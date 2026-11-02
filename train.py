"""
Unified GPU & Colab Training Pipeline for TCF-Net (Review 3)
Features:
  - Supports configurable backbones: 'distilbert-base-uncased' or 'google/muril-base-cased'
  - Multi-Task Dual Heads: Veracity (Real/Fake/Abstain) + Hate Speech (Hateful/Not Hateful)
  - Subject-Transfer Split Protocol (controls for publisher artifacts) vs. Random Split
  - Automatic Mixed Precision (AMP) on CUDA with GradScaler
  - CosineAnnealingLR scheduling and Early Stopping on Validation ROC-AUC
  - Exports Final Metrics (Accuracy, F1, Selective Risk @ 85% coverage, AURC) to artifacts/
"""

import os
import sys
import argparse
import json
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

from features.stylometric import batch_extract_stylometric
from features.diffusion import batch_simulate_cascade
from features.scrubber import scrub_text
from models.tcf_net import TCFNet, selective_loss
from data.loaders import load_dataset
from evaluation.metrics import (
    compute_classification_metrics,
    compute_risk_coverage_curve,
    export_evaluation_results
)

def parse_args():
    parser = argparse.ArgumentParser(description="TCF-Net Training Pipeline")
    parser.add_argument("--backbone", type=str, default="distilbert-base-uncased",
                        choices=["distilbert-base-uncased", "google/muril-base-cased"],
                        help="Transformer backbone for semantic encoding")
    parser.add_argument("--dataset", type=str, default="tcf_multitask",
                        choices=["tcf_multitask", "isot", "synthetic"],
                        help="Training corpus (tcf_multitask = persisted 5-language benchmark)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max corpus samples to load for training/eval (default: all)")
    parser.add_argument("--protocol", type=str, default="grouped",
                        choices=["grouped", "subject_transfer", "random"],
                        help="Evaluation protocol (grouped = identical texts never cross splits)")
    parser.add_argument("--no_ablation", action="store_true",
                        help="Skip baseline and modality ablation runs")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for fusion trunk")
    parser.add_argument("--target_coverage", type=float, default=0.85, help="Target coverage for abstention")
    parser.add_argument("--target_risk", type=float, default=0.01,
                        help="Maximum validation selective error used to calibrate the abstention threshold")
    parser.add_argument("--svd_dim", type=int, default=128, help="Lexical SVD dimension")
    parser.add_argument("--enable_hate", action="store_true", default=True, help="Enable multi-task hate speech head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Directory for exported artifacts")
    return parser.parse_args()

def extract_semantic_features(texts: list, backbone_name: str, device: str, cache_dir: str = None) -> np.ndarray:
    """
    Extracts semantic sentence embeddings using Transformer backbone (DistilBERT or MuRIL)
    with a lightweight TF-IDF+SVD fallback if transformers / weights are unavailable locally.
    Embeddings are cached on disk (keyed by backbone + corpus hash) so re-runs skip the encoder.
    """
    cache_path = None
    if cache_dir:
        import hashlib
        key = hashlib.md5(("\n".join(texts) + backbone_name).encode('utf-8')).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"emb_cache_{key}.npy")
        if os.path.exists(cache_path):
            print(f"[Encoder] Loaded cached embeddings: {cache_path}")
            return np.load(cache_path)
    emb = _encode(texts, backbone_name, device)
    if cache_path is not None and emb.shape[1] == 768:
        np.save(cache_path, emb)
    return emb

def _encode(texts: list, backbone_name: str, device: str) -> np.ndarray:
    try:
        from transformers import AutoTokenizer, AutoModel
        print(f"[Encoder] Loading transformer backbone: {backbone_name} ...")
        tok = AutoTokenizer.from_pretrained(backbone_name)
        model = AutoModel.from_pretrained(backbone_name).to(device)
        model.eval()

        use_amp = (device == 'cuda')
        embs = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts[i:i + batch_size])
            tokens = tok(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                    out = model(**tokens)
                    mask = tokens['attention_mask'].unsqueeze(-1).float()
                    # Attention-masked mean pooling
                    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embs.append(pooled.float().cpu().numpy())
        print(f"[Encoder] Extracted {len(texts)} semantic embeddings of dimension {embs[0].shape[1]}")
        return np.vstack(embs).astype(np.float32)
    except Exception as e:
        print(f"[Encoder] Transformer offline or unavailable ({e}). Falling back to high-dim SVD representation.")
        tf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
        X_tf = tf.fit_transform(texts)
        n_comp = min(512, max(X_tf.shape[1] - 1, 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        out_svd = svd.fit_transform(X_tf).astype(np.float32)
        if out_svd.shape[1] < 512:
            out_svd = np.pad(out_svd, ((0, 0), (0, 512 - out_svd.shape[1])), mode='constant')
        elif out_svd.shape[1] > 512:
            out_svd = out_svd[:, :512]
        return out_svd

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f"  TCF-Net Training Pipeline | Review 3")
    print(f"  Device           : {device.upper()}")
    print(f"  Backbone         : {args.backbone}")
    print(f"  Protocol         : {args.protocol}")
    print(f"  Target Coverage  : {args.target_coverage * 100:.1f}%")
    print(f"  Multi-Task Hate  : {args.enable_hate}")
    print(f"============================================================")

    # 1. Load Dataset
    df, source_name = load_dataset(max_samples=args.max_samples, seed=args.seed, dataset=args.dataset)

    # 2. Extract Multimodal Features
    print("\n[Step 1/5] Extracting Tri-Modal Features...")
    print("  -> Modality A (Semantic Meaning)...")
    X_text = extract_semantic_features(df['content'].tolist(), args.backbone, device, cache_dir=args.output_dir)

    print("  -> Modality B (Stylometric Linguistics - 22D)...")
    X_style = batch_extract_stylometric(df['style_src'].tolist())

    print("  -> Modality C (SIR Propagation Simulation - 8D)...")
    X_prop = batch_simulate_cascade(X_style, seed=args.seed)

    # 3. Evaluation Split Protocol
    print(f"\n[Step 2/5] Constructing Split ({args.protocol})...")
    y_veracity = df['label_veracity'].values.astype(np.int64)
    y_hate = df['label_hate'].values.astype(np.int64)
    all_ix = np.arange(len(df))

    test_subjects = {'worldnews', 'News', 'US_News', 'Middle-east'}
    active_protocol = None
    if args.protocol == 'grouped':
        # Template corpora repeat identical sentences; grouping by normalized text keeps every
        # duplicate inside one split so test accuracy is not inflated by memorization.
        from sklearn.model_selection import StratifiedGroupKFold
        groups = pd.factorize(df['content'])[0]
        strat = y_veracity * 10 + pd.factorize(df['lang'])[0]
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        pool, idx_te = next(outer.split(all_ix, strat, groups))
        inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed)
        tr_rel, val_rel = next(inner.split(pool, strat[pool], groups[pool]))
        idx_tr, idx_val = pool[tr_rel], pool[val_rel]
        active_protocol = "grouped"
    elif args.protocol == 'subject_transfer' and df['subject'].nunique() > 2:
        idx_te = all_ix[df['subject'].isin(test_subjects).values]
        pool = all_ix[~df['subject'].isin(test_subjects).values]
        if len(idx_te) > 100 and len(pool) > 100:
            idx_tr, idx_val = train_test_split(pool, test_size=0.15, stratify=y_veracity[pool], random_state=args.seed)
            active_protocol = "subject_transfer"
        else:
            active_protocol = "random"
    elif args.protocol != 'grouped':
        active_protocol = "random"

    if active_protocol == "random":
        idx_tr, tmp = train_test_split(all_ix, test_size=0.30, stratify=y_veracity, random_state=args.seed)
        idx_val, idx_te = train_test_split(tmp, test_size=0.50, stratify=y_veracity[tmp], random_state=args.seed)

    print(f"  Protocol: {active_protocol} | Train: {len(idx_tr)} | Val: {len(idx_val)} | Test: {len(idx_te)}")

    # 4. Lexical SVD Projection (fit strictly on train)
    print("  Fitting lexical TF-IDF -> SVD projection on Train...")
    # Character n-grams are script-agnostic: they handle Tamil/Devanagari combining marks and
    # Romanized code-mixed spellings that word tokenizers fragment.
    tfv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=30000, sublinear_tf=True, min_df=2)
    tfv.fit(df['content'].values[idx_tr])
    svd = TruncatedSVD(n_components=min(args.svd_dim, len(idx_tr) - 1), random_state=args.seed)
    svd.fit(tfv.transform(df['content'].values[idx_tr]))
    X_lex = svd.transform(tfv.transform(df['content'].values)).astype(np.float32)

    X_sem = np.hstack([X_text, X_lex]).astype(np.float32)
    print(f"  Combined Semantic Vector Dim: {X_sem.shape[1]}")

    # Standardize streams using Train statistics
    sc_t, sc_s, sc_p = StandardScaler(), StandardScaler(), StandardScaler()
    sc_t.fit(X_sem[idx_tr])
    sc_s.fit(X_style[idx_tr])
    sc_p.fit(X_prop[idx_tr])

    def pack_tensors(ix):
        return (
            torch.tensor(sc_t.transform(X_sem[ix]), dtype=torch.float32),
            torch.tensor(sc_s.transform(X_style[ix]), dtype=torch.float32),
            torch.tensor(sc_p.transform(X_prop[ix]), dtype=torch.float32),
            torch.tensor(y_veracity[ix], dtype=torch.long),
            torch.tensor(y_hate[ix], dtype=torch.long)
        )

    tr_pack = pack_tensors(idx_tr)
    val_pack = pack_tensors(idx_val)
    te_pack = pack_tensors(idx_te)

    # 5. Initialize & train TCF-Net (wrapped so modality ablations reuse the same loop)
    use_amp = (device == 'cuda')

    def train_tcfnet(use_style=True, use_prop=True, verbose=True):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = TCFNet(
            d_text=X_sem.shape[1],
            d_style=X_style.shape[1],
            d_prop=X_prop.shape[1],
            d_model=256,
            dropout=0.3,
            use_style=use_style,
            use_prop=use_prop,
            enable_hate_head=args.enable_hate
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        g = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(TensorDataset(*tr_pack), batch_size=args.batch_size, shuffle=True, generator=g)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_score = (-1.0, -1.0)
        best_state = None
        patience = 10
        wait = 0
        history = []

        for ep in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0

            for xt, xs, xp, y_v, y_h in train_loader:
                xt, xs, xp = xt.to(device), xs.to(device), xp.to(device)
                y_v, y_h = y_v.to(device), y_h.to(device)

                opt.zero_grad()
                with torch.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp):
                    out = model(xt, xs, xp)
                    loss, m_dict = selective_loss(
                        logits_veracity=out['logits_veracity'],
                        selection=out['selection'],
                        aux_veracity=out['aux_veracity'],
                        alpha=out['alpha'],
                        y_veracity=y_v,
                        target_coverage=args.target_coverage,
                        logits_hate=out.get('logits_hate'),
                        aux_hate=out.get('aux_hate'),
                        y_hate=y_h if args.enable_hate else None
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                train_loss += loss.item() * len(y_v)

            sched.step()
            train_loss /= len(idx_tr)

            # Validation pass
            model.eval()
            with torch.no_grad():
                v_xt, v_xs, v_xp, v_yv, _ = [t.to(device) for t in val_pack]
                v_out = model(v_xt, v_xs, v_xp)
                v_probs = torch.softmax(v_out['logits_veracity'], dim=-1)[:, 1].cpu().numpy()
                v_y = v_yv.cpu().numpy()
                try:
                    val_auc = float(roc_auc_score(v_y, v_probs))
                except Exception:
                    val_auc = 0.5
                val_acc = float(((v_probs >= 0.5).astype(int) == v_y).mean())
                val_cov = float(v_out['selection'].mean().item())

            history.append({
                'epoch': ep,
                'train_loss': round(train_loss, 4),
                'val_auc': round(val_auc, 4),
                'val_acc': round(val_acc, 4),
                'val_coverage': round(val_cov, 4)
            })

            if verbose and (ep % 5 == 0 or ep == args.epochs):
                print(f"  Epoch {ep:2d}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Coverage: {val_cov:.3f}")

            # Model selection on validation accuracy, ties broken by ROC-AUC
            score = (round(val_acc, 4), val_auc)
            if score > best_score:
                best_score = score
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stopping triggered at epoch {ep}.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model, history, best_score

    def predict(model, pack):
        with torch.no_grad():
            xt, xs, xp, yv, yh = [t.to(device) for t in pack]
            out = model(xt, xs, xp)
            res = {
                'p': torch.softmax(out['logits_veracity'], dim=-1)[:, 1].cpu().numpy(),
                'g': out['selection'].cpu().numpy(),
                'alpha': out['alpha'].cpu().numpy(),
                'y': yv.cpu().numpy(),
                'y_hate': yh.cpu().numpy()
            }
            if 'logits_hate' in out:
                res['p_hate'] = torch.softmax(out['logits_hate'], dim=-1)[:, 1].cpu().numpy()
        return res

    print("\n[Step 3/5] Initializing TCF-Net Model...")
    print("\n[Step 4/5] Training TCF-Net with Early Stopping...")
    model, history, best_score = train_tcfnet()
    best_auc = best_score[1]
    print(f"  Training Complete. Best Validation Acc: {best_score[0]:.4f} | ROC-AUC: {best_auc:.4f}")

    # 7. Final Test Evaluation & Risk-Coverage Curve
    print("\n[Step 5/5] Computing Final Test Metrics & Risk-Coverage Curves...")
    te = predict(model, te_pack)
    p_te, g_te, alpha_te, y_test_v = te['p'], te['g'], te['alpha'], te['y']

    # Abstention threshold is calibrated on the validation split, never on test.
    # Risk-targeted calibration: the lowest threshold whose validation selective error is <= target_risk,
    # placed midway to the next lower score so it sits inside the confidence gap.
    val_res = predict(model, val_pack)
    g_val = val_res['g']
    err_val = ((val_res['p'] >= 0.5).astype(int) != val_res['y']).astype(float)
    order = np.argsort(-g_val)
    g_sorted, risk_sorted = g_val[order], np.cumsum(err_val[order]) / np.arange(1, len(order) + 1)
    ok = np.where(risk_sorted <= args.target_risk)[0]
    if len(ok):
        k = ok.max()
        tau = float(g_sorted[k] if k == len(g_sorted) - 1 else (g_sorted[k] + g_sorted[k + 1]) / 2)
    else:
        tau = float(np.quantile(g_val, 1.0 - args.target_coverage))

    veracity_metrics = compute_classification_metrics(y_test_v, p_te)
    curve_df, risk_summary = compute_risk_coverage_curve(y_test_v, p_te, g_te)

    answered = g_te >= tau
    pred_te = (p_te >= 0.5).astype(int)
    risk_summary['tau_val_calibrated'] = tau
    risk_summary['coverage_at_tau'] = float(answered.mean())
    risk_summary['acc_at_tau'] = float((pred_te[answered] == y_test_v[answered]).mean()) if answered.any() else 0.0

    from sklearn.metrics import confusion_matrix
    veracity_metrics['confusion_matrix'] = confusion_matrix(y_test_v, pred_te, labels=[0, 1]).tolist()
    veracity_metrics['mean_gate_semantic'] = float(alpha_te[:, 0].mean())
    veracity_metrics['mean_gate_stylometric'] = float(alpha_te[:, 1].mean())
    veracity_metrics['mean_gate_propagation'] = float(alpha_te[:, 2].mean())

    hate_metrics = None
    if 'p_hate' in te and (te['y_hate'] >= 0).any():
        valid = te['y_hate'] >= 0
        hate_metrics = compute_classification_metrics(te['y_hate'][valid], te['p_hate'][valid])
        hate_metrics['confusion_matrix'] = confusion_matrix(
            te['y_hate'][valid], (te['p_hate'][valid] >= 0.5).astype(int), labels=[0, 1]).tolist()

    # Per-language and per-ambiguity breakdown
    te_df = pd.DataFrame({
        'text': df['raw'].values[idx_te],
        'language': df['lang'].values[idx_te],
        'ambiguity': df['ambiguity'].values[idx_te] if 'ambiguity' in df else 'na',
        'y_true': y_test_v,
        'p_fake': np.round(p_te, 4),
        'y_pred': pred_te,
        'confidence_g': np.round(g_te, 4),
        'answered': answered.astype(int),
        'gate_semantic': np.round(alpha_te[:, 0], 4),
        'gate_stylometric': np.round(alpha_te[:, 1], 4),
        'gate_propagation': np.round(alpha_te[:, 2], 4)
    })
    if 'p_hate' in te:
        te_df['hate_true'] = te['y_hate']
        te_df['p_hate'] = np.round(te['p_hate'], 4)
    te_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False, encoding='utf-8-sig')

    def breakdown(col):
        rows = []
        for key, grp in te_df.groupby(col):
            ans = grp[grp['answered'] == 1]
            row = {
                col: key,
                'samples': len(grp),
                'accuracy': round(float((grp['y_pred'] == grp['y_true']).mean()), 4),
                'coverage': round(float(grp['answered'].mean()), 4),
                'selective_accuracy': round(float((ans['y_pred'] == ans['y_true']).mean()), 4) if len(ans) else None
            }
            if 'p_hate' in grp:
                row['hate_accuracy'] = round(float(((grp['p_hate'] >= 0.5).astype(int) == grp['hate_true']).mean()), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    lang_df = breakdown('language')
    lang_df.to_csv(os.path.join(args.output_dir, "per_language_metrics.csv"), index=False, encoding='utf-8-sig')
    amb_df = breakdown('ambiguity')
    amb_df.to_csv(os.path.join(args.output_dir, "per_ambiguity_metrics.csv"), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION REPORT (Review 3)")
    print("=" * 60)
    print(f"  Dataset                           : {source_name} | Protocol: {active_protocol}")
    print(f"  Veracity Accuracy (100% Coverage)  : {veracity_metrics['accuracy']:.4f}")
    print(f"  Veracity Macro F1                 : {veracity_metrics['macro_f1']:.4f}")
    print(f"  Veracity ROC-AUC                  : {veracity_metrics['roc_auc']:.4f}")
    print(f"  Selective Acc @ tau (val-calib.)  : {risk_summary['acc_at_tau']:.4f}  (Coverage: {risk_summary['coverage_at_tau']:.4f})")
    print(f"  Selective Acc @ 85% Target Cov    : {risk_summary['acc_85cov']:.4f}  (Error: {risk_summary['error_85cov']:.4f})")
    print(f"  Selective Acc @ 70% Target Cov    : {risk_summary['acc_70cov']:.4f}  (Error: {risk_summary['error_70cov']:.4f})")
    print(f"  Selective Acc @ 50% Target Cov    : {risk_summary['acc_50cov']:.4f}  (Error: {risk_summary['error_50cov']:.4f})")
    print(f"  Area Under Risk Curve (AURC)      : {risk_summary['aurc']:.4f}")
    if hate_metrics:
        print(f"  Hate Speech Accuracy              : {hate_metrics['accuracy']:.4f}")
        print(f"  Hate Speech Macro F1              : {hate_metrics['macro_f1']:.4f}")
    print("-" * 60)
    print(lang_df.to_string(index=False))
    print("-" * 60)
    print(amb_df.to_string(index=False))
    print("=" * 60)

    # 8. Baselines & modality ablations on the identical split
    if not args.no_ablation:
        print("\n[Extra] Baselines and modality ablations on the same split...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.naive_bayes import MultinomialNB
        from scipy.sparse import hstack as sp_hstack

        X_char = tfv.transform(df['content'].values)
        tfw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit(df['content'].values[idx_tr])
        X_word = tfw.transform(df['content'].values)
        X_both = sp_hstack([X_char, X_word]).tocsr()

        def score(name, y_pred, y_prob=None):
            m = compute_classification_metrics(y_test_v, y_prob if y_prob is not None else y_pred.astype(float))
            m['accuracy'] = float((y_pred == y_test_v).mean())
            return {'model': name, 'accuracy': round(m['accuracy'], 4), 'macro_f1': round(m['macro_f1'], 4),
                    'roc_auc': round(m['roc_auc'], 4) if y_prob is not None else None}

        rows = []
        nb = MultinomialNB().fit(X_word[idx_tr], y_veracity[idx_tr])
        rows.append(score('TF-IDF (word) + Naive Bayes', nb.predict(X_word[idx_te]), nb.predict_proba(X_word[idx_te])[:, 1]))
        lr = LogisticRegression(max_iter=2000, C=4.0).fit(X_word[idx_tr], y_veracity[idx_tr])
        rows.append(score('TF-IDF (word) + Logistic Regression', lr.predict(X_word[idx_te]), lr.predict_proba(X_word[idx_te])[:, 1]))
        svm = LinearSVC(C=1.0).fit(X_both[idx_tr], y_veracity[idx_tr])
        rows.append(score('TF-IDF (word+char) + Linear SVM', svm.predict(X_both[idx_te])))
        st = LogisticRegression(max_iter=2000).fit(sc_s.transform(X_style[idx_tr]), y_veracity[idx_tr])
        rows.append(score('Stylometric (22-D) + Logistic Regression',
                          st.predict(sc_s.transform(X_style[idx_te])), st.predict_proba(sc_s.transform(X_style[idx_te]))[:, 1]))
        Xe = StandardScaler().fit(X_text[idx_tr])
        emb_lr = LogisticRegression(max_iter=3000).fit(Xe.transform(X_text[idx_tr]), y_veracity[idx_tr])
        rows.append(score(f'{args.backbone} embeddings + Logistic Regression',
                          emb_lr.predict(Xe.transform(X_text[idx_te])), emb_lr.predict_proba(Xe.transform(X_text[idx_te]))[:, 1]))

        for name, us, up in [('TCF-Net (Semantic only)', False, False),
                             ('TCF-Net (Semantic + Stylometric)', True, False)]:
            m_ab, _, _ = train_tcfnet(use_style=us, use_prop=up, verbose=False)
            r = predict(m_ab, te_pack)
            rows.append(score(name, (r['p'] >= 0.5).astype(int), r['p']))
        rows.append(score('TCF-Net (Full Tri-Modal, proposed)', pred_te, p_te))

        base_df = pd.DataFrame(rows)
        base_df.to_csv(os.path.join(args.output_dir, "baseline_comparison.csv"), index=False)
        print(base_df.to_string(index=False))

    # 9. Save Artifacts
    export_evaluation_results(veracity_metrics, risk_summary, curve_df, hate_metrics, args.output_dir)
    with open(os.path.join(args.output_dir, "run_info.json"), 'w', encoding='utf-8') as f:
        json.dump({'dataset': source_name, 'protocol': active_protocol, 'backbone': args.backbone,
                   'n_total': int(len(df)), 'n_train': int(len(idx_tr)), 'n_val': int(len(idx_val)),
                   'n_test': int(len(idx_te)), 'unique_texts': int(df['content'].nunique()),
                   'epochs_run': len(history), 'semantic_dim': int(X_sem.shape[1])}, f, indent=2)

    ckpt_path = os.path.join(args.output_dir, "tcfnet.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'dims': {
            'd_text': X_sem.shape[1],
            'd_style': X_style.shape[1],
            'd_prop': X_prop.shape[1],
            'd_model': 256
        },
        'cfg': vars(args),
        'abstention_threshold_85': tau
    }, ckpt_path)

    scalers_path = os.path.join(args.output_dir, "scalers.joblib")
    joblib.dump({
        'text': sc_t,
        'style': sc_s,
        'prop': sc_p,
        'tfv': tfv,
        'svd': svd,
        'backbone': args.backbone
    }, scalers_path)

    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)
    print(f"[Done] Checkpoint saved: {ckpt_path}")
    print(f"[Done] Scalers saved   : {scalers_path}")

if __name__ == "__main__":
    main()
