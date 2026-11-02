"""
Report Figure Generator for TCF-Net
Reads the artifacts written by train.py and renders the figures used in the project report
and IEEE paper into artifacts/figures/.

Usage:
    python -m evaluation.make_figures
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from features.stylometric import batch_extract_stylometric, STYLE_NAMES
from features.diffusion import simulate_cascade

ART = 'artifacts'
OUT = os.path.join(ART, 'figures')
INK = '#1f2937'
BLUE, ORANGE, GREEN, GREY, RED = '#2563eb', '#ea580c', '#16a34a', '#9ca3af', '#dc2626'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight'
})


def _box(ax, x, y, w, h, text, fc, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=INK, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1, shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    _box(ax, 0.1, 2.7, 1.5, 0.8, 'Input text\n(EN / TA / Tanglish /\nHinglish / HI)', '#f3f4f6', 8.5)
    _box(ax, 2.0, 4.6, 2.3, 0.9, 'Confound scrubber\n+ MuRIL encoder (768)\n+ char TF-IDF→SVD (128)', '#dbeafe', 8.5)
    _box(ax, 2.0, 2.65, 2.3, 0.9, 'Stylometric extractor\n22 deception cues', '#ffedd5', 8.5)
    _box(ax, 2.0, 0.7, 2.3, 0.9, 'SIR cascade simulator\n8 propagation features', '#dcfce7', 8.5)
    for y in (5.05, 3.1, 1.15):
        _arrow(ax, 1.6, 3.1, 2.0, y)
    _arrow(ax, 3.15, 2.65, 3.15, 1.6)

    for y, lab in ((4.75, 'Proj A\n896→256'), (2.8, 'Proj B\n22→256'), (0.85, 'Proj C\n8→256')):
        _box(ax, 4.7, y, 1.1, 0.6, lab, '#ffffff', 8.5)
        _arrow(ax, 4.3, y + 0.3, 4.7, y + 0.3)

    _box(ax, 6.25, 2.45, 1.45, 1.3, 'Gated fusion\nα = softmax(MLP\n[zt; zs; zp])', '#ede9fe', 8.5)
    for y in (5.05, 3.1, 1.15):
        _arrow(ax, 5.8, y, 6.25, 3.1)
    _box(ax, 8.05, 2.75, 1.0, 0.7, 'Shared\ntrunk 256', '#ffffff', 8.5)
    _arrow(ax, 7.7, 3.1, 8.05, 3.1)

    heads = [(4.9, 'f(x): Real / Fake', '#dbeafe'),
             (3.6, 'g(x): confidence\n→ ABSTAIN if g<τ', '#fef9c3'),
             (2.3, 'h(x): auxiliary\nveracity head', '#f3f4f6'),
             (1.0, 'Hate head:\nHateful / Not', '#fee2e2')]
    for y, lab, fc in heads:
        _box(ax, 9.45, y, 1.5, 0.75, lab, fc, 8.2)
        _arrow(ax, 9.05, 3.1, 9.45, y + 0.37)

    ax.text(5.5, 6.0, 'TCF-Net: Tri-Modal Credibility Fusion Network', ha='center', fontsize=12, weight='bold', color=INK)
    fig.savefig(os.path.join(OUT, 'fig_architecture.png'))
    plt.close(fig)


def pipeline():
    steps = ['Input\nclaim', 'Pre-processing\n(scrub, normalise)', 'Tri-modal\nfeature extraction',
             'Standardise\n(train-fit scalers)', 'Gated fusion\n+ trunk', 'Veracity, confidence\n& hate heads',
             'Decision:\nREAL / FAKE / ABSTAIN']
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1.9)
    ax.axis('off')
    w = 1.35
    for i, s in enumerate(steps):
        x = 0.1 + i * 1.57
        _box(ax, x, 0.45, w, 1.0, s, '#eef2ff' if i not in (0, 6) else '#f3f4f6', 8.2)
        if i < len(steps) - 1:
            _arrow(ax, x + w, 0.95, x + 1.57, 0.95)
    fig.savefig(os.path.join(OUT, 'fig_pipeline.png'))
    plt.close(fig)


def dataset_distribution():
    df = pd.read_csv('data/tcf_multitask_dataset.csv', encoding='utf-8-sig')
    df['language'] = df['language'].str.replace(r'\s*\(.*\)', '', regex=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ct = pd.crosstab(df['language'], df['veracity_class'])[['Real', 'Fake']]
    ct.plot.barh(stacked=True, ax=axes[0], color=[BLUE, ORANGE], width=0.7)
    axes[0].set_title('Veracity label by language')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('')
    ch = pd.crosstab(df['language'], df['hate_speech_class'])[['Non-Hateful', 'Hateful']]
    ch.plot.barh(stacked=True, ax=axes[1], color=[GREY, RED], width=0.7)
    axes[1].set_title('Hate speech label by language')
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('')
    amb = df['ambiguity_level'].value_counts()
    axes[2].bar(range(len(amb)), amb.values, color=[ORANGE, BLUE, GREY][:len(amb)])
    axes[2].set_xticks(range(len(amb)))
    axes[2].set_xticklabels([a.replace(' / ', '/\n') for a in amb.index], fontsize=9)
    axes[2].set_title('Ambiguity level')
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_dataset.png'))
    plt.close(fig)


def training_curves():
    h = pd.read_csv(os.path.join(ART, 'training_history.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(h['epoch'], h['train_loss'], color=BLUE, lw=2)
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('Epoch')
    axes[1].plot(h['epoch'], h['val_acc'], color=BLUE, lw=2, label='Val accuracy')
    axes[1].plot(h['epoch'], h['val_auc'], color=ORANGE, lw=2, label='Val ROC-AUC')
    axes[1].plot(h['epoch'], h['val_coverage'], color=GREEN, lw=2, ls='--', label='Mean g(x)')
    axes[1].set_title('Validation metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_training.png'))
    plt.close(fig)


def confusion():
    m = json.load(open(os.path.join(ART, 'final_metrics.json')))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    panels = [('veracity', 'Veracity (test)', ['Real', 'Fake']),
              ('hate_speech', 'Hate speech (test)', ['Not hateful', 'Hateful'])]
    for ax, (key, title, labels) in zip(axes, panels):
        cm = np.array(m[key]['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13,
                        color='white' if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_confusion.png'))
    plt.close(fig)


def risk_coverage():
    c = pd.read_csv(os.path.join(ART, 'risk_coverage_curve.csv'))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(c['coverage'] * 100, c['selective_risk'] * 100, color=BLUE, lw=2.2, label='TCF-Net (g(x) ranking)')
    ax.plot(c['coverage'] * 100, c['random_risk'] * 100, color=GREY, lw=2, ls='--', label='Random abstention')
    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Selective risk / error (%)')
    ax.set_title('Risk–coverage curve (test set)')
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, 'fig_risk_coverage.png'))
    plt.close(fig)


def per_language():
    d = pd.read_csv(os.path.join(ART, 'per_language_metrics.csv'), encoding='utf-8-sig')
    d['language'] = d['language'].str.replace(r'\s*\(.*\)', '', regex=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.27, d['accuracy'] * 100, 0.27, color=BLUE, label='Accuracy (100% coverage)')
    ax.bar(x, d['selective_accuracy'] * 100, 0.27, color=GREEN, label='Selective accuracy (τ)')
    ax.bar(x + 0.27, d['hate_accuracy'] * 100, 0.27, color=RED, label='Hate speech accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(d['language'])
    ax.set_ylim(min(50, (d['accuracy'].min() * 100) - 5), 101)
    ax.set_ylabel('%')
    ax.set_title('Per-language performance (test set)')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.savefig(os.path.join(OUT, 'fig_per_language.png'))
    plt.close(fig)


def baselines():
    path = os.path.join(ART, 'baseline_comparison.csv')
    if not os.path.exists(path):
        return
    b = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [BLUE if 'proposed' in m else GREY for m in b['model']]
    ax.barh(b['model'], b['accuracy'] * 100, color=colors)
    for i, v in enumerate(b['accuracy'] * 100):
        ax.text(v + 0.3, i, f'{v:.2f}', va='center', fontsize=9)
    ax.set_xlim(max(0, b['accuracy'].min() * 100 - 10), 100)
    ax.set_xlabel('Test accuracy (%)')
    ax.set_title('Baseline and ablation comparison')
    ax.invert_yaxis()
    fig.savefig(os.path.join(OUT, 'fig_baselines.png'))
    plt.close(fig)


def gates_and_style():
    m = json.load(open(os.path.join(ART, 'final_metrics.json')))['veracity']
    df = pd.read_csv('data/tcf_multitask_dataset.csv', encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    g = [m['mean_gate_semantic'], m['mean_gate_stylometric'], m['mean_gate_propagation']]
    axes[0].pie(g, labels=['Semantic', 'Stylometric', 'Propagation'], colors=[BLUE, ORANGE, GREEN],
                autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.45, edgecolor='white'))
    axes[0].set_title('Mean modality gate weight α (test)')

    X = batch_extract_stylometric(df['statement_text'].astype(str).tolist())
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = df['veracity_label'].values
    diff = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    order = np.argsort(np.abs(diff))[::-1][:10]
    axes[1].barh([STYLE_NAMES[i] for i in order], diff[order],
                 color=[ORANGE if diff[i] > 0 else BLUE for i in order])
    axes[1].axvline(0, color=INK, lw=0.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Mean z-score (Fake − Real)')
    axes[1].set_title('Most discriminative stylometric cues')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_gates_style.png'))
    plt.close(fig)


def sir_example():
    fake = "SHOCKING!! Scientists EXPOSED: 5G network secretly embedding tracking chips. Share before authorities delete this link!!!"
    real = "The World Health Organization has published quarterly clinical trial metrics on the COVID booster shot during its scheduled review."
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for text, lab, col in ((fake, 'Sensational (fake-style) claim', ORANGE), (real, 'Neutral (real-style) report', BLUE)):
        sv = batch_extract_stylometric([text])[0]
        excl, caps, sens = sv[STYLE_NAMES.index('excl_ratio')], sv[STYLE_NAMES.index('caps_word_ratio')], sv[STYLE_NAMES.index('sensational_density')]
        arous = np.clip(4.0 * excl + 3.0 * caps + 6.0 * sens, 0.0, 1.0)
        beta, gamma = 0.18 + 0.45 * arous, 0.22 - 0.08 * arous
        S, I, curve = 0.998, 0.002, []
        for _ in range(24):
            ni, nr = beta * S * I, gamma * I
            S, I = max(S - ni, 0), max(I + ni - nr, 0)
            curve.append(I)
        ax.plot(range(1, 25), np.array(curve) * 100, color=col, lw=2.2, label=f'{lab} (arousal={arous:.2f})')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Infected share (%)')
    ax.set_title('SIR propagation proxy (Modality C)')
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(OUT, 'fig_sir.png'))
    plt.close(fig)


def _short_lang(s):
    return s.str.replace(r'\s*\(.*\)', '', regex=True).str.replace('Tamil-English', 'Tanglish')


def roc_pr():
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    d = pd.read_csv(os.path.join(ART, 'test_predictions.csv'), encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for y, p, lab, col in ((d['y_true'], d['p_fake'], 'Veracity (Fake)', BLUE), (d['hate_true'], d['p_hate'], 'Hate speech', RED)):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=col, lw=2, label=f'{lab}  AUC={roc_auc_score(y, p):.3f}')
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=col, lw=2, label=f'{lab}  AP={average_precision_score(y, p):.3f}')
    axes[0].plot([0, 1], [0, 1], color=GREY, ls='--', lw=1)
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].set_title('ROC curves (test set)')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision–recall curves (test set)')
    for ax in axes:
        ax.legend(frameon=False, fontsize=9, loc='lower right' if ax is axes[0] else 'lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_roc_pr.png'))
    plt.close(fig)


def confidence_analysis():
    d = pd.read_csv(os.path.join(ART, 'test_predictions.csv'), encoding='utf-8-sig')
    tau = json.load(open(os.path.join(ART, 'final_metrics.json')))['selective_abstention']['tau_val_calibrated']
    cols = {'Borderline / Needs Fact-Check': ORANGE, 'Clear Fact': BLUE, 'Sensational Deception': GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0, 1, 41)
    axes[0].hist([d.loc[d.ambiguity == amb, 'confidence_g'] for amb in cols], bins=bins, stacked=True,
                 color=list(cols.values()), alpha=0.8, label=[amb.split(' /')[0] for amb in cols])
    for amb, col in cols.items():
        sub = d[d.ambiguity == amb]
        axes[1].scatter(sub['p_fake'], sub['confidence_g'], s=12, color=col, alpha=0.6, label=amb.split(' /')[0])
    for ax in axes:
        ax.axhline(tau, color=RED, ls='--', lw=1) if ax is axes[1] else ax.axvline(tau, color=RED, ls='--', lw=1)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Selection confidence g(x)')
    axes[0].set_ylabel('Test claims (log scale)')
    axes[0].set_title(f'Confidence distribution (τ = {tau:.3f}, dashed)')
    axes[1].set_xlabel('P(fake | x)')
    axes[1].set_ylabel('g(x)')
    axes[1].set_title('Prediction versus confidence')
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=8.5, loc='center right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_confidence.png'))
    plt.close(fig)


def gates_by_group():
    d = pd.read_csv(os.path.join(ART, 'test_predictions.csv'), encoding='utf-8-sig')
    d['language'] = _short_lang(d['language'])
    d['ambiguity'] = d['ambiguity'].str.split(' /').str[0]
    gcols = ['gate_semantic', 'gate_stylometric', 'gate_propagation']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title in ((axes[0], 'language', 'Mean gate weight by language'), (axes[1], 'ambiguity', 'Mean gate weight by claim type')):
        m = d.groupby(key)[gcols].mean()
        left = np.zeros(len(m))
        for c, col, lab in zip(gcols, (BLUE, ORANGE, GREEN), ('Semantic', 'Stylometric', 'Propagation')):
            ax.barh(m.index, m[c], left=left, color=col, label=lab)
            for i, (l_, v) in enumerate(zip(left, m[c])):
                ax.text(l_ + v / 2, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white')
            left += m[c].values
        ax.set_xlim(0, 1)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_gates_groups.png'))
    plt.close(fig)


def style_distributions():
    df = pd.read_csv('data/tcf_multitask_dataset.csv', encoding='utf-8-sig')
    X = batch_extract_stylometric(df['statement_text'].astype(str).tolist())
    feats = ['excl_ratio', 'caps_word_ratio', 'sensational_density', 'hedge_density', 'punct_burstiness', 'avg_sent_len']
    groups = [('Clear Fact', BLUE), ('Sensational Deception', ORANGE), ('Borderline / Needs Fact-Check', GREY)]
    fig, axes = plt.subplots(1, len(feats), figsize=(13, 3.4))
    for ax, f in zip(axes, feats):
        vals = [X[df.ambiguity_level == g, STYLE_NAMES.index(f)] for g, _ in groups]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6, showfliers=False)
        for patch, (_, col) in zip(bp['boxes'], groups):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Clear', 'Sens.', 'Border.'], fontsize=8.5)
        ax.set_title(f, fontsize=10)
    fig.suptitle('Stylometric feature distributions by claim type (full benchmark)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_style_dist.png'))
    plt.close(fig)


def sir_by_language():
    df = pd.read_csv('data/tcf_multitask_dataset.csv', encoding='utf-8-sig')
    df['lang'] = _short_lang(df['language'])
    X = batch_extract_stylometric(df['statement_text'].astype(str).tolist())
    excl, caps, sens = (X[:, STYLE_NAMES.index(k)] for k in ('excl_ratio', 'caps_word_ratio', 'sensational_density'))
    arous = np.clip(4.0 * excl + 3.0 * caps + 6.0 * sens, 0.0, 1.0)
    langs = sorted(df['lang'].unique())
    fig, axes = plt.subplots(1, len(langs), figsize=(13, 3.0), sharey=True)
    for ax, lg in zip(axes, langs):
        for cls, col in (('Fake', ORANGE), ('Real', BLUE)):
            a = arous[(df['lang'] == lg).values & (df['veracity_class'] == cls).values].mean()
            beta, gamma = 0.18 + 0.45 * a, 0.22 - 0.08 * a
            S_, I_, curve = 0.998, 0.002, []
            for _ in range(24):
                ni, nr = beta * S_ * I_, gamma * I_
                S_, I_ = max(S_ - ni, 0), max(I_ + ni - nr, 0)
                curve.append(I_)
            ax.plot(range(1, 25), np.array(curve) * 100, color=col, lw=2, label=f'{cls} (a={a:.2f})')
        ax.set_title(lg, fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel('Infected share (%)')
    fig.suptitle('Mean SIR propagation prior by language and veracity class', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_sir_lang.png'))
    plt.close(fig)


def confusion_by_language():
    d = pd.read_csv(os.path.join(ART, 'test_predictions.csv'), encoding='utf-8-sig')
    d['language'] = _short_lang(d['language'])
    langs = sorted(d['language'].unique())
    fig, axes = plt.subplots(1, len(langs), figsize=(13, 2.9))
    for ax, lg in zip(axes, langs):
        g = d[d.language == lg]
        cm = np.array([[((g.y_true == i) & (g.y_pred == j)).sum() for j in (0, 1)] for i in (0, 1)])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=11,
                        color='white' if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Real', 'Fake'], fontsize=8.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Real', 'Fake'], fontsize=8.5)
        ax.set_title(lg, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=8.5)
    axes[0].set_ylabel('Actual', fontsize=8.5)
    fig.suptitle('Veracity confusion matrices by language (test set, 100% coverage)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_confusion_lang.png'))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (architecture, pipeline, dataset_distribution, training_curves, confusion,
               risk_coverage, per_language, baselines, gates_and_style, sir_example,
               roc_pr, confidence_analysis, gates_by_group, style_distributions, sir_by_language,
               confusion_by_language):
        fn()
        print(f'[Figures] {fn.__name__} done')
    print(f'[Figures] Saved to {OUT}')


if __name__ == '__main__':
    main()
