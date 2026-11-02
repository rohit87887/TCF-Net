"""
Evaluation Metrics & Risk-Coverage Curve Computation for TCF-Net
Calculates:
  - Standard Classification Metrics: Accuracy, Precision, Recall, Macro F1, ROC-AUC, PR-AUC, Confusion Matrix
  - Selective Prediction (Abstention) Metrics:
      * Selective Risk vs. Coverage curve
      * Random Abstention comparison curve
      * Area Under Risk-Coverage Curve (AURC)
      * Error rates at specific coverage thresholds (100%, 85%, 70%, 50%)
  - Multi-task Hate Speech metrics
  - Exporter for JSON and CSV artifacts for Review 3 reports
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes binary classification performance metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Safe AUC handling when only one class is present in split
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(rec, prec))
    except Exception:
        pr_auc = 0.5

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_fake': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_fake': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_fake': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selection_scores: np.ndarray,
    cov_grid: np.ndarray = np.arange(0.05, 1.001, 0.01),
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Computes selective risk (error rate on answered samples) across the coverage spectrum [0.05, 1.00].
    Also generates the expected random abstention baseline.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    errors = (y_pred != y_true).astype(float)
    n_samples = len(errors)

    # Sort items by model confidence g(x) descending (most confident first)
    confidence_order = np.argsort(-selection_scores)
    sorted_errors = errors[confidence_order]

    rng = np.random.default_rng(seed)
    records = []
    sel_risks = []

    for c in cov_grid:
        k = max(int(c * n_samples), 1)
        sel_risk = float(sorted_errors[:k].mean())
        sel_acc = 1.0 - sel_risk

        # Random baseline: average of 10 random permutations of coverage k
        rand_estimates = [
            errors[rng.permutation(n_samples)[:k]].mean()
            for _ in range(10)
        ]
        rnd_risk = float(np.mean(rand_estimates))

        records.append({
            'coverage': round(float(c), 3),
            'samples_retained': k,
            'selective_risk': round(sel_risk, 4),
            'selective_accuracy': round(sel_acc, 4),
            'random_risk': round(rnd_risk, 4)
        })
        sel_risks.append(sel_risk)

    curve_df = pd.DataFrame(records)

    # Key operating thresholds
    def get_at_cov(target_c: float) -> Tuple[float, float]:
        k = max(int(target_c * n_samples), 1)
        risk = float(sorted_errors[:k].mean())
        return risk, 1.0 - risk

    r100, a100 = get_at_cov(1.00)
    r85, a85 = get_at_cov(0.85)
    r70, a70 = get_at_cov(0.70)
    r50, a50 = get_at_cov(0.50)

    # Area Under Risk Curve (AURC)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # np.trapz was removed in NumPy 2.x
    aurc = float(trapezoid(sel_risks, cov_grid))

    summary = {
        'error_100cov': r100,
        'acc_100cov': a100,
        'error_85cov': r85,
        'acc_85cov': a85,
        'error_70cov': r70,
        'acc_70cov': a70,
        'error_50cov': r50,
        'acc_50cov': a50,
        'aurc': aurc
    }

    return curve_df, summary

def export_evaluation_results(
    veracity_metrics: Dict[str, float],
    risk_summary: Dict[str, float],
    curve_df: pd.DataFrame,
    hate_metrics: Optional[Dict[str, float]] = None,
    output_dir: str = 'artifacts'
) -> str:
    """Exports structured metrics to artifacts/final_metrics.json and artifacts/risk_coverage_curve.csv."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    combined = {
        'veracity': veracity_metrics,
        'selective_abstention': risk_summary,
    }
    if hate_metrics:
        combined['hate_speech'] = hate_metrics

    json_path = os.path.join(output_dir, 'final_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

    csv_path = os.path.join(output_dir, 'risk_coverage_curve.csv')
    curve_df.to_csv(csv_path, index=False)

    print(f"[Metrics] Exported final metrics to: {json_path}")
    print(f"[Metrics] Exported risk-coverage curve to: {csv_path}")
    return json_path
