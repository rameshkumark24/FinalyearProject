"""
SFR Framework — Clinical Evaluation Metrics
===========================================
Computes clinical screening performance metrics:
- AUROC, AUPRC
- Sensitivity, Specificity, PPV (Precision), NPV
- Sensitivity at fixed clinical referral / specificity thresholds (90%, 95%)
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn import metrics


def compute_screening_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Compute comprehensive clinical screening metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_probs = np.asarray(y_pred_probs).astype(float)

    # Basic AUCs
    try:
        auroc = float(metrics.roc_auc_score(y_true, y_pred_probs))
    except Exception:
        auroc = 0.5

    try:
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_pred_probs)
        auprc = float(metrics.auc(recall_curve, precision_curve))
    except Exception:
        auprc = 0.0

    # Binary decisions at threshold
    y_pred_bin = (y_pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    referral_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Sensitivity at 90% and 95% specificity
    fpr, tpr, roc_thresh = metrics.roc_curve(y_true, y_pred_probs)
    # Specificity = 1 - FPR. For 90% spec, target FPR <= 0.10
    idx_spec90 = np.where(fpr <= 0.10)[0]
    sens_at_spec90 = float(tpr[idx_spec90[-1]]) if len(idx_spec90) > 0 else 0.0

    # For 95% spec, target FPR <= 0.05
    idx_spec95 = np.where(fpr <= 0.05)[0]
    sens_at_spec95 = float(tpr[idx_spec95[-1]]) if len(idx_spec95) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "referral_rate": float(referral_rate),
        "f1_score": float(f1),
        "sens_at_spec90": sens_at_spec90,
        "sens_at_spec95": sens_at_spec95,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
