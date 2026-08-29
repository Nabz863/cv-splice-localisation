"""Shared evaluation. Every method routes through this — do not duplicate."""
import numpy as np


def _check(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, true {true.shape}")
    return pred.astype(bool), true.astype(bool)


def confusion(pred_mask, true_mask):
    """Pixel counts. pred_mask/true_mask are boolean arrays, same shape."""
    p, t = _check(pred_mask, true_mask)
    return {
        "tp": int(( p &  t).sum()),
        "fp": int(( p & ~t).sum()),
        "fn": int((~p &  t).sum()),
        "tn": int((~p & ~t).sum()),
    }


def f1_iou(pred_mask, true_mask):
    """Pixel F1 and IoU for the tampered (positive) class.

    Convention for the degenerate case: if the ground truth is empty AND the
    prediction is empty, both metrics are 1.0 (a correct 'nothing here').
    If exactly one is empty, both are 0.0. Stated explicitly because 0/0 is a
    choice, not a fact, and every method must make the same one.
    """
    c = confusion(pred_mask, true_mask)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    if tp + fp + fn == 0:
        return {"f1": 1.0, "iou": 1.0, **c}
    return {
        "f1": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        **c,
    }


def image_auc(scores, labels):
    """Image-level ROC AUC via the rank (Mann-Whitney) identity.

    scores: per-image tamperedness score (higher = more likely tampered)
    labels: 1 = tampered, 0 = authentic
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties, else tied scores bias the result
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
