"""Metrics shared by every method, so all rungs are scored the same way."""
import numpy as np


# ---- Pixel-level F1 / IoU ----

def _check(pred, true):
    """Boolean copies of both masks; error if their shapes differ."""
    pred, true = np.asarray(pred), np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, true {true.shape}")
    return pred.astype(bool), true.astype(bool)


def confusion(pred_mask, true_mask):
    """TP/FP/FN/TN pixel counts for one image."""
    p, t = _check(pred_mask, true_mask)
    return {
        "tp": int(( p &  t).sum()),     # predicted spliced, is spliced
        "fp": int(( p & ~t).sum()),     # predicted spliced, is authentic
        "fn": int((~p &  t).sum()),     # predicted authentic, is spliced
        "tn": int((~p & ~t).sum()),     # predicted authentic, is authentic
    }


def f1_iou(pred_mask, true_mask):
    """F1 and IoU of the spliced class (both 1.0 if both masks are empty)."""
    c = confusion(pred_mask, true_mask)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    if tp + fp + fn == 0:                              # nothing to find, nothing found
        return {"f1": 1.0, "iou": 1.0, **c}
    return {
        "f1": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        **c,
    }


# ---- ROC AUC ----

def image_auc(scores, labels):
    """AUC = P(random tampered scores higher than random authentic); ties count half."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:                       # undefined with one class
        return float("nan")
    ranks = rankdata(scores, method="average")         # tied scores share the average rank
    # Mann-Whitney: (sum of positive ranks - its minimum) / number of pos-neg pairs
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pixel_auc(score, truth, n_sample=20000, seed=0):
    """AUC of a score map against a mask, on a fixed random sample of pixels (None if one class)."""
    truth = np.asarray(truth, bool)
    idx = np.random.default_rng(seed).choice(truth.size, size=min(n_sample, truth.size),
                                             replace=False)          # same pixels every call
    s, t = np.asarray(score, float).ravel()[idx], truth.ravel()[idx]
    if t.all() or not t.any():
        return None
    return image_auc(s, t.astype(int))                 # each pixel treated as one sample


def bootstrap_auc_ci(pos, neg, n=2000, seed=0, level=0.95):
    """Bootstrap confidence interval for image_auc (resample each class with replacement)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    lab = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    v = [image_auc(np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)], lab)
         for _ in range(n)]
    a = (1 - level) / 2 * 100                          # 2.5 for a 95% interval
    return float(np.percentile(v, a)), float(np.percentile(v, 100 - a))
