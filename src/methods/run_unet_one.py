"""Train and evaluate one (test, val) pair. Driven by run_unet_nested.py."""
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("WANDB", "1")

TEST_FOLD = int(os.environ["TEST_FOLD"])
VAL_FOLD = int(os.environ["VAL_FOLD"])

import run_unet as R          # honours all the env vars already
from patch_data import load_rows


def run_pair(k, v, rows):
    """Same body as R.run_fold, but with the val fold supplied rather than k+1."""
    orig = R.N_FOLDS
    R.N_FOLDS = 5
    # monkey-patch the pairing: run_fold derives val_fold = (k+1)%N_FOLDS
    src_val = (k + 1) % 5
    if v != src_val:
        # relabel so the intended val fold lands where run_fold expects it
        remap = {v: src_val, src_val: v}
        for r in rows:
            f = int(r["fold"])
            r["fold"] = str(remap.get(f, f))
    out = R.run_fold(k, rows)
    R.N_FOLDS = orig
    return out


if __name__ == "__main__":
    rows = load_rows()
    thr, f1, iou, ep, nan_ep = run_pair(TEST_FOLD, VAL_FOLD, rows)
    os.makedirs("results", exist_ok=True)
    with open(f"results/rung4_pair_t{TEST_FOLD}_v{VAL_FOLD}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_fold", "val_fold", "thr", "f1", "iou", "stopped_at", "first_nan"])
        w.writerow([TEST_FOLD, VAL_FOLD, thr, f"{f1:.6f}", f"{iou:.6f}", ep, nan_ep or ""])
    print(f"pair t{TEST_FOLD}v{VAL_FOLD}: F1 {f1:.4f}  IoU {iou:.4f}", flush=True)
