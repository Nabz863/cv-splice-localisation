"""Re-select (tau, beta) on the validation fold only, matching rung 4's protocol.
No recomputation -- the per-image counts are already stored."""
import csv, os
import numpy as np

d = np.load("results/rung2_grid.npz", allow_pickle=True)
counts, grid = d["counts"], d["grid"]
rows = list(csv.DictReader(open(os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv"))))
folds = np.array([int(r["fold"]) for r in rows])
N = 5

def f1_of(idx, gi):
    tp, fp, fn = counts[idx, 0, gi].sum(0)
    return 2*tp/(2*tp+fp+fn) if (tp+fp+fn) else 1.0

out = []
for k in range(N):
    val = (k + 1) % N                       # same val fold rung 4 uses
    vi = np.where(folds == val)[0]
    ti = np.where(folds == k)[0]
    gi = int(np.argmax([f1_of(vi, g) for g in range(len(grid))]))
    tp, fp, fn = counts[ti, 0, gi].sum(0)
    out.append((k, *grid[gi], 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)))
    print(f"fold {k}: tau {grid[gi][0]} beta {grid[gi][1]}  F1 {out[-1][3]:.4f}")

print(f"\nmean F1 {np.mean([o[3] for o in out]):.4f} +/- {np.std([o[3] for o in out]):.4f}")
with open("results/rung2_mrf_valselect.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["fold","tau","beta","f1","iou"]); w.writerows(out)
