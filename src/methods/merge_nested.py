"""Summarise rung-4 pair results. Works for the 5-pair and 20-pair protocols."""
import csv, glob, sys
from collections import defaultdict
import numpy as np

suffix = sys.argv[1] if len(sys.argv) > 1 else ""
paths = sorted(glob.glob(f"results/rung4_pair_t*v*{suffix}.csv"))
if not paths:
    raise SystemExit(f"no results/rung4_pair_t*v*{suffix}.csv found")

rows = []
for p in paths:
    with open(p) as f:
        r = list(csv.DictReader(f))
        if r:
            rows.append(r[0])
rows.sort(key=lambda r: (int(r["test_fold"]), int(r["val_fold"])))

print(f"{'test':>5}{'val':>5}{'thr':>6}{'F1':>9}{'IoU':>9}{'stop':>6}{'NaN@':>6}")
by_test = defaultdict(list)
for r in rows:
    k, v = int(r["test_fold"]), int(r["val_fold"])
    f1 = float(r["f1"]); by_test[k].append(f1)
    main = "  <- (k+1)" if v == (k + 1) % 5 else ""
    print(f"{k:>5}{v:>5}{float(r['thr']):>6.2f}{f1:>9.4f}{float(r['iou']):>9.4f}"
          f"{r['stopped_at']:>6}{r.get('first_nan_epoch') or '-':>6}{main}")

n_pairs = len(rows)
f1 = np.array([float(r["f1"]) for r in rows])
iou = np.array([float(r["iou"]) for r in rows])

if n_pairs > 5:
    print(f"\n{'test fold':>10}{'mean':>9}{'std':>8}{'range':>9}{'n':>4}")
    for k in sorted(by_test):
        a = np.array(by_test[k])
        print(f"{k:>10}{a.mean():>9.4f}{a.std():>8.4f}{a.max()-a.min():>9.4f}{len(a):>4}")
    # per-test-fold mean first, so every fold weighs equally
    fold_means = np.array([np.mean(by_test[k]) for k in sorted(by_test)])
    print(f"\nmean of per-fold means  F1 {fold_means.mean():.4f} "
          f"+/- {fold_means.std():.4f}")
    main_only = [float(r["f1"]) for r in rows
                 if int(r["val_fold"]) == (int(r["test_fold"]) + 1) % 5]
    if len(main_only) == 5:
        print(f"(k+1) pairing only      F1 {np.mean(main_only):.4f}  "
              f"bias {np.mean(main_only) - fold_means.mean():+.4f}")

print(f"\npairs: {n_pairs}")
print(f"mean F1  {f1.mean():.4f} +/- {f1.std():.4f}")
print(f"mean IoU {iou.mean():.4f} +/- {iou.std():.4f}")
print(f"non-finite in {sum(1 for r in rows if r.get('first_nan_epoch'))}/{n_pairs} runs")

with open(f"results/rung4_unet{suffix}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"wrote results/rung4_unet{suffix}.csv")
