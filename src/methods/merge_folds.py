"""Merge per-fold CSVs into one summary. Needed because Colab disconnects mid-run
and folds are then completed across several sessions."""
import csv, glob, os, sys
import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else ""
paths = sorted(glob.glob(f"results/rung4_fold*{tag}.csv"))
if not paths:
    raise SystemExit(f"no results/rung4_fold*{tag}.csv found")

rows = []
for p in paths:
    with open(p) as f:
        r = list(csv.DictReader(f))
        if r:
            rows.append(r[0])

rows.sort(key=lambda r: int(r["fold"]))
f1 = np.array([float(r["f1"]) for r in rows])
iou = np.array([float(r["iou"]) for r in rows])

print(f"{'fold':<6}{'thr':>6}{'F1':>9}{'IoU':>9}{'stop':>6}{'NaN@':>6}")
for r in rows:
    print(f"{r['fold']:<6}{float(r['thr']):>6.2f}{float(r['f1']):>9.4f}"
          f"{float(r['iou']):>9.4f}{r['stopped_at']:>6}"
          f"{r.get('first_nan_epoch') or '-':>6}")

print(f"\nfolds: {len(rows)}/5")
print(f"mean F1  {f1.mean():.4f} +/- {f1.std():.4f}")
print(f"mean IoU {iou.mean():.4f} +/- {iou.std():.4f}")
if len(rows) < 5:
    print(f"INCOMPLETE — resume with FOLD_START={max(int(r['fold']) for r in rows)+1}")

with open(f"results/rung4_unet{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"wrote results/rung4_unet{tag}.csv")
