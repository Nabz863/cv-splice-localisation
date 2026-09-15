"""Summarise rung 5: does the spatial prior still help over a learned unary?"""
import csv, glob
from collections import defaultdict
import numpy as np

paths = sorted(glob.glob("results/rung5_pair_t*v*.csv"))
if not paths:
    raise SystemExit("no results/rung5_pair_t*v*.csv found")

rows = [list(csv.DictReader(open(p)))[0] for p in paths]
rows.sort(key=lambda r: (int(r["test_fold"]), int(r["val_fold"])))

print(f"{'test':>5}{'val':>5}{'tau':>7}{'beta':>7}{'F1':>9}{'F1(b=0)':>10}{'gain':>9}")
by_test, by_test0 = defaultdict(list), defaultdict(list)
for r in rows:
    k = int(r["test_fold"])
    by_test[k].append(float(r["f1"])); by_test0[k].append(float(r["f1_beta0"]))
    print(f"{k:>5}{int(r['val_fold']):>5}{float(r['tau']):>7.2f}"
          f"{float(r['beta']):>7.2f}{float(r['f1']):>9.4f}"
          f"{float(r['f1_beta0']):>10.4f}{float(r['prior_gain']):>+9.4f}")

fm = np.array([np.mean(by_test[k]) for k in sorted(by_test)])
fm0 = np.array([np.mean(by_test0[k]) for k in sorted(by_test0)])
betas = np.array([float(r["beta"]) for r in rows])

print(f"\npairs: {len(rows)}")
print(f"MRF over U-Net logits   F1 {fm.mean():.4f} +/- {fm.std():.4f}")
print(f"beta = 0 (raw U-Net)    F1 {fm0.mean():.4f} +/- {fm0.std():.4f}")
print(f"prior contributes       {fm.mean()-fm0.mean():+.4f}")
print(f"\nbeta = 0 selected in {(betas == 0).sum()}/{len(betas)} pairs")
print(f"beta > 0 median        {np.median(betas[betas > 0]) if (betas > 0).any() else 0}")
print(f"\nfor comparison, over the ELA map (rung 1 -> rung 2): +0.0224")

with open("results/rung5_mrf_refine.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("wrote results/rung5_mrf_refine.csv")
