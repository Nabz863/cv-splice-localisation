"""Rung 5, step 3: collect the 20 per-pair results into results/rung5_mrf_refine{SUFFIX}.csv."""
import csv, glob, os                              # CSV files, finding files by pattern, paths
from collections import defaultdict               # a dict that starts each new key with an empty list
import numpy as np                                # maths

SUFFIX = os.environ.get("SUFFIX", "")             # "" = full data, "_n50" etc. = data-size runs
paths = sorted(glob.glob(f"results/rung5_pair_t?v?{SUFFIX}.csv"))   # one file per (test, val) pair
if not paths:
    raise SystemExit(f"no results/rung5_pair_t?v?{SUFFIX}.csv found")

rows = [list(csv.DictReader(open(p)))[0] for p in paths]             # the single row from each file
rows.sort(key=lambda r: (int(r["test_fold"]), int(r["val_fold"])))  # order by test fold, then val fold

# ---- Print one line per pair, and group F1 by test fold ----
print(f"{'test':>5}{'val':>5}{'tau':>7}{'beta':>7}{'F1':>9}{'F1(b=0)':>10}{'gain':>9}")
by_test, by_test0 = defaultdict(list), defaultdict(list)   # test fold -> list of F1s (with / without prior)
for r in rows:
    k = int(r["test_fold"])
    by_test[k].append(float(r["f1"])); by_test0[k].append(float(r["f1_beta0"]))   # rung 5 and rung 4 F1
    print(f"{k:>5}{int(r['val_fold']):>5}{float(r['tau']):>7.2f}"
          f"{float(r['beta']):>7.2f}{float(r['f1']):>9.4f}"
          f"{float(r['f1_beta0']):>10.4f}{float(r['prior_gain']):>+9.4f}")

# mean over validation folds within each test fold, then over test folds
fm = np.array([np.mean(by_test[k]) for k in sorted(by_test)])       # rung 5: one mean per test fold
fm0 = np.array([np.mean(by_test0[k]) for k in sorted(by_test0)])    # rung 4: one mean per test fold
betas = np.array([float(r["beta"]) for r in rows])                  # the beta each pair chose

# ---- Summary (mean +/- standard deviation across the 5 test folds) ----
print(f"\npairs: {len(rows)}")
print(f"MRF over U-Net logits   F1 {fm.mean():.4f} +/- {fm.std():.4f}")
print(f"beta = 0 (raw U-Net)    F1 {fm0.mean():.4f} +/- {fm0.std():.4f}")
print(f"prior contributes       {fm.mean()-fm0.mean():+.4f}")
print(f"\nbeta = 0 selected in {(betas == 0).sum()}/{len(betas)} pairs")    # how often validation turned the prior off
print(f"beta > 0 median        {np.median(betas[betas > 0]) if (betas > 0).any() else 0}")
print("\nfor the same comparison over ELA, see results/tables/main.csv (build_tables.py)")

# ---- Save all 20 rows in one file (build_tables.py reads it) ----
out = f"results/rung5_mrf_refine{SUFFIX}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())   # same columns as the per-pair files
    w.writeheader(); w.writerows(rows)
print(f"wrote {out}")
