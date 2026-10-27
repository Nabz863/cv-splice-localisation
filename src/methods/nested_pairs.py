"""Rung 2 under all 20 (test, validation) fold pairs: how much does the validation fold matter?

Reads results/rung2_grid.npz, writes results/rung2_nested_pairs.csv.
"""
import csv, itertools, os                         # CSV files, combinations, paths
import numpy as np                                # arrays and maths

d = np.load("results/rung2_grid.npz", allow_pickle=True)   # per-image counts from run_mrf.py
counts, grid = d["counts"], d["grid"]             # counts: (images, solvers, grid points, 3); grid: (tau, beta) list
rows = list(csv.DictReader(open(os.path.join(os.environ["DATASETS"], "casia2",
                                             "splice_manifest.csv"))))   # same image order as counts
folds = np.array([int(r["fold"]) for r in rows])  # each image's fold
N = 5                                             # number of folds


def f1_of(idx, gi):
    """Pooled F1 over images idx at grid point gi."""
    tp, fp, fn = counts[idx, 0, gi].sum(0)        # add up the counts of those images (solver 0 = graph cut)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0


print(f"{'test':>5}{'val':>5}{'tau':>7}{'beta':>7}{'test F1':>10}")
by_test, all_pairs = {k: [] for k in range(N)}, []   # test fold -> its F1s; list of every pair's result

for k, v in itertools.product(range(N), range(N)):   # every (test fold k, validation fold v)
    if k == v:                                    # a fold can't be both
        continue
    vi, ti = np.where(folds == v)[0], np.where(folds == k)[0]           # validation / test images
    gi = int(np.argmax([f1_of(vi, g) for g in range(len(grid))]))      # best grid point on validation
    tp, fp, fn = counts[ti, 0, gi].sum(0)         # that setting's counts on the test fold
    f1 = 2 * tp / (2 * tp + fp + fn)              # test F1
    by_test[k].append(f1)
    all_pairs.append((k, v, *grid[gi], f1))       # (test, val, tau, beta, F1)
    star = "  <- main protocol" if v == (k + 1) % N else ""   # mark the usual single pairing (val = test + 1)
    print(f"{k:>5}{v:>5}{grid[gi][0]:>7}{grid[gi][1]:>7}{f1:>10.4f}{star}")

# ---- How much does each test fold's F1 move with the validation fold? ----
print(f"\n{'test fold':>10}{'mean':>9}{'std':>8}{'range':>9}{'main':>9}")
for k in range(N):
    a = np.array(by_test[k])                      # the 4 F1s for this test fold
    main = [p[4] for p in all_pairs if p[0] == k and p[1] == (k + 1) % N][0]   # the usual pairing's F1
    print(f"{k:>10}{a.mean():>9.4f}{a.std():>8.4f}{a.max()-a.min():>9.4f}{main:>9.4f}")

main_mean = np.mean([p[4] for p in all_pairs if p[1] == (p[0] + 1) % N])   # mean over the 5 usual pairs
all_mean = np.mean([p[4] for p in all_pairs])                             # mean over all 20 pairs
print(f"\nmain protocol (5 pairs)  mean F1 {main_mean:.4f}")
print(f"all pairs     (20 pairs) mean F1 {all_mean:.4f}")
print(f"difference               {abs(main_mean - all_mean):.4f}")

# ---- Save (build_tables.py cross-checks its own recomputation against this) ----
with open("results/rung2_nested_pairs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test_fold", "val_fold", "tau", "beta", "test_f1"])
    w.writerows(all_pairs)
print("\nwrote results/rung2_nested_pairs.csv")
