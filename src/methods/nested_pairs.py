"""Rung 2 under all 20 (test, validation) fold pairs: how much does the validation fold matter?

Reads results/rung2_grid.npz, writes results/rung2_nested_pairs.csv.
"""
import csv, itertools, os
import numpy as np

d = np.load("results/rung2_grid.npz", allow_pickle=True)
counts, grid = d["counts"], d["grid"]
rows = list(csv.DictReader(open(os.path.join(os.environ["DATASETS"], "casia2",
                                             "splice_manifest.csv"))))
folds = np.array([int(r["fold"]) for r in rows])
N = 5


def f1_of(idx, gi):
    """Pooled F1 over images idx at grid point gi."""
    tp, fp, fn = counts[idx, 0, gi].sum(0)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0


print(f"{'test':>5}{'val':>5}{'tau':>7}{'beta':>7}{'test F1':>10}")
by_test, all_pairs = {k: [] for k in range(N)}, []

for k, v in itertools.product(range(N), range(N)):
    if k == v:
        continue
    vi, ti = np.where(folds == v)[0], np.where(folds == k)[0]           # validation / test images
    gi = int(np.argmax([f1_of(vi, g) for g in range(len(grid))]))      # best grid point on validation
    tp, fp, fn = counts[ti, 0, gi].sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn)
    by_test[k].append(f1)
    all_pairs.append((k, v, *grid[gi], f1))
    star = "  <- main protocol" if v == (k + 1) % N else ""
    print(f"{k:>5}{v:>5}{grid[gi][0]:>7}{grid[gi][1]:>7}{f1:>10.4f}{star}")

print(f"\n{'test fold':>10}{'mean':>9}{'std':>8}{'range':>9}{'main':>9}")
for k in range(N):
    a = np.array(by_test[k])
    main = [p[4] for p in all_pairs if p[0] == k and p[1] == (k + 1) % N][0]
    print(f"{k:>10}{a.mean():>9.4f}{a.std():>8.4f}{a.max()-a.min():>9.4f}{main:>9.4f}")

main_mean = np.mean([p[4] for p in all_pairs if p[1] == (p[0] + 1) % N])
all_mean = np.mean([p[4] for p in all_pairs])
print(f"\nmain protocol (5 pairs)  mean F1 {main_mean:.4f}")
print(f"all pairs     (20 pairs) mean F1 {all_mean:.4f}")
print(f"difference               {abs(main_mean - all_mean):.4f}")

with open("results/rung2_nested_pairs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test_fold", "val_fold", "tau", "beta", "test_f1"])
    w.writerows(all_pairs)
print("\nwrote results/rung2_nested_pairs.csv")
