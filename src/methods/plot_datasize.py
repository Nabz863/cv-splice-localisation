"""Plot F1 against training-set size, with the other rungs as reference lines."""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE, RUNG1, RUNG2 = 0.2351, 0.2621, 0.2845

pts = []
for suffix, label in [("_n50", 50), ("_n150", 150), ("_n400", 400),
                      ("_n800", 800), ("", 1093)]:
    paths = glob.glob(f"results/rung4_pair_t*v*{suffix}.csv")
    # the full run has no suffix, so filter out the sized files
    if suffix == "":
        paths = [p for p in paths if "_n" not in os.path.basename(p)[len("rung4_pair_t0v1"):]]
    if not paths:
        continue
    by_test = defaultdict(list)
    for p in paths:
        r = list(csv.DictReader(open(p)))[0]
        by_test[int(r["test_fold"])].append(float(r["f1"]))
    means = np.array([np.mean(v) for v in by_test.values()])
    pts.append((label, means.mean(), means.std(), len(paths)))

pts.sort()
print(f"{'n_train':>9}{'mean F1':>10}{'std':>8}{'pairs':>7}")
for n, m, s, k in pts:
    print(f"{n:>9}{m:>10.4f}{s:>8.4f}{k:>7}")

if len(pts) > 1:
    ns = [p[0] for p in pts]; ms = [p[1] for p in pts]; ss = [p[2] for p in pts]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(ns, ms, yerr=ss, marker="o", capsize=4, label="U-Net (rung 4)")
    plt.axhline(RUNG2, ls="--", c="tab:orange", label=f"MRF (rung 2) {RUNG2}")
    plt.axhline(RUNG1, ls="--", c="tab:green", label=f"ELA (rung 1) {RUNG1}")
    plt.axhline(CHANCE, ls=":", c="grey", label=f"chance {CHANCE}")
    plt.xscale("log")
    plt.xlabel("training masks"); plt.ylabel("pixel F1")
    plt.title("Does more labelled data help?")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/datasize_curve.png", dpi=120)
    print("\nwrote results/datasize_curve.png")
