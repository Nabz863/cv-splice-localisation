"""Qualitative figures: what each rung actually produces, including where it fails.

Splice mode builds one row per example with six panels — image, ground truth, the
z-scored ELA map, ELA+MRF, the U-Net probability map, U-Net+MRF — so the reader can
see the thing the numbers are describing: ELA responding to compression texture
across the whole frame, the prior turning a speckled map into blobs, and the learned
unary already being blob-like before the prior touches it.

Examples are chosen by rung 5's own per-image F1 at the five quantiles of that
distribution, best to worst, rather than hand-picked. The worst row is the point of
the figure as much as the best one.

Copy-move mode is the predicted-failure figure: four panels (image, ground truth,
ELA, ELA+MRF) on the copy-move set, where the pasted region carries the host's own
compression history, so there is no seam for ELA to find by construction.

CPU only. Imports no torch: the U-Net panels come from the cached logits.

Usage (from the repo root):
    python src/methods/fig_qualitative.py
    MODE=copymove python src/methods/fig_qualitative.py
    TAG=t0v2 N_ROWS=6 python src/methods/fig_qualitative.py

Writes results/fig_qualitative_{mode}.png (and .pdf, for the report)
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import v_ela
from mrf import solve_graphcut

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
MODE = os.environ.get("MODE", "splice")
TAG = os.environ.get("TAG", "t0v1")
N_ROWS = int(os.environ.get("N_ROWS", 5))
TAU_ELA = float(os.environ.get("TAU_ELA", 0.75))
BETA_ELA = float(os.environ.get("BETA_ELA", 8.0))
DPI = int(os.environ.get("DPI", 150))


def f1(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 1.0


def ela_of(path):
    img = np.asarray(Image.open(path).convert("RGB"))   # uint8, as run_mrf.py
    return v_ela(img)                    # robust z-score map, as run_mrf.py


def read_pair_params(tag):
    p = f"results/rung5_pair_{tag}.csv"
    if os.environ.get("TAU") and os.environ.get("BETA"):
        return float(os.environ["TAU"]), float(os.environ["BETA"])
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; pass TAU= and BETA= explicitly")
    r = list(csv.DictReader(open(p)))[0]
    return float(r["tau"]), float(r["beta"])


def panel(ax, data, kind, title):
    if kind == "rgb":
        ax.imshow(data)
    elif kind == "mask":
        ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    elif kind == "heat":
        lo, hi = np.percentile(data, [1, 99])
        ax.imshow(data, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def splice():
    tau, beta = read_pair_params(TAG)
    test_fold = int(TAG[1])
    rows = [r for r in csv.DictReader(open(os.environ.get("MANIFEST", os.path.join(ROOT, "splice_manifest.csv"))))
            if int(r["fold"]) == test_fold]

    z = np.load(f"results/logits/{TAG}.npz", allow_pickle=False)
    n = int(z["n_test"])
    if n != len(rows):
        raise SystemExit(f"cache has {n} test images, manifest fold {test_fold} has "
                         f"{len(rows)}; the cache was built from a different manifest")

    lgs, gts, scores = [], [], []
    for i in range(n):
        shape = tuple(z[f"test_shape_{i}"])
        gt = np.unpackbits(z[f"test_gt_{i}"])[:shape[0] * shape[1]].reshape(shape).astype(bool)
        lg = z[f"test_lg_{i}"].astype(np.float32)
        lgs.append(lg); gts.append(gt)
        scores.append(f1(np.asarray(solve_graphcut(lg, tau, beta), bool), gt))
    scores = np.array(scores)

    order = scores.argsort()[::-1]
    picks = [order[int(round(q * (n - 1)))] for q in np.linspace(0, 1, N_ROWS)]
    labels = ["best", "p75", "median", "p25", "worst"] if N_ROWS == 5 else \
             [f"q{int(100 - 100 * q)}" for q in np.linspace(0, 1, N_ROWS)]

    fig, axes = plt.subplots(len(picks), 6, figsize=(13, 2.3 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (i, lab) in enumerate(zip(picks, labels)):
        img = np.asarray(Image.open(rows[i]["image"]).convert("RGB"))
        gt, lg = gts[i], lgs[i]
        ela = ela_of(rows[i]["image"])
        e1 = np.asarray(ela >= TAU_ELA, bool)
        e2 = np.asarray(solve_graphcut(ela, TAU_ELA, BETA_ELA), bool)
        u4 = np.asarray(solve_graphcut(lg, tau, 0.0), bool)
        u5 = np.asarray(solve_graphcut(lg, tau, beta), bool)

        panel(axes[r, 0], img, "rgb", f"{lab}  {os.path.basename(rows[i]['image'])[:22]}")
        panel(axes[r, 1], gt, "mask", f"ground truth  ({gt.mean()*100:.1f}% tampered)")
        panel(axes[r, 2], ela, "heat", f"ELA   F1 {f1(e1, gt):.3f}")
        panel(axes[r, 3], e2, "mask", f"ELA+MRF   F1 {f1(e2, gt):.3f}")
        panel(axes[r, 4], 1 / (1 + np.exp(-lg)), "heat", f"U-Net p   F1 {f1(u4, gt):.3f}")
        panel(axes[r, 5], u5, "mask", f"U-Net+MRF   F1 {f1(u5, gt):.3f}")

    fig.suptitle(f"Splice localisation, test fold {test_fold} ({TAG}).  "
                 f"ELA tau={TAU_ELA}, beta={BETA_ELA};  U-Net tau={tau}, beta={beta}.  "
                 f"Rows are quantiles of rung 5 per-image F1.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "splice")
    print(f"rung 5 per-image F1 on fold {test_fold}: "
          f"mean {scores.mean():.4f}, median {np.median(scores):.4f}, "
          f"min {scores.min():.4f}, max {scores.max():.4f}")


def copymove():
    mf = os.environ.get("CM_MANIFEST", os.path.join(ROOT, "copymove_manifest.csv"))
    rows = list(csv.DictReader(open(mf)))
    pick = list(np.random.default_rng(0).permutation(len(rows))[:N_ROWS])

    fig, axes = plt.subplots(len(pick), 4, figsize=(9, 2.3 * len(pick)))
    axes = np.atleast_2d(axes)
    f1s = []
    for r, i in enumerate(pick):
        img = np.asarray(Image.open(rows[i]["image"]).convert("RGB"))
        gt = np.asarray(Image.open(rows[i]["mask"]).convert("L")) > 127
        ela = ela_of(rows[i]["image"])
        e1 = np.asarray(ela >= TAU_ELA, bool)
        e2 = np.asarray(solve_graphcut(ela, TAU_ELA, BETA_ELA), bool)
        f1s.append(f1(e2, gt))
        panel(axes[r, 0], img, "rgb", os.path.basename(rows[i]["image"])[:26])
        panel(axes[r, 1], gt, "mask", f"ground truth  ({gt.mean()*100:.1f}%)")
        panel(axes[r, 2], ela, "heat", f"ELA   F1 {f1(e1, gt):.3f}")
        panel(axes[r, 3], e2, "mask", f"ELA+MRF   F1 {f1(e2, gt):.3f}")

    fig.suptitle("Copy-move, splicing-selected (tau, beta) applied unchanged. Random examples; "
                 "set-wide micro F1:\nELA 0.128, ELA+MRF 0.139, chance 0.092 "
                 "(pooled tampered rate 4.85%).", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "copymove")
    print(f"ELA+MRF F1 on these {len(pick)} examples: {np.mean(f1s):.4f} "
          f"(set-wide micro: ELA 0.1278, ELA+MRF 0.1391, chance 0.0924)")


def save(fig, mode):
    os.makedirs("results", exist_ok=True)
    for ext in ("png", "pdf"):
        p = f"results/fig_qualitative_{mode}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    {"splice": splice, "copymove": copymove}[MODE]()
