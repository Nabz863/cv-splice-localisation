"""Qualitative figures: what each rung actually produces, including where it fails.

Splice mode builds one row per example with seven panels — image, ground truth, the
z-scored ELA map, ELA+MRF, SAM2's size-filtered union (with the oracle's best single
proposal outlined in gold), the U-Net probability map, U-Net+MRF — so the reader can
see the thing the numbers are describing: ELA responding to compression texture
across the whole frame, the prior turning a speckled map into blobs, and the learned
unary already being blob-like before the prior touches it.

Examples are chosen by rung 5's own per-image F1 at the five quantiles of that
distribution, best to worst, rather than hand-picked. The worst row is the point of
the figure as much as the best one.

Copy-move mode is the predicted-failure figure: four panels (image, ground truth,
ELA, ELA+MRF) on the copy-move set, where the pasted region carries the host's own
compression history, so there is no seam for ELA to find by construction.

CPU only. Imports no torch: the U-Net panels come from the cached logits and the
SAM2 panels from results/sam2_examples/ (sam2_examples.py, GPU); without those the
figure is drawn without the SAM2 column.

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
N_ROWS = int(os.environ.get("N_ROWS", 4))
SAM2_DIR = "results/sam2_examples"
TAU_ELA = float(os.environ.get("TAU_ELA", 0.75))
BETA_ELA = float(os.environ.get("BETA_ELA", 8.0))
DPI = int(os.environ.get("DPI", 300))


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


def pick_examples():
    """The splice figure's examples: rung 5's own per-image F1 on TAG's test fold,
    at N_ROWS evenly spaced quantiles, best to worst. sam2_examples.py imports this
    so that its SAM2 masks are for exactly these images."""
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
    qs = np.linspace(0, 1, N_ROWS)
    picks = [order[int(round(q * (n - 1)))] for q in qs]
    labels = ["best" if q == 0 else "worst" if q == 1 else f"p{round(100 * (1 - q))}" for q in qs]
    return dict(rows=[rows[i] for i in picks], lgs=[lgs[i] for i in picks],
                gts=[gts[i] for i in picks], labels=labels, tau=tau, beta=beta,
                test_fold=test_fold, scores=scores)


def sam2_masks(path):
    """Rung 3's masks for one image, written by sam2_examples.py, or None."""
    p = os.path.join(SAM2_DIR, os.path.basename(path).rsplit(".", 1)[0] + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    shape = tuple(z["shape"]); k = shape[0] * shape[1]
    un = lambda key: np.unpackbits(z[key])[:k].reshape(shape).astype(bool)
    return un("union"), un("oracle")


def splice():
    ex = pick_examples()
    tau, beta = ex["tau"], ex["beta"]
    sam = [sam2_masks(r["image"]) for r in ex["rows"]]
    have_sam = all(s is not None for s in sam)
    if not have_sam:
        print("  no SAM2 masks for these examples - run sam2_examples.py; drawing without rung 3")
    cols = ["image", "ground truth", "ELA (rung 1)", "ELA + MRF", "SAM2 (rung 3)",
            "U-Net (rung 4)", "U-Net + MRF"]
    if not have_sam:
        cols.remove("SAM2 (rung 3)")
    FS = 6.5
    fig, axes = plt.subplots(len(ex["rows"]), len(cols), figsize=(7.16, 1.0 * len(ex["rows"]) + 0.3))
    axes = np.atleast_2d(axes)

    def show(ax, data, kind, title, header):
        if kind == "rgb":
            ax.imshow(data, interpolation="lanczos")
        elif kind == "mask":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            lo, hi = np.percentile(data, [1, 99])
            ax.imshow(data, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title((header + "\n" if header else "") + title, fontsize=FS, pad=2, linespacing=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color("#8a8984")

    for r, (row, gt, lg, lab, sm) in enumerate(zip(ex["rows"], ex["gts"], ex["lgs"], ex["labels"], sam)):
        img = np.asarray(Image.open(row["image"]).convert("RGB"))
        ela = ela_of(row["image"])
        e1 = np.asarray(ela >= TAU_ELA, bool)
        e2 = np.asarray(solve_graphcut(ela, TAU_ELA, BETA_ELA), bool)
        u4 = np.asarray(solve_graphcut(lg, tau, 0.0), bool)
        u5 = np.asarray(solve_graphcut(lg, tau, beta), bool)
        h = (lambda c: c) if r == 0 else (lambda c: "")
        panels = [(img, "rgb", "", h("image")),
                  (gt, "mask", f"{gt.mean() * 100:.1f}% tampered", h("ground truth")),
                  (ela, "heat", f"F1 {f1(e1, gt):.2f}", h("ELA (rung 1)")),
                  (e2, "mask", f"F1 {f1(e2, gt):.2f}", h("ELA + MRF"))]
        if have_sam:
            panels.append((sm[0], "mask", f"F1 {f1(sm[0], gt):.2f}", h("SAM2 (rung 3)")))
        panels += [(1 / (1 + np.exp(-lg)), "heat", f"F1 {f1(u4, gt):.2f}", h("U-Net (rung 4)")),
                   (u5, "mask", f"F1 {f1(u5, gt):.2f}", h("U-Net + MRF"))]
        for c, (data, kind, title, header) in enumerate(panels):
            show(axes[r, c], data, kind, title, header)
        if have_sam:                          # the oracle's best proposal, outlined in gold
            axes[r, 4].contour(sm[1].astype(float), levels=[0.5], colors="#e3a21a", linewidths=0.6)
        axes[r, 0].set_ylabel(lab, fontsize=FS)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.93, bottom=0.01, wspace=0.06, hspace=0.32)
    save(fig, "splice")
    print(f"tau={tau}, beta={beta} ({TAG}); ELA tau={TAU_ELA}, beta={BETA_ELA}")
    for lab, row in zip(ex["labels"], ex["rows"]):
        print(f"  {lab}: {os.path.basename(row['image'])}")
    s = ex["scores"]
    print(f"rung 5 per-image F1 on fold {ex['test_fold']}: "
          f"mean {s.mean():.4f}, median {np.median(s):.4f}, min {s.min():.4f}, max {s.max():.4f}")


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
