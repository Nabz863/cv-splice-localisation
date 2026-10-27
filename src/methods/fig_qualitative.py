"""Qualitative figure (Fig. 5): image, truth, ELA, ELA+MRF, SAM2, U-Net, U-Net+MRF per row.

Rows are chosen at evenly spaced quantiles of rung 5's per-image F1 (not hand-picked).
MODE=copymove draws the copy-move version. Writes results/fig_qualitative_<mode>.png/.pdf.
Usage:  python src/methods/fig_qualitative.py
"""
import csv, os, sys                               # standard-library helpers
import numpy as np                                # arrays and maths
import matplotlib                                 # plotting library
matplotlib.use("Agg")                             # draw to files, no screen needed
import matplotlib.pyplot as plt                   # plotting
from PIL import Image                             # reads image files

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from noise_variants import v_ela                  # rung 1's score map
from mrf import solve_graphcut                    # the MRF solver (rungs 2 and 5)

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
MODE = os.environ.get("MODE", "splice")           # "splice" or "copymove"
TAG = os.environ.get("TAG", "t0v1")               # which (test, val) pair: test fold 0, val fold 1
N_ROWS = int(os.environ.get("N_ROWS", 4))         # images in the figure
SAM2_DIR = "results/sam2_examples"                # where sam2_examples.py saves its masks
TAU_ELA = float(os.environ.get("TAU_ELA", 0.75))  # rung 1-2 threshold
BETA_ELA = float(os.environ.get("BETA_ELA", 8.0)) # rung 2 prior strength
DPI = int(os.environ.get("DPI", 300))             # image resolution


# ---- Helpers ----

def f1(pred, gt):
    """Per-image F1."""
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())   # counts
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 1.0           # nothing to find and nothing predicted = perfect


def ela_of(path):
    """ELA score map of one image (rung 1)."""
    img = np.asarray(Image.open(path).convert("RGB"))
    return v_ela(img)


def read_pair_params(tag):
    """(tau, beta) the pair selected on validation (or TAU/BETA from the environment)."""
    p = f"results/rung5_pair_{tag}.csv"           # written by sweep_cached.py
    if os.environ.get("TAU") and os.environ.get("BETA"):   # values given by hand win
        return float(os.environ["TAU"]), float(os.environ["BETA"])
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found; pass TAU= and BETA= explicitly")
    r = list(csv.DictReader(open(p)))[0]          # first (only) row
    return float(r["tau"]), float(r["beta"])


def panel(ax, data, kind, title):
    """Draw one panel: an RGB image, a binary mask, or a heat map."""
    if kind == "rgb":
        ax.imshow(data)                           # colour photo
    elif kind == "mask":
        ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")   # black/white, no smoothing
    elif kind == "heat":
        lo, hi = np.percentile(data, [1, 99])     # colour range ignoring the extreme 1% each side
        ax.imshow(data, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])          # no axis ticks


def pick_examples():
    """Pick N_ROWS test images at evenly spaced quantiles of rung 5's per-image F1, best to worst."""
    tau, beta = read_pair_params(TAG)             # rung 5's selected setting for this pair
    test_fold = int(TAG[1])                       # "t0v1" -> 0
    rows = [r for r in csv.DictReader(open(os.environ.get("MANIFEST", os.path.join(ROOT, "splice_manifest.csv"))))
            if int(r["fold"]) == test_fold]      # test-fold images, in manifest order

    z = np.load(f"results/logits/{TAG}.npz", allow_pickle=False)   # U-Net outputs saved by cache_logits.py
    n = int(z["n_test"])                          # number of test images in the cache
    if n != len(rows):                            # cache and manifest must match
        raise SystemExit(f"cache has {n} test images, manifest fold {test_fold} has "
                         f"{len(rows)}; the cache was built from a different manifest")

    lgs, gts, scores = [], [], []                 # logits, true masks, rung 5 F1 per image
    for i in range(n):
        shape = tuple(z[f"test_shape_{i}"])       # (height, width)
        gt = np.unpackbits(z[f"test_gt_{i}"])[:shape[0] * shape[1]].reshape(shape).astype(bool)   # unpack the mask
        lg = z[f"test_lg_{i}"].astype(np.float32) # U-Net logits
        lgs.append(lg); gts.append(gt)
        scores.append(f1(np.asarray(solve_graphcut(lg, tau, beta), bool), gt))   # rung 5's F1 on this image
    scores = np.array(scores)

    order = scores.argsort()[::-1]                   # images sorted best to worst
    qs = np.linspace(0, 1, N_ROWS)                   # evenly spaced quantiles
    picks = [order[int(round(q * (n - 1)))] for q in qs]   # image at each quantile
    labels = ["best" if q == 0 else "worst" if q == 1 else f"p{round(100 * (1 - q))}" for q in qs]   # row labels
    return dict(rows=[rows[i] for i in picks], lgs=[lgs[i] for i in picks],
                gts=[gts[i] for i in picks], labels=labels, tau=tau, beta=beta,
                test_fold=test_fold, scores=scores)


def sam2_masks(path):
    """Rung 3's masks for one image, written by sam2_examples.py, or None."""
    p = os.path.join(SAM2_DIR, os.path.basename(path).rsplit(".", 1)[0] + ".npz")
    if not os.path.exists(p):                     # sam2_examples.py hasn't been run
        return None
    z = np.load(p)
    shape = tuple(z["shape"]); k = shape[0] * shape[1]   # image size; number of pixels
    un = lambda key: np.unpackbits(z[key])[:k].reshape(shape).astype(bool)   # unpack one stored mask
    return un("union"), un("oracle")


# ---- The two figures ----

def splice():
    """Splicing figure: one row per picked image, seven panels."""
    ex = pick_examples()
    tau, beta = ex["tau"], ex["beta"]
    sam = [sam2_masks(r["image"]) for r in ex["rows"]]   # SAM2 masks per image (or None)
    have_sam = all(s is not None for s in sam)    # draw the SAM2 column only if all are present
    if not have_sam:
        print("  no SAM2 masks for these examples - run sam2_examples.py; drawing without rung 3")
    cols = ["image", "ground truth", "ELA (rung 1)", "ELA + MRF", "SAM2 (rung 3)",
            "U-Net (rung 4)", "U-Net + MRF"]      # column headings
    if not have_sam:
        cols.remove("SAM2 (rung 3)")
    FS = 6.5                                      # font size
    fig, axes = plt.subplots(len(ex["rows"]), len(cols), figsize=(7.16, 1.0 * len(ex["rows"]) + 0.3))   # 7.16 in = IEEE page width
    axes = np.atleast_2d(axes)                    # always a 2-D grid, even with one row

    def show(ax, data, kind, title, header):
        """Like panel(), with the column heading above the first row and thinner borders."""
        if kind == "rgb":
            ax.imshow(data, interpolation="lanczos")   # smooth downscaling for photos
        elif kind == "mask":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            lo, hi = np.percentile(data, [1, 99])
            ax.imshow(data, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title((header + "\n" if header else "") + title, fontsize=FS, pad=2, linespacing=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():             # the panel's border lines
            sp.set_linewidth(0.4); sp.set_color("#8a8984")

    for r, (row, gt, lg, lab, sm) in enumerate(zip(ex["rows"], ex["gts"], ex["lgs"], ex["labels"], sam)):   # one figure row per image
        img = np.asarray(Image.open(row["image"]).convert("RGB"))
        ela = ela_of(row["image"])                # rung 1 score map
        e1 = np.asarray(ela >= TAU_ELA, bool)     # rung 1 mask
        e2 = np.asarray(solve_graphcut(ela, TAU_ELA, BETA_ELA), bool)   # rung 2 mask
        u4 = np.asarray(solve_graphcut(lg, tau, 0.0), bool)             # rung 4 mask (beta = 0 = threshold)
        u5 = np.asarray(solve_graphcut(lg, tau, beta), bool)            # rung 5 mask
        h = (lambda c: c) if r == 0 else (lambda c: "")                  # headings on the top row only
        panels = [(img, "rgb", "", h("image")),
                  (gt, "mask", f"{gt.mean() * 100:.1f}% tampered", h("ground truth")),
                  (ela, "heat", f"F1 {f1(e1, gt):.2f}", h("ELA (rung 1)")),
                  (e2, "mask", f"F1 {f1(e2, gt):.2f}", h("ELA + MRF"))]
        if have_sam:
            panels.append((sm[0], "mask", f"F1 {f1(sm[0], gt):.2f}", h("SAM2 (rung 3)")))
        panels += [(1 / (1 + np.exp(-lg)), "heat", f"F1 {f1(u4, gt):.2f}", h("U-Net (rung 4)")),   # sigmoid: logits -> probabilities
                   (u5, "mask", f"F1 {f1(u5, gt):.2f}", h("U-Net + MRF"))]
        for c, (data, kind, title, header) in enumerate(panels):
            show(axes[r, c], data, kind, title, header)
        if have_sam:                          # the oracle's best proposal, outlined in gold
            axes[r, 4].contour(sm[1].astype(float), levels=[0.5], colors="#e3a21a", linewidths=0.6)
        axes[r, 0].set_ylabel(lab, fontsize=FS)   # "best", "p67", ... at the left
    fig.subplots_adjust(left=0.03, right=0.995, top=0.93, bottom=0.01, wspace=0.06, hspace=0.32)   # tight margins
    save(fig, "splice")
    print(f"tau={tau}, beta={beta} ({TAG}); ELA tau={TAU_ELA}, beta={BETA_ELA}")
    for lab, row in zip(ex["labels"], ex["rows"]):   # which images were picked
        print(f"  {lab}: {os.path.basename(row['image'])}")
    s = ex["scores"]
    print(f"rung 5 per-image F1 on fold {ex['test_fold']}: "
          f"mean {s.mean():.4f}, median {np.median(s):.4f}, min {s.min():.4f}, max {s.max():.4f}")


def copymove():
    """Copy-move figure: image, truth, ELA, ELA+MRF on random copy-move images."""
    mf = os.environ.get("CM_MANIFEST", os.path.join(ROOT, "copymove_manifest.csv"))
    rows = list(csv.DictReader(open(mf)))         # every copy-move image
    pick = list(np.random.default_rng(0).permutation(len(rows))[:N_ROWS])   # seeded random picks

    fig, axes = plt.subplots(len(pick), 4, figsize=(9, 2.3 * len(pick)))
    axes = np.atleast_2d(axes)
    f1s = []                                      # rung 2 F1 of each example
    for r, i in enumerate(pick):
        img = np.asarray(Image.open(rows[i]["image"]).convert("RGB"))
        gt = np.asarray(Image.open(rows[i]["mask"]).convert("L")) > 127
        ela = ela_of(rows[i]["image"])
        e1 = np.asarray(ela >= TAU_ELA, bool)     # rung 1 mask
        e2 = np.asarray(solve_graphcut(ela, TAU_ELA, BETA_ELA), bool)   # rung 2 mask
        f1s.append(f1(e2, gt))
        panel(axes[r, 0], img, "rgb", os.path.basename(rows[i]["image"])[:26])   # name cut to 26 characters
        panel(axes[r, 1], gt, "mask", f"ground truth  ({gt.mean()*100:.1f}%)")
        panel(axes[r, 2], ela, "heat", f"ELA   F1 {f1(e1, gt):.3f}")
        panel(axes[r, 3], e2, "mask", f"ELA+MRF   F1 {f1(e2, gt):.3f}")

    fig.suptitle("Copy-move, splicing-selected (tau, beta) applied unchanged. Random examples; "
                 "set-wide micro F1:\nELA 0.128, ELA+MRF 0.139, chance 0.092 "
                 "(pooled tampered rate 4.85%).", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])        # leave room for the title
    save(fig, "copymove")
    print(f"ELA+MRF F1 on these {len(pick)} examples: {np.mean(f1s):.4f} "
          f"(set-wide micro: ELA 0.1278, ELA+MRF 0.1391, chance 0.0924)")


def save(fig, mode):
    """Save the figure as PNG and PDF."""
    os.makedirs("results", exist_ok=True)
    for ext in ("png", "pdf"):
        p = f"results/fig_qualitative_{mode}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")   # trim empty space around it
        print(f"wrote {p}")


if __name__ == "__main__":
    {"splice": splice, "copymove": copymove}[MODE]()   # run the function named by MODE
