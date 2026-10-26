"""Rung 1, step 1: rank the five cues by pixel AUC on 400 non-test images.

HOLDOUT=k excludes fold k (default 0). Writes results/cue_auc*.csv.
Usage:  python src/methods/compare_variants.py
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from noise_variants import VARIANTS
from eval.metrics import pixel_auc

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
HOLDOUT = int(os.environ.get("HOLDOUT", 0))     # test fold to leave out
TAG = "" if HOLDOUT == 0 else f"_holdout{HOLDOUT}"
N = 400                                         # number of images to score


def sample_rows():
    """N images from the four non-held-out folds, in a fixed random order."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
    rows = [r for r in rows if int(r["fold"]) != HOLDOUT]
    return list(np.random.default_rng(0).permutation(rows))[:N]


def one(r):
    """Pixel AUC of every cue on one image."""
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127      # mask -> True/False
    return {k: pixel_auc(f(img), truth) for k, f in VARIANTS.items()}


if __name__ == "__main__":
    rows = sample_rows()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:   # images in parallel
        out = list(ex.map(one, rows, chunksize=4))

    os.makedirs("results", exist_ok=True)
    with open(f"results/cue_auc_per_image{TAG}.csv", "w", newline="") as f:     # per-image AUCs
        w = csv.writer(f)
        w.writerow(["image"] + list(VARIANTS))
        for r, o in zip(rows, out):
            w.writerow([os.path.basename(r["image"])] + ["" if o[k] is None else f"{o[k]:.6f}"
                                                        for k in VARIANTS])

    print(f"pixel AUC over {len(out)} images (0.5 = no signal)\n")
    print(f"{'variant':<18}{'mean':>8}{'median':>9}{'>0.5':>8}")
    with open(f"results/cue_auc{TAG}.csv", "w", newline="") as f:              # one summary row per cue
        w = csv.writer(f)
        w.writerow(["variant", "n_images", "mean_auc", "median_auc", "frac_above_0p5"])
        for k in VARIANTS:
            a = np.array([o[k] for o in out if o[k] is not None])              # skip one-class images
            print(f"{k:<18}{a.mean():>8.4f}{np.median(a):>9.4f}{(a > 0.5).mean():>8.1%}")
            w.writerow([k, len(a), f"{a.mean():.6f}", f"{np.median(a):.6f}",
                        f"{(a > 0.5).mean():.6f}"])
    print(f"\nwrote results/cue_auc{TAG}.csv, results/cue_auc_per_image{TAG}.csv")
