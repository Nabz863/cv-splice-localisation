"""Rung 1, step 2: pick ELA's JPEG quality and window by pixel AUC on 300 non-test images.

HOLDOUT=k excludes fold k (default 0). Writes results/ela_tuning*.csv.
Usage:  python src/methods/tune_ela.py
"""
import csv, os, sys   # standard-library helpers: CSV files, paths, import path
from concurrent.futures import ProcessPoolExecutor   # runs a function on many images in parallel
import numpy as np   # arrays and maths
from PIL import Image   # reads image files

sys.path.insert(0, os.path.dirname(__file__))   # so files in this folder can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so src/ packages (eval, utils) can be imported
from noise_variants import v_ela   # rung 1's ELA score map
from eval.metrics import pixel_auc   # per-image pixel AUC

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder ($DATASETS is set by the user)
HOLDOUT = int(os.environ.get("HOLDOUT", 0))     # test fold to leave out
TAG = "" if HOLDOUT == 0 else f"_holdout{HOLDOUT}"   # output-file suffix, e.g. _holdout2 (empty for fold 0)
GRID = [(q, w) for q in (70, 80, 90, 95, 98) for w in (8, 16, 32)]   # (JPEG quality, window)


# ---- Which images to use ----
def sample_rows():
    """300 images from the non-held-out folds, in a fixed random order."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))   # every splicing image (one dict per row)
    rows = [r for r in rows if int(r["fold"]) != HOLDOUT]   # drop the test fold so tuning never sees it
    return list(np.random.default_rng(0).permutation(rows))[:300]   # shuffle with seed 0 (same every run), keep 300


# ---- Score one image at every setting ----
def one(r):
    """Pixel AUC of ELA at every grid setting, for one image."""
    img = np.array(Image.open(r["image"]).convert("RGB"))   # load the image as an RGB array
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127   # true mask (True = spliced)
    return [pixel_auc(v_ela(img, quality=q, w=w), truth) for q, w in GRID]   # one AUC per (quality, window)


if __name__ == "__main__":
    # ---- Run, print the table, keep the best setting ----
    rows = sample_rows()   # the images to score
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:   # images in parallel
        out = list(ex.map(one, rows, chunksize=4))   # one list of 15 AUCs per image

    print(f"{'quality':>8}{'window':>8}{'mean AUC':>11}{'median':>9}")   # column titles
    best = None   # best (mean AUC, quality, window) so far
    os.makedirs("results", exist_ok=True)   # create results/ if missing
    with open(f"results/ela_tuning{TAG}.csv", "w", newline="") as f:
        w_ = csv.writer(f)   # named w_ because w is the window size below
        w_.writerow(["quality", "window", "n_images", "mean_auc", "median_auc"])   # CSV header
        for i, (q, w) in enumerate(GRID):   # each setting (i = its column in out)
            a = np.array([r[i] for r in out if r[i] is not None])             # this setting, all images
            print(f"{q:>8}{w:>8}{a.mean():>11.4f}{np.median(a):>9.4f}")   # print this setting's mean and median
            w_.writerow([q, w, len(a), f"{a.mean():.6f}", f"{np.median(a):.6f}"])   # and save them
            if best is None or a.mean() > best[0]:                             # keep the best mean AUC
                best = (a.mean(), q, w)   # new best
    print(f"\nbest: quality={best[1]}, window={best[2]}  (AUC {best[0]:.4f})")   # the chosen ELA setting
    print(f"wrote results/ela_tuning{TAG}.csv")   # report what was written
