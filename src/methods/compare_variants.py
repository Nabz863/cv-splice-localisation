"""Rung 1, step 1: rank the five cues by pixel AUC on 400 non-test images.

HOLDOUT=k excludes fold k (default 0). Writes results/cue_auc*.csv.
Usage:  python src/methods/compare_variants.py
"""
import csv, os, sys   # standard-library helpers: CSV files, paths, import path
from concurrent.futures import ProcessPoolExecutor   # runs a function on many images in parallel
import numpy as np   # arrays and maths
from PIL import Image   # reads image files

sys.path.insert(0, os.path.dirname(__file__))   # so files in this folder can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so src/ packages (eval, utils) can be imported
from noise_variants import VARIANTS   # the candidate cues A-E (name -> function)
from eval.metrics import pixel_auc   # per-image pixel AUC

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder ($DATASETS is set by the user)
HOLDOUT = int(os.environ.get("HOLDOUT", 0))     # test fold to leave out
TAG = "" if HOLDOUT == 0 else f"_holdout{HOLDOUT}"   # output-file suffix, e.g. _holdout2 (empty for fold 0)
N = 400                                         # number of images to score


# ---- Which images to use ----
def sample_rows():
    """N images from the four non-held-out folds, in a fixed random order."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))   # every splicing image (one dict per row)
    rows = [r for r in rows if int(r["fold"]) != HOLDOUT]   # drop the test fold so tuning never sees it
    return list(np.random.default_rng(0).permutation(rows))[:N]   # shuffle with seed 0 (same every run), keep N


# ---- Score one image ----
def one(r):
    """Pixel AUC of every cue on one image."""
    img = np.array(Image.open(r["image"]).convert("RGB"))   # load the image as an RGB array
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127      # mask -> True/False
    return {k: pixel_auc(f(img), truth) for k, f in VARIANTS.items()}   # AUC of each cue's map on this image


if __name__ == "__main__":
    # ---- Run on all sampled images and save ----
    rows = sample_rows()   # the images to score
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:   # images in parallel
        out = list(ex.map(one, rows, chunksize=4))   # one dict of AUCs per image, in order

    os.makedirs("results", exist_ok=True)   # create results/ if missing
    with open(f"results/cue_auc_per_image{TAG}.csv", "w", newline="") as f:     # per-image AUCs
        w = csv.writer(f)
        w.writerow(["image"] + list(VARIANTS))   # header: image name then one column per cue
        for r, o in zip(rows, out):   # each image with its AUCs
            w.writerow([os.path.basename(r["image"])] + ["" if o[k] is None else f"{o[k]:.6f}"   # blank where AUC is undefined
                                                        for k in VARIANTS])

    print(f"pixel AUC over {len(out)} images (0.5 = no signal)\n")   # summary header
    print(f"{'variant':<18}{'mean':>8}{'median':>9}{'>0.5':>8}")   # column titles
    with open(f"results/cue_auc{TAG}.csv", "w", newline="") as f:              # one summary row per cue
        w = csv.writer(f)
        w.writerow(["variant", "n_images", "mean_auc", "median_auc", "frac_above_0p5"])
        for k in VARIANTS:   # each cue
            a = np.array([o[k] for o in out if o[k] is not None])              # skip one-class images
            print(f"{k:<18}{a.mean():>8.4f}{np.median(a):>9.4f}{(a > 0.5).mean():>8.1%}")   # mean, median, share of images above chance
            w.writerow([k, len(a), f"{a.mean():.6f}", f"{np.median(a):.6f}",   # same numbers to the CSV
                        f"{(a > 0.5).mean():.6f}"])
    print(f"\nwrote results/cue_auc{TAG}.csv, results/cue_auc_per_image{TAG}.csv")   # report what was written
