"""Rung 1, step 1: which physically motivated cue carries the splice signal?

Ranks the five candidate score maps in noise_variants.VARIANTS by pixel AUC
(threshold-free) on 400 images drawn from folds 1-4 with a fixed seed. Fold 0 is
excluded so that one test fold is untouched by this choice.

HOLDOUT=k leaves out fold k instead (outputs gain the suffix _holdout{k}).
Running k = 0..4 checks that the choice does not depend on which test fold
was excluded, i.e. that no test fold influenced the cue or its settings.

CPU, a few minutes. Writes results/cue_auc.csv (one row per cue) and
results/cue_auc_per_image.csv (one row per image), which build_tables.py reads.

Usage (from the repo root):
    python src/methods/compare_variants.py
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
HOLDOUT = int(os.environ.get("HOLDOUT", 0))     # the test fold kept out of this choice
TAG = "" if HOLDOUT == 0 else f"_holdout{HOLDOUT}"
N = 400


def sample_rows():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
    rows = [r for r in rows if int(r["fold"]) != HOLDOUT]
    return list(np.random.default_rng(0).permutation(rows))[:N]


def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    return {k: pixel_auc(f(img), truth) for k, f in VARIANTS.items()}


if __name__ == "__main__":
    rows = sample_rows()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:
        out = list(ex.map(one, rows, chunksize=4))

    os.makedirs("results", exist_ok=True)
    with open(f"results/cue_auc_per_image{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image"] + list(VARIANTS))
        for r, o in zip(rows, out):
            w.writerow([os.path.basename(r["image"])] + ["" if o[k] is None else f"{o[k]:.6f}"
                                                        for k in VARIANTS])

    print(f"pixel AUC over {len(out)} images (0.5 = no signal)\n")
    print(f"{'variant':<18}{'mean':>8}{'median':>9}{'>0.5':>8}")
    with open(f"results/cue_auc{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_images", "mean_auc", "median_auc", "frac_above_0p5"])
        for k in VARIANTS:
            a = np.array([o[k] for o in out if o[k] is not None])
            print(f"{k:<18}{a.mean():>8.4f}{np.median(a):>9.4f}{(a > 0.5).mean():>8.1%}")
            w.writerow([k, len(a), f"{a.mean():.6f}", f"{np.median(a):.6f}",
                        f"{(a > 0.5).mean():.6f}"])
    print(f"\nwrote results/cue_auc{TAG}.csv, results/cue_auc_per_image{TAG}.csv")
