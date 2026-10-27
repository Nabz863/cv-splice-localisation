"""Exploratory: pixel AUC of the noise cue on every image, and the all-tampered F1 to beat."""
import csv, os, sys                               # standard-library helpers
from concurrent.futures import ProcessPoolExecutor   # runs a function on many images in parallel
import numpy as np                                # arrays and maths
from PIL import Image                             # reads image files

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from noise_residual import score_map              # the sensor-noise score map being checked

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))   # every splicing image


# ---- One image: pixel AUC of the score map, plus the share of tampered pixels ----

def one(r):
    img = np.array(Image.open(r["image"]).convert("RGB"))         # load the image
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127    # true mask (True = spliced)
    s = score_map(img).ravel()                    # score per pixel, flattened to 1-D
    t = truth.ravel()                             # truth per pixel, flattened the same way
    if t.all() or not t.any():                    # AUC needs both classes present
        return None
    # pixel AUC from ranks, on a fixed sample of 20,000 pixels
    idx = np.random.default_rng(0).choice(len(t), size=min(20000, len(t)), replace=False)   # seeded sample
    s, t = s[idx], t[idx]                         # keep only the sampled pixels
    order = s.argsort()                           # pixel positions sorted by score
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)       # rank 1 = lowest score
    npos, nneg = int(t.sum()), int((~t).sum())    # number of tampered / authentic pixels
    if npos == 0 or nneg == 0:                    # the sample may have missed one class
        return None
    auc = (ranks[t].sum() - npos * (npos + 1) / 2) / (npos * nneg)   # Mann-Whitney formula for AUC
    return auc, truth.mean()                      # truth.mean() = fraction of pixels tampered


# ---- Run on every image and summarise ----

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 12)) as ex:   # up to 12 processes
        out = [o for o in ex.map(one, rows, chunksize=8) if o is not None]     # drop skipped images

    aucs = np.array([o[0] for o in out])          # AUC per image
    rates = np.array([o[1] for o in out])         # tampered fraction per image

    print(f"images scored: {len(aucs)}")
    print(f"\nmean tampered-pixel rate: {rates.mean():.4f}")
    p = rates.mean()                              # average tampered fraction
    print(f"F1 if we predict EVERY pixel tampered: {2*p/(1+p):.4f}   <-- the number to beat")   # chance F1
    print(f"\npixel-level AUC of the score map")
    print(f"  mean   {aucs.mean():.4f}   (0.5 = no signal)")
    print(f"  median {np.median(aucs):.4f}")
    print(f"  frac of images above 0.5: {(aucs > 0.5).mean():.3f}")   # share of images better than chance
