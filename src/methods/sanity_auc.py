"""Sanity check for pixel_auc: a blurred ground truth should score ~1.0, random noise ~0.5."""
import csv, os, sys                               # standard-library helpers
import numpy as np                                # arrays and maths
from PIL import Image                             # reads image files
from scipy import ndimage                         # image filters (Gaussian blur)

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so src/ packages can be imported
from eval.metrics import pixel_auc                # the function being checked

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))[:50]   # first 50 images

orc, rnd = [], []                                 # AUCs of the "oracle" and the random map
for r in rows:
    t = np.array(Image.open(r["mask"]).convert("L")) > 127                 # true mask
    orc.append(pixel_auc(ndimage.gaussian_filter(t.astype(float), 3), t))  # blurred truth: near-perfect score
    rnd.append(pixel_auc(np.random.default_rng(0).random(t.shape), t))     # random numbers: no signal

print(f"oracle (blurred GT): {np.mean([o for o in orc if o]):.4f}   expect ~1.0")   # skip None (one-class)
print(f"random:              {np.mean([o for o in rnd if o]):.4f}   expect ~0.5")
