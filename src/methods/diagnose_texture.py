"""Is the rung-1 score tracking texture rather than noise?"""
import csv, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from noise_residual import score_map

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))[:200]

cors = []
for r in rows:
    g = np.array(Image.open(r["image"]).convert("L"), float) / 255.0
    gx = ndimage.sobel(g, 0); gy = ndimage.sobel(g, 1)
    texture = ndimage.uniform_filter(np.hypot(gx, gy), size=32)
    s = score_map(np.array(Image.open(r["image"]).convert("RGB")))
    cors.append(np.corrcoef(s.ravel(), texture.ravel())[0, 1])

cors = np.array(cors)
print(f"corr(score, local texture energy): mean {cors.mean():+.3f}, median {np.median(cors):+.3f}")
print(f"|corr| > 0.3 in {(np.abs(cors) > 0.3).mean():.1%} of images")
