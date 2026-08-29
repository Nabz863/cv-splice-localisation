"""Does pixel_auc report what we think? Oracle should be 1.0, noise 0.5."""
import csv, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from compare_variants import pixel_auc

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))[:50]

orc, rnd = [], []
for r in rows:
    t = np.array(Image.open(r["mask"]).convert("L")) > 127
    orc.append(pixel_auc(ndimage.gaussian_filter(t.astype(float), 3), t))
    rnd.append(pixel_auc(np.random.default_rng(0).random(t.shape), t))

print(f"oracle (blurred GT): {np.mean([o for o in orc if o]):.4f}   expect ~1.0")
print(f"random:              {np.mean([o for o in rnd if o]):.4f}   expect ~0.5")
