import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from noise_variants import VARIANTS

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))
rows = list(np.random.default_rng(1).permutation(rows))[:6]

fig, axes = plt.subplots(len(rows), 2 + len(VARIANTS),
                         figsize=(3 * (2 + len(VARIANTS)), 3 * len(rows)))
for i, r in enumerate(rows):
    img = np.array(Image.open(r["image"]).convert("RGB"))
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127
    axes[i, 0].imshow(img); axes[i, 0].set_title("image" if i == 0 else "")
    axes[i, 1].imshow(truth, cmap="gray"); axes[i, 1].set_title("mask" if i == 0 else "")
    for j, (k, f) in enumerate(VARIANTS.items()):
        s = f(img)
        axes[i, 2 + j].imshow(np.clip(s, 0, 5), cmap="inferno")
        if i == 0: axes[i, 2 + j].set_title(k, fontsize=9)
    for a in axes[i]: a.set_axis_off()

os.makedirs("results", exist_ok=True)
plt.tight_layout(); plt.savefig("results/rung1_examples.png", dpi=90)
print("wrote results/rung1_examples.png")
