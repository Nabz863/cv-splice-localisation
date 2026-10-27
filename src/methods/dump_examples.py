"""Exploratory: draw 6 random images, their masks and every rung-1 cue -> results/rung1_examples.png."""
import csv, os, sys                               # standard-library helpers
import numpy as np                                # arrays and maths
import matplotlib; matplotlib.use("Agg")          # draw to files, no screen needed
import matplotlib.pyplot as plt                   # plotting
from PIL import Image                             # reads image files

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from noise_variants import VARIANTS               # the candidate cues A-E (name -> function)

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
rows = list(csv.DictReader(open(os.path.join(ROOT, "splice_manifest.csv"))))   # every splicing image
rows = list(np.random.default_rng(1).permutation(rows))[:6]       # 6 random images

# ---- One row per image: image, mask, then one panel per cue ----
fig, axes = plt.subplots(len(rows), 2 + len(VARIANTS),
                         figsize=(3 * (2 + len(VARIANTS)), 3 * len(rows)))   # 3 inches per panel
for i, r in enumerate(rows):                      # i = row of the figure
    img = np.array(Image.open(r["image"]).convert("RGB"))         # load the image
    truth = np.array(Image.open(r["mask"]).convert("L")) > 127    # true mask
    axes[i, 0].imshow(img); axes[i, 0].set_title("image" if i == 0 else "")          # column 0: image
    axes[i, 1].imshow(truth, cmap="gray"); axes[i, 1].set_title("mask" if i == 0 else "")   # column 1: mask
    for j, (k, f) in enumerate(VARIANTS.items()): # each cue
        s = f(img)                                # its score map
        axes[i, 2 + j].imshow(np.clip(s, 0, 5), cmap="inferno")     # clip z-scores at 5 for display
        if i == 0: axes[i, 2 + j].set_title(k, fontsize=9)          # titles on the top row only
    for a in axes[i]: a.set_axis_off()            # hide axis ticks

os.makedirs("results", exist_ok=True)             # create results/ if missing
plt.tight_layout(); plt.savefig("results/rung1_examples.png", dpi=90)   # save the figure
print("wrote results/rung1_examples.png")
