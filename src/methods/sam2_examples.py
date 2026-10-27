"""SAM2 masks (rung-3 union and oracle) for the few images in the qualitative figure (GPU).

Writes results/sam2_examples/<image>.npz; run fig_qualitative.py afterwards.
Usage:  SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt python src/methods/sam2_examples.py
"""
import csv, os, sys, time                         # standard-library helpers
import numpy as np                                # arrays and maths
import torch                                      # deep-learning library SAM2 runs on
from PIL import Image                             # reads image files

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from fig_qualitative import pick_examples, f1, TAG, SAM2_DIR   # same images and pair as the figure
from sam2.build_sam import build_sam2             # builds the SAM2 model
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator   # proposes masks with no prompt

CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_base_plus.pt")   # weights file
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"        # as run_sam2.py


def size_band(tag):
    """Size window this pair's validation fold selected for rung 3."""
    k, v = int(tag[1]), int(tag[3])               # tag looks like "t0v1": test fold 0, val fold 1
    for r in csv.DictReader(open("results/rung3_sam2_20pairs.csv")):
        if int(r["test_fold"]) == k and int(r["val_fold"]) == v:
            return float(r["min_frac"]), float(r["max_frac"])
    raise SystemExit(f"no rung 3 row for {tag} in results/rung3_sam2_20pairs.csv")


def iou(a, b):
    """IoU of two masks."""
    u = (a | b).sum()                             # pixels in either mask
    return (a & b).sum() / u if u else 1.0        # overlap / union


if __name__ == "__main__":
    if CHECKPOINT and not os.path.exists(CHECKPOINT) and not os.environ.get("SAM2_RANDOM_WEIGHTS"):
        raise SystemExit(f"checkpoint {CHECKPOINT} not found; set SAM2_CHECKPOINT")
    lo, hi = size_band(TAG)                       # the window rung 3 used for this pair
    ex = pick_examples()                          # the figure's images and their masks
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = None if os.environ.get("SAM2_RANDOM_WEIGHTS") else CHECKPOINT   # tests only
    gen = SAM2AutomaticMaskGenerator(build_sam2(MODEL_CFG, ckpt, device=dev))   # load SAM2
    os.makedirs(SAM2_DIR, exist_ok=True)
    print(f"{TAG}: size band [{lo}, {hi}] | {len(ex['rows'])} images | {dev}", flush=True)
    for lab, row, gt in zip(ex["labels"], ex["rows"], ex["gts"]):   # lab = row label in the figure
        t0 = time.time()
        img = np.array(Image.open(row["image"]).convert("RGB"))
        truth = np.array(Image.open(row["mask"]).convert("L")) > 127
        if truth.shape != gt.shape or not np.array_equal(truth, gt):   # must match the figure's mask
            raise SystemExit(f"{row['image']}: mask differs from the cached ground truth")
        with torch.inference_mode():              # no gradients needed
            masks = gen.generate(img)             # automatic proposals
        segs = [m["segmentation"].astype(bool) for m in masks]         # each proposal as a mask
        fracs = [float(m["area"]) / truth.size for m in masks]           # as run_sam2.py
        union = np.zeros_like(truth)                                      # rung 3's prediction
        for s, fr in zip(segs, fracs):
            if lo <= fr <= hi:                    # inside the window: add it
                union |= s
        oracle = max(segs, key=lambda s: iou(s, truth)) if segs else np.zeros_like(truth)   # best single proposal
        stem = os.path.basename(row["image"]).rsplit(".", 1)[0]          # file name without extension
        np.savez_compressed(os.path.join(SAM2_DIR, stem + ".npz"), shape=np.array(truth.shape),
                            union=np.packbits(union), oracle=np.packbits(oracle),   # packbits: 8 pixels per byte
                            min_frac=lo, max_frac=hi, n_masks=len(segs))
        print(f"  {lab:>6} {stem}: {len(segs)} proposals, union F1 {f1(union, truth):.3f}, "
              f"oracle F1 {f1(oracle, truth):.3f} ({time.time() - t0:.0f}s)", flush=True)
    print("next: python src/methods/fig_qualitative.py")
