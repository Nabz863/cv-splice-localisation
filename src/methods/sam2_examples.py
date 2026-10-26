"""SAM2 masks (rung-3 union and oracle) for the few images in the qualitative figure (GPU).

Writes results/sam2_examples/<image>.npz; run fig_qualitative.py afterwards.
Usage:  SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt python src/methods/sam2_examples.py
"""
import csv, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from fig_qualitative import pick_examples, f1, TAG, SAM2_DIR
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_base_plus.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"        # as run_sam2.py


def size_band(tag):
    """Size window this pair's validation fold selected for rung 3."""
    k, v = int(tag[1]), int(tag[3])
    for r in csv.DictReader(open("results/rung3_sam2_20pairs.csv")):
        if int(r["test_fold"]) == k and int(r["val_fold"]) == v:
            return float(r["min_frac"]), float(r["max_frac"])
    raise SystemExit(f"no rung 3 row for {tag} in results/rung3_sam2_20pairs.csv")


def iou(a, b):
    """IoU of two masks."""
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


if __name__ == "__main__":
    if CHECKPOINT and not os.path.exists(CHECKPOINT) and not os.environ.get("SAM2_RANDOM_WEIGHTS"):
        raise SystemExit(f"checkpoint {CHECKPOINT} not found; set SAM2_CHECKPOINT")
    lo, hi = size_band(TAG)
    ex = pick_examples()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = None if os.environ.get("SAM2_RANDOM_WEIGHTS") else CHECKPOINT   # tests only
    gen = SAM2AutomaticMaskGenerator(build_sam2(MODEL_CFG, ckpt, device=dev))
    os.makedirs(SAM2_DIR, exist_ok=True)
    print(f"{TAG}: size band [{lo}, {hi}] | {len(ex['rows'])} images | {dev}", flush=True)
    for lab, row, gt in zip(ex["labels"], ex["rows"], ex["gts"]):
        t0 = time.time()
        img = np.array(Image.open(row["image"]).convert("RGB"))
        truth = np.array(Image.open(row["mask"]).convert("L")) > 127
        if truth.shape != gt.shape or not np.array_equal(truth, gt):
            raise SystemExit(f"{row['image']}: mask differs from the cached ground truth")
        with torch.inference_mode():
            masks = gen.generate(img)
        segs = [m["segmentation"].astype(bool) for m in masks]
        fracs = [float(m["area"]) / truth.size for m in masks]           # as run_sam2.py
        union = np.zeros_like(truth)                                      # rung 3's prediction
        for s, fr in zip(segs, fracs):
            if lo <= fr <= hi:
                union |= s
        oracle = max(segs, key=lambda s: iou(s, truth)) if segs else np.zeros_like(truth)   # best single proposal
        stem = os.path.basename(row["image"]).rsplit(".", 1)[0]
        np.savez_compressed(os.path.join(SAM2_DIR, stem + ".npz"), shape=np.array(truth.shape),
                            union=np.packbits(union), oracle=np.packbits(oracle),
                            min_frac=lo, max_frac=hi, n_masks=len(segs))
        print(f"  {lab:>6} {stem}: {len(segs)} proposals, union F1 {f1(union, truth):.3f}, "
              f"oracle F1 {f1(oracle, truth):.3f} ({time.time() - t0:.0f}s)", flush=True)
    print("next: python src/methods/fig_qualitative.py")
