"""Pair each CASIA v2 tampered image with its mask and write the manifest CSV.

Usage:  python src/data/build_manifest.py          (splicing  -> splice_manifest.csv)
        python src/data/build_manifest.py Tp_S_    (copy-move -> copymove_manifest.csv)
"""
import os, csv, random, sys
from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path(os.environ["DATASETS"]) / "casia2"                    # dataset root folder
IMG_DIR = ROOT / "images" / "CASIA2.0_revised" / "Tp"             # tampered images
MASK_DIR = ROOT / "masks" / "CASIA2.0_Groundtruth"                # Pham et al. masks

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "Tp_D_"           # Tp_D_ = splicing, Tp_S_ = copy-move
OUT_NAME = "splice_manifest.csv" if PREFIX == "Tp_D_" else "copymove_manifest.csv"
EXCL_NAME = ("excluded_size_mismatch.csv" if PREFIX == "Tp_D_"
             else "excluded_size_mismatch_copymove.csv")

# ---- Pair images with masks ----
# mask file "X_gt.png" belongs to image "X.*": index masks by X
mask_files = {p.stem.removesuffix("_gt"): p for p in MASK_DIR.glob("*_gt.png")}

records, excluded = [], []
for img_path in sorted(IMG_DIR.glob(PREFIX + "*")):
    mask_path = mask_files.get(img_path.stem)
    if mask_path is None:                                         # no mask for this image: skip it
        continue
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        if im.size != mk.size:                                    # mask doesn't line up with the image
            reason = "transposed" if im.size == mk.size[::-1] else "size mismatch"
            excluded.append((img_path.stem, im.size, mk.size, reason))
            continue
        records.append({"image": str(img_path.resolve()), "mask": str(mask_path.resolve()),
                         "img_size": im.size, "mask_size": mk.size})

print(f"paired and usable: {len(records)}")
print(f"excluded (dimension mismatch): {len(excluded)}")
for b, s1, s2, reason in excluded:
    print(f"  {b}: image {s1} vs mask {s2} ({reason})")

# ---- Write outputs ----
with open(ROOT / EXCL_NAME, "w", newline="") as f:                # record what was excluded and why
    w = csv.writer(f)
    w.writerow(["basename", "img_size", "mask_size", "reason"])
    w.writerows(excluded)

# ---- Unused 80/10/10 split column (experiments use the folds from stratify_folds.py) ----
random.seed(0)
random.shuffle(records)
n = len(records)
n_train, n_val = int(0.8 * n), int(0.1 * n)
for r, s in zip(records, ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)):
    r["split"] = s

with open(ROOT / OUT_NAME, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "mask", "img_size", "mask_size", "split"])
    w.writeheader()
    w.writerows(records)

print(Counter(r["split"] for r in records))
