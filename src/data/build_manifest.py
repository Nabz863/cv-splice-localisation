"""Pair each CASIA v2 tampered image with its mask and write the manifest CSV.

Usage:  python src/data/build_manifest.py          (splicing  -> splice_manifest.csv)
        python src/data/build_manifest.py Tp_S_    (copy-move -> copymove_manifest.csv)
"""
import os, csv, random, sys   # standard-library helpers
from pathlib import Path   # path objects (Path("a") / "b")
from PIL import Image   # reads image sizes
from collections import Counter   # counts how many of each value

ROOT = Path(os.environ["DATASETS"]) / "casia2"                    # dataset root folder
IMG_DIR = ROOT / "images" / "CASIA2.0_revised" / "Tp"             # tampered images
MASK_DIR = ROOT / "masks" / "CASIA2.0_Groundtruth"                # Pham et al. masks

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "Tp_D_"           # Tp_D_ = splicing, Tp_S_ = copy-move
OUT_NAME = "splice_manifest.csv" if PREFIX == "Tp_D_" else "copymove_manifest.csv"   # output file name
EXCL_NAME = ("excluded_size_mismatch.csv" if PREFIX == "Tp_D_"   # list of images left out
             else "excluded_size_mismatch_copymove.csv")

# ---- Pair images with masks ----
# mask file "X_gt.png" belongs to image "X.*": index masks by X
mask_files = {p.stem.removesuffix("_gt"): p for p in MASK_DIR.glob("*_gt.png")}

records, excluded = [], []   # usable pairs; pairs left out
for img_path in sorted(IMG_DIR.glob(PREFIX + "*")):   # every tampered image of this type, sorted by name
    mask_path = mask_files.get(img_path.stem)   # its mask, or None
    if mask_path is None:                                         # no mask for this image: skip it
        continue
    with Image.open(img_path) as im, Image.open(mask_path) as mk:   # opening reads only the header (size), not the pixels
        if im.size != mk.size:                                    # mask doesn't line up with the image
            reason = "transposed" if im.size == mk.size[::-1] else "size mismatch"   # width and height swapped, or just different
            excluded.append((img_path.stem, im.size, mk.size, reason))   # remember it and why
            continue
        records.append({"image": str(img_path.resolve()), "mask": str(mask_path.resolve()),   # full paths to image and mask
                         "img_size": im.size, "mask_size": mk.size})

print(f"paired and usable: {len(records)}")   # report the counts
print(f"excluded (dimension mismatch): {len(excluded)}")
for b, s1, s2, reason in excluded:   # list each excluded image
    print(f"  {b}: image {s1} vs mask {s2} ({reason})")

# ---- Write outputs ----
with open(ROOT / EXCL_NAME, "w", newline="") as f:                # record what was excluded and why
    w = csv.writer(f)
    w.writerow(["basename", "img_size", "mask_size", "reason"])   # header
    w.writerows(excluded)   # one row per excluded image

# ---- Unused 80/10/10 split column (experiments use the folds from stratify_folds.py) ----
random.seed(0)   # same shuffle every run
random.shuffle(records)   # random order
n = len(records)   # number of usable pairs
n_train, n_val = int(0.8 * n), int(0.1 * n)   # 80% / 10% (the rest is test)
for r, s in zip(records, ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)):   # label each record
    r["split"] = s

with open(ROOT / OUT_NAME, "w", newline="") as f:   # write the manifest
    w = csv.DictWriter(f, fieldnames=["image", "mask", "img_size", "mask_size", "split"])   # columns
    w.writeheader()   # header row
    w.writerows(records)   # one row per image

print(Counter(r["split"] for r in records))   # how many in each split
