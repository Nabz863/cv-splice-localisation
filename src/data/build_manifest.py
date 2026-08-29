import os, csv, random
from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path(os.environ["DATASETS"]) / "casia2"
IMG_DIR = ROOT / "images" / "CASIA2.0_revised" / "Tp"
MASK_DIR = ROOT / "masks" / "CASIA2.0_Groundtruth"

mask_files = {p.stem.removesuffix("_gt"): p for p in MASK_DIR.glob("*_gt.png")}

records, excluded = [], []
for img_path in sorted(IMG_DIR.glob("Tp_D_*")):
    mask_path = mask_files.get(img_path.stem)
    if mask_path is None:
        continue
    with Image.open(img_path) as im, Image.open(mask_path) as mk:
        if im.size != mk.size:
            reason = "transposed" if im.size == mk.size[::-1] else "size mismatch"
            excluded.append((img_path.stem, im.size, mk.size, reason))
            continue
        records.append({"image": str(img_path.resolve()), "mask": str(mask_path.resolve()),
                         "img_size": im.size, "mask_size": mk.size})

print(f"paired and usable: {len(records)}")
print(f"excluded (dimension mismatch): {len(excluded)}")
for b, s1, s2, reason in excluded:
    print(f"  {b}: image {s1} vs mask {s2} ({reason})")

with open(ROOT / "excluded_size_mismatch.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["basename", "img_size", "mask_size", "reason"])
    w.writerows(excluded)

random.seed(0)
random.shuffle(records)
n = len(records)
n_train, n_val = int(0.8 * n), int(0.1 * n)
for r, s in zip(records, ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)):
    r["split"] = s

with open(ROOT / "splice_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "mask", "img_size", "mask_size", "split"])
    w.writeheader()
    w.writerows(records)

print(Counter(r["split"] for r in records))
