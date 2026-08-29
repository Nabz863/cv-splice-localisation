"""Assign 5-fold CV indices, stratified by host-image content category.

Categories smaller than the fold count cannot be stratified; they are dealt
round-robin with a per-category start offset so they do not all pile into fold 0.
"""
import csv, os, re, random
from collections import defaultdict, Counter

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
MANIFEST = os.path.join(ROOT, "splice_manifest.csv")
N_FOLDS, SEED = 5, 0

_cat = re.compile(r"([a-zA-Z]+)\d+")

def host_category(image_path):
    """CASIA filenames end ..._<donor>_<host>_<id>; the host is what we stratify on."""
    base = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = _cat.match(base.split("_")[-2])
    return m.group(1) if m else "unknown"

rows = list(csv.DictReader(open(MANIFEST)))
by_cat = defaultdict(list)
for r in rows:
    r["host_category"] = host_category(r["image"])
    by_cat[r["host_category"]].append(r)

rng = random.Random(SEED)
for i, (cat, members) in enumerate(sorted(by_cat.items())):
    rng.shuffle(members)
    for j, r in enumerate(members):
        r["fold"] = (j + i) % N_FOLDS

fields = ["image", "mask", "img_size", "mask_size", "host_category", "fold"]
with open(MANIFEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} pairs -> {N_FOLDS} folds")
print(f"{'fold':<6}{'n':>5}   category spread")
for k in range(N_FOLDS):
    fold = [r for r in rows if r["fold"] == k]
    spread = Counter(r["host_category"] for r in fold)
    print(f"{k:<6}{len(fold):>5}   {dict(sorted(spread.items()))}")
