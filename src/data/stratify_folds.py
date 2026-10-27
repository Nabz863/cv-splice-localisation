"""Add 5 cross-validation folds to splice_manifest.csv, balanced by host-image category."""
import csv, os, re, random   # standard-library helpers (re = text patterns)
from collections import defaultdict, Counter   # dict with default values; value counter

ROOT = os.path.join(os.environ["DATASETS"], "casia2")   # dataset folder
MANIFEST = os.path.join(ROOT, "splice_manifest.csv")   # written by build_manifest.py
N_FOLDS, SEED = 5, 0   # 5 folds; fixed seed so the folds are the same every run

_cat = re.compile(r"([a-zA-Z]+)\d+")                  # e.g. "nat00083" -> "nat"

def host_category(image_path):
    """CASIA filenames end ..._<donor>_<host>_<id>; return the host's category."""
    base = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]   # file name without folder or extension
    m = _cat.match(base.split("_")[-2])                       # second-to-last field is the host
    return m.group(1) if m else "unknown"   # the letters before the number

# ---- Group images by category ----
rows = list(csv.DictReader(open(MANIFEST)))   # every splicing image
by_cat = defaultdict(list)                            # category -> its images
for r in rows:
    r["host_category"] = host_category(r["image"])   # add the category to each row
    by_cat[r["host_category"]].append(r)   # and file the row under it

# ---- Deal each category across the folds ----
rng = random.Random(SEED)   # seeded random generator
for i, (cat, members) in enumerate(sorted(by_cat.items())):   # each category, in alphabetical order
    rng.shuffle(members)                              # random order within the category
    for j, r in enumerate(members):
        r["fold"] = (j + i) % N_FOLDS                 # deal round-robin; the offset i stops small
                                                      # categories all starting in fold 0

# ---- Save and summarise ----
fields = ["image", "mask", "img_size", "mask_size", "host_category", "fold"]   # columns kept (drops the old split column)
with open(MANIFEST, "w", newline="") as f:            # overwrite the manifest with the new columns
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")   # ignore any other keys in the rows
    w.writeheader()   # header row
    w.writerows(rows)   # one row per image

print(f"{len(rows)} pairs -> {N_FOLDS} folds")
print(f"{'fold':<6}{'n':>5}   category spread")
for k in range(N_FOLDS):                              # show each fold's size and category mix
    fold = [r for r in rows if r["fold"] == k]   # images in fold k
    spread = Counter(r["host_category"] for r in fold)   # how many of each category
    print(f"{k:<6}{len(fold):>5}   {dict(sorted(spread.items()))}")
