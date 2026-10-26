"""Add 5 cross-validation folds to splice_manifest.csv, balanced by host-image category."""
import csv, os, re, random
from collections import defaultdict, Counter

ROOT = os.path.join(os.environ["DATASETS"], "casia2")
MANIFEST = os.path.join(ROOT, "splice_manifest.csv")
N_FOLDS, SEED = 5, 0

_cat = re.compile(r"([a-zA-Z]+)\d+")                  # e.g. "nat00083" -> "nat"

def host_category(image_path):
    """CASIA filenames end ..._<donor>_<host>_<id>; return the host's category."""
    base = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]   # file name without folder or extension
    m = _cat.match(base.split("_")[-2])                       # second-to-last field is the host
    return m.group(1) if m else "unknown"

# ---- Group images by category ----
rows = list(csv.DictReader(open(MANIFEST)))
by_cat = defaultdict(list)                            # category -> its images
for r in rows:
    r["host_category"] = host_category(r["image"])
    by_cat[r["host_category"]].append(r)

# ---- Deal each category across the folds ----
rng = random.Random(SEED)
for i, (cat, members) in enumerate(sorted(by_cat.items())):
    rng.shuffle(members)                              # random order within the category
    for j, r in enumerate(members):
        r["fold"] = (j + i) % N_FOLDS                 # deal round-robin; the offset i stops small
                                                      # categories all starting in fold 0

# ---- Save and summarise ----
fields = ["image", "mask", "img_size", "mask_size", "host_category", "fold"]
with open(MANIFEST, "w", newline="") as f:            # overwrite the manifest with the new columns
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} pairs -> {N_FOLDS} folds")
print(f"{'fold':<6}{'n':>5}   category spread")
for k in range(N_FOLDS):                              # show each fold's size and category mix
    fold = [r for r in rows if r["fold"] == k]
    spread = Counter(r["host_category"] for r in fold)
    print(f"{k:<6}{len(fold):>5}   {dict(sorted(spread.items()))}")
