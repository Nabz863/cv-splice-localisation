"""Rung 3, step 2: 20-pair evaluation of SAM2 from the cache, plus image-level AUC on authentic images.

Picks each validation fold's size window, scores every test fold, and runs SAM2 on the
authentic images (GPU, resumable) for the AUC. Writes results/rung3_sam2_20pairs.csv,
rung3_sam2_folds.csv, image_auc_sam2_*.csv.   Usage:  python src/methods/run_sam2_auc.py
"""
import os                                         # environment variables
import sys                                        # import path
import csv                                        # CSV files
import time                                       # timing and retry waits
from pathlib import Path                          # path objects (Path("a") / "b")

import numpy as np                                # arrays and maths
import pandas as pd                               # tables (DataFrames)
import torch                                      # deep-learning library SAM2 runs on
from PIL import Image                             # reads image files
from sklearn.metrics import roc_auc_score         # standard AUC function


# ---- Size-filter grid (same as run_sam2.py) ----
N_FOLDS = 5                                       # number of folds
MIN_FRACS = [0.0, 0.001, 0.005, 0.01, 0.02]       # smallest allowed proposal (fraction of image)
MAX_FRACS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]   # largest allowed proposal
CONFIGS = [(lo, hi) for lo in MIN_FRACS for hi in MAX_FRACS if lo < hi]   # every valid window

DATASETS = Path(os.environ["DATASETS"])           # data folder set by the user
MANIFEST = DATASETS / "casia2" / "splice_manifest.csv"   # every splicing image
AUTH_DIR = DATASETS / "casia2" / "images" / "CASIA2.0_revised" / "Au"   # authentic images

CACHE_DIR = Path(                                 # where run_sam2.py saved its results
    os.environ.get(
        "SAM2_CACHE",
        "results/sam2_cache"
    )
)

COUNTS_CACHE = CACHE_DIR / "sam2_bplus_counts.csv"   # tampered images: tp/fp/fn per window
ORACLE_CACHE = CACHE_DIR / "sam2_bplus_oracle.csv"   # tampered images: best single proposal
AUTH_CACHE = CACHE_DIR / "sam2_bplus_auth_area.csv"  # authentic images: area score per window

RESULTS = Path("results")                         # output folder
PAIR_RESULTS = RESULTS / "rung3_sam2_20pairs.csv" # one row per (test, val) pair
FOLD_RESULTS = RESULTS / "rung3_sam2_folds.csv"   # one row per test fold
AUC_SUMMARY = RESULTS / "image_auc_sam2_summary.csv"

CHECKPOINT = os.environ.get(                      # SAM2 weights file
    "SAM2_CHECKPOINT",
    "checkpoints/sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml" # SAM2's model description

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}   # file types counted as images


# ---- Helpers ----

def metric(df):
    """Pooled F1 and IoU over a set of rows."""
    tp = df["tp"].sum()                           # add up counts over the rows
    fp = df["fp"].sum()
    fn = df["fn"].sum()

    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return f1, iou


def choose_config(counts, val_fold):
    """Size window with the best pooled F1 on the validation fold."""
    fold = counts[counts["fold"] == val_fold]     # validation images only
    best = None

    for lo, hi in CONFIGS:
        x = fold[                                 # rows for this window
            np.isclose(fold["min_frac"], lo)
            & np.isclose(fold["max_frac"], hi)
        ]

        f1, iou = metric(x)
        candidate = (f1, iou, lo, hi)

        if best is None or candidate[:2] > best[:2]:   # compare by F1, then IoU on ties
            best = candidate

    return best


def cfg_key(lo, hi):
    """String key for a size window."""
    return f"{float(lo):.6f}|{float(hi):.6f}"     # e.g. "0.010000|0.500000"


# ---- Authentic images: SAM2 area score per size window (cached, resumable) ----

def load_auth_cache():
    """Authentic-image scores computed so far."""
    if not AUTH_CACHE.exists():                   # first run: empty table
        return pd.DataFrame(
            columns=["image", "min_frac", "max_frac", "score"]
        )

    x = pd.read_csv(AUTH_CACHE)

    return x.drop_duplicates(                     # an interrupted run can leave repeats; keep the newest
        subset=["image", "min_frac", "max_frac"],
        keep="last"
    )


def completed_auth_images(auth, configs):
    """Authentic images that already have a score for every needed window."""
    if len(auth) == 0:
        return set()

    wanted = {cfg_key(lo, hi) for lo, hi in configs}   # windows each image needs

    x = auth.copy()
    x["cfg"] = [                                  # add a window key to every row
        cfg_key(lo, hi)
        for lo, hi in zip(x["min_frac"], x["max_frac"])
    ]

    done = set()

    for image, group in x.groupby("image"):       # each image's rows
        if wanted.issubset(set(group["cfg"])):    # has every needed window
            done.add(image)

    return done


def append_auth(rows):
    """Append new scores to the cache file."""
    if not rows:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        AUTH_CACHE,
        mode="a",                                 # add to the end of the file
        header=not AUTH_CACHE.exists(),           # header only when creating the file
        index=False
    )


def cache_authentic(configs):
    """Run SAM2 on every authentic image not yet cached; score = fraction of image predicted."""
    configs = sorted(set(configs))                # the distinct windows selected on validation

    paths = sorted(                               # every authentic image file
        p for p in AUTH_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if len(paths) != 7491:                        # CASIA v2 has 7491 authentic images
        raise RuntimeError(f"Expected 7491 authentic images, found {len(paths)}")

    auth = load_auth_cache()
    done = completed_auth_images(auth, configs)

    print("Authentic images:", len(paths))
    print("Already cached:", len(done))
    print("Still needed:", len(paths) - len(done))

    if len(done) == len(paths):                   # nothing left to do (no GPU needed)
        return

    sam2_repo = os.environ.get("SAM2_REPO", "sam2")          # optional local clone of the sam2 repo
    if sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)

    from sam2.build_sam import build_sam2         # imported here so the rest works without sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for SAM2 inference")

    model = build_sam2(                           # load the pretrained model
        MODEL_CFG,
        CHECKPOINT,
        device="cuda"
    )

    generator = SAM2AutomaticMaskGenerator(model) # default settings, no prompts

    buffer = []                                   # rows not yet written to disk
    n = 0                                         # images processed this run
    start = time.time()

    for path in paths:
        if str(path) in done:                     # already cached
            continue

        image = np.asarray(Image.open(path).convert("RGB"))
        image_area = image.shape[0] * image.shape[1]   # number of pixels

        with torch.inference_mode():              # no gradients needed
            masks = generator.generate(image)     # automatic proposals

        segs = [m["segmentation"].astype(bool) for m in masks]       # each proposal as a mask
        fracs = [float(m["area"]) / image_area for m in masks]       # each proposal's size

        for lo, hi in configs:
            pred = np.zeros(image.shape[:2], dtype=bool)

            for seg, frac in zip(segs, fracs):
                if lo <= frac <= hi:              # inside the window: add to the prediction
                    pred |= seg

            buffer.append({
                "image": str(path),
                "min_frac": lo,
                "max_frac": hi,
                "score": float(pred.mean())       # fraction of the image predicted spliced
            })

        n += 1

        if n % 10 == 0:                           # write to disk every 10 images
            append_auth(buffer)
            buffer = []

            rate = n / (time.time() - start)

            print(
                f"{len(done) + n}/{len(paths)} "
                f"{rate:.3f} img/s"
            )

    append_auth(buffer)                           # write whatever is left


def image_areas(counts):
    """Pixel count of every tampered image (retries reads on a flaky network drive)."""
    tp_dir = DATASETS / "casia2" / "images" / "CASIA2.0_revised" / "Tp"   # tampered images folder

    names = set(Path(str(x)).name for x in counts["image"].unique())   # file names we need
    areas = {}                                    # file name -> pixel count

    for path in tp_dir.iterdir():
        if path.name not in names:
            continue

        for attempt in range(5):                  # up to 5 tries
            try:
                with Image.open(path) as im:
                    w, h = im.size                # reads the header only, not the pixels

                areas[path.name] = w * h
                break

            except OSError:
                if attempt == 4:                  # last try failed: give up
                    raise
                time.sleep(3)                     # wait and retry

    if len(areas) != 1822:                        # one per splicing image
        raise RuntimeError(f"Expected 1822 image sizes, found {len(areas)}")

    return areas


# ---- Main ----

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(COUNTS_CACHE)            # from run_sam2.py
    oracle = pd.read_csv(ORACLE_CACHE)

    if counts["image"].nunique() != 1822:         # run_sam2.py must have finished
        raise RuntimeError("Incomplete tampered SAM2 cache")

    # ---- 1. choose a size window on each fold (used when that fold is the validation fold) ----
    selected = {}                                 # validation fold -> (min, max)

    print("VALIDATION FILTERS")

    for val_fold in range(N_FOLDS):
        f1, iou, lo, hi = choose_config(counts, val_fold)
        selected[val_fold] = (lo, hi)

        print(
            f"val {val_fold}: [{lo:.3f}, {hi:.3f}] "
            f"F1={f1:.4f} IoU={iou:.4f}"
        )

    # ---- 2. authentic-image scores for those windows ----
    cache_authentic(selected.values())

    auth = load_auth_cache()
    areas = image_areas(counts)

    # ---- 3. every (test, validation) pair: localisation F1/IoU and image-level AUC ----
    rows = []                                     # one result per pair

    print("\n20 PAIRS")

    for test_fold in range(N_FOLDS):
        oracle_test = oracle[oracle["fold"] == test_fold]
        oracle_f1, oracle_iou = metric(oracle_test)   # oracle doesn't depend on the validation fold

        for val_fold in range(N_FOLDS):
            if test_fold == val_fold:             # a fold can't be both
                continue

            lo, hi = selected[val_fold]           # window chosen on this validation fold

            test = counts[                        # test-fold rows at that window
                (counts["fold"] == test_fold)
                & np.isclose(counts["min_frac"], lo)
                & np.isclose(counts["max_frac"], hi)
            ].copy()

            f1, iou = metric(test)                # localisation score

            # tampered-image score = fraction of the image predicted spliced ((tp + fp) / area)
            test["score"] = [
                (tp + fp) / areas[Path(str(image)).name]
                for image, tp, fp in zip(
                    test["image"],
                    test["tp"],
                    test["fp"]
                )
            ]

            tam = pd.DataFrame({                  # tampered images, label 1
                "image": test["image"],
                "label": 1,
                "score": test["score"]
            })

            aut = auth[                           # authentic images at the same window
                np.isclose(auth["min_frac"], lo)
                & np.isclose(auth["max_frac"], hi)
            ][["image", "score"]].copy()

            aut["label"] = 0                      # authentic = label 0
            aut = aut[["image", "label", "score"]]   # same column order as tam

            if len(aut) != 7491:
                raise RuntimeError(
                    f"Expected 7491 authentic scores, found {len(aut)}"
                )

            combined = pd.concat([tam, aut], ignore_index=True)   # one table of both

            combined["test_fold"] = test_fold
            combined["val_fold"] = val_fold
            combined["min_frac"] = lo
            combined["max_frac"] = hi

            auc = roc_auc_score(                  # can the score separate tampered from authentic?
                combined["label"],
                combined["score"]
            )

            combined.to_csv(                      # save every image's score for this pair
                RESULTS / f"image_auc_sam2_t{test_fold}v{val_fold}.csv",
                index=False
            )

            rows.append({
                "test_fold": test_fold,
                "val_fold": val_fold,
                "min_frac": lo,
                "max_frac": hi,
                "f1": f1,
                "iou": iou,
                "auc": auc,
                "oracle_f1": oracle_f1,
                "oracle_iou": oracle_iou,
                "n_tampered": len(tam),
                "n_authentic": len(aut)
            })

            print(
                f"t{test_fold} v{val_fold}: "
                f"F1={f1:.4f} "
                f"IoU={iou:.4f} "
                f"AUC={auc:.4f}"
            )

    # ---- 4. save per pair, and per test fold (mean over its validation folds) ----
    pairs = pd.DataFrame(rows)

    folds = (                                     # average the 4 pairs of each test fold
        pairs
        .groupby("test_fold", as_index=False)
        .agg(
            f1=("f1", "mean"),
            iou=("iou", "mean"),
            auc=("auc", "mean"),
            oracle_f1=("oracle_f1", "mean"),
            oracle_iou=("oracle_iou", "mean")
        )
    )

    pairs.to_csv(PAIR_RESULTS, index=False)
    folds.to_csv(FOLD_RESULTS, index=False)

    pairs[                                        # AUC-only summary
        [
            "test_fold",
            "val_fold",
            "min_frac",
            "max_frac",
            "auc",
            "n_tampered",
            "n_authentic"
        ]
    ].to_csv(AUC_SUMMARY, index=False)

    print("\nFINAL")

    print(                                        # mean +/- spread over the 5 test folds
        f"SAM2 F1: {folds['f1'].mean():.4f} ± "
        f"{folds['f1'].std(ddof=0):.4f}"
    )

    print(
        f"SAM2 IoU: {folds['iou'].mean():.4f} ± "
        f"{folds['iou'].std(ddof=0):.4f}"
    )

    print(
        f"SAM2 AUC: {folds['auc'].mean():.4f} ± "
        f"{folds['auc'].std(ddof=0):.4f}"
    )


if __name__ == "__main__":
    main()
