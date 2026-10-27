
"""Rung 3, step 1 (GPU): run SAM2.1 Base+ (no prompts, no fine-tuning) on every splicing image.

For each image, stores tp/fp/fn of the union of proposals inside every size window, plus the
single best proposal (oracle), in SAM2_CACHE. Resumable. run_sam2_auc.py then does the 20-pair
evaluation (the 5-pair summary this script prints, results/rung3_sam2.csv, is not used).
Usage:  SAM2_CHECKPOINT=... SAM2_CACHE=... python src/methods/run_sam2.py [--limit N]
"""

import os                                         # environment variables, file replace
import csv                                        # reads the manifest
import time                                       # timing
import argparse                                   # command-line options (--limit)
import numpy as np                                # arrays and maths
import pandas as pd                               # tables (DataFrames) for the cache files
import torch                                      # deep-learning library SAM2 runs on

from pathlib import Path                          # path objects (Path("a") / "b")
import sys                                        # import path

REPO_ROOT = Path(__file__).resolve().parents[2]   # this file is src/methods/run_sam2.py -> go up 2 folders
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))            # so "src.eval.metrics" can be imported

from PIL import Image                             # reads image files

from sam2.build_sam import build_sam2             # builds the SAM2 model (installed separately)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator   # proposes masks with no prompt
from src.eval.metrics import f1_iou               # tp/fp/fn, F1 and IoU of one mask


# ---- Size-filter grid: keep proposals covering between min_frac and max_frac of the image ----
N_FOLDS = 5                                       # number of folds

MIN_FRACS = [0.0, 0.001, 0.005, 0.01, 0.02]       # smallest allowed proposal (fraction of image)
MAX_FRACS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]   # largest allowed proposal

CONFIGS = [                                       # every (min, max) pair where min < max
    (lo, hi)
    for lo in MIN_FRACS
    for hi in MAX_FRACS
    if lo < hi
]

DATASETS = Path(os.environ["DATASETS"])           # data folder set by the user
MANIFEST = DATASETS / "casia2" / "splice_manifest.csv"   # every splicing image

# ---- Paths (relative to the repo root unless set in the environment) ----
CHECKPOINT = os.environ.get(                      # SAM2 weights file
    "SAM2_CHECKPOINT",
    "checkpoints/sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml" # SAM2's model description (ships with the sam2 package)

CACHE_DIR = Path(                                 # where per-image results are saved
    os.environ.get(
        "SAM2_CACHE",
        "results/sam2_cache"
    )
)

COUNTS_CACHE = CACHE_DIR / "sam2_bplus_counts.csv"   # tp/fp/fn per image per size window
ORACLE_CACHE = CACHE_DIR / "sam2_bplus_oracle.csv"   # tp/fp/fn of the best single proposal

RESULTS = Path("results/rung3_sam2.csv")          # 5-pair summary (not used in the report)


# ---- Helpers ----

def load_manifest():
    """Rows of the splicing manifest."""
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))           # one dict per image


def metric_from_counts(tp, fp, fn):
    """F1 and IoU from pooled counts."""
    den_f1 = 2 * tp + fp + fn                     # F1 denominator
    den_iou = tp + fp + fn                        # IoU denominator

    f1 = 2 * tp / den_f1 if den_f1 else 1.0       # nothing to find and nothing predicted = perfect
    iou = tp / den_iou if den_iou else 1.0

    return f1, iou


def atomic_csv(df, path):
    """Write a CSV via a temporary file, so an interruption can't leave half a file."""
    tmp = Path(str(path) + ".tmp")                # e.g. counts.csv.tmp
    df.to_csv(tmp, index=False)                   # write everything to the temporary file
    os.replace(tmp, path)                         # then swap it in, in one step


def save_cache(count_records, oracle_records):
    """Save progress so far."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if missing

    atomic_csv(pd.DataFrame(count_records), COUNTS_CACHE)
    atomic_csv(pd.DataFrame(oracle_records), ORACLE_CACHE)


def load_cache():
    """Load any progress from a previous run."""
    if COUNTS_CACHE.exists():
        counts = pd.read_csv(COUNTS_CACHE).to_dict("records")   # list of dicts, one per row
    else:
        counts = []                               # first run: nothing saved yet

    if ORACLE_CACHE.exists():
        oracle = pd.read_csv(ORACLE_CACHE).to_dict("records")
    else:
        oracle = []

    return counts, oracle


def complete_images(count_records, oracle_records):
    """Images that already have results for every size window and an oracle row."""
    if not count_records or not oracle_records:
        return set()                              # nothing done yet

    cdf = pd.DataFrame(count_records)
    odf = pd.DataFrame(oracle_records)

    expected = len(CONFIGS)                       # rows each finished image should have

    good_counts = set(                            # images with exactly that many rows
        cdf.groupby("image")
           .size()
           .loc[lambda x: x == expected]
           .index
    )

    good_oracle = set(odf["image"])               # images with an oracle row

    return good_counts & good_oracle              # finished = in both


# ---- SAM2 on one image ----

def process_image(row, mask_generator):
    """Counts for every size window, plus the oracle (best single proposal by IoU)."""
    image = np.array(Image.open(row["image"]).convert("RGB"))     # load the image
    truth = np.array(Image.open(row["mask"]).convert("L")) > 127  # true mask (True = spliced)

    with torch.inference_mode():                  # no gradients needed (faster, less memory)
        masks = mask_generator.generate(image)        # automatic proposals, no prompt

    image_area = truth.size                       # number of pixels

    candidate_masks = [                           # each proposal as a True/False mask
        m["segmentation"].astype(bool)
        for m in masks
    ]

    candidate_fracs = [                           # each proposal's size as a fraction of the image
        float(m["area"]) / image_area
        for m in masks
    ]

    count_rows = []                               # one row per size window

    for lo, hi in CONFIGS:
        pred = np.zeros_like(truth, dtype=bool)   # start with an empty prediction

        for seg, frac in zip(candidate_masks, candidate_fracs):
            if lo <= frac <= hi:                     # proposal inside the size window ...
                pred |= seg                          # ... is added to the predicted mask

        m = f1_iou(pred, truth)                   # score the union

        count_rows.append({                       # keep the raw counts so F1 can be pooled later
            "image": row["image"],
            "fold": int(row["fold"]),
            "min_frac": lo,
            "max_frac": hi,
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        })

    # oracle: the single proposal with the highest IoU against the truth (diagnostic only)
    if candidate_masks:
        best = None
        best_iou = -1.0                           # anything beats -1

        for seg in candidate_masks:
            m = f1_iou(seg, truth)

            if m["iou"] > best_iou:               # new best proposal
                best_iou = m["iou"]
                best = m
    else:
        best = f1_iou(np.zeros_like(truth, dtype=bool), truth)   # no proposals: score an empty mask

    oracle_row = {
        "image": row["image"],
        "fold": int(row["fold"]),
        "n_masks": len(candidate_masks),          # how many proposals SAM2 made
        "tp": best["tp"],
        "fp": best["fp"],
        "fn": best["fn"],
    }

    return count_rows, oracle_row


# ---- Evaluation (5 fixed pairs; the 20-pair version is in run_sam2_auc.py) ----

def aggregate(df):
    """Pooled F1 and IoU over a set of rows."""
    tp = int(df["tp"].sum())                      # add up counts over the rows
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())

    f1, iou = metric_from_counts(tp, fp, fn)

    return f1, iou


def choose_validation_config(counts_df, val_fold):
    """Size window with the best pooled F1 on the validation fold."""
    fold_df = counts_df[counts_df["fold"] == val_fold]   # validation images only

    best = None

    for lo, hi in CONFIGS:
        x = fold_df[                              # rows for this size window
            np.isclose(fold_df["min_frac"], lo) &
            np.isclose(fold_df["max_frac"], hi)
        ]

        f1, iou = aggregate(x)

        candidate = (f1, iou, lo, hi)

        if best is None or candidate[:2] > best[:2]:   # compare by F1, then IoU on ties
            best = candidate

    return best


def final_results(count_records, oracle_records):
    """Print and save the 5-pair summary."""
    counts_df = pd.DataFrame(count_records)
    oracle_df = pd.DataFrame(oracle_records)

    rows = []                                     # one result per test fold

    print("\n=== FINAL 5-FOLD SAM2 RESULTS ===")
    print(
        f"{'test':<6}{'val':<6}"
        f"{'min':>8}{'max':>8}"
        f"{'F1':>10}{'IoU':>10}"
        f"{'oracle F1':>12}{'oracle IoU':>12}"
    )

    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS      # validation = the next fold

        val_f1, val_iou, lo, hi = choose_validation_config(   # pick the size window on validation
            counts_df, val_fold
        )

        test = counts_df[                         # test-fold rows at that window
            (counts_df["fold"] == test_fold) &
            np.isclose(counts_df["min_frac"], lo) &
            np.isclose(counts_df["max_frac"], hi)
        ]

        f1, iou = aggregate(test)                 # test score

        oracle_test = oracle_df[                  # oracle rows of the test fold
            oracle_df["fold"] == test_fold
        ]

        oracle_f1, oracle_iou = aggregate(oracle_test)

        rows.append({
            "test_fold": test_fold,
            "val_fold": val_fold,
            "min_frac": lo,
            "max_frac": hi,
            "f1": f1,
            "iou": iou,
            "oracle_f1": oracle_f1,
            "oracle_iou": oracle_iou,
        })

        print(
            f"{test_fold:<6}{val_fold:<6}"
            f"{lo:>8.3f}{hi:>8.3f}"
            f"{f1:>10.4f}{iou:>10.4f}"
            f"{oracle_f1:>12.4f}{oracle_iou:>12.4f}"
        )

    result_df = pd.DataFrame(rows)

    print(                                        # mean +/- spread over the 5 test folds
        f"\nmean F1       "
        f"{result_df['f1'].mean():.4f} +/- "
        f"{result_df['f1'].std(ddof=0):.4f}"
    )

    print(
        f"mean IoU      "
        f"{result_df['iou'].mean():.4f} +/- "
        f"{result_df['iou'].std(ddof=0):.4f}"
    )

    print(
        f"mean oracle F1 "
        f"{result_df['oracle_f1'].mean():.4f} +/- "
        f"{result_df['oracle_f1'].std(ddof=0):.4f}"
    )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)   # create results/ if missing
    result_df.to_csv(RESULTS, index=False)

    print(f"\nwrote {RESULTS}")


# ---- Main: load SAM2, process every image not already cached, save as it goes ----

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(                          # optional: only do a few images (for testing)
        "--limit",
        type=int,
        default=None,
        help="Process at most this many NEW images, useful for smoke testing."
    )

    args = parser.parse_args()

    rows = load_manifest()                        # every splicing image

    device = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one

    print("device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("model: SAM2.1 Hiera Base+")
    print("images:", len(rows))
    print("area-filter configs:", len(CONFIGS))

    print("\nLoading SAM2...")

    model = build_sam2(                           # load the pretrained model
        MODEL_CFG,
        CHECKPOINT,
        device=device
    )

    mask_generator = SAM2AutomaticMaskGenerator(model)   # default settings, no prompts

    count_records, oracle_records = load_cache()  # anything done by a previous run

    done = complete_images(count_records, oracle_records)

    print(f"cached complete images: {len(done)}/{len(rows)}")

    pending = [                                   # images still to do
        r for r in rows
        if r["image"] not in done
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print("processing now:", len(pending))

    start = time.time()

    for i, row in enumerate(pending, 1):          # i counts from 1

        # drop any partial rows left for this image by an interrupted run
        count_records = [
            x for x in count_records
            if x["image"] != row["image"]
        ]

        oracle_records = [
            x for x in oracle_records
            if x["image"] != row["image"]
        ]

        new_counts, new_oracle = process_image(   # run SAM2 on this image
            row,
            mask_generator
        )

        count_records.extend(new_counts)          # add its rows
        oracle_records.append(new_oracle)

        if i % 10 == 0 or i == len(pending):      # save every 10 images and at the end
            save_cache(count_records, oracle_records)

            elapsed = time.time() - start
            rate = elapsed / i                    # seconds per image

            total_done = len(
                complete_images(count_records, oracle_records)
            )

            print(
                f"{i}/{len(pending)} this run | "
                f"{total_done}/{len(rows)} total | "
                f"{rate:.2f}s/image"
            )

    done = complete_images(count_records, oracle_records)

    if len(done) == len(rows):                    # everything done: print the summary
        final_results(count_records, oracle_records)
    else:                                         # otherwise say how to continue
        print(
            f"\nCache currently contains "
            f"{len(done)}/{len(rows)} complete images."
        )
        print(
            "Run this script again with no --limit "
            "to resume from where it stopped."
        )


if __name__ == "__main__":
    main()
