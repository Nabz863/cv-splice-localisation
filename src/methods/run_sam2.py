
"""Rung 3: SAM2 zero-shot splice localisation.

Protocol:
- SAM2.1 Base+ automatic mask generation, no prompts and no fine-tuning.
- Candidate masks are filtered only by their fraction of image area.
- The size filter is selected on validation fold (k+1)%5.
- It is then applied unchanged to test fold k.
- Pixel F1/IoU are micro-averaged by summing TP/FP/FN across the fold.
- Oracle-best candidate is reported separately as an upper bound.
- SAM2 inference is cached so interrupted Colab runs can resume.
"""

import os
import csv
import time
import argparse
import numpy as np
import pandas as pd
import torch

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from src.eval.metrics import f1_iou


N_FOLDS = 5

MIN_FRACS = [0.0, 0.001, 0.005, 0.01, 0.02]
MAX_FRACS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]

CONFIGS = [
    (lo, hi)
    for lo in MIN_FRACS
    for hi in MAX_FRACS
    if lo < hi
]

DATASETS = Path(os.environ["DATASETS"])
MANIFEST = DATASETS / "casia2" / "splice_manifest.csv"

CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/content/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

CACHE_DIR = Path(
    os.environ.get(
        "SAM2_CACHE",
        "/content/drive/MyDrive/CV_Project/sam2_cache"
    )
)

COUNTS_CACHE = CACHE_DIR / "sam2_bplus_counts.csv"
ORACLE_CACHE = CACHE_DIR / "sam2_bplus_oracle.csv"

RESULTS = Path("results/rung3_sam2.csv")


def load_manifest():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def metric_from_counts(tp, fp, fn):
    den_f1 = 2 * tp + fp + fn
    den_iou = tp + fp + fn

    f1 = 2 * tp / den_f1 if den_f1 else 1.0
    iou = tp / den_iou if den_iou else 1.0

    return f1, iou


def atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def save_cache(count_records, oracle_records):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    atomic_csv(pd.DataFrame(count_records), COUNTS_CACHE)
    atomic_csv(pd.DataFrame(oracle_records), ORACLE_CACHE)


def load_cache():
    if COUNTS_CACHE.exists():
        counts = pd.read_csv(COUNTS_CACHE).to_dict("records")
    else:
        counts = []

    if ORACLE_CACHE.exists():
        oracle = pd.read_csv(ORACLE_CACHE).to_dict("records")
    else:
        oracle = []

    return counts, oracle


def complete_images(count_records, oracle_records):
    if not count_records or not oracle_records:
        return set()

    cdf = pd.DataFrame(count_records)
    odf = pd.DataFrame(oracle_records)

    expected = len(CONFIGS)

    good_counts = set(
        cdf.groupby("image")
           .size()
           .loc[lambda x: x == expected]
           .index
    )

    good_oracle = set(odf["image"])

    return good_counts & good_oracle


def process_image(row, mask_generator):
    image = np.array(Image.open(row["image"]).convert("RGB"))
    truth = np.array(Image.open(row["mask"]).convert("L")) > 127

    with torch.inference_mode():
        masks = mask_generator.generate(image)

    image_area = truth.size

    candidate_masks = [
        m["segmentation"].astype(bool)
        for m in masks
    ]

    candidate_fracs = [
        float(m["area"]) / image_area
        for m in masks
    ]

    count_rows = []

    for lo, hi in CONFIGS:
        pred = np.zeros_like(truth, dtype=bool)

        for seg, frac in zip(candidate_masks, candidate_fracs):
            if lo <= frac <= hi:
                pred |= seg

        m = f1_iou(pred, truth)

        count_rows.append({
            "image": row["image"],
            "fold": int(row["fold"]),
            "min_frac": lo,
            "max_frac": hi,
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        })

    if candidate_masks:
        best = None
        best_iou = -1.0

        for seg in candidate_masks:
            m = f1_iou(seg, truth)

            if m["iou"] > best_iou:
                best_iou = m["iou"]
                best = m
    else:
        best = f1_iou(np.zeros_like(truth, dtype=bool), truth)

    oracle_row = {
        "image": row["image"],
        "fold": int(row["fold"]),
        "n_masks": len(candidate_masks),
        "tp": best["tp"],
        "fp": best["fp"],
        "fn": best["fn"],
    }

    return count_rows, oracle_row


def aggregate(df):
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())

    f1, iou = metric_from_counts(tp, fp, fn)

    return f1, iou


def choose_validation_config(counts_df, val_fold):
    fold_df = counts_df[counts_df["fold"] == val_fold]

    best = None

    for lo, hi in CONFIGS:
        x = fold_df[
            np.isclose(fold_df["min_frac"], lo) &
            np.isclose(fold_df["max_frac"], hi)
        ]

        f1, iou = aggregate(x)

        candidate = (f1, iou, lo, hi)

        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return best


def final_results(count_records, oracle_records):
    counts_df = pd.DataFrame(count_records)
    oracle_df = pd.DataFrame(oracle_records)

    rows = []

    print("\n=== FINAL 5-FOLD SAM2 RESULTS ===")
    print(
        f"{'test':<6}{'val':<6}"
        f"{'min':>8}{'max':>8}"
        f"{'F1':>10}{'IoU':>10}"
        f"{'oracle F1':>12}{'oracle IoU':>12}"
    )

    for test_fold in range(N_FOLDS):
        val_fold = (test_fold + 1) % N_FOLDS

        val_f1, val_iou, lo, hi = choose_validation_config(
            counts_df, val_fold
        )

        test = counts_df[
            (counts_df["fold"] == test_fold) &
            np.isclose(counts_df["min_frac"], lo) &
            np.isclose(counts_df["max_frac"], hi)
        ]

        f1, iou = aggregate(test)

        oracle_test = oracle_df[
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

    print(
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

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(RESULTS, index=False)

    print(f"\nwrote {RESULTS}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many NEW images, useful for smoke testing."
    )

    args = parser.parse_args()

    rows = load_manifest()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("model: SAM2.1 Hiera Base+")
    print("images:", len(rows))
    print("area-filter configs:", len(CONFIGS))

    print("\nLoading SAM2...")

    model = build_sam2(
        MODEL_CFG,
        CHECKPOINT,
        device=device
    )

    mask_generator = SAM2AutomaticMaskGenerator(model)

    count_records, oracle_records = load_cache()

    done = complete_images(count_records, oracle_records)

    print(f"cached complete images: {len(done)}/{len(rows)}")

    pending = [
        r for r in rows
        if r["image"] not in done
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print("processing now:", len(pending))

    start = time.time()

    for i, row in enumerate(pending, 1):

        # Remove any incomplete old rows for this image
        count_records = [
            x for x in count_records
            if x["image"] != row["image"]
        ]

        oracle_records = [
            x for x in oracle_records
            if x["image"] != row["image"]
        ]

        new_counts, new_oracle = process_image(
            row,
            mask_generator
        )

        count_records.extend(new_counts)
        oracle_records.append(new_oracle)

        if i % 10 == 0 or i == len(pending):
            save_cache(count_records, oracle_records)

            elapsed = time.time() - start
            rate = elapsed / i

            total_done = len(
                complete_images(count_records, oracle_records)
            )

            print(
                f"{i}/{len(pending)} this run | "
                f"{total_done}/{len(rows)} total | "
                f"{rate:.2f}s/image"
            )

    done = complete_images(count_records, oracle_records)

    if len(done) == len(rows):
        final_results(count_records, oracle_records)
    else:
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
