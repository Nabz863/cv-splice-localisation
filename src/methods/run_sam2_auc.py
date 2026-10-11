import os
import sys
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score


N_FOLDS = 5
MIN_FRACS = [0.0, 0.001, 0.005, 0.01, 0.02]
MAX_FRACS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
CONFIGS = [(lo, hi) for lo in MIN_FRACS for hi in MAX_FRACS if lo < hi]

DATASETS = Path(os.environ["DATASETS"])
MANIFEST = DATASETS / "casia2" / "splice_manifest.csv"
AUTH_DIR = DATASETS / "casia2" / "images" / "CASIA2.0_revised" / "Au"

CACHE_DIR = Path(
    os.environ.get(
        "SAM2_CACHE",
        "/content/drive/MyDrive/CV_Project/sam2_cache"
    )
)

COUNTS_CACHE = CACHE_DIR / "sam2_bplus_counts.csv"
ORACLE_CACHE = CACHE_DIR / "sam2_bplus_oracle.csv"
AUTH_CACHE = CACHE_DIR / "sam2_bplus_auth_area.csv"

RESULTS = Path("results")
PAIR_RESULTS = RESULTS / "rung3_sam2_20pairs.csv"
FOLD_RESULTS = RESULTS / "rung3_sam2_folds.csv"
AUC_SUMMARY = RESULTS / "image_auc_sam2_summary.csv"

CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    "/content/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def metric(df):
    tp = df["tp"].sum()
    fp = df["fp"].sum()
    fn = df["fn"].sum()

    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return f1, iou


def choose_config(counts, val_fold):
    fold = counts[counts["fold"] == val_fold]
    best = None

    for lo, hi in CONFIGS:
        x = fold[
            np.isclose(fold["min_frac"], lo)
            & np.isclose(fold["max_frac"], hi)
        ]

        f1, iou = metric(x)
        candidate = (f1, iou, lo, hi)

        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return best


def cfg_key(lo, hi):
    return f"{float(lo):.6f}|{float(hi):.6f}"


def load_auth_cache():
    if not AUTH_CACHE.exists():
        return pd.DataFrame(
            columns=["image", "min_frac", "max_frac", "score"]
        )

    x = pd.read_csv(AUTH_CACHE)

    return x.drop_duplicates(
        subset=["image", "min_frac", "max_frac"],
        keep="last"
    )


def completed_auth_images(auth, configs):
    if len(auth) == 0:
        return set()

    wanted = {cfg_key(lo, hi) for lo, hi in configs}

    x = auth.copy()
    x["cfg"] = [
        cfg_key(lo, hi)
        for lo, hi in zip(x["min_frac"], x["max_frac"])
    ]

    done = set()

    for image, group in x.groupby("image"):
        if wanted.issubset(set(group["cfg"])):
            done.add(image)

    return done


def append_auth(rows):
    if not rows:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        AUTH_CACHE,
        mode="a",
        header=not AUTH_CACHE.exists(),
        index=False
    )


def cache_authentic(configs):
    configs = sorted(set(configs))

    paths = sorted(
        p for p in AUTH_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if len(paths) != 7491:
        raise RuntimeError(f"Expected 7491 authentic images, found {len(paths)}")

    auth = load_auth_cache()
    done = completed_auth_images(auth, configs)

    print("Authentic images:", len(paths))
    print("Already cached:", len(done))
    print("Still needed:", len(paths) - len(done))

    if len(done) == len(paths):
        return

    if "/content/sam2" not in sys.path:
        sys.path.insert(0, "/content/sam2")

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for SAM2 inference")

    model = build_sam2(
        MODEL_CFG,
        CHECKPOINT,
        device="cuda"
    )

    generator = SAM2AutomaticMaskGenerator(model)

    buffer = []
    n = 0
    start = time.time()

    for path in paths:
        if str(path) in done:
            continue

        image = np.asarray(Image.open(path).convert("RGB"))
        image_area = image.shape[0] * image.shape[1]

        with torch.inference_mode():
            masks = generator.generate(image)

        segs = [m["segmentation"].astype(bool) for m in masks]
        fracs = [float(m["area"]) / image_area for m in masks]

        for lo, hi in configs:
            pred = np.zeros(image.shape[:2], dtype=bool)

            for seg, frac in zip(segs, fracs):
                if lo <= frac <= hi:
                    pred |= seg

            buffer.append({
                "image": str(path),
                "min_frac": lo,
                "max_frac": hi,
                "score": float(pred.mean())
            })

        n += 1

        if n % 10 == 0:
            append_auth(buffer)
            buffer = []

            rate = n / (time.time() - start)

            print(
                f"{len(done) + n}/{len(paths)} "
                f"{rate:.3f} img/s"
            )

    append_auth(buffer)


def image_areas(counts):
    tp_dir = DATASETS / "casia2" / "images" / "CASIA2.0_revised" / "Tp"

    names = set(Path(str(x)).name for x in counts["image"].unique())
    areas = {}

    for path in tp_dir.iterdir():
        if path.name not in names:
            continue

        for attempt in range(5):
            try:
                with Image.open(path) as im:
                    w, h = im.size

                areas[path.name] = w * h
                break

            except OSError:
                if attempt == 4:
                    raise
                time.sleep(3)

    if len(areas) != 1822:
        raise RuntimeError(f"Expected 1822 image sizes, found {len(areas)}")

    return areas


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(COUNTS_CACHE)
    oracle = pd.read_csv(ORACLE_CACHE)

    if counts["image"].nunique() != 1822:
        raise RuntimeError("Incomplete tampered SAM2 cache")

    selected = {}

    print("VALIDATION FILTERS")

    for val_fold in range(N_FOLDS):
        f1, iou, lo, hi = choose_config(counts, val_fold)
        selected[val_fold] = (lo, hi)

        print(
            f"val {val_fold}: [{lo:.3f}, {hi:.3f}] "
            f"F1={f1:.4f} IoU={iou:.4f}"
        )

    cache_authentic(selected.values())

    auth = load_auth_cache()
    areas = image_areas(counts)

    rows = []

    print("\n20 PAIRS")

    for test_fold in range(N_FOLDS):
        oracle_test = oracle[oracle["fold"] == test_fold]
        oracle_f1, oracle_iou = metric(oracle_test)

        for val_fold in range(N_FOLDS):
            if test_fold == val_fold:
                continue

            lo, hi = selected[val_fold]

            test = counts[
                (counts["fold"] == test_fold)
                & np.isclose(counts["min_frac"], lo)
                & np.isclose(counts["max_frac"], hi)
            ].copy()

            f1, iou = metric(test)

            test["score"] = [
                (tp + fp) / areas[Path(str(image)).name]
                for image, tp, fp in zip(
                    test["image"],
                    test["tp"],
                    test["fp"]
                )
            ]

            tam = pd.DataFrame({
                "image": test["image"],
                "label": 1,
                "score": test["score"]
            })

            aut = auth[
                np.isclose(auth["min_frac"], lo)
                & np.isclose(auth["max_frac"], hi)
            ][["image", "score"]].copy()

            aut["label"] = 0
            aut = aut[["image", "label", "score"]]

            if len(aut) != 7491:
                raise RuntimeError(
                    f"Expected 7491 authentic scores, found {len(aut)}"
                )

            combined = pd.concat([tam, aut], ignore_index=True)

            combined["test_fold"] = test_fold
            combined["val_fold"] = val_fold
            combined["min_frac"] = lo
            combined["max_frac"] = hi

            auc = roc_auc_score(
                combined["label"],
                combined["score"]
            )

            combined.to_csv(
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

    pairs = pd.DataFrame(rows)

    folds = (
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

    pairs[
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

    print(
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
