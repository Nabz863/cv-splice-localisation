"""Training crops for the U-Net: 256x256 patches, half of them centred on spliced pixels."""
import csv, os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH = 256               # crop size in pixels
CROP_RNG = "per-worker"   # how crops are randomised; saved with each model (refine_meta.py)


class SpliceCrops(Dataset):
    """Dataset of image/mask crops. Each loader worker gets its own seeded random generator."""

    def __init__(self, rows, patch=PATCH, train=True, crops_per_image=8, seed=0):
        self.rows, self.patch, self.train = rows, patch, train
        self.n = crops_per_image if train else 1        # crops per image per epoch (1 for validation)
        self.seed = seed
        self._main, self._wseed, self._wrng = None, None, None

    @property
    def rng(self):
        """Random generator for this worker, seeded from (seed, torch's worker seed)."""
        info = torch.utils.data.get_worker_info()
        if info is None:                                  # no worker processes: one shared generator
            if self._main is None:
                self._main = np.random.default_rng(self.seed)
            return self._main
        if self._wseed != info.seed:                      # new worker or new epoch: reseed
            self._wseed = info.seed
            self._wrng = np.random.default_rng([self.seed, info.seed % (2 ** 63)])
        return self._wrng

    def __len__(self):
        return len(self.rows) * self.n

    def _load(self, r):
        """Image as float RGB in [0, 1], mask as 0/1 floats."""
        img = np.array(Image.open(r["image"]).convert("RGB"), np.float32) / 255.0
        msk = (np.array(Image.open(r["mask"]).convert("L")) > 127).astype(np.float32)
        return img, msk

    def __getitem__(self, i):
        img, msk = self._load(self.rows[i // self.n])     # item i is crop (i % n) of image (i // n)
        h, w = msk.shape
        p = self.patch

        # pad if the image is smaller than one patch
        if h < p or w < p:
            ph, pw = max(0, p - h), max(0, p - w)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            msk = np.pad(msk, ((0, ph), (0, pw)), mode="reflect")
            h, w = msk.shape

        # ---- Choose the crop position ----
        if self.train:
            ys, xs = np.nonzero(msk)                      # coordinates of spliced pixels
            if len(ys) and self.rng.random() < 0.5:      # half the crops on tampered pixels
                k = self.rng.integers(len(ys))            # pick a random spliced pixel ...
                y = int(np.clip(ys[k] - p // 2, 0, h - p))  # ... and centre the crop on it
                x = int(np.clip(xs[k] - p // 2, 0, w - p))
            else:                                         # otherwise a uniformly random crop
                y = int(self.rng.integers(0, h - p + 1))
                x = int(self.rng.integers(0, w - p + 1))
        else:
            y, x = (h - p) // 2, (w - p) // 2             # validation: fixed centre crop

        img, msk = img[y:y+p, x:x+p], msk[y:y+p, x:x+p]

        # ---- Augment (training only): random 90-degree rotation and flip ----
        if self.train:
            k = self.rng.integers(4)
            img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
            if self.rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]

        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))   # HWC -> CHW
        msk = torch.from_numpy(np.ascontiguousarray(msk))[None]                # add channel dim
        return img, msk


def load_rows():
    """All rows of the splicing manifest (image path, mask path, fold, ...)."""
    path = os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv")
    return list(csv.DictReader(open(path)))
