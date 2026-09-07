"""Patch sampling for rung 4.

Uniform random crops over-sample all-authentic background, so most patches carry
no boundary and the network wastes capacity on trivial examples. We bias half the
crops to contain tampered pixels.
"""
import csv, os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH = 256


class SpliceCrops(Dataset):
    def __init__(self, rows, patch=PATCH, train=True, crops_per_image=8, seed=0):
        self.rows, self.patch, self.train = rows, patch, train
        self.n = crops_per_image if train else 1
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows) * self.n

    def _load(self, r):
        img = np.array(Image.open(r["image"]).convert("RGB"), np.float32) / 255.0
        msk = (np.array(Image.open(r["mask"]).convert("L")) > 127).astype(np.float32)
        return img, msk

    def __getitem__(self, i):
        img, msk = self._load(self.rows[i // self.n])
        h, w = msk.shape
        p = self.patch

        # pad if the image is smaller than one patch
        if h < p or w < p:
            ph, pw = max(0, p - h), max(0, p - w)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            msk = np.pad(msk, ((0, ph), (0, pw)), mode="reflect")
            h, w = msk.shape

        if self.train:
            ys, xs = np.nonzero(msk)
            if len(ys) and self.rng.random() < 0.5:      # half the crops on tampered pixels
                k = self.rng.integers(len(ys))
                y = int(np.clip(ys[k] - p // 2, 0, h - p))
                x = int(np.clip(xs[k] - p // 2, 0, w - p))
            else:
                y = int(self.rng.integers(0, h - p + 1))
                x = int(self.rng.integers(0, w - p + 1))
        else:
            y, x = (h - p) // 2, (w - p) // 2

        img, msk = img[y:y+p, x:x+p], msk[y:y+p, x:x+p]

        if self.train:
            # flips and 90-degree rotations are exact label-preserving transforms
            # here: a forensic patch has no canonical orientation.
            k = self.rng.integers(4)
            img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
            if self.rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]

        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        msk = torch.from_numpy(np.ascontiguousarray(msk))[None]
        return img, msk


def load_rows():
    path = os.path.join(os.environ["DATASETS"], "casia2", "splice_manifest.csv")
    return list(csv.DictReader(open(path)))
