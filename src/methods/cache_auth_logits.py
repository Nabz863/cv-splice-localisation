"""Image-level AUC for rungs 4 and 5, phase 1: U-Net logits for authentic images.

The tampered side is already cached (results/logits/{tag}.npz holds that model's
test-fold logits), but the authentic images were never put through the network, so
image-level AUC for the learned rungs needs one forward pass over them.

GPU ONLY, and deliberately free of multiprocessing: same discipline as
cache_logits.py, for the same reason. The graph cuts and the AUC itself happen in
image_auc_unet.py, which imports no torch.

N_AUTH defaults to 2000 rather than all 7,492 because the cache is float16 maps on
disk (~200 KB/image) and the confidence interval is set by the smaller class, which
is the 1,822 tampered images either way. Set N_AUTH=0 to cache all of them.

Usage (from the repo root):
    python src/methods/cache_auth_logits.py
    TAG=t0v2 N_AUTH=0 python src/methods/cache_auth_logits.py

Writes results/logits/auth_{tag}.npz
"""
import csv, glob, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet
import json
import refine_meta

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")
NORM = os.environ.get("NORM", "sum")
TAG = os.environ.get("TAG", "t0v1")
N_AUTH = int(os.environ.get("N_AUTH", 2000))


@torch.no_grad()
def logits_full(model, path):
    """Same padding and un-padding as cache_logits.py, so the maps are comparable."""
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    return model(x)[0, 0].float().cpu().numpy()[:h, :w]


def main():
    out = f"results/logits/auth_{TAG}.npz"
    if os.path.exists(out):
        print(f"{out} exists already; delete it to recache")
        return

    saved = f"results/models/refine_{TAG}.pt"
    if not os.path.exists(saved):
        have = sorted(glob.glob("results/models/*.pt"))
        raise SystemExit(f"{saved} not found. models present: {have}")

    # The model must be the one whose test logits are in results/logits/{TAG}.npz.
    lg_meta = refine_meta.sidecar(f"results/logits/{TAG}.npz")
    if not os.path.exists(lg_meta):
        raise SystemExit(f"{lg_meta} missing: run cache_logits.py first")
    refine_meta.check(saved, json.load(open(lg_meta)))
    model = UNet(norm=NORM).to(DEV)
    model.load_state_dict(torch.load(saved, map_location=DEV))
    model.eval()
    print(f"loaded {saved} on {DEV}")

    auth_dir = os.environ.get("AUTH_DIR", os.path.join(os.environ["DATASETS"], "casia2",
                                                       "images", "CASIA2.0_revised", "Au"))
    paths = sorted(sum((glob.glob(os.path.join(auth_dir, e))
                        for e in ("*.jpg", "*.JPG", "*.tif", "*.TIF",
                                  "*.png", "*.PNG", "*.bmp", "*.BMP")), []))
    if not paths:
        raise SystemExit(f"no images in {auth_dir}")
    if N_AUTH and N_AUTH < len(paths):
        paths = sorted(np.random.default_rng(0).permutation(paths)[:N_AUTH])
    print(f"{len(paths)} authentic images from {auth_dir}\n")

    blob, kept, t0 = {}, [], time.time()
    for i, p in enumerate(paths):
        try:
            lg = logits_full(model, p).astype(np.float16)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)
            continue
        j = len(kept)
        blob[f"lg_{j}"] = lg
        blob[f"shape_{j}"] = np.array(lg.shape, np.int32)
        kept.append(p)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)

    blob["n"] = np.array(len(kept), np.int32)
    blob["paths"] = np.array(kept)

    os.makedirs("results/logits", exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", **blob)
    os.replace(out + ".tmp.npz", out)
    print(f"\ncached {len(kept)} authentic maps -> {out} "
          f"({os.path.getsize(out)/1e6:.0f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
