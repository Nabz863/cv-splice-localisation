"""Image-level AUC for rungs 4/5, step 1 (GPU): run one pair's U-Net on the authentic images.

The tampered side was already saved by cache_logits.py; image_auc_unet.py does the scoring.
Usage:  TAG=t0v1 N_AUTH=0 python src/methods/cache_auth_logits.py     (N_AUTH=0: all authentic images)
"""
import csv, glob, os, sys, time       # standard-library helpers (file paths, timing, ...)
import numpy as np                     # arrays and maths
import torch                           # PyTorch: runs the neural network
from PIL import Image                  # reads image files

sys.path.insert(0, os.path.dirname(__file__))   # let Python find our own modules in this folder
from unet import UNet                  # the network definition (rung 4)
import json                            # reads the JSON config file saved next to each model
import refine_meta                     # checks that a saved model matches the expected config

# ---- Settings (each can be overridden with an environment variable) ----
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # use the GPU if there is one
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):   # without a GPU this would take hours,
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")  # so stop
NORM = os.environ.get("NORM", "sum")           # which normalisation the constrained first layer used
TAG = os.environ.get("TAG", "t0v1")            # which (test, validation) pair's model: t0v1 = test fold 0, val fold 1
N_AUTH = int(os.environ.get("N_AUTH", 2000))   # how many authentic images to use (0 = all of them)


@torch.no_grad()                               # inference only: don't track gradients (faster, less memory)
def logits_full(model, path):
    """Whole-image logits, padded exactly as in cache_logits.py."""
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0   # load as RGB, scale pixels to 0..1
    h, w = img.shape[:2]                                                  # original height and width
    # The U-Net halves the size 3 times, so height and width must be multiples of 8.
    ph, pw = (-h) % 8, (-w) % 8                                           # rows/columns of padding needed
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")        # pad bottom/right by mirroring
    # PyTorch wants (batch, channels, height, width): move channels first and add a batch of 1
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    # Run the network, take the single output map, move it back to NumPy, crop the padding off
    return model(x)[0, 0].float().cpu().numpy()[:h, :w]


def main():
    out = f"results/logits/auth_{TAG}.npz"     # where this pair's authentic-image logits will be saved
    if os.path.exists(out):                    # already done: nothing to do
        print(f"{out} exists already; delete it to recache")
        return

    saved = f"results/models/refine_{TAG}.pt"  # the trained model for this pair (written by cache_logits.py)
    if not os.path.exists(saved):              # can't continue without it
        have = sorted(glob.glob("results/models/*.pt"))
        raise SystemExit(f"{saved} not found. models present: {have}")

    # ---- Load the model and check it is the one that produced this pair's test logits ----
    lg_meta = refine_meta.sidecar(f"results/logits/{TAG}.npz")    # the .json config saved next to the test logits
    if not os.path.exists(lg_meta):
        raise SystemExit(f"{lg_meta} missing: run cache_logits.py first")
    meta = json.load(open(lg_meta))                       # the config the logits were made under
    if "upsample" not in meta:                            # made before the decoder switch: refuse
        raise SystemExit(f"{lg_meta} predates the resize-then-convolve decoder; rerun "
                         f"cache_logits.py (slurm/rerun_unet.sbatch) first")
    refine_meta.check(saved, meta)                        # stop if the model was trained under a different config
    # Build the architecture the sidecar records, not whatever the default is now.
    model = UNet(norm=meta["constraint_norm"], upsample=meta["upsample"]).to(DEV)   # untrained network of the right shape
    model.load_state_dict(torch.load(saved, map_location=DEV))   # fill it with the trained weights
    model.eval()                                          # evaluation mode (BatchNorm uses its stored statistics)
    print(f"loaded {saved} on {DEV}")

    # ---- List the authentic images ----
    auth_dir = os.environ.get("AUTH_DIR", os.path.join(os.environ["DATASETS"], "casia2",
                                                       "images", "CASIA2.0_revised", "Au"))   # folder of authentic images
    # Find every image file in that folder (several extensions, upper and lower case) and sort the list
    paths = sorted(sum((glob.glob(os.path.join(auth_dir, e))
                        for e in ("*.jpg", "*.JPG", "*.tif", "*.TIF",
                                  "*.png", "*.PNG", "*.bmp", "*.BMP")), []))
    if not paths:                                         # wrong folder, or no data
        raise SystemExit(f"no images in {auth_dir}")
    if N_AUTH and N_AUTH < len(paths):                    # optionally use only a subset ...
        paths = sorted(np.random.default_rng(0).permutation(paths)[:N_AUTH])   # ... chosen randomly but with a fixed seed
    print(f"{len(paths)} authentic images from {auth_dir}\n")

    # ---- Run the model on each image ----
    blob, kept, t0 = {}, [], time.time()       # blob: arrays to save; kept: images that worked; t0: start time
    for i, p in enumerate(paths):              # i = position in the list, p = file path
        try:
            lg = logits_full(model, p).astype(np.float16)   # logits for this image; float16 halves the file size
        except Exception as e:                                  # unreadable image: skip, but say so
            print(f"  skip {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)
            continue
        j = len(kept)                          # index among the images that succeeded
        blob[f"lg_{j}"] = lg                   # store the logit map under the name lg_0, lg_1, ...
        blob[f"shape_{j}"] = np.array(lg.shape, np.int32)   # and its height/width
        kept.append(p)                         # remember which file this was
        if (i + 1) % 250 == 0:                 # progress message every 250 images
            print(f"  {i+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)

    blob["n"] = np.array(len(kept), np.int32)  # number of images saved
    blob["paths"] = np.array(kept)             # their file paths, in order

    # ---- Save everything in one compressed .npz file ----
    os.makedirs("results/logits", exist_ok=True)              # create the folder if it doesn't exist
    np.savez_compressed(out + ".tmp.npz", **blob)             # write to a temporary name first ...
    os.replace(out + ".tmp.npz", out)                         # ... then rename, so a crash never leaves a half-written file
    print(f"\ncached {len(kept)} authentic maps -> {out} "
          f"({os.path.getsize(out)/1e6:.0f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":                    # only run main() when this file is run as a script
    main()
