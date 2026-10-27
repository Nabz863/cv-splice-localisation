"""Rung 4/5, step 1 (GPU): train each (test, validation) pair's U-Net and save its logits.

Writes results/models/refine_<tag>.pt, results/logits/<tag>.npz (validation + test logits)
and a row in results/rung5_training{SUFFIX}.csv. N_TRAIN=n trains on n masks (data-size curve).
Usage:  BATCH=8 WORKERS=2 python src/methods/cache_logits.py
"""
import csv, os, sys, time                    # standard-library helpers
import numpy as np                            # arrays and maths
import torch                                  # PyTorch: neural networks on the GPU
from PIL import Image                         # reads image files
from torch.utils.data import DataLoader       # feeds the network batches of training crops

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # so "utils" (in src/) can be imported
sys.path.insert(0, os.path.dirname(__file__))                       # so files in this folder can be imported
from utils.seed import set_seed               # fixes every random number generator
from unet import UNet, dice_bce               # the network and its loss function
from patch_data import SpliceCrops, load_rows, PATCH, CROP_RNG   # training crops and the manifest
import refine_meta                            # saves/checks the config file next to each model

DEV = "cuda" if torch.cuda.is_available() else "cpu"     # use the GPU if there is one
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):     # training on CPU would take days, so stop
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

# ---- Settings (environment variables override the defaults) ----
EPOCHS = int(os.environ.get("EPOCHS", 90))          # maximum passes over the training data
PATIENCE = int(os.environ.get("PATIENCE", 10))      # stop after this many epochs without validation improvement
BATCH = int(os.environ.get("BATCH", 8))             # crops per training step
N_FOLDS = int(os.environ.get("N_FOLDS", 5))         # number of cross-validation folds
WORKERS = int(os.environ.get("WORKERS", 2))         # background processes that load images
NORM = os.environ.get("NORM", "sum")                # constrained-layer normalisation ("sum" is the paper's)
WD = float(os.environ.get("WD", 1e-4))              # weight decay (a small penalty on large weights)
CLIP = float(os.environ.get("CLIP", 1.0))           # gradient clipping: cap the gradient's size at 1
N_CAL = int(os.environ.get("N_CAL", 150))           # validation images used to pick thresholds
NESTED = bool(int(os.environ.get("NESTED", 1)))     # 1 = all 20 (test, val) pairs
N_TRAIN = int(os.environ.get("N_TRAIN", 0))          # 0 = all training images
SUFFIX = f"_n{N_TRAIN}" if N_TRAIN else ""          # file-name suffix for data-size runs, e.g. "_n50"
AMP_DTYPE = torch.bfloat16                          # number format for the forward pass (keeps fp32's range)
CROPS = 8                                           # SpliceCrops default
# everything that defines a trained model; saved next to each model and logits file
CONFIG = dict(epochs_max=EPOCHS, patience=PATIENCE, batch=BATCH, workers=WORKERS, lr=1e-3,
              weight_decay=WD, grad_clip=CLIP, constraint_norm=NORM, amp_dtype="bfloat16",
              n_cal=N_CAL, crops_per_image=CROPS, patch=PATCH, n_train=N_TRAIN,
              crop_rng=CROP_RNG, nonfinite_counted=1)


@torch.no_grad()                                      # inference only: no gradients needed
def logits_full(model, path):
    """Run the model on a whole image (padded to a multiple of 8) and return its logits."""
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0   # load, scale pixels to 0..1
    h, w = img.shape[:2]                              # original height and width
    ph, pw = (-h) % 8, (-w) % 8                       # 3 poolings need sides divisible by 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")        # pad bottom/right by mirroring
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)          # to (1, channels, H, W) on the GPU
    return model(x)[0, 0].float().cpu().numpy()[:h, :w]   # crop the padding off again


# ---- Training ----

def train_one(test_fold, val_fold, rows, tag):
    """Train one U-Net on the three remaining folds, early-stopping on the validation fold."""
    set_seed(test_fold * 10 + val_fold)               # one fixed seed per pair
    tr = [r for r in rows if int(r["fold"]) not in (test_fold, val_fold)]   # training folds
    if N_TRAIN:                                       # data-size runs use only N_TRAIN of them ...
        tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]       # fixed random subset
    va = [r for r in rows if int(r["fold"]) == val_fold]                    # validation fold

    os.makedirs("results/ckpt_refine", exist_ok=True)   # folder for resumable checkpoints
    ck = f"results/ckpt_refine/{tag}.pt"              # checkpoint, so an interrupted run can resume

    # DataLoaders hand out shuffled batches of random 256x256 crops (training) or fixed centre crops (validation)
    dl = DataLoader(SpliceCrops(tr, train=True, crops_per_image=CROPS, seed=test_fold),
                    batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,  # validation batches (fixed crops, no shuffling)
                     shuffle=False, num_workers=WORKERS)

    model = UNet(norm=NORM).to(DEV)                   # a fresh, randomly initialised network on the GPU
    # AdamW = the optimiser that updates the weights (learning rate 0.001, with weight decay)
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD) if WD
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)     # cosine learning-rate decay
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))        # mixed-precision loss scaling

    # best = lowest validation loss so far; best_state = weights at that point; bad = epochs since it improved
    best, best_state, bad, start_ep, stopped = float("inf"), None, 0, 0, EPOCHS
    nf_train = nf_val = 0                         # count of batches skipped for NaN/inf loss
    if os.path.exists(ck):                         # resume from a checkpoint
        refine_meta.check(ck, CONFIG)              # only if it was made under the same config
        st = torch.load(ck, map_location=DEV, weights_only=False)                  # load everything saved
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])          # weights and optimiser
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])    # schedule and scaler
        best, best_state, bad, start_ep = st["best"], st["best_state"], st["bad"], st["epoch"]
        nf_train, nf_val = st["nf_train"], st["nf_val"]  # skipped-batch counts so far
        print(f"  resumed {tag} from epoch {start_ep}", flush=True)

    model.constrain()                              # apply the first-layer constraint before training starts
    for ep in range(start_ep, EPOCHS):             # one loop iteration = one epoch
        # ---- one training epoch ----
        model.train()                              # training mode (BatchNorm uses batch statistics)
        tot, n, t0, bad_tr, bad_va = 0.0, 0, time.time(), 0, 0   # running loss, crop count, timer, bad batches
        for x, y in dl:                            # x = batch of image crops, y = their masks
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)   # move to the GPU
            opt.zero_grad()                        # clear gradients from the previous step
            with torch.amp.autocast(DEV, dtype=AMP_DTYPE, enabled=(DEV == "cuda")):   # bfloat16 forward pass
                loss = dice_bce(model(x), y)       # predict, then measure how wrong the prediction is
            if not torch.isfinite(loss):           # skip (and count) a NaN/inf batch
                bad_tr += 1
                continue
            scaler.scale(loss).backward()          # backpropagation: compute each weight's gradient
            if CLIP > 0:                           # gradient clipping (norm 1)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # cap the total gradient size
            scaler.step(opt); scaler.update()      # update the weights
            model.constrain()                      # restore the first-layer constraint
            tot += loss.item() * x.size(0); n += x.size(0)   # add up the loss for the epoch average
        sched.step()                               # lower the learning rate a little (cosine schedule)
        nf_train += bad_tr
        if n == 0:                                 # every batch failed: training has broken down
            print(f"  {tag} ep {ep+1}: every training batch non-finite - stopping", flush=True)
            break

        # ---- validation loss ----
        model.eval()                               # evaluation mode
        vl, vn = 0.0, 0                            # total validation loss and crop count
        with torch.no_grad():                      # no gradients needed for validation
            for xv, yv in vdl:
                xv, yv = xv.to(DEV), yv.to(DEV)         # move the batch to the GPU
                l = dice_bce(model(xv), yv)             # validation loss
                if torch.isfinite(l):
                    vl += l.item() * xv.size(0); vn += xv.size(0)  # add to the running total
                else:
                    bad_va += 1                         # count skipped validation batches
        nf_val += bad_va                                # add to the model's total
        vl = vl / vn if vn else float("inf")       # average validation loss
        print(f"  {tag} ep {ep+1}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f} "
              f"({time.time()-t0:.0f}s)"
              + (f"  NON-FINITE batches: train {bad_tr}, val {bad_va}" if bad_tr or bad_va
                 else ""), flush=True)

        # ---- early stopping: keep the best weights, stop after PATIENCE epochs without improvement ----
        if vl < best - 1e-4:                       # improved (by more than a tiny margin)
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}   # copy the weights
        else:
            bad += 1                                    # count skipped training batches
            if bad >= PATIENCE:                    # no improvement for PATIENCE epochs: stop
                stopped = ep + 1
                print(f"  early stop at epoch {ep+1} (best val {best:.4f})", flush=True)
                break

        # save a checkpoint (write to .tmp, then rename, so a crash can't leave a broken file)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_state": best_state, "bad": bad,
                    "epoch": ep + 1, "nf_train": nf_train, "nf_val": nf_val}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)                     # swap in the new checkpoint in one step
        refine_meta.write(ck, CONFIG)              # record the config next to the checkpoint

    if best_state is not None:
        model.load_state_dict(best_state)          # restore the best-validation weights
    model.eval()
    if os.path.exists(ck):                         # training finished: the checkpoint is no longer needed
        os.remove(ck)
        os.remove(refine_meta.sidecar(ck))              # its settings file too

    # ---- save the model, its config, and a row in the training log ----
    os.makedirs("results/models", exist_ok=True)
    saved = f"results/models/refine_{tag}.pt"           # final model file
    torch.save(model.state_dict(), saved)          # the final (best) weights
    refine_meta.write(saved, CONFIG, stopped_at=stopped, best_val=best,
                      nonfinite_train=nf_train, nonfinite_val=nf_val)

    log = f"results/rung5_training{SUFFIX}.csv"    # one row per trained model
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:          # "a" = append to the file
        w = csv.writer(f)
        if new:                                    # write the header row the first time only
            w.writerow(["test_fold", "val_fold", "stopped_at", "best_val", "nonfinite_train",
                        "nonfinite_val"] + list(refine_meta.KEYS))
        w.writerow([test_fold, val_fold, stopped, f"{best:.6f}", nf_train, nf_val]  # one row per model
                   + [CONFIG[k] for k in refine_meta.KEYS])
    return model


# ---- Logits ----

def cache_pair(test_fold, val_fold, rows):
    """Train (or load) one pair's model and save its logits on 150 validation and all test images."""
    tag = f"t{test_fold}v{val_fold}{SUFFIX}"      # e.g. "t0v1" = test fold 0, validation fold 1
    out = f"results/logits/{tag}.npz"
    if os.path.exists(out):                        # already done for this config
        refine_meta.check(out, CONFIG)
        print(f"{tag}: cached already", flush=True)
        return

    saved = f"results/models/refine_{tag}.pt"           # final model file for this pair
    if os.path.exists(saved):                      # a trained model exists: load it instead of retraining
        refine_meta.check(saved, CONFIG)
        model = UNet(norm=NORM).to(DEV)                 # fresh model ...
        model.load_state_dict(torch.load(saved, map_location=DEV))  # ... with the saved weights
        model.eval()                                    # evaluation mode
        print(f"{tag}: loaded {saved}", flush=True)
    else:
        print(f"{tag}: training", flush=True)
        model = train_one(test_fold, val_fold, rows, tag)  # no saved model: train one

    va = [r for r in rows if int(r["fold"]) == val_fold]    # validation images
    te = [r for r in rows if int(r["fold"]) == test_fold]   # test images
    cal = list(np.random.default_rng(1).permutation(va))[:N_CAL]   # fixed 150 validation images

    t0 = time.time()                                    # start a timer
    blob = {}                                      # every array to save, by name
    for split, subset in (("cal", cal), ("test", te)):   # "cal" = validation images used for choosing thresholds
        for i, r in enumerate(subset):
            lg = logits_full(model, r["image"]).astype(np.float16)  # float16 halves the file size
            m = (np.array(Image.open(r["mask"]).convert("L")) > 127)   # ground-truth mask as True/False
            blob[f"{split}_lg_{i}"] = lg
            blob[f"{split}_gt_{i}"] = np.packbits(m)      # 1 bit/pixel
            blob[f"{split}_shape_{i}"] = np.array(m.shape, np.int32)   # needed to unpack the bits later
        blob[f"n_{split}"] = np.array(len(subset), np.int32)          # how many images in this split

    os.makedirs("results/logits", exist_ok=True)        # output folder
    np.savez_compressed(out + ".tmp.npz", **blob)  # write to a temporary name ...
    os.replace(out + ".tmp.npz", out)              # ... then rename (never a half-written file)
    refine_meta.write(out, CONFIG)                 # record the config next to the logits
    mb = os.path.getsize(out) / 1e6
    print(f"{tag}: cached {len(cal)} cal + {len(te)} test "
          f"({mb:.0f} MB, {time.time()-t0:.0f}s)\n", flush=True)


if __name__ == "__main__":
    rows = load_rows()                             # every splicing image, with its fold
    # all 20 ordered (test, validation) pairs, or 5 fixed pairs (val = test + 1) if NESTED=0
    pairs = ([(k, v) for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
             if NESTED else [(k, (k + 1) % N_FOLDS) for k in range(N_FOLDS)])
    print(f"device {DEV} | {len(pairs)} pairs | no multiprocessing in this phase", flush=True)
    print("config " + " ".join(f"{k}={v}" for k, v in CONFIG.items()), flush=True)
    for k, v in pairs:                             # train and cache each pair in turn
        cache_pair(k, v, rows)
    print("phase 1 done. now run: python3 src/methods/sweep_cached.py", flush=True)
