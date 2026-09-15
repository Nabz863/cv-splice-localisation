"""Extension: the MRF over the U-Net's logits instead of the ELA map.

Same energy, same solver, same grid-search calibration as rung 2 -- only the
unary changes. With a learned unary this is the dense-CRF refinement
configuration used by DeepLab, so it is arguably a CRF rather than an MRF;
noted as such in the report.

THE QUESTION. Over the ELA map the Potts prior bought +0.0224 F1 (0.2621 ->
0.2845). Does it still help once the evidence is strong? If the gain largely
vanishes, the prior was compensating for a noisy unary rather than contributing
independent spatial information -- which is a real finding either way.

PROTOCOL. Identical to rung 4: the same 20 (test, val) pairs, the same training
folds, the same bf16 / weight-decay / clipping configuration. (tau, beta) are
grid-searched on the validation fold and applied unchanged to test. beta = 0
reduces exactly to thresholding the U-Net's logits, which is rung 4 -- so the
control is built in and every gain is attributable to the prior alone.

Tau is in LOGIT space here, not the z-score space rung 2 used: 0 corresponds to
p(tampered) = 0.5. Hence the negative values in the grid.
"""
import csv, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seed
from unet import UNet, dice_bce
from patch_data import SpliceCrops, load_rows
from mrf import solve_graphcut, energy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("GPU unavailable - refusing CPU fallback. Set ALLOW_CPU=1 to override.")

EPOCHS = int(os.environ.get("EPOCHS", 90))
PATIENCE = int(os.environ.get("PATIENCE", 10))
BATCH = int(os.environ.get("BATCH", 8))
N_FOLDS = int(os.environ.get("N_FOLDS", 5))
WORKERS = int(os.environ.get("WORKERS", 2))
NORM = os.environ.get("NORM", "sum")
WD = float(os.environ.get("WD", 1e-4))
CLIP = float(os.environ.get("CLIP", 1.0))
NESTED = bool(int(os.environ.get("NESTED", 1)))
N_TRAIN = int(os.environ.get("N_TRAIN", 0))     # 0 = all; for smoke tests
N_CAL = int(os.environ.get("N_CAL", 150))       # calibration images
CUT_WORKERS = int(os.environ.get("CUT_WORKERS", 10))
USE_WANDB = bool(int(os.environ.get("WANDB", 1)))
AMP_DTYPE = torch.bfloat16

TAUS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]        # logit space
BETAS = [0.0, 4.0, 16.0, 32.0, 64.0, 128.0, 256.0]
GRID = [(t, b) for t in TAUS for b in BETAS]

if USE_WANDB:
    import wandb


@torch.no_grad()
def logits_full(model, path):
    """Full-resolution logits. Pad to a multiple of 8 so three pools and three
    upsamples land back on the original size exactly."""
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    return model(x)[0, 0].float().cpu().numpy()[:h, :w]


def train_one(test_fold, val_fold, rows, tag):
    """Identical recipe to rung 4 so the comparison is like-for-like."""
    set_seed(test_fold * 10 + val_fold)
    tr = [r for r in rows if int(r["fold"]) not in (test_fold, val_fold)]
    va = [r for r in rows if int(r["fold"]) == val_fold]
    if N_TRAIN:
        tr = list(np.random.default_rng(0).permutation(tr))[:N_TRAIN]

    ck = f"results/ckpt_refine/{tag}.pt"
    os.makedirs("results/ckpt_refine", exist_ok=True)

    dl = DataLoader(SpliceCrops(tr, train=True, seed=test_fold), batch_size=BATCH,
                    shuffle=True, num_workers=WORKERS, drop_last=True)
    vdl = DataLoader(SpliceCrops(va, train=False), batch_size=BATCH,
                     shuffle=False, num_workers=WORKERS)

    model = UNet(norm=NORM).to(DEV)
    opt = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WD) if WD
           else torch.optim.Adam(model.parameters(), lr=1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))

    best, best_state, bad, start_ep = float("inf"), None, 0, 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        best, best_state, bad, start_ep = st["best"], st["best_state"], st["bad"], st["epoch"]
        print(f"  resumed {tag} from epoch {start_ep} (best val {best:.4f})", flush=True)

    model.constrain()
    for ep in range(start_ep, EPOCHS):
        model.train()
        tot, n, t0 = 0.0, 0, time.time()
        for x, y in dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(DEV, dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            if CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt); scaler.update()
            model.constrain()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        if n == 0:
            break

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for xv, yv in vdl:
                xv, yv = xv.to(DEV), yv.to(DEV)
                l = dice_bce(model(xv), yv)
                if torch.isfinite(l):
                    vl += l.item() * xv.size(0); vn += xv.size(0)
        vl = vl / vn if vn else float("inf")
        print(f"  {tag} ep {ep+1}/{EPOCHS}  train {tot/n:.4f}  val {vl:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if USE_WANDB:
            wandb.log({"epoch": ep + 1, "train/loss": tot / n, "val/loss": vl})

        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {ep+1} (best val {best:.4f})", flush=True)
                break

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "best_state": best_state, "bad": bad,
                    "epoch": ep + 1}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if os.path.exists(ck):
        os.remove(ck)

    # Persist the weights. Downstream sweeps need the model, not just its score.
    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), f"results/models/refine_{tag}.pt")
    return model


def _cuts_one(args):
    """Graph cuts for one image across the whole grid. CPU-bound, so this runs
    in a worker process; the logits are computed on the GPU beforehand."""
    lg, t = args
    c = np.zeros((len(GRID), 3), np.int64)
    e = np.zeros(len(GRID))
    for gi, (tau, beta) in enumerate(GRID):
        x = solve_graphcut(lg, tau, beta)
        c[gi] = ((x & t).sum(), (x & ~t).sum(), (~x & t).sum())
        e[gi] = energy(lg, x, tau, beta)
    return c, e


def counts_over(model, subset, label=""):
    """Confusion counts for every (tau, beta), plus the energy reached.

    Two phases: logits on the GPU (fast, serial), then graph cuts fanned across
    CPU workers. Single-threaded this is 49 solves per image and dominates the
    runtime -- at high beta a single solve can take most of a second.
    """
    t0 = time.time()
    payload = []
    for r in subset:
        lg = logits_full(model, r["image"]).astype(np.float32)
        t = np.array(Image.open(r["mask"]).convert("L")) > 127
        payload.append((lg, t))
        if len(payload) % 50 == 0:
            torch.cuda.empty_cache()
    print(f"    {label} logits done ({time.time()-t0:.0f}s), "
          f"{len(payload)*len(GRID)} cuts across {CUT_WORKERS} workers", flush=True)

    c = np.zeros((len(GRID), 3), np.int64)
    e = np.zeros(len(GRID))
    # spawn, not fork: forking a process with a live CUDA context corrupts it,
    # and the next CUDA call fails with an illegal memory access.
    with ProcessPoolExecutor(max_workers=CUT_WORKERS,
                             mp_context=mp.get_context("spawn")) as ex:
        for i, (ci, ei) in enumerate(ex.map(_cuts_one, payload, chunksize=2)):
            c += ci; e += ei
            if (i + 1) % 50 == 0:
                print(f"    {label} {i+1}/{len(payload)} ({time.time()-t0:.0f}s)",
                      flush=True)
    return c, e / max(len(subset), 1)


def f1_of(counts, gi):
    tp, fp, fn = counts[gi]
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0


def run_pair(test_fold, val_fold, rows):
    tag = f"t{test_fold}v{val_fold}"
    if USE_WANDB:
        wandb.init(project="cv-splice-localisation", name=f"refine-{tag}",
                   group="rung5-mrf-refine",
                   config={"test_fold": test_fold, "val_fold": val_fold,
                           "epochs_max": EPOCHS, "patience": PATIENCE,
                           "batch": BATCH, "lr": 1e-3, "weight_decay": WD,
                           "grad_clip": CLIP, "amp_dtype": "bfloat16",
                           "taus": str(TAUS), "betas": str(BETAS)})

    saved = f"results/models/refine_{tag}.pt"
    if os.path.exists(saved):
        model = UNet(norm=NORM).to(DEV)
        model.load_state_dict(torch.load(saved, map_location=DEV))
        model.eval()
        print(f"  loaded {saved} - skipping training", flush=True)
    else:
        model = train_one(test_fold, val_fold, rows, tag)
    va = [r for r in rows if int(r["fold"]) == val_fold]
    te = [r for r in rows if int(r["fold"]) == test_fold]

    cal = list(np.random.default_rng(1).permutation(va))[:N_CAL]
    cc, _ = counts_over(model, cal, "cal")

    gi = int(np.argmax([f1_of(cc, g) for g in range(len(GRID))]))
    tau, beta = GRID[gi]

    # beta = 0 on the same validation set: the raw U-Net control, chosen the
    # same way, so the comparison isolates the prior and nothing else.
    zero = [g for g, (t, b) in enumerate(GRID) if b == 0.0]
    gi0 = zero[int(np.argmax([f1_of(cc, g) for g in zero]))]
    tau0 = GRID[gi0][0]

    ct, et = counts_over(model, te, "test")
    f1, iou = f1_of(ct, gi), ct[gi][0] / ct[gi].sum() if ct[gi].sum() else 0.0
    tp, fp, fn = ct[gi]
    iou = tp / (tp + fp + fn)
    f1_0 = f1_of(ct, gi0)

    if USE_WANDB:
        wandb.log({"test/f1": f1, "test/iou": iou, "test/f1_beta0": f1_0,
                   "test/tau": tau, "test/beta": beta, "test/tau_beta0": tau0,
                   "test/prior_gain": f1 - f1_0})
        wandb.finish()

    os.makedirs("results", exist_ok=True)
    with open(f"results/rung5_pair_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_fold", "val_fold", "tau", "beta", "f1", "iou",
                    "tau_beta0", "f1_beta0", "prior_gain"])
        w.writerow([test_fold, val_fold, tau, beta, f"{f1:.6f}", f"{iou:.6f}",
                    tau0, f"{f1_0:.6f}", f"{f1 - f1_0:.6f}"])
    return tau, beta, f1, iou, f1_0


if __name__ == "__main__":
    rows = load_rows()
    pairs = ([(k, v) for k in range(N_FOLDS) for v in range(N_FOLDS) if k != v]
             if NESTED else [(k, (k + 1) % N_FOLDS) for k in range(N_FOLDS)])
    pending = [(k, v) for k, v in pairs
               if not os.path.exists(f"results/rung5_pair_t{k}v{v}.csv")]

    print(f"device {DEV} | {len(rows)} pairs | {len(GRID)} (tau, beta) points | "
          f"amp=bf16 wd={WD} clip={CLIP}", flush=True)
    print(f"{len(pairs)} (test, val) pairs | {len(pairs)-len(pending)} done | "
          f"{len(pending)} to run\n", flush=True)

    for k, v in pending:
        t0 = time.time()
        tau, beta, f1, iou, f1_0 = run_pair(k, v, rows)
        print(f"t{k}v{v}: tau {tau} beta {beta}  F1 {f1:.4f} (beta=0: {f1_0:.4f}, "
              f"gain {f1-f1_0:+.4f})  IoU {iou:.4f}  [{time.time()-t0:.0f}s]\n",
              flush=True)

    print("summarise: python3 src/methods/merge_refine.py", flush=True)
