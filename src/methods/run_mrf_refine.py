"""Extension: MRF over the U-Net's logits instead of the ELA map.

Same energy, same solver, same (tau, beta) search -- only the unary changes.
With a learned unary this is the dense-CRF refinement configuration used by
DeepLab, so it is arguably a CRF rather than an MRF; noted in the report.

Question: does a spatial prior still help once the evidence is strong? Over ELA
it bought +0.023 F1. If it buys much less here, the prior was compensating for a
noisy unary rather than adding independent information.
"""
import csv, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from utils.seed import set_seed
from unet import UNet
from patch_data import SpliceCrops, load_rows
from mrf import solve_graphcut
from torch.utils.data import DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", 12))
TAUS = [-1.0, -0.5, 0.0, 0.5, 1.0]        # logit space: 0 = p(tampered) 0.5
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
GRID = [(t, b) for t in TAUS for b in BETAS]


@torch.no_grad()
def logits_full(model, path):
    img = np.array(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(DEV)
    return model(x)[0, 0].float().cpu().numpy()[:h, :w]


def train_fold(k, rows):
    set_seed(k)
    tr = [r for r in rows if int(r["fold"]) != k]
    dl = DataLoader(SpliceCrops(tr, train=True, seed=k), batch_size=8,
                    shuffle=True, num_workers=4, drop_last=True)
    from unet import dice_bce
    model = UNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    model.constrain()
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                loss = dice_bce(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            model.constrain()
        sched.step()
        print(f"  fold {k} ep {ep+1}/{EPOCHS} loss {loss.item():.4f}", flush=True)
    model.eval()
    return model


def counts_for(model, subset):
    c = np.zeros((len(GRID), 3), np.int64)
    for r in subset:
        lg = logits_full(model, r["image"])
        t = np.array(Image.open(r["mask"]).convert("L")) > 127
        for gi, (tau, beta) in enumerate(GRID):
            x = solve_graphcut(lg, tau, beta)
            c[gi] += ((x & t).sum(), (x & ~t).sum(), (~x & t).sum())
    return c


if __name__ == "__main__":
    rows = load_rows()
    out = []
    for k in range(5):
        t0 = time.time()
        model = train_fold(k, rows)
        tr = [r for r in rows if int(r["fold"]) != k]
        te = [r for r in rows if int(r["fold"]) == k]
        cal = list(np.random.default_rng(1).permutation(tr))[:150]

        cc = counts_for(model, cal)
        f1s = [2*a/(2*a+b+d) if (a+b+d) else 0 for a, b, d in cc]
        gi = int(np.argmax(f1s))
        tau, beta = GRID[gi]

        ct = counts_for(model, te)
        tp, fp, fn = ct[gi]
        f1, iou = 2*tp/(2*tp+fp+fn), tp/(tp+fp+fn)

        # beta = 0 on the same fold = the raw U-Net, for a like-for-like control
        gi0 = int(np.argmax([f1s[g] if GRID[g][1] == 0 else -1 for g in range(len(GRID))]))
        tp0, fp0, fn0 = ct[gi0]
        f1_0 = 2*tp0/(2*tp0+fp0+fn0)

        print(f"fold {k}: tau {tau} beta {beta}  F1 {f1:.4f} (beta=0: {f1_0:.4f})  "
              f"IoU {iou:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        out.append((k, tau, beta, f1, iou, f1_0))

    f1s = [o[3] for o in out]; f0s = [o[5] for o in out]
    print(f"\nmean F1 with prior    {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"mean F1 at beta=0     {np.mean(f0s):.4f}")
    print(f"prior contributes     {np.mean(f1s)-np.mean(f0s):+.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/rung5_mrf_refine.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold","tau","beta","f1","iou","f1_beta0"])
        w.writerows(out)
