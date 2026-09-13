"""Data-size curve: how does F1 move with the number of training masks?

This is the Week 6 topic from the course brief, and it answers the question that
hung over the project after the dataset size was corrected from 649 to 1822 --
whether the training-set size is the binding constraint on rung 4.

Uses the same nested protocol as the full run: for each training-set size, every
(test, val) pair, so the numbers are directly comparable to rung 4's 0.6979.
Smaller sets mean fewer steps per epoch, so these are much cheaper than the full
run -- roughly proportional to n.
"""
import os, subprocess, sys, time

SIZES = [int(x) for x in os.environ.get("SIZES", "50,150,400,800").split(",")]
BASE = dict(os.environ, NESTED="1", EPOCHS="90", PATIENCE="10",
            WD="1e-4", CLIP="1.0", WORKERS=os.environ.get("WORKERS", "2"))

for n in SIZES:
    done = len([f for f in os.listdir("results")
                if f.startswith("rung4_pair_t") and f.endswith(f"_n{n}.csv")])
    if done == 20:
        print(f"n={n}: already complete, skipping", flush=True)
        continue
    print(f"\n=== n_train = {n} ({done}/20 pairs done) ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "src/methods/run_unet.py"],
                       env=dict(BASE, N_TRAIN=str(n)))
    if r.returncode != 0:
        print(f"n={n} interrupted (exit {r.returncode}) — rerun this script to resume",
              flush=True)
        break
    print(f"  n={n} done [{time.time()-t0:.0f}s]", flush=True)

print("\nsummarise with:", flush=True)
for n in SIZES:
    print(f"  python3 src/methods/merge_nested.py _n{n}", flush=True)
