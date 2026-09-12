"""Full nested CV for rung 4: every (test, val) pair, not just (k, k+1).

Rung 2's 20-pair enumeration showed the fixed (k+1)%5 pairing is biased upward
by 0.0055 F1 -- it scored above its fold's all-pairs mean in 5 of 5 folds,
because a systematic rule propagates any adjacent-fold correlation identically
through every fold. Enumerating all 20 removes that.

Each pair writes its own CSV and completed pairs are skipped, so an interrupted
session resumes rather than restarts.
"""
import csv, itertools, os, subprocess, sys, time

N_FOLDS = 5
ENV = dict(os.environ, EPOCHS=os.environ.get("EPOCHS", "60"),
           PATIENCE=os.environ.get("PATIENCE", "8"),
           WORKERS=os.environ.get("WORKERS", "2"),
           NORM="sum", WD="0", CLIP="1.0")

pairs = [(k, v) for k, v in itertools.product(range(N_FOLDS), range(N_FOLDS)) if k != v]
done = [p for p in pairs if os.path.exists(f"results/rung4_pair_t{p[0]}_v{p[1]}.csv")]
todo = [p for p in pairs if p not in done]

print(f"{len(pairs)} pairs total | {len(done)} done | {len(todo)} to run", flush=True)
if done:
    print("  done:", ", ".join(f"t{k}v{v}" for k, v in done), flush=True)

for k, v in todo:
    t0 = time.time()
    print(f"\n=== test fold {k}, val fold {v} ===", flush=True)
    r = subprocess.run([sys.executable, "src/methods/run_unet_one.py"],
                       env=dict(ENV, TEST_FOLD=str(k), VAL_FOLD=str(v)))
    if r.returncode != 0:
        print(f"pair t{k}v{v} FAILED (exit {r.returncode}) - stopping", flush=True)
        break
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

print("\nrun src/methods/merge_nested.py to summarise", flush=True)
