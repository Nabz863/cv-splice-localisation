#!/bin/bash
#SBATCH --job-name=cv-datasize
#SBATCH --output=results/logs/rerun_datasize_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
# Wits MS cluster: nodes with dead GPUs (Slurm can't see this, so exclude them by name).
#SBATCH --exclude=mscluster44,mscluster50,mscluster65,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82,mscluster62,mscluster83
#
# Data-size curve (Fig. 8): the same rung 4/5 pipeline trained on 50/150/400/800 masks,
# then the fp16 stabiliser comparison (Fig. 3). ~20 h on one RTX 3090. Runs alongside
# rerun_unet.sbatch. Safe to resubmit: finished sizes are skipped.
#
# Submit from the repo root:   mkdir -p results/logs && DATASETS=/path/to/data sbatch slurm/rerun_datasize.sbatch
set -euo pipefail

# ---- Environment ----
if [ -d /usr/local/cuda-12.0 ]; then                       # Wits cluster's CUDA install, if present
  export PATH=/usr/local/cuda-12.0/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
  export CUDA_HOME=/usr/local/cuda-12.0
fi
: "${DATASETS:?set DATASETS to the folder that contains casia2/}"
export WANDB_MODE=offline

cd "${SLURM_SUBMIT_DIR:-.}"                                 # the repo root you submitted from
set +u; source venv/bin/activate; set -u
echo "node: $(hostname)   commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo '+uncommitted')"

# ---- Stop early if this node has no working GPU ----
python3 - <<'PY'
import socket, sys, torch
if not torch.cuda.is_available():
    print(f"NO GPU on {socket.gethostname()} - add to the exclude list", flush=True)
    sys.exit(1)
print(f"GPU ok on {socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
PY

# ---- First launch only: move old data-size outputs to results/stale/ ----
if [ ! -e results/.rerun_datasize_started ]; then
  STALE=results/stale/datasize_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$STALE"
  shopt -s nullglob
  for f in results/models/refine_t?v?_n* results/logits/t?v?_n* results/ckpt_refine/t?v?_n* \
           results/rung5_pair_t?v?_n*.csv results/rung5_mrf_refine_n*.csv \
           results/rung5_training_n*.csv; do
    if [ -e "$f" ]; then mv "$f" "$STALE/"; fi
  done
  shopt -u nullglob
  touch results/.rerun_datasize_started
fi

# ---- For each training-set size: train 20 models, sweep (tau, beta), summarise ----
for n in 50 150 400 800; do
  if [ -e "results/rung5_mrf_refine_n$n.csv" ]; then echo "n=$n: done already"; continue; fi
  echo "=== n_train = $n ==="
  NESTED=1 BATCH=8 WORKERS=2 N_TRAIN=$n python3 src/methods/cache_logits.py     # GPU
  SUFFIX=_n$n CUT_WORKERS=12 python3 src/methods/sweep_cached.py                # CPU
  SUFFIX=_n$n python3 src/methods/merge_refine.py
  rm -f results/logits/t?v?_n$n.npz results/logits/t?v?_n$n.json               # ~2.3 GB, no longer needed
done

# ---- fp16 stabiliser comparison (Fig. 3) ----
ONLY=baseline,clip,wd1e-3 python3 src/methods/compare_stabilisers.py

rm -f results/.rerun_datasize_started
echo "data-size rerun complete."
