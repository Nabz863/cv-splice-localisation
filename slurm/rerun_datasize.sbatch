#!/bin/bash
#SBATCH --job-name=cv-datasize
#SBATCH --output=/home-mscluster/nmahomed/cv-project/results/logs/rerun_datasize_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
# Nodes whose GPUs are dead. SLURM does no GPU accounting on this partition
# (GRES=null), so the scheduler cannot see these and will happily assign them.
#SBATCH --exclude=mscluster44,mscluster50,mscluster65,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82
#
# The data-size curve, through the same pipeline as Table 1 (cache_logits.py ->
# sweep_cached.py -> merge_refine.py), so the curve's full-data point IS Table
# 1's rung 4 and every size also gets rung 5 (the prior):
#   for n in 50 150 400 800: train 20 models on n masks, sweep (tau, beta),
#   summarise to results/rung5_mrf_refine_n{n}.csv, then delete that size's
#   cached logits (~2.3 GB) since nothing else reads them.
# Then the fp16 stabiliser comparison (compare_stabilisers.py), ~2 h, which
# uses the same crop loader.
#
# Runs alongside slurm/rerun_unet.sbatch. Safe to resubmit: finished pairs and
# sizes are skipped, and outputs from another config are refused.
#
# Submit from the repo root:  sbatch slurm/rerun_datasize.sbatch
set -euo pipefail

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.0
export DATASETS=${DATASETS:-/datasets/nmahomed}
export WANDB_MODE=offline

cd ~/cv-project
set +u; source venv/bin/activate; set -u
echo "node: $(hostname)   commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo '+uncommitted')"

python3 - <<'PY'
import socket, sys, torch
if not torch.cuda.is_available():
    print(f"NO GPU on {socket.gethostname()} - add to the exclude list", flush=True)
    sys.exit(1)
print(f"GPU ok on {socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
PY

# First launch only: move any previous data-size outputs aside. Nothing is deleted.
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

for n in 50 150 400 800; do
  if [ -e "results/rung5_mrf_refine_n$n.csv" ]; then echo "n=$n: done already"; continue; fi
  echo "=== n_train = $n ==="
  NESTED=1 BATCH=8 WORKERS=2 N_TRAIN=$n python3 src/methods/cache_logits.py     # GPU
  SUFFIX=_n$n CUT_WORKERS=12 python3 src/methods/sweep_cached.py                # CPU
  SUFFIX=_n$n python3 src/methods/merge_refine.py
  rm -f results/logits/t?v?_n$n.npz results/logits/t?v?_n$n.json
done

ONLY=baseline,clip,wd1e-3 python3 src/methods/compare_stabilisers.py

rm -f results/.rerun_datasize_started
echo "data-size rerun complete."
