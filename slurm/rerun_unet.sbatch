#!/bin/bash
#SBATCH --job-name=cv-rerun
#SBATCH --output=/home-mscluster/nmahomed/cv-project/results/logs/rerun_unet_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
# Nodes whose GPUs are dead. SLURM does no GPU accounting on this partition
# (GRES=null), so the scheduler cannot see these and will happily assign them.
#SBATCH --exclude=mscluster44,mscluster50,mscluster65,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82
#
# Regenerates every rung 4/5 result from scratch under ONE training config, the
# one run_unet.py used for the data-size curve (batch 8, 2 loader workers):
#   train 20 models + cache logits   cache_logits.py          GPU, ~15-20 h
#   (tau, beta) sweep, Table 1 rows  sweep_cached.py, merge_refine.py   CPU
#   image-level AUC, all 20 pairs    cache_auth_logits.py (GPU), image_auc_unet.py,
#                                    select_summary_unet.py (CPU)
#   where the prior's gain falls     mechanism.py over all 20 pairs      CPU
#   qualitative figure               fig_qualitative.py                  CPU
# GPU and CPU steps run as separate python processes (see cache_logits.py).
#
# Safe to resubmit if it dies: finished pairs are skipped, a half-trained pair
# resumes from its checkpoint, and anything produced under another config is
# refused rather than reused (src/methods/refine_meta.py).
#
# Submit from the repo root:  sbatch slurm/rerun_unet.sbatch
set -euo pipefail

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.0
export DATASETS=${DATASETS:-/datasets/nmahomed}
export WANDB_MODE=offline

cd ~/cv-project
set +u; source venv/bin/activate; set -u
echo "node: $(hostname)   commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo '+uncommitted')"
nvidia-smi || echo "nvidia-smi failed on $(hostname)"

python3 - <<'PY'
import socket, sys, torch
if not torch.cuda.is_available():
    print(f"NO GPU on {socket.gethostname()} - add to the exclude list", flush=True)
    sys.exit(1)
print(f"GPU ok on {socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
PY

TAGS=""
for k in 0 1 2 3 4; do for v in 0 1 2 3 4; do
  [ "$k" != "$v" ] && TAGS="$TAGS t${k}v${v}"
done; done

# 1. First launch only: move every output of the OLD rung 4/5 models aside. They
#    were trained under two different configs and have no record of which; the
#    new code refuses them anyway. Nothing is deleted.
if [ ! -e results/.rerun_unet_started ]; then
  STALE=results/stale/$(date +%Y%m%d_%H%M%S)
  mkdir -p "$STALE"
  shopt -s nullglob
  for f in results/models/refine_* results/logits/* results/ckpt_refine/* \
           results/rung5_pair_*.csv results/rung5_mrf_refine.csv results/rung5_training.csv \
           results/image_auc_unet_t*.csv results/image_auc_select_t*.csv; do
    mv "$f" "$STALE/"
  done
  shopt -u nullglob
  echo "old rung 4/5 outputs moved to $STALE"
  touch results/.rerun_unet_started
fi

# 2. Train each pair's model and cache its validation and test logits (GPU).
NESTED=1 BATCH=8 WORKERS=2 python3 src/methods/cache_logits.py

# 3. (tau, beta) sweep over the cached logits and the 20-pair summary (CPU).
CUT_WORKERS=12 python3 src/methods/sweep_cached.py
python3 src/methods/merge_refine.py

# 4. Image-level AUC for every pair. The authentic-image logits are large
#    (~0.4 GB per pair), so each pair's are deleted once its AUC is written;
#    set KEEP_AUTH=1 to keep them.
for t in $TAGS; do
  if [ -e "results/image_auc_select_$t.csv" ]; then echo "$t: AUC done already"; continue; fi
  TAG=$t N_AUTH=2000 python3 src/methods/cache_auth_logits.py          # GPU
  TAG=$t WORKERS=12 python3 src/methods/image_auc_unet.py              # CPU
  TAG=$t WORKERS=12 python3 src/methods/select_summary_unet.py         # CPU
  [ "${KEEP_AUTH:-0}" = 1 ] || rm -f "results/logits/auth_$t.npz"
done

# 5. Where the prior's corrections fall, over all 20 pairs; qualitative figure.
TAGS=$(echo $TAGS | tr ' ' ',') WORKERS=12 python3 src/methods/mechanism.py
TAG=t0v1 python3 src/methods/fig_qualitative.py

rm -f results/.rerun_unet_started
echo "rung 4/5 rerun complete. Next: bash scripts/rerun_cpu.sh, then build_tables.py"
