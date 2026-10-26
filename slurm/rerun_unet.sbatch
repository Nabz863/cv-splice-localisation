#!/bin/bash
#SBATCH --job-name=cv-rerun
#SBATCH --output=results/logs/rerun_unet_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
# Wits MS cluster: nodes with dead GPUs (Slurm can't see this, so exclude them by name).
#SBATCH --exclude=mscluster44,mscluster50,mscluster65,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82,mscluster62,mscluster83
#
# Rungs 4 and 5 on full data, all 20 (test, val) pairs, from scratch (~15-20 h on one RTX 3090):
#   train + cache logits (GPU) -> (tau, beta) sweep (CPU) -> image-level AUC -> mechanism -> Fig. 5
# Safe to resubmit: finished pairs are skipped and a half-trained pair resumes.
#
# Submit from the repo root:   mkdir -p results/logs && DATASETS=/path/to/data sbatch slurm/rerun_unet.sbatch
set -euo pipefail

# ---- Environment ----
if [ -d /usr/local/cuda-12.0 ]; then                       # Wits cluster's CUDA install, if present
  export PATH=/usr/local/cuda-12.0/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:${LD_LIBRARY_PATH:-}
  export CUDA_HOME=/usr/local/cuda-12.0
fi
: "${DATASETS:?set DATASETS to the folder that contains casia2/}"
export WANDB_MODE=offline                                   # log W&B runs locally, no internet needed

cd "${SLURM_SUBMIT_DIR:-.}"                                 # the repo root you submitted from
set +u; source venv/bin/activate; set -u
echo "node: $(hostname)   commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo '+uncommitted')"
nvidia-smi || echo "nvidia-smi failed on $(hostname)"

# ---- Stop early if this node has no working GPU ----
python3 - <<'PY'
import socket, sys, torch
if not torch.cuda.is_available():
    print(f"NO GPU on {socket.gethostname()} - add to the exclude list", flush=True)
    sys.exit(1)
print(f"GPU ok on {socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
PY

TAGS=""                                                     # t0v1 t0v2 ... t4v3 (20 pairs)
for k in 0 1 2 3 4; do for v in 0 1 2 3 4; do
  [ "$k" != "$v" ] && TAGS="$TAGS t${k}v${v}"
done; done

# ---- 1. First launch only: move old rung 4/5 outputs to results/stale/ (nothing is deleted) ----
if [ ! -e results/.rerun_unet_started ]; then
  STALE=results/stale/$(date +%Y%m%d_%H%M%S)
  mkdir -p "$STALE"
  shopt -s nullglob
  for f in results/models/refine_t?v?.* results/logits/t?v?.* results/logits/auth_* \
           results/ckpt_refine/t?v?.* results/rung5_pair_t?v?.csv \
           results/rung5_mrf_refine.csv results/rung5_training.csv results/rung5_counts/* \
           results/image_auc_unet_t?v?.csv results/image_auc_select_t?v?.csv; do
    if [ -e "$f" ]; then mv "$f" "$STALE/"; fi
  done
  shopt -u nullglob
  echo "old rung 4/5 outputs moved to $STALE"
  touch results/.rerun_unet_started
fi

# ---- 2. Train each pair's U-Net and cache its validation and test logits (GPU) ----
NESTED=1 BATCH=8 WORKERS=2 python3 src/methods/cache_logits.py

# ---- 3. Choose (tau, beta) on validation, score on test; summarise (CPU) ----
CUT_WORKERS=12 python3 src/methods/sweep_cached.py
python3 src/methods/merge_refine.py
CUT_WORKERS=12 python3 src/methods/sweep_perimage.py      # per-image counts, ~1.5 h

# ---- 4. Image-level AUC per pair (authentic-image logits are ~1.5 GB, deleted after use) ----
for t in $TAGS; do
  if [ -e "results/image_auc_select_$t.csv" ]; then echo "$t: AUC done already"; continue; fi
  TAG=$t N_AUTH=0 python3 src/methods/cache_auth_logits.py             # GPU
  TAG=$t WORKERS=12 python3 src/methods/image_auc_unet.py              # CPU
  TAG=$t WORKERS=12 python3 src/methods/select_summary_unet.py         # CPU
  [ "${KEEP_AUTH:-0}" = 1 ] || rm -f "results/logits/auth_$t.npz"
done

# ---- 5. Where the prior's gain comes from (Fig. 6), and the qualitative figure (Fig. 5) ----
TAGS=$(echo $TAGS | tr ' ' ',') WORKERS=12 python3 src/methods/mechanism.py
TAG=t0v1 python3 src/methods/fig_qualitative.py

rm -f results/.rerun_unet_started
echo "rung 4/5 rerun complete."
