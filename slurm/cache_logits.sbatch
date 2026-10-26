#!/bin/bash
#SBATCH --job-name=cv-cache
#SBATCH --output=results/logs/cache_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
# Wits MS cluster: nodes with dead GPUs (Slurm can't see this, so exclude them by name).
#SBATCH --exclude=mscluster44,mscluster50,mscluster65,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82,mscluster62,mscluster83
#
# Step 2 of rerun_unet.sbatch on its own: train the 20 U-Nets and cache their logits.
# Submit from the repo root:   mkdir -p results/logs && DATASETS=/path/to/data sbatch slurm/cache_logits.sbatch

if [ -d /usr/local/cuda-12.0 ]; then                       # Wits cluster's CUDA install, if present
  export PATH=/usr/local/cuda-12.0/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
  export CUDA_HOME=/usr/local/cuda-12.0
fi
: "${DATASETS:?set DATASETS to the folder that contains casia2/}"
export WANDB_MODE=offline

cd "${SLURM_SUBMIT_DIR:-.}"
source venv/bin/activate

echo "node: $(hostname)"
nvidia-smi || echo "nvidia-smi failed on $(hostname)"

# stop early if this node has no working GPU
python3 - <<'PY' || exit 1
import socket, sys, torch
if not torch.cuda.is_available():
    print(f"NO GPU on {socket.gethostname()} - add to the exclude list", flush=True)
    sys.exit(1)
print(f"GPU ok on {socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
PY

NESTED=1 WORKERS=2 BATCH=8 python3 src/methods/cache_logits.py
