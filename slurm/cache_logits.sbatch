#!/bin/bash
#SBATCH --job-name=cv-cache
#SBATCH --output=/home-mscluster/nmahomed/cv-project/results/cache_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
# Nodes whose GPUs are dead. SLURM does no GPU accounting on this partition
# (GRES=null), so the scheduler cannot see these and will happily assign them.
#SBATCH --exclude=mscluster44,mscluster50,mscluster65,mscluster46,mscluster47,mscluster48,mscluster51,mscluster63,mscluster64,mscluster69,mscluster70,mscluster74,mscluster75,mscluster76,mscluster82

export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.0
export DATASETS=/datasets/nmahomed
export WANDB_MODE=offline

cd ~/cv-project
source venv/bin/activate

echo "node: $(hostname)"
nvidia-smi || echo "nvidia-smi failed on $(hostname)"

# Fail fast and loudly rather than running at CPU speed for hours.
python3 - <<'PY' || exit 1
import socket, sys, torch
if not torch.cuda.is_available():
    print(f"NO GPU on {socket.gethostname()} - add to the exclude list", flush=True)
    sys.exit(1)
print(f"GPU ok on {socket.gethostname()}: {torch.cuda.get_device_name(0)}", flush=True)
PY

NESTED=1 WORKERS=4 BATCH=16 python3 src/methods/cache_logits.py
