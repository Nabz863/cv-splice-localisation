#!/bin/bash
#SBATCH --job-name=cv-sweep
#SBATCH --output=results/logs/sweep_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#
# Step 3 of rerun_unet.sbatch on its own: the (tau, beta) sweep over cached logits. CPU only.
# Submit from the repo root:   mkdir -p results/logs && sbatch slurm/sweep_cached.sbatch

cd "${SLURM_SUBMIT_DIR:-.}"
source venv/bin/activate

echo "node: $(hostname)"
python3 -c "import numpy, maxflow; print('deps ok')" || exit 1     # check dependencies first

CUT_WORKERS=12 python3 src/methods/sweep_cached.py
