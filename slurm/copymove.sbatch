#!/bin/bash
#SBATCH --job-name=cv-copymove
#SBATCH --output=results/logs/copymove_%j.log
#SBATCH --time=04:00:00
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#
# Copy-move transfer test (Sec. VI), CPU only.
# Submit from the repo root:   mkdir -p results/logs && DATASETS=/path/to/data sbatch slurm/copymove.sbatch

: "${DATASETS:?set DATASETS to the folder that contains casia2/}"
cd "${SLURM_SUBMIT_DIR:-.}"
source venv/bin/activate 2>/dev/null || true
WORKERS=12 python3 src/methods/copymove_matched.py
