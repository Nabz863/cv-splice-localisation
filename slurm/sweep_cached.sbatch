#!/bin/bash
#SBATCH --job-name=cv-sweep
#SBATCH --output=/home-mscluster/nmahomed/cv-project/results/sweep_%j.log
#SBATCH --partition=bigbatch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
# No GPU needed -- this phase imports no torch at all, so dead-GPU nodes are fine.

export DATASETS=/datasets/nmahomed
cd ~/cv-project
source venv/bin/activate

echo "node: $(hostname)"
python3 -c "import numpy, maxflow; print('deps ok')" || exit 1

CUT_WORKERS=12 python3 src/methods/sweep_cached.py
