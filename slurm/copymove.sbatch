#!/bin/bash
#SBATCH --job-name=cv-copymove
#SBATCH --output=results/copymove_%j.log
#SBATCH --time=04:00:00
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

export DATASETS=/datasets/nmahomed
cd ~/cv-project
source venv/bin/activate 2>/dev/null || true
WORKERS=12 python3 src/methods/run_copymove.py
