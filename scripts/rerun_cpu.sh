#!/bin/bash
# CPU-only reruns that make every rung 1-2 number in the report come from a file:
#   rung 1 cue comparison      compare_variants.py  -> results/cue_auc*.csv
#   ELA quality/window sweep   tune_ela.py          -> results/ela_tuning.csv
#   noise cue vs texture       diagnose_texture.py  -> results/texture_corr.csv
#   ICM vs graph cuts          icm_vs_graphcut.py   -> results/rung2_grid_icm.npz
# then the unit tests, every table, and every figure. Roughly 20-40 minutes on
# 12 cores; nothing here needs a GPU, so it can run alongside rerun_unet.sbatch,
# but build_tables.py at the end needs that job's outputs.
#
# Run from the repo root:   DATASETS=/path/to/datasets bash scripts/rerun_cpu.sh
# Logs go to results/logs/rerun_cpu_<timestamp>.log as well as the terminal.
set -euo pipefail
: "${DATASETS:?set DATASETS to the folder that contains casia2/}"
mkdir -p results/logs
LOG=results/logs/rerun_cpu_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo '+uncommitted')"

python3 src/eval/test_metrics.py
python3 src/methods/test_mrf.py
python3 src/report/test_overlap.py

python3 src/methods/sanity_auc.py
python3 src/methods/compare_variants.py
python3 src/methods/tune_ela.py
python3 src/methods/diagnose_texture.py
python3 src/methods/icm_vs_graphcut.py

if [ -e results/.rerun_unet_started ] || [ ! -e results/rung5_training.csv ]; then
  echo "rerun_unet.sbatch has not finished - skipping build_tables.py and make_figures.py"
  exit 0
fi
python3 src/report/build_tables.py
python3 src/report/make_figures.py
