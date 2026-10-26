# Source this before running anything:   DATASETS=/your/data/folder source setup_env.sh
# DATASETS must be the folder that contains casia2/ (see README.txt, "Data").
: "${DATASETS:?set DATASETS to the folder that contains casia2/}"
export DATASETS
