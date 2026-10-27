#!/bin/bash
# Retrain every rung 4/5 U-Net and rebuild every table and figure (~26 h on the cluster).
# Run ONCE from the repo root on the login node:  DATASETS=/path/to/datasets bash scripts/retrain_unet.sh
# The login node only runs git, grep and sbatch; all computation happens in the submitted jobs.
# If a GPU job dies, resubmit that job directly (it resumes); FORCE=1 starts everything over.
set -euo pipefail
cd "$(dirname "$0")/.."                                    # repo root
: "${DATASETS:?set DATASETS to the folder that contains casia2/}"   # stop if DATASETS is not set
mkdir -p results/logs

# ---- Refuse to run twice by accident ----
MARK=results/.retrain_resize_submitted                      # written at the end of this script
if [ -e "$MARK" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "Already submitted ($(cat "$MARK")). To resume a failed job, sbatch it directly."
  echo "To discard the new outputs and start over: FORCE=1 bash $0"
  exit 1
fi

# ---- Record exactly which code runs (HEAD, plus any uncommitted changes as a patch) ----
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP=results/logs/retrain_code_$STAMP.patch
{
  git diff HEAD --binary                                    # changes to tracked files
  git ls-files --others --exclude-standard -- src scripts slurm \
    | while read -r f; do git diff --no-index --binary /dev/null "$f" || true; done   # new files
} > "$SNAP"
if [ -s "$SNAP" ]; then
  echo "code: $(git rev-parse --short HEAD) + uncommitted changes, saved to $SNAP"
else
  rm "$SNAP"; SNAP=""                                       # nothing uncommitted
  echo "code: $(git rev-parse --short HEAD), no uncommitted changes"
fi

# ---- Cheap check: models from the old decoder must be refused, not reused ----
grep -q '"upsample"' src/methods/refine_meta.py \
  || { echo "refine_meta.py does not record the decoder - old models would be reused"; exit 1; }

# ---- Fresh start: the GPU jobs move every old output to results/stale/ when these markers are absent ----
for m in results/.rerun_unet_started results/.rerun_datasize_started; do
  if [ -e "$m" ]; then rm "$m"; echo "cleared stale marker $m"; fi
done

# ---- Submit. Job scripts are passed on stdin and run with bash (sbatch --wrap uses sh, which has no "source") ----
cpu_job() {   # usage: cpu_job NAME TIME CPUS DEPENDENCY COMMANDS
  sbatch --parsable --job-name="$1" --partition=bigbatch --time="$2" --cpus-per-task="$3" \
         ${4:+--dependency=$4 --kill-on-invalid-dep=yes} \
         --output="$PWD/results/logs/${1}_%j.log" <<EOF
#!/bin/bash
set -euo pipefail
cd "$PWD"
set +u; source venv/bin/activate; set -u
export DATASETS="$DATASETS" OMP_NUM_THREADS=$3
$5
EOF
}
J0=$(cpu_job cv-checks 0:20:00 2 "" \
       "python3 src/methods/test_unet.py && python3 src/methods/test_mrf.py")   # unit tests gate everything
J1=$(sbatch --parsable --dependency=afterok:$J0 --kill-on-invalid-dep=yes slurm/rerun_unet.sbatch)       # 20 full-data models
J2=$(sbatch --parsable --dependency=afterok:$J0 --kill-on-invalid-dep=yes slurm/rerun_datasize.sbatch)   # 80 data-size models + stabilisers
J3=$(cpu_job cv-tables 6:00:00 12 "afterok:$J1:$J2" "bash scripts/rerun_cpu.sh")   # tests, rung 1-2 reruns, tables, figures

echo "submitted $(date '+%a %d %b %H:%M'): checks=$J0 unet=$J1 datasize=$J2 tables=$J3" \
     "code=$(git rev-parse --short HEAD)${SNAP:+ + $SNAP}" | tee "$MARK"
echo "Watch with: squeue -u \$USER    Logs: results/logs/"
echo "Leave src/, scripts/ and slurm/ untouched until the tables job has finished."
