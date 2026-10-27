#!/bin/bash
# Retrain every rung 4/5 U-Net with the resize-then-convolve decoder (unet.py)
# and rebuild every table and figure from the new models.
#
# Run ONCE, on the cluster login node, from the repo root:
#     bash scripts/retrain_unet.sh
#
# What it does:
#   0. records the code being run: HEAD's hash, plus any uncommitted changes
#      (tracked and new files) as results/logs/retrain_code_<time>.patch, which
#      applies to that HEAD. Commit the same changes later and the results trace
#      to the commit; until then they trace to HEAD + the patch.
#   1. CPU checks: test_unet.py (decoder, sidecars, overfit) and test_mrf.py
#   2. clears the start markers of any earlier, unfinished rerun, so that both
#      GPU jobs begin by moving EVERY old-decoder output to results/stale/
#      (nothing is deleted; refine_meta.py would refuse those files anyway)
#   3. submits, in parallel:
#        slurm/rerun_unet.sbatch      20 full-data models, rungs 4/5, per-image
#                                     sweep, image-level AUC, mechanism,
#                                     qualitative figure                ~20-24 h
#        slurm/rerun_datasize.sbatch  80 models at n = 50/150/400/800, then the
#                                     fp16 stabiliser comparison        ~20-24 h
#      and a CPU job that waits for both to succeed, then runs
#      scripts/rerun_cpu.sh: tests, rung 1-2 reruns, build_tables.py and
#      make_figures.py                                                  ~1 h
#
# If a GPU job dies, resubmit THAT job directly (sbatch slurm/rerun_unet.sbatch);
# it resumes. Do not rerun this script: it would move the new, partly finished
# outputs aside and start again. FORCE=1 does exactly that, deliberately.
set -euo pipefail
cd "$(dirname "$0")/.."
export DATASETS=${DATASETS:-/datasets/nmahomed}
mkdir -p results/logs

MARK=results/.retrain_resize_submitted
if [ -e "$MARK" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "Already submitted ($(cat "$MARK")). To resume a failed job, sbatch it directly."
  echo "To discard the new outputs and start over: FORCE=1 bash $0"
  exit 1
fi

# 0. Record exactly which code runs. The jobs read the working tree, so do not
#    edit tracked code until all three jobs have finished.
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP=results/logs/retrain_code_$STAMP.patch
{
  git diff HEAD --binary
  git ls-files --others --exclude-standard -- src scripts slurm \
    | while read -r f; do git diff --no-index --binary /dev/null "$f" || true; done
} > "$SNAP"
if [ -s "$SNAP" ]; then
  echo "code: $(git rev-parse --short HEAD) + uncommitted changes, saved to $SNAP:"
  git status --short -- src scripts slurm
else
  rm "$SNAP"; SNAP=""
  echo "code: $(git rev-parse --short HEAD), no uncommitted changes"
fi

# 1. CPU checks before spending ~24 GPU-hours.
set +u; source venv/bin/activate; set -u
python3 src/methods/test_unet.py
python3 src/methods/test_mrf.py
grep -q '"upsample"' src/methods/refine_meta.py \
  || { echo "refine_meta.py does not record the decoder - old models would be reused"; exit 1; }

# 2. Fresh start for both jobs.
for m in results/.rerun_unet_started results/.rerun_datasize_started; do
  if [ -e "$m" ]; then rm "$m"; echo "cleared stale marker $m"; fi
done

# 3. Submit.
J1=$(sbatch --parsable slurm/rerun_unet.sbatch)
J2=$(sbatch --parsable slurm/rerun_datasize.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J1:$J2 --kill-on-invalid-dep=yes \
       --job-name=cv-tables --partition=bigbatch --time=6:00:00 --cpus-per-task=12 \
       --output="$PWD/results/logs/retrain_tables_%j.log" \
       --wrap "cd '$PWD' && set +u && source venv/bin/activate && set -u && \
               DATASETS='$DATASETS' bash scripts/rerun_cpu.sh")

echo "submitted $(date '+%a %d %b %H:%M'): unet=$J1 datasize=$J2 tables=$J3" \
     "code=$(git rev-parse --short HEAD)${SNAP:+ + $SNAP}" | tee "$MARK"
cat <<EOF

Watch:   squeue -u \$USER
Logs:    results/logs/rerun_unet_$J1.log
         results/logs/rerun_datasize_$J2.log
         results/logs/retrain_tables_$J3.log   (starts when both GPU jobs succeed)
Done when the tables log ends with make_figures.py's output. Until then, leave
src/, scripts/ and slurm/ untouched: later steps read the code when they start.
Then commit the code and results/ together; the commit will match ${SNAP:-HEAD}.
EOF
