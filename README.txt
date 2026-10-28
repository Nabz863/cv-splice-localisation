LOCALISING SPLICED REGIONS IN IMAGES: A CONTROLLED COMPARISON
COMS4036A/COMS7050A Computer Vision, Wits
Nabeel Mahomed (2671907), Jonathan Lewis (2660425)

Pixel-level splice localisation on CASIA v2.0 (1,822 splicing image/mask pairs),
comparing five methods ("rungs") under one protocol:

  Rung 1  Error Level Analysis (ELA), thresholded      src/methods/noise_variants.py (v_ela)
  Rung 2  ELA + Potts MRF, exact graph cut             src/methods/mrf.py
  Rung 3  SAM2.1, zero-shot                            src/methods/run_sam2.py, run_sam2_auc.py
  Rung 4  U-Net with a constrained first layer         src/methods/unet.py, cache_logits.py
  Rung 5  U-Net + the same MRF                         src/methods/sweep_cached.py

At beta = 0, rung 2 is exactly rung 1 and rung 5 is exactly rung 4 (checked by
src/methods/test_mrf.py), so the difference between them is the prior's effect.

The U-Net's decoder upsamples by nearest-neighbour resize + 3x3 conv (src/methods/unet.py).
An earlier version used 2x2 transposed convolutions; its results are at commit 4b22355.

Every number in the report is computed by src/report/build_tables.py from files in
results/, and every figure is drawn by src/report/make_figures.py from results/tables/.
The committed rung 4/5 results were produced by the code in commit f01abdb (rungs 1-3
are unchanged since 4b22355); later commits change only comments, documentation, default
file paths and the job-submission script, not any computation.


======================================================================================
1. SETUP
======================================================================================

Requirements: Linux, Python 3.10+, an NVIDIA GPU for rungs 3-5 (we used one RTX 3090
per job for the U-Nets and a Colab A100 for SAM2). Rungs 1-2 and all analysis are CPU only.

All commands are run from the repository root.

    git clone https://github.com/Nabz863/cv-splice-localisation.git
    cd cv-splice-localisation
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt         # install a CUDA build of torch for GPU steps
    mkdir -p results/logs

Rung 3 only (SAM2):

    pip install git+https://github.com/facebookresearch/sam2.git
    mkdir -p checkpoints
    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

No path in the code is absolute. Data is found through one environment variable:

    export DATASETS=/path/to/your/data      # the folder that CONTAINS casia2/


======================================================================================
2. DATA
======================================================================================

We use CASIA v2.0 (Dong et al., 2013) with the ground-truth masks of Pham et al. (2019).
Both are available from https://github.com/namtpham/casia2groundtruth (masks in the repo,
a link to the images in its README). We found no licence for either; they are used for
non-commercial research only and are not included in this repository.

Expected layout:

    $DATASETS/casia2/images/CASIA2.0_revised/Tp/     tampered images (Tp_D_* splicing, Tp_S_* copy-move)
    $DATASETS/casia2/images/CASIA2.0_revised/Au/     7,491 authentic images
    $DATASETS/casia2/masks/CASIA2.0_Groundtruth/     masks, named <image name>_gt.png

Build the manifests and folds (seconds):

    python src/data/build_manifest.py           # -> $DATASETS/casia2/splice_manifest.csv   (1,822 pairs; 6 excluded for size mismatch)
    python src/data/build_manifest.py Tp_S_     # -> $DATASETS/casia2/copymove_manifest.csv (3,284 pairs)
    python src/data/stratify_folds.py           # adds 5 folds, stratified by host-image category


======================================================================================
3. QUICK CHECK: REBUILD THE TABLES AND FIGURES FROM THE COMMITTED RESULTS (~1 minute, CPU)
======================================================================================

All intermediate results are committed in results/ (except trained models and cached
logits, which are large). With the data set up as in section 2:

    python src/report/build_tables.py      # -> results/tables/*.csv  (every number in the report)
    python src/report/make_figures.py      # -> results/figures/*     (every figure except Fig. 5)

build_tables.py cross-checks files that should agree and stops if they don't.
Note: results/figures/ file names are not the report's figure numbers; section 6 maps them.

Fig. 5 (results/fig_qualitative_splice.pdf) is the exception: fig_qualitative.py draws it
from pair t0v1's cached U-Net logits, which are too large for GitHub (115 MB). The trained
t0v1 model is committed (results/models/refine_t0v1.pt, 8 MB), so regenerate its logits
without retraining (minutes; a CPU works, though its logits can differ from a GPU's in the
last digits), then redraw Fig. 5; the SAM2 column uses the committed results/sam2_examples/:
    ALLOW_CPU=1 BATCH=8 WORKERS=2 python -c "import sys; sys.path.insert(0, 'src/methods'); import cache_logits as c; c.cache_pair(0, 1, c.load_rows())"
    TAG=t0v1 python src/methods/fig_qualitative.py
Step E (slurm/rerun_unet.sbatch) also regenerates everything, Fig. 5 included.


======================================================================================
4. FULL REPRODUCTION, IN ORDER
======================================================================================

Each step writes into results/; later steps read earlier outputs. GPU times are for
one RTX 3090.

Step A. Tests (seconds)
    python src/eval/test_metrics.py
    python src/methods/test_mrf.py
    python src/methods/test_unet.py
    python src/report/test_overlap.py
    python src/methods/sanity_auc.py

Step B. Rung 1: choose the cue and its settings (CPU, minutes per run)
    for k in 0 1 2 3 4; do                          # repeat with each test fold held out
      HOLDOUT=$k python src/methods/compare_variants.py   # -> results/cue_auc*.csv   (ELA wins)
      HOLDOUT=$k python src/methods/tune_ela.py           # -> results/ela_tuning*.csv (q=90, w=32)
    done
    python src/methods/diagnose_texture.py                # -> results/texture_corr.csv

Step C. Rungs 1 and 2 (CPU)
    python src/methods/run_mrf.py              # -> results/rung2_grid.npz (every image, every (tau, beta))
    python src/methods/nested_pairs.py         # -> results/rung2_nested_pairs.csv (20 pairs)
    python src/methods/icm_vs_graphcut.py      # -> results/rung2_grid_icm.npz (greedy solver comparison)

    (scripts/rerun_cpu.sh runs steps A, B and icm_vs_graphcut.py, then section 3.)

Step D. Rung 3: SAM2 (GPU; we ran it on Colab)
    SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt SAM2_CACHE=results/sam2_cache \
        python src/methods/run_sam2.py         # SAM2 on all 1,822 images; resumable; cache in SAM2_CACHE
    SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_base_plus.pt SAM2_CACHE=results/sam2_cache \
        python src/methods/run_sam2_auc.py     # 20-pair evaluation + authentic images
                                               # -> results/rung3_sam2_20pairs.csv, image_auc_sam2_*.csv

Step E. Rungs 4 and 5 (GPU; two jobs, run in parallel, ~20 h each)
    On a Slurm cluster, one command runs the tests, both GPU jobs and then
    scripts/rerun_cpu.sh, each as its own job:
        DATASETS=$DATASETS bash scripts/retrain_unet.sh
    or submit the two GPU jobs by hand:
        DATASETS=$DATASETS sbatch slurm/rerun_unet.sbatch       # 20 full-data U-Nets, rung 5 sweep,
                                                                # image-level AUC, mechanism, Fig. 5
        DATASETS=$DATASETS sbatch slurm/rerun_datasize.sbatch   # 80 U-Nets at 50/150/400/800 masks,
                                                                # then the fp16 stabiliser runs (Fig. 3)
    Without Slurm, the same scripts run as plain bash from the repo root:
        bash slurm/rerun_unet.sbatch
        bash slurm/rerun_datasize.sbatch
    (The #SBATCH partition and node-exclude lines are Wits MS cluster settings; change
    them for another cluster. Optional: python src/methods/overfit_batch.py first,
    a one-batch wiring check.)

Step E2. The earlier transposed-convolution U-Net (optional)
    Its results are committed at 4b22355 (git checkout 4b22355 shows them; Table I there
    has U-Net 0.6943 and U-Net + MRF 0.7356). To retrain it with the current code, in a
    SEPARATE clone (the GPU jobs move the current rung 4/5 outputs to results/stale/):
        UPSAMPLE=transpose DATASETS=$DATASETS bash scripts/retrain_unet.sh
    sbatch passes UPSAMPLE on to cache_logits.py, which builds the 2x2 transposed-conv
    decoder and records it in each model's sidecar; every later step reads the decoder
    from there. Two things always use the default (resize) decoder: the fp16 stabiliser
    runs (Fig. 3) and the drawing of Fig. 2. The architecture and seeds match 4b22355, so
    this should reproduce those numbers, but we have not rerun it to check.

Step F. Further analyses at the fixed operating point tau = 0.75, beta = 8 (CPU)
    python src/methods/per_category.py         # -> results/per_category.csv     (Sec. V-F)
    python src/methods/copymove_matched.py     # -> results/copymove_*           (Sec. VI)
    python src/methods/image_auc_classical.py  # -> results/image_auc_classical.csv (Table II, rungs 1-2)
    Optional, GPU: python src/methods/sam2_examples.py, then
                   python src/methods/fig_qualitative.py    (adds the SAM2 column to Fig. 5)

Step G. Tables and figures (CPU, ~1 min): section 3.


======================================================================================
5. SEEDS AND DETERMINISM
======================================================================================

  Folds                  stratify_folds.py, seed 0
  U-Net, pair (k, v)     set_seed(10k + v) in cache_logits.py (Python, NumPy, PyTorch,
                         CUDA; cuDNN deterministic, benchmark off)
  Training crops         each data-loader worker has its own generator, seeded from
                         (k, torch's per-worker seed) in patch_data.py
  Data-size subsets      NumPy default_rng(0)
  Validation images      150 per pair, default_rng(1)
  Stabiliser runs        set_seed(0) for every candidate
  Rung 1 image samples   default_rng(0); pixel-AUC pixel samples, seed 0
  Bootstrap intervals    default_rng(0), 2,000 resamples

Every trained model and logits file has a JSON file next to it recording its training
configuration (src/methods/refine_meta.py); a file made under a different configuration
is refused, never silently reused; this includes the decoder type. With the earlier
transposed-convolution decoder, two independent retrains of all 100 U-Nets reproduced
every stopping epoch, validation loss and rung 4/5 number exactly; we have not repeated
this for the current decoder.


======================================================================================
6. WHERE EACH REPORT NUMBER COMES FROM
======================================================================================

  Table I (main results)        results/tables/main.csv
  Table II (image-level AUC)    results/tables/auc.csv
  Report figure                 drawn from                                 file
  Fig. 1  method overview       make_figures.py fig_method                  results/figures/fig0_method.pdf
  Fig. 2  U-Net                 make_figures.py fig_unet (sizes: unet.py)   results/figures/fig10_unet.pdf
  Fig. 3  training stability    results/tables/stabiliser.csv (wandb/)      results/figures/fig7_stabiliser.pdf
  Fig. 4  prior gain per pair   results/tables/gains.csv                    results/figures/fig2_prior_gain.pdf
  Fig. 5  qualitative examples  src/methods/fig_qualitative.py (logits)     results/fig_qualitative_splice.pdf
  Fig. 6  mechanism             results/tables/mechanism*.csv               results/figures/fig8_mechanism.pdf
  Fig. 7  selection criterion   results/tables/selection*.csv               results/figures/fig12_selection.pdf
  Fig. 8  data-size curve       results/tables/datasize.csv                 results/figures/fig3_datasize.pdf
  Fig. 9  SAM2                  results/tables/sam2.csv                     results/figures/fig11_sam2.pdf
  (fig1_main_f1, fig4_per_category, fig5_image_auc, fig6_copymove and fig9_perimage_f1 are
  drawn too but not used in the report; Tables I and II carry those numbers.)
  Sec. V-F per category         results/tables/category.csv
  Sec. VI copy-move             results/tables/copymove.csv
  Rung 1 choice, ICM            results/tables/cues.csv, icm.csv, icm_energy.csv

build_tables.py's main() has one labelled section per table, naming its source files.


======================================================================================
7. REPOSITORY LAYOUT
======================================================================================

  src/data/      manifest and folds
  src/eval/      metrics (F1, IoU, AUC, bootstrap) shared by every method
  src/methods/   the five rungs, their sweeps, and the analyses
  src/report/    build_tables.py, make_figures.py, figure overlap checker
  slurm/         cluster jobs for the GPU steps
  scripts/       retrain_unet.sh (submits every job), rerun_cpu.sh (CPU steps + tables + figures)
  results/       every intermediate result, tables and figures (models/logits not committed)
  wandb/         offline training logs (stabiliser runs)
