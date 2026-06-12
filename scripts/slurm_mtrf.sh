#!/bin/bash
# =============================================================================
# SLURM Array Job: mTRF (lagged shared-weight Ridge) — Workstream A
# =============================================================================
#
# High-pass cutoff sweep: 6 layers x 3 cutoffs = 18 tasks.
# Each task runs evaluate_features_mtrf for ONE (layer, cutoff), ALL electrodes,
# single_lag mode, lags 0-800 ms @ 20 ms, scored with UNCORRECTED Pearson r
# (NC-on-high-passed-EEG is invalid; Kadir explicitly OK'd uncorrected r).
#
# Each task writes its OWN dir (concurrent writers to one HDF5 corrupt it):
#   outputs/results/whisper-base-mtrf-hp<cut>/layer_<layer>/
# where <cut> in {0p5, 1p0, 2p0} for cutoffs {0.5, 1.0, 2.0} Hz.
#
# Array map (--array=0-17):  cutoff_idx = task/6 ; layer = task%6
#   tasks  0..5  -> 0.5 Hz, layers 0..5
#   tasks  6..11 -> 1.0 Hz, layers 0..5
#   tasks 12..17 -> 2.0 Hz, layers 0..5
#
# Submit:
#   sbatch --array=0-17 scripts/slurm_mtrf.sh
#
# Inspect after:
#   python scripts/plot_mtrf_scores.py \
#       --nohp_dir outputs/results/whisper-base-mtrf-hp1p0/layer_2 \
#       --hp_dir   outputs/results/whisper-base-mtrf-hp1p0/layer_2   # etc.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=mtrf_hp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=04:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/mtrf_hp_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/mtrf_hp_%A_%a.err

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
LAYERS_CONFIG="configs/extraction/audio/whisper_base_layers.json"
FEATURES_DIR="outputs/features/whisper-base-delta-t/merged/"
NEURAL_H5="outputs/neural_data/broderick2018_30s.h5"
MODEL_ID="whisper-base"

LAG_MIN=0
LAG_MAX=800
LAG_STEP=20
N_TRAIN_SAMPLES=200
TEST_STRIDE=1

CUTOFFS=(0.5 1.0 2.0)

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
source env.sh
echo "Python: $(which python)"
mkdir -p logs

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CUT_IDX=$(( TASK_ID / 6 ))
LAYER_ID=$(( TASK_ID % 6 ))
HIGHPASS=${CUTOFFS[$CUT_IDX]}
HP_LABEL=$(echo "$HIGHPASS" | tr '.' 'p')
OUT_DIR="outputs/results/whisper-base-mtrf-hp${HP_LABEL}/layer_${LAYER_ID}"

echo "CUTOFF SWEEP  task=${TASK_ID}  layer=${LAYER_ID}  highpass=${HIGHPASS} Hz  -> ${OUT_DIR}"
mkdir -p "$OUT_DIR"

# ── Run (all electrodes: no --roi_allowlist; whole_brain excluded by default) ──
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python -m mbs.evaluation.evaluate_features_mtrf \
    --model_id               "$MODEL_ID" \
    --target_feature_layers  "$LAYERS_CONFIG" \
    --features_dir           "$FEATURES_DIR" \
    --data_hdf5_path         "$NEURAL_H5" \
    --output_dir             "$OUT_DIR" \
    --mode                   single_lag \
    --layer_id               "$LAYER_ID" \
    --lag_min_ms             "$LAG_MIN" \
    --lag_max_ms             "$LAG_MAX" \
    --lag_step_ms            "$LAG_STEP" \
    --highpass_hz            "$HIGHPASS" \
    --n_train_time_samples   "$N_TRAIN_SAMPLES" \
    --test_time_stride       "$TEST_STRIDE" \
    --nc_threshold           0.0 \
    --noise_ceiling_correct  false \
    --overwrite              true

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS  task=${TASK_ID}  layer=${LAYER_ID}  hp=${HIGHPASS}  -> ${OUT_DIR}"
    ls -lh "$OUT_DIR"
else
    echo "FAILED   task=${TASK_ID}  exit=${EXIT_CODE}"
fi
echo "End: $(date)"
exit $EXIT_CODE
