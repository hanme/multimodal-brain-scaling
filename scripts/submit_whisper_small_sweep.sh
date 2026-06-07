#!/bin/bash
# =============================================================================
# Submit window/stride sweep for whisper-small × Broderick 2018
# =============================================================================
#
# Tests 3 stride options (fixed 30 s Whisper window) to find the best
# sample-size / stimulus-overlap trade-off for whisper-small (d=768):
#
#   w30s30  30 s window, 30 s stride  ~  88 train stimuli  (no overlap)
#   w30s10  30 s window, 10 s stride  ~ 252 train stimuli  (current baseline)
#   w30s05  30 s window,  5 s stride  ~ 496 train stimuli  (most samples)
#
# Steps performed:
#   1. Create EEG HDF5 files for new strides (short SLURM jobs, ~5 min each).
#      w30s10 symlinks to the existing broderick2018_30s.h5.
#   2. Submit Delta-T feature extraction array jobs (~6 h wall time each).
#
# After all extraction jobs complete, evaluate with:
#   scripts/submit_eval_whisper_small_sweep.sh   (see bottom of this file)
#
# Run from the repo root with the venv active:
#   cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#   source env.sh
#   bash scripts/submit_whisper_small_sweep.sh
# =============================================================================

set -euo pipefail

REPO="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
BIDS_ROOT="/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea"
MODEL_ID="whisper-small"
CHUNK_SIZE=8    # stimuli per SLURM task (8 × ~45 min ≈ 6 h, within 12 h limit)
BATCH_T=4

cd "$REPO"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — EEG HDF5 files
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p outputs/neural_data

# w30s10: reuse existing file (same parameters)
if [ ! -e outputs/neural_data/broderick2018_w30s10.h5 ]; then
    ln -s broderick2018_30s.h5 outputs/neural_data/broderick2018_w30s10.h5
    echo "Symlinked outputs/neural_data/broderick2018_w30s10.h5 -> broderick2018_30s.h5"
else
    echo "outputs/neural_data/broderick2018_w30s10.h5 already exists, skipping."
fi

# w30s05 and w30s30: submit short SLURM jobs
for STRIDE in 5.0 30.0; do
    STRIDE_PAD=$(printf '%02d' "${STRIDE%.*}")
    OUT_H5="outputs/neural_data/broderick2018_w30s${STRIDE_PAD}.h5"
    if [ -e "$OUT_H5" ]; then
        echo "${OUT_H5} already exists, skipping."
        continue
    fi
    JOB=$(sbatch --parsable \
        --job-name="eeg_fmt_w30s${STRIDE_PAD}" \
        --time=00:20:00 \
        --mem=24G \
        --output="$REPO/logs/eeg_fmt_w30s${STRIDE_PAD}_%j.out" \
        --error="$REPO/logs/eeg_fmt_w30s${STRIDE_PAD}_%j.err" \
        --wrap="source $REPO/env.sh && python -m mbs.data_prep.format_eeg_hdf5 \
          --bids_root $BIDS_ROOT \
          --output_path $REPO/$OUT_H5 \
          --window_duration 30.0 \
          --window_stride   $STRIDE \
          --target_sr       50 \
          --n_test_runs     4 \
          --seed            42")
    echo "Submitted EEG formatting job ${JOB}: broderick2018_w30s${STRIDE_PAD}.h5"
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Feature extraction array jobs
# Note: extraction does NOT depend on EEG formatting (EEG is only needed
# for evaluation). All 3 extraction jobs can run in parallel immediately.
# ─────────────────────────────────────────────────────────────────────────────

# Array sizes: over-allocate slightly; empty tasks (start >= n_dataset) exit cleanly.
#   w30s30: ~110 stimuli / 8 = 14 tasks  -> array 0-14  (15 tasks)
#   w30s10: ~314 stimuli / 8 = 40 tasks  -> array 0-39  (40 tasks)
#   w30s05: ~630 stimuli / 8 = 79 tasks  -> array 0-79  (80 tasks)

declare -A ARRAY_MAX=( ["5.0"]="79" ["10.0"]="39" ["30.0"]="14" )

for STRIDE in 5.0 10.0 30.0; do
    STRIDE_PAD=$(printf '%02d' "${STRIDE%.*}")
    N_MAX="${ARRAY_MAX[$STRIDE]}"
    OUT_BASE="outputs/features/${MODEL_ID}-w30s${STRIDE_PAD}-delta-t"

    if [ -d "$OUT_BASE" ]; then
        echo "WARNING: ${OUT_BASE}/ already exists — skipping submission to avoid overwrite."
        echo "  Delete it first if you want to re-run: rm -rf $OUT_BASE"
        continue
    fi

    JOB=$(MODEL_ID="$MODEL_ID" \
          WINDOW_DUR="30.0" \
          WINDOW_STRIDE="$STRIDE" \
          CHUNK_SIZE="$CHUNK_SIZE" \
          BATCH_T="$BATCH_T" \
          sbatch --parsable \
              --array="0-${N_MAX}" \
              --job-name="delta_t_${MODEL_ID}-w30s${STRIDE_PAD}" \
              --export="ALL,MODEL_ID=${MODEL_ID},WINDOW_DUR=30.0,WINDOW_STRIDE=${STRIDE},CHUNK_SIZE=${CHUNK_SIZE},BATCH_T=${BATCH_T}" \
              "$REPO/scripts/slurm_extract_delta_t_generic.sh")
    echo "Submitted extraction job ${JOB}: ${MODEL_ID} w30s${STRIDE_PAD}  (array 0-${N_MAX})"
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — After extraction completes: run evaluation
# ─────────────────────────────────────────────────────────────────────────────
cat <<'EOF'

─────────────────────────────────────────────────────────────────
After all extraction jobs finish, merge and evaluate each config:

for STRIDE_PAD in 05 10 30; do
  REPO=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
  FEAT_DIR=$REPO/outputs/features/whisper-small-w30s${STRIDE_PAD}-delta-t
  MERGED=$FEAT_DIR/merged

  mkdir -p $MERGED
  cp $FEAT_DIR/chunk_*/feats*.h5 $MERGED/

  sbatch --job-name=eval_ws_w30s${STRIDE_PAD} \
         --time=06:00:00 --mem=48G --cpus-per-task=8 \
         --output=$REPO/logs/eval_ws_w30s${STRIDE_PAD}_%j.out \
         --error=$REPO/logs/eval_ws_w30s${STRIDE_PAD}_%j.err \
         --wrap="source $REPO/env.sh && python -m mbs.evaluation.evaluate_features_temporal \
           --model_id whisper-small \
           --target_feature_layers $REPO/configs/extraction/audio/whisper_small_layers.json \
           --features_dir $MERGED \
           --data_hdf5_path $REPO/outputs/neural_data/broderick2018_w30s${STRIDE_PAD}.h5 \
           --output_dir $REPO/outputs/results/whisper-small-w30s${STRIDE_PAD}-delta-t/"
done

What to look for in results (temporal_scores_summary.json):
  - peak_score at Fz, T7, T8 (single electrodes, not clusters)
  - mean_score across time bins at those ROIs
  - The stride that gives the highest scores is the winner
  - Expect w30s05 (most stimuli) to beat w30s10 > w30s30 for larger models
  - If w30s30 ≈ w30s05, sample size is not the bottleneck
─────────────────────────────────────────────────────────────────
EOF
