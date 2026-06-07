#!/bin/bash
# =============================================================================
# Resubmit: eval_temporal_full — whisper-base delta-t (full dataset)
# =============================================================================
# Previous run (54913812) was killed after ~8.5 h (3/6 layers done).
# The output HDF5 was corrupt because the file was held open at kill time.
#
# Fixes applied in evaluate_features_temporal.py:
#   - HDF5 opened in append mode ("a") — survives mid-run kills
#   - Summary JSON written after each dataset — no data lost on kill
#   - Already-computed keys skipped on resume
#
# Usage (from anywhere):
#   bash /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/scripts/resubmit_eval_temporal_full.sh
# =============================================================================

set -euo pipefail

REPO="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
OUTPUT_DIR="$REPO/outputs/results/whisper-base-delta-t-full"
SCORES_H5="$OUTPUT_DIR/temporal_scores.h5"

cd "$REPO"

# Delete the corrupt HDF5 from the killed run so we start clean.
# (h5py cannot open a file with a bad object header in append mode.)
if [ -f "$SCORES_H5" ]; then
    echo "Removing corrupt HDF5: $SCORES_H5"
    rm "$SCORES_H5"
fi

JOB=$(sbatch --parsable \
    --job-name=eval_temporal_full \
    --time=48:00:00 \
    --mem=48G \
    --cpus-per-task=8 \
    --output="$REPO/logs/eval_temporal_full_%j.out" \
    --error="$REPO/logs/eval_temporal_full_%j.err" \
    --wrap="source $REPO/env.sh && python -m mbs.evaluation.evaluate_features_temporal \
      --model_id whisper-base \
      --target_feature_layers $REPO/configs/extraction/audio/whisper_base_layers.json \
      --features_dir $REPO/outputs/features/whisper-base-delta-t/merged \
      --data_hdf5_path $REPO/outputs/neural_data/broderick2018_30s.h5 \
      --output_dir $OUTPUT_DIR")

echo "Submitted job $JOB: eval_temporal_full (48 h wall time)"
echo "Logs: $REPO/logs/eval_temporal_full_${JOB}.out"
