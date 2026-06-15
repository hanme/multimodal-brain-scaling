#!/bin/bash
# =============================================================================
# Cross-dataset mTRF transfer — train one dataset, test the other (out-of-domain), ANY whisper size.
# CPU only. Plan §13 follow-up. Array: 0 = D1->D2, 1 = D2->D1.
#
#   sbatch --array=0-1 scripts/slurm_cross_mtrf.sh                      # whisper-base (default)
#   MODEL_ID=whisper-small sbatch --array=0-1 --export=ALL scripts/slurm_cross_mtrf.sh
#
# Each task fits the source ONCE and scores both the source's own test (in-domain reference)
# and the target's test (transfer). Resume-friendly (OVERWRITE=false default).
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=xfer_mtrf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=48:00:00
# NOTE: shares fit_parcel_mtrf, now gcv_mode='eigen' (see slurm_mtrf_parcels.sh). 32 cores + 48h
# for the heavier eigen path on the D1-side fit (d1->d2 transfer).
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/xfer_mtrf_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/xfer_mtrf_%A_%a.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

MODEL_ID="${MODEL_ID:-whisper-base}"
source scripts/_whisper_features.sh
# Optional PCA-on-features (see slurm_mtrf_parcels.sh). Separate '-pca' output dir. Usage: PCA_VAR=0.95.
PCA_ARG=""; SUF=""
if [ -n "${PCA_VAR:-}" ]; then PCA_ARG="--pca_var ${PCA_VAR}"; SUF="-pca"; fi
D1_H5=outputs/neural_data/broderick2018_30s.h5;  D1_FEAT=$(resolve_feat "")           || { echo "no D1 features for $MODEL_ID"; exit 1; }
D2_H5=outputs/neural_data/surprisal_30s.h5;       D2_FEAT=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; }

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) S_TAG=d1; S_H5=$D1_H5; S_FT=$D1_FEAT; T_TAG=d2; T_H5=$D2_H5; T_FT=$D2_FEAT; OUT=d1-to-d2 ;;
  1) S_TAG=d2; S_H5=$D2_H5; S_FT=$D2_FEAT; T_TAG=d1; T_H5=$D1_H5; T_FT=$D1_FEAT; OUT=d2-to-d1 ;;
  *) echo "bad array id $TASK (use 0-1)"; exit 1 ;;
esac

echo "=================== transfer: $MODEL_ID  $S_TAG -> $T_TAG ==================="
python -m mbs.evaluation.evaluate_cross_dataset_mtrf \
  --model_id "$MODEL_ID" \
  --target_feature_layers "$LAYERS" \
  --source_tag "$S_TAG" --source_data_hdf5 "$S_H5" --source_features_dir "$S_FT" \
  --target_tag "$T_TAG" --target_data_hdf5 "$T_H5" --target_features_dir "$T_FT" \
  --output_dir "outputs/results/${MODEL_ID}-mtrf-xfer${SUF}-$OUT/" \
  --parcels_from "$D1_H5" \
  $PCA_ARG \
  --highpass_hz 0.5 --lag_max_ms 800 --n_train_time_samples 200 --overwrite "${OVERWRITE:-false}"

echo "DONE $MODEL_ID  $S_TAG -> $T_TAG"
