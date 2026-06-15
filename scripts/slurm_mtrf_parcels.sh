#!/bin/bash
# =============================================================================
# Parcel-level mTRF encoder — D1 (sanity) / D2 (replication) / D3 (pooled), ANY whisper size.
# CPU only (sklearn RidgeCV). Plan §13. Scores test_d1/test_d2 SEPARATELY for D3.
#
# ARRAY job — one dataset per task (parallel, isolated wall clock):
#   sbatch --array=0-2 scripts/slurm_mtrf_parcels.sh                       # whisper-base (default)
#   MODEL_ID=whisper-small sbatch --array=0-2 --export=ALL scripts/slurm_mtrf_parcels.sh
#   MODEL_ID=whisper-medium sbatch --array=2 --export=ALL scripts/slurm_mtrf_parcels.sh   # just d3
#
# Features resolve to our local copy if present, else Sophie's read-only tree (see
# scripts/_whisper_features.sh). Same canonical 4 parcels for every model/dataset.
# OVERWRITE=false (default) resumes; OVERWRITE=true recomputes.
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=mtrf_parcels
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=48:00:00
# NOTE: RidgeCV now uses gcv_mode='eigen' (avoids the LAPACK SVD int32 overflow that segfaulted
# small/medium on D1/D3). Eigen does an n×n Gram eigendecomposition (n = n_windows·200), so it is
# heavier than svd for the big fitting sets: 32 cores give the LAPACK threads + RAM headroom
# (D3 Gram ≈ 50 GB in float64), and 'standard' has no wall cap so 48h is comfortable.
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/mtrf_parcels_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/mtrf_parcels_%A_%a.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

MODEL_ID="${MODEL_ID:-whisper-base}"
source scripts/_whisper_features.sh
PARCELS_FROM=outputs/neural_data/broderick2018_30s.h5   # canonical parcel membership (D1)

# Optional PCA-on-features (variance-preserving, PC count varies by model/layer). When set, results
# go to a separate '-pca' output dir so they don't clobber the raw-feature (eigen) runs, and the
# narrow design lets D3 fit without overflow. Usage: PCA_VAR=0.95 sbatch ... (default: off).
PCA_ARG=""; SUF=""
if [ -n "${PCA_VAR:-}" ]; then PCA_ARG="--pca_var ${PCA_VAR}"; SUF="-pca"; fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) TAG=d1; NEURAL=outputs/neural_data/broderick2018_30s.h5;  FEAT=$(resolve_feat "")          || { echo "no D1 features for $MODEL_ID"; exit 1; } ;;
  1) TAG=d2; NEURAL=outputs/neural_data/surprisal_30s.h5;      FEAT=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; } ;;
  2) TAG=d3; NEURAL=outputs/neural_data/d3_combined_30s.h5;    FEAT=$(resolve_feat "-d3" || true) ;;
  *) echo "bad array id $TASK (use 0-2)"; exit 1 ;;
esac

# d3: if no prebuilt union dir exists anywhere, build it locally from D1 u D2 (names don't collide)
if [ "$TAG" = "d3" ] && [ -z "${FEAT:-}" ]; then
  D1=$(resolve_feat "")          || { echo "no D1 features for $MODEL_ID"; exit 1; }
  D2=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; }
  FEAT="outputs/features/${MODEL_ID}-delta-t-d3/merged"
  echo "[setup] building local D3 feature dir = D1 u D2 for $MODEL_ID ..."
  mkdir -p "$FEAT"; cp "$D1"/*.h5 "$FEAT"/; cp "$D2"/*.h5 "$FEAT"/
  echo "[setup] D3 feature files: $(ls "$FEAT" | wc -l)"
fi

echo "=================== mTRF parcels: $MODEL_ID / $TAG${SUF}  (feats: $FEAT) ==================="
python -m mbs.evaluation.evaluate_features_mtrf_parcels \
  --model_id "$MODEL_ID" \
  --target_feature_layers "$LAYERS" \
  --data_hdf5_path "$NEURAL" \
  --features_dir   "$FEAT" \
  --output_dir     "outputs/results/${MODEL_ID}-mtrf-parcels${SUF}-$TAG/" \
  --parcels_from   "$PARCELS_FROM" \
  $PCA_ARG \
  --highpass_hz 0.5 --lag_max_ms 800 --n_train_time_samples 200 --overwrite "${OVERWRITE:-false}"

echo "DONE $MODEL_ID/$TAG -> outputs/results/${MODEL_ID}-mtrf-parcels${SUF}-$TAG/mtrf_parcel_scores.h5"
