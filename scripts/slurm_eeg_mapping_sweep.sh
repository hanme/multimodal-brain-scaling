#!/bin/bash
# =============================================================================
# Model->EEG mTRF layer sweep on D2 with CV-on-train layer selection.
# ARRAY: one task per (model x target_level). 8 tasks = {tiny,base,small,medium} x {parcels,electrodes}.
# CPU only (sklearn RidgeCV, gcv_mode='eigen'). The medium/electrodes task is the long pole
# (24 layers x 5-fold CV on a wide design) — use PCA_VAR to tame it if it runs long.
#
#   sbatch --array=0-7 scripts/slurm_eeg_mapping_sweep.sh
#   PCA_VAR=0.95 sbatch --array=6,7 --export=ALL scripts/slurm_eeg_mapping_sweep.sh   # just medium
# After all tasks finish:
#   python scripts/plot_eeg_mapping.py --target_level parcels
#   python scripts/plot_eeg_mapping.py --target_level electrodes
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=eeg_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=72:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/eeg_sweep_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/eeg_sweep_%A_%a.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs outputs/results/eeg_mapping
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)
LEVELS=(parcels electrodes)
SIGF=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 2))]}
LEVEL=${LEVELS[$((TASK % 2))]}
# D2 features: local copy if present, else Sophie's read-only tree.
FEAT="outputs/features/${MODEL}-delta-t-surprisal/merged"
[ -d "$FEAT" ] || FEAT="${SIGF}/${MODEL}-delta-t-surprisal/merged"

PCA_ARG=""; [ -n "${PCA_VAR:-}" ] && PCA_ARG="--pca_var ${PCA_VAR}"

echo "=== eeg_mapping_sweep: ${MODEL} / ${LEVEL}  (feats: ${FEAT}) ==="
python scripts/eeg_mapping_sweep.py \
  --model_id "$MODEL" --target_level "$LEVEL" \
  --features_dir "$FEAT" --neural outputs/neural_data/surprisal_30s.h5 \
  $PCA_ARG \
  --out "outputs/results/eeg_mapping/${MODEL}__${LEVEL}__D2.json"

echo "DONE ${MODEL}/${LEVEL}"
