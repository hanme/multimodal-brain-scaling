#!/bin/bash
# =============================================================================
# Held-out fit-quality BAR figure across the whisper size ladder (D2 only).
# For each model (tiny/base/small/medium) refits the mTRF at its best D2 layer with the
# 5-parcel definition (frontal/central/temporal/parietal/occipital, NC floor r>0.2 from
# surprisal_30s.h5) and plots per-parcel held-out TEST r as bars — one panel per model.
# CPU only (sklearn RidgeCV); the OOM on the login node is why this goes through a compute node.
#
#   sbatch scripts/slurm_fit_quality_bars.sh
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=fq_bars
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=04:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/fq_bars_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/fq_bars_%j.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs outputs/figures/fit_quality
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "=================== fit-quality bars: all whisper models / D2 ==================="
python scripts/plot_fit_quality_bars.py \
  --neural outputs/neural_data/surprisal_30s.h5 \
  --out outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png

echo "DONE -> outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png"
