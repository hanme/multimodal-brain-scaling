#!/bin/bash
# =============================================================================
# CPU (Jed): aggregate the attention-encoder group-by-part CV folds into eeg_mapping-schema JSONs
# + the same layer-selection / fit-quality figures as the mTRF. Run AFTER kuma_probe_d2_cv.sh.
# Pure JSON/h5 reads (no feature loading) -> cheap; runs fine on a compute node.
#
#   sbatch scripts/jed_collect_encoder_cv.sh
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=enc_cv_agg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=00:30:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/enc_cv_agg_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/enc_cv_agg_%j.err

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs outputs/results/eeg_mapping_encoder

MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)
LEVELS=(parcels electrodes)
for MODEL in "${MODELS[@]}"; do
  for LEVEL in "${LEVELS[@]}"; do
    CVDIR="outputs/results/${MODEL}-probe-group-d2-${LEVEL}-cv"
    [ -d "$CVDIR" ] || { echo "skip ${MODEL}/${LEVEL} (no CV dir)"; continue; }
    echo "=== aggregate ${MODEL}/${LEVEL} ==="
    python scripts/eeg_mapping_encoder_cv.py \
      --model_id "$MODEL" --target_level "$LEVEL" --cv_dir "$CVDIR" \
      --out "outputs/results/eeg_mapping_encoder/${MODEL}__${LEVEL}__D2.json"
  done
done

for LEVEL in "${LEVELS[@]}"; do
  python scripts/plot_eeg_mapping.py \
    --results_dir outputs/results/eeg_mapping_encoder --target_level "$LEVEL" \
    --out_dir outputs/figures/eeg_mapping_encoder
done
echo "DONE -> outputs/results/eeg_mapping_encoder/*.json ; outputs/figures/eeg_mapping_encoder/*.png"
