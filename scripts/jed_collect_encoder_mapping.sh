#!/bin/bash
# =============================================================================
# CPU (Jed): turn the trained attention-encoder sweep into eeg_mapping-schema JSONs + the SAME
# layer-selection / fit-quality figures the mTRF uses (plot_eeg_mapping.py). Run on a COMPUTE node,
# not the login node (per-layer feature arrays OOM the head node -> exit 137).
#
# For every finished outputs/results/<model>-probe-group-d2-<level>/ it computes the per-layer
# VALIDATION r (train-carved split, reproduced from the checkpoint) for honest layer selection, then
# writes outputs/results/eeg_mapping_encoder/<model>__<level>__D2.json and the two figures.
#
#   sbatch scripts/jed_collect_encoder_mapping.sh
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=enc_mapping
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=02:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/enc_mapping_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/enc_mapping_%j.err

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs outputs/results/eeg_mapping_encoder

MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)
LEVELS=(parcels electrodes)
SIGF=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features

for MODEL in "${MODELS[@]}"; do
  for LEVEL in "${LEVELS[@]}"; do
    RDIR="outputs/results/${MODEL}-probe-group-d2-${LEVEL}"
    [ -f "$RDIR/attn_probe_temporal_scores.h5" ] || { echo "skip ${MODEL}/${LEVEL} (no scores h5)"; continue; }
    FEAT="outputs/features/${MODEL}-delta-t-surprisal/merged"
    [ -d "$FEAT" ] || FEAT="${SIGF}/${MODEL}-delta-t-surprisal/merged"
    echo "=== collect ${MODEL}/${LEVEL} (feats: ${FEAT}) ==="
    python scripts/eeg_mapping_encoder.py \
      --model_id "$MODEL" --target_level "$LEVEL" \
      --results_dir "$RDIR" --features_dir "$FEAT" \
      --neural outputs/neural_data/surprisal_30s.h5 \
      --out "outputs/results/eeg_mapping_encoder/${MODEL}__${LEVEL}__D2.json"
  done
done

# Same plotter as the mTRF, pointed at the encoder JSONs.
for LEVEL in "${LEVELS[@]}"; do
  python scripts/plot_eeg_mapping.py \
    --results_dir outputs/results/eeg_mapping_encoder --target_level "$LEVEL" \
    --out_dir outputs/figures/eeg_mapping_encoder
done

echo "DONE -> outputs/results/eeg_mapping_encoder/*.json ; outputs/figures/eeg_mapping_encoder/*.png"
