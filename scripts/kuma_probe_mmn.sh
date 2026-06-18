#!/bin/bash
# =============================================================================
# Workstream B for the MMN deliverable: train the attention encoder on D2 and SAVE a reusable
# checkpoint (model__<layer>.pt) so it can be applied to the MMN tones + Sophie's own stimuli.
# GPU (Kuma L40S). SUBMIT FROM KUMA. /work is shared, so D2 neural h5 + features are in place.
#
#   sbatch scripts/kuma_probe_mmn.sh                                  # whisper-small, blocks.10
#   MODEL_ID=whisper-small LAYER_ID=10 sbatch scripts/kuma_probe_mmn.sh
#
# Differs from kuma_probe_d2d3.sh in three deliberate ways:
#   * --parcels_from = the D2 neural file (NOT broderick) -> D2-native 5 parcels INCLUDING central
#     + FCz (fronto-central, where the MMN is largest; dropped under the canonical D1 scheme only
#     because Broderick's central electrodes are unreliable). The cross-dataset scaling TABLES keep
#     the canonical-4 scheme; this single-dataset MMN illustration uses D2's own reliable parcels.
#   * --save_model true -> persists the trained mapping as a checkpoint (the actual deliverable).
#   * --layer_id ${LAYER_ID} -> trains only the figure layer (fast); drop it to scan all layers.
# Probe hyperparams match run_probe_d2d3.sh (the adopted D1-sweep config).
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_mmn
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_mmn_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_mmn_%j.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date) on $(hostname)"
source env.sh
mkdir -p logs
nvidia-smi || true

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA unavailable. Fix the cu126 torch build first (handover ENV NOTE), then resubmit."
  exit 1
fi

MODEL_ID="${MODEL_ID:-whisper-small}"
LAYER_ID="${LAYER_ID:-10}"
source scripts/_whisper_features.sh
D2_FEAT=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; }
NEURAL="outputs/neural_data/surprisal_30s.h5"
OUT="outputs/results/${MODEL_ID}-probe-group-d2-mmn/"

echo "=== probe (group) for MMN: $MODEL_ID / D2 / layer_id=$LAYER_ID  (feats: $D2_FEAT) ==="
echo "    parcels_from = $NEURAL (D2-native 5 parcels incl. central) | checkpoint ON"

python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
  --model_id "$MODEL_ID" \
  --target_feature_layers "$LAYERS" \
  --data_hdf5_path "$NEURAL" \
  --features_dir   "$D2_FEAT" \
  --output_dir     "$OUT" \
  --readout_level group \
  --parcels_from   "$NEURAL" \
  --highpass_hz 0.5 --lookback_ms 800 \
  --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
  --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
  --amp false --device cuda --overwrite true \
  --layer_id "$LAYER_ID" \
  --save_model true

echo "DONE -> ${OUT}model__blocks.${LAYER_ID}.pt  (+ attn_probe_temporal_scores.h5)"
echo "Next (CPU, on jed): python scripts/insilico_mmn_attn.py \\"
echo "  --checkpoint ${OUT}model__blocks.${LAYER_ID}.pt \\"
echo "  --mmn_features_root outputs/features/${MODEL_ID}-mmn --method method_09 \\"
echo "  --features_dir $D2_FEAT --neural $NEURAL \\"
echo "  --out_dir outputs/figures/insilico_mmn_small --data_dir outputs/insilico_mmn_predictions_small"
echo "End: $(date)"
