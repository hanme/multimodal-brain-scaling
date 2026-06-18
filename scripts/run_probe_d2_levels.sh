#!/bin/bash
# =============================================================================
# Learned temporal attention encoder on D2 ONLY, BOTH target levels (parcels + electrodes) —
# GPU body, any whisper size. The §16/§20 redo: MSE loss (now the engine default), EEG targets
# z-scored per target with the scaling stored in the checkpoint (predictions invert to real
# units), MIRAGE-style checkpoint selection on a validation split carved from TRAIN (the test
# split is never touched during selection). Targets/NC come from D2 (surprisal_30s.h5) — the
# same definitions as the §20 mTRF sweep (scripts/eeg_targets.py).
#
# Run on an interactive GPU node (or wrap in an sbatch like kuma_probe_d2d3.sh):
#   srun --partition l40s --gres=gpu:1 --cpus-per-task 8 --time 02:00:00 --pty bash
#   source env.sh && MODEL_ID=whisper-small bash scripts/run_probe_d2_levels.sh
# =============================================================================
set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA not available — fix the cu126 torch build first (handover ENV NOTE). Aborting."
  exit 1
fi

MODEL_ID="${MODEL_ID:-whisper-small}"
source scripts/_whisper_features.sh
D2_FEAT=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; }
D2_NEURAL="outputs/neural_data/surprisal_30s.h5"

probe_level () {  # $1 = parcels | electrodes
  echo "=================== encoder (group): $MODEL_ID / D2 / $1  (feats: $D2_FEAT) ==================="
  python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
    --model_id "$MODEL_ID" \
    --target_feature_layers "$LAYERS" \
    --data_hdf5_path "$D2_NEURAL" \
    --features_dir   "$D2_FEAT" \
    --output_dir     "outputs/results/${MODEL_ID}-probe-group-d2-$1/" \
    --readout_level group \
    --target_level  "$1" \
    --parcels_from "$D2_NEURAL" \
    --highpass_hz 0.5 --lookback_ms 800 --nc_threshold 0.2 \
    --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
    --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
    --val_frac 0.2 --eval_every 5 \
    --amp false --device cuda --overwrite true
}

probe_level parcels
probe_level electrodes

echo "ALL DONE -> outputs/results/${MODEL_ID}-probe-group-d2-{parcels,electrodes}/attn_probe_temporal_scores.h5"
echo "          checkpoints (with eeg_mu/eeg_sd): .../model__<layer>.pt"
