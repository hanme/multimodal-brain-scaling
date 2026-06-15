#!/bin/bash
# =============================================================================
# Learned temporal probe (Workstream B) on D2 + D3 — GPU body, ANY whisper size. Plan §13 Q2.
# Called by scripts/kuma_probe_d2d3.sh (sbatch), or run directly on an interactive GPU node:
#   srun --partition l40s --gres=gpu:1 --cpus-per-task 8 --time 01:00:00 --pty bash
#   source env.sh && MODEL_ID=whisper-small bash scripts/run_probe_d2d3.sh
#
# Adopted best config from the D1 sweep: d_model=64, num_latents=4, 1 layer, wd=1e-2,
# dropout=0.3, amp off. Group readout, canonical 4 parcels, test_d1/test_d2 scored separately.
# =============================================================================
set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
mkdir -p logs

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA not available — fix the cu126 torch build first (handover ENV NOTE). Aborting."
  exit 1
fi

MODEL_ID="${MODEL_ID:-whisper-base}"
source scripts/_whisper_features.sh
D2_FEAT=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; }
D3_FEAT=$(resolve_feat "-d3" || true)
if [ -z "${D3_FEAT:-}" ]; then
  D1=$(resolve_feat "")          || { echo "no D1 features for $MODEL_ID"; exit 1; }
  D2=$(resolve_feat "-surprisal") || { echo "no D2 features for $MODEL_ID"; exit 1; }
  D3_FEAT="outputs/features/${MODEL_ID}-delta-t-d3/merged"
  echo "[setup] building local D3 feature dir = D1 u D2 for $MODEL_ID ..."
  mkdir -p "$D3_FEAT"; cp "$D1"/*.h5 "$D3_FEAT"/; cp "$D2"/*.h5 "$D3_FEAT"/
fi

probe () {  # $1=tag  $2=neural_h5  $3=feature_dir
  echo "=================== probe (group): $MODEL_ID / $1  (feats: $3) ==================="
  python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
    --model_id "$MODEL_ID" \
    --target_feature_layers "$LAYERS" \
    --data_hdf5_path "$2" \
    --features_dir   "$3" \
    --output_dir     "outputs/results/${MODEL_ID}-probe-group-$1/" \
    --readout_level group \
    --parcels_from outputs/neural_data/broderick2018_30s.h5 \
    --highpass_hz 0.5 --lookback_ms 800 \
    --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
    --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
    --amp false --device cuda --overwrite true
}

probe d2 outputs/neural_data/surprisal_30s.h5   "$D2_FEAT"   # Q1: probe replication on D2
probe d3 outputs/neural_data/d3_combined_30s.h5 "$D3_FEAT"   # Q2: more data — does the gap shrink?

echo "ALL DONE -> outputs/results/${MODEL_ID}-probe-group-{d2,d3}/attn_probe_temporal_scores.h5"
