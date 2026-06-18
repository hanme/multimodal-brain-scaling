#!/bin/bash
# =============================================================================
# GPU (Kuma L40S): attention-encoder group-by-part CV sweep on D2.
# ARRAY: one task per (model x level x fold). 32 tasks = {tiny,base,small,medium} x
# {parcels,electrodes} x {fold 0..3}. Each task trains ALL layers on 3 of the 4 audiobook-part
# groups and validates on the held-out group (non-overlapping selection split — no 20 s window
# leak). No checkpoints saved here (selection only); per-layer val r + clean test r are written.
#
#   sbatch --array=0-31 scripts/kuma_probe_d2_cv.sh
# TASK -> model = TASK/8 ; level = (TASK%8)/4 ; fold = TASK%4
# After all 32 finish: aggregate with scripts/jed_collect_encoder_cv.sh.
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_cv
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_cv_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_cv_%A_%a.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs
nvidia-smi

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA unavailable. Fix the cu126 torch build first (handover ⚠️ ENV NOTE)."; exit 1
fi

MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)
LEVELS=(parcels electrodes)
SIGF=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features
D2_NEURAL="outputs/neural_data/surprisal_30s.h5"
NFOLDS=4

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 8))]}
LEVEL=${LEVELS[$(((TASK % 8) / 4))]}
FOLD=$((TASK % 4))
LAYERS="configs/extraction/audio/${MODEL//-/_}_layers.json"
FEAT="outputs/features/${MODEL}-delta-t-surprisal/merged"
[ -d "$FEAT" ] || FEAT="${SIGF}/${MODEL}-delta-t-surprisal/merged"
OUT="outputs/results/${MODEL}-probe-group-d2-${LEVEL}-cv/fold${FOLD}"

echo "=== encoder CV: ${MODEL} / D2 / ${LEVEL} / fold ${FOLD} of ${NFOLDS}  (feats: ${FEAT}) ==="
python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
  --model_id "$MODEL" \
  --target_feature_layers "$LAYERS" \
  --data_hdf5_path "$D2_NEURAL" \
  --features_dir   "$FEAT" \
  --output_dir     "$OUT" \
  --readout_level group \
  --target_level  "$LEVEL" \
  --parcels_from "$D2_NEURAL" \
  --val_mode grouped --n_folds "$NFOLDS" --fold_idx "$FOLD" \
  --highpass_hz 0.5 --lookback_ms 800 --nc_threshold 0.2 \
  --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
  --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
  --eval_every 5 --save_model false \
  --amp false --device cuda --overwrite true

echo "DONE ${MODEL}/${LEVEL}/fold${FOLD} AT $(date)"
