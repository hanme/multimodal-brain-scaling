#!/bin/bash
# =============================================================================
# GPU (Kuma L40S): train the FINAL reusable attention-encoder checkpoint at each model's
# CV-chosen layer (read from outputs/results/eeg_mapping_encoder/<model>__<level>__D2.json).
# ARRAY: 8 tasks = {tiny,base,small,medium} x {parcels,electrodes}. One layer/task -> cheap.
# Produces the MMN checkpoint (with eeg_mu/eeg_sd) + clean test r. Run AFTER jed_collect_encoder_cv.sh.
#
#   sbatch --array=0-7 scripts/kuma_probe_d2_final.sh
# TASK -> model = TASK/2 ; level = TASK%2
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_final
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_final_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_final_%A_%a.error

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

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 2))]}
LEVEL=${LEVELS[$((TASK % 2))]}
LAYERS="configs/extraction/audio/${MODEL//-/_}_layers.json"
FEAT="outputs/features/${MODEL}-delta-t-surprisal/merged"
[ -d "$FEAT" ] || FEAT="${SIGF}/${MODEL}-delta-t-surprisal/merged"
JSON="outputs/results/eeg_mapping_encoder/${MODEL}__${LEVEL}__D2.json"
[ -f "$JSON" ] || { echo "🛑 no CV JSON $JSON — run jed_collect_encoder_cv.sh first"; exit 1; }

CHOSEN=$(python -c "import json;print(json.load(open('$JSON'))['chosen_layer'])")
# layer index in the model's layer config (driver selects by --layer_id)
LID=$(python -c "import json;ls=[e['name'] if isinstance(e,dict) else e for e in json.load(open('$LAYERS'))];print(ls.index('$CHOSEN'))")
echo "=== final checkpoint: ${MODEL} / ${LEVEL} / chosen ${CHOSEN} (idx ${LID}) ==="

python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
  --model_id "$MODEL" \
  --target_feature_layers "$LAYERS" \
  --data_hdf5_path "$D2_NEURAL" \
  --features_dir   "$FEAT" \
  --output_dir     "outputs/results/${MODEL}-probe-group-d2-${LEVEL}" \
  --readout_level group \
  --target_level  "$LEVEL" \
  --parcels_from "$D2_NEURAL" \
  --layer_id "$LID" \
  --val_mode grouped --n_folds 4 --fold_idx 0 \
  --highpass_hz 0.5 --lookback_ms 800 --nc_threshold 0.2 \
  --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
  --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
  --eval_every 5 --save_model true \
  --amp false --device cuda --overwrite true

echo "DONE ${MODEL}/${LEVEL} final AT $(date)"
