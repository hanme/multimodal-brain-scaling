#!/bin/bash
# =============================================================================
# GPU (Kuma L40S): D2 attention encoder sweep — the encoder twin of slurm_eeg_mapping_sweep.sh.
# ARRAY: one task per (model x target_level). 8 tasks = {tiny,base,small,medium} x {parcels,electrodes}.
# Each task trains ONE (model, level) on its own GPU, so all run in parallel (subject to GPU
# availability) instead of the single-job kuma_probe_d2_levels.sh (one model, levels sequential).
#
# §16/§20 config: MSE loss, EEG z-scored + scaling stored in the checkpoint, MIRAGE-style
# checkpoint selection on a train-carved validation split. D2 only.
#
#   sbatch --array=0-7 scripts/kuma_probe_d2_sweep.sh                 # all 4 models x 2 levels
#   sbatch --array=4,5 scripts/kuma_probe_d2_sweep.sh                 # just whisper-small (both levels)
#   OVERWRITE=false sbatch --array=6,7 --time 08:00:00 --export=ALL \
#       scripts/kuma_probe_d2_sweep.sh                                # RESUME: only the missing layers
# OVERWRITE (default true) -> false reuses an existing scores.h5 and runs ONLY the layers not yet in
# it (the driver skips present layer keys), so a walltime-truncated model finishes the remainder.
# After all tasks finish:
#   outputs/results/<model>-probe-group-d2-{parcels,electrodes}/attn_probe_temporal_scores.h5
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_d2_swp
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_d2_swp_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_d2_swp_%A_%a.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs
nvidia-smi

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA unavailable. Fix the cu126 torch build first (handover ⚠️ ENV NOTE), then resubmit."
  exit 1
fi

MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)
LEVELS=(parcels electrodes)
SIGF=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features
D2_NEURAL="outputs/neural_data/surprisal_30s.h5"
OVERWRITE="${OVERWRITE:-true}"   # false => resume: keep existing scores.h5, run only missing layers

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 2))]}
LEVEL=${LEVELS[$((TASK % 2))]}
LAYERS="configs/extraction/audio/${MODEL//-/_}_layers.json"
# D2 features: local copy if present, else Sophie's read-only tree.
FEAT="outputs/features/${MODEL}-delta-t-surprisal/merged"
[ -d "$FEAT" ] || FEAT="${SIGF}/${MODEL}-delta-t-surprisal/merged"

echo "=== encoder (group): ${MODEL} / D2 / ${LEVEL}  (feats: ${FEAT}) ==="
python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
  --model_id "$MODEL" \
  --target_feature_layers "$LAYERS" \
  --data_hdf5_path "$D2_NEURAL" \
  --features_dir   "$FEAT" \
  --output_dir     "outputs/results/${MODEL}-probe-group-d2-${LEVEL}/" \
  --readout_level group \
  --target_level  "$LEVEL" \
  --parcels_from "$D2_NEURAL" \
  --highpass_hz 0.5 --lookback_ms 800 --nc_threshold 0.2 \
  --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
  --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
  --val_frac 0.2 --eval_every 5 \
  --amp false --device cuda --overwrite "$OVERWRITE"

echo "DONE ${MODEL}/${LEVEL} AT $(date)"
