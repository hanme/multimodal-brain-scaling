#!/bin/bash
# =============================================================================
# GPU (Kuma L40S): transcribe an audio file in cache/ to Markdown with openai-whisper.
# Much faster than CPU — whisper-medium does ~2 h of audio in ~15-30 min. No ffmpeg needed
# (soundfile decodes mp3). Submit from the kuma cluster.
#
#   sbatch scripts/kuma_transcribe.sh                                  # newest cache/ audio, medium
#   AUDIO=cache/Fried_depression_scales.mp3 MODEL=medium sbatch scripts/kuma_transcribe.sh
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=transcribe
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/transcribe_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/transcribe_%j.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs cache/whisper_models
nvidia-smi

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA unavailable — fix the cu126 torch build first (handover ENV NOTE), or use jed_transcribe.sh."
  exit 1
fi

MODEL="${MODEL:-medium}"
ARGS=(--model "$MODEL" --device cuda --model_cache_dir cache/whisper_models)
[ -n "${AUDIO:-}" ] && ARGS+=(--audio "$AUDIO")
[ -n "${LANG_CODE:-}" ] && ARGS+=(--language "$LANG_CODE")

python scripts/transcribe_audio.py "${ARGS[@]}"
echo "DONE AT $(date)"
