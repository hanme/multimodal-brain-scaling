#!/bin/bash
# =============================================================================
# CPU (Jed): transcribe an audio file in cache/ to Markdown with openai-whisper.
# Whisper medium on CPU runs at a few× real-time, fine for short clips; use a smaller model
# (MODEL=small / base) for long files or impatience. No ffmpeg needed (soundfile decodes mp3).
#
#   sbatch scripts/jed_transcribe.sh                                  # newest cache/ audio, medium
#   AUDIO=cache/talk.mp3 MODEL=small sbatch scripts/jed_transcribe.sh # specific file + model
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=transcribe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=02:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/transcribe_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/transcribe_%j.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs cache/whisper_models
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

MODEL="${MODEL:-medium}"
ARGS=(--model "$MODEL" --device cpu --model_cache_dir cache/whisper_models)
[ -n "${AUDIO:-}" ] && ARGS+=(--audio "$AUDIO")
[ -n "${LANG_CODE:-}" ] && ARGS+=(--language "$LANG_CODE")

python scripts/transcribe_audio.py "${ARGS[@]}"
echo "DONE AT $(date)"
