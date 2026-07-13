#!/bin/bash
# =============================================================================
# SLURM job: prefetch audio-model weights into cache/model_weights (ONE process).
# =============================================================================
#
# The delta-t extraction runs as a big SLURM array; if every task downloads the same model into the
# shared cache/model_weights at once, HF's file locking on the /work filesystem is unreliable and the
# config.json / weights get corrupted (JSONDecodeError etc.). Run this single-process job FIRST so the
# cache is warm and valid; the array tasks then only READ it. Submit the arrays with a dependency:
#
#   PF=$(sbatch --parsable scripts/slurm_prefetch_audio_models.sh)
#   sbatch --dependency=afterok:$PF --array=0-19 --export=ALL,MODEL_ID=whisper-large,... scripts/slurm_extract_delta_t_d2.sh
#
# force_download=True (wav2vec2) overwrites any already-corrupted cache; whisper.load_model
# re-verifies its checksum and self-heals. Set MODELS=... to prefetch a subset.
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=prefetch_audio
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=02:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/prefetch_audio_%j.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/prefetch_audio_%j.err

set -euo pipefail

MODELS="${MODELS:-whisper-large wav2vec2-medium wav2vec2-large}"
CACHE="${MODEL_CACHE_DIR:-cache/model_weights}"

cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling
source env.sh
mkdir -p logs "$CACHE"

echo "=== prefetch audio models -> ${CACHE} ==="
echo "Models: ${MODELS}"
echo "Start: $(date)"

for M in $MODELS; do
    echo "--- $M ---"
    python - "$M" "$CACHE" <<'PY'
import sys
from mbs.extraction.modeling.backbones.audio_models import load_model_audio
model_id, cache = sys.argv[1], sys.argv[2]
# force_download only matters for the wav2vec2 (HF) path; whisper ignores it and self-heals.
backbone, transform = load_model_audio(model_id, model_cache_dir=cache, force_download=True)
print(f"OK prefetched {model_id} -> {type(backbone).__name__}")
PY
done

echo "End: $(date)"
