#!/bin/bash
# =============================================================================
# SLURM (CPU, jed standard): Method A in-silico MMN at PARCEL level — fit FIR mTRF on
#   D2 (Cortical Surprisal) at MODEL_ID's committed §1.5 layer, apply to all 10 MMN
#   methods in one run (insilico_mmn.py loops --methods all internally). ~1-2 min
#   compute once scheduled; the Ridge fit needs >> the login-node memory cap.
#
# Submit one model:
#   sbatch --export=ALL,MODEL_ID=whisper-tiny scripts/slurm_insilico_mmn.sh
# Submit several remaining models as an array (TASK -> MODELS[TASK]):
#   sbatch --array=0,2 scripts/slurm_insilico_mmn.sh        # whisper-tiny, whisper-small
# Extra args after the script name are forwarded to insilico_mmn.py.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=insilico_mmn
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5500M
#SBATCH --time=00:30:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/insilico_mmn_%A_%a.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/insilico_mmn_%A_%a.err

PROJECT_DIR="/work/upschrimpf1/sigfstea/multimodal-brain-scaling"
cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs outputs/figures outputs/insilico_mmn_predictions

# committed §1.5 mTRF-parcels layer per model (do not re-select)
declare -A MTRF_PARCELS_LAYER=(
    [whisper-tiny]=blocks.0  [whisper-base]=blocks.0
    [whisper-small]=blocks.3 [whisper-medium]=blocks.11
)
MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_ID="${MODELS[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_ID="${MODEL_ID:-whisper-base}"
fi
LAYER="${MTRF_PARCELS_LAYER[$MODEL_ID]}"
[ -n "$LAYER" ] || { echo "unknown MODEL_ID=$MODEL_ID"; exit 1; }
TRAIN_FEATURES="outputs/features/${MODEL_ID}-delta-t-surprisal/merged"
# whisper-base keeps its historical MMN-features root (outputs/features); every other model's
# MMN delta_T features live under a model-scoped root so they don't collide.
if [ "$MODEL_ID" = "whisper-base" ]; then MMN_ROOT="outputs/features"; else MMN_ROOT="outputs/features/${MODEL_ID}-mmn"; fi

echo "Start: $(date) on $(hostname)   MODEL_ID=$MODEL_ID  layer=$LAYER  features=$TRAIN_FEATURES  mmn_root=$MMN_ROOT"

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python scripts/insilico_mmn.py \
    --layer "$LAYER" \
    --train_features "$TRAIN_FEATURES" \
    --mmn_features_root "$MMN_ROOT" \
    --lag_max_ms 800 \
    --out_dir "outputs/figures/insilico_mmn/${MODEL_ID}" \
    --data_dir "outputs/insilico_mmn_predictions/${MODEL_ID}" \
    "$@"
EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS ${MODEL_ID}/parcels" || echo "FAILED ${MODEL_ID}/parcels exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
