#!/bin/bash
# =============================================================================
# SLURM Array Job: Delta-T (causal) feature extraction — generic / parametrised
# =============================================================================
#
# Parameters are passed via environment variables (use --export=ALL or
# export them before calling sbatch).  All have sensible defaults so the
# script can also be sourced for inspection.
#
# Required env vars (with defaults shown):
#   MODEL_ID       = whisper-small   # e.g. whisper-tiny/base/small/medium/large
#   WINDOW_DUR     = 30.0            # stimulus window duration in seconds
#   WINDOW_STRIDE  = 10.0            # stride in seconds
#   CHUNK_SIZE     = 8               # stimuli per array task
#   BATCH_T        = 4               # time-step truncations per forward pass
#
# Output goes to:
#   outputs/features/{MODEL_ID}-w{W}s{S}-delta-t/chunk_{TASK_ID}/
# where W = window (integer s), S = stride (zero-padded integer s).
#
# Usage (from repo root):
#   export MODEL_ID=whisper-small WINDOW_DUR=30.0 WINDOW_STRIDE=5.0
#   export CHUNK_SIZE=8 BATCH_T=4
#   sbatch --array=0-79 --export=ALL scripts/slurm_extract_delta_t_generic.sh
#
# Or use scripts/submit_whisper_small_sweep.sh to submit all configs at once.
# =============================================================================

# ── SBATCH directives ────────────────────────────────────────────────────────
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=delta_t_generic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=12:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/delta_t_%x_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/delta_t_%x_%A_%a.err

# ── Parameters (read from env, fall back to defaults) ────────────────────────
MODEL_ID="${MODEL_ID:-whisper-small}"
WINDOW_DUR="${WINDOW_DUR:-30.0}"
WINDOW_STRIDE="${WINDOW_STRIDE:-10.0}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
BATCH_T="${BATCH_T:-4}"

# ── Derived paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
DATA_ROOT="/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/stimuli"

# Layer config: whisper-small -> whisper_small_layers.json (replaces - with _)
LAYERS_CONFIG="configs/extraction/audio/${MODEL_ID//-/_}_layers.json"

# Output directory encodes model + window + stride for unambiguous file naming.
# W = window as integer (e.g. "30"), S = stride zero-padded to 2 digits (e.g. "05").
WIN_INT="${WINDOW_DUR%.*}"
STRIDE_PAD=$(printf '%02d' "${WINDOW_STRIDE%.*}")
OUTPUT_BASE="outputs/features/${MODEL_ID}-w${WIN_INT}s${STRIDE_PAD}-delta-t"

# ── Header ────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "DELTA-T FEATURE EXTRACTION  (generic)"
echo "Model:   ${MODEL_ID}   window=${WINDOW_DUR}s   stride=${WINDOW_STRIDE}s"
echo "Layers:  ${LAYERS_CONFIG}"
echo "Output:  ${OUTPUT_BASE}/chunk_<N>/"
echo "Array task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"
echo "========================================================================"

# ── Environment ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
source env.sh
echo "Python: $(which python)"
echo ""

# ── Validate inputs ───────────────────────────────────────────────────────────
if [ ! -f "$LAYERS_CONFIG" ]; then
    echo "ERROR: layer config not found: $LAYERS_CONFIG"
    exit 1
fi

# ── Compute stim range for this task ─────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
STIM_START=$(( TASK_ID * CHUNK_SIZE ))
CHUNK_DIR="${OUTPUT_BASE}/chunk_${TASK_ID}"

echo "Stimuli: start=${STIM_START}  count=${CHUNK_SIZE}"
echo "Output:  ${CHUNK_DIR}"
echo ""

mkdir -p "$CHUNK_DIR"

# ── Run extraction ────────────────────────────────────────────────────────────
OMP_NUM_THREADS=1 python -m mbs.extraction.extract_features_delta_t \
    --model_id              "$MODEL_ID" \
    --data_root             "$DATA_ROOT" \
    --target_feature_layers "$LAYERS_CONFIG" \
    --output_dir            "$CHUNK_DIR" \
    --window_duration       "$WINDOW_DUR" \
    --window_stride         "$WINDOW_STRIDE" \
    --batch_t               "$BATCH_T" \
    --t_stride              1 \
    --stim_start_idx        "$STIM_START" \
    --n_stimuli             "$CHUNK_SIZE" \
    --save_every            "$CHUNK_SIZE"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS  task=${TASK_ID}  stim_start=${STIM_START}"
    echo "Files written:"
    ls -lh "$CHUNK_DIR"
else
    echo "FAILED   task=${TASK_ID}  exit=${EXIT_CODE}"
fi

echo "End: $(date)"
exit $EXIT_CODE
