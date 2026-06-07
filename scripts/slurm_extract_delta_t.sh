#!/bin/bash
# =============================================================================
# SLURM Array Job: Delta-T (causal) feature extraction for Whisper-base
# =============================================================================
#
# Each array task processes CHUNK_SIZE stimuli (stim_start_idx = task_id * CHUNK_SIZE).
# All chunks write to separate subdirs; merge with the commands below.
#
# PILOT (3 stimuli, full t_stride=1, ~40 min):
#   sbatch --array=0 scripts/slurm_extract_delta_t.sh
#
# FULL RUN (314 stimuli across 20 tasks, ~4-5 h each):
#   Edit MODE="full" below, then:
#   sbatch --array=0-19 scripts/slurm_extract_delta_t.sh
#
# After all tasks finish, merge + evaluate:
#   File names are globally unique (encoded stim_start_idx), so a flat cp works:
#   mkdir -p outputs/features/whisper-base-delta-t/merged/
#   cp outputs/features/whisper-base-delta-t/chunk_*/feats*.h5 \
#      outputs/features/whisper-base-delta-t/merged/
#   mbs-evaluate-temporal \
#     --model_id whisper-base \
#     --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
#     --features_dir outputs/features/whisper-base-delta-t/merged/ \
#     --data_hdf5_path outputs/neural_data/broderick2018_30s.h5 \
#     --output_dir outputs/results/whisper-base-delta-t/
# =============================================================================

# ── SBATCH directives (must be before any executable code) ───────────────────
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=delta_t_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=12:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/delta_t_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/delta_t_%A_%a.err

# ── Mode switch ───────────────────────────────────────────────────────────────
# Set to "pilot" for the first 3-stimulus test job, "full" for the real run.
MODE="full"   # "pilot" | "full"

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
DATA_ROOT="/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/stimuli"
LAYERS_CONFIG="configs/extraction/audio/whisper_base_layers.json"
MODEL_ID="whisper-base"
BATCH_T=4

if [ "$MODE" = "pilot" ]; then
    CHUNK_SIZE=3         # stimuli per task
    T_STRIDE=1           # full temporal resolution
    SAVE_EVERY=3
    OUTPUT_DIR="outputs/features/whisper-base-delta-t-slurm-pilot"
else
    CHUNK_SIZE=16        # 20 tasks × 16 = 320 >= 314 stimuli
    T_STRIDE=1
    SAVE_EVERY=8
    OUTPUT_DIR="outputs/features/whisper-base-delta-t"
fi

# =============================================================================
echo "========================================================================"
echo "DELTA-T FEATURE EXTRACTION  (mode=${MODE})"
echo "Array task ${SLURM_ARRAY_TASK_ID} / Job ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"
echo "========================================================================"

# ── Environment ───────────────────────────────────────────────────────────────
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

source env.sh
echo "Python: $(which python)"
echo ""

# ── Compute stim range for this task ─────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
STIM_START=$(( TASK_ID * CHUNK_SIZE ))
CHUNK_DIR="${OUTPUT_DIR}/chunk_${TASK_ID}"

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
    --window_duration       30.0 \
    --window_stride         10.0 \
    --batch_t               $BATCH_T \
    --t_stride              $T_STRIDE \
    --stim_start_idx        $STIM_START \
    --n_stimuli             $CHUNK_SIZE \
    --save_every            $SAVE_EVERY

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
