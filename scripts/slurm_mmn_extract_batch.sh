#!/bin/bash
# =============================================================================
# SLURM Array: delta_T feature extraction for MMN stimuli -- FEW-SUBMISSIONS variant.
#   One array task = ONE method (all its clips extracted in a single python call, so the
#   model loads once), so a whole model is a SINGLE submission of a 48-task array -- vs
#   48 submissions x 16-task arrays in slurm_mmn_extract.sh (336 submissions total).
#   ~16 clips/method; well under the 24 h limit even for the large models.
#   Output is identical (mmn-<method>-delta-t/ with one feats_delta_t-*.h5 per clip).
#
# Submit one whisper model (30 s window, default stim root):
#   sbatch --export=ALL,MODEL_ID=whisper-small --array=0-47 scripts/slurm_mmn_extract_batch.sh
# Submit one wav2vec2 model (10 s window + 10 s stim root):
#   sbatch --export=ALL,MODEL_ID=wav2vec2-medium,MMN_STIM_ROOT=$PWD/outputs/mmn_stimuli_wav2vec2,WINDOW_DUR=10.0,WINDOW_STRIDE=10.0 \
#          --array=0-47 scripts/slurm_mmn_extract_batch.sh
#   Cap concurrent tasks with %, e.g. --array=0-47%12.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=mmn_extract_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=24:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/mmn_extract_batch_%A_%a.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/mmn_extract_batch_%A_%a.err

PROJECT_DIR="/work/upschrimpf1/sigfstea/multimodal-brain-scaling"
MODEL_ID="${MODEL_ID:-whisper-base}"
MMN_STIM_ROOT="${MMN_STIM_ROOT:-${PROJECT_DIR}/outputs/mmn_stimuli}"
WINDOW_DUR="${WINDOW_DUR:-30.0}"
WINDOW_STRIDE="${WINDOW_STRIDE:-30.0}"
LAYERS_CONFIG="configs/extraction/audio/${MODEL_ID//-/_}_layers.json"

# 24 Frequency method ids -> 48 method dirs (regular + counter), indexed by the array task id.
IDS=(09 10 12 17 18 19 20 21 27 28 29 30 31 32 33 37 43 44 53 55 60 72 74 75)
METHODS=()
for id in "${IDS[@]}"; do METHODS+=("method_${id}" "method_${id}_counter"); done   # 48

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
METHOD="${METHODS[$TASK_ID]}"
[ -n "$METHOD" ] || { echo "no method for array index $TASK_ID (expected 0-47)"; exit 1; }

DATA_ROOT="${MMN_STIM_ROOT}/${METHOD}"
if [ "$MODEL_ID" = "whisper-base" ]; then MMN_ROOT="outputs/features"; else MMN_ROOT="outputs/features/${MODEL_ID}-mmn"; fi
OUTPUT_DIR="${MMN_ROOT}/mmn-${METHOD}-delta-t"

cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs "$OUTPUT_DIR"

N_WAV=$(ls "${DATA_ROOT}"/*.wav 2>/dev/null | wc -l)
[ "$N_WAV" -gt 0 ] || { echo "no wavs in $DATA_ROOT"; exit 1; }
echo "Start: $(date) on $(hostname)  MODEL_ID=$MODEL_ID  METHOD=$METHOD  clips=$N_WAV  win=${WINDOW_DUR}/${WINDOW_STRIDE}"
echo "  data_root=$DATA_ROOT  -> $OUTPUT_DIR  (insilico --mmn_features_root $MMN_ROOT)"

# One call extracts all N_WAV clips of this method (model loaded once); save_every=1 checkpoints
# each clip so a timeout still leaves partial progress (the completeness check catches short dirs).
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} python -m mbs.extraction.extract_features_delta_t \
    --model_id              "$MODEL_ID" \
    --data_root             "$DATA_ROOT" \
    --target_feature_layers "$LAYERS_CONFIG" \
    --output_dir            "$OUTPUT_DIR" \
    --window_duration       "$WINDOW_DUR" \
    --window_stride         "$WINDOW_STRIDE" \
    --batch_t               16 \
    --t_stride              1 \
    --stim_start_idx        0 \
    --n_stimuli             "$N_WAV" \
    --save_every            1

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS $MODEL_ID/$METHOD" || echo "FAILED $MODEL_ID/$METHOD exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
