#!/bin/bash
# =============================================================================
# SLURM Array: delta_T feature extraction for MMN stimuli (in-silico MMN POC)
# =============================================================================
# One task per stimulus wav in outputs/mmn_stimuli/<METHOD>/ (1 standard + 15 deviants).
# delta_T (causal) features, whisper-base, all blocks, t_stride=1 (~13 min/stim).
# Parallel over the 16 stimuli -> ~13 min wall.
#
# Submit:   sbatch --array=0-15 scripts/slurm_mmn_extract.sh
#           sbatch --export=ALL,MMN_METHOD=method_37 --array=0-15 scripts/slurm_mmn_extract.sh
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=mmn_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=01:30:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/mmn_extract_%A_%a.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/mmn_extract_%A_%a.err

PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
METHOD="${MMN_METHOD:-method_09}"
MODEL_ID="${MODEL_ID:-whisper-base}"
DATA_ROOT="${PROJECT_DIR}/outputs/mmn_stimuli/${METHOD}"
LAYERS_CONFIG="configs/extraction/audio/${MODEL_ID//-/_}_layers.json"
# whisper-base keeps its historical path (outputs/features/mmn-<method>-delta-t); any other model
# gets a model-scoped root so its MMN features don't collide. Point insilico_mmn.py's
# --mmn_features_root at the dir that CONTAINS mmn-<method>-delta-t (printed below).
if [ "$MODEL_ID" = "whisper-base" ]; then MMN_ROOT="outputs/features"; else MMN_ROOT="outputs/features/${MODEL_ID}-mmn"; fi
OUTPUT_DIR="${MMN_ROOT}/mmn-${METHOD}-delta-t"

cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs "$OUTPUT_DIR"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "MMN extract: $MODEL_ID / $METHOD  stimulus index ${TASK_ID} of $(ls ${DATA_ROOT}/*.wav | wc -l)"
echo "  -> $OUTPUT_DIR   (insilico --mmn_features_root $MMN_ROOT)"

# window_duration 30 / stride 30 -> exactly one 30 s segment per clip (matches Broderick training window)
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} python -m mbs.extraction.extract_features_delta_t \
    --model_id              "$MODEL_ID" \
    --data_root             "$DATA_ROOT" \
    --target_feature_layers "$LAYERS_CONFIG" \
    --output_dir            "$OUTPUT_DIR" \
    --window_duration       30.0 \
    --window_stride         30.0 \
    --batch_t               16 \
    --t_stride              1 \
    --stim_start_idx        "$TASK_ID" \
    --n_stimuli             1 \
    --save_every            1

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS task=${TASK_ID}" || echo "FAILED task=${TASK_ID} exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
