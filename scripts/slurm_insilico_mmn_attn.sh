#!/bin/bash
# =============================================================================
# SLURM (CPU, jed standard): Method B in-silico MMN + fit-quality figures from a trained
#   attention-encoder checkpoint (outputs/results/<model>-probe-group-d2-<level>/model__<layer>.pt,
#   produced by kuma_probe_d2_final.sh). One task = one (model, level, method) triple.
#
# IMPORTANT: insilico_mmn_attn.py opens its output h5 in truncate ("w") mode on every call, so
# every (model, level, method) triple MUST get its own --data_dir/--out_dir, or later runs
# silently clobber earlier ones sharing the same dir. This script enforces that automatically.
#
# 160 canonical slots = 4 models x 2 levels x 20 methods (10 regular + 10 counter), flattened as:
#   model = TASK / 40 ; level = (TASK / 20) % 2 ; method = METHODS[TASK % 20]
#
# Regular methods:    slots 0–9 (parcels) and 20–29 (electrodes) per model block
# Counter methods:    slots 10–19 (parcels) and 30–39 (electrodes) per model block
#
# Submit only the counter slots (new):
#   sbatch --array=10-19,30-39,50-59,70-79,90-99,110-119,130-139,150-159 scripts/slurm_insilico_mmn_attn.sh
# Submit all 160:
#   sbatch --array=0-159 scripts/slurm_insilico_mmn_attn.sh
# Or a single triple via env vars (no array):
#   sbatch --export=ALL,MODEL_ID=whisper-tiny,LEVEL=parcels,METHOD=method_37 scripts/slurm_insilico_mmn_attn.sh
# Extra args after the script name are forwarded to insilico_mmn_attn.py.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=insilico_mmn_attn
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5500M
#SBATCH --time=00:30:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/insilico_mmn_attn_%A_%a.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/insilico_mmn_attn_%A_%a.err

PROJECT_DIR="/work/upschrimpf1/sigfstea/multimodal-brain-scaling"
cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs outputs/figures/insilico_mmn outputs/insilico_mmn_predictions

MODELS=(whisper-tiny whisper-base whisper-small whisper-medium)
LEVELS=(parcels electrodes)
# same order as the METHODS registry in insilico_mmn.py (regular first, then counterbalanced)
METHODS=(method_75 method_74 method_72 method_60 method_53 method_55 method_37 method_43 method_44 method_27
         method_75_counter method_74_counter method_72_counter method_60_counter method_53_counter
         method_55_counter method_37_counter method_43_counter method_44_counter method_27_counter)

# committed §1.5 encoder layer per (model, level) — do not re-select
declare -A ENC_LAYER=(
    [whisper-tiny_parcels]=blocks.3    [whisper-tiny_electrodes]=blocks.3
    [whisper-base_parcels]=blocks.2    [whisper-base_electrodes]=blocks.0
    [whisper-small_parcels]=blocks.10  [whisper-small_electrodes]=blocks.10
    [whisper-medium_parcels]=blocks.4  [whisper-medium_electrodes]=blocks.3
)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK=$SLURM_ARRAY_TASK_ID
    MODEL_ID="${MODELS[$((TASK / 40))]}"
    LEVEL="${LEVELS[$((TASK / 20 % 2))]}"
    METHOD="${METHODS[$((TASK % 20))]}"
else
    MODEL_ID="${MODEL_ID:-whisper-base}"
    LEVEL="${LEVEL:-parcels}"
    METHOD="${METHOD:-method_37}"
fi

LAYER="${ENC_LAYER[${MODEL_ID}_${LEVEL}]}"
[ -n "$LAYER" ] || { echo "unknown MODEL_ID/LEVEL combo: ${MODEL_ID}/${LEVEL}"; exit 1; }

CHECKPOINT="outputs/results/${MODEL_ID}-probe-group-d2-${LEVEL}/model__${LAYER}.pt"
FEATURES_DIR="outputs/features/${MODEL_ID}-delta-t-surprisal/merged"
NEURAL="outputs/neural_data/surprisal_30s.h5"
if [ "$MODEL_ID" = "whisper-base" ]; then MMN_FEATURES_ROOT="outputs/features"; else MMN_FEATURES_ROOT="outputs/features/${MODEL_ID}-mmn"; fi
OUT_DIR="outputs/figures/insilico_mmn/${MODEL_ID}-${LEVEL}/${METHOD}"
DATA_DIR="outputs/insilico_mmn_predictions/${MODEL_ID}-${LEVEL}/${METHOD}"

[ -f "$CHECKPOINT" ] || { echo "🛑 missing checkpoint $CHECKPOINT — run kuma_probe_d2_final.sh first"; exit 1; }

echo "Start: $(date) on $(hostname)   MODEL_ID=$MODEL_ID  LEVEL=$LEVEL  METHOD=$METHOD  layer=$LAYER"
echo "  checkpoint=$CHECKPOINT  out_dir=$OUT_DIR  data_dir=$DATA_DIR"

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python scripts/insilico_mmn_attn.py \
    --checkpoint "$CHECKPOINT" \
    --mmn_features_root "$MMN_FEATURES_ROOT" \
    --method "$METHOD" \
    --features_dir "$FEATURES_DIR" \
    --neural "$NEURAL" \
    --baseline_start_mult -3.0 --baseline_end_mult 0.0 \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    "$@"
EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS ${MODEL_ID}/${LEVEL}/${METHOD}" || echo "FAILED ${MODEL_ID}/${LEVEL}/${METHOD} exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
