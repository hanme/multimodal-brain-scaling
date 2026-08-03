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
#
# A separate screen (e.g. the trailing-floor soafix re-run) MUST redirect all four paths, or it
# will read the literature features and OVERWRITE the committed literature predictions:
#   sbatch --export=ALL,MODEL_ID=whisper-tiny,\
# MMN_FEATURES_ROOT=outputs/features/whisper-tiny-mmn-soafix,\
# STIM_ROOT=outputs/mmn_stimuli_soafix,\
# DATA_DIR=outputs/insilico_mmn_predictions_soafix/whisper-tiny \
#          scripts/slurm_insilico_mmn.sh
# Redirecting the inputs while leaving DATA_DIR at its default is refused (see the guard below):
# that combination scores new features straight over the committed baseline, which has no backup.
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
mkdir -p logs      # DATA_DIR/OUT_DIR are created below, once any redirect has been applied

# committed §1.5 mTRF-parcels layer per model (do not re-select)
declare -A MTRF_PARCELS_LAYER=(
    [whisper-tiny]=blocks.0  [whisper-base]=blocks.0
    [whisper-small]=blocks.3 [whisper-medium]=blocks.11
    [whisper-large]=blocks.21
    # wav2vec2 layers from the D2 sweep (outputs/results/eeg_mapping/{model}__parcels__D2.json).
    [wav2vec2-medium]=encoder.layers.2 [wav2vec2-large]=encoder.layers.12
)
MODELS=(whisper-tiny whisper-base whisper-small whisper-medium whisper-large wav2vec2-medium wav2vec2-large)

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    MODEL_ID="${MODELS[$SLURM_ARRAY_TASK_ID]}"
else
    MODEL_ID="${MODEL_ID:-whisper-base}"
fi
LAYER="${MTRF_PARCELS_LAYER[$MODEL_ID]}"
[ -n "$LAYER" ] || { echo "unknown MODEL_ID=$MODEL_ID"; exit 1; }
TRAIN_FEATURES="outputs/features/${MODEL_ID}-delta-t-surprisal/merged"
# whisper-base keeps its historical MMN-features root (outputs/features); every other model's
# MMN delta_T features live under a model-scoped root so they don't collide. MMN_FEATURES_ROOT
# overrides both, so a separate screen can point at its own features.
if [ -n "${MMN_FEATURES_ROOT:-}" ]; then MMN_ROOT="$MMN_FEATURES_ROOT"
elif [ "$MODEL_ID" = "whisper-base" ]; then MMN_ROOT="outputs/features"
else MMN_ROOT="outputs/features/${MODEL_ID}-mmn"; fi
# wav2vec2 was mapped on 10 s D2 windows (surprisal_10s.h5); whisper on 30 s (surprisal_30s.h5).
case "$MODEL_ID" in wav2vec2-*) TRAIN_NEURAL="outputs/neural_data/surprisal_10s.h5";; *) TRAIN_NEURAL="outputs/neural_data/surprisal_30s.h5";; esac
# Capture what the CALLER set, before defaults below make everything look explicit -- the guard
# needs to distinguish "redirected the inputs and forgot the output" from a plain literature run.
REDIRECTED_IN="${MMN_FEATURES_ROOT:-}${STIM_ROOT:-}"
# wav2vec2 MMN clips are 10 s and staged in a separate root; whisper clips are 30 s in the default.
# stim_dir feeds the final-tone-onset used for time-locking, so it MUST match the feature clips.
if [ -z "${STIM_ROOT:-}" ]; then
    case "$MODEL_ID" in wav2vec2-*) STIM_ROOT="outputs/mmn_stimuli_wav2vec2";; *) STIM_ROOT="outputs/mmn_stimuli";; esac
fi
# Where the predictions h5 and figures land.
DATA_DIR="${DATA_DIR:-outputs/insilico_mmn_predictions/${MODEL_ID}}"
OUT_DIR="${OUT_DIR:-outputs/figures/insilico_mmn/${MODEL_ID}}"
# Condition registry + SOA/duration lookups all come from this one CSV.
METADATA_CSV="${METADATA_CSV:-data/metadata/literature_frequency_intensity_duration_metadata.csv}"

# Redirecting the inputs but not the output is the one combination that destroys data: it scores a
# different screen's features straight over the committed literature predictions, which have no
# backup. The electrode wrapper leaves this to its submitter; here it is enforced in-script.
if [ -n "$REDIRECTED_IN" ]; then
    case "$DATA_DIR" in
        outputs/insilico_mmn_predictions|outputs/insilico_mmn_predictions/*)
            echo "REFUSING: MMN_FEATURES_ROOT/STIM_ROOT were redirected but DATA_DIR is still"
            echo "  $DATA_DIR -- the committed literature predictions root, which has no backup."
            echo "  Set DATA_DIR to a sibling, e.g. outputs/insilico_mmn_predictions_soafix/${MODEL_ID}."
            exit 1 ;;
    esac
fi

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "Start: $(date) on $(hostname)   MODEL_ID=$MODEL_ID  layer=$LAYER  features=$TRAIN_FEATURES  neural=$TRAIN_NEURAL  stim=$STIM_ROOT  mmn_root=$MMN_ROOT"
echo "  metadata=$METADATA_CSV  data_dir=$DATA_DIR  out_dir=$OUT_DIR  extra args: $*"

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python scripts/insilico_mmn.py \
    --layer "$LAYER" \
    --train_features "$TRAIN_FEATURES" \
    --train_neural "$TRAIN_NEURAL" \
    --mmn_features_root "$MMN_ROOT" \
    --stimuli_root "$STIM_ROOT" \
    --metadata_csv "$METADATA_CSV" \
    --lag_max_ms 800 \
    --baseline_start_mult -3.0 --baseline_end_mult 0.0 \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    "$@"
EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS ${MODEL_ID}/parcels" || echo "FAILED ${MODEL_ID}/parcels exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
