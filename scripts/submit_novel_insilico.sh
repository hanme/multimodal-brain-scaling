#!/bin/bash
# =============================================================================
# Submit the electrode-level in-silico MMN for the novel grid: all 6 models, one job each.
#
# The counterpart to submit_novel_extraction.sh, and it exists for the same reason plus a
# sharper one. slurm_insilico_mmn_electrodes.sh defaults every path to the LITERATURE run, so a
# screen that forgets DATA_DIR does not fail -- it overwrites
# outputs/insilico_mmn_predictions/<model>/electrode_predictions__*.h5, which is the comparison
# baseline this whole search is measured against, and has no backup. Retyping five env vars per
# model, twice (Phase 1 and Phase 2), is not worth that risk.
#
# --save_plots false is passed by default: at 1806 conditions the per-method figures are ~2
# matplotlib renders each, one a ~50-subplot montage, and would dominate the walltime. The
# prediction h5 is written either way and is what all downstream scoring reads. Re-run the
# winners with SAVE_PLOTS=true METHODS=<consensus dirs> for figures.
#
#   DRY_RUN=1 scripts/submit_novel_insilico.sh              # print, submit nothing
#   scripts/submit_novel_insilico.sh                        # Phase 1
#   METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv \
#   PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_novel_phase2 \
#       scripts/submit_novel_insilico.sh                    # Phase 2
#   SAVE_PLOTS=true METHODS=method_1042,method_1042_counter \
#   PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_novel_figs \
#       scripts/submit_novel_insilico.sh                    # figures for the consensus set
# =============================================================================
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
METADATA_CSV="${METADATA_CSV:-data/metadata/novel_grid_frequency_metadata.csv}"
FEATURES_TAG="${FEATURES_TAG:-novel}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-outputs/insilico_mmn_predictions_novel}"
FIGURES_ROOT="${FIGURES_ROOT:-outputs/figures/insilico_mmn_electrodes_novel}"
WHISPER_STIM="${WHISPER_STIM:-outputs/mmn_stimuli_novel}"
WAV2VEC2_STIM="${WAV2VEC2_STIM:-outputs/mmn_stimuli_novel_wav2vec2}"
MODELS="${MODELS:-whisper-tiny whisper-base whisper-small whisper-medium wav2vec2-medium wav2vec2-large}"
SAVE_PLOTS="${SAVE_PLOTS:-false}"
METHODS="${METHODS:-all}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$METADATA_CSV" ] || { echo "metadata CSV not found: $METADATA_CSV"; exit 1; }

# The guard this script exists for. Anything under the literature predictions root would clobber
# the committed baseline; refuse rather than warn.
case "$PREDICTIONS_ROOT" in
    outputs/insilico_mmn_predictions|outputs/insilico_mmn_predictions/*)
        echo "REFUSING: PREDICTIONS_ROOT=$PREDICTIONS_ROOT is the literature run's root."
        echo "  Writing there would overwrite the committed literature predictions this search"
        echo "  is compared against. Use e.g. outputs/insilico_mmn_predictions_novel."
        exit 1 ;;
esac

# The features being scored must cover the conditions this CSV names. If the grid was rebuilt
# after extraction, method ids refer to different frequency pairs in the features than in the
# CSV, and the scoring is silently wrong rather than absent.
first_id=$(awk -F, 'NR>1 && $4=="Frequency" {print $2; exit}' "$METADATA_CSV")
last_id=$(awk -F, 'NR>1 && $4=="Frequency" {id=$2} END {print id}' "$METADATA_CSV")
for model in $MODELS; do
    root="outputs/features/${model}-mmn-${FEATURES_TAG}"
    [ -d "$root" ] || continue          # not extracted yet; the driver reports it per method
    for id in "$first_id" "$last_id"; do
        [ -d "$root/mmn-method_${id}-delta-t" ] || {
            echo "REFUSING: $root/mmn-method_${id}-delta-t is missing."
            echo "  $METADATA_CSV names method_${id}, but the extracted features do not have it."
            echo "  The grid was rebuilt after extraction, so ids no longer line up. Re-extract"
            echo "  against the current grid before scoring."
            exit 1; }
    done
done

N_COND=$(awk -F, 'NR>1 && $4=="Frequency" {n++} END {print 2*n}' "$METADATA_CSV")
echo "metadata    : $METADATA_CSV ($N_COND conditions incl. counter)"
echo "features    : outputs/features/<model>-mmn-${FEATURES_TAG}"
echo "predictions : ${PREDICTIONS_ROOT}/<model>"
echo "save_plots  : $SAVE_PLOTS   methods: $METHODS"
[ "$DRY_RUN" = "1" ] && echo "DRY RUN -- nothing will be submitted"
echo

for model in $MODELS; do
    case "$model" in
        wav2vec2-*) stim="$WAV2VEC2_STIM" ;;
        *)          stim="$WHISPER_STIM"  ;;
    esac
    [ -d "$stim" ] || { echo "stimulus root missing: $stim"; exit 1; }

    exports="ALL,MODEL_ID=${model},METADATA_CSV=${METADATA_CSV}"
    exports="${exports},MMN_FEATURES_ROOT=outputs/features/${model}-mmn-${FEATURES_TAG}"
    exports="${exports},STIM_ROOT=${stim}"
    exports="${exports},DATA_DIR=${PREDICTIONS_ROOT}/${model}"
    exports="${exports},OUT_DIR=${FIGURES_ROOT}/${model}"

    cmd=(sbatch --export="$exports" scripts/slurm_insilico_mmn_electrodes.sh
         --save_plots "$SAVE_PLOTS")
    [ "$METHODS" != "all" ] && cmd+=(--methods "$METHODS")

    if [ "$DRY_RUN" = "1" ]; then
        echo "${cmd[@]}"
    else
        echo "submitting $model"
        "${cmd[@]}" || { echo "FAILED to submit $model"; exit 1; }
    fi
done

echo
[ "$DRY_RUN" = "1" ] || echo "submitted $(echo "$MODELS" | wc -w | tr -d ' ') jobs. \
Each writes ${PREDICTIONS_ROOT}/<model>/electrode_predictions__<layer>.h5 with $N_COND groups."
