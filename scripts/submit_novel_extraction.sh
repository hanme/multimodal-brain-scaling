#!/bin/bash
# =============================================================================
# Submit the MMN delta_T extraction arrays for the novel grid: all 6 models, both stimulus
# families, split into as many arrays as the cluster's MaxArraySize requires.
#
# This is the expensive step (~123 CHF for Phase 1), and every one of its knobs is a way to
# spend that money on the wrong thing:
#   * a missing MMN_FEATURES_ROOT sends whisper-base at the literature features' bare
#     outputs/features root;
#   * a missing MMN_NAME_BY_STIM_ID makes the Phase-2 top-up re-extract all 16 clips per dir
#     rather than the 14 it is missing;
#   * wav2vec2 needs the 10 s stimulus root AND the 10 s window, or it silently extracts
#     whisper-length clips.
# Encoding them once here is the point: the per-model differences are in one place instead of
# in a doc the operator retypes twice.
#
#   DRY_RUN=1 scripts/submit_novel_extraction.sh          # print the sbatch lines, submit nothing
#   scripts/submit_novel_extraction.sh                    # Phase 1
#   METHOD_LIST=$PWD/outputs/novel_methods_phase2.txt scripts/submit_novel_extraction.sh
# =============================================================================
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
METHOD_LIST="${METHOD_LIST:-${PROJECT_DIR}/outputs/novel_methods_phase1.txt}"
FEATURES_TAG="${FEATURES_TAG:-novel}"
WHISPER_STIM="${WHISPER_STIM:-${PROJECT_DIR}/outputs/mmn_stimuli_novel}"
WAV2VEC2_STIM="${WAV2VEC2_STIM:-${PROJECT_DIR}/outputs/mmn_stimuli_novel_wav2vec2}"
MODELS="${MODELS:-whisper-tiny whisper-base whisper-small whisper-medium wav2vec2-medium wav2vec2-large}"
CONCURRENCY="${CONCURRENCY:-200}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$METHOD_LIST" ] || { echo "METHOD_LIST not found: $METHOD_LIST (run stage_novel_stimuli.sh)"; exit 1; }
N=$(grep -c . "$METHOD_LIST")
[ "$N" -gt 0 ] || { echo "METHOD_LIST is empty: $METHOD_LIST"; exit 1; }

# Cluster array cap. Query it rather than assume: a list longer than the cap is rejected at
# submission, and the novel grid's 1056 entries exceed the common default of 1001.
if MAX_ARRAY=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2}') \
   && [ -n "$MAX_ARRAY" ]; then
    echo "cluster MaxArraySize = $MAX_ARRAY"
else
    MAX_ARRAY="${MAX_ARRAY_FALLBACK:-1001}"
    echo "could not query MaxArraySize (not on a controller?) -- assuming $MAX_ARRAY"
fi
CHUNK=$((MAX_ARRAY - 1))                      # indices are 0..MaxArraySize-1
N_CHUNKS=$(( (N + CHUNK - 1) / CHUNK ))

echo "method list : $METHOD_LIST ($N conditions)"
echo "submissions : $N_CHUNKS array(s) per model x $(echo "$MODELS" | wc -w | tr -d ' ') models"
echo "features tag: -mmn-${FEATURES_TAG}"
[ "$DRY_RUN" = "1" ] && echo "DRY RUN -- nothing will be submitted"
echo

for model in $MODELS; do
    case "$model" in
        wav2vec2-*) stim="$WAV2VEC2_STIM"; win="10.0" ;;
        *)          stim="$WHISPER_STIM";  win="30.0" ;;
    esac
    [ -d "$stim" ] || { echo "stimulus root missing: $stim"; exit 1; }

    for ((c = 0; c < N_CHUNKS; c++)); do
        offset=$((c * CHUNK))
        last=$((N - offset - 1))
        [ "$last" -ge "$CHUNK" ] && last=$((CHUNK - 1))

        exports="ALL,MODEL_ID=${model},METHOD_LIST=${METHOD_LIST}"
        exports="${exports},MMN_STIM_ROOT=${stim},WINDOW_DUR=${win},WINDOW_STRIDE=${win}"
        exports="${exports},MMN_FEATURES_ROOT=outputs/features/${model}-mmn-${FEATURES_TAG}"
        exports="${exports},MMN_NAME_BY_STIM_ID=true,TASK_OFFSET=${offset}"

        cmd=(sbatch --export="$exports" --array="0-${last}%${CONCURRENCY}"
             scripts/slurm_mmn_extract_batch.sh)
        if [ "$DRY_RUN" = "1" ]; then
            # Plain echo, not printf %q: the point of a dry run is a line you can read and
            # paste, and %q escapes every comma in the --export list.
            echo "${cmd[@]}"
        else
            echo "submitting ${model} chunk $((c + 1))/${N_CHUNKS} (offset ${offset}, 0-${last})"
            "${cmd[@]}" || { echo "FAILED to submit ${model} chunk ${c}"; exit 1; }
        fi
    done
done

echo
[ "$DRY_RUN" = "1" ] || echo "submitted. Completeness check when done: every one of the "\
"$((N * $(echo "$MODELS" | wc -w | tr -d ' '))) feature dirs must hold its full clip count."
