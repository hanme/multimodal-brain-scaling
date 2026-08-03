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
METADATA_CSV="${METADATA_CSV:-data/metadata/novel_grid_frequency_metadata.csv}"
FEATURES_TAG="${FEATURES_TAG:-novel}"
WHISPER_STIM="${WHISPER_STIM:-${PROJECT_DIR}/outputs/mmn_stimuli_novel}"
WAV2VEC2_STIM="${WAV2VEC2_STIM:-${PROJECT_DIR}/outputs/mmn_stimuli_novel_wav2vec2}"
MODELS="${MODELS:-whisper-tiny whisper-base whisper-small whisper-medium wav2vec2-medium wav2vec2-large}"
CONCURRENCY="${CONCURRENCY:-200}"
DRY_RUN="${DRY_RUN:-0}"
# Clips per array task. 0 = one task per condition, extracting all its clips serially -- the
# historical layout. Wall clock is set by that serial loop, NOT by CONCURRENCY: a 24-condition x
# 7-model run is 168 tasks against %200, so the throttle never binds and ~15x of the parallelism
# already asked for sits idle. At ~13 min/clip, 16 clips serially is ~3.5 h; CLIPS_PER_TASK=4 is
# ~55 min for 4x the model loads, which is the sweet spot before load overhead dominates.
CLIPS_PER_TASK="${CLIPS_PER_TASK:-0}"
CLIPS_PER_METHOD="${CLIPS_PER_METHOD:-16}"

[ -f "$METHOD_LIST" ] || { echo "METHOD_LIST not found: $METHOD_LIST (run slurm_stage_novel_stimuli.sh)"; exit 1; }
N=$(grep -c . "$METHOD_LIST")
[ "$N" -gt 0 ] || { echo "METHOD_LIST is empty: $METHOD_LIST"; exit 1; }

# The method list and the metadata CSV MUST describe the same conditions. If the grid is
# rebuilt and the stimuli are not re-staged, the ids silently remap: method_1001 is one
# frequency pair in the staged audio and a different one in the CSV that scores it. Nothing
# downstream catches that -- the features are internally consistent, just describing the wrong
# pairs -- so refuse here, where it is still free.
[ -f "$METADATA_CSV" ] || { echo "metadata CSV not found: $METADATA_CSV"; exit 1; }
expected=$(awk -F, 'NR>1 && $4=="Frequency" {print "method_"$2; print "method_"$2"_counter"}' \
           "$METADATA_CSV" | sort)
actual=$(grep . "$METHOD_LIST" | sort)
if [ "$expected" != "$actual" ]; then
    n_exp=$(echo "$expected" | grep -c .)
    echo "REFUSING: METHOD_LIST does not match METADATA_CSV."
    echo "  $METHOD_LIST      : $N conditions"
    echo "  $METADATA_CSV : $n_exp conditions"
    echo "  The grid was rebuilt without re-staging the stimuli, so method ids refer to"
    echo "  different frequency pairs in each. Re-run, in order:"
    echo "    python scripts/build_novel_grid_csv.py"
    echo "    rm -rf outputs/stim_gen_novel outputs/mmn_stimuli_novel outputs/mmn_stimuli_novel_wav2vec2"
    echo "    rm -f  $METHOD_LIST"
    echo "    sbatch scripts/slurm_generate_stimuli.sh ...   # then slurm_stage_novel_stimuli.sh"
    exit 1
fi

# ...and the staged audio must actually be there for those conditions.
for probe in "$(head -1 "$METHOD_LIST")" "$(tail -1 "$METHOD_LIST")"; do
    for root in "$WHISPER_STIM" "$WAV2VEC2_STIM"; do
        [ -d "$root/$probe" ] || {
            echo "REFUSING: $root/$probe is missing -- the method list names conditions that"
            echo "  were never staged. Run scripts/slurm_stage_novel_stimuli.sh first."
            exit 1; }
    done
done

# Cluster array cap. Query it rather than assume: a list longer than the cap is rejected at
# submission, and the novel grid's 1806 entries exceed the common default of 1001.
if MAX_ARRAY=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/ /,"",$2); print $2}') \
   && [ -n "$MAX_ARRAY" ]; then
    echo "cluster MaxArraySize = $MAX_ARRAY"
else
    MAX_ARRAY="${MAX_ARRAY_FALLBACK:-1001}"
    echo "could not query MaxArraySize (not on a controller?) -- assuming $MAX_ARRAY"
fi
CHUNK=$((MAX_ARRAY - 1))                      # indices are 0..MaxArraySize-1

# The array indexes (condition, clip-chunk) pairs, so size it for the expanded space.
if [ "$CLIPS_PER_TASK" -gt 0 ]; then
    PER_METHOD=$(( (CLIPS_PER_METHOD + CLIPS_PER_TASK - 1) / CLIPS_PER_TASK ))
else
    PER_METHOD=1
fi
N_TASKS=$(( N * PER_METHOD ))
N_CHUNKS=$(( (N_TASKS + CHUNK - 1) / CHUNK ))

echo "method list : $METHOD_LIST ($N conditions)"
if [ "$CLIPS_PER_TASK" -gt 0 ]; then
    echo "clip split  : $CLIPS_PER_TASK clips/task x $PER_METHOD chunks per condition "\
"(declared $CLIPS_PER_METHOD clips/condition) -> $N_TASKS tasks per model"
else
    echo "clip split  : none -- one task per condition, all clips serial "\
"(set CLIPS_PER_TASK=4 to parallelise; the %${CONCURRENCY} throttle is otherwise slack)"
fi
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
        last=$((N_TASKS - offset - 1))
        [ "$last" -ge "$CHUNK" ] && last=$((CHUNK - 1))

        exports="ALL,MODEL_ID=${model},METHOD_LIST=${METHOD_LIST}"
        exports="${exports},MMN_STIM_ROOT=${stim},WINDOW_DUR=${win},WINDOW_STRIDE=${win}"
        exports="${exports},MMN_FEATURES_ROOT=outputs/features/${model}-mmn-${FEATURES_TAG}"
        exports="${exports},MMN_NAME_BY_STIM_ID=true,TASK_OFFSET=${offset}"
        exports="${exports},CLIPS_PER_TASK=${CLIPS_PER_TASK},CLIPS_PER_METHOD=${CLIPS_PER_METHOD}"

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
