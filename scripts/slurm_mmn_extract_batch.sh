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
#
# Arbitrary condition sets (e.g. the 992-dir novel frequency grid) come from a METHOD_LIST file
# of method-dir names, one per line, indexed by SLURM_ARRAY_TASK_ID + TASK_OFFSET. TASK_OFFSET
# exists so a list longer than the cluster's MaxArraySize can be submitted as several arrays:
#   sbatch --export=ALL,MODEL_ID=whisper-small,METHOD_LIST=$PWD/outputs/novel_methods_phase1.txt,\
# MMN_STIM_ROOT=$PWD/outputs/mmn_stimuli_novel,MMN_FEATURES_ROOT=outputs/features/whisper-small-mmn-novel,\
# MMN_NAME_BY_STIM_ID=true --array=0-495 scripts/slurm_mmn_extract_batch.sh
#   sbatch ... ,TASK_OFFSET=496 --array=0-495 scripts/slurm_mmn_extract_batch.sh
#
# CLIPS_PER_TASK splits a method's clips over several tasks so they run in PARALLEL rather than
# serially, which is what actually sets the wall clock (the array %throttle is usually slack).
# The array must be sized for the expanded index space -- submit_novel_extraction.sh does that
# arithmetic for you, so prefer it over hand-submitting:
#   sbatch --export=ALL,MODEL_ID=whisper-large,CLIPS_PER_TASK=4,MMN_NAME_BY_STIM_ID=true,... \
#          --array=0-191%200 scripts/slurm_mmn_extract_batch.sh     # 48 methods x 4 chunks
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

# Condition list: from METHOD_LIST if given, else the 24 literature Frequency method ids ->
# 48 method dirs (regular + counter). Either way it is indexed by the array task id.
if [ -n "${METHOD_LIST:-}" ]; then
    [ -f "$METHOD_LIST" ] || { echo "METHOD_LIST not found: $METHOD_LIST"; exit 1; }
    # mapfile is unavailable on some login shells; read is portable and handles a missing
    # trailing newline.
    METHODS=()
    while IFS= read -r line || [ -n "$line" ]; do
        [ -n "$line" ] && METHODS+=("$line")
    done < "$METHOD_LIST"
else
    IDS=(09 10 12 17 18 19 20 21 27 28 29 30 31 32 33 37 43 44 53 55 60 72 74 75)
    METHODS=()
    for id in "${IDS[@]}"; do METHODS+=("method_${id}" "method_${id}_counter"); done   # 48
fi
N_METHODS=${#METHODS[@]}

# CLIPS_PER_TASK splits a method's clips across SEVERAL array tasks instead of extracting them all
# serially in one. Wall clock is set by the serial loop inside a task, not by the array throttle:
# 24 conditions x 7 models is only 168 tasks against the submitter's %200, so the default leaves
# most of the requested parallelism idle. At ~13 min/clip, 16 clips serially is ~3.5 h; chunks of 4
# are ~55 min for 4x the model loads.
#
# 0 (the default) keeps the historical one-task-per-method layout byte-for-byte.
#
# The index space becomes (method, clip-chunk) pairs, ordered method-major, so TASK_OFFSET chunking
# against MaxArraySize still works unchanged. CLIPS_PER_METHOD is the DECLARED clip count -- the
# index->method map has to be computable before the method dir is known, so it cannot be discovered
# from disk. Tasks whose chunk starts past a short dir's actual clip count exit 0 as a no-op.
CLIPS_PER_TASK="${CLIPS_PER_TASK:-0}"
CLIPS_PER_METHOD="${CLIPS_PER_METHOD:-16}"

TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))

if [ "$CLIPS_PER_TASK" -gt 0 ]; then
    # Chunked tasks share one output directory, so they must not be able to collide. Stimulus-id
    # naming gives each clip its own h5 and makes that safe by construction; the legacy
    # start/batch names only happen not to collide, and depend on glob order for alignment.
    if [ "${MMN_NAME_BY_STIM_ID:-false}" != "true" ]; then
        echo "REFUSING: CLIPS_PER_TASK=$CLIPS_PER_TASK needs MMN_NAME_BY_STIM_ID=true."
        echo "  Several tasks write into ${MMN_FEATURES_ROOT:-<root>}/mmn-<method>-delta-t at once;"
        echo "  only stimulus-id naming guarantees one file per clip and no clobbering."
        exit 1
    fi
    N_CHUNKS_PER_METHOD=$(( (CLIPS_PER_METHOD + CLIPS_PER_TASK - 1) / CLIPS_PER_TASK ))
    METHOD_IDX=$(( TASK_ID / N_CHUNKS_PER_METHOD ))
    CHUNK_IDX=$(( TASK_ID % N_CHUNKS_PER_METHOD ))
    STIM_START=$(( CHUNK_IDX * CLIPS_PER_TASK ))
    N_STIM=$CLIPS_PER_TASK
    N_TASKS=$(( N_METHODS * N_CHUNKS_PER_METHOD ))
else
    METHOD_IDX=$TASK_ID
    CHUNK_IDX=0
    STIM_START=0
    N_STIM=0                      # resolved to the dir's full clip count once it is known
    N_TASKS=$N_METHODS
fi

# Guard explicitly: a bash array index past the end yields "" but a NEGATIVE one silently wraps
# to the end of the list and would extract the wrong condition.
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$N_TASKS" ]; then
    if [ "$CLIPS_PER_TASK" -gt 0 ]; then
        echo "no method for index $TASK_ID (${N_METHODS} methods x ${N_CHUNKS_PER_METHOD} chunks = ${N_TASKS} tasks, valid 0-$((N_TASKS-1)))"
    else
        echo "no method for index $TASK_ID (list has $N_METHODS entries, valid 0-$((N_METHODS-1)))"
    fi
    exit 1
fi
METHOD="${METHODS[$METHOD_IDX]}"

DATA_ROOT="${MMN_STIM_ROOT}/${METHOD}"
# whisper-base historically wrote to the bare outputs/features; MMN_FEATURES_ROOT overrides that
# (and every other model's default) so a separate screen -- e.g. the novel grid -- can be routed
# to its own root without colliding with the committed literature features.
if [ -n "${MMN_FEATURES_ROOT:-}" ]; then MMN_ROOT="$MMN_FEATURES_ROOT"
elif [ "$MODEL_ID" = "whisper-base" ]; then MMN_ROOT="outputs/features"
else MMN_ROOT="outputs/features/${MODEL_ID}-mmn"; fi
OUTPUT_DIR="${MMN_ROOT}/mmn-${METHOD}-delta-t"

cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs "$OUTPUT_DIR"

N_WAV=$(ls "${DATA_ROOT}"/*.wav 2>/dev/null | wc -l)
[ "$N_WAV" -gt 0 ] || { echo "no wavs in $DATA_ROOT"; exit 1; }

# Resolve the clip slice now that the real count is known. Unchunked means "all of them".
if [ "$CLIPS_PER_TASK" -le 0 ]; then
    N_STIM=$N_WAV
elif [ "$STIM_START" -ge "$N_WAV" ]; then
    # The dir holds fewer clips than CLIPS_PER_METHOD declared. Not an error here -- a short dir is
    # what check_novel_features.py exists to catch -- but this task has nothing to do.
    echo "no clips for $METHOD chunk $CHUNK_IDX: starts at $STIM_START but the dir holds $N_WAV"
    echo "  (CLIPS_PER_METHOD=$CLIPS_PER_METHOD declared; run the completeness check if unexpected)"
    exit 0
elif [ $(( STIM_START + N_STIM )) -gt "$N_WAV" ]; then
    N_STIM=$(( N_WAV - STIM_START ))
fi

echo "Start: $(date) on $(hostname)  MODEL_ID=$MODEL_ID  METHOD=$METHOD (task $TASK_ID of $N_TASKS)  clips=$N_WAV  win=${WINDOW_DUR}/${WINDOW_STRIDE}"
echo "  clip slice: $N_STIM of $N_WAV from index $STIM_START (chunk $CHUNK_IDX, CLIPS_PER_TASK=$CLIPS_PER_TASK)"
echo "  data_root=$DATA_ROOT  -> $OUTPUT_DIR  (insilico --mmn_features_root $MMN_ROOT)"

# One call extracts all N_WAV clips of this method (model loaded once); save_every=1 checkpoints
# each clip so a timeout still leaves partial progress (the completeness check catches short dirs).
#
# MMN_NAME_BY_STIM_ID=true makes this idempotent: each h5 is named after its clip, so a
# resubmitted task re-does only what is missing, and clips ADDED to a method dir later (the novel
# search's Phase 2, which grows each dir from 2 to 16 wavs) cost only the new clips and cannot
# clobber the ones already extracted. It defaults to FALSE because the committed literature
# features were written with the legacy start/batch names, and a directory holding both schemes
# would be double-loaded by load_layer_features. Set it for new screens (which own their root via
# MMN_FEATURES_ROOT), not for topping up the literature ones.
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} python -m mbs.extraction.extract_features_delta_t \
    --model_id              "$MODEL_ID" \
    --data_root             "$DATA_ROOT" \
    --target_feature_layers "$LAYERS_CONFIG" \
    --output_dir            "$OUTPUT_DIR" \
    --window_duration       "$WINDOW_DUR" \
    --window_stride         "$WINDOW_STRIDE" \
    --batch_t               16 \
    --t_stride              1 \
    --stim_start_idx        "$STIM_START" \
    --n_stimuli             "$N_STIM" \
    --save_every            1 \
    --name_by_stim_id       "${MMN_NAME_BY_STIM_ID:-false}" \
    --overwrite             "${MMN_OVERWRITE:-false}"

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS $MODEL_ID/$METHOD" || echo "FAILED $MODEL_ID/$METHOD exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
