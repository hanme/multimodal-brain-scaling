#!/bin/bash
# =============================================================================
# SLURM Array Job: Delta-T (causal) feature extraction on D2 (Cortical Surprisal)
# for the NEW models — whisper-large and wav2vec2-medium/large.
# =============================================================================
#
# One script for the D2 model->EEG mapping features of the models added by the
# "enable large + wav2vec2" prereq. It differs from slurm_extract_delta_t_generic.sh
# (which does D1 / Broderick) only in the data_root + output naming:
#   * DATA_ROOT   = the Cortical Surprisal audiobook WAVs (source of surprisal_30s.h5),
#                   recovered from the stimulus_ids baked into surprisal_30s.h5.
#   * OUTPUT_BASE = outputs/features/${MODEL_ID}-delta-t-surprisal  (merge -> merged/).
#
# Per-model windowing (set via env; the extractor auto-picks mel vs raw-waveform by model_id):
#   whisper-large   :  WINDOW_DUR=30  WINDOW_STRIDE=10   -> maps against surprisal_30s.h5 (EXISTS)
#   wav2vec2-medium :  WINDOW_DUR=10  WINDOW_STRIDE=5    -> maps against surprisal_10s.h5 (MUST BE BUILT)
#   wav2vec2-large  :  WINDOW_DUR=10  WINDOW_STRIDE=5    -> maps against surprisal_10s.h5 (MUST BE BUILT)
#
# ⚠️  The wav2vec2 (10 s) features need a 10 s / 5 s EEG target (surprisal_10s.h5) for the
#     mapping sweep — surprisal_30s.h5 will NOT align. See the handoff runbook.
#
# GPU vs CPU: the extractor auto-selects the device and does NOT hard-fail without a GPU.
# For CPU (jed partition), comment out --partition/--gres and raise --cpus-per-task.
#
# Usage (from the repo root, after `source env.sh`):
#   # size the array first (prints "Stimuli: N (idx ...)" = window count):
#   export MODEL_ID=whisper-large WINDOW_DUR=30 WINDOW_STRIDE=10
#   python -m mbs.extraction.extract_features_delta_t --model_id "$MODEL_ID" \
#     --data_root "$DATA_ROOT" \
#     --target_feature_layers "configs/extraction/audio/${MODEL_ID//-/_}_layers.json" \
#     --output_dir /tmp/probe --window_duration "$WINDOW_DUR" --window_stride "$WINDOW_STRIDE" --n_stimuli 1
#   # submit (CHUNK_SIZE * (max array idx + 1) must cover the window count):
#   export MODEL_ID=whisper-large WINDOW_DUR=30 WINDOW_STRIDE=10 CHUNK_SIZE=16
#   sbatch --array=0-19 --export=ALL scripts/slurm_extract_delta_t_d2.sh
#   export MODEL_ID=wav2vec2-medium WINDOW_DUR=10 WINDOW_STRIDE=5 CHUNK_SIZE=32
#   sbatch --array=0-19 --export=ALL scripts/slurm_extract_delta_t_d2.sh
#   export MODEL_ID=wav2vec2-large  WINDOW_DUR=10 WINDOW_STRIDE=5 CHUNK_SIZE=32
#   sbatch --array=0-19 --export=ALL scripts/slurm_extract_delta_t_d2.sh
#   # merge chunks:
#   mkdir -p outputs/features/${MODEL_ID}-delta-t-surprisal/merged
#   cp outputs/features/${MODEL_ID}-delta-t-surprisal/chunk_*/feats*.h5 \
#      outputs/features/${MODEL_ID}-delta-t-surprisal/merged/
# =============================================================================

# ── SBATCH directives ────────────────────────────────────────────────────────
# NOTE: confirm --chdir (your working clone) and --partition/--gres for your cluster.
#       These follow the GPU (kuma / l40s) convention used by scripts/kuma_probe_*.sh.
#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=delta_t_d2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition l40s
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/delta_t_%x_%A_%a.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/delta_t_%x_%A_%a.err

# ── Parameters (read from env, fall back to defaults) ────────────────────────
MODEL_ID="${MODEL_ID:-wav2vec2-medium}"      # whisper-large | wav2vec2-medium | wav2vec2-large
WINDOW_DUR="${WINDOW_DUR:-10.0}"             # 30 for whisper-large; 10 for wav2vec2
WINDOW_STRIDE="${WINDOW_STRIDE:-5.0}"        # 10 for whisper-large; 5  for wav2vec2
CHUNK_SIZE="${CHUNK_SIZE:-32}"              # stimulus WINDOWS per array task
BATCH_T="${BATCH_T:-16}"                     # truncations per forward pass (memory knob)

# ── Derived paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/work/upschrimpf1/sigfstea/multimodal-brain-scaling"
# Cortical Surprisal audiobook WAVs — source of surprisal_30s.h5 / existing whisper
# -delta-t-surprisal features (recovered from the stimulus_ids inside surprisal_30s.h5).
DATA_ROOT="${DATA_ROOT:-/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/data/cortical_suprisal_dataset/audiobooks}"

LAYERS_CONFIG="configs/extraction/audio/${MODEL_ID//-/_}_layers.json"
OUTPUT_BASE="outputs/features/${MODEL_ID}-delta-t-surprisal"

# ── Header ────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "DELTA-T FEATURE EXTRACTION  (D2 Cortical Surprisal)"
echo "Model:   ${MODEL_ID}   window=${WINDOW_DUR}s   stride=${WINDOW_STRIDE}s"
echo "Data:    ${DATA_ROOT}"
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
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: CUDA not available — running on CPU (fine, just slower for large models)."
fi
echo ""

# ── Validate inputs ───────────────────────────────────────────────────────────
if [ ! -f "$LAYERS_CONFIG" ]; then
    echo "ERROR: layer config not found: $LAYERS_CONFIG"
    exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: data_root not found: $DATA_ROOT"
    exit 1
fi

# ── Compute stim range for this task ─────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
STIM_START=$(( TASK_ID * CHUNK_SIZE ))
CHUNK_DIR="${OUTPUT_BASE}/chunk_${TASK_ID}"

echo "Stimulus windows: start=${STIM_START}  count=${CHUNK_SIZE}"
echo "Output:  ${CHUNK_DIR}"
echo ""

mkdir -p "$CHUNK_DIR" logs

# ── Run extraction ────────────────────────────────────────────────────────────
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} python -m mbs.extraction.extract_features_delta_t \
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
    ls -lh "$CHUNK_DIR"
else
    echo "FAILED   task=${TASK_ID}  exit=${EXIT_CODE}"
fi

echo "End: $(date)"
exit $EXIT_CODE
