#!/bin/bash
# =============================================================================
# SLURM array: model->EEG mTRF layer sweep (D2, CV-on-train selection) for the NEW models.
# 6 tasks = {whisper-large, wav2vec2-medium, wav2vec2-large} x {parcels, electrodes}.
# CPU only (sklearn RidgeCV). Self-contained: merges each model's delta-t chunks (once, with an
# atomic lock so the two level-tasks don't race), picks the right EEG target, then sweeps.
#
# EEG target per model (window length must match the features):
#   whisper-large  -> surprisal_30s.h5   (30 s / 10 s features)
#   wav2vec2-*     -> surprisal_10s.h5   (10 s / 5 s features; build with slurm_build_surprisal_10s.sh)
#
# Submit AFTER the corresponding slurm_extract_delta_t_d2.sh arrays finish (use --dependency), e.g.:
#   sbatch --array=0,1 scripts/slurm_eeg_mapping_sweep_d2.sh                 # whisper-large only
#   PCA_VAR=0.95 sbatch --array=2-5 --export=ALL scripts/slurm_eeg_mapping_sweep_d2.sh   # wav2vec2 (tame wide sweeps)
#   sbatch --array=0-5 scripts/slurm_eeg_mapping_sweep_d2.sh                 # all six
# After it finishes: chosen_layer + test_r_chosen are in
#   outputs/results/eeg_mapping/<model>__<level>__D2.json
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=eeg_sweep_d2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=72:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/eeg_sweep_d2_%A_%a.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/eeg_sweep_d2_%A_%a.err

set -euo pipefail
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling
source env.sh
mkdir -p logs outputs/results/eeg_mapping
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

MODELS=(whisper-large wav2vec2-medium wav2vec2-large)
LEVELS=(parcels electrodes)

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK / 2))]}
LEVEL=${LEVELS[$((TASK % 2))]}

# EEG target: whisper-large maps against the 30 s EEG; wav2vec2 against the 10 s EEG.
case "$MODEL" in
  whisper-*)  NEURAL=outputs/neural_data/surprisal_30s.h5 ;;
  wav2vec2-*) NEURAL=outputs/neural_data/surprisal_10s.h5 ;;
  *) echo "ERROR: unknown model $MODEL"; exit 1 ;;
esac
if [ ! -f "$NEURAL" ]; then
  echo "ERROR: EEG target not found: $NEURAL"
  [ "${MODEL#wav2vec2-}" != "$MODEL" ] && echo "  -> run scripts/slurm_build_surprisal_10s.sh first."
  exit 1
fi

BASE="outputs/features/${MODEL}-delta-t-surprisal"
MERGED="${BASE}/merged"

# ── Merge chunks -> merged/ (once; atomic lock so the two level-tasks don't collide) ──────────
if [ ! -f "${MERGED}/.merge_done" ]; then
  if mkdir "${BASE}/.merge_lock" 2>/dev/null; then
    echo "Merging ${MODEL} chunks -> ${MERGED}"
    mkdir -p "$MERGED"
    cp ${BASE}/chunk_*/feats*.h5 "$MERGED"/ 2>/dev/null || true
    touch "${MERGED}/.merge_done"
    rmdir "${BASE}/.merge_lock"
  else
    echo "Another task is merging ${MODEL}; waiting..."
    for _ in $(seq 1 120); do [ -f "${MERGED}/.merge_done" ] && break; sleep 5; done
  fi
fi
if ! ls "${MERGED}"/*.h5 >/dev/null 2>&1; then
  echo "ERROR: no merged features in ${MERGED} (did extraction finish?)"; exit 1
fi

PCA_ARG=""; [ -n "${PCA_VAR:-}" ] && PCA_ARG="--pca_var ${PCA_VAR}"

echo "=== eeg_mapping_sweep D2: ${MODEL} / ${LEVEL}  (feats: ${MERGED}  neural: ${NEURAL}) ==="
python scripts/eeg_mapping_sweep.py \
  --model_id "$MODEL" --target_level "$LEVEL" \
  --features_dir "$MERGED" --neural "$NEURAL" \
  $PCA_ARG \
  --out "outputs/results/eeg_mapping/${MODEL}__${LEVEL}__D2.json"

echo "DONE ${MODEL}/${LEVEL}"
