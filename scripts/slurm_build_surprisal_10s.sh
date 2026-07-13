#!/bin/bash
# =============================================================================
# SLURM job: build surprisal_10s.h5 (10 s / 5 s Cortical Surprisal EEG target)
# for the wav2vec2 D2 mapping, then copy it into the multimodal-brain-scaling clone.
# =============================================================================
#
# Re-runs the EXISTING parametrized formatter (format_eeg_hdf5_surprisal) that produced
# surprisal_30s.h5, at window=10 s / stride=5 s. Keeps --n_test_parts 3 --seed 42 so the
# held-out parts match the 30 s file (test = AUNP02, BROP02, BROP03) -> comparable test r.
# CPU only. Confirm --data_root/--audio_root against the formatter's usage docstring:
#   sed -n '14,25p' src/mbs/data_prep/format_eeg_hdf5_surprisal.py
#
#   sbatch scripts/slurm_build_surprisal_10s.sh          # from either clone; paths are absolute
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis
#SBATCH --job-name=build_surprisal_10s
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/logs/build_surprisal_10s_%j.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/logs/build_surprisal_10s_%j.err

set -euo pipefail

# ── Parameters (override via env if needed) ──────────────────────────────────
WINDOW_DUR="${WINDOW_DUR:-10.0}"
WINDOW_STRIDE="${WINDOW_STRIDE:-5.0}"
TARGET_SR="${TARGET_SR:-50}"
N_TEST_PARTS="${N_TEST_PARTS:-3}"
SEED="${SEED:-42}"

TA_DIR=/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis
MBS_CLONE=/work/upschrimpf1/sigfstea/multimodal-brain-scaling
DATA_ROOT="${DATA_ROOT:-data/cortical_suprisal_dataset}"
AUDIO_ROOT="${AUDIO_ROOT:-data/cortical_suprisal_dataset/audiobooks}"
OUT_LOCAL="outputs/neural_data/surprisal_10s.h5"

cd "$TA_DIR"
mkdir -p logs outputs/neural_data

# Use the temporal-analysis project's own venv (it owns the surprisal formatter + deps).
PY="$TA_DIR/.venv/bin/python"
[ -x "$PY" ] || PY=python

echo "=== build surprisal_10s.h5  (window=${WINDOW_DUR}s stride=${WINDOW_STRIDE}s sr=${TARGET_SR}) ==="
echo "Python: $PY"; "$PY" --version
echo "Start: $(date)"

"$PY" -m mbs.data_prep.format_eeg_hdf5_surprisal \
    --data_root      "$DATA_ROOT" \
    --audio_root     "$AUDIO_ROOT" \
    --output_path    "$OUT_LOCAL" \
    --window_duration "$WINDOW_DUR" \
    --window_stride   "$WINDOW_STRIDE" \
    --target_sr       "$TARGET_SR" \
    --n_test_parts    "$N_TEST_PARTS" \
    --seed            "$SEED" \
    --overwrite       true

# Copy into the multimodal-brain-scaling clone so eeg_mapping_sweep.py can read it.
mkdir -p "$MBS_CLONE/outputs/neural_data"
cp "$OUT_LOCAL" "$MBS_CLONE/outputs/neural_data/surprisal_10s.h5"

echo "=== sanity check ==="
"$PY" - <<PY
import h5py
f = h5py.File("$OUT_LOCAL", "r")
print("window_s:", f.attrs.get("window_duration_s"), "stride_s:", f.attrs.get("window_stride_s"),
      "target_sr:", f.attrs.get("target_sr"))
for s in ("train", "test"):
    node = f[s]["stimulus_ids"]
    ids = node[()] if not isinstance(node, h5py.Group) else node[list(node.keys())[0]][()]
    ex = ids[0].decode() if hasattr(ids[0], "decode") else ids[0]
    print(f"{s}: {len(ids)} windows  e.g. {ex}")
PY

echo "Copied to: $MBS_CLONE/outputs/neural_data/surprisal_10s.h5"
echo "End: $(date)"
