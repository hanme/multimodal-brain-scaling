#!/bin/bash
# =============================================================================
# SLURM: in-silico MMN at the ELECTRODE level — fit FIR mTRF on Broderick (per-electrode
#         targets), apply to the MMN stimulus pairs, write one topographic MMN figure per
#         pair + an auto MMN verdict. Electrode counterpart of slurm_insilico_mmn.sh.
# =============================================================================
# Compute node (the Ridge fit exceeds the ~8 GB login-node cap). ~1-2 min once scheduled.
# One whisper layer per task; loops over all MMN methods. Layer scan = array.
#
# Defaults train the mapping on D2 (Cortical Surprisal, human-speech audiobook EEG) at
# whisper-small/blocks.10 — D1/Broderick gave ~zero held-out on central electrodes.
#   sbatch scripts/slurm_insilico_mmn_electrodes.sh                 # D2, whisper-small, blocks.10
# Override the training dataset/model with --train_neural / --train_features (old --broderick_*
# names still accepted). Extra args after the script name are forwarded to the python driver.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=insilico_mmn_elec
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5500M
#SBATCH --time=00:30:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/insilico_mmn_elec_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/insilico_mmn_elec_%j.err

PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs outputs/figures/insilico_mmn_electrodes

echo "Start: $(date) on $(hostname)"

LAYER_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LAYER="blocks.${SLURM_ARRAY_TASK_ID}"
    LAYER_ARGS=(--layer "$LAYER")
    echo "Array task ${SLURM_ARRAY_TASK_ID} -> layer ${LAYER}"
fi

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python scripts/insilico_mmn_electrodes.py "${LAYER_ARGS[@]}" "$@"
EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
