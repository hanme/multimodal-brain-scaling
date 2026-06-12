#!/bin/bash
# =============================================================================
# SLURM: in-silico MMN — fit FIR mTRF on Broderick, apply to MMN stimuli,
#         plot deviant / standard / (deviant-standard) for method_09.
# =============================================================================
# Runs on a compute node because the Ridge fit (X ~ 11700 x 20992, upcast to
# float64 for the SVD + lagged-design transients) needs >> the ~8 GB login-node
# cgroup cap. ~1-2 min compute once scheduled.
#
# Each task fits the Broderick->EEG mapping ONCE for one whisper layer and loops over all
# MMN methods (one figure per method). Layer scan = array 0-5 (blocks.0 .. blocks.5).
#
# Submit (single layer):   sbatch scripts/slurm_insilico_mmn.sh
# Submit (full layer scan):sbatch --array=0-5 scripts/slurm_insilico_mmn.sh
# Extra args after the script name are forwarded to insilico_mmn.py.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=insilico_mmn
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5500M
#SBATCH --time=00:30:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/insilico_mmn_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/insilico_mmn_%j.err

PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs outputs/figures

echo "Start: $(date) on $(hostname)"

# Layer scan when run as an array (task i -> blocks.i); single default layer otherwise.
LAYER_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LAYER="blocks.${SLURM_ARRAY_TASK_ID}"
    LAYER_ARGS=(--layer "$LAYER")
    echo "Array task ${SLURM_ARRAY_TASK_ID} -> layer ${LAYER}"
fi

OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python scripts/insilico_mmn.py "${LAYER_ARGS[@]}" "$@"
EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
