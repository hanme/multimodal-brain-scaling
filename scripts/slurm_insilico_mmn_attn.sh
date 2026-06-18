#!/bin/bash
# =============================================================================
# SLURM (CPU, jed standard): Workstream-B in-silico MMN + fit-quality figures from a trained
# attention-encoder checkpoint. The per-window network inference is light but the throttled,
# shared login node makes it crawl -> run it on a compute node instead.
#
# Submit:  sbatch scripts/slurm_insilico_mmn_attn.sh --checkpoint <...> --mmn_features_root <...> ...
# All args after the script name are forwarded verbatim to scripts/insilico_mmn_attn.py.
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=insilico_mmn_attn
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5500M
#SBATCH --time=00:30:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/insilico_mmn_attn_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/insilico_mmn_attn_%j.err

PROJECT_DIR="/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling"
cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs outputs/figures

echo "Start: $(date) on $(hostname)"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} python scripts/insilico_mmn_attn.py "$@"
EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
