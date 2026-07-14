#!/bin/bash
# =============================================================================
# SLURM (CPU, jed standard): generate the frequency-oddball MMN stimuli.
#   00aa_generate_audio_stimuli.py synthesizes 1 standard + 15 deviants per method
#   per direction (regular + counter) for each model family (whisper 30 s, wav2vec2
#   10 s), filtering the metadata to change_type=="Frequency". Parallel over
#   --n_workers = the allocated cores; single node, ~minutes for the 24-method set.
#
# Submit:   sbatch scripts/slurm_generate_stimuli.sh
#           sbatch --export=ALL,OUTPUT_DIR=outputs/stim_gen,METADATA_CSV=... scripts/slurm_generate_stimuli.sh
# Bump --cpus-per-task below (and it flows into --n_workers) for more parallelism.
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=gen_stimuli
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/gen_stimuli_%j.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/gen_stimuli_%j.err

PROJECT_DIR="/work/upschrimpf1/sigfstea/multimodal-brain-scaling"
METADATA_CSV="${METADATA_CSV:-data/metadata/literature_frequency_intensity_duration_metadata.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stim_gen}"

cd "$PROJECT_DIR" || { echo "cannot cd"; exit 1; }
source env.sh
mkdir -p logs "$OUTPUT_DIR"

echo "Start: $(date) on $(hostname)   cores=${SLURM_CPUS_PER_TASK:-1}  metadata=$METADATA_CSV  out=$OUTPUT_DIR"

python scripts/00aa_generate_audio_stimuli.py \
    --metadata_csv "$METADATA_CSV" \
    --output_dir   "$OUTPUT_DIR" \
    --n_workers    "${SLURM_CPUS_PER_TASK:-4}"

EXIT_CODE=$?
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS generate_stimuli" || echo "FAILED generate_stimuli exit=${EXIT_CODE}"
echo "End: $(date)"
exit $EXIT_CODE
