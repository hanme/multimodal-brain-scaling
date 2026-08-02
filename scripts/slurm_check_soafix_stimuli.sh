#!/bin/bash
# =============================================================================
# Trailing-floor stimulus check as a batch job — the gate between generating the
# regenerated literature stimuli and staging/extracting them.
#
# check_soafix_stimuli.py md5s ~3,000 wavs (the new flat tree against the staged baseline) and
# runs the pipeline's own onset detector over a few hundred more. That is minutes of solid Lustre
# IO, which does not belong on the login node even though it is not CPU-heavy.
#
# THE EXIT CODE IS THE POINT. 0 = 12 ids byte-identical and 12 rebuilt with the expected trailing
# audio; non-zero = the layout moved where it must not have, and the whole comparison against the
# committed baseline is invalid. Check it before staging rather than eyeballing the log:
#     sacct -j <jobid> --format=JobID,State,ExitCode
#
# No --chdir directive: sbatch already defaults the working directory to wherever you submitted
# from, so submit from the project root and the relative log paths below land in logs/. PROJECT_DIR
# overrides it if you need to submit from elsewhere.
#
#   sbatch scripts/slurm_check_soafix_stimuli.sh                        # the default 'key' scope
#   sbatch scripts/slurm_check_soafix_stimuli.sh --onset_scope all      # time every clip
#   NEW_SRC=outputs/stim_gen_other sbatch scripts/slurm_check_soafix_stimuli.sh
# =============================================================================

#SBATCH --job-name=check_soafix_stim
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/check_soafix_stim_%j.out
#SBATCH --error=logs/check_soafix_stim_%j.error

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
NEW_SRC="${NEW_SRC:-outputs/stim_gen_soafix}"
BASELINE_WHISPER="${BASELINE_WHISPER:-outputs/mmn_stimuli}"
BASELINE_WAV2VEC2="${BASELINE_WAV2VEC2:-outputs/mmn_stimuli_wav2vec2}"
METADATA_CSV="${METADATA_CSV:-data/metadata/literature_frequency_intensity_duration_metadata.csv}"
TRAILING_FLOOR_MS="${TRAILING_FLOOR_MS:-400}"

cd "${PROJECT_DIR:-.}" || { echo "cannot cd to $PROJECT_DIR"; exit 1; }
source env.sh
mkdir -p logs

[ -d "$NEW_SRC" ] || { echo "new stimulus tree not found: $NEW_SRC"; exit 1; }
[ -f "$METADATA_CSV" ] || { echo "metadata CSV not found: $METADATA_CSV"; exit 1; }

echo "Start: $(date) on $(hostname)"
echo "  new tree=${NEW_SRC}  floor=${TRAILING_FLOOR_MS} ms"
echo "  baseline=${BASELINE_WHISPER} | ${BASELINE_WAV2VEC2}"
echo "  extra args: $*"

# "$@" LAST so trailing flags (--onset_scope, --tol_ms) override the defaults above.
python scripts/check_soafix_stimuli.py \
    --new_src           "$NEW_SRC" \
    --baseline_whisper  "$BASELINE_WHISPER" \
    --baseline_wav2vec2 "$BASELINE_WAV2VEC2" \
    --metadata_csv      "$METADATA_CSV" \
    --trailing_floor_ms "$TRAILING_FLOOR_MS" \
    "$@"

EXIT_CODE=$?
echo "End: $(date)"
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS check_soafix_stimuli" \
                     || echo "FAILED check_soafix_stimuli exit=${EXIT_CODE} -- DO NOT STAGE"
exit $EXIT_CODE
