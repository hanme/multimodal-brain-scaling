#!/bin/bash
# =============================================================================
# Score and verify the trailing-floor re-screen in one batch job — Step 7 without the sync.
#
# Runs analyze_mmn_s7_roi.py over the new prediction h5s, then verify_soafix_predictions.py over
# both the h5s and the resulting CSV. Doing it here rather than after the rsync means a bad screen
# is caught before anything is pulled, and only the small CSVs need to travel if you just want the
# numbers.
#
# THE EXIT CODE IS THE POINT. 0 = every epoch reaches the criteria window, no trough on a final
# sample, no truncated recovery search, and the 24 unchanged conditions score bit-identically to
# the committed baseline. Non-zero = something leaked between the runs:
#     sacct -j <jobid> --format=JobID,State,ExitCode
#
# No --chdir directive: sbatch already defaults the working directory to wherever you submitted
# from, so submit from the project root and the relative log paths below land in logs/. PROJECT_DIR
# overrides it if you need to submit from elsewhere.
#
#   sbatch scripts/slurm_verify_soafix_predictions.sh
#   OUT_CSV=outputs/results_soafix/rerun.csv sbatch scripts/slurm_verify_soafix_predictions.sh
#
# Runs equally well outside SLURM (it is a plain script with directives), which is how the same
# check is repeated locally after the h5s are synced back. env.sh is the cluster's module + venv
# setup, so skip it and bring your own interpreter:
#   conda activate mbs-env
#   USE_ENV_SH=0 PROJECT_DIR=$PWD bash scripts/slurm_verify_soafix_predictions.sh
# =============================================================================

#SBATCH --job-name=verify_soafix_pred
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_soafix_pred_%j.out
#SBATCH --error=logs/verify_soafix_pred_%j.error

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-outputs/insilico_mmn_predictions_soafix}"
BASELINE_PREDICTIONS_ROOT="${BASELINE_PREDICTIONS_ROOT:-outputs/insilico_mmn_predictions}"
BASELINE_S7_CSV="${BASELINE_S7_CSV:-outputs/results_24freq_7models/mmn_s7_roi.csv}"
OUT_CSV="${OUT_CSV:-outputs/results_soafix/mmn_s7_roi.csv}"
DIP_UV_THRESHOLD="${DIP_UV_THRESHOLD:-0.75}"
USE_ENV_SH="${USE_ENV_SH:-1}"

cd "${PROJECT_DIR:-.}" || { echo "cannot cd to $PROJECT_DIR"; exit 1; }
[ "$USE_ENV_SH" = "1" ] && source env.sh
mkdir -p logs "$(dirname "$OUT_CSV")"

# analyze_mmn_s7_roi.py defaults --out to outputs/results_with_counter/mmn_s7_roi.csv, and the
# baseline this run is compared against lives one directory over. Refuse to write over either --
# a clobbered baseline is unrecoverable and would silently make the comparison vacuous.
case "$OUT_CSV" in
    outputs/results_24freq_7models/*|outputs/results_with_counter/*|"$BASELINE_S7_CSV")
        echo "REFUSING: OUT_CSV=$OUT_CSV is a committed results file."
        echo "  Overwriting it destroys the baseline this screen is measured against."
        echo "  Use e.g. outputs/results_soafix/mmn_s7_roi.csv."
        exit 1 ;;
esac

[ -d "$PREDICTIONS_ROOT" ] || { echo "predictions root not found: $PREDICTIONS_ROOT"; exit 1; }

n_h5=$(find "$PREDICTIONS_ROOT" -name 'electrode_predictions__*.h5' | wc -l | tr -d ' ')
echo "Start: $(date) on $(hostname)"
echo "  predictions=${PREDICTIONS_ROOT} (${n_h5} h5, expect 7)"
echo "  baseline   =${BASELINE_S7_CSV}"
echo "  out        =${OUT_CSV}  (X=${DIP_UV_THRESHOLD})"
echo "  extra args : $*"
[ "$n_h5" -eq 7 ] || echo "  WARNING: expected 7 prediction files, one per model -- MODELS was"
[ "$n_h5" -eq 7 ] || echo "           probably left at the six SEARCH_MODELS, dropping whisper-large"

python scripts/analyze_mmn_s7_roi.py \
    --predictions_root  "$PREDICTIONS_ROOT" \
    --dip_uv_threshold  "$DIP_UV_THRESHOLD" \
    --out               "$OUT_CSV"
SCORE_EXIT=$?
if [ $SCORE_EXIT -ne 0 ]; then
    echo "FAILED analyze_mmn_s7_roi exit=${SCORE_EXIT}"
    exit $SCORE_EXIT
fi

# "$@" LAST so trailing flags override the defaults above.
python scripts/verify_soafix_predictions.py \
    --predictions_root          "$PREDICTIONS_ROOT" \
    --baseline_predictions_root "$BASELINE_PREDICTIONS_ROOT" \
    --baseline_s7_csv           "$BASELINE_S7_CSV" \
    --new_s7_csv                "$OUT_CSV" \
    "$@"

EXIT_CODE=$?
echo "End: $(date)"
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS verify_soafix_predictions" \
                     || echo "FAILED verify_soafix_predictions exit=${EXIT_CODE}"
exit $EXIT_CODE
