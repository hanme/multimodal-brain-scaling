#!/bin/bash
# =============================================================================
# Feature completeness/integrity check as a batch job.
#
# check_novel_features.py opens every HDF5 in every method dir and reads a feature block out of
# each. At Phase-2 scale that is 254 dirs x 16 clips x 6 models = 24,384 files, which is far too
# slow (and too much IO) for the login node. The work is per-directory and shares no state, so it
# parallelises exactly: this asks for a node's worth of cores and hands the count straight to
# --n_workers.
#
# THE EXIT CODE IS THE POINT. 0 = every directory clean, non-zero = problems found. Check it
# before submitting the next stage rather than eyeballing the log:
#     sacct -j <jobid> --format=JobID,State,ExitCode
#
# No --chdir directive: sbatch already defaults the working directory to wherever you submitted
# from, so submit from the project root and the relative log paths below land in logs/. PROJECT_DIR
# overrides it if you need to submit from elsewhere.
#
#   sbatch scripts/slurm_check_novel_features.sh                        # Phase 2, all 6 models
#   EXPECT_CLIPS=2 METHOD_LIST=outputs/novel_methods_phase1.txt \
#       sbatch scripts/slurm_check_novel_features.sh                    # Phase 1
#   MODELS=whisper-tiny sbatch scripts/slurm_check_novel_features.sh    # one model
#   sbatch scripts/slurm_check_novel_features.sh --show 200             # extra args forwarded
# =============================================================================

#SBATCH --job-name=check_novel_feats
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/check_novel_feats_%j.out
#SBATCH --error=logs/check_novel_feats_%j.error

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
MODELS="${MODELS:-all}"
METHOD_LIST="${METHOD_LIST:-outputs/novel_methods_phase2.txt}"
EXPECT_CLIPS="${EXPECT_CLIPS:-16}"
FEATURES_TAG="${FEATURES_TAG:-novel}"

cd "${PROJECT_DIR:-.}" || { echo "cannot cd to $PROJECT_DIR"; exit 1; }
source env.sh
mkdir -p logs

[ -f "$METHOD_LIST" ] || { echo "METHOD_LIST not found: $METHOD_LIST"; exit 1; }

echo "Start: $(date) on $(hostname)  cores=${SLURM_CPUS_PER_TASK:-1}"
echo "  models=${MODELS}  list=${METHOD_LIST} ($(grep -c . "$METHOD_LIST") conditions)"
echo "  expect_clips=${EXPECT_CLIPS}  features_tag=${FEATURES_TAG}"
echo "  extra args: $*"

# "$@" LAST so trailing flags (--show, --features_root, --n_workers) override the defaults above.
python scripts/check_novel_features.py \
    --models       "$MODELS" \
    --method_list  "$METHOD_LIST" \
    --expect_clips "$EXPECT_CLIPS" \
    --features_tag "$FEATURES_TAG" \
    --n_workers    "${SLURM_CPUS_PER_TASK:-4}" \
    "$@"

EXIT_CODE=$?
echo "End: $(date)"
[ $EXIT_CODE -eq 0 ] && echo "SUCCESS check_novel_features" \
                     || echo "FAILED check_novel_features exit=${EXIT_CODE}"
exit $EXIT_CODE
