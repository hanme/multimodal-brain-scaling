#!/bin/bash
# =============================================================================
# Phase-2 view of report_extraction_cost.sh: the same measured-vs-predicted table, pinned to the
# Phase-2 shape (14 new clips per method dir, 290 dirs) and rolled up per MODEL rather than per
# JOB, plus the one number the Phase-2 budget was written against -- CHF per pair.
#
# Why a wrapper and not a second copy: the sacct arithmetic, the MODEL_ID= log lookup and the
# literature per-clip table live in report_extraction_cost.sh and stay there. This sets
# CLIPS_PER_DIR/N_DIRS_FULL, re-aggregates its rows by model (Phase 1 needed two array chunks per
# model, so one model could span several job ids), and prices the result per pair.
#
# The headline check is the last block: novel_search_common.CHF_PER_PAIR_PHASE2 = 1.325 is the
# overhead-corrected rate the 145-pair selection was priced with, against a flat-actuals upper
# bound of 1.432 CHF/pair carried over from Phase 1. Landing at or under 1.325 confirms the
# model-load overhead really did amortise 14 ways instead of 2. Reporting only: nothing in the
# search branches on it.
#
#   scripts/report_extraction_cost_phase2.sh 65901234 65901235
#   scripts/report_extraction_cost_phase2.sh $(sacct --name=mmn_extract_batch --starttime=today \
#       --format=JobID --parsable2 --noheader | grep -oE '^[0-9]+' | sort -u)
#
# Off-cluster: run the base script on the cluster, paste its table into a file, summarise here.
#   COST_TABLE=phase2_table.txt scripts/report_extraction_cost_phase2.sh
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-$HERE/report_extraction_cost.sh}"

# Phase-2 shape. Each selected pair is 2 method dirs, each already holding the standard + N7/var1
# from Phase 1 and gaining the other 14 clips of the full deviant grid. The pair count is not a
# constant of the design -- it is however many pairs cleared n_agree >= MIN_AGREE_PHASE2 (127 on
# the 2026-07-27 ranking), so re-read it if the screen is re-run:
#   N_PAIRS=$(($(wc -l < outputs/results_novel_search/phase2_selected_pairs.csv) - 1))
CLIPS_PER_DIR="${CLIPS_PER_DIR:-14}"
N_PAIRS="${N_PAIRS:-127}"
N_DIRS_FULL="${N_DIRS_FULL:-$((2 * N_PAIRS))}"
N_MODELS="${N_MODELS:-6}"
CHF_PER_PAIR_TARGET="${CHF_PER_PAIR_TARGET:-1.325}"   # novel_search_common.CHF_PER_PAIR_PHASE2
CHF_PER_PAIR_CAP="${CHF_PER_PAIR_CAP:-1.432}"         # Phase-1 flat actuals, the upper bound
COST_TABLE="${COST_TABLE:-}"

if [ -n "$COST_TABLE" ]; then
    [ -f "$COST_TABLE" ] || { echo "COST_TABLE not found: $COST_TABLE"; exit 1; }
    raw=$(cat "$COST_TABLE")
    echo "summarising pasted table: $COST_TABLE"
else
    [ $# -gt 0 ] || { echo "usage: $0 <jobid> [<jobid> ...]   (or COST_TABLE=<file> $0)"; exit 1; }
    [ -x "$BASE" ] || [ -f "$BASE" ] || { echo "base script not found: $BASE"; exit 1; }
    raw=$(CLIPS_PER_DIR="$CLIPS_PER_DIR" N_DIRS_FULL="$N_DIRS_FULL" bash "$BASE" "$@")
    status=$?
    [ "$status" -eq 0 ] || { echo "$raw"; exit "$status"; }
fi

echo "$raw" | grep -F '(no completed task records)' && echo

# Roll the base script's per-job rows up by model. pred/m is identical across chunks of the same
# model, so it is taken from the first row rather than recomputed -- the per-clip table stays in
# one place, in the base script.
echo "$raw" | awk -v k="$CLIPS_PER_DIR" -v N="$N_DIRS_FULL" -v np="$N_PAIRS" \
                  -v nm="$N_MODELS" -v tgt="$CHF_PER_PAIR_TARGET" -v cap="$CHF_PER_PAIR_CAP" '
$2 ~ /^[0-9]+$/ && NF >= 7 && $3 ~ /^[0-9.]+$/ {
    m = $1
    if (!(m in seen)) { seen[m] = 1; order[++n_models] = m; pred[m] = $4 }
    tasks[m] += $2
    coreh[m] += $3 * $2            # per-task mean x tasks = core-h for that job
    chf[m]   += $6
}
END {
    CHF_PER_CORE_H = 0.0055
    printf "%-17s %6s %10s %10s %8s %9s %11s\n", "MODEL", "TASKS", "core-h/m", "pred/m", "RATIO", "CHF used", "full model"
    for (i = 0; i < 80; i++) printf "-"; printf "\n"

    tot_tasks = 0; tot_chf = 0; tot_coreh = 0; tot_full = 0
    for (i = 1; i <= n_models; i++) {
        m = order[i]
        per = coreh[m] / tasks[m]
        ratio = (pred[m] > 0) ? sprintf("%.2f", per / pred[m]) : "-"
        full = per * N * CHF_PER_CORE_H
        printf "%-17s %6d %10.3f %10s %8s %9.2f %11.1f\n", m, tasks[m], per, pred[m], ratio, chf[m], full
        tot_tasks += tasks[m]; tot_chf += chf[m]; tot_coreh += coreh[m]; tot_full += full
    }
    for (i = 0; i < 80; i++) printf "-"; printf "\n"
    printf "%-17s %6d %10s %10s %8s %9.2f %11.1f\n", "TOTAL", tot_tasks, "", "", "", tot_chf, tot_chf

    expected = N * nm
    printf "\ncore-h used: %.1f    CHF actually used by these jobs: %.2f\n", tot_coreh, tot_chf
    printf "array tasks accounted for: %d of %d expected (%d dirs x %d models)\n", tot_tasks, expected, N, nm
    if (tot_tasks < expected)
        printf "  INCOMPLETE -- %d task(s) missing; each model below is projected to %d dirs from its own rate.\n", expected - tot_tasks, N
    else if (tot_tasks > expected)
        printf "  MORE tasks than expected -- resubmissions counted twice? De-duplicate the job ids.\n"

    # Price per pair on the FULL 290-dir run. Each model is projected from ITS OWN measured rate
    # (the "full model" column) and the projections summed -- never one model rescaled to stand in
    # for the rest, which with a 27x spread between whisper-tiny and whisper-medium would be off
    # by an order of magnitude. A model with no completed tasks is left OUT and named, because
    # there is no measured rate to project it from.
    if (n_models == 0) { print "\nno completed task records in any job -- nothing to price."; exit }
    per_pair = tot_full / np
    printf "\nmeasured CHF per pair: %.3f   (%.2f CHF over %d pairs, both directions, %d model(s))\n",
           per_pair, tot_full, np, n_models
    if (n_models < nm) {
        printf "  PARTIAL: %d of %d models have no completed tasks and are NOT in this figure.\n",
               nm - n_models, nm
        printf "  It is a floor, not a Phase-2 total -- not comparable to %.3f until all %d land.\n", tgt, nm
    } else {
        printf "  vs CHF_PER_PAIR_PHASE2 = %.3f (overhead-corrected plan): %+.1f%%\n",
               tgt, 100 * (per_pair - tgt) / tgt
        printf "  vs %.3f (Phase-1 flat-actuals cap):                %+.1f%%\n",
               cap, 100 * (per_pair - cap) / cap
        printf "  Phase-2 total at this rate: %.0f CHF\n", tot_full
    }
}'

echo
echo "ratio = measured / predicted, against the literature per-clip figures x $CLIPS_PER_DIR clips"
echo "+ 0.27 core-h task overhead. 'full model' extrapolates the measured per-method rate to"
echo "$N_DIRS_FULL dirs; once every task has landed it equals 'CHF used' for that model, so a gap"
echo "there is the completeness check, not an estimate."
