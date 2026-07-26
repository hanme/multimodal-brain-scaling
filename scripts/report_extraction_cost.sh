#!/bin/bash
# =============================================================================
# Measured cost of MMN extraction jobs, per model, against the predicted table.
#
# The per-clip figures the budget was built on came from the literature screen and are turning
# out pessimistic (whisper-tiny: 0.39 core-h/method measured against 1.03 predicted). This reads
# what jobs actually used and rescales the remaining run from it, so the plan tracks reality
# rather than the estimate.
#
# The model is not in the job name, so it is recovered from each job's log, which the extract
# wrapper prints as "MODEL_ID=<model>".
#
#   scripts/report_extraction_cost.sh 65811607 65813260
#   scripts/report_extraction_cost.sh $(sacct --name=mmn_extract_batch --starttime=today \
#       --format=JobID --parsable2 --noheader | grep -oE '^[0-9]+' | sort -u)
# =============================================================================
set -uo pipefail

CHF_PER_CORE_H=0.0055
CLIPS_PER_DIR="${CLIPS_PER_DIR:-2}"     # 2 in Phase 1, 14 in Phase 2
N_DIRS_FULL="${N_DIRS_FULL:-1056}"      # method dirs in a full Phase-1 model
LOGDIR="${LOGDIR:-logs}"

[ $# -gt 0 ] || { echo "usage: $0 <jobid> [<jobid> ...]"; exit 1; }
command -v sacct >/dev/null || { echo "sacct not found -- run this on the cluster"; exit 1; }

# Predicted core-h per clip, from the literature screen. A case rather than an associative
# array so this runs under bash 3.2 too and can be exercised off the cluster.
pred_core_h_per_clip() {
    case "$1" in
        whisper-tiny)    echo 0.38 ;;
        whisper-base)    echo 0.62 ;;
        whisper-small)   echo 1.81 ;;
        whisper-medium)  echo 5.87 ;;
        wav2vec2-medium) echo 0.39 ;;
        wav2vec2-large)  echo 0.74 ;;
        *)               echo 0    ;;
    esac
}

printf '%-17s %6s %10s %10s %8s %9s %11s\n' \
       MODEL TASKS "core-h/m" "pred/m" RATIO "CHF used" "full model"
printf '%.0s-' {1..80}; echo

total_used=0
for job in "$@"; do
    model=$(grep -hoE 'MODEL_ID=[A-Za-z0-9._-]+' "$LOGDIR"/*_"${job}"_*.out 2>/dev/null \
            | head -1 | cut -d= -f2)
    [ -n "$model" ] || model=$(grep -hoE 'MODEL_ID=[A-Za-z0-9._-]+' "$LOGDIR"/*"${job}"*.out \
            2>/dev/null | head -1 | cut -d= -f2)
    model="${model:-unknown}"

    # one row per array task; drop the .batch/.extern/.<step> accounting rows
    read -r core_h n <<<"$(sacct -j "$job" --format=JobID,CPUTimeRAW --parsable2 --noheader \
        | grep -vE '\.(batch|extern|[0-9]+)\|' \
        | awk -F'|' '$2 ~ /^[0-9]+$/ {s += $2; n++} END {printf "%.4f %d", s/3600, n+0}')"
    [ "${n:-0}" -gt 0 ] || { printf '%-17s %6s  (no completed task records)\n' "$model" "-"; continue; }

    per_method=$(awk -v s="$core_h" -v n="$n" 'BEGIN{printf "%.3f", s/n}')
    pred=$(awk -v c="$(pred_core_h_per_clip "$model")" -v k="$CLIPS_PER_DIR" 'BEGIN{printf "%.3f", k*c+0.27}')
    ratio=$(awk -v a="$per_method" -v p="$pred" 'BEGIN{printf "%s", (p>0)? sprintf("%.2f", a/p) : "-"}')
    chf=$(awk -v s="$core_h" -v r="$CHF_PER_CORE_H" 'BEGIN{printf "%.2f", s*r}')
    full=$(awk -v a="$per_method" -v N="$N_DIRS_FULL" -v r="$CHF_PER_CORE_H" \
           'BEGIN{printf "%.1f", a*N*r}')
    total_used=$(awk -v t="$total_used" -v c="$chf" 'BEGIN{printf "%.2f", t+c}')

    printf '%-17s %6d %10s %10s %8s %9s %11s\n' \
           "$model" "$n" "$per_method" "$pred" "$ratio" "$chf" "$full"
done

printf '%.0s-' {1..80}; echo
echo "CHF actually used by these jobs: $total_used"
echo
echo "ratio = measured / predicted. Below 1 means the literature per-clip figures are"
echo "pessimistic; 'full model' extrapolates the measured rate to $N_DIRS_FULL method dirs."
echo "For Phase 2 re-run with CLIPS_PER_DIR=14 N_DIRS_FULL=290."
