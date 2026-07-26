#!/bin/bash
# =============================================================================
# Stage generated MMN wavs into the per-method directories the extractor reads,
# and emit the METHOD_LIST that addresses them.
#
# 00aa_generate_audio_stimuli.py writes a FLAT tree:
#     <src>/audio_outputs_{regular,counter}/{whisper,wav2vec2}/method_<id>_*.wav
# but extract_features_delta_t.py takes one --data_root per condition, and
# insilico_mmn reads <stimuli_root>/<method>/:
#     <root>/method_<id>/          <- regular direction
#     <root>/method_<id>_counter/  <- counterbalanced direction
# Nothing else in the repo bridges those two layouts. This does, for every method in the
# metadata CSV, and writes the matching method-dir list for slurm_mmn_extract_batch.sh.
#
# The two families are staged to separate roots because their clips differ in length (whisper
# 30 s, wav2vec2 10 s) and stim_dir feeds the final-tone-onset used for time-locking -- it MUST
# match the feature clips.
#
# Idempotent: re-running over a directory that has gained clips (the novel search's Phase-2
# top-up, 2 wavs -> 16) just copies the new ones.
#
# Phase 1 is ~4,200 copies and Phase 2 ~9,300, so this belongs on a compute node rather than
# the login node. It is a plain script with SBATCH directives, so it runs either way -- sbatch it
# for the full grid, run it directly for a handful of methods.
#
#   sbatch scripts/slurm_stage_novel_stimuli.sh                       # Phase 1
#   sbatch --export=ALL,METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv,\
# SRC=outputs/stim_gen_novel_phase2,METHOD_LIST_OUT=outputs/novel_methods_phase2.txt \
#          scripts/slurm_stage_novel_stimuli.sh                       # Phase 2
# =============================================================================

#SBATCH --chdir /work/upschrimpf1/sigfstea/multimodal-brain-scaling
#SBATCH --job-name=stage_novel
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/stage_novel_%j.out
#SBATCH --error=/work/upschrimpf1/sigfstea/multimodal-brain-scaling/logs/stage_novel_%j.err

set -uo pipefail
mkdir -p logs

METADATA_CSV="${METADATA_CSV:-data/metadata/novel_grid_frequency_metadata.csv}"
SRC="${SRC:-outputs/stim_gen_novel}"
WHISPER_ROOT="${WHISPER_ROOT:-outputs/mmn_stimuli_novel}"
WAV2VEC2_ROOT="${WAV2VEC2_ROOT:-outputs/mmn_stimuli_novel_wav2vec2}"
METHOD_LIST_OUT="${METHOD_LIST_OUT:-outputs/novel_methods_phase1.txt}"

[ -f "$METADATA_CSV" ] || { echo "metadata CSV not found: $METADATA_CSV"; exit 1; }
[ -d "$SRC" ] || { echo "source tree not found: $SRC (run slurm_generate_stimuli.sh first)"; exit 1; }

# method_id is column 2, change_type column 4 -- the same Frequency filter the generator and
# build_methods_from_csv apply, so the staged set matches the condition registry exactly.
IDS=$(awk -F, 'NR>1 && $4=="Frequency" {print $2}' "$METADATA_CSV")
N_IDS=$(echo "$IDS" | grep -c .)
[ "$N_IDS" -gt 0 ] || { echo "no change_type==Frequency rows in $METADATA_CSV"; exit 1; }

echo "staging $N_IDS methods from $SRC"
echo "  whisper  -> $WHISPER_ROOT"
echo "  wav2vec2 -> $WAV2VEC2_ROOT"

mkdir -p "$(dirname "$METHOD_LIST_OUT")"
: > "$METHOD_LIST_OUT"

copied=0
missing=0
for id in $IDS; do
    for spec in "whisper:${WHISPER_ROOT}" "wav2vec2:${WAV2VEC2_ROOT}"; do
        fam="${spec%%:*}"; root="${spec##*:}"
        for cfg in regular counter; do
            suffix=""; [ "$cfg" = "counter" ] && suffix="_counter"
            dst="${root}/method_${id}${suffix}"
            mkdir -p "$dst"
            # Files are named method_<id>_*.wav in BOTH direction trees; the direction is carried
            # by the source subdirectory, not the filename.
            if compgen -G "${SRC}/audio_outputs_${cfg}/${fam}/method_${id}_*.wav" > /dev/null; then
                cp "${SRC}/audio_outputs_${cfg}/${fam}/method_${id}_"*.wav "$dst/"
                copied=$((copied + 1))
            else
                echo "  MISSING ${SRC}/audio_outputs_${cfg}/${fam}/method_${id}_*.wav"
                missing=$((missing + 1))
            fi
        done
    done
    printf 'method_%s\nmethod_%s_counter\n' "$id" "$id" >> "$METHOD_LIST_OUT"
done

echo
echo "staged $copied method/family/direction groups ($missing missing)"
echo "method list -> $METHOD_LIST_OUT ($(wc -l < "$METHOD_LIST_OUT") entries, expect $((2 * N_IDS)))"

for root in "$WHISPER_ROOT" "$WAV2VEC2_ROOT"; do
    n_dirs=$(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'method_*' | wc -l | tr -d ' ')
    n_wavs=$(find "$root" -mindepth 2 -name '*.wav' | wc -l | tr -d ' ')
    # Every condition dir should hold the same clip count; a short one means a partial generation.
    short=$(for d in "$root"/method_*; do
                c=$(find "$d" -name '*.wav' | wc -l | tr -d ' ')
                echo "$c"
            done | sort -u | tr '\n' ' ')
    echo "  $root: $n_dirs dirs, $n_wavs wavs, clips-per-dir {${short% }}"
done

[ "$missing" -eq 0 ] || { echo "FAILED: $missing groups had no source wavs"; exit 1; }
echo "SUCCESS stage_novel_stimuli"
