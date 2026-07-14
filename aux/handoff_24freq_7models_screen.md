# Handoff — 24-frequency × 7-model MMN screen (metric & threshold decisions)

**Spec:** `aux/prompt_24freq_7models_metric_selection.md`.
**Status (2026-07-13):** Local **code prep DONE** (this session). Cluster compute + local scoring/
plots/decisions **pending** (need the 24-method × 7-model prediction h5s synced back).

The screen extends the counterbalanced MMN work from **10 methods × 4 whisper models** to **24
frequency-oddball methods × 7 models**, mTRF only, at two sites (**frontal parcel**, **FCz
electrode**), to deliver two decisions: (1) the MMN metric — **S2** vs **S7** and, if S7, the µV
floor **X ∈ {0.5, 0.75, 1.0, 1.5}**; (2) a **#/48** "brain-like enough" cutoff feeding the
downstream novel-stimulus search.

---

## Code changes made this session (local, uncommitted)

1. **`scripts/insilico_mmn.py`** — `METHODS` is now **derived from the metadata CSV**
   (`build_methods_from_csv()`), yielding **48 entries** = the 24 `change_type=="Frequency"` methods
   × {regular, counter} (ids 9,10,12,17,18,19,20,21,27,28,29,30,31,32,33,37,43,44,53,55,60,72,74,75).
   `insilico_mmn_electrodes.py` imports `METHODS`, so both drivers are covered. `--methods all` now
   runs all 48. (SOA is still looked up per-method from the CSV.)
2. **`scripts/analyze_mmn_criteria_s5_s6.py`** — `UV_SWEEP` now includes **0.75**:
   `(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)`. `analyze_mmn_s7_roi.py` imports it, so its CSV carries an
   S7@0.75 column automatically.
3. **`scripts/analyze_mmn_screen_24freq.py`** — **new**. Consumes the screen's long `mmn_s7_roi.csv`
   (mTRF, two sites) → tidy CSV + summary CSVs + the 4 figures + console tables (below). Style mirrors
   `aux/analysis_with_counter/plots/deviance_scaling_plots.py` (Okabe–Ito, 7-model colors/markers).
   Concordance uses a `--ref_x` headline floor, **default 0.5 µV**.
4. **Method zero-padding fix** — `build_methods_from_csv()` now emits `method_{id:02d}` (→ `method_09`),
   matching the generator's `method_{id:02d}` filenames and the `mmn_stimuli/method_XX` /
   `mmn-method_XX-delta-t` path convention. Only method 9 was affected (all other ids are ≥10).
5. **Default X = 0.5 µV** — `analyze_mmn_criteria_s5_s6.py` `--dip_uv_threshold` default 1.0→**0.5**;
   `analyze_mmn_s7_roi.py` was already 0.5. `UV_SWEEP` still carries 0.75 (and 0.5/1.0/1.5).
6. **SLURM wrappers extended to 7 models** — `slurm_insilico_mmn.sh` and `..._electrodes.sh` now have
   layer maps + a 7-entry `MODELS` array covering whisper-large (`blocks.21`) and wav2vec2-medium/large
   (**`encoder.layers.0` placeholder**), and select `--train_neural` (`surprisal_10s.h5` for wav2vec2,
   else `surprisal_30s.h5`). `slurm_mmn_extract.sh` gained `MMN_STIM_ROOT` + `WINDOW_DUR`/`WINDOW_STRIDE`
   env overrides (defaults unchanged) so wav2vec2's 10 s clips extract with a 10 s window.

**Smoke-tested** against the existing whisper×10 (regular-only) predictions: scoring + the new script
run clean; S7 ≤ S2 with 0 violations; all CSVs + 5 plots produced. All three SLURM scripts pass
`bash -n`.

---

## Per-model run config

| Model | chosen_layer (parcels / electrodes) | `--train_neural` | `--train_features` | MMN extract window |
| ----- | ----------------------------------- | ---------------- | ------------------ | ------------------ |
| whisper-tiny   | `blocks.0` / `blocks.0` | `surprisal_30s.h5` | `outputs/features/whisper-tiny-delta-t-surprisal/merged` | 30 s |
| whisper-base   | `blocks.0` / `blocks.0` | `surprisal_30s.h5` | `outputs/features/whisper-base-delta-t-surprisal/merged` | 30 s |
| whisper-small  | `blocks.3` / `blocks.1` | `surprisal_30s.h5` | `outputs/features/whisper-small-delta-t-surprisal/merged` | 30 s |
| whisper-medium | `blocks.11` / `blocks.12` | `surprisal_30s.h5` | `outputs/features/whisper-medium-delta-t-surprisal/merged` | 30 s |
| whisper-large  | `blocks.21` / `blocks.21` | `surprisal_30s.h5` | `outputs/features/whisper-large-delta-t-surprisal/merged` | 30 s |
| wav2vec2-medium | **`encoder.layers.0` placeholder** → swap real chosen_layer when JSON lands | `surprisal_10s.h5` | `outputs/features/wav2vec2-medium-delta-t-surprisal/merged` | **10 s** |
| wav2vec2-large  | **`encoder.layers.0` placeholder** → swap real chosen_layer when JSON lands | `surprisal_10s.h5` | `outputs/features/wav2vec2-large-delta-t-surprisal/merged` | **10 s** |

`chosen_layer` for whisper tiny/base/small/medium is already in `outputs/results/eeg_mapping/`;
whisper-large + wav2vec2 JSONs must be synced from the cluster (whisper-large is known = `blocks.21`).

---

## Part 2 — Cluster runbook (concrete; run from the cluster PROJECT_DIR after `source env.sh`)

Assumes D2/surprisal features exist for all 7 models (`outputs/features/{model}-delta-t-surprisal/merged`)
and layer configs exist for all 7. The SLURM wrappers already encode per-model layers/paths (extended
this session). **All three wrappers now target the sigfstea tree**
`/work/upschrimpf1/sigfstea/multimodal-brain-scaling` (`slurm_mmn_extract.sh` was repointed from the
mehrer tree so extraction outputs land where the in-silico step reads them). Sync the edited scripts
to that tree before running.

The 24 method ids and the 48 method-dir names:
```bash
IDS="09 10 12 17 18 19 20 21 27 28 29 30 31 32 33 37 43 44 53 55 60 72 74 75"
METHODS=""; for id in $IDS; do METHODS="$METHODS method_${id} method_${id}_counter"; done   # 48
```

**1. Generate stimuli (filters change_type==Frequency internally):** run on a compute node — the
generator uses `--n_workers` = all cores, so don't run it on the login node.
```bash
sbatch scripts/slurm_generate_stimuli.sh     # 16 cores, --n_workers=$SLURM_CPUS_PER_TASK, -> outputs/stim_gen/
# (equivalent direct run, e.g. inside salloc: python scripts/00aa_generate_audio_stimuli.py \
#   --metadata_csv data/metadata/literature_frequency_intensity_duration_metadata.csv \
#   --output_dir outputs/stim_gen --n_workers 16)
```
Emits `outputs/stim_gen/audio_outputs_{regular,counter}/{whisper,wav2vec2}/`.

**2. Stage into per-family stimulus roots** (whisper 30 s clips vs wav2vec2 10 s clips; all 5 whisper
models share the `whisper/` clips, both wav2vec2 share `wav2vec2/`):
```bash
SRC=outputs/stim_gen
for id in $IDS; do
  for fam_root in "whisper:outputs/mmn_stimuli" "wav2vec2:outputs/mmn_stimuli_wav2vec2"; do
    fam=${fam_root%%:*}; root=${fam_root##*:}
    mkdir -p $root/method_${id} $root/method_${id}_counter
    cp $SRC/audio_outputs_regular/$fam/method_${id}_*.wav $root/method_${id}/
    cp $SRC/audio_outputs_counter/$fam/method_${id}_*.wav $root/method_${id}_counter/
  done
done   # each dir now holds 16 wavs (1 standard + 15 deviants)
```

**3. MMN feature extraction** — 336 array jobs (7 models × 48 dirs × 16 tasks). Throttle if the queue
caps submissions.
```bash
# whisper (30 s window, default MMN_STIM_ROOT=outputs/mmn_stimuli)
for m in whisper-tiny whisper-base whisper-small whisper-medium whisper-large; do
  for meth in $METHODS; do
    sbatch --export=ALL,MODEL_ID=$m,MMN_METHOD=$meth --array=0-15 scripts/slurm_mmn_extract.sh
  done
done
# wav2vec2 (10 s window + 10 s stimuli root)
for m in wav2vec2-medium wav2vec2-large; do
  for meth in $METHODS; do
    sbatch --export=ALL,MODEL_ID=$m,MMN_METHOD=$meth,MMN_STIM_ROOT=$PWD/outputs/mmn_stimuli_wav2vec2,WINDOW_DUR=10.0,WINDOW_STRIDE=10.0 \
           --array=0-15 scripts/slurm_mmn_extract.sh
  done
done
```
→ features at `outputs/features/{model}-mmn/mmn-method_XX-delta-t/` (whisper-base: `outputs/features/`).

**4. In-silico MMN** — after all extractions finish (the driver skips any method with a missing feature
dir). One job per model does all 48 methods; the wrappers set layer/`train_neural`/`data_dir` per model:
```bash
# parcels → frontal (array 0-6 = the 7 MODELS)   |   electrodes → FCz
sbatch --array=0-6 scripts/slurm_insilico_mmn.sh
sbatch --array=0-6 scripts/slurm_insilico_mmn_electrodes.sh
# or one model at a time:
sbatch --export=ALL,MODEL_ID=whisper-large scripts/slurm_insilico_mmn.sh
sbatch --export=ALL,MODEL_ID=wav2vec2-medium scripts/slurm_insilico_mmn_electrodes.sh
```
→ `outputs/insilico_mmn_predictions/{model}/predictions__{layer}.h5` +
`electrode_predictions__{layer}.h5`. wav2vec2 uses `encoder.layers.0` (placeholder).

**5. Sync** all `outputs/insilico_mmn_predictions/{model}/*.h5` back to the local repo.

---

## Part 3 — Local scoring + analysis (after sync)

```
conda activate mbs-env
# 1. score (X=0.75 now in the sweep)
python scripts/analyze_mmn_s7_roi.py \
  --predictions_root outputs/insilico_mmn_predictions \
  --out outputs/results_24freq_7models/mmn_s7_roi.csv        # confirm S7<=S2 = 0 violations
# 2. tidy CSV + figures + tables
python scripts/analyze_mmn_screen_24freq.py \
  --s7_csv outputs/results_24freq_7models/mmn_s7_roi.csv \
  --out_dir outputs/results_24freq_7models                   # default --expected_conditions 48
```
Produces under `outputs/results_24freq_7models/`: `mmn_screen_24freq.csv` (tidy: model, site,
method_id, direction, S2, S7@{0.5,0.75,1,1.5}, trough_uv), `summary_counts_by_site.csv`,
`per_model_spearman.csv`, and `plots/` (Fig 1 X-vs-#/48 by model; Fig 2 pooled vs S2; Fig 3
deviance-scaling both sites). Console: summary counts, ranking-stability across X, frontal↔FCz
concordance.

**3. Write the decisions memo** `aux/analysis_24freq_7models/screen_results_and_decisions.md`
(style like `results_analysis_with_counter.md`): **metric** (S2 vs S7 + X — justified by whether S7
changes the model *ranking* vs S2, the µV-trough distribution + ~4× shrinkage caveat, clear-vs-marginal
separation) and **threshold** (#/48 cutoff — tied to the gap structure and the downstream search).

**Expected invariants after the full run:** tidy CSV = **672 rows** (7 × 2 × 48); exactly **48
conditions per model per site**; **S7 ≤ S2** everywhere; all **7 models** present at both sites.

---

## Caveats to carry into the memo

- **wav2vec2 provisional:** `encoder.layers.0` placeholder → re-run wav2vec2-medium/large × {parcels,
  electrodes} when the real `chosen_layer` JSONs land.
- **Shrinkage/scale:** predictions are ~4× amplitude-shrunk; X is calibrated to the model's own trough
  distribution (median ≈ −0.8 µV), not literature EEG µV — central to interpreting the X choice.
- **wav2vec2 comparability:** mapped on 10 s/10 s windows with `PCA_VAR=0.95` (whisper 30 s/10 s,
  `pca_var=None`); its test r is a loose sanity gate, and its MMN features must use a 10 s window.
- **Commits:** none unless asked; never add a `Co-Authored-By: Claude` trailer.
