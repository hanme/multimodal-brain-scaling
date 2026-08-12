# Repository Operational Overview — `multimodal-brain-scaling` (auditory-EEG / MMN fork)

**Generated:** 2026-06-18
**Last updated:** 2026-08-03 (**results-artifact map** — §1.3 now lists the in-silico prediction roots and the three screen-result directories (`results_soafix_full`, `results_24freq_7models`, `results_novel_search`) and says which memo section reads which; §14 adds the two results memos to the reading order. Prior 2026-07-27 entry: **§17 Phase-1 results landed** — 184.74 CHF spent, top tier not distinguishable from chance, empty consensus set, 110 pairs recommended for Phase 2; see §17.3–§17.4 and `aux/analysis_novel_search/novel_stimulus_search_results.md`. Prior 2026-07-25 entry: new **§17 Stimuli Search Phase** — the novel tone-pair grid search, its pipeline prerequisites, cost model and run guide. Prior 2026-07-13 entry: enabled whisper-large + wav2vec2-{medium,large} for the D2 mTRF mapping — see the **2026-07-13** changelog line below, §1.4, §1.5, and `aux/handoff_enable_large_wav2vec2_models.md`. Earlier 2026-06-21 doc pass: relabeled the in-use MMN stimulus design as **Definition 1** — literature classic-oddball, frequency-deviant — after confirming via direct literature read-through that all 10 sourced papers use this design, not the final-tone-controlled design originally assumed. **Definition 2** is now reserved for that final-tone-controlled design, which is the target the stimulus set is being revised toward — no true Definition 2 literature source has been found yet. See §1.6, §1.7, §5, §13.8, §16.1–§16.2)

**2026-07-25 — novel tone-pair stimulus search (§17):** a 903-pair semitone-spaced frequency grid screened in two budgeted phases against 6 models, mTRF-only, FCz, S7 @ 0.75 µV. Mappings and committed layers are frozen and reused. Required making the condition registry `--metadata_csv`-driven (it was built at import time from the literature sheet), giving the extractor stimulus-named HDF5s and a working `--overwrite` resume, and adding path overrides to the extraction/in-silico wrappers so a second screen cannot overwrite the literature features or predictions. New `scripts/{build_novel_grid_csv,novel_search_common,rank_novel_phase1,build_novel_phase2_csv,rank_novel_phase2}.py` + `tests/test_novel_search.py`.

**Changelog pointer:** the most recent work touches the **model→EEG mapping** (§4) and the **layer-selection CV** — the committed per-model layers in §1.5 were finalised on 2026-06-18 after switching from leaky random folds to group-by-part CV. See §4.1–§4.2 and Open Questions §13.1. The MMN-design terminology (Definition 1 vs 2) was relabeled on 2026-06-21 — see §16.1/§16.2.

**2026-07-13 — three new models enabled for the D2 mTRF mapping** (prereq for the 24×7 screen; full log in `aux/handoff_enable_large_wav2vec2_models.md`):
- **whisper-large** (`large-v3`), **wav2vec2-medium** (`facebook/wav2vec2-base`, 12 layers), **wav2vec2-large** (`facebook/wav2vec2-large`, 24 layers) — pretrained. Each now has D2/surprisal delta-t features (`outputs/features/{model}-delta-t-surprisal/merged`) and a CV-chosen mTRF layer per level in `outputs/results/eeg_mapping/{model}__{level}__D2.json`.
- **New code:** `load_wav2vec2` in `audio_models.py` (raw-waveform backbone; `whisper-large`→`large-v3`); a raw-waveform causal path in `extract_features_delta_t.py` (whisper mel path byte-identical) + an over-provision guard; `configs/extraction/audio/wav2vec2_{medium,large}_layers.json`.
- **New SLURM scripts (jed/CPU, `--partition=standard`, 5 GB/CPU):** `slurm_prefetch_audio_models.sh` (avoids concurrent-download HF-cache corruption — run first), `slurm_extract_delta_t_d2.sh`, `slurm_build_surprisal_10s.sh`, `slurm_eeg_mapping_sweep_d2.sh`.
- **Caveats:** wav2vec2 was mapped on **10 s/10 s** windows (features extracted at 10 s/5 s and reused as the even-offset subset), vs whisper's 30 s/10 s — so wav2vec2 test r isn't strictly comparable to whisper's. Both used `pca_var=None`; **no PCA was applied to either.** The sweep is slow under sklearnex (patched `RidgeCV` lacks `gcv_mode='eigen'`); numpy's OpenBLAS is capped at 2 threads in this env.

**Scope.** This document covers every executable / workflow-relevant script in the repository:
- Python modules with a CLI / `__main__` (under `src/mbs/` and `scripts/`).
- Registered console entry points (`pyproject.toml [project.scripts]`).
- Shell and SLURM job scripts under `scripts/`.

**Not covered (deliberately):**
- **Config files** (`configs/**/*.yaml`, `configs/extraction/audio/*.json`) — described where a script consumes them, but not individually documented. They exist and are extensive.
- **Tests** (`tests/*.py`) — exist (pytest); referenced only in §12.
- **Library/helper modules without a CLI** (e.g. `src/mbs/evaluation/attn_probe/engine_temporal.py`, `model.py`, `eeg_targets.py` helpers) — documented as dependencies of the CLIs that use them, not as standalone entries.
- **Notebooks** — the `visualization/main` and `visualization/supp` trees contain notebook/figure artifacts (commit `2cedb01` "Add visualization notebooks"); not documented here.
- **Literature text** (`docs/literature/*.txt`, `docs/literature/extract_pdf_text.py`) — reference material for stimulus design, not pipeline code.

**Orientation.** This repository is a **fork of the published ICML-2026 "Multimodal Scaling Laws for … Visual Cortex" codebase** that has been **extended for auditory EEG encoding and in-silico Mismatch-Negativity (MMN)** work. The document is centered on that **active auditory-EEG/MMN pipeline** (§2–§8). The original **vision→brain scaling pipeline is now reference-only and lives in §10**.

---

## Table of Contents

- [1. Repository at a Glance](#1-repository-at-a-glance)
  - [1.1 Environment setup](#11-environment-setup)
  - [1.2 Entry points](#12-entry-points)
  - [1.3 Key directories](#13-key-directories)
  - [1.4 Neural datasets (D1/D2/D3)](#14-neural-datasets-d1d2d3)
  - [1.5 Models and committed layers](#15-models-and-committed-layers)
  - [1.6 Shapes & conventions worth stating once](#16-shapes--conventions-worth-stating-once)
  - [1.7 Current status / results](#17-current-status--results)
- [2. Data Preparation](#2-data-preparation)
  - [2.1 `src/mbs/data_prep/format_eeg_hdf5.py`](#21-srcmbsdata_prepformat_eeg_hdf5py)
- [3. Feature Extraction](#3-feature-extraction)
  - [3.1 `src/mbs/extraction/extract_features_delta_t.py`](#31-srcmbsextractionextract_features_delta_tpy)
- [4. Model→EEG Mapping (fit & score)](#4-modeleeg-mapping-fit--score)
  - [4.1 `src/mbs/evaluation/evaluate_features_mtrf.py`](#41-srcmbsevaluationevaluate_features_mtrfpy)
  - [4.2 `src/mbs/evaluation/evaluate_features_mtrf_parcels.py`](#42-srcmbsevaluationevaluate_features_mtrf_parcelspy)
  - [4.3 `src/mbs/evaluation/evaluate_cross_dataset_mtrf.py`](#43-srcmbsevaluationevaluate_cross_dataset_mtrfpy)
  - [4.4 `src/mbs/evaluation/evaluate_features_attn_probe_temporal.py`](#44-srcmbsevaluationevaluate_features_attn_probe_temporalpy)
  - [4.5 `src/mbs/evaluation/evaluate_features_temporal.py`](#45-srcmbsevaluationevaluate_features_temporalpy)
  - [4.6 `scripts/eeg_mapping_sweep.py`](#46-scriptseeg_mapping_sweeppy)
  - [4.7 `scripts/eeg_mapping_encoder_cv.py`](#47-scriptseeg_mapping_encoder_cvpy)
  - [4.8 `scripts/eeg_targets.py` (shared helper)](#48-scriptseeg_targetspy-shared-helper)
- [5. In-silico MMN](#5-in-silico-mmn)
  - [5.1 `scripts/insilico_mmn.py`](#51-scriptsinsilico_mmnpy)
  - [5.2 `scripts/insilico_mmn_electrodes.py`](#52-scriptsinsilico_mmn_electrodespy)
  - [5.3 `scripts/insilico_mmn_attn.py`](#53-scriptsinsilico_mmn_attnpy)
  - [5.4 `scripts/build_mmn_results_table.py`](#54-scriptsbuild_mmn_results_tablepy)
- [6. Analysis / Scoring / Visualization](#6-analysis--scoring--visualization)
- [7. Diagnostics](#7-diagnostics)
- [8. Orchestration (shell / SLURM)](#8-orchestration-shell--slurm)
- [10. Legacy / reference-only: vision→brain scaling](#10-legacy--reference-only-visionbrain-scaling)
- [11. High-Level Workflow Overview](#11-high-level-workflow-overview)
- [12. Dependency and Execution Graph](#12-dependency-and-execution-graph)
- [13. Open Questions and Unclear Areas](#13-open-questions-and-unclear-areas)
- [14. Recommended Reading Order](#14-recommended-reading-order)
- [15. How to Run the Full Workflow](#15-how-to-run-the-full-workflow)
- [16. MMN Stimulus-Design Correction & Verdict-Criterion Rewrite (2026-06-18)](#16-mmn-stimulus-design-correction--verdict-criterion-rewrite-2026-06-18)
- [17. Stimuli Search Phase — novel tone-pair grid (2026-07-25)](#17-stimuli-search-phase--novel-tone-pair-grid-2026-07-25)
  - [17.1 What the search is and why the pipeline needed changing](#171-what-the-search-is-and-why-the-pipeline-needed-changing)
  - [17.2 Code changes](#172-code-changes)
  - [17.3 New scripts](#173-new-scripts)
  - [17.4 Cost model and budget](#174-cost-model-and-budget)
  - [17.5 Step-by-step run guide](#175-step-by-step-run-guide)
  - [17.6 Invariants and failure modes](#176-invariants-and-failure-modes)
- [18. MMN "realness" checks at FCz — N-effect and deviance scaling (2026-08-11)](#18-mmn-realness-checks-at-fcz--n-effect-and-deviance-scaling-2026-08-11)
  - [18.1 The result in one paragraph](#181-the-result-in-one-paragraph)
  - [18.2 Why a cluster re-run was needed at all](#182-why-a-cluster-re-run-was-needed-at-all)
  - [18.3 Code changes](#183-code-changes)
  - [18.4 New artefacts](#184-new-artefacts)
  - [18.5 The two re-runs](#185-the-two-re-runs)
  - [18.6 Reporting decisions worth not re-litigating](#186-reporting-decisions-worth-not-re-litigating)
  - [18.7 Interpretation limits](#187-interpretation-limits-short-form-full-list-in-the-memo)
  - [18.8 Failure modes specific to this phase](#188-failure-modes-specific-to-this-phase)

---

## 1. Repository at a Glance

**What it is.** A `src`-layout Python package (`mbs`, "multimodal-brain-scaling") plus research configs and scripts. The upstream purpose is scaling-law analysis of vision models against neural data. This fork adds an **auditory-EEG track**: feed audio into a Whisper-family encoder, learn a model→EEG mapping (two interchangeable methods), and use it to predict EEG for designed MMN tone stimuli (the "in-silico MMN").

**Where it runs.** Locally for light work, but the real pipeline runs on the **EPFL SCITAS cluster** (`jed` = CPU partition, `kuma` = GPU partition `l40s`). As of 2026-06-18, Sophie's working clone (where all new code runs and all new outputs are written) lives at:
`/work/upschrimpf1/sigfstea/multimodal-brain-scaling` — see §1.1 for the clone command. Several scripts already default to feature paths under this tree (e.g. `outputs/features/whisper-small-delta-t-surprisal/merged`), since it's also where the D2 mapping features were extracted (§5).

An earlier/reference repo install some handover docs (`aux/XX_handover_for_Sophie.md`, `aux/01_setup.md`) cite a different cluster path:
`/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling`. That tree (and `/work/upschrimpf1/mehrer/datasets/...`) is still fine to **read** from for input datasets (e.g. the Broderick BIDS data, §2.1) that aren't duplicated into Sophie's clone — but all generated `outputs/` (features, predictions, figures, results) for this MMN deliverable should land under the `sigfstea` clone, not the `mehrer` tree.

### 1.1 Environment setup

**Clone (one-time, 2026-06-18 — repo pushed to GitHub this round):**
```bash
ssh jed
cd /work/upschrimpf1/sigfstea
git clone https://github.com/hanme/multimodal-brain-scaling.git
cd multimodal-brain-scaling
```
If a clone already exists there from earlier work, `git pull origin main` instead of cloning fresh. All outputs for the MMN deliverable (§5, §16) should be written under this tree's `outputs/`, even when an input dataset is read from `mehrer`'s tree (§1, above).

```bash
# On the cluster, every session:
module load gcc/13.2.0 python/3.11.7
source env.sh            # = the two lines above + activate .venv

# One-time create (auditory extras):
UV_CACHE_DIR=/work/upschrimpf1/mehrer/.cache/uv \
uv sync --python python3.11 --extra evaluation --extra analysis --extra visualization --extra dev --extra audio
```

`env.sh` (verified) is just:
```bash
module load gcc/13.2.0 python/3.11.7
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
```

The **`audio` extra** (`openai-whisper`, `mne`, `soundfile`) gates the auditory track; **`evaluation`** gates `timm`/`transformers`/`tables`/`brainscore-vision`. Full setup, GitHub-SSH, and dataset download instructions: `aux/01_setup.md`.

### 1.2 Entry points

Registered console scripts (`pyproject.toml [project.scripts]`) — all real and verified:

| Command | Module:function | Track |
|---|---|---|
| `mbs-download-artifacts` | `mbs.download_artifacts:main` | legacy/vision (§10) |
| `mbs-train` | `mbs.training.train:cli` | legacy/vision (§10) |
| `mbs-extract-features` | `mbs.extraction.extract_features:cli` | legacy/vision (§10) |
| `mbs-extract-features-delta-t` | `mbs.extraction.extract_features_delta_t:cli` | **auditory (§3.1)** |
| `mbs-evaluate-all-layers` | `mbs.evaluation.evaluate_features_all_layers:cli` | legacy/vision (§10) |
| `mbs-evaluate-committed-layers` | `mbs.evaluation.evaluate_features_committed_layers:cli` | legacy/vision (§10) |
| `mbs-evaluate-attn-probe` | `mbs.evaluation.evaluate_features_attn_probe:cli` | legacy/vision (§10) |
| `mbs-fit-curves` | `mbs.analysis.curve_fitting.start_fitting:cli` | legacy/vision (§10) |

The core **auditory mapping/scoring modules are NOT registered as console scripts** — run them with `python -m`:

| Module | Section |
|---|---|
| `python -m mbs.data_prep.format_eeg_hdf5` | §2.1 |
| `python -m mbs.extraction.extract_features_delta_t` | §3.1 (or `mbs-extract-features-delta-t`) |
| `python -m mbs.evaluation.evaluate_features_mtrf` | §4.1 |
| `python -m mbs.evaluation.evaluate_features_mtrf_parcels` | §4.2 |
| `python -m mbs.evaluation.evaluate_cross_dataset_mtrf` | §4.3 |
| `python -m mbs.evaluation.evaluate_features_attn_probe_temporal` | §4.4 |
| `python -m mbs.evaluation.evaluate_features_temporal` | §4.5 |
| `python scripts/*.py` | §4.6–§7 |

### 1.3 Key directories

| Path | Contents |
|---|---|
| `src/mbs/` | the `mbs` package: `data_prep/`, `extraction/`, `evaluation/`, `training/`, `analysis/`, `metrics/`, `modeling/`, `visualization/`, `core/` |
| `src/mbs/evaluation/attn_probe/` | attention-encoder model, engine, dataset, checkpoint I/O (Workstream B internals) |
| `scripts/` | all shell/SLURM job scripts + auditory analysis/plot/diagnostic Python (`insilico_mmn*`, `eeg_*`, `plot_*`, `diagnose_*`, `score_*`) |
| `configs/extraction/audio/` | per-model Whisper layer lists `whisper_{tiny,base,small,medium,large}_layers.json` |
| `configs/{models,training,evaluation,analysis}/` | legacy/vision research configs (§10) |
| `data/audio_stimuli/` | 384 MMN tone WAVs `method_NN_*.wav` (standard + deviant); source for `outputs/mmn_stimuli/` (§5, §16.1) |
| `data/metadata/` | `literature_frequency_intensity_duration_metadata.csv` (MMN paradigm definitions) |
| `outputs/neural_data/` | formatted EEG HDF5s (`broderick2018_30s.h5` = D1, `surprisal_30s.h5` = D2, `d3_combined_30s.h5` = D3) — consumed everywhere; gitignored, produced on cluster |
| `outputs/features/` | extracted delta-T feature HDF5s, per model, per stimulus set (`merged/` after combining SLURM chunks) |
| `outputs/results/` | mapping JSONs, mTRF/probe score HDF5s, MMN prediction HDF5s, checkpoints (`model__<layer>.pt`) |
| `outputs/insilico_mmn_predictions*/` | in-silico MMN prediction HDF5s, one dir per model, `predictions__<layer>.h5` (parcel) + `electrode_predictions__<layer>.h5` (electrode). `…_soafix/` backs the current literature screen; `…_novel{,_phase2}/` the §17 grid search. The unsuffixed dir is the original literature run — **wrappers default to it, so a second screen must redirect (§17.6)** |
| `outputs/results_soafix_full/` | `mmn_s7_roi.csv` — the current **literature screen**: 24 frequency methods × {regular, counter} = 48 conditions per model per site, mTRF only, 7 models. Source for §7/8/8b/8c/10/11 of the counter memo (§14). `outputs/results_24freq_7models/` holds an earlier screen vintage, retained for reference |
| `outputs/results_novel_search/` | `grid_index.csv` + `phase{1,2}_mmn_s7_roi.csv` — the §17 novel grid search |
| `outputs/figures/` | all rendered figures |
| `aux/` | handover notes, project plan, setup, screening plan — **read these first** (§14) |
| `aux/analysis_with_counter/` | `results_analysis_with_counter.md` (the literature-screen results memo) + `plots/` — the figures and stats CSVs it links, regenerated by the three scripts in that dir |
| `docs/literature/` | extracted MMN-paper text for stimulus design |

### 1.4 Neural datasets (D1/D2/D3)

The auditory mapping is trained/tested on naturalistic-speech EEG, group-averaged, windowed 30 s at 10 s stride, resampled to **50 Hz (20 ms/bin)**:

| Tag | File (`outputs/neural_data/`) | Source | Notes |
|---|---|---|---|
| **D1** | `broderick2018_30s.h5` | Broderick 2018 (OpenNeuro ds004408, "Old man and the sea") | Built by §2.1. Fronto-central NC is poor (Cz r≈0.16) → not used for MMN. |
| **D2** | `surprisal_30s.h5` | Weissbart "Cortical Surprisal" speech EEG (13 subj) | **Primary training set for the MMN deliverable** (healthy FCz, r≈0.99). 157 train / 43 test 30 s windows. |
| **D3** | `d3_combined_30s.h5` | D1 ∪ D2 pooled | Used in pooling / per-dataset-scored experiments (no formatter for it in this checkout — see §13.4). |

**D2 10 s variant (2026-07-13):** `surprisal_10s.h5` = the same Weissbart EEG re-formatted at **10 s windows / 10 s stride** (the wav2vec2 mapping target — 10 s clips match wav2vec2's regime). Built by re-running `format_eeg_hdf5_surprisal.py` (in the *temporal-analysis* project, not this checkout) with `--window_duration 10 --window_stride 10`, then copied into `outputs/neural_data/`. Stimulus IDs use 16 kHz sample offsets stepping by 160000, so they align with 10 s-window delta-t features. (An earlier 10 s/**5 s** build was kept as `surprisal_10s_stride5.h5`.)

Noise ceiling (NC) is stored as **% variance explained** (`max_nc=100`); `load_neural_data` recovers Pearson r via `sqrt(nc/100)`. Channels/parcels are kept only if NC r > threshold (0.2 for the auditory targets).

### 1.5 Models and committed layers

Whisper layer lists live in `configs/extraction/audio/whisper_<size>_layers.json` (format: `[{"name":"blocks.0","position":0.0}, …]`). whisper-base has 6 blocks (`blocks.0`–`blocks.5`); small=12, medium=24, large=32, tiny=4.

**Committed per-model mapping layers (2026-06-18, fixed — do not re-select).** Source of truth: `chosen_layer` in `outputs/results/eeg_mapping{,_encoder}/<model>__<level>__D2.json`. From `aux/XX_handover_for_Sophie.md` §1:

| model | A mTRF · parcels | A mTRF · electrodes | B enc · parcels | B enc · electrodes |
|---|---|---|---|---|
| whisper-tiny | `blocks.0` | `blocks.0` | `blocks.3` | `blocks.3` |
| whisper-base | `blocks.0` | `blocks.0` | `blocks.2` | `blocks.0` |
| whisper-small | `blocks.3` | `blocks.1` | `blocks.10` | `blocks.10` |
| whisper-medium | `blocks.11` | `blocks.12` | `blocks.4` | `blocks.3` |

**New models (2026-07-13), Method A mTRF only** — source of truth `chosen_layer` in
`outputs/results/eeg_mapping/{model}__{level}__D2.json`; per-model×level table in
`aux/handoff_enable_large_wav2vec2_models.md` ("Results").

| model | A mTRF · parcels | A mTRF · electrodes | mean test r (parc / elec) |
|---|---|---|---|
| whisper-large | `blocks.21` | `blocks.21` | +0.160 / +0.180 ✅ |
| wav2vec2-medium | ⏳ pending (use `encoder.layers.0`) | ⏳ pending (use `encoder.layers.0`) | sweep running |
| wav2vec2-large | ⏳ pending (use `encoder.layers.0`) | ⏳ pending (use `encoder.layers.0`) | sweep running |

whisper-large blocks are `blocks.0`–`blocks.31`; wav2vec2 layers are `encoder.layers.0`–`{11 or 23}`.
**Until the 4 wav2vec2 sweeps finish, use `--layer encoder.layers.0` as a placeholder** in the
in-silico step; swap in the real `chosen_layer` when the JSONs land. **Method B (encoder) was not run
for these three.** Mapping-config caveat: whisper-large used 30 s/10 s (like the other
whisper models); wav2vec2-{medium,large} used **10 s/10 s** (features extracted at
10 s/5 s and reused as the even-offset subset). **All of them used `pca_var=None`** — the window
length is the only fit-protocol difference. See the 2026-07-13 changelog line above.

### 1.6 Shapes & conventions worth stating once

- **Feature HDF5** (`features/<layer-with-dashes>`): `[n_stim, T, d_model]` float16; `blocks.0` → dataset key `blocks-0`. Stimulus ids in `ids`. Whisper `T = 1500` for a 30 s clip (20 ms bins).
- **Targets:** *parcels* = 5 coarse 10-20 clusters (frontal/central/temporal/parietal/occipital), each a raw average over its NC-surviving member channels; *electrodes* = every channel passing NC r>0.2 (≈47 on D2). Defined once in `scripts/eeg_targets.py` (§4.8) and `attn_probe/dataset_temporal.py`.
- **Lags / lookback:** mTRF lag window 0–800 ms at 20 ms (`lags_in_bins`); the attention encoder uses `lookback = round(lookback_ms/20)+1` tokens.
- **Group-by-part CV:** windows from the same audiobook part overlap by 20 s, so layer-selection folds are grouped by part (`grouped_kfold`, k=4) to avoid leakage. Mirrored in the mTRF sweep and the encoder dataset.
- **MMN read-out:** time 0 = final/eliciting-tone onset; MMN = `deviant_mean − standard`, baseline-corrected on the pre-onset window, negative peak in the 100–250 ms (parcels) / 100–240 ms (electrodes) band. Design in use is **Definition 1 (literature classic-oddball, frequency-deviant)** — the deviant train's final tone differs in frequency from the standard's (§5, §16.1). **Caveat:** the current 10-method stimulus set was generated under the assumption that it implemented Definition 2 (final tone physically identical, standard/deviant distinguished only by preceding context), but every sourced paper actually uses Definition 1 — see §16.2. A true Definition 2 literature source has not yet been identified.

### 1.7 Current status / results

From the handover (`aux/XX_handover_for_Sophie.md`, 2026-06-18):
- ✅ **Mapping redone and complete** for both methods × 4 models × {parcels, electrodes}; chosen layers committed (§1.5).
- ✅ **Method A (mTRF)** ready to use now.
- ⏳ **Method B (attention encoder)** usable once final per-layer checkpoints (`model__<layer>.pt`) are produced by `scripts/kuma_probe_d2_final.sh`.
- Result artifacts present: `outputs/results/eeg_mapping*/*.json`, `outputs/results/*-probe-group-d2*/attn_probe_temporal_summary.json`, `outputs/results/*-mtrf-*/`.
- ✅ The 8 placeholder MMN frequency pairs (`method_09`, `method_12(+_counter)`, old `method_37/44/55`) have been **replaced** with the 10-pair literature classic-oddball **Definition 1** set (§5, §16.1). `outputs/mmn_stimuli`/`data/audio_stimuli` still need populating/verifying on the cluster (Part B of the handover runbook) before features can be (re-)extracted for them.
- ⚠️ **This Definition 1 stimulus set is provisional** — chosen to get the pipeline running end-to-end, not yet fully vetted against the literature; 1–2 methods may be swapped in/out later. It was built on the assumption it would be a Definition 2 (final-tone-controlled) design — it is not; see §16.2 for the caveat, the correction, and current inclusion criteria. A genuine Definition 2 literature source is still being sought.

---

## 2. Data Preparation

### 2.1 `src/mbs/data_prep/format_eeg_hdf5.py`

**Purpose:** Convert the raw Broderick 2018 BIDS EEG dataset into the `mbs` temporal HDF5 schema (D1). Loads each subject×run BrainVision file, crops to the audio duration, resamples to 50 Hz, segments into overlapping windows, group-averages across subjects, computes split-half Spearman-Brown noise ceilings, and maps standard 10-20 labels onto BioSemi128 channels.

**How to run:**
```bash
python -m mbs.data_prep.format_eeg_hdf5 \
  --bids_root  /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea \
  --output_path outputs/neural_data/broderick2018_30s.h5 \
  --window_duration 30.0 \
  --window_stride   10.0 \
  --target_sr       50 \
  --n_test_runs     4 \
  --seed            42
```
(Driven on the cluster by the EEG-format step inside `scripts/submit_whisper_small_sweep.sh`, §8.)

**Required inputs:**
- `--bids_root`: BIDS tree with `sub-XXX/eeg/sub-XXX_task-listening_run-NN_eeg.vhdr` and `stimuli/audioNN.wav`.

**Key parameters:**

| Arg | Default | Purpose |
|---|---|---|
| `--bids_root` | (required) | BIDS root of Broderick ds004408 |
| `--output_path` | (required) | Output HDF5 path |
| `--window_duration` | `30.0` | Window length (s) |
| `--window_stride` | `10.0` | Stride between windows (s) |
| `--target_sr` | `50` | EEG resample rate (Hz) = model bins/s |
| `--n_test_runs` | `4` | Runs held out as the test split |
| `--overwrite` | `False` | Regenerate if output exists |
| `--seed` | `42` | RNG for test-run choice + NC subject halves |

**Produced outputs:** one HDF5 (D1). Schema (verified):
```
attrs: subjects=["group"], rois=[...], splits=["train","test"], max_nc=100.0,
       temporal=True, T_model=<window_size>, time_step_ms, window_duration_s,
       window_stride_s, target_sr, n_subjects, dataset="broderick2018",
       test_runs=[...], channel_names=[...]
train/stimulus_ids                          # "audioNN_SSSSSSS" (start sample @16kHz)
train/neural_data/group/<roi>   [n_stim, T, n_roi_ch]
test/stimulus_ids
test/neural_data/group/<roi>
noise_ceilings/group/<roi>      [T, n_roi_ch]   # % variance explained
```
ROIs = single 10-20 electrodes (nearest BioSemi128 within 25 mm) + 5 `*_cluster` parcels + `whole_brain`.

**Step-by-step:** (1) enumerate subjects/runs, randomly pick `n_test_runs`; (2) discover channel names; (3) map 10-20 labels → BioSemi128 indices, build cluster ROIs + `whole_brain` (`_find_roi_indices`); (4) per subject×run: load, crop to audio, resample, accumulate per-window sums into two random subject halves; (5) group sums = half1+half2; (6) build per-ROI `[n_stim,T,n_ch]` arrays; (7) split-half SB-corrected NC per ROI (`_noise_ceiling_from_halves`); (8) write HDF5.

**Key functions:** `_load_run_eeg`, `_find_roi_indices`, `_noise_ceiling_from_halves`, `_get_segment_starts`, `_stim_id`, `main`.

**Main dependencies:** `mne` (lazy), `scipy.io.wavfile`, `h5py`, `numpy`, `mbs.core.str2bool`.

**Relationship to other scripts:** produces D1, the canonical NC source for parcels (`--parcels_from broderick2018_30s.h5` in §4.2/§4.3/§4.4). Stimulus-id convention must match the extractor (§3.1).

**Common failure modes:** missing `mne` (audio extra); `.vhdr` filename mismatch (expects `task-listening`, `run-NN`); MNE montage lookup failure → falls back to `whole_brain` only (silently degrades ROIs); EEG shorter than a window after resample rounding → that window dropped.

> **Note:** this formatter is **Broderick-specific**. D2 (`surprisal_30s.h5`) and D3 (`d3_combined_30s.h5`) are consumed throughout but there is **no formatter for them in this checkout** (§13.4).

---

## 3. Feature Extraction

### 3.1 `src/mbs/extraction/extract_features_delta_t.py`

**Purpose:** "Delta-T" (causal) feature extraction for Whisper-family audio models. For each stimulus and each encoder time bin `t`, build a mel spectrogram truncated to frames `[0, 2(t+1))` (rest filled with the per-clip silence value), run the encoder, and keep the layer output at position `t`. The result mimics a streaming/causal encoder and is written in the same schema as the temporal `extract_features.py`, so the evaluators consume it unmodified.

**Supports / scope:** any Whisper model id loadable by `mbs.extraction.modeling.backbones.audio_models.load_whisper`.

**How to run:**
```bash
python -m mbs.extraction.extract_features_delta_t \
  --model_id whisper-base \
  --data_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/stimuli \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --output_dir outputs/features/whisper-base-delta-t \
  --window_duration 30.0 --window_stride 10.0 \
  --batch_t 4 --t_stride 1 --save_every 8
```
Alternative: `mbs-extract-features-delta-t …` (console script). Cluster array drivers: `scripts/slurm_extract_delta_t.sh`, `scripts/slurm_extract_delta_t_generic.sh`, `scripts/slurm_mmn_extract.sh` (§8).

**Required inputs:**
- `--data_root`: directory of `*.wav` stimuli.
- `--target_feature_layers`: JSON list `[{"name":"blocks.0"}, …]`.

**Key parameters:**

| Arg | Default | Purpose |
|---|---|---|
| `--model_id` | (required) | Whisper model id |
| `--data_root` | (required) | Dir of `.wav` files |
| `--target_feature_layers` | (required) | Layer JSON |
| `--output_dir` | (required) | Feature HDF5 dir |
| `--window_duration` | `30.0` | Sliding window (s) |
| `--window_stride` | `10.0` | Window stride (s) |
| `--batch_t` | `16` | # truncated mels per forward pass (memory/speed) |
| `--t_stride` | `1` | Sub-sample time bins (50 → 1 s pilot) |
| `--stim_start_idx` | `0` | Start stimulus index (parallelism) |
| `--n_stimuli` | `0` | Process N stimuli (0 = all remaining) |
| `--save_every` | `8` | Stimuli accumulated per output HDF5 |
| `--model_cache_dir` | `cache/model_weights` | Weights cache |
| `--seed` | `42` | — |
| `--overwrite` | `False` | — |

**Produced outputs:** one or more HDF5 chunk files per task:
```
feats_delta_t-start_<NNNNN>-batch_<i>-seed_42.h5
  features/<layer-dashes>   [n_stim, T_out, d_model] float16 (gzip)
  ids                       [n_stim] str
  attrs: model_id, backbone_source="audio", target_feature_layers,
         extraction_mode="delta_t", config_json
```
Chunks from parallel tasks must be combined into a `merged/` directory (consumed downstream as `--features_dir .../merged`).

**Step-by-step:** (1) load Whisper + transform; (2) wrap encoder in `HookedEncoder` for the requested layers; (3) build `AudioSegmentDataset` (16 kHz, raw waveform); (4) per stimulus compute full mel once; (5) `extract_delta_t` runs batched truncated-mel passes and collects position-`t` vectors per layer; (6) flush every `save_every` stimuli.

**Key functions:** `extract_delta_t`, `_truncate_mel`, `_silence_value`, `_write_batch`, `main`/`cli`.

**Main dependencies:** `torch`, `h5py`, `tqdm`; internal `load_whisper`, `HookedEncoder`, `AudioSegmentDataset`.

**Relationship to other scripts:** its `merged/` output is the `--features_dir`/`--train_features`/`--mmn_features_root` input for everything in §4–§5. The stimulus-id scheme matches §2.1 so EEG and features align.

**Common failure modes:** no `.wav` in `--data_root` (assertion); layer name absent from the model; parallel chunks left un-merged → downstream id-alignment drops rows with a "matched IDs" warning.

---

## 4. Model→EEG Mapping (fit & score)

Two interchangeable mappings (model features → EEG): **Method A = closed-form lagged Ridge (mTRF)** and **Method B = gradient-trained attention encoder**. Both fit on D2 train, score held-out test, select layers via group-by-part CV, and write a parallel `heldout_r` schema.

### 4.1 `src/mbs/evaluation/evaluate_features_mtrf.py`

**Purpose:** Lagged shared-weight RidgeCV ("mTRF") for temporal audio features vs EEG, over ROIs/channels in the neural HDF5. Two modes: `single_lag` (encoding-vs-latency curve, `scores[n_lags, n_ch]`) and `fir` (one multivariate-FIR r per channel, `scores[1, n_ch]`). This is the **shared engine**: many scripts import its `lags_in_bins`, `build_lagged_design`, `highpass_along_time`, `sample_time_indices`, `pearson_along_time`.

**How to run:**
```bash
python -m mbs.evaluation.evaluate_features_mtrf \
  --model_id whisper-base \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --features_dir outputs/features/whisper-base-delta-t/merged/ \
  --data_hdf5_path outputs/neural_data/broderick2018_30s.h5 \
  --output_dir outputs/results/whisper-base-mtrf-hp0p5/layer_0 \
  --mode single_lag --layer_id 0 \
  --lag_min_ms 0 --lag_max_ms 800 --lag_step_ms 20 \
  --highpass_hz 0.5 --n_train_time_samples 200 --overwrite true
```
Cluster driver: `scripts/slurm_mtrf.sh` (layer×high-pass sweep, §8).

**Required inputs:** layer JSON; `--features_dir` (merged delta-T); `--data_hdf5_path` (EEG HDF5 with `noise_ceilings`).

**Key parameters (selected):**

| Arg | Default | Purpose |
|---|---|---|
| `--mode` | `single_lag` | `single_lag` curve vs `fir` mTRF |
| `--lag_min_ms`/`--lag_max_ms`/`--lag_step_ms` | `0`/`800`/`20` | lag grid |
| `--highpass_hz` | `0.0` | high-pass EEG+features (off by default) |
| `--n_train_time_samples` | `200` | random training time points/segment |
| `--test_time_stride` | `1` | stride when scoring |
| `--feature_pca` | `0` | PCA components (>0 enables) |
| `--standardize_features` | `True` | z-score features on train stats |
| `--nc_threshold` | `0.0` | drop channels with mean NC ≤ this |
| `--noise_ceiling_correct` | `True` | divide r by NC |
| `--roi_allowlist` | `""` | comma-sep ROI subset |
| `--exclude_whole_brain` | `True` | skip `whole_brain` |
| `--layer_id` | `None` | run only this layer index |
| `--overwrite` | `False` | — |

**Produced outputs:** `mtrf_scores.h5` with `attrs: lags_bins, lags_ms, mode` and per-result group `<layer-dashes>/<subject>/<roi>` containing `scores_raw [n_rows,n_ch]`, `scores_nc`, `kept_channels`; plus `mtrf_scores_summary.json`. (Note the `<layer>/<subject>/<roi>` key layout — see §13.3.)

**Step-by-step:** load layers; read subjects/ROIs/NC; build lags; per (layer, subject) assemble one channel block across ROIs sharing id-alignment, mask by NC, `fit_score_block` (one RidgeCV per lag set with `alpha_per_target`), split back to ROIs, write.

**Key functions:** `lags_in_bins`, `sample_time_indices`, `build_lagged_design`, `pearson_along_time`, `highpass_along_time`, `fit_score_block`, `main`/`cli`.

**Relationship:** the engine for §4.2, §4.6, §5.1–§5.3 (they import its functions). Reads §3.1 features + §2.1 EEG.

**Common failure modes:** features not 3D (layer skipped); no id overlap (skip with message); chosen RidgeCV alpha at grid edge (widen alphas).

### 4.2 `src/mbs/evaluation/evaluate_features_mtrf_parcels.py`

**Purpose:** Parcel-level mTRF encoder with **per-dataset held-out scoring** (the MMN-free, importable counterpart of the fit inside `insilico_mmn.py`). Fits one lagged RidgeCV (features → 5 NC-parcels) on train, scores each held-out split separately (for D3, `test_d1`/`test_d2` never pooled). Provides `fit_parcel_mtrf`/`score_parcel_mtrf`/`cross_score_dataset` reused by §4.3.

**How to run:**
```bash
python -m mbs.evaluation.evaluate_features_mtrf_parcels \
  --model_id whisper-base \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --data_hdf5_path outputs/neural_data/surprisal_30s.h5 \
  --features_dir   outputs/features/whisper-base-delta-t-surprisal/merged \
  --output_dir     outputs/results/whisper-base-mtrf-parcels-d2/ \
  --parcels_from   outputs/neural_data/broderick2018_30s.h5 \
  --highpass_hz 0.5 --lag_max_ms 800 --n_train_time_samples 200 --overwrite true
```
Cluster driver: `scripts/slurm_mtrf_parcels.sh` (array over D1/D2/D3, §8).

**Key parameters:** `--parcels_from` (default `outputs/neural_data/broderick2018_30s.h5` — defines canonical parcel membership; NC recomputed on `--data_hdf5_path`), `--nc_threshold` (0.2), `--lag_max_ms` (800), `--pca_var` (optional, e.g. 0.95 — PCA features before lagging to dodge LAPACK overflow on wide models / pooled D3), `--layer_id`, `--overwrite`, `--use_wide_range_alphas`.

**Produced outputs:** `mtrf_parcel_scores.h5`:
```
attrs: highpass_hz, fs=50, lag_max_ms, nc_threshold, pca_var, splits
<layer-dashes>/parcels, parcel_nc_r,
              heldout_r__<split>, heldout_r_nc__<split>   # per split (test | test_d1 | test_d2)
              attrs: n_pcs (if PCA)
```
plus `mtrf_parcel_summary.json`.

**Key functions:** `fit_parcel_mtrf`, `score_parcel_mtrf`, `cross_score_dataset`, `_aligned`, `main`/`cli`.

**Main dependencies:** imports lag/score helpers from §4.1, and `build_parcels`/`recompute_parcel_nc`/`list_test_splits`/`load_parcel_eeg`/`parcel_nc_vector` from `attn_probe/dataset_temporal.py`.

**Relationship:** shares the `heldout_r__<split>` schema with §4.4 so mTRF and encoder sit side by side; its fit/score functions are reused by §4.3.

### 4.3 `src/mbs/evaluation/evaluate_cross_dataset_mtrf.py`

**Purpose:** Pure out-of-domain transfer: fit the parcel-mTRF on a SOURCE train split, then from the same fit score both the source's own held-out test (in-domain) and the TARGET dataset's test (transfer). Source feature standardisation is applied to the target (no peeking).

**How to run:**
```bash
python -m mbs.evaluation.evaluate_cross_dataset_mtrf \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --source_tag d1 --source_data_hdf5 outputs/neural_data/broderick2018_30s.h5 \
  --source_features_dir outputs/features/whisper-base-delta-t/merged/ \
  --target_tag d2 --target_data_hdf5 outputs/neural_data/surprisal_30s.h5 \
  --target_features_dir outputs/features/whisper-base-delta-t-surprisal/merged/ \
  --output_dir outputs/results/whisper-base-mtrf-xfer-d1-to-d2/ \
  --highpass_hz 0.5 --lag_max_ms 800 --n_train_time_samples 200 --overwrite true
```
Cluster driver: `scripts/slurm_cross_mtrf.sh` (array d1→d2 and d2→d1, §8).

**Key parameters:** `--source_tag/--source_data_hdf5/--source_features_dir`, `--target_tag/--target_data_hdf5/--target_features_dir` (all required), `--parcels_from`, `--nc_threshold` (0.2), `--lag_max_ms` (800), `--pca_var`, `--layer_id`, `--overwrite`.

**Produced outputs:** `cross_mtrf_scores.h5` with `attrs: source, target` and per layer `parcels`, `parcel_nc_r`, `heldout_r__<source_tag>_<split>` (in-domain), `heldout_r__<target_tag>_<split>` (transfer); plus `cross_mtrf_summary.json`.

**Key functions:** `main`/`cli`; reuses `fit_parcel_mtrf`/`score_parcel_mtrf`/`cross_score_dataset` from §4.2.

### 4.4 `src/mbs/evaluation/evaluate_features_attn_probe_temporal.py`

**Purpose:** **Method B** — the learned temporal attention probe (MIRAGE-style). A shared latent-attention trunk attends over the lookback window; a subject head reads out target EEG. Trained with **MSE** + random time-point sampling, MIRAGE-style checkpoint selection on a held-out fold, scored along time on the built-in test split(s). EEG targets are z-scored (stats stored in the checkpoint so predictions invert to real units). Optionally saves a reusable `model__<layer>.pt` for in-silico MMN.

**Supports / scope:** `--readout_level {individual,group}` × `--target_level {parcels,electrodes}`; per-layer or all-layers.

**How to run (final checkpoint at a chosen layer, GPU):**
```bash
python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
  --model_id whisper-small \
  --target_feature_layers configs/extraction/audio/whisper_small_layers.json \
  --data_hdf5_path outputs/neural_data/surprisal_30s.h5 \
  --features_dir   <whisper-small D2 feats>/merged \
  --output_dir     outputs/results/whisper-small-probe-group-d2-parcels \
  --readout_level group --target_level parcels \
  --parcels_from outputs/neural_data/surprisal_30s.h5 \
  --highpass_hz 0.5 --lookback_ms 800 --nc_threshold 0.2 \
  --d_model 64 --num_latents 4 --cross_attn_layers 1 --dropout 0.3 \
  --weight_decay 1e-2 --epochs 200 --lr 1e-3 --n_train_time_samples 200 \
  --layer_id 10 --save_model true --amp false --device cuda --overwrite true
```
Cluster drivers (§8): `kuma_probe_d2_cv.sh` (CV folds) → `jed_collect_encoder_cv.sh` (§4.7) → `kuma_probe_d2_final.sh` (final ckpt); also `run_probe_d2d3.sh`, `kuma_probe_mmn.sh`.

**Key parameters (selected):**

| Arg | Default | Purpose |
|---|---|---|
| `--readout_level` | `group` | one head per subject vs single group head |
| `--target_level` | `parcels` | parcels vs electrodes |
| `--parcels_from` | `outputs/neural_data/surprisal_30s.h5` | canonical target membership (NC recomputed on data) |
| `--lookback_ms` | `800.0` | attention lookback |
| `--highpass_hz` | `0.5` | high-pass features+EEG |
| `--nc_threshold` | `0.2` | NC floor |
| `--val_mode` | `grouped` | grouped (non-overlapping fold) vs random |
| `--n_folds` / `--fold_idx` | `4` / `0` | grouped CV fold selection |
| `--eval_every` | `5` | epochs between val-Pearson checkpoint selection |
| `--d_model`/`--num_latents`/`--cross_attn_layers`/`--nhead`/`--dropout`/`--pos_mode` | `256`/`16`/`2`/`8`/`0.1`/`learned` | probe capacity |
| `--epochs`/`--lr`/`--weight_decay`/`--batch_size`/`--n_train_time_samples` | `200`/`1e-3`/`1e-4`/`512`/`200` | training |
| `--device`/`--amp` | `cuda`/`True` | — |
| `--layer_id` | `None` | only this layer index |
| `--save_model` | `True` | write `model__<layer>.pt` |

**Produced outputs:** `attn_probe_temporal_scores.h5`:
```
attrs: readout_level, target_level, val_mode, n_folds, fold_idx,
       lookback_ms, lookback_bins, highpass_hz, fs=50, nc_threshold
<layer-dashes>/parcels, parcel_nc_r,
              heldout_r__<split> [P], heldout_r_nc__<split> [P],
              heldout_r__val [P]              # selection fold (when val present)
              heldout_r_persubj__<split>      # individual readout only
```
plus `attn_probe_temporal_summary.json` and (if `--save_model`) `model__<layer>.pt` (state_dict + ProbeConfig + feature mu/sd + lookback + parcels + eeg_mu/eeg_sd + meta — see §5.3).

**Key functions:** `run_layer`, `_aligned_feats`, `main`/`cli`; uses `train_temporal_probe`/`score_heldout` (`engine_temporal.py`), `ProbeConfig`/`SingleRoiProbeSystem` (`model.py`), `save_probe_checkpoint` (`checkpoint.py`), and the parcel/electrode builders in `dataset_temporal.py`.

**Relationship:** the §4.7 collector reads its `heldout_r__val`/`heldout_r__test` across folds to pick the layer; its checkpoints feed §5.3.

**Common failure modes:** `--readout_level group` requires a `group` subject in the HDF5; `individual` requires per-subject EEG (re-run §2.1 with subject storage); features not 3D → layer skipped.

### 4.5 `src/mbs/evaluation/evaluate_features_temporal.py`

**Purpose:** Baseline **per-time-step** RidgeCV (fit independently at each of the T time bins, score by across-stimuli Pearson r). Superseded by the mTRF (§4.1) for continuous stimuli but still used for the D1 full-dataset temporal-score diagnostics consumed by §6 plots.

**How to run:**
```bash
python -m mbs.evaluation.evaluate_features_temporal \
  --model_id whisper-base \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --features_dir   outputs/features/whisper-base-delta-t/merged \
  --data_hdf5_path outputs/neural_data/broderick2018_30s.h5 \
  --output_dir     outputs/results/whisper-base-delta-t-full
```
Cluster driver: `scripts/resubmit_eval_temporal_full.sh` (§8).

**Key parameters:** `--noise_ceiling_correct` (True), `--use_wide_range_alphas` (False), `--overwrite` (False), `--seed` (42).

**Produced outputs:** `temporal_scores.h5` keyed `<layer-dashes>/<subject>/<roi>` → `scores[T, n_ch]`; plus `temporal_scores_summary.json`. Resumable (append mode; skips computed keys).

**Relationship:** its `temporal_scores.h5` is the input to §6 `plot_temporal_scores.py` and `plot_score_distributions.py`.

### 4.6 `scripts/eeg_mapping_sweep.py`

**Purpose:** mTRF **layer sweep on one dataset (default D2) with CV-on-train layer selection**. For each layer: fit on train, score with group-by-part k-fold CV within train (selection signal); pick max-CV layer; refit on all train and score held-out test. Writes one JSON per (model × target level) that `plot_eeg_mapping.py` renders.

**How to run:**
```bash
python scripts/eeg_mapping_sweep.py \
  --model_id whisper-small --target_level parcels \
  --features_dir <whisper-small D2 feats>/merged \
  --neural outputs/neural_data/surprisal_30s.h5 \
  --out outputs/results/eeg_mapping/whisper-small__parcels__D2.json
```
Cluster driver: `scripts/slurm_eeg_mapping_sweep.sh` (8-task array = 4 models × 2 levels, §8).

**Key parameters:**

| Arg | Default | Purpose |
|---|---|---|
| `--model_id` | (required) | — |
| `--target_level` | (required) | `parcels`\|`electrodes` |
| `--features_dir` | (required) | D2 features (merged) |
| `--neural` | `outputs/neural_data/surprisal_30s.h5` | EEG HDF5 |
| `--layers_config` | `""` | override; else `configs/extraction/audio/<model>_layers.json` |
| `--n_folds` | `4` | group-by-part CV folds |
| `--nc_r_threshold` | `0.2` | NC floor |
| `--highpass_hz` | `0.5` | high-pass |
| `--lag_max_ms` | `800.0` | lag window |
| `--n_train_time_samples` | `120` | training time points |
| `--alpha_log_min/max`, `--alpha_n` | `1.0`/`7.0`/`25` | ridge alpha grid |
| `--pca_var` | `None` | PCA features before lagging |
| `--seed` | `42` | — |
| `--out` | (required) | output JSON |

**Produced outputs:** JSON with `layers`, `positions`, `targets`, `nc_r`, `cv_r_by_layer`, `test_r_by_layer`, `cv_score_by_layer`, `chosen_idx`, `chosen_layer`, `test_r_chosen`, `cv_score_chosen`. **This `chosen_layer` is the committed mTRF layer in §1.5.**

**Key functions:** `score_layer`, `fit_predict`, `maybe_pca`, `layers_for`, `main`. Reuses §4.1 helpers + §4.8 targets.

### 4.7 `scripts/eeg_mapping_encoder_cv.py`

**Purpose:** Aggregate the attention-encoder group-by-part CV folds (from `kuma_probe_d2_cv.sh`) into one `eeg_mapping`-schema JSON per (model × level), so `plot_eeg_mapping.py` renders Method B exactly like Method A. Per layer it averages `heldout_r__val` and `heldout_r__test` over the present folds; the chosen layer maximises mean-fold val r.

**How to run:**
```bash
python scripts/eeg_mapping_encoder_cv.py \
  --model_id whisper-small --target_level parcels \
  --cv_dir outputs/results/whisper-small-probe-group-d2-parcels-cv \
  --out outputs/results/eeg_mapping_encoder/whisper-small__parcels__D2.json
```
Cluster driver: `scripts/jed_collect_encoder_cv.sh` (loops 4 models × 2 levels, then calls §6 `plot_eeg_mapping.py`, §8).

**Key parameters:** `--cv_dir` (required, has `fold*/attn_probe_temporal_scores.h5`), `--layers_config` (default derived), `--n_folds` (4), `--out` (required).

**Produced outputs:** JSON with the same key set as §4.6 plus `method="attn_encoder_cv"`, `selection`. **`chosen_layer` here is the committed encoder layer in §1.5.**

### 4.8 `scripts/eeg_targets.py` (shared helper)

**Purpose:** Dataset-agnostic EEG **target builders** shared by §4.6 and §5.1–§5.3. A target is `(name, member_channels, reliability_r)`. Provides parcels (5 coarse 10-20 clusters), electrodes (each NC-passing channel), the group-by-part CV folder, and the per-split loader.

**Not a CLI** (no `__main__`). Key exports: `FS=50.0`, `TIME_STEP_MS=20.0`, `CLUSTERS`, `build_parcels`, `build_electrodes`, `build_targets`, `load_split_targets(..., return_ids=…)`, `grouped_kfold`, `part_group`, `channel_r`, `montage_pos`, `decode`.

> **Note:** the function is named `load_split_targets` here. Three scripts (§5.3, §6.1, §6.3) used to import the stale name `load_split_parcels` and fail at import time — fixed 2026-06-18, see §13.2.

---

## 5. In-silico MMN

Drive the model→EEG mapping with designed MMN tone stimuli and read `deviant − standard` time-locked to the final tone. MMN stimulus dirs are `outputs/mmn_stimuli/<method>` with features at `outputs/features/mmn-<method>-delta-t` (built by `scripts/slurm_mmn_extract.sh`, §8). **Method registry (`METHODS` in `insilico_mmn.py`, updated 2026-06-18, §16.1):** 10 literature classic-oddball (**Definition 1**) pairs sourced from `data/metadata/literature_frequency_intensity_duration_metadata.csv` — `method_75, method_74, method_72, method_60, method_53, method_55, method_37, method_43, method_44, method_27` (Karger_2014 … Schall_1999a). Each method has 1 standard file + 15 deviant files (`N∈{3,5,7} × var∈{1..5}`); the deviant train's *last* tone differs in frequency from the standard's. **This Definition 1 stimulus set is provisional, not yet fully vetted — see §16.2.** (The 8-entry identity-MMN registry this section used to describe — `method_37, method_12, method_44, method_09, method_55` + `_counter` variants, final tone physically identical in std/dev (Definition 2, discontinued, not currently used) — is gone; see §16.1 for why the old and new rounds collide on some `method_id` numbers without being the same stimuli.)

### 5.1 `scripts/insilico_mmn.py`

**Purpose:** **Method A in-silico MMN at parcel level.** Fit a FIR mTRF on the training EEG (default D2) at one layer with electrodes aggregated into NC-floored parcels, apply the single mapping to each MMN method's delta-T features, time-lock to the final-tone onset, then split into two separate computations (`finalize_method()`, §16.3): **(a)** mean-only baseline correction (window = `3×SOA` ms before onset, SOA from the metadata CSV) for the **plotted** `dev_b`/`std_b`/`diff_b` traces, in the original prediction units; **(b)** a full z-score of dev/std within that same window, differenced, most-negative point in `[100,240]` ms → the **verdict-only** `baseline_normalized_peak` (never plotted, just annotated as a number) plus an `n7v1_peak` diagnostic from the single `N7/var1` deviant trace alone. Plots restrict rows to frontal/central/temporal parcels. Optionally also scores held-out test r (`--eval_heldout`).

**How to run:**
```bash
python scripts/insilico_mmn.py \
  --train_features <D2 feats>/merged \
  --train_neural outputs/neural_data/surprisal_30s.h5 \
  --mmn_features_root outputs/features \
  --layer blocks.3 --lag_max_ms 800 --methods all
```
Cluster driver: `scripts/slurm_insilico_mmn.sh` (optional `--layer blocks.{0..5}` array, §8).

**Key parameters (selected):** `--train_features`/`--train_neural` (aliases `--broderick_features`/`--broderick_neural`; default D2 whisper-small feats + `surprisal_30s.h5`), `--mmn_features_root` (`outputs/features`), `--stimuli_root` (`outputs/mmn_stimuli`), `--layer` (`blocks.3`), `--methods` (`all`), `--nc_r_threshold` (0.2), `--highpass_hz` (0.5), `--lag_max_ms` (**800**, fixed 2026-06-18 — was `500`, §13.7), `--metadata_csv` (the literature CSV, for the SOA lookup), `--n_train_time_samples` (120), `--eval_heldout` (True), `--alpha_log_min/max`/`--alpha_n` (1/7/25), `--win_pre_ms`/`--win_post_ms` (150/500 — plot range only, the baseline/verdict window comes from SOA), `--out_dir` (`outputs/figures/insilico_mmn`), `--data_dir` (`outputs/insilico_mmn_predictions`).

**Produced outputs:**
- Figures `outputs/figures/insilico_mmn/insilico_mmn__<method>__<layer>.png` — 3 rows (frontal/central/temporal) × 3 columns (deviant/standard/deviant−standard, all mean-baseline-corrected), 3rd column annotated with `baseline_normalized_peak`.
- Predictions `outputs/insilico_mmn_predictions/predictions__<layer>.h5`:
```
attrs: layer, highpass_hz, lag_max_ms, fs=50, time_step_ms=20, nc_r_threshold, note
parcels [S12], parcel_members, parcel_nc_r, (heldout_r, heldout_r_nc if eval)
<method>/ time_ms, standard [n_t,P], deviant_mean [n_t,P], deviants [n_dev,n_t,P], deviant_ids,
          peak [P] (z-scored baseline_normalized_peak), n7v1_peak [P] (N7/var1 diagnostic)
```

**Step-by-step:** build parcels (NC floor) → `fit_mapping` once for the layer → load the SOA table once (`load_soa_table`) → for each method: load delta-T feats, predict per stimulus (`predict_timecourse`), detect final-tone onset (`detect_final_tone_onset_s`), then `finalize_method()` (time-lock, mean-correct for plotting, z-score for the verdict peak), plot + store.

**Key functions:** `fit_mapping`, `evaluate_heldout`, `predict_timecourse`, `analyze_method`, `finalize_method`, `load_soa_table`, `soa_for_method`, `plot_method`, `detect_final_tone_onset_s`, `main`. Imports `build_parcels`/`load_split_targets` from §4.8 and lag helpers from §4.1.

**Common failure modes:** missing standard or deviant in a method dir → method skipped; MMN feature dir absent → skipped; ridge alpha at grid edge (warning).

### 5.2 `scripts/insilico_mmn_electrodes.py`

**Purpose:** Electrode-level in-silico MMN with a **topographic** 10-20 montage plot and an **automatic MMN verdict** — the z-scored `baseline_normalized_peak` from `finalize_method()` (§5.1/§16.3), averaged over a fronto-central ROI; negative beyond `--mmn_thresh` → "MMN present". The plotted traces are still the mean-baseline-corrected `diff_b`, never the z-scored arrays. Reuses `fit_mapping`/`analyze_method` from §5.1 with electrodes as singleton targets.

**How to run:**
```bash
python scripts/insilico_mmn_electrodes.py --layer blocks.1 --lag_max_ms 800 --methods all
```
Cluster driver: `scripts/slurm_insilico_mmn_electrodes.sh` (§8).

**Key parameters (extra vs §5.1):** `--layer` (default `blocks.10`), `--mmn_lo_ms`/`--mmn_hi_ms` (100/240 — **plot-shading only** as of 2026-06-18; the verdict window is fixed inside `finalize_method`, not parameterized here), `--mmn_thresh` (0.0), `--win_post_ms` (400), `--metadata_csv` (SOA lookup, default `insilico_mmn.DEFAULT_SOA_CSV`), `--out_dir` (`outputs/figures/insilico_mmn_electrodes`); FC ROI = `Fz,FCz,Cz,FC1,FC2,F1,F2` (intersected with surviving electrodes).

**Produced outputs:** figures `insilico_mmn_electrodes__<method>__<layer>.png`; predictions `electrode_predictions__<layer>.h5` with per-method `roi_baseline_normalized_peak`/`mmn_present` attrs (renamed from `roi_mmn_amp` 2026-06-18) plus `peak`/`n7v1_peak` datasets; prints a per-method MMN screen summary (`N/total pairs show an MMN`).

**Key functions:** `mmn_metric` (now just ROI-averages the precomputed `peak` array, no longer recomputes a raw mean amplitude), `plot_topo`, `main`; `build_electrodes`/`montage_pos` from §4.8.

### 5.3 `scripts/insilico_mmn_attn.py`

**Purpose:** **Method B in-silico MMN** — same time-locking/plots as §5.1 but the predictor is a **trained attention-encoder checkpoint** (`model__<layer>.pt` from §4.4) instead of an on-the-spot ridge. Also renders a held-out fit-quality figure (recorded vs predicted parcel EEG).

**How to run:**
```bash
python scripts/insilico_mmn_attn.py \
  --checkpoint outputs/results/whisper-small-probe-group-d2-parcels/model__blocks.10.pt \
  --mmn_features_root outputs/features --method method_37 \
  --features_dir <D2 feats>/merged \
  --neural outputs/neural_data/surprisal_30s.h5 \
  --out_dir outputs/figures/insilico_mmn/whisper-small \
  --data_dir outputs/insilico_mmn_predictions/whisper-small
```
(Model-namespaced `--out_dir`/`--data_dir` — see §5.4/§15 Stage E for why; the argparse defaults still say `..._small` and need overriding per model.) Cluster driver: `scripts/slurm_insilico_mmn_attn.sh` (§8).

**Key parameters:** `--checkpoint` (required), `--mmn_features_root` (required), `--method` (`method_37`, fixed 2026-06-18 — was the now-nonexistent `method_09`), `--metadata_csv` (SOA lookup, default `insilico_mmn.DEFAULT_SOA_CSV`), `--features_dir`/`--neural` (required, for fit-quality), `--win_pre_ms`/`--win_post_ms` (150/500 — plot range only), `--window_idx` (-1), `--device` (`cpu`), `--out_dir`/`--data_dir`, `--tag` (`attn`).

**Produced outputs:** `insilico_mmn__<method>__<layer>__attn.png`, `fit_quality__<tag>__<layer>__attn.png`, and `predictions__<layer>__attn.h5` (same per-method schema as §5.1).

**Key functions:** `analyze_method_attn` (now just predicts per stimulus, then delegates to the shared `finalize_method` from §5.1/§16.3 instead of duplicating the baseline logic), `fit_quality_figure`, `parcels_from_ckpt`, `main`. Uses `load_probe_checkpoint`/`predict_timecourse`/`predictions_to_units` from `attn_probe/checkpoint.py`; reuses `METHODS`/`finalize_method`/`load_soa_table`/`soa_for_method`/`plot_method`/`detect_final_tone_onset_s` from §5.1; `load_split_targets` from `eeg_targets` (§4.8) for the fit-quality split.

> **Amplitude:** Method B is MSE-trained in z-units; invert with `predictions_to_units` before reading magnitudes. The additive mean cancels in `deviant − standard`.

### 5.4 `scripts/build_mmn_results_table.py`

**Purpose:** Combine the per-model/method prediction HDF5s from §5.1 (mTRF) and §5.3 (encoder) into one results table — the "Combined results table" deliverable: 10 pairs × 4 models × {mTRF, encoder} × `baseline_normalized_peak` per parcel, plus the `n7v1_peak` diagnostic per parcel. Added 2026-06-18 (§16.3).

**How to run:**
```bash
python scripts/build_mmn_results_table.py \
  --predictions_root outputs/insilico_mmn_predictions --out outputs/results/mmn_results_table.csv
```

**Expects the model-namespaced directory convention** (not the §5.1/§5.3 defaults — pass `--out_dir`/`--data_dir` per model when running those):
```
outputs/insilico_mmn_predictions/<model>/predictions__<layer>.h5         # mTRF
outputs/insilico_mmn_predictions/<model>/predictions__<layer>__attn.h5   # encoder
```

**Key functions:** `rows_from_h5` (one row per `<model, mapping, method>`, columns `<parcel>_peak`/`<parcel>_n7v1_peak`), `main`. No analysis logic of its own — pure HDF5→CSV reshaping via `h5py`/`csv`.

**Outputs:** one CSV at `--out` (default `outputs/results/mmn_results_table.csv`); writes nothing (with a printed warning) if `--predictions_root` doesn't exist or has no matching HDF5s yet.

---

## 6. Analysis / Scoring / Visualization

All Python; abbreviated template. (Detailed argparse verified from source.)

### 6.1 `scripts/score_mtrf_fitquality.py`
**Purpose:** Re-score mTRF fit quality "B-identically" — correlate along all valid time points across all held-out test windows (matching the attention-encoder scoring) rather than the lighter random-sample eval inside §5.1.
**Run:**
```bash
python scripts/score_mtrf_fitquality.py \
  --features_dir <D2 feats>/merged --neural outputs/neural_data/surprisal_30s.h5 \
  --layer blocks.10 --out outputs/results/mtrf_fitquality_b_identical.json
```
**Key params:** `--features_dir`/`--neural` (required), `--layer` (`blocks.10`), `--nc_r_threshold` (0.2), `--highpass_hz` (0.5), `--lag_max_ms` (800), `--n_train_time_samples` (200), alpha grid (1/7/25), `--out` (`""` → stdout only).
**Outputs:** stdout per-parcel table + optional JSON. Imports `load_split_targets` from `eeg_targets` (fixed 2026-06-18, was the broken `load_split_parcels` — §13.2).

### 6.2 `scripts/plot_eeg_mapping.py`
**Purpose:** From `eeg_mapping[_encoder]` JSONs (§4.6/§4.7), draw layer-selection curves (CV vs test r, chosen layer circled) and held-out test-r bars.
**Run:** `python scripts/plot_eeg_mapping.py --results_dir outputs/results/eeg_mapping --target_level parcels`
**Key params:** `--results_dir` (`outputs/results/eeg_mapping`), `--target_level` (**required**, `parcels|electrodes`), `--out_dir` (`outputs/figures/eeg_mapping`).
**Outputs:** `layer_selection__<level>__D2.png`, `test_fit_quality__<level>__D2.png`.

### 6.3 `scripts/plot_fit_quality.py`
**Purpose:** Recorded vs predicted parcel EEG on one held-out test window (the "is the fit good?" figure), using the §5.1 mTRF.
**Key params:** `--features_dir`/`--neural`/`--out` (required), `--layer` (`blocks.10`), `--nc_r_threshold` (0.2), `--highpass_hz` (0.5), `--lag_max_ms` (800), `--n_train_time_samples` (200), `--window_idx` (-1 = median), `--max_seconds` (0), `--label` (`mTRF`).
**Outputs:** one PNG at `--out`. Imports `load_split_targets` from `eeg_targets` (fixed 2026-06-18, was the broken `load_split_parcels` — §13.2).

### 6.4 `scripts/plot_fit_quality_bars.py`
**Purpose:** Cross-model bar chart of per-parcel held-out test r at each model's hardcoded best layer (D2, mTRF). Models/layers/feature dirs hardcoded (tiny `blocks.1`, base `blocks.5`, small `blocks.10`, medium `blocks.22`).
**Run:** `python scripts/plot_fit_quality_bars.py --neural outputs/neural_data/surprisal_30s.h5 --out outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png`
**Key params:** `--neural` (`surprisal_30s.h5`), `--out` (required), NC/HP/lag/alpha as §6.3, `--n_eval_time_samples` (400). Cluster driver `scripts/slurm_fit_quality_bars.sh`.
**Outputs:** one PNG. Calls `build_parcels`/`fit_mapping` from §5.1 (uses `eval_heldout`, never needed `load_split_parcels`/`load_split_targets`).

### 6.5 `scripts/plot_mtrf_scores.py`
**Purpose:** Diagnostic comparing no-highpass vs 1 Hz high-pass mTRF lag curves per ROI (slow-autocorrelation confound check). Reads §4.1 `mtrf_scores.h5`.
**Key params:** `--nohp_dir` (`…/whisper-base-mtrf-full/layer_2`), `--hp_dir` (`…/whisper-base-mtrf-hp1/layer_2`), `--layer` (`blocks-2`), `--rois` (`Fz,T7,T8,AF3,FT7,TP7,Pz`), `--out` (`outputs/figures/mtrf_highpass_diagnostic.png`). Reads `<layer>/group/<roi>` (see §13.3).

### 6.6 `scripts/plot_temporal_scores.py`
**Purpose:** Per-time-step encoding curves (8 electrodes, all 6 base layers overlaid; raw + Gaussian-smoothed) from §4.5 `temporal_scores.h5`.
**Key params:** `--scores_path` (`…/whisper-base-delta-t-full/temporal_scores.h5`), `--output_path` (`…/whisper_base_temporal_scores.png`), `--sigma_bins` (25 ≈ 500 ms). Assumes 1500 bins @20 ms, `<layer>/group/<electrode>` keys.

### 6.7 `scripts/plot_score_distributions.py`
**Purpose:** Tests whether per-time-bin r distributions exceed 0 (one-sample t with AR(1)/`n_eff` correction); violin plots (8 electrodes × 6 layers) + JSON. Reads §4.5 `temporal_scores.h5`.
**Key params:** `--scores_path` (same default as §6.6), `--output_dir` (`outputs/figures`).
**Outputs:** `score_distribution_summary.json`, `whisper_base_score_distributions.png`.

### 6.8 `scripts/plot_mtrf_scores.py` etc. — note
§6.5–§6.7 expect HDF5 keys `<layer>/group/<roi>`; the writers (§4.1, §4.5) use `<layer>/<subject>/<roi>`, i.e. `subject="group"` for these datasets — consistent here, but a footgun for non-group data (§13.3).

---

## 7. Diagnostics

Both are **no-argparse** scripts (hardcoded paths, run from repo root, stdout only).

### 7.1 `scripts/diagnose_roi_mapping.py`
**Purpose:** Diagnose standard 10-20 → BioSemi128 mapping (same logic as §2.1): nearest-channel distances, collision check, stored NC per ROI, optional raw FCz signal stats.
**Run:** `python scripts/diagnose_roi_mapping.py`
**Reads:** `outputs/neural_data/broderick2018_30s.h5`; optional sub-001 BrainVision; MNE montages. **Targets:** `Fz,FCz,Cz,Pz,F3,F4,C3,C4,T7,T8`.

### 7.2 `scripts/diagnose_extra_electrodes.py`
**Purpose:** Extract NC for standard electrodes not in the stored ROI list (e.g. Fp2, C2) by nearest-neighbour to BioSemi128 + reading `whole_brain` NC.
**Run:** `python scripts/diagnose_extra_electrodes.py`
**Reads:** `broderick2018_30s.h5` (`noise_ceilings/group/whole_brain [T,128]`); first sub-001 `.vhdr`; MNE montages.

---

## 8. Orchestration (shell / SLURM)

All cluster scripts `cd` to the handover repo root and `source env.sh`. **jed** = CPU (no `--partition` or `--partition=standard`); **kuma** = GPU (`--partition l40s`, `--gres=gpu:1`). `/work` is shared across both. Logs → `logs/`.

**Shared helper (sourced, not submitted):**
- `scripts/_whisper_features.sh` — resolves `MODEL_ID` (default `whisper-base`) to its delta-T feature dir + layer config (`SOPHIE_FEAT`, `LAYERS`).

**Feature extraction (jed):**

| Script | Invokes | Notes |
|---|---|---|
| `slurm_extract_delta_t.sh` | `mbs.extraction.extract_features_delta_t` | whisper-base on Broderick; internal `MODE` pilot/full; 12 h, 1 CPU |
| `slurm_extract_delta_t_generic.sh` | same | parametrised (`MODEL_ID` default `whisper-small`, `WINDOW_DUR`/`WINDOW_STRIDE`/`CHUNK_SIZE`/`BATCH_T`) |
| `slurm_mmn_extract.sh` | same | MMN WAVs (`MMN_METHOD` default still says `method_09` in the script — stale, always pass `MMN_METHOD` explicitly to one of the §16.1 10 methods); 16-task array; `--window_stride 30` |
| `slurm_mmn_extract_batch.sh` | same | one method per array task by default (model loads once). Condition list from `METHOD_LIST` + `TASK_OFFSET`, else the hardcoded 24 literature ids; `MMN_FEATURES_ROOT` / `MMN_NAME_BY_STIM_ID` / `MMN_OVERWRITE` (§17.2); **`CLIPS_PER_TASK`** splits a method's clips over several tasks so they run in parallel — wall clock is set by that serial loop, not by the array `%throttle`, which is usually slack (default `0` = the historical whole-method layout; needs `MMN_NAME_BY_STIM_ID=true`, and `CLIPS_PER_METHOD` declares the per-method clip count the index map assumes); 24 h, 8 CPU |
| `submit_whisper_small_sweep.sh` | format_eeg_hdf5 + `slurm_extract_delta_t_generic.sh` | window/stride ablation (w30s05/10/30); orchestrator |

**mTRF / Workstream A (jed):**

| Script | Invokes | Notes |
|---|---|---|
| `slurm_mtrf.sh` | `evaluate_features_mtrf` | whisper-base/Broderick; 18-task array (6 layers × 3 high-pass cutoffs 0.5/1.0/2.0) |
| `slurm_mtrf_parcels.sh` | `evaluate_features_mtrf_parcels` | D1/D2/D3 array (0–2); `MODEL_ID`/`PCA_VAR`/`OVERWRITE`; 48 h, 32 CPU |
| `slurm_cross_mtrf.sh` | `evaluate_cross_dataset_mtrf` | d1→d2 & d2→d1 (array 0–1); `PCA_VAR` optional; 48 h, 32 CPU |
| `slurm_eeg_mapping_sweep.sh` | `eeg_mapping_sweep.py` | 8-task array (4 models × 2 levels); 72 h, 32 CPU |
| `slurm_fit_quality_bars.sh` | `plot_fit_quality_bars.py` | cross-model D2 bars; 4 h, 16 CPU |
| `resubmit_eval_temporal_full.sh` | `evaluate_features_temporal` (via `sbatch --wrap`) | recovers killed D1 temporal job; deletes corrupt h5 first |

**Attention encoder / Workstream B (kuma GPU unless noted):**

| Script | Invokes | Notes |
|---|---|---|
| `kuma_probe_d2_cv.sh` | `evaluate_features_attn_probe_temporal` | CV folds, array 0–31 (4 models × 2 levels × 4 folds); `--save_model false` |
| `jed_collect_encoder_cv.sh` (**jed**) | `eeg_mapping_encoder_cv.py` + `plot_eeg_mapping.py` | aggregate folds → JSON + figures |
| `kuma_probe_d2_final.sh` | same probe module | final checkpoint at CV-chosen layer, array 0–7; `--save_model true`; reads `eeg_mapping_encoder/*.json` |
| `kuma_probe_mmn.sh` | same probe module | MMN-deliverable ckpt (`MODEL_ID` default whisper-small, `LAYER_ID` default 10) |
| `kuma_probe_d2d3.sh` → `run_probe_d2d3.sh` | same probe module | D2 + D3 group probes (4 parcels from Broderick) |
| `jed_probe_tests.sh` (**jed**) | `pytest tests/test_attn_probe_temporal.py` | CPU validation |

**In-silico MMN (jed, `partition=standard`):**

| Script | Invokes | Notes |
|---|---|---|
| `slurm_insilico_mmn.sh` | `insilico_mmn.py` | optional layer array `blocks.{0..5}`; forwards extra args via `"$@"` |
| `slurm_insilico_mmn_electrodes.sh` | `insilico_mmn_electrodes.py` | optional layer array; topographic. `MMN_FEATURES_ROOT`/`STIM_ROOT`/`METADATA_CSV`/`DATA_DIR`/`OUT_DIR` overrides — **set `DATA_DIR` for any non-literature screen or it overwrites the committed literature predictions** (§17.6); 12 h |
| `slurm_generate_stimuli.sh` | `00aa_generate_audio_stimuli.py` | `METADATA_CSV`/`OUTPUT_DIR`; forwards extra args via `"$@"` (`--trial_levels`/`--num_variations`/`--models`, §17.2); 1 h, 16 CPU |
| `slurm_stage_novel_stimuli.sh` (**login node**, not sbatch) | `cp` + `mkdir` | flat generator tree → per-condition dirs + `METHOD_LIST` (§17.3) |
| `submit_novel_extraction.sh` (**login node**, not sbatch) | `sbatch slurm_mmn_extract_batch.sh` | 6 models × the array splits `MaxArraySize` requires; `MODELS` overrides the model set (pass it explicitly or whisper-large is silently skipped); `CLIPS_PER_TASK` sizes the array for the expanded (method, clip-chunk) index space; `DRY_RUN=1` to preview (§17.3) |
| `submit_novel_insilico.sh` (**login node**, not sbatch) | `sbatch slurm_insilico_mmn_electrodes.sh` | 6 models, all paths redirected; refuses the literature predictions root (§17.6) |
| `slurm_insilico_mmn_attn.sh` | `insilico_mmn_attn.py` | checkpoint-driven; all args via `"$@"` |

**Legacy/example (local, not SLURM):** `train_example.sh`, `extract_example.sh`, `evaluate_example.sh`, `fit_curves_example.sh` (§10).

> **Referenced-but-absent scripts** (named in `aux/project_plan_20260611.md`, not in this checkout): `scripts/jed_collect_encoder_mapping.sh`, `scripts/kuma_probe_d2_levels.sh`, `scripts/run_probe_d2_levels.sh`. See §13.5.

---

## 10. Legacy / reference-only: vision→brain scaling

The original ICML-2026 codebase. Fully functional but **not part of the active auditory work**; documented here at a glance (entry points verified from `pyproject.toml`; usage from `README.md`). See the README for the canonical instructions.

| Command / module | Role | Example invocation |
|---|---|---|
| `mbs-download-artifacts` (`mbs.download_artifacts`) | Restore published result CSVs from Hugging Face | `uv run mbs-download-artifacts --artifacts-dir artifacts` |
| `mbs-train` (`mbs.training.train`) | Fine-tune a vision backbone with neural supervision | `bash scripts/train_example.sh` |
| `mbs-extract-features` (`mbs.extraction.extract_features`) | Extract vision-model features (h5 / THINGS / brain_score) | `bash scripts/extract_example.sh` |
| `mbs-evaluate-all-layers` (`…evaluate_features_all_layers`) | All-layer ridge sweep | — |
| `mbs-evaluate-committed-layers` (`…evaluate_features_committed_layers`) | Committed-layer ridge evaluation | `bash scripts/evaluate_example.sh` |
| `mbs-evaluate-attn-probe` (`…evaluate_features_attn_probe`) | Attention-based readout (non-temporal) | — |
| `mbs-fit-curves` (`mbs.analysis.curve_fitting.start_fitting`) | Fit scaling curves from result CSVs | `bash scripts/fit_curves_example.sh` |

Also reference-only: `src/mbs/training/**`, `src/mbs/analysis/curve_fitting/**`, `src/mbs/metrics/**` (RSA/CKA/RidgeGCV), `src/mbs/modeling/**`, most of `configs/{models,training,evaluation,analysis}/`, and `docs/literature/extract_pdf_text.py`.

---

## 11. High-Level Workflow Overview

The active auditory pipeline goes raw audio + raw EEG → model→EEG mapping → in-silico MMN:

1. **Format EEG** (§2.1): Broderick BIDS → `broderick2018_30s.h5` (D1). (D2 `surprisal_30s.h5` / D3 supplied separately — §13.4.)
2. **Extract features** (§3.1): speech stimuli → delta-T HDF5 chunks → merge into `<model>-delta-t[-surprisal]/merged`.
3. **Fit & select the mapping** on D2:
   - Method A: `eeg_mapping_sweep.py` (§4.6) → committed mTRF layer.
   - Method B: `kuma_probe_d2_cv.sh` (§4.4) → `eeg_mapping_encoder_cv.py` (§4.7) → committed encoder layer → `kuma_probe_d2_final.sh` checkpoint.
4. **Extract MMN stimulus features** (§8 `slurm_mmn_extract.sh`).
5. **Run in-silico MMN** (§5): Method A `insilico_mmn.py` / `insilico_mmn_electrodes.py`, or Method B `insilico_mmn_attn.py`.
6. **Read the MMN** from `predictions__<layer>*.h5`: plotted `deviant_mean − standard` is mean-baseline-corrected (window = `3×SOA` ms); the verdict is the separately z-scored `baseline_normalized_peak` in the 100–240 ms band (§16.3), never the plotted trace itself.
7. **Combine results** (§5.4): `build_mmn_results_table.py` → one CSV across all models/methods/mapping types.

End-to-end (Method A, parcels, whisper-small, on the cluster):
```bash
source env.sh
# (D1 EEG; D2 assumed present)
python -m mbs.data_prep.format_eeg_hdf5 --bids_root $BIDS --output_path outputs/neural_data/broderick2018_30s.h5
sbatch scripts/slurm_extract_delta_t_generic.sh           # MODEL_ID=whisper-small → merge to /merged
sbatch scripts/slurm_eeg_mapping_sweep.sh                 # → outputs/results/eeg_mapping/*.json (chosen layer)
sbatch scripts/slurm_mmn_extract.sh                       # MMN tone features
python scripts/insilico_mmn.py --train_features <D2 feats>/merged \
  --train_neural outputs/neural_data/surprisal_30s.h5 --layer blocks.3 --lag_max_ms 800 --methods all
```

---

## 12. Dependency and Execution Graph

```
Broderick BIDS ──format_eeg_hdf5.py(§2.1)──► broderick2018_30s.h5 (D1)
(D2 surprisal_30s.h5, D3 d3_combined_30s.h5  ── supplied/built elsewhere, §13.4)
                                            │
speech WAVs ──extract_features_delta_t(§3.1)─► <model>-delta-t[-surprisal]/merged ─┐
                                                                                   │
        ┌──────────────────────────────────────────────────────────────────────-─┤
        │                                                                          │
   MTRF (A)                                                                   ATTN ENCODER (B)
   evaluate_features_mtrf(§4.1) ── slurm_mtrf.sh                             evaluate_features_attn_probe_temporal(§4.4)
   evaluate_features_mtrf_parcels(§4.2) ── slurm_mtrf_parcels.sh                 │  ▲ kuma_probe_d2_cv.sh (folds)
   evaluate_cross_dataset_mtrf(§4.3) ── slurm_cross_mtrf.sh                      │  │
   eeg_mapping_sweep.py(§4.6) ── slurm_eeg_mapping_sweep.sh                      │  └─► fold h5
        │  └─► eeg_mapping/*.json ──┐                                            │       │
        │                          │                                  eeg_mapping_encoder_cv.py(§4.7) ◄┘
        │                          │  jed_collect_encoder_cv.sh ──► eeg_mapping_encoder/*.json ──► chosen layer
        │                          ▼                                            │
        │                  plot_eeg_mapping.py(§6.2) ◄───────────────────────────┘
        │                                                              kuma_probe_d2_final.sh ──► model__<layer>.pt
        ▼                                                                          │
MMN WAVs ──slurm_mmn_extract.sh──► mmn-<method>-delta-t                            │
        │                                                                          │
   insilico_mmn.py(§5.1) / insilico_mmn_electrodes.py(§5.2)          insilico_mmn_attn.py(§5.3) ◄─ checkpoint
        └──────────────► predictions__<layer>*.h5 ◄───────────────────────────────┘
                                   │                  (model-namespaced dirs)
                         finalize_method(): mean-corrected dev/std/diff (plotted) +
                         z-scored baseline_normalized_peak / n7v1_peak (verdict only)
                                   │
                       build_mmn_results_table.py(§5.4) ──► mmn_results_table.csv

(diagnostics: plot_temporal_scores / plot_score_distributions ◄ evaluate_features_temporal(§4.5);
 plot_mtrf_scores ◄ evaluate_features_mtrf; score_mtrf_fitquality / plot_fit_quality / plot_fit_quality_bars ◄ §5.1 mTRF)
```

**Script dependency table** (Reads the output of / invokes):

| Script | Reads / invokes |
|---|---|
| `format_eeg_hdf5.py` (§2.1) | Broderick BIDS |
| `extract_features_delta_t.py` (§3.1) | speech/MMN WAVs |
| `evaluate_features_mtrf.py` (§4.1) | §3.1 features, §2.1 EEG |
| `evaluate_features_mtrf_parcels.py` (§4.2) | §3.1 features, D2/D3 EEG; imports §4.1 |
| `evaluate_cross_dataset_mtrf.py` (§4.3) | §3.1 features, two EEG sets; imports §4.2 |
| `evaluate_features_attn_probe_temporal.py` (§4.4) | §3.1 features, EEG; writes checkpoints |
| `evaluate_features_temporal.py` (§4.5) | §3.1 features, §2.1 EEG |
| `eeg_mapping_sweep.py` (§4.6) | §3.1 features, D2 EEG; imports §4.1, §4.8 |
| `eeg_mapping_encoder_cv.py` (§4.7) | §4.4 fold h5 |
| `insilico_mmn.py` (§5.1) | §3.1 (speech + MMN) features, D2 EEG; imports §4.1, §4.8 |
| `insilico_mmn_electrodes.py` (§5.2) | imports §5.1, §4.8 |
| `insilico_mmn_attn.py` (§5.3) | §4.4 checkpoint, MMN features; imports §5.1, checkpoint.py |
| `build_mmn_results_table.py` (§5.4) | §5.1/§5.3 `predictions__<layer>*.h5` (model-namespaced dirs) |
| `score_mtrf_fitquality.py` (§6.1) | imports §5.1, §4.8 |
| `plot_eeg_mapping.py` (§6.2) | §4.6/§4.7 JSON |
| `plot_fit_quality.py` (§6.3) | imports §5.1, §4.8 |
| `plot_fit_quality_bars.py` (§6.4) | imports §5.1, D2 EEG |
| `plot_mtrf_scores.py` (§6.5) | §4.1 `mtrf_scores.h5` |
| `plot_temporal_scores.py` / `plot_score_distributions.py` (§6.6/§6.7) | §4.5 `temporal_scores.h5` |
| `diagnose_*` (§7) | §2.1 EEG, BIDS |

---

## 13. Open Questions and Unclear Areas

### 13.1 Layer-selection CV leakage — **Resolved/Fixed**
The earlier layer-selection CV carved folds from overlapping 30 s/10 s-stride windows (20 s overlap → leakage). **Fix:** switched to non-overlapping **group-by-part** CV (`grouped_kfold`, folds grouped by audiobook `.wav`), re-ran both methods, and **committed the chosen layers** (§1.5). Verified in `eeg_targets.py`, `eeg_mapping_sweep.py`, `attn_probe/dataset_temporal.py`, and commit `3394c24`.

### 13.2 Broken `load_split_parcels` import in three scripts — **Resolved (2026-06-18, see §16.3)**
`scripts/insilico_mmn_attn.py` (§5.3), `scripts/plot_fit_quality.py` (§6.3), and `scripts/score_mtrf_fitquality.py` (§6.1) all did `from insilico_mmn import (… load_split_parcels …)`, but `insilico_mmn.py` never defined or re-exported `load_split_parcels` — the function was renamed to `load_split_targets` and moved to `scripts/eeg_targets.py` (§4.8) and these three call sites were never updated. **Fixed:** all three now `from eeg_targets import load_split_targets` and call that name directly. Method A's main driver `insilico_mmn.py` was already correct.

### 13.3 HDF5 key convention `<layer>/group/<roi>` vs `<layer>/<subject>/<roi>` — **Needs confirmation**
The plot scripts §6.5–§6.7 hardcode `subject="group"` in the key path, while the writers §4.1/§4.5 use `<layer>/<subject>/<roi>`. For the current group-averaged datasets `subject` is always `"group"`, so they match — but any per-subject run would silently mismatch. Confirm before reusing the plotters on individual-subject score files.

### 13.4 D2 / D3 formatters not in this checkout — **Open**
`surprisal_30s.h5` (D2) and `d3_combined_30s.h5` (D3) are consumed pervasively, but only the Broderick (D1) formatter (§2.1) is present. The D2/D3 builders (the handover/plan mention a `cortical_suprisal_dataset` and a `build_d3_features` effort) live in Sophie's larger cluster copy, not here. To rebuild D2/D3 you need that code or the prebuilt HDF5s. The files are gitignored / not present locally.

### 13.5 Referenced-but-absent shell scripts — **Open (doc drift)**
`aux/project_plan_20260611.md` references `scripts/jed_collect_encoder_mapping.sh`, `scripts/kuma_probe_d2_levels.sh`, `scripts/run_probe_d2_levels.sh` — none exist in this checkout. Treat them as planned/not-yet-committed.

### 13.7 `insilico_mmn.py` default `--lag_max_ms 500` vs sweep 800 — **Resolved (2026-06-18)**
The driver default was 500 ms even though the handover instructed `--lag_max_ms 800` to match the sweep that chose the committed layers, and `score_mtrf_fitquality.py`/`plot_fit_quality.py` already defaulted to 800. **Fixed:** `insilico_mmn.py`'s `--lag_max_ms` default is now `800.0`, matching the other two drivers; the electrode driver's default was also brought in line.

### 13.8 MMN frequency pairs are placeholders — **Resolved (2026-06-18, see §16.1)**
The 8 shipped `method_*` pairs described a now-superseded identity-MMN design (Definition 2, discontinued) and are gone. `METHODS` in `insilico_mmn.py` now holds the 10-pair literature classic-oddball **Definition 1** set (`data/metadata/literature_frequency_intensity_duration_metadata.csv`), already present as `data/audio_stimuli/`, but it is **provisional, not yet fully vetted** (§16.2). See §16.1 for why this is a different design from the old `method_37/44/55`, not a stale copy of it.

### 13.10 Shell-script `MMN_METHOD`/`--method` defaults still say `method_09` — **Open (doc drift, low risk)**
`scripts/slurm_mmn_extract.sh` (`METHOD="${MMN_METHOD:-method_09}"`) and `scripts/kuma_probe_mmn.sh` (echoed example) still default/refer to the old-registry `method_09`, which no longer exists in `METHODS` or `data/audio_stimuli/` (§16.1). The three Python drivers' own `--method`/`--methods` defaults were updated to the new registry (`method_37`, §5.1–§5.3), but these two shell scripts were left as-is since they're outside this round's Python-only scope. Low risk because the runbook (§15 Stage D) always passes `MMN_METHOD` explicitly — just don't rely on the shell scripts' bare defaults.

---

## 14. Recommended Reading Order

1. `aux/XX_handover_for_Sophie.md` — the focused "run it" guide for the MMN deliverable; defines §1.5 layers and Methods A/B.
2. `aux/01_setup.md` — environment, cluster, dataset download.
3. `aux/mmn_screening_plan.md` — MMN stimulus selection + evaluation criteria.
4. `aux/project_plan_20260611.md` — the full technical log (scaling, env/GPU, superseded methods, MMN design §17, repo layout); large — skim by section.
5. **The two results memos**, for what the screens actually found:
   - `aux/analysis_with_counter/results_analysis_with_counter.md` — the **literature screen**. Sections 7/8/8b/8c/10/11 are the current 24-method × 7-model mTRF vintage (`outputs/results_soafix_full/`, denominators /48 and /336); Sections 0b/1–6/9 are the older 10-method × 4-model screen (`outputs/results_with_counter/`). **The two vintages are not directly comparable** — the memo's data-vintage flag at the top says which is which, and it is the first thing to read.
   - `aux/analysis_novel_search/novel_stimulus_search_results.md` — the §17 novel-grid search.
6. `scripts/eeg_targets.py` (§4.8) — parcels/electrodes/CV definitions used everywhere.
7. `src/mbs/evaluation/evaluate_features_mtrf.py` (§4.1) — the shared mTRF engine.
8. `scripts/insilico_mmn.py` (§5.1) — the end-to-end Method A MMN driver.
9. `src/mbs/evaluation/evaluate_features_attn_probe_temporal.py` (§4.4) — Method B training/scoring.
10. `README.md` — only for the legacy vision pipeline (§10).

---

## 15. How to Run the Full Workflow

**Prerequisites:** (repo root is Sophie's clone, §1.1 — not the `mehrer` tree; only input datasets are read from `mehrer`)
```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling
source env.sh
mbs-extract-features-delta-t --help     # verify env
ls outputs/neural_data/surprisal_30s.h5 # D2 must exist (see §13.4) -- in THIS tree, not mehrer's
```

**Stage A — EEG formatting (D1; D2/D3 supplied).** `--bids_root` reads the raw dataset from `mehrer`'s tree (read-only input); `--output_path` writes into Sophie's own `outputs/` (never back into `mehrer`):
```bash
python -m mbs.data_prep.format_eeg_hdf5 \
  --bids_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea \
  --output_path outputs/neural_data/broderick2018_30s.h5 \
  --window_duration 30.0 --window_stride 10.0 --target_sr 50 --n_test_runs 4 --seed 42
```

**Stage B — feature extraction (per model).** Pilot first (cheap smoke test) by editing `MODE="pilot"` in `slurm_extract_delta_t.sh` (3 stimuli) or using `--n_stimuli 3 --t_stride 50` directly, then full:
```bash
MODEL_ID=whisper-small sbatch scripts/slurm_extract_delta_t_generic.sh   # array over chunks
# merge chunk_*/ into a single merged/ dir before evaluation
```

**Stage C — fit & select the mapping (D2):**
```bash
# Method A:
sbatch scripts/slurm_eeg_mapping_sweep.sh        # → outputs/results/eeg_mapping/<model>__<level>__D2.json
# Method B:
sbatch scripts/kuma_probe_d2_cv.sh               # CV folds (GPU)
sbatch scripts/jed_collect_encoder_cv.sh         # aggregate → eeg_mapping_encoder/*.json + figures
sbatch scripts/kuma_probe_d2_final.sh            # final model__<layer>.pt at chosen layer
```

**Stage D — MMN stimulus features (10 methods × 4 models = 40 combinations, §16.1):**
```bash
MMN_METHOD=method_37 MODEL_ID=whisper-small sbatch scripts/slurm_mmn_extract.sh
# repeat for method_75,74,72,60,53,55,43,44,27 and for whisper-tiny/base/medium
```

**Stage E — in-silico MMN (use committed §1.5 layers, `--lag_max_ms 800`).** Pass model-namespaced `--out_dir`/`--data_dir` — tiny and base share `chosen_layer = blocks.0` for mTRF parcels, so the default shared output dir would silently overwrite one with the other:
```bash
# Method A, parcels (whisper-small → blocks.3):
python scripts/insilico_mmn.py \
  --train_features <D2 feats>/merged --train_neural outputs/neural_data/surprisal_30s.h5 \
  --mmn_features_root outputs/features --layer blocks.3 --lag_max_ms 800 --methods all \
  --out_dir outputs/figures/insilico_mmn/whisper-small --data_dir outputs/insilico_mmn_predictions/whisper-small
# Method A, electrodes (whisper-small → blocks.1; topography + verdict):
python scripts/insilico_mmn_electrodes.py \
  --train_features <D2 feats>/merged --train_neural outputs/neural_data/surprisal_30s.h5 \
  --layer blocks.1 --lag_max_ms 800 --methods all
# Method B (after Stage C checkpoint exists):
python scripts/insilico_mmn_attn.py \
  --checkpoint outputs/results/whisper-small-probe-group-d2-parcels/model__blocks.10.pt \
  --mmn_features_root outputs/features --method method_37 \
  --features_dir <D2 feats>/merged --neural outputs/neural_data/surprisal_30s.h5 \
  --out_dir outputs/figures/insilico_mmn/whisper-small --data_dir outputs/insilico_mmn_predictions/whisper-small
```

**Stage F — read the MMN** (`outputs/insilico_mmn_predictions/<model>/predictions__<layer>*.h5`, §5.1/§5.3/§16.3): the plotted `deviant − standard` is mean-baseline-corrected over a `3×SOA`-ms pre-onset window (same units as the raw prediction); the verdict is the **separately z-scored** `baseline_normalized_peak` (most-negative point in the 100–240 ms band) plus the `n7v1_peak` single-trial diagnostic — both stored per-target in the h5, never the plotted trace itself. Same criterion at parcel and electrode level. Snippet in `aux/XX_handover_for_Sophie.md` §2 (pre-dates the z-score split — see §16.3 here for the current formula).

**Stage G — combined results table (§5.4):**
```bash
python scripts/build_mmn_results_table.py \
  --predictions_root outputs/insilico_mmn_predictions --out outputs/results/mmn_results_table.csv
```

**Resource reference (from SLURM headers):**

| Stage | Cluster | Time | CPUs/GPU |
|---|---|---|---|
| EEG format | jed | ~20 min | — |
| Delta-T extract | jed | 12 h/array task | 1 CPU |
| mTRF parcels / cross / sweep | jed | 48–72 h | 32 CPU |
| Encoder CV / final | kuma | 72 h | 1 GPU, 8 CPU |
| MMN extract | jed | 1.5 h | 2 CPU |
| In-silico MMN | jed (standard) | 30 min | 8–12 CPU |

**Fallback for partial failures:** the mTRF/encoder/temporal writers open the scores HDF5 in append mode and skip completed `(layer, subject, roi)` keys, so a killed job resumes by re-submitting (use `--overwrite true` only to force a clean rebuild). `resubmit_eval_temporal_full.sh` deletes a corrupt `temporal_scores.h5` before resubmitting — the pattern to copy for a corrupted output.

---

## 16. MMN Stimulus-Design Correction & Verdict-Criterion Rewrite (2026-06-18)

This round closed out §13.2/§13.7/§13.8 above and replaced the placeholder MMN registry with the literature stimulus set already shipped in `data/audio_stimuli/`. Four things worth recording, none derivable from a quick code read:

### 16.1 The new `method_37/44/55` are NOT the old `method_37/44/55`
Two unrelated stimulus-generation rounds happen to reuse the same `method_id` numbers. The **old** round (still referenced in `aux/XX_handover_for_Sophie.md` and, until this commit, hardcoded in `insilico_mmn.py`'s `METHODS`) was a physically-controlled identity-MMN design — **discontinued, not currently used**: one standard/deviant pair per method, with the final/eliciting tone *physically identical* in both — deviance lived only in the preceding context (e.g. old `method_37` = "1050→1000 Hz downward," final tone fixed at 1000 Hz). The **new** round (this round's deliverable, **currently in use**) is a literature-replication classic oddball design: per method, 1 standard file + 15 deviant files (`N∈{3,5,7} × var∈{1..5}`), and the deviant condition's *last* tone differs in frequency from the standard's (e.g. new `method_37` = Javitt_2000a, standard 1000 Hz repeats, deviant train ends on 1050 Hz). Confirmed against `data/metadata/literature_frequency_intensity_duration_metadata.csv`: all 10 requested ids map exactly to Karger_2014(75), Domjan_2012(74), Bodatsch_2011(72), Umbricht_2003a(60), Salisbury_2002a(53), Shinozaki_2002a(55), Javitt_2000a(37), Michie_2000b(43), Michie_2000c(44), Schall_1999a(27); `data/audio_stimuli/` already has all 10 × 16 files. **Implication:** if you ever see `outputs/mmn_stimuli/method_{37,44,55}` not matching `data/audio_stimuli/method_{37,44,55}`, that is not a stale-cache bug — it's two different experiments sharing a number. Don't cross-reference them; always populate `outputs/mmn_stimuli/` fresh from `data/audio_stimuli/` (or the verified cluster source) for this design.

**Definition label correction (2026-06-21):** a direct read-through of the methods/epoching sections of all 10 sourced papers (Schall_1999a, Javitt_2000a, Michie_2000b/c, Salisbury_2002a, Shinozaki_2002a, Umbricht_2003a, Bodatsch_2011, Domjan_2012, Karger_2014) confirmed every one uses the literature classic-oddball design — none use the physically-controlled identity design. The classic-oddball design (currently in use, frequency pairs sourced directly from these papers) is now **Definition 1**. The discontinued physically-controlled identity-MMN design — the methodologically newer paradigm, designed to rule out the physical-stimulus-difference confound in classic oddball MMN — is now **Definition 2**, and is not currently used; no literature source implementing it has yet been identified for this project.

### 16.2 Definition 1 stimulus set is provisional, not yet fully vetted
**Stimulus sources are provisional, not yet fully vetted.** The 10 methods (§16.1) were selected to get the pipeline running end-to-end, with the expectation that 1–2 may be swapped in/out later once the literature list is reviewed more carefully. Current inclusion criteria: most recent publication year per paper, only one method permitted per paper, frequency variants only (no duration/intensity deviants in this round). Treat the per-method rows (`aux/results_analysis.md` Tables 1a/1c/2a/2c, Table 4) as more provisional than the per-model rollups (1b/1d/1e/2b/2d/2e) — a future swap could shift individual method rows without necessarily changing the overall per-model picture much, or could change it a lot; this hasn't been stress-tested.

### 16.3 Code changes made this round
- `scripts/insilico_mmn.py`: `METHODS` replaced with the 10-pair set (§16.1); added `load_soa_table()`/`soa_for_method()` (reads `standard_soa` from the metadata CSV); replaced the mean-only, fixed-150ms `bc()` baseline correction with `finalize_method()` — a shared A/B helper that keeps the **plotted** `dev_b`/`std_b`/`diff_b` as mean-only baseline correction (now over a `3×SOA`-ms window, same units as before) but adds a **z-scored, verdict-only** `baseline_normalized_peak` (full z-score of dev/std within the same baseline window, differenced, most-negative point in `[100,240]` ms) plus an `n7v1_peak` diagnostic from the single `N7/var1` deviant trace alone. The z-scored arrays are never plotted — only the resulting scalar is annotated on the figures. `plot_method()` now restricts rows to frontal/central/temporal and annotates the third column with `baseline_normalized_peak`. `--lag_max_ms` default changed `500→800` (§13.7).
- `scripts/insilico_mmn_attn.py`: fixed the `load_split_parcels`→`load_split_targets` import (§13.2); `analyze_method_attn()` now delegates to the shared `finalize_method()` instead of duplicating the baseline logic.
- `scripts/insilico_mmn_electrodes.py`: `mmn_metric()` now reads the precomputed `peak` array (ROI-averaged) instead of recomputing a raw mean amplitude; `--mmn_lo_ms`/`--mmn_hi_ms` are now plot-shading-only (the verdict window is fixed inside `finalize_method`).
- `scripts/score_mtrf_fitquality.py`, `scripts/plot_fit_quality.py`: same import fix as above.
- New `scripts/build_mmn_results_table.py`: assembles the combined CSV (10 pairs × 4 models × {mTRF, encoder} × `baseline_normalized_peak` per parcel, plus the `N7/var1` diagnostic) from the per-model/method prediction HDF5s.
- Two small leftover-default fixes for consistency with the new registry: `insilico_mmn_attn.py`'s `--method` default `method_09→method_37`; its module-docstring example and `insilico_mmn_electrodes.py`'s docstring example updated to real method ids. The shell-script defaults (`slurm_mmn_extract.sh`'s `MMN_METHOD`, `kuma_probe_mmn.sh`'s echoed example) still say `method_09` — left as-is, flagged in §13.10, since the runbook always passes `MMN_METHOD` explicitly.

### 16.4 Method B status: CV-done, final checkpoints not yet trained
The encoder layer-selection CV sweep is complete and committed (§1.5/§21 of `project_plan_20260611.md`) — `outputs/results/eeg_mapping_encoder/*.json` are all populated with `chosen_layer`. The separate **final reusable checkpoint** job (`model__<layer>.pt`, needed before `insilico_mmn_attn.py` can run at all) has not been run: `find outputs/results -name "model__*.pt"` returns nothing as of this writing. That's a cluster-side prerequisite (`sbatch --array=0-7 scripts/kuma_probe_d2_final.sh`) — see `aux/XX_handover_for_Sophie.md` §4 / the plan file's Part B runbook.

### 16.5 Testing and how this round was shipped
New `tests/test_insilico_mmn.py` (15 tests): import-smoke regression for the §13.2 fix (all 5 touched scripts + `build_mmn_results_table.py` must import cleanly and must not define/leak `load_split_parcels`), `METHODS`/`load_soa_table`/`soa_for_method` correctness against the real metadata CSV, `finalize_method()`'s mean-only-vs-z-scored split (synthetic traces with a known baseline SD and a planted bump, verified the plotted arrays stay far from unit variance while the verdict peak is correctly negative/z-scored, including the N7/var1-more-extreme-than-the-3-deviant-average property — checked robust across 200 random seeds, not just the one in the committed test), and `build_mmn_results_table.py`'s HDF5→CSV reshaping. Full existing suite re-run alongside it: 121 passed, 4 skipped (optional deps: `lightning`, `sklearnex`), 1 pre-existing unrelated failure (`test_static.py`'s private-cluster-path scan already had ~27 offenders across files never touched this round — confirmed no *new* offender was introduced).

One environment gotcha worth recording: the `mbs-env` conda environment's editable `mbs` install pointed at a sibling checkout (`multimodal-brain-scaling-1`), not this repo. Tests were run with `PYTHONPATH="$(pwd)/src:$(pwd)/scripts"` prepended to force resolution against this checkout; if `mbs-env` gets re-pointed at this repo (or the `sigfstea` clone, §1.1) via `pip install -e .` this won't be necessary.

Shipped as 5 commits on `main`, pushed to `origin` (`https://github.com/hanme/multimodal-brain-scaling.git`): the registry/verdict rewrite, the three-file import fix, the electrode verdict switch, the table builder + tests, and this write-up.

---

## 17. Stimuli Search Phase — novel tone-pair grid (2026-07-25)

**Scope.** A two-phase search for frequency-oddball tone pairs that do *not* appear in the literature, looking for stimuli that drive stronger and more model-consistent MMN responses than the 24-method literature set. **Not covered:** any re-fitting. The model→EEG mTRF mappings and the committed §1.5 per-model **electrode** layers are frozen and reused exactly; this phase only feeds new stimuli through them.

**Spec:** `aux/prompt_novel_stimuli_search_phases12.md`. **Memo:** `aux/analysis_novel_search/novel_stimulus_search_results.md` — all five sections filled from the completed Phase 1 (2026-07-27). The memo is a **results document and carries no costing**; run scale and disk requirements are in §17.4 below.

**Phase-1 outcome, in one line.** The grid's top tier (17 direction-instances at 6/6) is **not distinguishable from chance** — 12.0 expected from the six models' own marginal S7 rates, p = 0.103 — and **no pair of 903 reaches 6/6 in both directions**, so the consensus set is empty. The mean-µV heatmap stripes by row and column rather than banding parallel to the diagonal, and the two directions of a pair produce near mirror-image difference waves: the models are responding to particular frequencies, not to the change between two tones. This reproduces the literature screen's near-chance result (§11 of `results_analysis_with_counter.md`) on a 37× larger space.

**If a Phase-2 subset is selected**, the natural cut points are the tiers, since the ranking has no elbow: **17 pairs** with a 6/6 direction, **127** with a direction at ≥5 (the two do not overlap, so 17 + 110 = 127 exactly), then 387 at ≥4. See memo §5 for the structure and §17.4 here for what each costs.

### 17.1 What the search is and why the pipeline needed changing

43 frequencies give 903 unordered pairs. Each pair is synthesized in both directions by the existing counterbalancing, so 903 rows cover all **1,806 ordered** frequency combinations — emitting reversed rows too would double the cost for nothing. The diagonal is excluded: a same-frequency "deviant" synthesizes to a waveform byte-identical to the standard, so the difference is exactly zero by construction.

The grid is a **uniform ladder**: 43 rungs from 200 Hz in exact **1.500-semitone** steps, reaching 7611 Hz at **8.8% deviance per step**.

```
 200   218   238   259   283   308   336   367
 400   436   476   519   566   617   673   734
 800   872   951  1037  1131  1234  1345  1467
1600  1745  1903  2075  2263  2468  2691  2934
3200  3490  3805  4150  4525  4935  5382  5869
6400  6979  7611
```

**Why semitones, not hertz.** Pitch is ratio-based, so an equal-Hz grid over this range would put a 13.5-semitone gap at the bottom and 0.55-semitone steps at the top — a 24× difference in perceptual step size, most of the points wasted in the treble. Equal ratio steps also mean "k steps apart" denotes the same deviance everywhere, and that the two directions of a pair carry equal deviance *magnitude* (+1.500 st one way, −1.500 the other), which is what makes the frequency-preference test interpretable.

**Why 1.5 specifically.** 12 / 1.5 = 8 exactly, so eight steps is an octave and **200/400/800/1600/3200/6400** are all grid points — true 2:1 pairs exist. Any step that does not divide 12 evenly loses that: at the 1.494 st needed to land 43 points exactly on 7500 Hz, 12/1.494 = 8.03 and *no* pair anywhere would be an exact octave.

**Why the top is 7611, not 7500.** 200 → 7500 Hz is 62.746 semitones and 62.746 / 1.5 = 41.83 steps — not an integer, so a strict 1.5-semitone ladder cannot land on 7500. The options were to stop short (6979 Hz), append 7500 as an irregular rung, or overshoot by one. This overshoots: 7611 Hz is still comfortably under the **8 kHz Nyquist** of the 16 kHz sample rate every model takes (above which a tone aliases to a frequency the metadata does not claim), and every step stays identical with no irregular rung to caveat in the analysis.

**The cost is resolution.** The finest deviance this grid can express is **8.8%**, coarser than 8 of the 24 literature frequency methods. It is a coarse-to-fine search of a wide space, not a threshold study; a follow-up would need a finer grid over whatever region this one flags.

Rungs are anchored to 200 Hz (`200 × 2**(1.5i/12)`) rather than interpolated between endpoints; interpolating would accumulate rounding error.

- **Phase 1** — all 903 pairs, 4 clips each (standard + N7/var1 deviant, × 2 directions). Cheap screen.
- **Phase 2** — the pairs **5 or 6 of the 6 models agreed on** (`n_agree >= 5` in at least one direction) — **127** of the 903 — topped up to the full 32 clips (all 15 deviants), re-scored and re-ranked.

Fixed decisions (do not revisit): mTRF only; **S7 = S2 ∧ (trough_uv ≤ −0.75 µV)**; **FCz** electrode only; 6 models (whisper-large excluded — it alone was 51% of the literature screen's cost).

Five pipeline properties blocked this, each of which would have failed *silently* rather than loudly:

1. **`slurm_generate_stimuli.sh` did not forward `"$@"`.** `--trial_levels 7 --num_variations 1` would have been dropped and the full 16-clip grid synthesized — an 8× Phase-1 overshoot.
2. **`extract_features_delta_t.py` named HDF5s `feats_delta_t-start_<idx>-batch_<n>-seed_42.h5`** — start/batch counters, no stimulus identity. Phase 1's 2 clips land at `start_00000-batch_{0,1}`; in Phase 2 the 14 new clips occupy sorted indices 0–9 and 11–14 of the now-16-wav directory, so any contiguous `--stim_start_idx`/`--n_stimuli` subset **overwrites Phase 1's standard**. `analyze_method` returns `None` without a standard, so the whole of Phase 2 would have scored nothing.
3. **`--overwrite` was declared and never referenced** (one hit in the whole file); `_write_batch` opens mode `"w"`. There was no resume: any timed-out task re-extracted from scratch.
4. **`slurm_insilico_mmn_electrodes.sh` read none of the paths a separate screen needs** — only `MODEL_ID`. Pointed at the novel grid it would have read the *literature* features and **overwritten `outputs/insilico_mmn_predictions/{model}/electrode_predictions__*.h5`**, destroying the comparison baseline.
5. **`insilico_mmn.py:82` built `METHODS` at import time** from the hardcoded literature CSV, so `--metadata_csv` could not drive the condition registry.

### 17.2 Code changes

All are back-compatible: defaults reproduce the literature runs exactly.

| File | Change |
|---|---|
| `scripts/insilico_mmn.py` | Module-level `METHODS = build_methods_from_csv()` **removed**; the registry is built inside `main()` from `args.metadata_csv`, and `load_soa_table`/`load_duration_map` are routed to the same file. `DEFAULT_SOA_CSV` is now an alias of `DURATION_CSV` rather than a duplicated literal. |
| `scripts/insilico_mmn_electrodes.py` | Same registry change. New **`--save_plots`** (default `true`) gates both per-method figures *and* the `compute_criteria_table` call behind them. |
| `scripts/insilico_mmn_attn.py` | Same registry change (it imported `METHODS` and would not have imported otherwise). Not otherwise touched. |
| `scripts/00aa_generate_audio_stimuli.py` | New `--trial_levels` / `--num_variations` / `--models`, carried in a picklable `StimulusGrid` NamedTuple rather than module constants — macOS *spawns* pool workers, so mutating the constants in `main()` would not reach them. Filenames and the RNG seed are untouched. |
| `src/mbs/extraction/extract_features_delta_t.py` | New `--name_by_stim_id` (default `false`) names each HDF5 `feats_delta_t-<stim_id>-seed_42.h5`, asserting `--save_every 1`. `--overwrite` now actually works: with `false`, a stimulus whose output exists is skipped. A guard refuses to mix the two naming schemes in one directory (see §17.6). Post-loop flush added so a skipped *final* stimulus cannot strand an unwritten batch. |
| `scripts/slurm_generate_stimuli.sh` | Appends `"$@"`. |
| `scripts/slurm_mmn_extract_batch.sh` | Condition list from `METHOD_LIST` (one dir name per line) indexed by `SLURM_ARRAY_TASK_ID + TASK_OFFSET`, falling back to the hardcoded 24 literature ids when unset; explicit bounds guard (a *negative* bash index silently wraps to the end of the array). New `MMN_FEATURES_ROOT`, `MMN_NAME_BY_STIM_ID`, `MMN_OVERWRITE`. |
| `scripts/slurm_insilico_mmn_electrodes.sh` | `--time` `00:30:00` → `12:00:00`. New `MMN_FEATURES_ROOT` / `STIM_ROOT` / `METADATA_CSV` / `DATA_DIR` / `OUT_DIR` overrides; `--metadata_csv` now actually passed through. |
| `tests/test_insilico_mmn.py` | The stale `len(METHODS) == 10` assertion (failing since the registry became CSV-driven — the real count is 48) replaced with a 24-ids × 2-directions check, plus a regression that the registry is *not* a module attribute. |

### 17.3 New scripts

| Script | Purpose |
|---|---|
| `scripts/build_novel_grid_csv.py` | The grid → `data/metadata/novel_grid_frequency_metadata.csv` (903 rows, ids 1001–1903, header byte-identical to the literature sheet) + `outputs/results_novel_search/grid_index.csv` (`method_id, f_low, f_high, ratio, semitones, pct_deviance`). Asserts no duplicate pairs, no diagonal, no off-grid frequency, no id collision with the literature 1–76. |
| `scripts/novel_search_common.py` | The ranking criteria and cost model, **imported by both phases** so they provably rank identically — if they diverged, the Phase-1→Phase-2 rank stability would be measuring the code rather than the stimuli. |
| `scripts/rank_novel_phase1.py` | Ranks the 1,806 direction-instances, selects the pairs at `n_agree >= --min_agree` (default 5), emits `phase1_ranked_directions.csv`, `phase2_selected_pairs.csv`, `outputs/novel_methods_phase2.txt`. |
| `scripts/build_novel_phase2_csv.py` | Verbatim row-subset of the grid CSV for the selected pairs → `data/metadata/novel_grid_phase2_subset.csv`. |
| `scripts/rank_novel_phase2.py` | Final ranking, `consensus_set.csv`, and the Phase-1↔Phase-2 Spearman ρ. |
| `aux/analysis_novel_search/plots/novel_search_plots.py` | The three figures (`--results_dir`/`--out_dir` overridable). Runs as soon as Phase 1 lands; skips figure 3 until Phase 2 exists. |
| `aux/analysis_novel_search/plots/phase1_results.py` | Every table and figure the Phase-1 memo cites: agreement tiers at X = 0.75 and 0.50, the strong-pair waveform panels (per-model, and direction-collapsed in z units — read from the prediction HDF5s, not the CSV), the per-model µV boxplot, the top-30 table, the mean-µV heatmap, the chance baseline, the novel-vs-literature comparison, and the four ranking-structure panels. Every table also lands as a CSV beside its figure. Asserts the slice invariants (10,836 rows, all 6 models, S7 ⊆ S2, both directions per pair) rather than assuming them. `--results_dir`/`--out_dir`/`--predictions_root`/`--literature_csv` overridable; `--wave_units uv` and `--wave_ylim` tune the direction panels; **`--cost_overlay`** adds a cost axis to the yield panel (off by default). |
| `scripts/slurm_stage_novel_stimuli.sh` | Bridges the generator's flat output tree to the per-condition dirs the extractor and `insilico_mmn` read, and emits the `METHOD_LIST`. Idempotent, so the Phase-2 top-up re-stages in place. Self-verifying: fails naming any method with no source wavs. |
| `scripts/submit_novel_extraction.sh` | Submits the extraction arrays for all 6 models, encoding the per-model window/stimulus-root/feature-root differences once. Queries `MaxArraySize` and splits via `TASK_OFFSET`. `DRY_RUN=1` prints the sbatch lines. |
| `scripts/submit_novel_insilico.sh` | Submits in-silico MMN for all 6 models with every path redirected, and **refuses** to write into the literature predictions root. Serves both phases and the figures-for-winners re-run. |
| `scripts/report_extraction_cost.sh` | Post-hoc: measured core-h/method per model from `sacct` against the predicted table. Recovers the model from each job's log. Parameterized via `CLIPS_PER_DIR` / `N_DIRS_FULL`. |
| `scripts/report_extraction_cost_phase2.sh` | Phase-2 wrapper over the above: pins the Phase-2 shape (14 clips/dir, 254 dirs) and rolls the per-job rows up **per model**. Also serves as a completeness check — a model short of its 254 tasks is flagged rather than extrapolated over. `COST_TABLE=<file>` summarises a pasted table off-cluster. |
| `scripts/check_novel_features.py` | Completeness + integrity check on extracted features: clip counts, naming scheme, every layer present, shape/dtype consistency, ids match filenames, exactly one standard, values finite and non-zero. Run after every extraction submission; exits non-zero on any problem. |

**Ranking criteria** (identical in both phases). The unit is the **direction-instance**, not the pair — 903 pairs → 1,806 instances, matching the existing convention that regular and counter each count as one MMN observation.
- `n_agree` = how many of the 6 models show S7 at FCz, X = 0.75 µV (0–6).
- `mean_uv` = mean `trough_uv` across **only the agreeing models**; undefined when `n_agree == 0` (averaging in traces that failed S2 would mix in latencies that are not MMN latencies).
- Sort `n_agree` **desc**, then `mean_uv` **asc** (most negative wins). `n_agree == 0` sorts last. Remaining exact ties break on `(pair_id, direction)` and are logged.

**Selection** *(revised 2026-07-30; replaced a walk down the ranking to a fixed 145 pairs)*. A pair goes to Phase 2 iff **at least one of its directions reaches `n_agree >= 5`** — `MIN_AGREE_PHASE2` in `novel_search_common.py`, i.e. 5 or 6 of the 6 models saw an MMN. Both directions of every selected pair are then tested, including one that scored far below the bar, because a strong asymmetry between directions is precisely the frequency-preference artifact the counterbalanced design exists to detect, and it is only measurable if the reversal is carried. Requiring *both* directions to clear 5/6 would presuppose that answer and leave 4 pairs.

The pair count is an **output of the criterion, not an input to it**: 131 direction-instances qualify over **127 distinct pairs** (17 topping out at 6/6, 110 at 5/6), carrying 123 sub-threshold reversals along with them, for 254 method dirs. It moves if the Phase-1 scores move. The former top-145 rule filled its last 18 slots with pairs whose best direction only reached 4/6 — selected to meet a quota rather than because the models agreed — so ranking position no longer buys entry.

### 17.4 Scale of the run

Nothing in the search branches on cost. `scripts/report_extraction_cost.sh` and
`scripts/report_extraction_cost_phase2.sh` report measured spend from `sacct` after the fact; the
constants they use live in `novel_search_common.py` and are reporting-only.

| | method dirs | clips/dir | clips | array tasks (× 6 models) |
|---|---|---|---|---|
| Phase 1 | 1806 | 2 | 3,612 | 10,836 |
| Phase 2 (127 pairs) | 254 | 14 (new only) | 3,556 | 1,524 |

**`--save_plots false` on the in-silico step is not optional.** At 1,806 conditions × 2 figures ×
6 models, per-method plotting is ~21,700 matplotlib renders — one of them a ~50-subplot montage —
and would dominate the walltime. The prediction HDF5 is written either way and is what all
downstream scoring reads.

**Disk.** Extraction hooks *every* layer (24 for whisper-medium and wav2vec2-large, 12 for whisper-small and wav2vec2-medium, 6 for whisper-base, 4 for whisper-tiny) — deliberately not narrowed, so the features stay usable for layer-wise MMN profiles, re-selection, and the attention encoder once retrained. float16, T = 1500 (whisper) / 499 (wav2vec2) ⇒ **148.9 MB per clip across the 6 models** (measured ~0.89× that in practice): **≈500 GB** for Phase 1 and **≈470 GB** for Phase 2, **≈1 TB total**. Confirm free quota before Stage D and again before Stage I — a mid-array `No space left on device` throws away everything the array had done.

### 17.5 Step-by-step run guide

**Prerequisites:** everything in §17.2 pulled to the cluster; the frozen mTRF training features and the committed electrode layers already in place (§1.5). Local steps run in `mbs-env` with `PYTHONPATH="$(pwd)/src:$(pwd)/scripts"` (§16.5).

**Run every cluster command from the repo root** (`cd /work/.../multimodal-brain-scaling`). The submitters hand `sbatch` a relative `scripts/slurm_*.sh` path and resolve `outputs/` against the current directory; from anywhere else they abort with a missing-path message rather than doing something odd. Stage A needs no environment at all (standard library only, so plain `python3` works before `source env.sh`); every other local step needs `mbs-env`.

**Stage A — build the grid (seconds). Run it on BOTH machines.** `/data/` and `outputs/` are gitignored, so neither CSV travels with the repo — only the generator does. The output is deterministic (a pure function of the constants in the script), so running it in both places gives byte-identical files; do not rsync them.
```bash
python scripts/build_novel_grid_csv.py     # → data/metadata/novel_grid_frequency_metadata.csv (903 rows)
                                           # → outputs/results_novel_search/grid_index.csv
```
The cluster needs the **metadata CSV** (Stages C and E); the local machine needs the **grid index** (Stage G, for the frequency/deviance columns).

**Stage B — pre-flight checks (cluster, before the first fan-out).** `df` reports the whole shared filesystem (5.5 P, 1.6 P free as of 2026-07-26), which says nothing about whether *you* may write ~1 TB — on a shared `/work` it is a per-user or per-group quota that binds, so check that too.
```bash
scontrol show config | grep -i MaxArraySize      # measured 2026-07-26: 10001 -> 1806 fits in ONE array
df -h /work/upschrimpf1/sigfstea                 # filesystem free space -- NOT your quota
lfs quota -u $USER /work 2>/dev/null || mmlsquota --block-size auto 2>/dev/null || quota -s
lfs quota -g upschrimpf1 /work 2>/dev/null       # the group quota is the one that usually binds
```

**Stage C — synthesize audio (~minutes), then stage it.** The trailing flags are the whole point of the `"$@"` fix; without them you get 16 clips per method, not 2.
```bash
sbatch --export=ALL,METADATA_CSV=data/metadata/novel_grid_frequency_metadata.csv,\
OUTPUT_DIR=outputs/stim_gen_novel scripts/slurm_generate_stimuli.sh \
       --trial_levels 7 --num_variations 1 --models whisper,wav2vec2

# generator writes a FLAT tree; the extractor and insilico_mmn read per-condition dirs.
# slurm_stage_novel_stimuli.sh bridges them and emits the METHOD_LIST, then self-verifies.
scripts/slurm_stage_novel_stimuli.sh          # → outputs/mmn_stimuli_novel{,_wav2vec2}/method_<id>{,_counter}/
                                        # → outputs/novel_methods_phase1.txt (1806 entries)
```
It prints the per-root directory and wav counts and the set of clips-per-dir; expect `1806 dirs, 3612 wavs, clips-per-dir {2}` for each root. It exits non-zero naming any method whose source wavs are missing — a partial generation must not quietly shrink the grid, since the in-silico driver skips a missing feature dir without erroring.

**Stage D — Phase-1 extraction (the expensive step).** `submit_novel_extraction.sh` submits all 6 models, picking the 10 s window and wav2vec2 stimulus root for the wav2vec2 pair, a per-model feature root (which is what keeps whisper-base off the literature run's bare `outputs/features`), and stimulus-id naming. It queries `MaxArraySize` and splits the 1806-entry list into as many arrays as needed. On jed that is **10001**, so the whole list fits in one `--array=0-1805%200` per model and `TASK_OFFSET` stays 0 — six submissions, not twelve. The splitting path still exists for a cluster with a tighter cap.
```bash
DRY_RUN=1 scripts/submit_novel_extraction.sh    # read the 6 sbatch lines first
scripts/submit_novel_extraction.sh
```

⚠️ **Gate on one model × 10 methods before fanning out.** `scripts/report_extraction_cost.sh <jobid>` prints measured core-h per method against the predicted table and flags anything running long.

Then verify the output, because a short or malformed extraction does not announce itself:
```bash
python scripts/check_novel_features.py --model_id whisper-tiny \
  --method_list outputs/novel_methods_gate.txt --expect_clips 2
```
It reports clip counts, naming scheme, per-layer presence, shape/dtype, ids-vs-filenames, exactly-one-standard, and the projected full-grid disk from the measured MB/clip. After each full model: same command with `--method_list outputs/novel_methods_phase1.txt`; after Phase 2, `--models all --expect_clips 16`.

**Stage E — Phase-1 in-silico MMN.** `submit_novel_insilico.sh` submits one job per model with every path redirected. That redirection is the point: `slurm_insilico_mmn_electrodes.sh` defaults each path to the **literature** run, so a screen that forgets `DATA_DIR` silently overwrites `outputs/insilico_mmn_predictions/<model>/` — the comparison baseline, with no backup. The script refuses any `PREDICTIONS_ROOT` under that path.
```bash
DRY_RUN=1 scripts/submit_novel_insilico.sh      # read the 6 sbatch lines first
scripts/submit_novel_insilico.sh                # --save_plots false by default
```
`--save_plots false` is what keeps this inside 12 h: at 1806 conditions the per-method figures are ~2 renders each, one a ~50-subplot montage. `--lag_max_ms 800` and the `-3.0 / 0.0` baseline multipliers are hardcoded in the wrapper and **must not** be overridden — they are what make these results comparable to the literature screen. Sanity check: the held-out r printed by `fit_mapping` should match the literature run's log for that model, confirming the frozen mapping refit is bit-identical.

**Stage F — sync and score (local).**
```bash
for m in whisper-tiny whisper-base whisper-small whisper-medium wav2vec2-medium wav2vec2-large; do
  mkdir -p outputs/insilico_mmn_predictions_novel/$m
  rsync -av "sigfstea@jed.hpc.epfl.ch:/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/insilico_mmn_predictions_novel/$m/"'*.h5' \
        "outputs/insilico_mmn_predictions_novel/$m/"
done
python scripts/analyze_mmn_s7_roi.py \
  --predictions_root outputs/insilico_mmn_predictions_novel \
  --dip_uv_threshold 0.75 \
  --out outputs/results_novel_search/phase1_mmn_s7_roi.csv
```

**Stage G — rank and select.** Phase 2 takes every pair with a direction at `n_agree >= 5` (`--min_agree`); the count follows from the data, and no budget arithmetic enters — nothing reads `sacct`.
```bash
python scripts/rank_novel_phase1.py                            # → 127 pairs, 254 method dirs
python aux/analysis_novel_search/plots/novel_search_plots.py    # figures 1–2
```

**Stage H — Phase-2 audio (the 14 missing deviants).**
```bash
python scripts/build_novel_phase2_csv.py
sbatch --export=ALL,METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv,\
OUTPUT_DIR=outputs/stim_gen_novel_phase2 scripts/slurm_generate_stimuli.sh \
       --trial_levels 3,5,7 --num_variations 5 --models whisper,wav2vec2

# same staging script, pointed at the subset; the dirs go from 2 wavs to 16 in place
METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv \
SRC=outputs/stim_gen_novel_phase2 \
METHOD_LIST_OUT=outputs/novel_methods_phase2.txt \
    scripts/slurm_stage_novel_stimuli.sh      # expect clips-per-dir {16 2}, NOT {16}
```
The mixed `{16 2}` is correct and expected: only the 127 selected pairs are topped up, and the other 776 keep the 2 clips Phase 1 gave them. A bare `{16}` would mean you regenerated the whole grid.

Then **checksum the overlap**: the regenerated standard and `N7_var1` deviant must be byte-identical to Phase 1's, or the Phase-1 features are invalid for reuse and the 28-clip saving evaporates. (Verified locally in advance — `generate_deviant_sequence` seeds on `(method_id, N, v)` alone, independent of grid size, row count and worker count.)

**Stage I — Phase-2 extraction.** No index arithmetic: with `MMN_NAME_BY_STIM_ID=true` and `MMN_OVERWRITE` unset, pointing the extractor at the now-16-wav directory extracts **exactly the 14 missing clips** and skips the 2 that exist. Re-run Stage D's submitter against the Phase-2 list — and override **both** env vars:
```bash
METHOD_LIST=$PWD/outputs/novel_methods_phase2.txt \
METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv \
    scripts/submit_novel_extraction.sh        # 254 dirs = 127 pairs × 2, one array per model
```
`METADATA_CSV` is **not** optional here: the submitter refuses when the method list and the metadata CSV describe different condition sets, and the default full-grid CSV yields 1,806 against the list's 254. Verify the plan implies **3,556** new clips (254 × 14), not 4,064 (254 × 16) — the 16-clip figure means you are re-extracting Phase 1 for nothing. Gate on one model × 5 methods first, then run `scripts/report_extraction_cost_phase2.sh <jobids>` on the finished arrays.

**Stage J — Phase-2 in-silico, final ranking, deliverables.** Same submitter, pointed at the subset and a separate predictions root (the features stay in the Phase-1 roots — Phase 2 topped those directories up in place, so `FEATURES_TAG` does not change):
```bash
METADATA_CSV=data/metadata/novel_grid_phase2_subset.csv \
PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_novel_phase2 \
    scripts/submit_novel_insilico.sh
# ...then rsync back as in Stage F, and:
python scripts/analyze_mmn_s7_roi.py \
  --predictions_root outputs/insilico_mmn_predictions_novel_phase2 \
  --dip_uv_threshold 0.75 --out outputs/results_novel_search/phase2_mmn_s7_roi.csv
python scripts/rank_novel_phase2.py
python aux/analysis_novel_search/plots/novel_search_plots.py    # all three figures
```
Finally, per-pair waveform figures for the headline set only:
```bash
SAVE_PLOTS=true METHODS=method_1042,method_1042_counter \
PREDICTIONS_ROOT=outputs/insilico_mmn_predictions_novel_figs \
    scripts/submit_novel_insilico.sh
```

**Resource reference:**

| Stage | Where | Time | CPUs |
|---|---|---|---|
| A — build grid | login node **and** local | seconds | — |
| C — generate stimuli | jed (standard) | 1 h | 16 |
| C / H — `slurm_stage_novel_stimuli.sh` | login node (not sbatch) | ~minutes | — |
| D / I — `submit_novel_extraction.sh` → MMN extract | login node → jed | 24 h/array task | 8 |
| E / J — `submit_novel_insilico.sh` → in-silico MMN | login node → jed (standard) | 12 h | 12 |
| F–G, plots | local | minutes | — |

### 17.6 Invariants and failure modes

**Check at each stage:**

| After | Expect |
|---|---|
| Stage A | 903 rows; header byte-identical to the literature sheet; ids 1001–1903 |
| Stage C | 1806 dirs per family root × exactly 2 wavs; 30 s (whisper) / 10 s (wav2vec2); FFT peaks at the intended frequencies |
| Stage D | 1806 × 6 = 10,836 feature dirs, 2 `feats_delta_t-*.h5` each |
| Stage E | 1806 conditions per model in the predictions h5 |
| Stage F | 10,836 rows after filtering to FCz / mTRF / X = 0.75; S7 ≤ S2 (script-asserted); all 6 models present |
| Stage G | 1,806 ranked direction-instances; 131 qualifying at `n_agree >= 5` over 127 unique pairs; 254 method dirs in the METHOD_LIST |
| Stage I | exactly 3,556 **new** clips (254 × 14), never 4,064 |

**Failure modes specific to this phase:**

- **A method whose feature dir is missing is silently skipped** by the in-silico driver — a partial extraction quietly shrinks the grid rather than erroring. Count the feature dirs before Stage E.
- **Never mix HDF5 naming schemes in one feature directory.** `load_layer_features` globs `*.h5` and concatenates, while the id→row map keeps only the *last* occurrence of each id, so a mixed directory misaligns features against ids **without raising**. `extract_features_delta_t.py` now refuses to start in that situation; the fix is a fresh `--output_dir`, never deleting half the files. This is also why `MMN_NAME_BY_STIM_ID` defaults to **false** — the committed literature features use the legacy names.
- **`MMN_FEATURES_ROOT` must be set for whisper-base** in particular: it is the one model whose default MMN feature root is the bare `outputs/features`, which would collide with the literature features rather than sitting in a model-scoped subdirectory.
- **A stale `METHOD_LIST` after a grid rebuild is the worst failure here, and it happened.** Rebuilding the grid renumbers every `method_id`, so if the stimuli are not re-staged, `method_1001` is one frequency pair in the audio and a different one in the CSV that scores it. The features come out internally consistent and simply describe the wrong pairs — no check downstream catches it. Both submitters now compare the method list against the metadata CSV condition-by-condition and refuse on any mismatch, and the extraction submitter also checks the named conditions were actually staged. The tell before the guard existed: the array range. `--array=0-1055` means 1056 conditions; the 43-point grid is `0-1805`.
- **Forgetting `DATA_DIR` in Stage E overwrites the literature predictions** — the comparison baseline. There is no backup; the wrapper defaults to the literature path by design for the literature run. `submit_novel_insilico.sh` sets it for you and refuses any `PREDICTIONS_ROOT` under `outputs/insilico_mmn_predictions/`, so use it rather than hand-rolling the sbatch line.
- **`--save_plots false` is not optional at 1,806 conditions.** With plots on, expect ~4 h/model of single-threaded matplotlib on top of the compute, likely exceeding the 12 h walltime for the slower models.

---

## 18. MMN "realness" checks at FCz — N-effect and deviance scaling (2026-08-11)

**Scope.** Two dose-response checks on whether the in-silico MMN behaves like a human/clinical MMN, both asking the same question along axes the MMN literature has strong priors about: does the trough deepen as the paradigm makes the oddball more surprising? **(a) the N-effect** — more standards between deviants → deeper MMN; **(b) deviance scaling** — larger standard→deviant separation → deeper MMN. **Not covered:** any re-fitting. The mTRF mappings and the committed §1.5 electrode layers are frozen; the one cluster re-run here is a re-*apply* of the existing mapping, needed only because the driver was discarding data (§18.2).

**Site:** the **FCz electrode only** — no parcels at any stage. mTRF only; the encoder is out of scope (it clears S7@0.75 in 16 of 360 cells across all ROIs). Six models — whisper-large excluded, for two independent reasons: its predicted µV run ~20–35× every other model (median S7-passing LIT trough at FCz −37.3 µV against a −1.08…−1.75 µV band for the other six), which would dominate any raw-µV pooled mean and force a symlog axis; and it was never run in the novel search, so keeping it would make `lit` and `p2` incomparable.

**Everything lives in `aux/analysis_MMN_realness_checks/`.** Memo: `mmn_dose_response_fcz_memo.md`. Figure-reading guide: `README.md` — read that before the figures; 34 PNGs is not a self-describing set.

### 18.1 The result in one paragraph

**The models are dose-sensitive in the MMN-like direction on both axes, but far more in *whether* a mismatch response occurs than in *how deep* it is.** Across the N range the NOVEL-P2 trough deepens by 0.085 µV on a ~1.3 µV baseline (~6%) while the probability of producing a criterion-passing MMN rises 54% → 59% (Cochran–Armitage z = +6.36, p = 2e-10). Human MMN amplitude is graded; this looks closer to a near-threshold detection process. Hold that loosely, though: ridge shrinkage plus a linear mTRF readout compresses amplitude, and removing one layer of censoring (S7 → S2) nearly doubled the measured effect, so some of the flatness is demonstrably measurement rather than model. For deviance the effect is real but small and **range-limited**: ρ = −0.15 up to two octaves, reversing to +0.10 above, where a "deviant" is arguably a different sound. Both are written up with their limits in the memo.

**Two results that only appeared because the analysis was decomposed:**

- **LIT's headline ρ = −0.31 dissolves.** Split at the point NOVEL-P2 begins, it is flat within each half (−0.09, p = 0.51 below 3.16 st; −0.09, p = 0.48 in the 4.5–12 st overlap). The whole correlation is a −1.13 → −1.36 µV *step* between two dense method clusters that also differ in SOA (200–1000 ms) and duration (50–200 ms). It is a two-cluster contrast, not a gradient, and should not be quoted alone.
- **The pooled N line averages over models that disagree in direction.** In LIT, whisper-medium deepens in 77% of its stimuli (median −0.30 µV) while wav2vec2-medium (35%), whisper-base (38%) and whisper-small (40%) run the other way. Within NOVEL-P2 the *rate* effect is likewise carried entirely by the three larger models (whisper-medium z = 4.33, wav2vec2-medium z = 4.15, wav2vec2-large z = 5.55, all p < 1e-4) while whisper tiny/base/small are flat. Suggestive for a scaling project; six models across two architectures, so a hypothesis, not a law.

### 18.2 Why a cluster re-run was needed at all

Each method's 15 deviants are a 3 × 5 grid of `N ∈ {3,5,7}` × `variation ∈ {1..5}`, so **N lives entirely inside the average the pipeline takes**. `analyze_method → finalize_method` returns `dev_stack [15, n_t, n_elec]` and `dev_ids`, but `insilico_mmn_electrodes.py` wrote only `deviant_mean` and discarded the rest. Verified absent in every committed electrode h5 for both sources, though `n_deviants = 15` confirms the trials were computed. One patch, two re-runs — deviance (§18.4) needed none of this and shipped first.

The `deviants` stack *does* survive in the **parcel** h5s (and in `insilico_mmn_attn.py`'s output), which is what let the entire per-trial path be validated on real data before any cluster time was requested: per-trial scoring reproduced the pipeline's own stored `n7v1_peak` to **2.1–3.0 × 10⁻⁷** across all 288 (model × method) cells, and emitted exactly the expected 4,320 rows. Those parcel numbers are a plumbing check and appear in no figure.

### 18.3 Code changes

**`scripts/insilico_mmn_electrodes.py`** now also writes, per method group:

| dataset | shape / dtype | contents |
|---|---|---|
| `deviants_fc` | `[n_dev, n_t, n_fc]` float32, gzip-4 | `dev_stack` sliced to the fronto-central cluster |
| `deviants_fc_electrodes` | `S8` | that axis's electrode names, in slice order |
| `deviant_ids` | `S40` | labels the trial axis of both it and `deviant_mean` |

- **Fronto-central slice, not the full stack.** `CLUSTERS["frontal"] + CLUSTERS["central"]`, intersected with the electrodes that survived the NC floor rather than assumed present. The full 47-electrode stack would take each h5 from ~26 MB to ~230 MB for 40 electrodes of unused scalp; this costs ~35 MB/model and removes a second cluster round-trip if Fz is ever wanted. The analysis is FCz and nothing else — this is a storage decision only.
- **`n_dev` is never hardcoded to 15.** It is sized from `dev_stack`, because `n_deviants` is `--trial_levels × --num_variations` and other configs of this same driver legitimately produce 1. The 15-trial assertion lives in the analysis script.
- **Nothing else moved** — not the parcel driver, the mTRF fit, the layer map or the baseline multipliers, so both runs stay comparable to their committed screens.

Three tests in `tests/test_insilico_mmn.py`, the load-bearing one being that `deviants_fc[:, :, FCz].mean(0)` reproduces the stored `deviant_mean` FCz column to float32 tolerance — the guarantee that the per-trial stack and the committed method-level results describe the same traces.

### 18.4 New artefacts

Under `aux/analysis_MMN_realness_checks/`:

| file | what it is |
|---|---|
| `mmn_dose_response_common.py` | the shared vocabulary: sources, sets, gates, palette, the one summary statistic, the balanced-N filter |
| `analyze_mmn_per_trial_n.py` | per-trial FCz scoring → the two tidy CSVs (4,320 + 22,860 rows) |
| `n_effect_plots.py` | Plot 1 (N) — pooled, per-model, pass-rate, change-distribution + 4 stats CSVs |
| `deviance_scaling_s7gated.py` | Plot 2 (deviance) — pooled, per-model, pass-rate, overlap diagnostic + stats CSV |
| `README.md` | **the figure-reading guide** — vocabulary, the five figure families, five ways to misread them |
| `handoff_per_trial_deviants_n_effect.md` | the cluster runbook (§18.5) |
| `plots/<gate>/<set>/*.png`, `svgs/…` | 34 figures, mirrored trees |

Scoring is imported, never reimplemented: `compute_z_diff`, `zscore_baseline`, `trace_stats`, `decide` from `analyze_mmn_criteria`; `uv_diff_wave` from `analyze_mmn_criteria_s5_s6`; knobs identical to `analyze_mmn_s7_roi.py` (WINDOW 100–240, RECOVERY_MS 120, RECOVERY_FRAC 0.5, CTRL_WINDOW 300–440, EDGE_GUARD_BINS 1).

### 18.5 The two re-runs

Both go to **new sibling roots** — `outputs/insilico_mmn_predictions_soafix_pertrial` and `..._novel_phase2_pertrial` — so the committed h5s are never written. `insilico_mmn_electrodes.py` opens with `h5py.File(path, "w")`, so pointing it at the existing roots destroys them; the siblings are what make the re-run reversible. Both names clear `submit_novel_insilico.sh`'s guard, which refuses only `outputs/insilico_mmn_predictions` and paths beneath it. Budget roughly double the prediction footprint (NOVEL-P2 is ~602 MB, plus ~35 MB/model of `deviants_fc`); check `lfs quota -g upschrimpf1 /work` first.

Commands and the three verification gates are in `handoff_per_trial_deviants_n_effect.md`. All three passed on 2026-08-10. The one worth restating: **gate 3 must score into new CSV paths.** `outputs/results_soafix_full/mmn_s7_roi.csv` and `outputs/results_novel_search/phase2_mmn_s7_roi.csv` are the committed inputs the deviance analysis reads, and re-scoring in place would silently swap them for re-run output. Gate 3 is now an explicit MATCH/DRIFT comparison against the committed FCz S2/S7 table, which reproduced exactly — the patch adds a dataset and perturbs no verdict.

> **Gate 1 matters more than it looks.** Gate 3 passing is *also* consistent with the patch never having been applied: if the cluster's `git pull` did not take, the re-run produces byte-identical verdicts with no `deviants_fc` written. Gate 1 is the only check that proves the patch ran.

### 18.6 Reporting decisions worth not re-litigating

Each of these was changed at least once during review; the reasoning is in the code comments and `README.md`.

- **Two gates, both shipped.** `S7@0.75` (shape + µV floor) is the headline; **`S2`** (shape only) is the uncensored companion under `plots/s2/`. The floor removes the shallow tail *by construction* — exactly the traces a weak dose-response moves — so S7 numbers are a **lower bound**. Not a small effect: S7 keeps 48% of LIT's S2 traces and 73% of NOVEL-P2's, and the paired N7−N3 shift is −0.085 µV at S2 (p = 9e-5) against −0.047 µV at S7 (p = 0.004). The pooled cell-level ρ only reaches significance at S2.
- **The N amplitude figures use a *balanced* set** — only stimuli producing a criterion-passing MMN at N=3 *and* 5 *and* 7. Without it the gate admits a different population at each N and the mean moves because membership changed, not because anything deepened. That was not hypothetical: the unbalanced LIT panel rose visibly (unpaired p = 0.003) while paired it was null (p = 0.23, 50% of stimuli — a coin flip). The **rate** figures are deliberately not balanced; their denominator is every condition by definition.
- **One summary statistic everywhere: median with a 95% bootstrap CI**, with `__mean_sem` companions on an identical axis so the choice is auditable. 0 of 31 per-bin distributions are consistent with normality (Shapiro–Wilk), skewness −2 to −3; a median has no closed-form standard error, so ±1.96·SE would be unsupported. It does not manufacture a result — the N=3→7 change agrees to ≤0.02 µV between the two statistics.
- **Linear semitone x-axis, not log.** Semitones are already log-frequency, so a log axis is log-of-log in Hz; it compressed the >24 st region where the trend reverses; and it does not even give more even bins (lit_p2 widths span 7.5× linearly against 11.7× in log₂).
- **The `p2_top100` fourth panel is selected on the outcome twice** — NOVEL-P2 is already the ≥5/6-agreement subset, and ranking within it by agreement and depth selects again on the plotted quantity (S7 pass rate 83% vs p2's 56%). It bounds the best responders; it does not estimate an effect size, and no statistic is computed on it.
- **Figures are reproducible.** Fixed `svg.hashsalt`, `metadata={"Date": None}` on SVG saves, and a seeded bootstrap: identical input gives byte-identical output. Without this every re-render rewrote all tracked SVGs with thousands of no-op line changes.

### 18.7 Interpretation limits (short form; full list in the memo)

- **N is confounded with oddball probability by construction.** The generator sets rare-tone probability to 1/(N+1) (`00aa_generate_audio_stimuli.py:339`), so N = 3/5/7 is 25% / 16.7% / 12.5%. A deepening trough is MMN-like but cannot separate local spacing from global rarity.

  > **The generator's labels are inverted relative to the roles — check this before reading that code.** `generate_deviant_sequence`'s own docstring is explicit: *"the background (frequent) tone is the **deviant** frequency and the rare (oddball) tone is the **standard** frequency"*. So in a deviant sequence the suffix `["standard"] + ["deviant"] * N + ["standard"]` is **one oddball, N frequent tones, one oddball**, and `p_standard = 1.0 / (N + 1)` is the **oddball** probability, not the standard's. Read in isolation, both lines say the opposite of what they mean. The figures label the axis in *role* terms — "N standards between deviants", oddball probability 25% / 16.7% / 12.5% — which is correct, and matches the classic MMN reading, precisely because the code's names invert. The time-locked final tone is always the rare oddball.
- **NOVEL-P2 is selected on the outcome variable** (`n_agree ≥ 5` at FCz in phase 1), which is essentially the quantity being plotted. It answers *"among stimuli that reliably evoke a model MMN, does amplitude scale?"* — not *"does deviance drive amplitude?"*. Its range is also truncated from below at 4.50 st: 0 of 42 pairs below 2 st survived, so the small-deviance end where the trend is most diagnostic is absent and LIT is the only coverage there.
- **The `lit_p2` deviance slope is partly a between-source contrast** — LIT supplies nearly all conditions below 4.5 st, NOVEL-P2 nearly all above 12. Inside the 4.50–12 st overlap the two sources agree closely (ρ = −0.089 vs −0.094) but **both are non-significant**: agreement on a null, which rules out a source artefact without establishing a deviance effect. Never quote the `lit_p2` ρ without it. The caution does **not** apply to the `lit_p2` N-effect, where both sources span the identical N.
- **Predicted µV are not literature EEG µV.** Ridge shrinks the amplitude and the 0.75 µV floor is calibrated to the models' own distribution, not a clinical scale. Only direction and monotonicity transfer.

### 18.8 Failure modes specific to this phase

- **`deviants_fc` absent** → `analyze_mmn_per_trial_n.py` exits with the re-run instruction rather than producing nothing. The committed electrode h5s are in this state until the §18.5 re-runs land.
- **Scoring the re-run into the committed CSV paths** destroys the deviance analysis's inputs. See gate 3.
- **A stale `phase2_final_ranking.csv`** would silently change which stimuli the `p2_top100` panel contains; `set_frame` asserts all 100 ranked methods match the scored data and refuses otherwise.
- **`whisper-large` leaking back in** would break `lit`↔`p2` comparability. It is filtered at load time in `load_fcz`, not at plot time, so it cannot reach a panel or an appendix table.
- **The per-trial CSVs are untracked**, matching the existing `mmn_s7_roi.csv` convention. Regenerating them needs the `_pertrial` prediction roots, not the committed ones.
