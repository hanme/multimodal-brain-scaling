# Project Plan: Adapting mbs for Auditory EEG Encoding Models

> ⚠️ **PARTIALLY SUPERSEDED (2026-06-12).** The *encoding method* and its layer conclusion in
> **Phase 4b** below (per-time-bin Ridge; "blocks.2 is best") are **superseded** by
> `aux/project_plan_20260611.md` (Workstream A: mTRF lagged shared-weight Ridge). The old
> per-bin lag curve was a slow-autocorrelation confound; the de-confounded answer is a proper
> auditory TRF (~120–140 ms) with **mid-depth (~blocks-3)** layers best, not blocks-2.
> Everything else here — Phases 0–3 infrastructure, the Delta_T decision, the Phase 5 model-spec
> table, and the Phase 6 MMN/in-silico plan — **remains canonical.** See the handover's
> "UPDATE 2026-06-11/12" section for the summary.

**Goal:** Use the `mbs` ridge regression pipeline (originally built for visual brain-model alignment) to
train auditory EEG encoding models — i.e., linear readouts from audio model layers to EEG electrode
responses. The immediate scientific goal is to identify which layer of which audio model (Whisper
variants, wav2vec2, AST, VGGish) best predicts EEG responses in healthy subjects, as a validation
and layer-selection tool feeding into Sophie's MMN unit-selection pipeline.

**Connection to Sophie's pipeline:** see `aux/schizophrenia_pipeline_Sophie_2026.md`.
Sophie's pipeline selects model units that behaviorally mimic MMN. This pipeline asks whether those
same units predict real EEG — a neural validity check. The best-predicting layer also provides a
principled answer to "which layer should Sophie use?" rather than defaulting to the deepest one.

---

## Scientific context and two-dataset strategy

| Dataset | Paradigm | Purpose here |
|---|---|---|
| ds004408 (OpenNeuro) | Naturalistic speech (audiobook) | Phase 1–4: build and validate pipeline machinery |
| MMN EEG dataset (TBD) | Oddball tones, standard/deviant | Phase 5+: actual scientific question |

**ds004408** (Di Liberto / Broderick / Lalor lab): 128-ch EEG at 512 Hz, subjects listened to
"The Old Man and the Sea" audiobook segments. WAV stimuli with TextGrid word/phoneme alignments.
Good for establishing that the pipeline works, selecting the best Whisper layer, and producing a
first encoding model. Not a MMN dataset — different paradigm from Sophie's stimuli.

**MMN EEG dataset:** needed to directly validate Sophie's MMN units. Ask Gokce / Sophie / lab
whether one already exists (patient + control). ERP-CORE (open, N=40, has MMN paradigm) is a
fallback. Once obtained, Phase 3 data formatting applies identically.

**Key design decision (discuss with Gokce):**
For naturalistic speech, Whisper outputs `[1500 time bins, d_model]` per 30s segment — not a single
feature vector. Two options:
- **Mean pool over time** (simple): one vector per segment → plugs directly into existing Ridge code
- **TRF / lagged regression** (principled): keep time series, fit a linear filter per lag

Start with mean pooling for the pilot. It reuses 100% of the existing `mbs` evaluation code with
zero modification. Switch to TRF if mean pooling gives poor encoding scores.

---

## Development conventions

**Test-driven development:** all new code is covered by tests in `tests/` before (or alongside) the
implementation. Run the fast suite at any point with:

```bash
python -m pytest tests/ -m "not slow" -q
```

Tests that download model weights or read real EEG files are marked `@pytest.mark.slow` and excluded
from the default run. The number of failing tests is the implementation progress tracker: every phase
completed turns a block of red tests green.

**Adding a new auditory model (Phase 5 extension):** follow the exact same pattern used for Whisper.
For each new model:

1. **Write tests first** — add shape tests and a slow integration test to `tests/test_backbone_audio.py`,
   window/ID tests to `tests/test_datasets_audio.py` if the dataset handling differs, and update
   `tests/test_import_smoke.py::test_audio_module_imports` with any new module names.
2. **Register** — add the model ID to `audio_models.py` and document the dotted layer paths in
   `aux/XX_handover_for_Sophie-md` (the layer path table).
3. **Verify** — run `python -m pytest tests/test_backbone_audio.py -m "slow"` to confirm the forward
   pass produces the expected shape before wiring into the full pipeline.

This ensures every model in Sophie's list has documented, verified behavior before it enters a science
run.

---

## Phase 0 — Environment ✅ Done

- [x] Fork `epflneuroailab/multimodal-brain-scaling` → `hanme/multimodal-brain-scaling`
- [x] Clone fork to `/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/`
- [x] Complete `uv sync --python python3.11 --extra evaluation --extra analysis --extra visualization --extra dev --extra audio`
      (run with `module load gcc/13.2.0 python/3.11.7` active; cache at `/work/upschrimpf1/mehrer/.cache/uv`)
- [x] Verify: `mbs-extract-features --help` works

---

## Phase 1 — Audio model backbone loader ✅ Done (2026-06-01)

**File:** `src/mbs/extraction/modeling/backbones/audio_models.py`

- [x] Added `openai-whisper` to `pyproject.toml` under `audio` optional dependency group
- [x] Implemented `WhisperBackboneWrapper` + `WhisperTransform`:
  - Loads by ID (`whisper-tiny/base/small/medium/large`)
  - `WhisperTransform`: waveform → `[n_mels, 3000]` log-mel via `whisper.log_mel_spectrogram`
  - `load_whisper(model_id)` → `(WhisperBackboneWrapper, WhisperTransform)`
  - Layer paths: `blocks.{i}` (factory prepends `backbone.` automatically)
- [x] Registered `audio` backbone source in `create_backbone()` registry
- [x] Smoke test: Whisper-base loads and produces `[1500, 512]` per layer

**Model IDs to support (Sophie's 9 models):**

| Model ID string | Source |
|---|---|
| `whisper-tiny` | openai-whisper |
| `whisper-base` | openai-whisper |
| `whisper-small` | openai-whisper |
| `whisper-medium` | openai-whisper |
| `whisper-large` | openai-whisper |
| `wav2vec2-base` | HuggingFace `facebook/wav2vec2-base-960h` |
| `wav2vec2-large` | HuggingFace `facebook/wav2vec2-large-960h` |
| `ast` | HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` |
| `vggish` | torchhub (CPU only) |

Start with whisper-base only. Add others once pipeline is end-to-end.

---

## Phase 2 — Audio stimulus dataset loader ✅ Done (2026-06-01)

**File:** `src/mbs/extraction/data/datasets_audio.py`

- [x] Implemented `AudioSegmentDataset`: directory of `.wav` files → `(waveform_np, stimulus_id)` pairs
  - Resamples to 16 kHz, converts to mono, sub-segments into sliding windows
  - Stimulus ID format: `audioXX_SSSSSSS` (matches EEG formatter exactly)
- [x] Added `--dataset_type audio` to `mbs-extract-features` CLI
- [x] Full temporal activations stored as `[n_stimuli, T, d_model]` (default)
- [x] `--mean_pool_time true` flag for mean-pool pilots → `[n_stimuli, d_model]`

---

## Phase 3 — EEG data formatter

**New script:** `src/mbs/data_prep/format_eeg_hdf5.py` *(written and run — output at `outputs/neural_data/broderick2018_30s.h5`)*

**Dataset facts (verified by direct inspection, 2026-06-01):**

| Property | Value |
|---|---|
| Local path | `/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea` |
| Subjects | 19 (sub-001 … sub-019) |
| Runs per subject | 20 (run-01 … run-20, one per audio segment) |
| Audio stimuli | 20 WAV files (`stimuli/audio01.wav` … `audio20.wav`), stereo 44100 Hz, **177–202 s each (~3 min)** |
| EEG format | BrainVision (`.vhdr/.eeg/.vmrk`), Brain Products amplifier |
| EEG sampling rate | 512 Hz |
| EEG channels | 128, named A1–A32, B1–B32, C1–C32, D1–D32 (BioSemi Active2 layout) |
| Event markers | **None** — vmrk files are empty. Each run IS the continuous EEG during one audio. |
| Temporal alignment | Per README: "starts are aligned, EEG longer to a varying extent" — EEG and audio both start at t=0, EEG continues 15–25 s past audio end. Formatter trims EEG to audio duration. |
| Sub-segmentation needed | Yes — audio is ~3 min but Whisper/wav2vec2 need 30s/10s windows |

**Stimulus count with different window/stride settings (Whisper, 30s window):**

| Stride | Windows/run | Total stimuli | Train (16 runs) | Test (4 runs) |
|---|---|---|---|---|
| 30 s (non-overlap) | ~6 | ~120 | ~96 | ~24 |
| 10 s | ~16 | ~320 | ~256 | ~64 |
| 5 s | ~31 | ~620 | ~496 | ~124 |

For whisper-base (d=512), a stride of 5–10 s gives 256–496 training stimuli — still somewhat underdetermined per time step. See Open Question 1. The formatter defaults to 10 s stride.

Convert raw EEG data to the HDF5 format that `mbs-evaluate-committed-layers` expects.
The existing format (from `evaluation_helpers.py`) is:

```
neural_benchmark.h5
├── attrs: subjects, rois, splits, max_nc
├── train/
│   ├── stimulus_ids          [n_train_stimuli]
│   └── neural_data/
│       └── {subject}/
│           └── {roi}         [n_train_stimuli, n_channels]  (or [n_stimuli, T, n_channels])
├── test/
│   └── (same structure)
└── noise_ceilings/
    └── {subject}/
        └── {roi}             [n_channels]  (Pearson-r scale, i.e. sqrt(split-half reliability))
```

**Status: ✅ Done — output at `outputs/neural_data/broderick2018_30s.h5`**

- [x] Parse ds004408 BIDS structure and match to stimulus WAV IDs
- [x] Sub-segment continuous EEG into 30s windows (10s stride), downsample to 50 Hz (Whisper grid)
- [x] ROIs defined: Fz, FCz, Cz, T7, T8, Pz, F3, F4, frontal_cluster, temporal_cluster, central_cluster, whole_brain
- [x] Noise ceilings: split-half across subjects, Spearman-Brown corrected → `[T, n_ch]`
- [x] Train/test split: 4 runs held out as test (runs [2, 9, 13, 14]); 252 train / 62 test stimuli
- [x] Validated against `load_neural_data()` schema

---

## Phase 4a — End-to-end pilot run, mean-pool (whisper-base × ds004408)

**Purpose:** quick sanity check only. Collapses the temporal axis on both sides so the existing
(non-temporal) `mbs-evaluate-all-layers` code can be used unchanged. Does NOT produce the main
scientific output — that is Phase 4b.

### What "mean-pool" means here

| Side | Shape before | Shape after | How |
|---|---|---|---|
| Model features | `[n_stim, T=1500, d_model]` | `[n_stim, d_model]` | `--mean_pool_time true` in extract step |
| EEG neural data | `[n_stim, T=1500, n_ch]` | `[n_stim, n_ch]` | `collapse_temporal_hdf5.py` averages over T |
| Noise ceiling | `[T=1500, n_ch]` | `[n_ch]` | same script, mean over T |

Mean-pooling the model features compresses 1500 × 20ms = 30s of context into one vector per stimulus.
Mean-pooling the EEG compresses 30s of brain response into one vector per stimulus.
The Ridge then maps model → brain at the level of whole-stimulus averages.

**This is a weak test.** The temporal structure (which latency of the EEG does which layer predict?)
is the scientifically interesting question and requires Phase 4b. Phase 4a is only used to verify
that there is *any* predictive signal before investing in Phase 4b compute.

### Step-by-step commands (executed 2026-06-01)

```bash
# Step 1: extract mean-pooled features [n_stim, d_model]
python -m mbs.extraction.extract_features \
  --model_id whisper-base \
  --backbone_source audio \
  --dataset_type audio \
  --data_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/stimuli \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --output_dir outputs/features/whisper-base-meanpool/ \
  --mean_pool_time true \
  --window_duration 30.0 \
  --window_stride   10.0 \
  --max_feature_dim 0 \
  --batch_size 8 \
  --num_workers 0

# Step 2: create mean-pooled EEG HDF5 from the temporal one
#   collapse_temporal_hdf5.py averages [n_stim, T, n_ch] → [n_stim, n_ch]
#   and [T, n_ch] noise ceilings → [n_ch] (mean over time bins)
python -m mbs.data_prep.collapse_temporal_hdf5 \
  --input_path  outputs/neural_data/broderick2018_30s.h5 \
  --output_path outputs/neural_data/broderick2018_30s_meanpool.h5

# Step 3: layer sweep (existing code, unchanged)
mbs-evaluate-all-layers \
  --model_id whisper-base \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --features_dir outputs/features/whisper-base-meanpool/ \
  --data_hdf5_path outputs/neural_data/broderick2018_30s_meanpool.h5 \
  --output_dir outputs/layer_search/whisper-base-meanpool/ \
  --exclude_whole_brain false \
  --overwrite true
```

### Bugs fixed during Phase 4a bringup (2026-06-01)

Several issues were discovered and fixed while executing Phase 4a for the first time.
All fixes are already in the codebase and do not need to be re-applied.

| File | Bug | Fix |
|---|---|---|
| `extract_features.py` `parse_args` | `--backbone_source` did not include `audio` as valid choice | Added `"audio"` to `choices` |
| `extract_features.py` `main` | `--dtype float16` (default) caused mixed-dtype error on CPU (Whisper LayerNorm calls `x.float()` internally) | Auto-downgrade to float32 when device is CPU |
| `configs/extraction/audio/whisper_base_layers.json` | Layer names had `backbone.blocks.{i}` (already prefixed), but factory prepends `backbone.` again → double prefix | Changed to `blocks.{i}` (path relative to backbone, without prefix); factory adds `backbone.` |
| `configs/extraction/audio/whisper_base_layers.json` | Missing `position` field required by `mbs-evaluate-all-layers` | Added `"position": 0.0…1.0` (normalized depth, 6 blocks evenly spaced) |
| `evaluation_helpers.py` `load_neural_data` | Returned raw byte strings; `load_layer_features` returns decoded strings → dict lookup failed | Added `decode('utf-8')` after loading stimulus IDs |
| `evaluation_helpers.py` `compute_metrics` and `pearsonr_score` | sklearn squeezes single-output predictions to 1D; `y_true` shape `(n, 1)` vs `y_pred` shape `(n,)` caused `pearsonr` to fail | Added `y_pred = y_pred.reshape(-1, 1)` when shapes mismatch |

### Status ✅ Done (2026-06-01)

All three steps complete. Results in `outputs/layer_search/whisper-base-meanpool/`.

**Results (held-out test set, pearsonr):**

| ROI | Best layer | pearsonr | pearsonr_nc |
|---|---|---|---|
| Fz | blocks.4 | 0.867 | > 1 (NC underestimate) |
| T7 | blocks.4 | > 0.82 | — |
| T8 | blocks.4 | > 0.82 | — |
| Pz | blocks.4 | > 0.82 | — |

Success criterion exceeded. Phase 4b (temporal) is the main scientific output — see below.

---

## Design decision: Full_T vs Delta_T activation extraction

> **Decided 2026-06-02: we use Delta_T.**

Sophie's pipeline extracts activations in two modes:

- **Full_T:** single forward pass on the complete stimulus waveform → `[T, d_model]`.
  Each time bin t has "seen" the entire future of the stimulus (non-causal).
- **Delta_T:** T separate forward passes, each on a progressively zero-padded waveform
  (`[s_1, …, s_i, 0, 0, …]`), collecting the output at position i. Causal: each bin
  sees only past context.

**Decision: we use Delta_T.** The reason is scientific, not just preference:

The brain at time t had only heard audio up to t. Using Full_T features at t gives the
model information the brain never had. For the **MMN EEG dataset** (Phase 6) this is a
fundamental confound: the deviant is *surprising* precisely because the brain did not see
it coming. A Full_T model at the deviant position already encoded it from future context,
so its representation does not carry the prediction-error signal the brain shows.

Delta_T is implemented in `src/mbs/extraction/extract_features_delta_t.py`
(CLI: `mbs-extract-features-delta-t`, or `python -m mbs.extraction.extract_features_delta_t`).
SLURM submission script: `scripts/slurm_extract_delta_t.sh`.

**Compute cost:** ~13 min per stimulus on CPU (whisper-base, 1500 bins, batch_t=4).
Full 314-stimulus run: ~3.5–5 h on 20 parallel SLURM tasks. See Phase 4b and
`XX_handover_for_Sophie.md` Section 5b for full commands.

Note: Phase 4a (mean-pool pilot) used Full_T, which is acceptable because the temporal
axis was collapsed anyway — the goal was only to confirm a signal exists.

---

## Phase 4b — Temporal evaluation (whisper-base × ds004408) ✅ Done (2026-06-07)

**Scripts:**
- Feature extraction: `src/mbs/extraction/extract_features_delta_t.py` (Delta_T causal extractor)
- Evaluation: `src/mbs/evaluation/evaluate_features_temporal.py`
- SLURM: `scripts/slurm_extract_delta_t.sh` (whisper-base) / `scripts/slurm_extract_delta_t_generic.sh` (generic)

Uses full-resolution Delta_T features `[n_stimuli, T, d_model]` and EEG `[n_stimuli, T, n_channels]`
aligned to the model's time grid (20ms/step for Whisper). Fits one Ridge per time step:

```
for t in range(T_model):
    X = features[:, t, :]          # [n_stimuli, d_model]
    y = eeg[:, t, :]               # [n_stimuli, n_channels]
    score[t, :] = RidgeCV(X, y)    # Pearson-r per channel, noise-ceiling corrected
```

**Status (2026-06-04):**
- [x] `extract_features_delta_t.py` written, tested, bugs fixed (see handover doc Section 6b)
- [x] SLURM pilot (job 54867238, 3 stim × 1500 bins) — completed ~70 min
- [x] Full SLURM run (job 54867710, 20 tasks × 16 stim) — completed; 312/314 stimuli (2 lost to flush bug, fixed)
- [x] Merged features at `outputs/features/whisper-base-delta-t/merged/` (39 files)
- [x] EEG HDF5 expanded to **67 ROIs** (62 single electrodes + 5 clusters + whole_brain)
- [x] Temporal evaluation — job 54930384, ~31.5 h wall time, → `outputs/results/whisper-base-delta-t-full/`

**Results (ds004408, naturalistic speech, NC-corrected Pearson r, mean over 1500 time bins):**

| Electrode | NC (raw) | blocks.0 | blocks.1 | blocks.2 (best) | blocks.3 | blocks.4 | blocks.5 |
|---|---|---|---|---|---|---|---|
| Fz | 50.4% | 0.035 | 0.043 | **0.110** | 0.048 | 0.047 | 0.039 |
| T7 | 75.3% | -0.003 | 0.001 | **0.050** | 0.017 | 0.010 | -0.006 |
| FT7 | 91.1% | 0.005 | 0.008 | **0.060** | 0.021 | 0.013 | 0.003 |
| AF3 | 94.6% | 0.003 | 0.010 | **0.069** | 0.025 | 0.016 | 0.003 |

**Key finding: blocks.2 is the best layer** across all high-NC speech electrodes. This differs from
Phase 4a (mean-pool: blocks.4 best at Fz) — the temporal evaluation reveals an earlier, more
sensory layer drives the continuous EEG envelope response.

**How to read these numbers — what the evaluator actually does:**
For each (layer, electrode) pair, the evaluator fits **T=1500 independent Ridge regressions**,
one per time step: `X_train[:, t, :] [n_stim, d_model]` → `y_train[:, t] [n_stim]`. Each
regression uses all training stimuli at a single moment in time (rows = stimuli, columns =
model features at that time step). The result is a prediction score curve `scores[T]` — one
Pearson r per time bin, NC-corrected.

The numbers in the table are `mean(scores[T])`, a scalar summary that collapses the temporal
structure. They are useful for ranking layers but discard the scientifically interesting
content. **The primary output is the full `scores[T]` curve**, which shows which time lags
a given layer can predict. Visualization should plot these curves per layer at Fz, T7, FT7, AF3.

There is no "committed layer per time step" — each layer produces its own independent curve.
The time × layer interaction (does blocks.2 predict early lags, blocks.4 late lags?) is the
key question for the MMN dataset.

**Why Phase 4a and 4b give different best layers:**
Phase 4a collapsed 30s → one vector; blocks.4 (higher-level) won because it captures
semantic/phonemic content that varies across whole segments. Phase 4b fits at 20ms resolution;
blocks.2 (earlier) wins on average because the dominant continuous-EEG signal is the auditory
envelope-following response, a low-level acoustic feature. Both are correct for their question.

**Statistical validation — are other layers genuinely above zero or just noise?**

A one-sample t-test of scores[T] against 0 (script: `scripts/plot_score_distributions.py`,
figure: `outputs/figures/whisper_base_score_distributions.png`) reveals three tiers:

| Tier | Layers | Mean range | Evidence |
|---|---|---|---|
| Unambiguous signal | blocks.2 | 0.037–0.110 | t = 10–21, 62–70% of bins > 0, consistent across all electrodes |
| Weak but real | blocks.1, 3, 4 | 0.009–0.048 | t = 2–10, 52–58% bins > 0, significant at most electrodes |
| Noise | blocks.0, blocks.5 | −0.007–0.015 | t < 2, ~50% bins > 0, not significant at AF3/FT7/T7/TP7/Fpz |

blocks.2 mean is **2–5× larger** than the next best layer. Fz is an exception — all 6 layers
are significant there because Fz's 50% NC and frontal position picks up a broader mixture of
processing stages than the more lateralized auditory electrodes.

**Autocorrelation correction (option 1 implemented):** The t-test treats T=1500 time bins as
independent, but ρ₁ ≈ 0.71–0.82 across all layers/electrodes. The implemented correction uses
the AR(1) effective sample size `n_eff = T × (1−ρ₁) / (1+ρ₁)` → n_eff ≈ 160–460. The
corrected t-statistic is `t_corr = mean / (std / √n_eff)` with `df = n_eff − 1`. After
correction the tier structure sharpens:

| Tier | Layers | Corrected result |
|---|---|---|
| Robust signal | blocks.2 | *** at all electrodes (d=0.26–0.54, t_corr=4.6–10.3) |
| Marginal | blocks.3 | * or ** at AF3/P9/Pz/Fz only; ns elsewhere |
| Noise | blocks.0, 1, 4, 5 | ns at all speech electrodes (Fz excepted — all layers sig there) |

**Three approaches to correct autocorrelation-inflated p-values** (option 1 implemented):

*Option 1 — n_eff from lag-1 autocorrelation (AR(1)) — implemented in `plot_score_distributions.py`:*
```
ρ₁ = lag-1 autocorrelation of scores[T]
n_eff = T × (1 − ρ₁) / (1 + ρ₁)
t_corr = mean / (std / √n_eff),   df = n_eff − 1
```
Assumes exponential ACF decay. Fast, 3 lines. Will over-correct if autocorrelation decays
slower than AR(1) (common in EEG signals with long-range structure).

*Option 2 — n_eff from full ACF (more accurate, not yet implemented):*
```
n_eff = T / (1 + 2 × Σₖ ρₖ)   summed until ρₖ ≈ 0
```
Plug n_eff into the same corrected t formula. Better for slow-decaying ACF. Still just numpy
ACF + one substitution; ~5 extra lines over option 1.

*Option 3 — circular shift permutation test (non-parametric, not yet implemented):*
Randomly circular-shift scores[T] N=1000–5000 times. Each shift preserves the exact
autocorrelation structure but destroys any consistent offset from zero. p = fraction of
null means ≥ observed mean. No distributional assumptions. Ideal for paper reporting; ~10
lines of code.

**Visualization scripts** (both runnable from repo root with venv active):
```bash
python scripts/plot_temporal_scores.py      # time-course curves (smoothed + raw overlay)
python scripts/plot_score_distributions.py  # violin plots of scores[T] vs. zero
```
Figures: `outputs/figures/whisper_base_temporal_scores.png`,
`outputs/figures/whisper_base_score_distributions.png`,
`outputs/figures/score_distribution_summary.json`

**ROI filtering:** 28 of 61 single electrodes have raw NC stored as exactly 0.0% in the HDF5.
Exclude these by filtering `raw_NC > 0`; the 33 remaining electrodes have genuine signal (0.78%–94.6%).
NC-correction divides by ≈1e-3 for the zero-NC electrodes, inflating scores to nonsensical values
(e.g. C2 mean ≈ 49). See Open Question 4 and `XX_handover_for_Sophie.md` for the full electrode list.

**ROI set:** 67 ROIs covering the full extended 10-20 electrode layout, filtered to ≤25 mm from
nearest BioSemi128 channel.

---

## Phase 4c — Dataset validation (next priority, 2026-06-07)

Before scaling to more models, establish that the whisper-base results generalise beyond
Broderick 2018. Two questions:

**Question 1 — Replication:** Does a different naturalistic speech EEG dataset yield similar
goodness-of-fit scores (same best layer, similar NC-corrected r at speech electrodes)?
- If yes: pipeline is robust, Broderick is a representative training set.
- If no: results are dataset-specific; need to understand why before scaling.

**Question 2 — Combining datasets:** Does pooling multiple datasets as training data increase
scores, or are returns quickly diminishing?
- If increasing: invest in collecting / pooling more data before scaling to other models.
- If diminishing: Broderick alone is sufficient; scaling data is not the bottleneck.

**Tasks:**
- [ ] Identify a suitable second naturalistic speech EEG dataset (discuss with Gokce/Sophie)
- [ ] Run Phase 3 formatter on the new dataset (same schema, same 50 Hz grid)
- [ ] Evaluate whisper-base blocks.2 on new dataset; compare scores to Broderick results in
      `outputs/figures/`
- [ ] Train on Broderick, test on new dataset (cross-dataset generalisation)
- [ ] Train on combined datasets, test on held-out set; compare to Broderick-only

---

## Phase 5 — Scale to other models (revised priority order, 2026-06-07)

**VGGish excluded** from temporal evaluation: T=10 bins at 1s resolution is too coarse to
meaningfully predict EEG at 20ms resolution. May still be run for mean-pool comparison only
as a baseline, but not a priority.

**Order of implementation:**
1. **Whisper variants (tiny, small, medium, large)** — same loader already implemented, only
   configs and SLURM scripts needed. Enables cross-model-size comparison within the Whisper
   family before any new code.
2. **wav2vec2-base / wav2vec2-large** — more involved: different input format (raw waveform,
   10s window), different time grid (499 bins not 1500), separate EEG HDF5 needed. Implement
   after Whisper variants.
3. **AST** — mean-pool only (2D patch tokens, not temporal). Lower priority.

Repeat Phases 4a + 4b for all models in Sophie's pipeline (excluding VGGish temporal).
T and U_total values below are confirmed from Sophie's `split_info.json` files
(see `aux/00_schizophrenia_pipeline_Sophie_2026.md` for the verification command).

### Model specs (verified)

| Model | Input | Duration | T (time bins) | hop (ms) | U_total | Temporal eval? |
|---|---|---|---|---|---|---|
| whisper-tiny | 80-bin log-mel | 30 s | 1 500 | 20.0 | 28 800 | Yes |
| whisper-base | 80-bin log-mel | 30 s | 1 500 | 20.0 | 56 832 | Yes |
| whisper-small | 80-bin log-mel | 30 s | 1 500 | 20.0 | 168 192 | Yes |
| whisper-medium | 80-bin log-mel | 30 s | 1 500 | 20.0 | 445 440 | Yes |
| whisper-large | 128-bin log-mel | 30 s | 1 500 | 20.0 | 741 120 | Yes |
| wav2vec2-base | raw waveform (normalised) | 10 s | 499 | ≈20.0 | 229 888 | Yes |
| wav2vec2-large | raw waveform (normalised) | 10 s | 499 | ≈20.0 | 600 576 | Yes |
| vggish | 0.96 s window chunks [N×1×96×64] | 10 s | 10 | 1 000.0 | 4 928 | Mean-pool only |
| ast | 128-bin log-mel → 2D patch tokens | 10 s | 1 214 | ≈8.2 | 241 920 | Mean-pool only |

Notes:
- **wav2vec2:** T=499 not ~1500 — CNN stride 320 samples = 20 ms; 160 000 samples / 320 = 500 frames, processor trims one.
- **AST:** T=1214 is the transformer token axis (2D patches flattened), not a 1D temporal axis. hop_ms≈8.2 is a ratio only.
- **VGGish:** CPU only (torchvggish has no GPU support). T=10 bins = one per 0.96 s window.
- **whisper-large:** n_mels=128 (not 80); `WhisperTransform` already reads `model.dims.n_mels` so this is handled.

### Preprocessing per model (for `audio_models.py`)

**Whisper (all sizes):** already implemented in Phase 1.
- Library: `openai-whisper` (not HuggingFace)
- `whisper.log_mel_spectrogram(waveform, n_mels=model.dims.n_mels)` → `[n_mels, 3000]`
- `whisper.pad_or_trim(waveform)` to exactly 480 000 samples (30 s)
- Encoder only (`model.encoder`); layer paths: `backbone.blocks.{i}`

**wav2vec2-base / wav2vec2-large:**
- Library: HuggingFace `transformers`
- `Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")` (or `-large-960h`)
- Processor normalises raw waveform (mean 0, unit variance) → `input_values` tensor; no spectrogram
- Duration: exactly **10 s** (160 000 samples at 16 kHz)
- Hook full model (CNN + transformer); layer paths TBD from probe pass
- Add to `load_model_audio()` under `elif model_id.startswith("wav2vec2-")`

**VGGish:**
- Library: `torchvggish` (torchhub)
- `torchvggish.waveform_to_examples(audio_np, 16000)` → `[N_windows, 1, 96, 64]`
- Duration: exactly **10 s** → N_windows=10
- **Force CPU** (torchvggish does not support CUDA)
- Add to `load_model_audio()` under `elif model_id == "vggish"`

**AST:**
- Library: HuggingFace `transformers`
- `ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")`
- 128 mel bins, 25 ms window, 10 ms hop, AudioSet normalisation (mean=−4.268, std=4.569)
- Duration: exactly **10 s** (padded/truncated to 1024 frames ≈ 10.24 s at 100 fps)
- Add to `load_model_audio()` under `elif model_id == "ast"`

### Implementation tasks

- [ ] `audio_models.py` — add `load_wav2vec2()`, `load_vggish()`, `load_ast()` loaders
- [ ] Add `torchvggish`, `torchaudio` to `pyproject.toml` `audio` extra if not already present
- [ ] Write slow tests for each new model in `tests/test_backbone_audio.py` (follow Phase 1 pattern)
- [ ] Run Phases 4a + 4b for each model; collect layer sweep results
- [ ] Add `configs/evaluation/layer_commitment/layer_commitments_audio.json` with committed layers

For AST and VGGish: mean-pool evaluation (Phase 4a) only.
For Whisper and wav2vec2: run both Phase 4a and Phase 4b (temporal).

---

## Phase 6 — MMN electrode-level analysis via trained Ridge mapping (2026-06-07)

The trained Ridge mapping (Broderick 2018, Phase 4b) lets us generate *in-silico* electrode
responses to any audio without collecting new EEG. This is Sophie's most direct next step and
does not require patient data.

**6a — In-silico MMN (no new data needed):**
- [ ] Feed Sophie's exact MMN stimuli (standard + deviant tone sequences) into whisper-base
      to extract Delta_T features at blocks.2 (best predicting layer)
- [ ] Apply the trained Ridge weights (from Phase 4b) to those features → get predicted
      electrode-level EEG time courses for each MMN stimulus
- [ ] Run Sophie's existing MMN analysis on those predicted electrode responses:
  - deviant minus standard difference wave at Fz
  - Does the model show an MMN-like component at 100–200ms?
  - Is the effect present at the electrode and latency expected from real MMN data?
- [ ] Compare the in-silico MMN across model architectures once Phase 5 results are available:
  - Does a model with "schizophrenia-like" properties show a reduced/absent in-silico MMN?

This is the scientifically novel contribution: using a brain-aligned encoding model to
*predict* what a patient's EEG should look like for a paradigm the model was never trained on.

**6b — Real patient MMN data (requires data access, later priority):**

Once an MMN EEG dataset is available (patient + control):

- [ ] Run Phase 3 formatter on MMN data:
  - Stimuli = individual tone sequences (standard and deviant), ~200ms each
  - Epoch = 0–500ms post-stimulus onset (25 time steps at 20ms/step)
  - Average across repetitions per condition before storing
  - ROIs: Fz and FCz as single-electrode ROIs (primary MMN channels) + clusters
- [ ] Compare real patient MMN profiles against in-silico MMN predictions from Phase 6a:
  - Does the model whose in-silico MMN best matches controls also best match healthy brains?
  - Does the model whose in-silico MMN is reduced/delayed match schizophrenia profiles?
- [ ] Cross-reference best-predicting layer with Sophie's MMN unit selection:
  - Do units selected by Sophie's pipeline come disproportionately from the
    best-predicting layer found in Phase 4b?

---

## Open questions (discuss with Gokce)

1. **Stimulus count — underdetermined regression:** With 30s windows at 10s stride we get ~256
   training stimuli vs. d=512 features for whisper-base. RidgeCV handles this via regularisation,
   but it is worth discussing with Kadir whether a 5s stride (~496 stimuli) or shorter windows
   (10s, which would also allow wav2vec2 compatibility) would be better. Using shorter 10s windows
   instead of 30s windows would give ~620 stimuli AND match the wav2vec2 input duration — a single
   HDF5 file could then serve both models.

2. **Noise ceiling source:** ds004408 has single-trial continuous recordings (no repetitions per
   stimulus). Split-half across the 19 subjects gives a cross-subject NC. This is conceptually
   different from Kadir's THINGS-EEG (many repetitions per image, within-subject NC). Cross-subject
   NC is a lower bound — discuss with Kadir whether this affects score interpretation.

3. **EEG sampling rate and downsampling (confirmed):** ds004408 is at 512 Hz. MNE's `raw.resample()`
   applies a zero-phase FIR anti-alias filter before downsampling — correct. The 10× reduction
   (512 → 50 Hz) is well within MNE's standard usage.

4. **28 zero-NC electrodes — resolved (2026-06-07):** After the 67-ROI expansion, 28 of 61 single
   electrodes have raw NC stored as exactly 0.0% in `broderick2018_30s.h5`. These fall into two
   groups:

   - **Expected zeros (genuine physiology):** FCz, Cz, C1/C2, CP* — motor/central strip does not
     respond consistently across subjects during passive audiobook listening. Confirmed by
     `scripts/diagnose_roi_mapping.py` (2026-06-04): FCz → BioSemi C23, correct mapping, 9.8 mm
     distance. The 0% NC reflects the paradigm, not a bug.
   - **Unexpected zeros:** O1, P1–P3, P5, F1/F2, F5/F6, FC1/FC2, FC5/FC6, FT8, PO3, AF8, C4, C6.
     These are not motor-strip and should have some signal. Likely cause: `format_eeg_hdf5.py`
     channel mapping gaps — some standard 10-20 names were not found in the BioSemi128 montage
     and ended up with zero-filled data, producing NC = 0.

   **Resolution: filter `raw_NC > 0` in all analyses**, leaving 33 electrodes with genuine signal
   (NC range 0.78%–94.6%). No reprocessing needed. The full list and NC values are in
   `XX_handover_for_Sophie.md`. FCz is still the primary electrode for Phase 6 (MMN), where the
   time-locked oddball response will produce high cross-subject agreement.

5. **Which MMN EEG dataset to use:** ERP-CORE (open, N=40, has MMN paradigm) vs. lab-internal
   patient data. ERP-CORE is controls-only; for the schizophrenia comparison we need patient data.

6. **Sophie's pipeline compatibility:** should this encoding model replace or augment Sophie's unit
   selection? I.e., select units that are (a) MMN-like AND (b) predictive of EEG at the MMN latency?
