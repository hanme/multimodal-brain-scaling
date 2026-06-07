# Handover: Auditory EEG Encoding Models — Sophie

**Status: living document, updated as plan executes.**
Last updated: 2026-06-07
See `02_project_plan_make_compatible_for_auditory_EEG.md` for the full technical roadmap.
See `00_schizophrenia_pipeline_Sophie_2026.md` for the MMN unit-selection pipeline this feeds into.

---

## What this is

This repo is the computational backbone for turning audio models (Whisper, wav2vec2, AST, VGGish)
into **EEG encoding models**: linear mappings from a model layer's activations to recorded EEG
electrode signals. Once trained, such a model can tell us:

- Which layer of which audio model best predicts human EEG responses to sound
- Whether that model-to-brain alignment differs between healthy controls and schizophrenia patients
- Which layer Sophie's MMN unit selection should target, grounded in actual neural data

---

## 1. The original repo (Kadir Gokce / epflneuroailab)

**Repository:** `epflneuroailab/multimodal-brain-scaling` (we work on fork `hanme/multimodal-brain-scaling`)

Kadir's paper studies how well *visual* AI models predict *visual* brain responses
(fMRI, EEG, MEG) across 600+ models and 8 neural benchmarks. His key finding:
scaling model size and training data improves brain alignment, and intermediate-to-deep
layers (normalized depth ~0.73 for EEG) are generally the best predictors.

### Source tree

```
src/mbs/
├── extraction/                # Feature extraction from models
│   ├── extract_features.py    # CLI entry point: mbs-extract-features
│   ├── data/
│   │   ├── dataloaders.py     # create_dataloader() factory
│   │   └── datasets.py        # THINGSDataset, H5Dataset, BrainScoreDataset
│   └── modeling/
│       ├── backbones/
│       │   ├── __init__.py    # create_backbone() registry: timm | spvvs | hf
│       │   ├── timm_models.py # load vision models from timm
│       │   ├── hf_models.py   # load VLMs from HuggingFace (Qwen, V-JEPA2, ...)
│       │   └── scaling_models.py  # load SPVVS-trained vision models
│       ├── encoder_feature_extractor.py  # torch.fx-based extractor (timm/spvvs)
│       └── encoder_hooks.py   # Hook-based extractor (used for HF models)
│                              # -> HookedEncoder + HookFeatureExtractor
├── evaluation/                # Ridge regression scoring
│   ├── evaluate_features_all_layers.py      # mbs-evaluate-all-layers
│   ├── evaluate_features_committed_layers.py # mbs-evaluate-committed-layers
│   └── utils/evaluation_helpers.py
│       # load_neural_data()   reads HDF5 neural benchmark
│       # load_layer_features() reads HDF5 feature files
│       # get_pipeline()       returns RidgeCV sklearn pipeline
│       # compute_metrics()    Pearson-r, noise-ceiling correction
├── analysis/                  # Scaling curve fitting
└── training/                  # Model fine-tuning (not needed here)
```

### End-to-end workflow (original, visual)

```
mbs-extract-features            ->  per-layer HDF5 feature files  (n_stimuli x d_model)
mbs-evaluate-all-layers         ->  Pearson-r score per layer per ROI  (layer search)
mbs-evaluate-committed-layers   ->  final scores using the best layer per ROI
mbs-fit-curves                  ->  fit scaling laws across model families
```

---

## 2. What's already there and directly reusable

### 2a. Hook-based feature extractor (`encoder_hooks.py`)

This is the most important piece of infrastructure. `HookedEncoder` attaches
`torch.nn.Module.register_forward_hook` callbacks to *any* named submodule in *any*
PyTorch model. You specify layers as dotted paths (e.g. `"encoder.blocks.4"`) and
get back a dict of captured activations — no model surgery required.

```python
# How it works (simplified)
encoder = HookedEncoder(
    backbone=my_model,
    feat_layers={"encoder.blocks.4": "block_4", "encoder.blocks.8": "block_8"},
)
feats = encoder(inputs)   # feats["block_4"].shape == [batch, T, d_model]
```

**This works for Whisper, wav2vec2, AST, and VGGish without modification** — all we
need is the correct dotted path to each transformer block or conv layer. The hook
infrastructure is model-agnostic.

### 2b. Ridge regression evaluation

`evaluation_helpers.py` contains the full scoring pipeline:

- `load_neural_data(path, subject, roi, split)` — reads the HDF5 neural benchmark,
  returns `(stimulus_ids, neural_data, noise_ceiling)`
- `get_pipeline()` — returns a `sklearn.Pipeline([('regressor', RidgeCV(alphas=...))])`
  with a wide alpha grid (0.01 to 10^7)
- `compute_metrics()` — Pearson-r raw and noise-ceiling-corrected

**Two evaluation modes:**

| Mode | Features shape | EEG shape | Output | Code change |
|---|---|---|---|---|
| Mean-pool (Phase 4a) | `[n_stimuli, d_model]` | `[n_stimuli, n_ch]` | 1 score per ROI | None — existing code |
| Temporal (Phase 4b) | `[n_stimuli, T, d_model]` | `[n_stimuli, T, n_ch]` | `score[T, n_ch]` | New evaluator script |

In the temporal mode, a Ridge is fit independently for each time step t:
`X[:, t, :] → y[:, t, :]`. This produces a prediction score time series at each electrode —
analogous to an ERP but measuring how well the model predicts the EEG at each latency.
For the MMN, the key question is whether the score peaks at ~100–200ms at Fz.

### 2c. HDF5 feature format (unchanged)

Feature files written by `mbs-extract-features` look like:

```
feats_30000-bs_32-batch_0-seed_42.h5
├── features/
│   └── {layer_name}     [n_stimuli_in_batch, d_model]  float16
├── ids                  [n_stimuli_in_batch]  str
└── attrs: model_id, backbone_source, target_feature_layers, config_json
```

### 2d. HDF5 neural data format (need to populate, not modify reader)

`load_neural_data()` expects:

```
neural_benchmark.h5
├── attrs: subjects, rois, splits, max_nc
├── train/
│   ├── stimulus_ids              [n_train_stimuli]
│   └── neural_data/{subj}/{roi}  [n_stimuli, n_channels]
├── test/  (same structure)
└── noise_ceilings/{subj}/{roi}   [n_channels]
```

We need to *create* this file from the raw EEG dataset (Phase 3).
The reader code is untouched.

---

## 2b. ds004408 dataset facts (verified 2026-06-01)

The validation dataset is **Broderick 2018 / Di Liberto 2015**, OpenNeuro ds004408.

**Local path:** `/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea`

| Property | Value |
|---|---|
| Subjects | 19 (sub-001 … sub-019) |
| Runs per subject | 20 (one per audio segment) |
| Audio stimuli | 20 WAV files, stereo 44100 Hz, **177–202 s each (~3 min)** |
| EEG format | BrainVision, Brain Products amplifier |
| EEG rate | 512 Hz, 128 channels |
| Channel names | A1–A32, B1–B32, C1–C32, D1–D32 (BioSemi Active2 layout) |
| Event markers | **None** — each run IS the continuous EEG for one audio segment |
| Alignment | README confirms: "starts are aligned, EEG longer to a varying extent" |

**Key architectural consequence:** audio files are ~3 min, far longer than Whisper's 30s context.
The formatter sub-segments each run into 30s windows at a configurable stride (default 10s).
With 10s stride this gives ~16 windows/run × 20 runs = **~320 stimuli** total (256 train / 64 test
after holding out 4 runs). This is below whisper-base's feature dimension (d=512), which is a mild
under-determination; RidgeCV handles it via regularisation, but discusses with Kadir.

**No event markers** means there is no onset timing to extract — alignment is purely continuous
(EEG sample 0 = audio sample 0). The formatter trims EEG to the audio duration and sub-segments
both together.

---

## 3. What we added / changed

### 3a. `pyproject.toml` — removed `scaling-primate-vvs`, added `audio` extra

`scaling-primate-vvs` (Kadir's SPVVS checkpoint loader) was removed from the `training`
and `evaluation` extras — it is only needed for loading SPVVS vision checkpoints,
irrelevant here, and caused a 2.5+ hour hang during `uv sync`. The `audio` extra adds
the packages needed for audio model loading and EEG preprocessing:

```toml
audio = [
    "openai-whisper",   # Whisper model + mel spectrogram utilities
    "mne",              # EEG preprocessing (epoching, filtering)
    "soundfile",        # .wav I/O
    "librosa",          # audio resampling
]
```

### 3b. `src/mbs/extraction/modeling/backbones/audio_models.py` — audio backbone loader
*(Phase 1, status: **Done**)*

Registers audio models in the existing `create_backbone()` factory under
`backbone_source = "audio"`. Implements:

- `load_model_audio(model_id, **kwargs)` — dispatches to per-model loaders
- `load_whisper(model_id)` — loads Whisper encoder via `openai-whisper`, returns
  `(model.encoder, AudioPreprocessor)`
- `AudioPreprocessor` — callable: `.wav` path -> 16 kHz mel spectrogram tensor
  (matches `whisper.log_mel_spectrogram()`)

The layer names for `HookedEncoder` for each Whisper size:

| Layer path | What it is |
|---|---|
| `backbone.backbone.blocks.0` | encoder transformer block 0 |
| `backbone.backbone.blocks.N` | encoder transformer block N |

For Whisper-base, N goes from 0 to 5 (6 blocks). Output shape per block:
`[batch, T=1500, d_model=512]`.

### 3c. `src/mbs/extraction/data/datasets_audio.py` — audio stimulus dataset
*(Phase 2, status: **Done**)*

Implements `AudioSegmentDataset`: given a folder of `.wav` files, returns
`(waveform_tensor, stimulus_id)` pairs. Handles resampling to 16 kHz,
mono conversion, and padding/truncation to 30s (Whisper) or model-specific length.

Adds `--dataset_type audio` to the `mbs-extract-features` CLI. By default, features are
stored at **full temporal resolution**: `[n_stimuli, T, d_model]` where T = number of
model time steps (e.g. 1500 for Whisper at 20ms/step). An optional `--mean_pool_time`
flag collapses to `[n_stimuli, d_model]` for a quick compatibility pilot.

### 3d. `src/mbs/data_prep/format_eeg_hdf5.py` — BIDS EEG → mbs HDF5
*(Phase 3, status: **Written and executed** — output at `outputs/neural_data/broderick2018_30s.h5`)*

Converts ds004408 (or any similarly structured BIDS EEG dataset) into the `neural_benchmark.h5`
format. Written, unit-tested, and run on the full Broderick 2018 dataset.

Key design decisions informed by the actual dataset:
- **No event markers** in ds004408 — alignment is purely continuous (sample 0 of EEG = sample 0
  of audio, per README). No onset-extraction step needed.
- **Sub-segmentation:** each ~3-min audio/EEG run is split into 30s windows (default 10s stride)
  to produce independent "stimuli" for the regression. Stimulus IDs match `AudioSegmentDataset`
  convention exactly: `audioXX_SSSSSSS` where SSSSSSS is the start sample at 16 kHz.
- **Cross-subject average:** 19 subjects are averaged into a single "group" response. Split-half
  across subjects gives the noise ceiling (Spearman-Brown corrected).
- Downsampling EEG from 512 Hz to **model's time grid** using MNE's anti-aliased resample:
  - Whisper / wav2vec2: **50 Hz** (20 ms/step)
  - VGGish: **1 Hz** (1000 ms/step)
  - AST: no per-timepoint alignment (2D patch tokens; mean-pool only)
- **ROI discovery** uses MNE's `biosemi128` montage to map standard 10-20 names (Fz, FCz, Cz,
  etc.) to the nearest BioSemi channel index. Also outputs `whole_brain` (all 128 ch).
- Noise ceiling stored as **% variance explained** (r² × 100), `max_nc=100.0`, so
  `load_neural_data()` recovers Pearson r via `sqrt(nc_stored / 100)`.

#### Setup — install `mne` before running

`mne` is part of the `audio` extra in `pyproject.toml`. Install it once:

```bash
# From the repo root, with the venv active:
uv sync --extra audio
# or, if uv sync is not available:
pip install mne
```

#### To run (Whisper-compatible, 50 Hz, 30s windows with 10s stride)

```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh   # or however the venv is activated on the compute node

python -m mbs.data_prep.format_eeg_hdf5 \
  --bids_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea \
  --output_path outputs/neural_data/broderick2018_30s.h5 \
  --window_duration 30.0 \
  --window_stride   10.0 \
  --target_sr       50 \
  --n_test_runs     4 \
  --seed            42
```

| Argument | Meaning | Change for other models |
|---|---|---|
| `--target_sr 50` | Downsample EEG to 50 Hz (Whisper / wav2vec2 grid) | Use `1` for VGGish; omit for AST |
| `--window_duration 30.0` | 30 s windows (Whisper context length) | Keep 30s for wav2vec2/AST too |
| `--window_stride 10.0` | 10 s stride → ~16 windows/run × 20 runs = ~320 stimuli | Can reduce to 5s for more stimuli |
| `--n_test_runs 4` | Hold out 4 runs (~20%) as test set | Keep consistent across models |
| `--output_path` | One HDF5 per `target_sr` (50 Hz and 1 Hz need separate files) | Change filename accordingly |

For VGGish (1 Hz grid):
```bash
python -m mbs.data_prep.format_eeg_hdf5 \
  --bids_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea \
  --output_path outputs/neural_data/broderick2018_30s_1hz.h5 \
  --window_duration 30.0 \
  --window_stride   10.0 \
  --target_sr       1 \
  --n_test_runs     4 \
  --seed            42
```

The formatter prints tqdm progress bars per subject and per ROI. Full run on 19 subjects ×
20 runs took **~3 min** on the compute node (dominated by MNE BrainVision I/O).

#### Output: verified run (2026-06-01)

```
Subjects: 19 | runs: 20 | train: 16 | test: 4
Window: 30.0s / stride: 10.0s | target_sr: 50 Hz
Test runs: [2, 9, 13, 14]
Channels: 128 | ROIs: 14
Stimuli: train=252, test=62
Written: outputs/neural_data/broderick2018_30s.h5
```

**Noise ceiling summary** (r_SB² × 100, averaged across 1500 time bins):

| ROI | NC mean | Note |
|---|---|---|
| T7 | 75.3% | Temporal cortex, strong auditory envelope-following |
| T8 | 73.3% | Temporal cortex |
| Pz | 74.4% | Parietal, strong auditory |
| temporal_cluster | 74.3% | T7+T8 combined |
| Fz | 50.4% | **Key MMN electrode — very good** |
| frontal_cluster | 23.5% | Fz+F3+F4+FCz combined |
| F3 | 23.2% | Frontal |
| F4 | 20.4% | Frontal |
| whole_brain | 26.5% | All 128 channels, averaged |
| Cz | 2.6% | Central |
| central_cluster | 1.1% | Cz+C3+C4 |
| C3 | 0.8% | Motor cortex — low auditory response expected |
| C4 | 0.0% | Motor cortex |
| FCz | 0.0% | Fronto-central vertex — unexpectedly low; see note |

**Note on near-zero FCz/Cz/C3/C4 NC:** These NC values are averaged across *all 1500 time bins*.
For continuous naturalistic speech (audiobook), auditory cortex drives strong envelope-following
at temporal and frontal electrodes, but motor/central regions have much weaker sustained responses.
The near-zero NC is not an error — it reflects genuine cross-subject variability at those sites.
FCz being 0% is somewhat unexpected (it is a classical MMN electrode); this may indicate that
the BioSemi128 channel nearest to the standard FCz 10-20 position is not well-matched in this dataset,
or that the continuous-speech paradigm is simply weak there relative to an oddball MMN paradigm.
**Recommend inspecting FCz channel in the raw data before drawing conclusions about that ROI.**

### 3e. `src/mbs/evaluation/evaluate_features_temporal.py` — temporal evaluator
*(Phase 4b, status: **Done** — CLI + full implementation)*

New CLI `mbs-evaluate-temporal`. For each (layer, electrode) pair it fits **T separate Ridge
regressions**, one per time step, and stores the resulting prediction score curve `scores[T]`.

**Mechanics in detail:** for a given layer L and electrode E:

```
for t in 0..T-1:
    X_train = feat_train[:, t, :]   # [n_stim=250, d_model]  — model activations at time t
    y_train = eeg_train[:, t, :]    # [n_stim=250, n_ch=1]   — EEG amplitude at time t
    ridge = RidgeCV(alphas=...).fit(X_train, y_train)

    X_test = feat_test[:, t, :]     # [n_stim=62,  d_model]
    y_pred = ridge.predict(X_test)
    scores[t] = pearsonr(eeg_test[:, t, :], y_pred) / nc[t]  # NC-corrected
```

Each of the 1500 time steps is treated as an independent regression problem: take all stimuli
at that moment in time (rows = stimuli, columns = model features), map to EEG, test on held-out
stimuli. There is **no single committed layer** — each layer yields its own `scores[T]` curve.

**What `mean_score` and `peak_score` in the summary JSON are:**
- `mean_score = nanmean(scores[0:T])` — average prediction over all time bins. Used as a
  scalar summary to rank layers (blocks.2 best for ds004408). This collapses the temporal
  structure and should be treated as a rough ranking tool, not the primary output.
- `peak_score = nanmax(scores[0:T])` — best prediction at any single time bin.

**The scientifically interesting output is the full `scores[T]` curve**, not the mean.
Different time lags may be dominated by different layers — e.g. early lags (0–100ms) may favour
sensory layers (blocks.0–2) while later lags (100–300ms) may favour higher-level layers
(blocks.3–5). This time × layer interaction is the key question for the MMN dataset: does the
best-predicting layer at the MMN latency (~100–200ms at Fz) match the layer Sophie uses for
unit selection?

**Why Phase 4a (mean-pool) and Phase 4b (temporal) give different best layers:**
- Phase 4a collapsed 30s of EEG to a single vector before regressing → blocks.4 won. A higher-level
  layer captures more of the semantic/phonemic content that varies across 30s segments.
- Phase 4b fits at each 20ms bin → blocks.2 wins on average. The dominant signal in continuous
  naturalistic EEG is the auditory cortex envelope-following response, which is a low-level
  acoustic feature better encoded in earlier layers.
Both results are correct — they answer different questions.

**Output stored in HDF5:** `scores[T, n_ch]` per (layer, subject, roi) in
`outputs/results/{model}-delta-t-full/temporal_scores.h5`. Key for the MMN: read `scores[:, 0]`
at key `blocks-2/group/Fz` and plot vs. time to see the prediction time course at the primary MMN electrode.

### 3f. `src/mbs/extraction/extract_features_delta_t.py` — Delta_T (causal) feature extractor
*(Phase 4b-pre, status: **Done** — 2026-06-02)*

New CLI `mbs-extract-features-delta-t`. Registered in `pyproject.toml`.

**Why Delta_T and not Full_T:** Whisper is a non-causal model — its representation at time bin t
has "seen" the full 30s audio including the future. Using Full_T features to predict EEG at time t
introduces an information asymmetry: the model knew what was coming, the brain did not.
For the **MMN EEG dataset** this is a fundamental confound — the whole point of MMN is that the
deviant is *surprising*. A Full_T model at the deviant position already encoded it from context,
so its representation cannot carry the prediction-error signal the brain shows. Delta_T is
therefore **not optional** for the MMN dataset. For ds004408 (naturalistic speech, no event
structure), it is also the principled choice, and keeps the feature type consistent with
Sophie's unit-selection pipeline which uses Delta_T throughout.

**What it does:** for each stimulus and each time step t:
1. Build a truncated mel spectrogram — keep frames `[0, 2*(t+1))`, fill the rest with
   the per-stimulus silence value (computed from Whisper's global mel normalization:
   `silence_val = mel_full.max() - 2.0`).
2. Run the Whisper encoder on a batch of `batch_t` such truncated spectrograms in one
   forward pass.
3. Collect output at position t from each item in the batch.

Output format is identical to `extract_features.py` temporal mode — `[n_stim, T_out, d_model]`
per layer — so `evaluate_features_temporal.py` works without modification.

**Compute cost (measured 2026-06-02, whisper-base, CPU):**

| Scenario | Time |
|---|---|
| 1 stimulus × 10 bins (t_stride=150, pilot) | ~6.5s |
| 1 stimulus × 1500 bins (t_stride=1, full) | **~13 min** (375 batches × ~2.1s each) |
| 16-stimulus SLURM task (1 CPU core) | **~3.5 h** |
| All 314 stimuli on 20 parallel SLURM tasks | **~3.5–5 h wall time** |

**Important:** tqdm only reports progress when a full stimulus finishes. At t_stride=1 this means
no output for ~13 minutes — the job is not hanging, it is working.

**Disk space (whisper-base, all 314 stimuli, t_stride=1):**
- Raw: 1500 bins × 512 d_model × float16 × 6 layers × 314 stimuli ≈ **2.8 GB uncompressed**
- Stored with gzip (opts=4): **~1–1.5 GB on disk**
- Each 16-stimulus chunk file: ~50–70 MB

GPU option: at batch_t=1500, all 1500 truncations fit in one forward pass per stimulus. A GPU
node would reduce the full run from ~4 h to under 30 min. See Section 5b for the SLURM script.

**Key parameters:**

| Arg | Default | Purpose |
|---|---|---|
| `--batch_t` | 16 | Forward passes per batch (increase on GPU, e.g. 512) |
| `--t_stride` | 1 | Sub-sample bins: `50` → 30 bins at 1s resolution (pilot) |
| `--n_stimuli` | 0 (all) | Limit stimulus count (pilot) |
| `--stim_start_idx` | 0 | First stimulus index (for parallelising across processes) |
| `--save_every` | 8 | Stimuli per output HDF5 file |

---

## 4. The extensible hook architecture

The central design principle: **we do not write model-specific feature extraction
code per model**. Instead, we use the general `HookedEncoder` that already exists
in `encoder_hooks.py`, and we only need to provide two things per new model:

1. **A loader** (`load_model_audio` dispatch) that returns the model + preprocessor
2. **Layer name strings** (dotted paths into the model's module tree)

Below is the mapping for Sophie's 9 models:

| Model | Loader | Layer path pattern | Output shape per layer |
|---|---|---|---|
| `whisper-tiny` | `openai-whisper` | `blocks.{i}` | `[T=1500, 384]` |
| `whisper-base` | `openai-whisper` | `blocks.{i}` | `[T=1500, 512]` |
| `whisper-small` | `openai-whisper` | `blocks.{i}` | `[T=1500, 768]` |
| `whisper-medium` | `openai-whisper` | `blocks.{i}` | `[T=1500, 1024]` |
| `whisper-large` | `openai-whisper` | `blocks.{i}` | `[T=1500, 1280]` |
| `wav2vec2-base` | HuggingFace `facebook/wav2vec2-base-960h` | `encoder.layers.{i}` | `[T~50/s, 768]` |
| `wav2vec2-large` | HuggingFace `facebook/wav2vec2-large-960h` | `encoder.layers.{i}` | `[T~50/s, 1024]` |
| `ast` | HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` | `audio_spectrogram_transformer.encoder.layer.{i}` | `[1214 tokens, 768]` |
| `vggish` | `torch.hub` `harritaylor/torchvggish` | `features.{i}` (conv layers) | `[T=10, 512]` |

After mean-pool (`activation.mean(dim=-2)`) all shapes reduce to `[d_model]` —
which is what the Ridge regression sees. Mean-pooling is model-agnostic.

**Note on layer path convention:** paths in the JSON configs are relative to the backbone module's internal structure — the factory prepends `backbone.` automatically. E.g. `blocks.0` in the JSON resolves to `WhisperBackboneWrapper.backbone.blocks[0]`. Do **not** include `backbone.` in the JSON.

### Adding a new model: checklist

1. Add a `load_<model>(model_id)` function in `audio_models.py` returning `(model, preprocessor)`
2. Add a dispatch branch in `load_model_audio()`
3. Add the model to the `MODEL_LOADERS` list in `backbones/__init__.py`
4. Specify the layer names in a target-layers JSON config (same format as existing visual models)
5. That's it — extraction, evaluation, and scoring all work without further changes

---

## 5. How to run the pipeline (Whisper-base x ds004408)

Once Phase 1–3 are complete, two modes are available:

### 5a. Mean-pool pilot (quick sanity check — not the main scientific output)

**What mean-pooling does here:** both the model features and the EEG neural data have their
temporal axis collapsed to a single vector before regression:

| | Before | After | Purpose |
|---|---|---|---|
| Model features | `[n_stim, T=1500, d_model]` | `[n_stim, d_model]` | `--mean_pool_time true` |
| EEG | `[n_stim, T=1500, n_ch]` | `[n_stim, n_ch]` | `collapse_temporal_hdf5.py` |
| Noise ceiling | `[T=1500, n_ch]` | `[n_ch]` | same script, mean over T |

This is a weak test (30s of brain response averaged into one vector). It is only used to confirm
that *any* predictive signal exists before investing in Phase 5b temporal evaluation.

```bash
# Step 1: extract mean-pooled features (already done 2026-06-01)
python -m mbs.extraction.extract_features \
  --model_id whisper-base \
  --backbone_source audio \
  --dataset_type audio \
  --data_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/stimuli \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --output_dir outputs/features/whisper-base-meanpool/ \
  --mean_pool_time true \
  --window_duration 30.0 --window_stride 10.0 \
  --max_feature_dim 0 --batch_size 8 --num_workers 0

# Step 2: create mean-pooled EEG HDF5 from the temporal one
python -m mbs.data_prep.collapse_temporal_hdf5 \
  --input_path  outputs/neural_data/broderick2018_30s.h5 \
  --output_path outputs/neural_data/broderick2018_30s_meanpool.h5

# Step 3: layer sweep (existing mbs-evaluate-all-layers, unchanged)
mbs-evaluate-all-layers \
  --model_id whisper-base \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --features_dir outputs/features/whisper-base-meanpool/ \
  --data_hdf5_path outputs/neural_data/broderick2018_30s_meanpool.h5 \
  --output_dir outputs/layer_search/whisper-base-meanpool/ \
  --exclude_whole_brain false
```

**Success criterion:** noise-ceiling-corrected Pearson r > 0.1 in at least one ROI.

**What the reported metrics mean:** the evaluator uses the train/test split baked into the HDF5
(runs [2, 9, 13, 14] held out as test, the rest as train). It fits Ridge on the train split,
then predicts on the held-out test split. The metrics in the output JSON therefore mean:

| Field | What it is |
|---|---|
| `pearsonr` | Pearson r between predicted and actual EEG on the **held-out test runs** |
| `pearsonr_nc` | Same, divided by the noise ceiling (values ≤ 1 mean below ceiling; > 1 can occur due to noise) |
| `cv_score` | 5-fold cross-validation score *within* the train split — used for stability monitoring only |

The `pearsonr` / `pearsonr_nc` in the results are honest held-out estimates, not CV scores.
The Fz result (pearsonr = 0.867, blocks.4) is from those 4 unseen runs.

### 5b. Temporal evaluation — Delta_T (main scientific output)

**Why Delta_T:** we use the causal `mbs-extract-features-delta-t` extractor (see Section 3f),
not `mbs-extract-features`. Delta_T is required because the EEG at time t was produced by a
brain that had only heard audio up to t — using Full_T features (which encode the full future)
would give an unfair information advantage and could bias layer selection.

For ds004408 (naturalistic speech), do not expect a sharp peak at 50–300 ms — that criterion
applies to the MMN EEG dataset (Phase 6). Here, look for (a) scores meaningfully above zero
across most time bins, and (b) a consistent layer ranking (blocks.4 best at Fz from Phase 4a).

#### Timing and disk space

| Scenario | Wall time |
|---|---|
| 3-stimulus SLURM pilot (t_stride=1) | ~40 min |
| 16-stimulus SLURM task (1 CPU core) | ~3.5 h |
| Full 314-stimulus run (20 SLURM tasks in parallel) | **~3.5–5 h** |
| Full run on GPU (batch_t=1500) | ~30 min |

Disk: ~**1–1.5 GB** on disk (gzip-compressed float16) for all 314 stimuli × 6 layers.

**Note:** tqdm only updates when a full stimulus finishes (~13 min each). Do not interpret
silence in the log as a hang — check `squeue` to confirm the job is still running.

#### Step 1 — SLURM pilot (3 stimuli × 1500 bins, ~40 min)

A ready-made SLURM script is at `scripts/slurm_extract_delta_t.sh`. It uses SLURM array jobs:
each `SLURM_ARRAY_TASK_ID` maps to a chunk of 16 stimuli (full run) or 3 stimuli (pilot).

```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling

# Submit pilot (array task 0 only, 3 stimuli, t_stride=1)
sbatch --time=02:00:00 --array=0 scripts/slurm_extract_delta_t.sh

# Monitor
squeue -u $USER
tail -f logs/delta_t_<JOBID>_0.out
```

Logs go to `logs/delta_t_<JOBID>_<TASK>.out` / `.err`.

Pilot success output: `SUCCESS  task=0  stim_start=0` at the end of the log,
and an HDF5 file in `outputs/features/whisper-base-delta-t-slurm-pilot/chunk_0/`.

#### Step 2 — full SLURM run (20 tasks × 16 stimuli, ~4 h wall time)

Edit line `MODE="pilot"` → `MODE="full"` in `scripts/slurm_extract_delta_t.sh`, then:

```bash
sbatch --array=0-19 scripts/slurm_extract_delta_t.sh
```

Each of the 20 tasks writes to `outputs/features/whisper-base-delta-t/chunk_<N>/`.
The job uses 1 CPU and 6.9 GB RAM per task (cluster MaxMemPerCPU limit).

**Important:** `#SBATCH` directives must appear before any executable code in the script.
The script is already correctly structured — don't move the `#SBATCH` block below the
`MODE=` variable assignments or the cluster will silently ignore the directives.

#### Step 3 — temporal evaluation (after features complete)

The evaluator loads features from a single directory. After the parallel run, either:
- (a) merge all `chunk_N/` directories into one by moving the HDF5 files, or
- (b) run `mbs-evaluate-temporal` once per chunk and combine summaries.

Option (a) is simpler. HDF5 filenames now encode `stim_start_idx` (the global stimulus
offset for each SLURM task), so filenames are globally unique and a flat `cp` works:

```bash
REPO=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
mkdir -p $REPO/outputs/features/whisper-base-delta-t/merged
cp $REPO/outputs/features/whisper-base-delta-t/chunk_*/feats*.h5 \
   $REPO/outputs/features/whisper-base-delta-t/merged/

mbs-evaluate-temporal \
  --model_id whisper-base \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --features_dir outputs/features/whisper-base-delta-t/merged/ \
  --data_hdf5_path outputs/neural_data/broderick2018_30s.h5 \
  --output_dir outputs/results/whisper-base-delta-t/
```

**Note on the first full run (job 54867710, 2026-06-02):** chunk_19 is missing its
last 2 stimuli (indices 312–313) due to a since-fixed flush bug (`stim_idx == total-1`
was wrong when `stim_start_idx > 0`; fixed to `stim_idx == end - 1`). All other 19
chunks are complete. The evaluation (job 54912412) ran on 312/314 stimuli — results
are valid. If you need the complete 314-stimulus run, re-extract chunk_19 with the
fixed code and re-evaluate:

```bash
# Re-extract missing stimuli 312–313 (only ~26 min, 2 stimuli × 13 min each)
python -m mbs.extraction.extract_features_delta_t \
  --model_id whisper-base \
  --data_root /work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea/stimuli \
  --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
  --output_dir outputs/features/whisper-base-delta-t/chunk_19/ \
  --stim_start_idx 312 --n_stimuli 2 --save_every 8 --t_stride 1 --batch_t 4
# Then re-merge (the new file will be feats_delta_t-start_00312-batch_0-seed_42.h5)
cp outputs/features/whisper-base-delta-t/chunk_19/feats_delta_t-start_00312*.h5 \
   outputs/features/whisper-base-delta-t/merged/
```

**Success criterion (ds004408):** mean prediction score above zero across most time bins for
Fz, T7, T8. Layer ranking should be consistent with Phase 4a (blocks.2–4 best). A sharp peak
at 50–300 ms is NOT expected for naturalistic speech — that criterion applies to Phase 6 (MMN).

---

## 6. Known challenges and open questions

### Temporal resolution: model-specific time grids

The EEG downsampling rate is **model-specific** — it is set by the model's architecture,
not a fixed constant. The model is always the bottleneck: you cannot predict EEG at finer
temporal resolution than the model provides, because the model has no information below
that resolution. Downsampling EEG to match the model loses nothing on the prediction side.

| Model | Model time grid | EEG target rate |
|---|---|---|
| Whisper (all sizes) | 20 ms/step (1500 bins / 30 s) | 50 Hz |
| wav2vec2 | ≈20 ms/step (499 bins / 10 s) | 50 Hz |
| VGGish | 1000 ms/step (10 bins / 10 s) | 1 Hz |
| AST | 1214 patch tokens (2D, not purely temporal) | mean-pool only |

For Whisper, 50 Hz gives 25 time points in a 500 ms MMN epoch — sufficient to resolve
the MMN peak at 100–200 ms.

### Full_T vs Delta_T: **decision — we use Delta_T** (2026-06-02)

Sophie's pipeline extracts activations in two modes:

- **Full_T:** single forward pass on the full stimulus → `[T, d_model]` per layer.
  Non-causal: time bin t has "seen" the entire future stimulus.
- **Delta_T:** T separate forward passes, each on a zero-padded truncation of the waveform
  (`[s_1,…,s_i, 0,…,0]`), collecting output at position i. Causal: each bin sees only past context.

**We use Delta_T.** The reason is scientific, not just a preference:

The brain at time t had only heard audio up to t. Using Full_T features at t gives the model
information the brain never had, creating an information asymmetry that can bias scores and
layer selection. For the **MMN EEG dataset** (Phase 6) this is a fundamental confound: the
deviant tone is *surprising* precisely because the brain did not see it coming. A Full_T model
at the deviant position already encoded it from future context, so its representation does not
carry the prediction-error signal the brain shows. Delta_T is therefore **not optional** for
the MMN dataset. For ds004408 (naturalistic speech) it is also the principled choice and keeps
the feature type consistent with Sophie's unit-selection pipeline throughout.

**Compute cost on CPU:** ~15 s per forward pass (Whisper-base). Full run (314 stimuli ×
1500 bins) costs ~97 wall-clock hours on a 20-core node (each core handles a chunk of stimuli
in parallel). Use a GPU node for the full run — see Section 5b for both strategies.

Mean-pool (Phase 4a) was a quick sanity check using Full_T, which is acceptable because
(a) the temporal axis was collapsed anyway, and (b) the goal was only to confirm a signal
exists before investing in the full Delta_T temporal run.

### FCz (and Cz, C3, C4) noise ceiling is ~0% — expected for this paradigm

In the Broderick 2018 (audiobook) dataset, FCz NC = 0.0%, Cz = 2.6%, C3 = 0.8%, C4 = 0.0%.
This was diagnosed with `scripts/diagnose_roi_mapping.py` (2026-06-04) and confirmed to be
**genuine physiology, not a bug**. Key findings from the diagnostic:

- FCz → BioSemi channel **C23**, distance 9.8 mm (same as Fz → C21 at 9.8 mm). Mapping is correct.
- No channel collisions (each standard name maps to a unique BioSemi electrode).
- The fronto-central / central strip simply does not drive consistent cross-subject responses
  during **passive audiobook listening**. The NC is an honest measure of cross-subject agreement,
  and motor/central cortex has no reason to respond consistently to speech in this paradigm.

Compare: temporal (T7/T8 ~75%) and parietal/frontal (Pz 74%, Fz 50%) are driven by auditory
cortex envelope-following and attention responses that ARE consistent across subjects.

**FCz is still the primary electrode of interest for Phase 6 (MMN).** The oddball paradigm
generates a sharp, time-locked deviant response at FCz/Fz at 100–200 ms that will produce
strong cross-subject agreement (and thus high NC) in that dataset. The Broderick NC numbers
should not be used to judge FCz's viability — different paradigm, different response.

**For the Broderick temporal evaluation, the scientifically meaningful electrodes are:**
Fz (50%), T7/T8/Pz (~73–75%), F3/F4 (~20–23%). Ignore NC-corrected scores at FCz/Cz/C3/C4.

### wav2vec2: variable-length output

wav2vec2 outputs roughly 1 frame per 20ms (~50 Hz), but the number of frames depends on
audio length. Pad/truncate all stimuli to the same length before extraction, or rely on
mean pooling which handles variable length naturally.

### AST: tokens are not temporal

AST converts the spectrogram into 1214 patch tokens (frequency x time patches).
Mean pooling over all 1214 tokens is the simplest approach; a more principled
alternative pools only tokens covering the stimulus time window.

### VGGish: coarse time resolution

VGGish processes 0.96s frames at 1s hops, giving T=10 bins for a 10s clip.
Expect lower encoding scores than transformer models due to the coarse resolution.

### Stimulus count and regression reliability

Ridge regression with d=512 (whisper-base) features and N stimuli:

- Need N >> 512 to avoid underdetermined regression
- ds004408 has segments from a single audiobook (~60 segments) — likely insufficient
  at the coarse segmentation level; plan to sub-segment into shorter clips
- Discuss with Kadir: THINGS-EEG used 22,248 image stimuli; we probably need at least ~1,000

### Noise ceiling for cross-subject data

ds004408 likely has single-trial recordings per subject per segment (naturalistic paradigm).
The noise ceiling will be cross-subject (split subjects in half, not trials).
This gives a lower bound on the ceiling compared to within-subject split-half.

---

## 6b. Bugs fixed (all already in codebase)

### Phase 4a bringup (2026-06-01)

| File | Symptom | Fix |
|---|---|---|
| `extract_features.py` | `error: argument --backbone_source: invalid choice: 'audio'` | Added `"audio"` to `--backbone_source` choices |
| `extract_features.py` | `RuntimeError: mixed dtype (CPU)` — Whisper LayerNorm calls `x.float()` internally | Auto-downgrade to float32 when `device == cpu` |
| `configs/extraction/audio/*.json` | `KeyError: backbone.backbone.blocks.0` — factory double-prepends `backbone.` | Layer names in JSON must be `blocks.{i}`, not `backbone.blocks.{i}` |
| `configs/extraction/audio/*.json` | `KeyError: 'position'` — evaluator requires normalized depth field | Added `"position": 0.0…1.0` to each layer entry |
| `evaluation_helpers.py` | `KeyError: b'audio01_0000000'` — IDs returned as bytes, mapping used strings | Added `.decode('utf-8')` in `load_neural_data` |
| `evaluation_helpers.py` | `ValueError: x and y must have the same length` — sklearn squeezes single-channel output to 1D | Added `y_pred.reshape(-1,1)` guard in `compute_metrics` and `pearsonr_score` |

### Phase 4b bringup (2026-06-02 – 2026-06-04)

| File | Symptom | Fix |
|---|---|---|
| `extract_features_delta_t.py` | All chunk files had identical names (`batch_0`, `batch_1`) → flat `cp` to merged/ collided and only chunk_0 survived | Filename now encodes `stim_start_idx`: `feats_delta_t-start_NNNNN-batch_K-seed_42.h5` |
| `extract_features_delta_t.py` | Last batch of last chunk never flushed — 2 stimuli lost from chunk_19 | Flush condition was `stim_idx == total-1`; fixed to `stim_idx == end-1` (correct when `stim_start_idx > 0`) |
| `evaluate_features_temporal.py` | `AttributeError: 'dict' object has no attribute 'replace'` — layer list JSON loaded as dicts | Extract `entry["name"]` from each entry after JSON load |
| `evaluate_features_temporal.py` | `ValueError: x and y must have the same length along axis` — sklearn squeezes single-ch output to 1D | Added `y_pred = y_pred.reshape(y_ref.shape)` after `model.predict()` |
| `evaluate_features_temporal.py` | 2 missing stimulus IDs (from flush bug) caused entire layer × ROI to be `continue`d → empty results | Filter to matched-ID subset instead of skipping; warn but proceed |

## 7. Implementation status

| Phase | Component | Status |
|---|---|---|
| 0 | Environment setup (uv, Python 3.11) | Done |
| 0 | Fork + clone `hanme/multimodal-brain-scaling` | Done |
| 1 | `audio_models.py` — Whisper backbone loader + `WhisperTransform` | **Done** |
| 1 | `pyproject.toml` — `audio` extra | **Done** |
| 1 | Register `audio` in backbone registry | **Done** |
| 2 | `datasets_audio.py` — `AudioPreprocessor`, `AudioSegmentDataset` | **Done** |
| 2 | `_create_audio_hook_feature_extractor` (bypass image-shape inference) | **Done** |
| 2 | `evaluate_features_temporal.py` — per-timepoint Ridge CLI | **Done** |
| 3 | Download ds004408 | **Done** — `/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea` |
| 3 | `format_eeg_hdf5.py` — BIDS EEG → `[n_stimuli, T_model, n_ch]` HDF5 | **Done** — `outputs/neural_data/broderick2018_30s.h5` (252 train / 62 test; **67 ROIs**) |
| 4a | Mean-pool pilot: whisper-base × ds004408 (sanity check) | **Done** — results in `outputs/layer_search/whisper-base-meanpool/`; pearsonr >0.82 at T7/T8/Fz |
| 4b-pre | `extract_features_delta_t.py` — Delta_T causal extractor | **Done** (2026-06-02) |
| 4b-pre | `scripts/slurm_extract_delta_t.sh` — SLURM array job script | **Done** (2026-06-02) |
| 4b | Delta_T features: whisper-base × ds004408 (312/314 stimuli) | **Done** — job 54867710; merged at `outputs/features/whisper-base-delta-t/merged/` |
| 4b | Temporal evaluation: whisper-base × ds004408 (67 ROIs) | **Done** — job 54930384, ~31.5 h; results at `outputs/results/whisper-base-delta-t-full/` |
| 4b-sweep | Window/stride sweep infrastructure for whisper-small | **Done** (2026-06-04) — see `scripts/submit_whisper_small_sweep.sh` |
| 5 | `configs/extraction/audio/whisper_small_layers.json` (12 blocks) | **Done** (2026-06-04) |
| 5 | `load_wav2vec2`, `load_vggish`, `load_ast` in `audio_models.py` | TODO |
| 5 | Scale runs: wav2vec2 (temporal), AST + VGGish (mean-pool) | TODO |
| 6 | MMN EEG dataset integration (dataset TBD) | TODO |

---

## ⏭ Immediate next steps (pick up here — last updated 2026-06-04)

### Current state

| Item | Status |
|---|---|
| Phase 4a (mean-pool pilot) | ✅ Done — pearsonr > 0.82 at Fz/T7/T8; best layer `blocks.4` |
| Delta_T features (312/314 stimuli) | ✅ Done — `outputs/features/whisper-base-delta-t/merged/` (39 files) |
| Temporal evaluation (67 ROIs) | ✅ Done — job 54930384, ~31.5 h; results at `outputs/results/whisper-base-delta-t-full/` |
| whisper-small sweep infrastructure | ✅ Done — `scripts/submit_whisper_small_sweep.sh` |

**Activate the environment** (needed every session):
```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh       # loads gcc/13.2.0 + python/3.11.7 and activates .venv
```

---

### Phase 4b results (2026-06-07)

Job 54930384 completed after ~31.5 h wall time. Results at
`outputs/results/whisper-base-delta-t-full/` (`temporal_scores.h5` + `temporal_scores_summary.json`).

**Key finding: blocks.2 is the best layer** (NC-corrected Pearson r, mean over 1500 time bins):

| Electrode | NC (raw) | blocks.0 | blocks.1 | blocks.2 (best) | blocks.3 | blocks.4 | blocks.5 |
|---|---|---|---|---|---|---|---|
| Fz | 50.4% | 0.035 | 0.043 | **0.110** | 0.048 | 0.047 | 0.039 |
| T7 | 75.3% | -0.003 | 0.001 | **0.050** | 0.017 | 0.010 | -0.006 |
| FT7 | 91.1% | 0.005 | 0.008 | **0.060** | 0.021 | 0.013 | 0.003 |
| AF3 | 94.6% | 0.003 | 0.010 | **0.069** | 0.025 | 0.016 | 0.003 |

This differs from Phase 4a (mean-pool: blocks.4 best at Fz). The temporal evaluation captures the
auditory cortex envelope-following response, which is driven by a more sensory (earlier) layer than
the whole-stimulus average.

**Statistical validation of layer signals** (`scripts/plot_score_distributions.py`,
figure: `outputs/figures/whisper_base_score_distributions.png`):

A one-sample t-test of scores[T] against 0 across 8 electrodes reveals three tiers:

| Tier | Layers | Mean range | Evidence |
|---|---|---|---|
| Unambiguous signal | **blocks.2** | 0.037–0.110 | t = 10–21, 62–70% of bins > 0, all electrodes |
| Weak but real | blocks.1, 3, 4 | 0.009–0.048 | t = 2–10, 52–58% bins > 0, significant at most electrodes |
| Noise | blocks.0, blocks.5 | −0.007–0.015 | t < 2, ~50% bins > 0, not significant at AF3/FT7/T7/TP7/Fpz |

blocks.2 mean is 2–5× larger than the next best layer across all speech electrodes. Fz is an
exception — all 6 layers reach significance at Fz, because its lower and more midline position
picks up a broader mixture of processing stages.

**Autocorrelation correction (option 1 implemented):** ρ₁ ≈ 0.71–0.82 across layers/electrodes,
giving n_eff ≈ 160–460 from T=1500. Implemented in `scripts/plot_score_distributions.py` via
`n_eff = T × (1−ρ₁) / (1+ρ₁)` and `t_corr = mean / (std / √n_eff)` with `df = n_eff − 1`.
After correction the tier structure sharpens — only blocks.2 is *** everywhere; blocks.3 holds
* or ** at a few electrodes; blocks.0/1/4/5 are ns at all speech electrodes.

Three approaches exist for this correction (see `02_project_plan_make_compatible_for_auditory_EEG.md`
for full detail):
- **Option 1 (implemented):** AR(1) n_eff = T×(1−ρ₁)/(1+ρ₁). Fast, assumes exponential ACF decay.
- **Option 2:** Full-ACF n_eff = T / (1 + 2Σρₖ). More accurate for slow-decaying autocorrelation.
- **Option 3:** Circular shift permutation — randomly shift the time series, preserving autocorrelation structure; p = fraction of null means ≥ observed. Non-parametric, ideal for paper reporting.

Robust evidence = Cohen's d and ratio of means, not p-values.

**Electrode NC map — 33 valid electrodes (raw NC > 0), sorted by NC:**

| Electrode | NC (raw) | Note |
|---|---|---|
| AF3 | 94.6% | Left anterior frontal — highest NC |
| FT7 | 91.1% | Left fronto-temporal |
| P9 | 88.0% | Left posterior temporal (extreme lateral) |
| TP7 | 84.3% | Left temporo-parietal |
| Fpz | 79.9% | Frontal pole midline |
| P7 | 78.5% | Left posterior temporal |
| T7 | 75.3% | Left temporal — primary auditory cortex |
| Pz | 74.4% | Parietal midline |
| T8 | 73.3% | Right temporal |
| FC3 | 71.3% | Left frontal-central |
| AFz | 64.8% | Anterior frontal midline |
| O2 | 64.3% | Right occipital (note: O1 = 0%, mapping gap) |
| Fp2 | 59.0% | Right frontal pole |
| P6 | 58.0% | Left posterior parietal |
| F7 | 57.4% | Left frontal |
| C5 | 57.4% | Left central |
| PO4 | 54.8% | Right parieto-occipital |
| Fz | 50.4% | **Primary MMN electrode** |
| P10 | 47.1% | Right posterior temporal |
| TP8 | 43.3% | Right temporo-parietal |
| Fp1 | 41.7% | Left frontal pole |
| Oz | 36.0% | Occipital midline |
| FC4 | 29.5% | Right frontal-central |
| CP6 | 25.2% | Right centro-parietal |
| F3 | 23.2% | Left frontal |
| F4 | 20.4% | Right frontal |
| AF4 | 12.0% | Right anterior frontal |
| P8 | 11.9% | Right posterior temporal |
| AF7 | 11.6% | Left anterior frontal |
| F8 | 10.8% | Right frontal |
| POz | 6.3% | Parieto-occipital midline |
| Cz | 2.6% | Central midline — near zero |
| C3 | 0.8% | Left motor cortex — near zero |

**Strong left-hemisphere dominance** (language lateralization): AF3 (94.6%) vs AF4 (12.0%),
FT7 (91.1%) vs FT8 (0%), FC3 (71.3%) vs FC4 (29.5%), TP7 (84.3%) vs TP8 (43.3%).

**28 electrodes excluded (raw NC = 0.0% in HDF5):**
P4, AF8, C1, C2, C4, C6, CP1, CP2, CP3, CP4, CP5, CPz, F1, F2, F5, F6, FC1, FC2, FC5, FC6,
FCz, FT8, O1, P1, P2, P3, P5, PO3.

These fall into two groups: (a) genuine physiology — FCz/Cz/C1/C2/CP* do not respond
consistently to continuous speech; (b) mapping gaps in `format_eeg_hdf5.py` where standard
10-20 names were not matched to BioSemi channels (O1, P1–P3, F1/F2, FC1/FC2, etc.).
**Filter rule: exclude any electrode where raw NC == 0.** Note FCz is still the primary
electrode of interest for Phase 6 (MMN oddball), where the time-locked deviant response will
produce high cross-subject agreement and non-zero NC.

---

### Next steps (pick up here — 2026-06-07)

**1. Results analysis / visualization** ✅ Done (2026-06-07)

| Script | Output | What it shows |
|---|---|---|
| `scripts/plot_temporal_scores.py` | `outputs/figures/whisper_base_temporal_scores.png` | scores[T] time-course per layer × 8 electrodes (smoothed + raw) |
| `scripts/plot_score_distributions.py` | `outputs/figures/whisper_base_score_distributions.png` | Violin distributions of scores[T] vs. zero, with t-test annotations |
| — | `outputs/figures/score_distribution_summary.json` | Full statistical summary (mean, SE, t, p, fraction>0 per layer × electrode) |

Key finding: blocks.2 is unambiguously the best layer (2–5× larger mean than any other).
All other layers hover near zero at speech-sensitive electrodes; see statistical validation above.

**2. Dataset validation (do before scaling to more models)**

Before investing in other models, check that whisper-base blocks.2 results generalise:

- Does a *different* naturalistic speech EEG dataset give similar goodness-of-fit and the same
  best layer? (pick one dataset; discuss with Gokce which is most accessible)
- Does pooling Broderick + a second dataset improve scores, or are returns quickly diminishing?

If generalisation holds: Broderick alone is a sufficient training set → scale to more models.
If not: understand why before scaling (dataset-specific artefact, very different electrode
coverage, etc.)

**3. Scale to other audio models (Whisper variants first)**

Priority order (from easiest to most involved):

1. **Whisper tiny / small / medium / large** — same feature extraction code, only configs and
   SLURM scripts need to be changed. Run Phase 4a (mean-pool) first for a quick cross-size
   comparison, then Phase 4b (temporal) for the winner.
2. **wav2vec2-base / wav2vec2-large** — different input format (raw waveform) and different
   time grid (499 bins at ~20ms, aligned with EEG), but otherwise the same pipeline.
   More setup: a new EEG HDF5 reformatted to 10s windows is needed.
3. **AST** — mean-pool only (no temporal resolution). Lowest priority.

**VGGish: excluded from temporal evaluation.** Its output is 10 bins at 1s resolution —
far too coarse to predict EEG at 20ms. Can be included in mean-pool comparisons as a baseline.

**4. In-silico MMN analysis via trained Ridge mapping (no new data needed)**

This is the key step connecting the encoding model to Sophie's schizophrenia pipeline. The
idea: the trained linear mapping (Ridge weights from Phase 4b, Broderick 2018) can predict
electrode-level EEG responses to *any* audio, including MMN stimuli — without collecting new
EEG.

Steps:
1. Feed Sophie's MMN stimuli (standard + deviant tone sequences) through whisper-base to get
   Delta_T features at blocks.2.
2. Apply the trained Ridge weights → get predicted electrode-level EEG time courses.
3. Compute the deviant-minus-standard difference wave at Fz (in the predicted responses).
4. Does the model show an MMN-like component at 100–200ms? If yes: the model's internal
   representations contain MMN-relevant information, consistent with it predicting naturalistic
   speech EEG.
5. Once multiple models are available (step 3 above): compare the in-silico MMN across
   architectures — does a model with "schizophrenia-like" properties show a reduced/absent MMN?

This approach does not require patient EEG and can be run as soon as the Ridge mapping exists.
Patient EEG (Phase 6b in the project plan) is the follow-up to validate the predictions
against real data.

**Summary of priorities:**

| # | Task | Blocker | Estimated effort |
|---|---|---|---|
| 1 | Dataset validation (Step 2 above) | Need to identify a second naturalistic speech EEG dataset | 1–2 days formatting + 1 SLURM job |
| 2 | Whisper family (tiny/small/medium/large) Phase 4a | None | 1–2 days configs + SLURM submission |
| 3 | In-silico MMN on whisper-base mapping | Sophie needs her MMN stimuli as audio files | 1 day scripting |
| 4 | wav2vec2 integration | 10s window EEG HDF5 not yet created | 2–3 days |
| 5 | Real patient MMN comparison | MMN EEG dataset (patient + control) | TBD |
