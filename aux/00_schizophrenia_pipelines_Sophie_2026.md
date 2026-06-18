# Sophie Sigfstead's Schizophrenia Pipeline (2026)

**Repo location:** `/work/upschrimpf1/sigfstea/scz_updated_pipeline_071226/`

The project studies whether audio AI models produce MMN-like (mismatch negativity) neural responses.
MMN is the brain's automatic response to an unexpected "deviant" sound in a stream of repeated "standard" tones,
and is robustly impaired in schizophrenia patients. The goal is to identify which internal units of audio models
produce the most MMN-like signal, and then search for stimuli that maximally drive those units ("supernatural MMN").

There is also an earlier prototype version at `/work/upschrimpf1/sigfstea/unit_identification_pipeline/`
(summer 2025, no time axis, single deviant per method, no counterbalancing). The updated pipeline is the one to use.

---

## Directory survey of `/work/upschrimpf1/sigfstea/` (surveyed 2026-06-12)

Read-only walk of Sophie's work area, done to locate the MMN stimuli + the multi-model code
before the in-silico MMN / Workstream-B work. There are **two parallel codebases**: the
**encoding** side (model → EEG, where our mTRF lives) and the **MMN unit/stimulus** side (two
generations). They are currently separate; the join is conceptual (our mTRF says *which
layer/units* predict EEG; the MMN pipeline provides *what to feed through* that mapping).

### Sizes (du, 2026-06-12)

| Dir | Size | What it is |
|---|---|---|
| `scz_updated_pipeline_071226/` | **1.7 TB** | **Current** MMN/scz pipeline (Apr–May 2026) — the "updated, use this one" |
| `multimodal-brain-scaling/` | 174 GB | Her copy of the auditory-EEG **encoding** repo (fork of `hanme`) |
| `miniconda3/` | 48 GB | Her Python env (+ `Miniconda3-latest-…sh` installer) |
| `unit_identification_pipeline/` | 14 GB | **Original** 2025 MMN unit-ID pipeline (git repo, Jun–Nov 2025) |
| `cortical_suprisal_dataset/` | n/a | 2nd speech EEG dataset — **no read access (700 perms)**; also mirrored in her encoding repo `data/` |

(Stray junk: `=12`, `unit_identification_pipeline/=4.20.0` — accidental conda-log redirects.)

### 1. `scz_updated_pipeline_071226/` — current MMN pipeline (1.7 TB)
This is where "Sophie's stimuli" and the "other-model extensions" actually live.
- `scripts/`: `00aa_generate_audio_stimuli.py` (**MMN tone generator** — sine tones, standard vs
  deviant, counterbalanced), `phase_00a_generate_activations.py` (Full_T + Delta_T activations),
  `phase_01_unit_selection.py`, full_t-vs-delta_t analysis scripts, SLURM submitters.
- `data/audio_outputs_literature/`: **~3,072 generated MMN `.wav` stimuli** (regular +
  counterbalanced, with metadata), organized **per model** (`ast/ vggish/ wav2vec2/ whisper/` —
  different input durations).
- `data/activations/` (+ `test_activations/`, `new_test_activations/`): **~200k `.npy`** Full_T
  and Delta_T "method" activations **and deviant−standard differences**, for **all 9 models**
  (whisper tiny→large, wav2vec2-base/large, ast, vggish). → the in-silico-MMN inputs are
  **already computed on disk**; read selectively (per model/method), don't scan the whole tree.
- `metadata/`: literature paradigm CSVs (frequency/intensity/duration), stimuli-search metadata.
- `analyses/`: several versioned full_t-vs-delta_t analyses + `unit_selection_methods_analysis/`.

### 2. `unit_identification_pipeline/` — original 2025 pipeline (14 GB, git repo)
Sophie's Jun–Nov 2025 summer project, with a README. Phase 1 = localize MMN-like units in
models; Phase 2 = search for "supernatural MMN" stimuli. Scripts: `audio_input_creation.py`,
`retrieve_model_activations.py`, `unit_id_statistical_testing.py`, `stimuli_response_computation.py`
(+ cluster/GPU/parallel variants). Contains `cross_val_pipeline_v3/` (code/configs/results,
multiple `step_04_analysis_results_*`), bootstrap samples, `statistical_results/`, and docs
(incl. a VGGish implementation guide). Model list even includes **Auristream**. Superseded by #1.

### 3. `multimodal-brain-scaling/` — her copy of the encoding repo (174 GB)
The model→EEG side (same fork we work in). Beyond our copy it has: a 2nd speech dataset
(`cortical_suprisal_dataset`), a large **multi-dataset "D3" pooling** effort
(`build_d3_features`, `diagnose_d3_*`, `aux/03_…`, `aux/06_…`), big results-analysis docs
(`05_whisper_results_analysis.md`), and the same `attn_probe/` seed. **Only `load_whisper`** —
the other-model loaders are NOT here; the multi-model code is the activation-extraction in #1/#2.

### Resolutions for our project
- **"The stimuli from Sophie"** = the ~3,072 MMN `.wav`s already generated in
  `scz_updated_pipeline_071226/data/audio_outputs_literature/` (per-model) — in the MMN repo,
  not the encoding repo.
- **"Extensions for the other models"** = multi-model activation extraction in the MMN repos
  (whisper sizes, wav2vec2, ast, vggish, auristream) — as `retrieve_model_activations*.py` /
  `phase_00a_generate_activations.py`, NOT as `audio_models.py` loaders in the encoding repo.
- **Caveat:** these paths are on Sophie's account and not under our git; confirm with her which
  is the live copy before depending on them (her dir isn't a git repo; `.DS_Store` ⇒ Mac-synced,
  so newer work may be local/uncommitted).

---

## Pipeline overview

```
Phase 0aa  →  Phase 0A  →  Phase 01  →  (Phase 0B/02, stimulus search — not yet run)
audio gen     activations   unit ID
```

### Phase 0aa — Audio stimulus generation
**Script:** `scripts/00aa_generate_audio_stimuli.py`
**Submit:** `scripts/submit_phase_0aa_literature.sh` / `submit_phase_0aa_stimuli_search.sh`

Generates `.wav` files for all (model, method_id) combinations.

- **Sample rate:** 16 kHz universal
- **Tone synthesis:** pure sine waves, 5 ms linear rise/fall ramps, amplitude in dB SPL (94 dB reference)
- **Stimulus type — standard:** K tone slots, all at standard frequency
- **Stimulus type — deviant:** stochastic prefix + fixed suffix ending in deviant tone;
  three trial levels N ∈ {3, 5, 7} × 5 random variations = **15 deviant files per method per model**
- **Counterbalancing:** a second set with standard/deviant frequencies swapped is also generated
- **Output dirs:** `data/audio_outputs_literature/audio_outputs_regular/{model}/` and `audio_outputs_counter/{model}/`
- **Metadata:** `data/audio_outputs_literature/audio_outputs_regular_metadata/metadata.csv` (one row per file)

> **Why the counterbalanced set matters — the two MMN constructions (physical control).** The MMN is
> read time-locked to the **last (eliciting) tone**. Two ways to build the standard you subtract:
> - **Definition 1 — uncontrolled:** standard `{1000,1000,1000,1000}` vs deviant `{1000,1000,1000,1200}`.
>   Surprise = the 1000→1200 shift at the end, but the **last tone differs physically** (1000 vs 1200),
>   so `deviant − standard` confounds surprise with the probe tone's acoustics.
> - **Definition 2 — physically controlled:** standard `{1200,1200,1200,1200}` vs deviant
>   `{1000,1000,1000,1200}`. Same surprise, but the **last tone is physically identical (1200 Hz) in
>   both** → the subtraction isolates surprise and holds the probe-tone acoustics fixed.
>
> The **counterbalanced (`audio_outputs_counter`) set is exactly the mechanism that delivers
> Definition 2**: averaging each physical tone equally across the standard and deviant roles matches the
> eliciting-tone acoustics across conditions. This is the design **Weber 2022** uses; the classic
> founding papers (Sams 1985, Tiitinen 1994) used the looser Definition 1 (oddball with physically
> different standard vs deviant tones). Our encoding-side in-silico MMN (`method_09`, identity-MMN) is
> also Definition 2 — the final tone is literally identical, deviance only in the context.
> Full write-up: `aux/project_plan_20260611.md §17`.

Model-specific durations and formats:

| Model group | Duration | Format |
|---|---|---|
| whisper (all sizes) | 30 s | raw `.wav` (16 kHz mono) |
| wav2vec2, vggish, ast | 10 s | raw `.wav` (16 kHz mono) |

### Phase 0A — Activation extraction
**Script:** `scripts/phase_00a_generate_activations.py`
**Submit:** `scripts/submit_phase_0a_array.sh` (one SLURM job per model)

For each (model, method_id):
1. Probe pass on a zero waveform to determine T (time bins) and which layers have a time axis
2. **Full_T activations:** single forward pass on the complete waveform → shape `[T, U]`
3. **Delta_T activations:** T separate forward passes on progressively truncated waveforms (causal simulation) → shape `[T, U]`
4. The 15 deviant activations are averaged in-place (float64 accumulation); then `difference = deviant_avg − standard` is computed

Only layers whose time dimension equals T are retained. VGGish runs on CPU; Whisper uses encoder only; wav2vec2 and AST hook the full model.

Output (6 files per method per model):
```
data/activations_full_t_method/{model}/method_{ID:02d}_standard_activation_full_t.npy
data/activations_full_t_method/{model}/method_{ID:02d}_deviant_avg_activation_full_t.npy
data/activations_full_t_method_difference/{model}/method_{ID:02d}_full_t_difference.npy
data/activations_delta_t_method/{model}/          [same three files with delta_t suffix]
```

### Phase 01 — MMN unit selection
**Script:** `scripts/phase_01_unit_selection.py`
**Submit:** `scripts/submit_phase_01.sh` (one job per model × activation_type = 18 jobs)

Run per (model, activation_type) pair end-to-end:

- **Step 1:** deterministic 18/6 train/held-out split of the 24 discovered method_ids (seed=42)
- **Step 2:** 60 metric×window combinations (5 metrics × 3 P-windows × 4 Q-windows)
  - P windows (baseline before final tone onset): P_1cyc, P_2cyc, P_3cyc
  - Q windows (response after final tone): Q_fto_150ms, Q_fto_end, Q_fte_150ms, Q_fte_end
  - Metrics: B_NP, B_fixed, method_7, V1, V4
  - One-sample t-test (H1: mean < 0) + BH-FDR correction; top **5%** of units selected
- **Step 3:** Bootstrap stability (B=100 resamplings of training methods, seed=12345)
- **Step 4:** Visualizations — held-out difference/standard/deviant traces, bootstrap histograms, p-value summary tables

Output: `data/unit_selection/{activation_type}/primary_split/{model}/`

---

## Per-model input format details
**Source:** `scripts/phase_00a_generate_activations.py` — `load_and_preprocess_audio()`, `run_forward_pass()`, `load_model()`

The pipeline always reads `.wav` files (no mp3/flac support — `torchaudio.load()` is called directly).
The loading function resamples to 16 kHz if needed and converts to mono before handing off to each model.
What varies per model is **how that raw waveform is then preprocessed before the forward pass**:

### Whisper (tiny / base / small / medium / large)
- **Input to model:** log-mel spectrogram, computed by `whisper.log_mel_spectrogram(audio_np, n_mels=model.dims.n_mels)`
  - n_mels = 80 for tiny/base/small/medium; 128 for large
  - Standard Whisper STFT: 25 ms window, 10 ms hop → 100 frames/sec
- **Duration:** exactly **30 s** (30 000 ms) — the probe waveform and all stimuli are generated at this length
- **No HuggingFace processor** — uses the `openai-whisper` library directly
- **Only the encoder** is hooked; the decoder is never run
- **Device:** CUDA if available, CPU otherwise; AMP (`torch.cuda.amp.autocast`) on CUDA

### wav2vec2-base / wav2vec2-large
- **Input to model:** raw waveform passed through `Wav2Vec2Processor` (`facebook/wav2vec2-base-960h` / `facebook/wav2vec2-large-960h`)
  - Processor normalises the waveform (mean 0, unit variance) and returns `input_values` tensor
  - No spectrogram — model ingests raw samples directly
- **Duration:** exactly **10 s** (10 000 ms)
- **Device:** CUDA if available

### VGGish
- **Input to model:** mel-spectrogram windows computed internally by `torchvggish.waveform_to_examples(audio_np, 16000)`
  - Produces frames of shape [N_windows, 1, 96, 64] (0.96 s non-overlapping windows of 64-bin log-mel)
  - The model sees multiple windows simultaneously (batch = number of 0.96 s chunks in the audio)
- **Duration:** exactly **10 s** → ~10 windows of 0.96 s
- **Device:** forced **CPU** (hardcoded, GPU not supported via torchvggish)
- **No HuggingFace processor**

### AST (Audio Spectrogram Transformer)
- **Input to model:** log-mel spectrogram computed by `ASTFeatureExtractor` (`MIT/ast-finetuned-audioset-10-10-0.4593`)
  - 128 mel bins, 25 ms window, 10 ms hop, AudioSet normalisation (mean=−4.268, std=4.569)
  - Time dimension padded/truncated to 1024 frames (≈ 10.24 s at 100 frames/s)
- **Duration:** exactly **10 s**
- **Device:** CUDA if available

### Input → output transformation (empirical, from actual run data)

The pipeline stores one value per (time bin, unit) per layer. All layers are concatenated into a flat
unit axis of size U_total. The final activation tensor per stimulus per model is shape **[T, U_total]**
where T is the number of time bins and U_total is the total number of units across all valid layers.

T and U_total are not hardcoded — they are discovered at runtime by a probe forward pass
(`run_probe_pass()` in `scripts/phase_00a_generate_activations.py`, lines ~420–504) and then
persisted to `split_info.json` by Phase 01. The numbers below come from those saved files and can
be verified with:

```bash
# verify T, hop_ms, U_total for each model that has been run:
for f in /work/upschrimpf1/sigfstea/scz_updated_pipeline_071226/analyses/unit_selection_methods_analysis/mmn_unit_selection_full_t/primary_split/*/split_info.json; do
  echo "=== $(basename $(dirname $f)) ==="
  python3 -c "import json; d=json.load(open('$f')); print('T=',d.get('T'), 'hop_ms=',d.get('hop_ms'), 'U_total=',d.get('U_total'))"
done
```

The _why_ behind each T value follows from the model architecture and is explained per model below.

#### Whisper (all sizes) — 30 s input
- `.wav` 16 kHz → `whisper.log_mel_spectrogram(audio_np, n_mels=model.dims.n_mels)` (line ~305 in `phase_00a_generate_activations.py`)
  → 3000 mel frames at 10 ms/frame (80-bin log-mel for tiny/base/small/medium; 128-bin for large)
- Whisper's encoder has two conv layers; the second uses stride 2, halving time → **T = 1500 bins, one every 20 ms**
- All encoder transformer blocks share T = 1500; total coverage = 1500 × 20 ms = 30 000 ms
- `analyze_full_t_vs_delta_t_v2.py` line 45 hard-codes `T = 1500; HOP_MS = 20` for cross-check

| Size | U_total (sum of U across all valid layers) |
|---|---|
| tiny | 28 800 |
| base | 56 832 |
| small | 168 192 |
| medium | 445 440 |
| large | 741 120 |

#### wav2vec2-base / wav2vec2-large — 10 s input
- `.wav` 16 kHz → `Wav2Vec2Processor(waveform, sampling_rate=16000)` (line ~330 in `phase_00a_generate_activations.py`)
  → normalises raw waveform (mean 0, unit variance), returns `input_values` tensor; no spectrogram
- wav2vec2's CNN feature extractor has a total stride of 320 samples = 20 ms at 16 kHz; 160 000 samples / 320 = 500
  frames, processor trims one → **T = 499 bins, one every ≈20 ms**

| Size | U_total |
|---|---|
| base | 229 888 |
| large | 600 576 |

#### VGGish — 10 s input
- `.wav` 16 kHz → `torchvggish.waveform_to_examples(audio_np, SAMPLE_RATE)` (line ~319 in `phase_00a_generate_activations.py`)
  → non-overlapping 0.96 s windows → shape [N_windows, 1, 96, 64]; for 10 s audio N_windows = 10
- The probe pass detects N_windows as the "batch" dimension and treats it as the time axis
  (see `run_probe_pass()` lines ~460–462: 4D tensors with `act.shape[0] > 1` contribute to the T vote)
- **T = 10 bins, hop = 1000 ms** — one bin per second; by far the coarsest temporal resolution
- U_total = **4 928** (CNN activations spatially pooled per layer, then concatenated across layers)

Note: VGGish's "time bins" are whole 0.96 s chunks, not fine-grained time steps.

#### AST (Audio Spectrogram Transformer) — 10 s input
- `.wav` 16 kHz → `ASTFeatureExtractor(waveform, sampling_rate=16000)` (line ~340 in `phase_00a_generate_activations.py`)
  → 128-bin log-mel spectrogram, AudioSet normalisation (mean=−4.268, std=4.569), padded to 1024 time frames
- AST patch-embeds the 2D spectrogram with 16×16 patches at stride 10 in both time and frequency:
  (1024 − 16)/10 + 1 = 101 time patches ... but the HF model `MIT/ast-finetuned-audioset-10-10-0.4593`
  uses its own patch grid → 1212 content patches + 2 special tokens = **T = 1214 sequence positions**
- This is the transformer _token_ axis, not a 1D temporal axis — AST flattens 2D spatial patches into a sequence
- **hop_ms ≈ 8.24 ms** is purely a ratio (10 000 ms / 1214) used by Phase 01 for window alignment; it does not
  correspond to a true uniform temporal stride
- U_total = **241 920**

### Summary table

Numbers verified from `split_info.json` files (see verification command above).

| Model | Input | Duration | T (time bins) | hop (ms) | U_total |
|---|---|---|---|---|---|
| whisper-tiny | `.wav` 16 kHz → 80-bin log-mel | 30 s | 1 500 | 20.0 | 28 800 |
| whisper-base | `.wav` 16 kHz → 80-bin log-mel | 30 s | 1 500 | 20.0 | 56 832 |
| whisper-small | `.wav` 16 kHz → 80-bin log-mel | 30 s | 1 500 | 20.0 | 168 192 |
| whisper-medium | `.wav` 16 kHz → 80-bin log-mel | 30 s | 1 500 | 20.0 | 445 440 |
| whisper-large | `.wav` 16 kHz → 128-bin log-mel | 30 s | 1 500 | 20.0 | 741 120 |
| wav2vec2-base | `.wav` 16 kHz → raw waveform (normalised) | 10 s | 499 | ≈20.0 | 229 888 |
| wav2vec2-large | `.wav` 16 kHz → raw waveform (normalised) | 10 s | 499 | ≈20.0 | 600 576 |
| vggish | `.wav` 16 kHz → 0.96 s window chunks [N×1×96×64] | 10 s | 10 | 1 000.0 | 4 928 |
| ast | `.wav` 16 kHz → 128-bin log-mel → 2D patch tokens | 10 s | 1 214 | ≈8.2 | 241 920 |

**Duration flexibility:** the pipeline has no built-in support for variable durations. Durations are fixed constants
(`MODEL_DURATION_MS` dict). Changing them would require regenerating audio and rerunning activation extraction.
Stimuli shorter than the target duration would need padding; longer stimuli would be truncated (this is not
currently handled — the audio generation script always produces exactly the right length).

---

## Models (9 total)
**Defined in:** `scripts/phase_00a_generate_activations.py` (`CV_MODELS`, `MODEL_SUBDIR_MAP`)

| Model | Architecture | HF / source identifier |
|---|---|---|
| whisper-tiny | Transformer encoder | openai/whisper-tiny |
| whisper-base | Transformer encoder | openai/whisper-base |
| whisper-small | Transformer encoder | openai/whisper-small |
| whisper-medium | Transformer encoder | openai/whisper-medium |
| whisper-large | Transformer encoder | openai/whisper-large |
| wav2vec2-base | Self-supervised transformer | facebook/wav2vec2-base-960h |
| wav2vec2-large | Self-supervised transformer | facebook/wav2vec2-large-960h |
| vggish | CNN (AudioSet) | torchhub (CPU only) |
| ast | Audio Spectrogram Transformer | MIT/ast-finetuned-audioset-10-10-0.4593 |

---

## Stimulus sets
**Metadata:** `metadata/`

### Literature stimuli
**File:** `metadata/literature_frequency_intensity_duration_metadata.csv`

76 stimulus definitions from published MMN research.
The 30 frequency-change methods are used for unit identification (frequency MMN is the most robust paradigm).
Example methods: Kathmann_1995a (600→1000 Hz), Alain_1998a (1000→1122 Hz), Hirayasu_1998a (1000→1200 Hz).
See `metadata/README.md` for the full list of method IDs and column definitions.

### Comprehensive search stimuli
**File:** `metadata/stimuli_search_comprehensive_metadata.csv`

~3,500 pure-tone parameter sweep entries for the later stimulus search phase (Phase 0B/02, not yet run).
Coverage: 200–8000 Hz frequency × 40–90 dB intensity combinations; fixed 75 ms duration, 500 ms ISI.

---

## Key parameter reference

| Parameter | Value |
|---|---|
| Sample rate | 16 kHz |
| Tone ramps | 5 ms linear rise/fall |
| dB reference | 94 dB SPL full-scale |
| Deviant variants per method | 15 (N ∈ {3,5,7} × 5 variations) |
| Whisper audio window | 30 s |
| All other models | 10 s |
| Train / held-out split | 18 / 6 methods (seed=42) |
| Unit selection threshold | top 5% by BH-FDR p-value |
| Bootstrap iterations | 100 (seed=12345) |
| Metric × window combos | 60 (5 × 3 × 4) |
