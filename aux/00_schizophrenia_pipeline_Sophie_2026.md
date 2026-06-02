# Sophie Sigfstead's Schizophrenia Pipeline (2026)

**Repo location:** `/work/upschrimpf1/sigfstea/scz_updated_pipeline_071226/`

The project studies whether audio AI models produce MMN-like (mismatch negativity) neural responses.
MMN is the brain's automatic response to an unexpected "deviant" sound in a stream of repeated "standard" tones,
and is robustly impaired in schizophrenia patients. The goal is to identify which internal units of audio models
produce the most MMN-like signal, and then search for stimuli that maximally drive those units ("supernatural MMN").

There is also an earlier prototype version at `/work/upschrimpf1/sigfstea/unit_identification_pipeline/`
(summer 2025, no time axis, single deviant per method, no counterbalancing). The updated pipeline is the one to use.

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
