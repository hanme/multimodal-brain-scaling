# Handoff — enable whisper-large + wav2vec2 for the mTRF D2 mapping

**Date:** 2026-07-12. Prereq for `aux/prompt_24freq_7models_metric_selection.md` (the 24×7 screen).
Scope of THIS work: make whisper-large, wav2vec2-medium, wav2vec2-large usable by the delta-t
extractor + `eeg_mapping_sweep.py`, and hand back a chosen layer + held-out test r per model × level.
**No MMN stimuli / in-silico MMN / S2–S7 scoring here.**

## Decisions locked in
| Model | HF / whisper id | Layers | D2 window | EEG target |
| ----- | --------------- | ------ | --------- | ---------- |
| whisper-large   | `large-v3` (128 mel, 32 blocks) | 32 | **30 s / 10 s** | `surprisal_30s.h5` ✅ exists |
| wav2vec2-medium | `facebook/wav2vec2-base` (pretrained) | 12 | **10 s / 5 s** | `surprisal_10s.h5` ⚠️ must be built |
| wav2vec2-large  | `facebook/wav2vec2-large` (pretrained) | 24 | **10 s / 5 s** | `surprisal_10s.h5` ⚠️ must be built |

- wav2vec2 = **pretrained** (self-supervised, no ASR fine-tuning).
- Whisper delta-t path for tiny/base/small/medium is **unchanged** — the wav2vec2 support is a
  parallel code path; existing whisper features stay reproducible.

## Local code changes (done + statically verified on CPU)
- `src/mbs/extraction/modeling/backbones/audio_models.py`
  - `load_wav2vec2()` (base→medium, large→large; `Wav2Vec2Model` + `Wav2Vec2FeatureExtractor`,
    raw-waveform transform, `.eval()` to disable SpecAugment); routed via `load_model_audio`.
  - `load_whisper()`: `whisper-large` now pins **large-v3** (guarded; tiny–medium untouched).
- `src/mbs/extraction/extract_features_delta_t.py`
  - New **raw-waveform causal path** for wav2vec2 (`extract_delta_t_waveform`, `_truncate_waveform`,
    `_wav2vec2_norm_stats`, `_infer_wav2vec2_frame_count`). Mirrors the Whisper mel-truncation logic
    in the waveform domain: for frame `t`, keep samples `[0,(t+1)*320)`, silence the future,
    normalize with **full-window** stats, read encoder frame `t`. Whisper mel path is byte-identical.
  - Loader call switched `load_whisper` → `load_model_audio` (numeric no-op for whisper).
- `configs/extraction/audio/wav2vec2_medium_layers.json` (12), `wav2vec2_large_layers.json` (24)
  — hook paths `encoder.layers.{i}`, positions `i/(L-1)`.
- `scripts/slurm_extract_delta_t_d2.sh` — D2 delta-t extraction for all three new models.

**CPU smoke test** (`scratchpad/smoke_wav2vec2_deltat.py`, faithful stub, no network/GPU): frame-count
inference, causal truncation, hook resolution on `encoder.layers.{i}`, and `[T,d]` float16 finite
output all pass on CPU. The real weights need `transformers` (present on the cluster; not in local mbs-env).

## D2 stimuli data_root (recovered — confirmed by you on jed)
```
/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/data/cortical_suprisal_dataset/audiobooks
```
Stems in `surprisal_30s.h5`: train = AUNP01,03–08 + BROP01 + FLOP01–04; test = AUNP02, BROP02, BROP03.
This is the `--data_root` the extractor must use so window IDs (`{stem}_{start_sample}`) align with the EEG.

---

## Track A — whisper-large (UNBLOCKED, uses existing surprisal_30s.h5)
```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling && source env.sh

# 1. size the array (prints "Stimuli: N ..." = window count at 30 s / 10 s)
export MODEL_ID=whisper-large WINDOW_DUR=30 WINDOW_STRIDE=10
export DATA_ROOT=/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/data/cortical_suprisal_dataset/audiobooks
python -m mbs.extraction.extract_features_delta_t --model_id "$MODEL_ID" --data_root "$DATA_ROOT" \
  --target_feature_layers configs/extraction/audio/whisper_large_layers.json \
  --output_dir /tmp/probe --window_duration 30 --window_stride 10 --n_stimuli 1   # note N

# 2. extract (GPU recommended — 32 blocks; CPU works but slow). CHUNK_SIZE*(maxidx+1) >= N
export CHUNK_SIZE=16
sbatch --array=0-19 --export=ALL scripts/slurm_extract_delta_t_d2.sh

# 3. merge
mkdir -p outputs/features/whisper-large-delta-t-surprisal/merged
cp outputs/features/whisper-large-delta-t-surprisal/chunk_*/feats*.h5 \
   outputs/features/whisper-large-delta-t-surprisal/merged/

# 4. layer sweep (both levels) against the EXISTING 30 s EEG
for LVL in parcels electrodes; do
  python scripts/eeg_mapping_sweep.py --model_id whisper-large --target_level $LVL \
    --features_dir outputs/features/whisper-large-delta-t-surprisal/merged \
    --neural outputs/neural_data/surprisal_30s.h5 \
    --out outputs/results/eeg_mapping/whisper-large__${LVL}__D2.json
done
```

## Track B — wav2vec2-medium & wav2vec2-large (extraction UNBLOCKED; sweep BLOCKED on surprisal_10s.h5)

**Extraction** (same script, 10 s / 5 s):
```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling && source env.sh
export DATA_ROOT=/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/data/cortical_suprisal_dataset/audiobooks
for M in wav2vec2-medium wav2vec2-large; do
  export MODEL_ID=$M WINDOW_DUR=10 WINDOW_STRIDE=5 CHUNK_SIZE=32
  sbatch --array=0-19 --export=ALL scripts/slurm_extract_delta_t_d2.sh
done
# then merge each into outputs/features/${M}-delta-t-surprisal/merged/  (as in Track A step 3)
```

### ⚠️ BLOCKER: build `surprisal_10s.h5` (needed before the wav2vec2 sweep)
The mTRF aligns features↔EEG by exact 10 s window ID on a 50 Hz grid. `surprisal_30s.h5` is 30 s
windows and **will not align** with 10 s wav2vec2 features (the sweep would drop every row). You need
a 10 s / 5 s re-window of the **raw Weissbart Cortical Surprisal EEG** — not in this checkout (this
repo only has the pre-formatted 30 s h5).

**The raw pipeline was located** (2026-07-12) in the temporal-analysis project:
`/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis/`
- Raw EEG: `data/cortical_suprisal_dataset/P00.h5 … P12.h5` (13 subjects)
- Stimuli: `data/cortical_suprisal_dataset/audiobooks/{AUNP,BROP,FLOP}*.wav` + `stimulus_order.csv`
- Prep: `align_data.py`, `preprocess.py`
- **Formatter: a `format_eeg_hdf5_surprisal` module** (covered by `tests/test_format_eeg_hdf5_surprisal.py`)
  — this is what produced `surprisal_30s.h5`.

**Confirmed a one-line re-run** — `src/mbs/data_prep/format_eeg_hdf5_surprisal.py` (in the
temporal-analysis project) already exposes `--window_duration/--window_stride/--target_sr`
(surprisal_30s.h5 was built with `30.0 / 10.0 / 50`). IDs use 16 kHz sample offsets, so at
`window=10, stride=5` they align with the wav2vec2 feature windows. Keep `--n_test_parts 3 --seed 42`
(defaults) → same held-out parts as the 30 s file (test = AUNP02, BROP02, BROP03), so test r stays
comparable to whisper.

```bash
# Run in the TEMPORAL-ANALYSIS project (it owns the surprisal formatter + raw P*.h5).
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis
source .venv/bin/activate        # or: uv run --  (that project's env)
# confirm the exact --data_root/--audio_root against the usage example: sed -n '14,25p' src/mbs/data_prep/format_eeg_hdf5_surprisal.py
python -m mbs.data_prep.format_eeg_hdf5_surprisal \
  --data_root  data/cortical_suprisal_dataset \
  --audio_root data/cortical_suprisal_dataset/audiobooks \
  --output_path outputs/neural_data/surprisal_10s.h5 \
  --window_duration 10.0 --window_stride 5.0 --target_sr 50 \
  --n_test_parts 3 --seed 42 --overwrite true

# copy into the multimodal-brain-scaling clone so the sweep can read it:
cp outputs/neural_data/surprisal_10s.h5 \
   /work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/neural_data/

# sanity check:
python - <<'PY'
import h5py
f = h5py.File("outputs/neural_data/surprisal_10s.h5","r")
print("window_s:", f.attrs.get("window_duration_s"), "stride_s:", f.attrs.get("window_stride_s"))
for s in ("train","test"):
    node = f[s]["stimulus_ids"]
    ids = node[()] if not isinstance(node, h5py.Group) else node[list(node.keys())[0]][()]
    print(s, "windows:", len(ids), "e.g.", (ids[0].decode() if hasattr(ids[0],"decode") else ids[0]))
PY
```

**Then the sweep:**
```bash
for M in wav2vec2-medium wav2vec2-large; do
  for LVL in parcels electrodes; do
    python scripts/eeg_mapping_sweep.py --model_id $M --target_level $LVL \
      --features_dir outputs/features/${M}-delta-t-surprisal/merged \
      --neural outputs/neural_data/surprisal_10s.h5 \
      --out outputs/results/eeg_mapping/${M}__${LVL}__D2.json
  done
done
```

## Acceptance / sanity gate
`eeg_mapping_sweep.py` writes `chosen_layer` + `test_r_chosen` per model × level. Spot-check the mean
held-out test r is in the whisper ballpark (else the model/layer/features are broken — stop & flag):

| reference (existing) | parcels mean test r | electrodes mean test r |
| --- | --- | --- |
| whisper-small | +0.073 | +0.118 |
| whisper-medium | +0.078 | +0.079 |

Deliverable to the screen prompt: per model × level → chosen layer, mean test r, feature path.

## CPU vs GPU
The extractor auto-selects device (`cuda` if available, else CPU) and does not hard-fail on CPU.
Delta-t cost = O(frames) full forward passes per window — same structure as the whisper delta-t you
already ran CPU-only. wav2vec2-medium ≈ whisper-base on CPU (practical); wav2vec2-large is ~4× heavier
per frame → prefer GPU or use more/smaller array tasks. For a CPU job: comment out `--partition`/
`--gres` in `slurm_extract_delta_t_d2.sh` and raise `--cpus-per-task`.
