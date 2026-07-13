# Handoff — enable whisper-large + wav2vec2 for the mTRF D2 mapping

**Started:** 2026-07-12. **Completed:** 2026-07-13. Prereq for
`aux/prompt_24freq_7models_metric_selection.md` (the 24×7 screen).
Scope of THIS work: make whisper-large, wav2vec2-medium, wav2vec2-large usable by the delta-t
extractor + `eeg_mapping_sweep.py`, and hand back a chosen layer + held-out test r per model × level.
**No MMN stimuli / in-silico MMN / S2–S7 scoring here.**

## STATUS (2026-07-13): whisper-large ✅ done · wav2vec2 ⏳ sweeps running

- **whisper-large — COMPLETE and validated.** CV-chosen layer `blocks.21` for both levels; mean
  held-out test r = **+0.160 (parcels)**, **+0.180 (electrodes)** — above the whisper-small/medium
  ballpark, so it passes the sanity gate. Ready to use in the screen now.
- **wav2vec2-medium / wav2vec2-large — sweeps still running** (`{model}__{level}__D2.json` not yet
  written). **Interim: use `encoder.layers.0` as a placeholder `--layer`** for all four wav2vec2
  configs so the screen can proceed; swap in the real `chosen_layer` from the JSONs when the sweeps
  finish (4 values: {medium,large} × {parcels,electrodes}). See **Results** below.

Full numbers in the **Results** section.

**Deviations from the original plan** (see "Execution log" for details):
- **wav2vec2 D2 mapping uses 10 s windows at 10 s stride** (not 5 s). Features were extracted once at
  10 s / **5 s** and **reused** — the 10 s/10 s windows are exactly the even-offset subset (same
  waveforms → identical feature vectors + IDs), so no re-extraction was needed; only the EEG was
  rebuilt at 10 s/10 s and the sweep matches the subset by window ID.
- **wav2vec2 sweeps used `PCA_VAR=0.95`** (whisper used `pca_var=None`) to keep the wide-design mTRF
  tractable → wav2vec2 test r is **not strictly apples-to-apples** with the whisper ballpark. ⚠️
- A **prefetch job** was added (concurrent array downloads corrupt the shared HF cache), and every
  job runs **jed CPU only** (`--partition=standard`, 5 GB/CPU, no GPU).
- The layer sweep is **slow under sklearnex** (its patched RidgeCV lacks the `gcv_mode='eigen'`
  fast path); an `MBS_NO_SKLEARNEX` opt-out was added, then reverted at the user's request.

## Decisions locked in (as executed)
| Model | HF / whisper id | Layers | Features (window/stride) | mTRF window | EEG target | PCA |
| ----- | --------------- | ------ | ------------------------ | ----------- | ---------- | --- |
| whisper-large   | `large-v3` (128 mel, 32 blocks) | 32 | 30 s / 10 s | 30 s / 10 s | `surprisal_30s.h5` ✅ | `pca_var=None` |
| wav2vec2-medium | `facebook/wav2vec2-base` (pretrained) | 12 | 10 s / 5 s | **10 s / 10 s** | `surprisal_10s.h5` (10 s/10 s) ✅ | `PCA_VAR=0.95` |
| wav2vec2-large  | `facebook/wav2vec2-large` (pretrained) | 24 | 10 s / 5 s | **10 s / 10 s** | `surprisal_10s.h5` (10 s/10 s) ✅ | `PCA_VAR=0.95` |

- wav2vec2 = **pretrained** (self-supervised, no ASR fine-tuning).
- Whisper delta-t path for tiny/base/small/medium is **unchanged** — the wav2vec2 support is a
  parallel code path; existing whisper features stay reproducible.
- **wav2vec2 features are 10 s/5 s but the mapping is fit on 10 s/10 s** windows: the sweep matches
  features↔EEG by window ID, and the 10 s/10 s EEG IDs are the even-offset subset of the 10 s/5 s
  feature IDs — so the 5 s-stride windows at odd offsets are simply never matched (ignored).

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

## Everything runs as SLURM jobs — nothing on the login node

Four scripts, all `sbatch`-submittable. **Prefetch the model weights first** (single process) — the
extraction array must NOT be the first thing to download a model, or concurrent tasks corrupt the
shared `cache/model_weights` (HF file-locking is unreliable on `/work`). Array over-provisioning is
safe (a task past the last window no-ops), and the sweep script **merges the delta-t chunks itself**.
Chain stages with `--dependency` on the returned job ids.

| Stage | Script | Notes |
| --- | --- | --- |
| **Prefetch weights** | `scripts/slurm_prefetch_audio_models.sh` | CPU; **run FIRST**, one process; force-downloads all 3 models |
| Extract D2 features | `scripts/slurm_extract_delta_t_d2.sh` | CPU array (jed, `--partition=standard`); per-model `MODEL_ID/WINDOW_DUR/WINDOW_STRIDE` |
| Build 10 s EEG | `scripts/slurm_build_surprisal_10s.sh` | CPU; only needed for wav2vec2 |
| Layer sweep (merge + sweep) | `scripts/slurm_eeg_mapping_sweep_d2.sh` | CPU array over model × level; picks 30 s vs 10 s EEG automatically |

```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling

# 0. PREFETCH all model weights once (heals any cache already corrupted by a racing array).
PF=$(sbatch --parsable scripts/slurm_prefetch_audio_models.sh)
```

## Track A — whisper-large (UNBLOCKED, uses existing surprisal_30s.h5)
```bash
# 1. extract (CPU/jed), after prefetch. Over-provision the array freely — extra tasks no-op.
EXJ=$(sbatch --parsable --dependency=afterok:$PF \
      --array=0-19 --export=ALL,MODEL_ID=whisper-large,WINDOW_DUR=30,WINDOW_STRIDE=10,CHUNK_SIZE=16 \
      scripts/slurm_extract_delta_t_d2.sh)

# 2. sweep both levels after extraction finishes (merge is done inside the sweep job)
sbatch --dependency=afterok:$EXJ --array=0,1 scripts/slurm_eeg_mapping_sweep_d2.sh
```

## Track B — wav2vec2-medium & wav2vec2-large (10 s / 5 s; needs surprisal_10s.h5)

> **As-run note (2026-07-13):** the commands below are the original 10 s/**5 s** plan. The mapping was
> ultimately fit at 10 s/**10 s** (rebuild the EEG with `--export=ALL,WINDOW_STRIDE=10` on
> `slurm_build_surprisal_10s.sh`, keep the 10 s/5 s features, sweep with `PCA_VAR=0.95`). See the
> "Execution log" below.

```bash
# 1. build the 10 s EEG target (independent — can run any time)
B10=$(sbatch --parsable scripts/slurm_build_surprisal_10s.sh)

# 2. extract each wav2vec2 model (CPU/jed), after prefetch
EXM=$(sbatch --parsable --dependency=afterok:$PF \
      --array=0-19 --export=ALL,MODEL_ID=wav2vec2-medium,WINDOW_DUR=10,WINDOW_STRIDE=5,CHUNK_SIZE=32 \
      scripts/slurm_extract_delta_t_d2.sh)
EXL=$(sbatch --parsable --dependency=afterok:$PF \
      --array=0-19 --export=ALL,MODEL_ID=wav2vec2-large,WINDOW_DUR=10,WINDOW_STRIDE=5,CHUNK_SIZE=32 \
      scripts/slurm_extract_delta_t_d2.sh)

# 3. sweep (tasks 2-5 = wav2vec2 medium/large × parcels/electrodes) once features + 10 s EEG exist.
#    PCA_VAR tames the wide electrode sweeps.
PCA_VAR=0.95 sbatch --dependency=afterok:$B10:$EXM:$EXL --array=2-5 --export=ALL scripts/slurm_eeg_mapping_sweep_d2.sh
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

This is packaged as **`scripts/slurm_build_surprisal_10s.sh`** (CPU): it runs the formatter in the
temporal-analysis project's `.venv` at `10/5/50`, copies `surprisal_10s.h5` into the mbs clone's
`outputs/neural_data/`, and prints a sanity check. Before first submit, confirm `--data_root`/
`--audio_root` inside the script against the formatter's usage docstring
(`sed -n '14,25p' src/mbs/data_prep/format_eeg_hdf5_surprisal.py`). The sweep step (Track B) is what
consumes it.

## Results (delivered to the screen prompt)

Per model × level: CV-chosen layer + mean held-out test r, from
`outputs/results/eeg_mapping/{model}__{level}__D2.json`. Regenerate with:
```bash
cd /work/upschrimpf1/sigfstea/multimodal-brain-scaling
python - <<'PY'
import json, statistics, os
for M in ["whisper-large","wav2vec2-medium","wav2vec2-large"]:
    for L in ["parcels","electrodes"]:
        p=f"outputs/results/eeg_mapping/{M}__{L}__D2.json"
        if not os.path.exists(p): print(f"{M:16s} {L:11s} MISSING"); continue
        d=json.load(open(p)); tr=[x for x in d.get("test_r_chosen",[]) if x==x]
        m=statistics.fmean(tr) if tr else float("nan")
        print(f"{M:16s} {L:11s} chosen={str(d['chosen_layer']):18s} meanTESTr={m:+.3f} nT={len(tr)} pca={d.get('pca_var')}")
PY
```

| Model | Level | chosen_layer | mean test r | features dir |
| --- | --- | --- | --- | --- |
| whisper-large | parcels | `blocks.21` | **+0.160** | `outputs/features/whisper-large-delta-t-surprisal/merged` |
| whisper-large | electrodes | `blocks.21` | **+0.180** | `outputs/features/whisper-large-delta-t-surprisal/merged` |
| wav2vec2-medium | parcels | _pending_ → use `encoder.layers.0` | _pending_ | `outputs/features/wav2vec2-medium-delta-t-surprisal/merged` |
| wav2vec2-medium | electrodes | _pending_ → use `encoder.layers.0` | _pending_ | `outputs/features/wav2vec2-medium-delta-t-surprisal/merged` |
| wav2vec2-large | parcels | _pending_ → use `encoder.layers.0` | _pending_ | `outputs/features/wav2vec2-large-delta-t-surprisal/merged` |
| wav2vec2-large | electrodes | _pending_ → use `encoder.layers.0` | _pending_ | `outputs/features/wav2vec2-large-delta-t-surprisal/merged` |

**wav2vec2 placeholder:** the four wav2vec2 sweeps were still running at doc time. Until each
`{model}__{level}__D2.json` lands, pass **`--layer encoder.layers.0`** for the wav2vec2 in-silico runs;
then replace with the real `chosen_layer` (4 values) and re-run those models. whisper-large's numbers
above are final. (Regenerate the whole table with the command above.)

## Execution log (2026-07-13) — what actually happened

Scripts added during execution (all committed to `main`, jed/CPU, `--partition=standard`, 5 GB/CPU):
- `scripts/slurm_prefetch_audio_models.sh` — **run first**; single process force-downloads all 3
  models into `cache/model_weights`. Needed because a concurrent extraction array corrupted the
  shared HF cache (`config.json` JSONDecodeError) — HF file locking is unreliable on `/work`.
- `scripts/slurm_build_surprisal_10s.sh` — builds the Cortical Surprisal EEG at 10 s windows via the
  formatter in the **temporal-analysis** project (`.../multimodal-brain-scaling-temporal-analysis`,
  its own `.venv`; needs `PYTHONPATH=<that>/src`), then copies into the mbs clone. `--data_root` =
  the `cortical_suprisal_dataset` dir (P00–P12.h5), audio derived as `<data_root>/audiobooks`.
- `scripts/slurm_eeg_mapping_sweep_d2.sh` — array over the 3 new models × {parcels,electrodes};
  self-merges chunks (atomic lock) and auto-selects the 30 s vs 10 s EEG.
- `extract_features_delta_t.py` gained an over-provision guard (array tasks past the last window
  no-op instead of `IndexError`), so arrays can be sized generously.

Sequence that ran: prefetch → extract (whisper-large 30 s/10 s; wav2vec2 10 s/5 s) → build
`surprisal_10s.h5` → sweep. whisper-large completed with `pca_var=None`. wav2vec2 was first run at
10 s/5 s (very slow) then **switched to 10 s/10 s** to halve the mapping windows: the EEG was rebuilt
at 10 s/10 s (old one kept as `surprisal_10s_stride5.h5`), the existing 10 s/5 s features were reused,
and the sweeps re-run with `PCA_VAR=0.95`.

**Known operational gotchas (for future runs / the screen):**
- **sklearnex cripples the sweep.** `evaluation_helpers.py` unconditionally calls
  `sklearnex.patch_sklearn()`; its patched `RidgeCV` doesn't implement `gcv_mode='eigen'`, so it
  re-solves per (alpha × target) instead of one eigendecomposition → ~50–100× slower on the wide
  electrode designs. Stock scikit-learn is the reference implementation (same results). Left enabled
  per the user's choice; if a future sweep is intolerably slow, that's the first thing to disable.
- **numpy's BLAS is capped at 2 threads** (`OpenBLAS … MAX_THREADS=2` in this env), so 32-CPU sweep
  tasks only use ~2 cores for the linear algebra. Not fixed (needs a re-linked numpy); noted so nobody
  expects CPU count alone to help the sweep.
- **Comparability caveat:** wav2vec2 mappings differ from whisper's in two ways — 10 s/10 s vs 30 s/10 s
  windows, and `PCA_VAR=0.95` vs `pca_var=None`. Treat the whisper test-r ballpark below as a
  loose sanity gate for wav2vec2, not an exact comparison.

## Acceptance / sanity gate
`eeg_mapping_sweep.py` writes `chosen_layer` + `test_r_chosen` per model × level. Spot-check the mean
held-out test r is in the whisper ballpark (else the model/layer/features are broken — stop & flag):

| reference (existing) | parcels mean test r | electrodes mean test r |
| --- | --- | --- |
| whisper-small | +0.073 | +0.118 |
| whisper-medium | +0.078 | +0.079 |

Deliverable to the screen prompt: per model × level → chosen layer, mean test r, feature path.

## Cluster (jed / CPU only — no GPU)
All three scripts run on jed CPU (`--partition=standard`, `--mem-per-cpu=5G` = 1 CPU per 5 GB RAM,
`source env.sh`), matching `scripts/slurm_insilico_mmn.sh`. The extractor auto-selects CPU. Delta-t
cost = O(frames) full forward passes per window — the same structure as the whisper delta-t already
run CPU-only. wav2vec2-medium ≈ whisper-base on CPU (practical); wav2vec2-large / whisper-large are
heavier, so extraction runs 8 CPUs/task for 24 h and parallelizes across the array (raise `CHUNK_SIZE`
+ array size, or bump `--cpus-per-task`, if a task runs long). The sweep uses 32 CPUs/task.
