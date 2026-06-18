# Temporal EEG Encoding — Project Plan (2026-06-11)

Author: H. Mehrer, after the 2026-06-11 conversation with Kadir Gökce.
Scope: how to predict EEG from audio-model features over time in `multimodal-brain-scaling`
(auditory adaptation of Kadir's vision pipeline). This plan supersedes the ad-hoc per-time-bin
approach currently in `evaluate_features_temporal.py`.

---

## 0. TL;DR / the decision

The current per-time-bin Ridge readout, the classical mTRF, and Kadir's learned attention
probe are **three points on one spectrum** — they all predict `EEG[time, channel]` from model
features and differ only in how the **time axis of the readout** is handled:

| | Readout weights | Lookback (lags) | Fit | Score axis | Status here |
|---|---|---|---|---|---|
| **(1) Per-bin Ridge** | separate per time bin | none (zero-lag) | closed-form | across stimuli @ fixed offset | retired (ERP/MMN-only) |
| **(2) mTRF** | one set, shared across time | window (e.g. 50–800 ms) | closed-form ridge | along time | ✅ **DONE** — `evaluate_features_mtrf.py`, held-out-validated (2026-06-13) |
| **(3) Learned temporal probe** | shared trunk + subject heads | window | gradient (SGD) | along time | ✅ built + benchmarked (2026-06-13) — **loses to mTRF at this data scale**; §11 |

**Decision:** retire (1) as a primary metric; (2) mTRF is **built and validated** (the current
method); now adapt the existing `attn_probe/` into (3), Kadir's current method (MIRAGE,
Gökce/Al-Khamissi 2026), which reportedly beats ridge. The three methods we will compare and
make **selectable** are: **(2) mTRF (current), (3a) probe / per-subject heads (Kadir individual),
(3b) probe / single group head (Kadir group)**. See §3.

Why (1) is wrong for continuous speech:
- **Zero-lag.** It fits `feature[t] → EEG[t]` at the same bin; auditory cortex lags the
  stimulus ~50–200 ms and responds as a temporal spread (a convolution), not instantaneously.
- **Wrong score axis.** `evaluate_features_temporal.py:78` correlates predicted vs actual
  *across stimuli at a fixed within-window offset t*. With 30 s / 10 s overlapping windows,
  offset `t` is an arbitrary absolute moment, not a shared condition. The standard encoding
  score correlates **along the time axis** of held-out runs.
- (1) is acceptable *only* for discrete ERP/MMN epochs, where per-latency-across-stimuli is the
  evoked waveform. Keep it scoped to that, not continuous speech.

---

## 1. Where things stand (verified in code)

- **Features.** `extract_features_delta_t.py` produces **causal** time-resolved features:
  `feature[t]` sees audio only up to `t` (mel frames ≥ `2(t+1)` set to silence). Whisper:
  3000 mel frames → 1500 encoder bins = 50 Hz. Output HDF5 `features/<layer> = [n_stim, T, d]`.
  *Whisper-only right now* (`load_whisper`); other backbones not yet wired (see §6).
- **EEG.** `format_eeg_hdf5.py` (Broderick 2018, ds004408; 19 subj, 128-ch, group-averaged):
  30 s / 10 s windows, resampled to `target_sr=50` Hz → `neural_data = [n_stim, T=1500, n_ch]`.
  Time axes of features and EEG therefore **match** (both 50 Hz, 1500 bins). EEG is cropped to
  audio onset (zero stimulus→EEG delay built in). Train/test split is **by run** (`test_runs`
  held out) — good, avoids overlapping-window leakage between splits.
- **Noise ceiling.** Already **per-time-point and per-channel**: `[T, n_ch]`, cross-subject
  split-half + Spearman-Brown, stored as % variance (`max_nc=100`, recover r via `sqrt(nc/100)`).
- **Current temporal eval.** `evaluate_features_temporal.py` = method (1) above. Fits an
  independent `RidgeCV` per bin, `feature[:,t,:] → EEG[:,t,:]`, scores across stimuli per bin.
- **attn_probe.** `src/mbs/evaluation/attn_probe/` is the seed of method (3): shared
  `LatentAttentionTrunk` + `SubjectHeadBank` (one linear head per subject), AdamW, MSE loss,
  `RunningPearson` metric. **But it is still vision-shaped:** `dataset.py` loads target
  `y = [n_stim, N_subject]` (one vector per stimulus, no time axis), and the trunk tokenizes
  over patches/`N`, not time. It does **not** yet ingest a time-resolved EEG target.

### 1b. What Kadir's repos actually did (for the record)

- **Gökce 2024 / `_Kadir_orig`** — when targets are temporal `[n_stim, T, n_ch]`, he **flattens
  T into the output** and fits **one** ridge from a **single per-stimulus feature vector**
  (`evaluate_features_all_layers.py:167-178`). So the *output* is per-time-and-channel, but the
  *features* have no time axis and there are no per-bin models. He also **masks channels by NC**
  before fitting (`noise_ceiling > 0.1`).
- **Gökce/Al-Khamissi 2026 (MIRAGE) / `ambitious-brain-model`** — the relevant precedent:
  time-resolved features → attention layer-aggregator → fusion → **temporal transformer** →
  **subject-specific readout heads**, trained by SGD, **scored by Pearson over all time points**
  (paper §§ lines 80–113). Headline: the temporal encoder gains "beyond what ridge regression
  can capture." `attn_probe/` is the seed of this.

### 1c. Kadir's verbal guidance (2026-06-11), decoded

- NCs are **per channel and per time point**; **average over time only** to collapse to a single
  value (e.g. for layer ranking).
- **Fit over the 50–800 ms** lag/latency window (the "look back a few time points").
- **Exclude low-NC channels before fitting.**
- **No repeats →** cross-subject NC is the right fallback; ideally also bring in a dataset with
  repeats for a within-subject ceiling (he named "Aude" as a contact/source).
- **Higher explained variance from the iterative (gradient) fit**, not the closed-form ridge.
- **One readout shared across time**, applied at all time points, **but randomly sample time
  points during training** so the readout can't overfit the strong temporal autocorrelation.
  (Same idea as his "train only on future" leakage caution.)
- **One head per subject is fine** — `SubjectHeadBank` already supports it.

---

## 2. Workstream A — closed-form mTRF baseline ✅ DONE

`src/mbs/evaluation/evaluate_features_mtrf.py` (the old `evaluate_features_temporal.py` is
re-scoped to ERP/MMN only). For each output time `t`, stack features over a lag window
`L={l_min…l_max}` and predict `EEG[t]` from `[feature[t−l] for l in L]` with one **shared**
`RidgeCV(alpha_per_target=True)` across all `(segment, t)` rows. Score: predict held-out runs,
concatenate over time, correlate **along time** per channel. NC-mask channels before fitting;
random time-point subsampling per segment; 0.5 Hz high-pass to de-confound slow drift. Built,
tested (`tests/test_evaluate_features_mtrf.py`, 11 tests), run across cutoffs × 6 layers, and
**held-out-validated** (built-in run-level split; `insilico_mmn.py:evaluate_heldout`, 2026-06-13).
Full results and design log in §11. The reusable pure functions (`lags_in_bins`,
`sample_time_indices`, `build_lagged_design`, `pearson_along_time`, `mask_channels`,
`highpass_along_time`, `_standardize`) are imported directly by Workstream B (§3).

---

## 3. Workstream B — learned temporal probe (DETAILED PLAN, locked 2026-06-13)

Adapt the existing `attn_probe/` (vision: one vector per stimulus) to time-resolved EEG =
method (3) / MIRAGE-style. **Decisions locked with Hannes (2026-06-13):**
1. Build **per-subject heads first** (Kadir individual level), **then** the single-group-head
   variant (closer to the mTRF, fully comparable) — in that order. The three methods
   {mTRF / probe-individual / probe-group} must be **easily selectable** (see §3.6).
2. Predict the **same 4 NC-parcels** as Method A (frontal/temporal/parietal/occipital).
3. **Encoding comparison only** for now; MMN integration is planned but deferred (§3.7).
4. **Run all 6 whisper-base layers.**

### 3.0 The core reframing (why this is cheap)

Method A predicts `EEG[t]` from a **flattened** lag stack `[feature[t−l] for l in L]` → `[L·d]`
→ linear Ridge. Method B predicts the same `EEG[t]` from the **same lookback window kept as a
sequence** `[L, d]` → attention trunk → head. **The lookback window IS the token sequence**
(tokens = lag/time positions). This maps onto existing infrastructure with almost no new
architecture:
- `TokenAdapter` 3-D path ([model.py:61-62](../src/mbs/evaluation/attn_probe/model.py#L61))
  passes `(B,N,C)` straight through as tokens → set `N = lookback length`, `C = d_feature`.
- `SinusoidalPosEnc` / `LearnedPosEnc` ([model.py:96-135](../src/mbs/evaluation/attn_probe/model.py#L96))
  become the temporal position encoding over lag positions (MIRAGE uses learned absolute).
- `LatentAttentionTrunk` ([model.py:205](../src/mbs/evaluation/attn_probe/model.py#L205)):
  window → latents → flatten. **Unchanged.**
- `LinearHead` → `n_parcel`; `SubjectHeadBank`
  ([model.py:319](../src/mbs/evaluation/attn_probe/model.py#L319)) is the per-subject mechanism
  (group variant = a single `"group"` head). **Unchanged.**

So Method B = Method A with the linear FIR replaced by a learned attention-over-lags. The real
work is the **dataset** (windowed samples), the **engine** (Pearson-over-time loss + along-time
eval + our EEG loader), and a **driver** that mirrors `evaluate_features_mtrf.py:main` so outputs
are directly comparable.

### 3.1 Reuse vs. build

**Reuse unchanged** — from `attn_probe/model.py`: `LatentAttentionTrunk`, `TokenAdapter`,
`Sinusoidal/LearnedPosEnc`, `LinearHead`/`LowRankHead`, `SubjectHeadBank`, `SingleRoiProbeSystem`,
`ProbeConfig`. From `attn_probe/engine.py`: `set_seed`, `build_lr_scheduler`, the
AdamW/AMP/grad-accum skeleton. From `attn_probe/metrics.py`: `RunningPearson`. From
`evaluate_features_mtrf.py`: `lags_in_bins`, `sample_time_indices`, `highpass_along_time`,
`pearson_along_time`, `mask_channels`, `_standardize`. Parcel construction + NC loading from
`insilico_mmn.py:load_split_parcels`.

**Build new (4 files + 1 SLURM):**
1. `attn_probe/dataset_temporal.py` — windowed temporal dataset (see §3.3).
2. `attn_probe/engine_temporal.py` — train + along-time eval; 1−r loss; random-time sampling.
3. `evaluation/evaluate_features_attn_probe_temporal.py` — CLI driver, structured to mirror
   `evaluate_features_mtrf.py:main` so the output h5 is parallel.
4. `tests/test_attn_probe_temporal.py` — synthetic lag-recovery + shape/causality/leakage asserts.
5. `scripts/slurm_attn_probe_temporal.sh` — GPU array (one task/layer), `--array=0-5`.

### 3.2 Data prep prerequisite — per-subject EEG (NEW, gates 3a)

The current `outputs/neural_data/broderick2018_30s.h5` stores **only** `subjects=['group']`
(19-subject average, each ROI `[n_stim, T, 1]`). The **per-subject (Kadir individual)** variant
needs per-subject EEG + per-subject NC stored. Step **B-data**: extend
`format_eeg_hdf5.py` with a `--store_subjects` mode that writes `neural_data/<sub-XXX>/<roi>` for
all 19 subjects (keep `group` too), plus a per-subject NC (cross-subject NC is undefined per
individual → use that subject's own split-half across its runs if available, else fall back to
the group NC for masking only; settle in §3.6/§4). Group variant (3b) needs **no** re-extraction.
*Confirm the formatter's current averaging path before coding this.*

### 3.3 Data flow (concrete shapes)

Per layer, reusing the mTRF / insilico loaders:
```
feats  [n_stim, T=1500, d=512]   delta_T whisper-base, standardized w/ TRAIN mu/sd
eeg    [n_stim, T=1500, P=4]     NC-masked parcels, high-passed 0.5 Hz   (per subject OR group)
```
Windowing (in the dataset; mirrors `build_lagged_design` but **keeps the lag axis**):
```
sample (stimulus s, output-time t):  X = feats[s, t−L+1 : t+1, :]  → [L, d]   (window = tokens)
                                      Y = eeg[s, t, :]              → [P]
```
- `L` = lookback length in bins (0–800 ms → 40 bins; matches Method A's window).
- **Causality preserved:** look back only; `delta_T` features are already causal.
- **Random time sampling:** `sample_time_indices(T, L, n_train_time_samples, rng)` per stimulus
  per epoch (Kadir's autocorrelation fix — already implemented).
- **Batching:** flatten (stimulus × sampled-times) → `[B, L, d]` → trunk → `[B, P]`.

### 3.4 Eval (along time, Kadir-style — identical metric to A)

For each held-out window predict all valid `t` in order, concatenate predictions over time across
the test windows, `pearson_along_time(Y, Yhat)` per parcel → **the same `heldout_r` Method A
reports.** Per-subject variant: average r over subjects (engine already does this). This is the
apples-to-apples comparison.

### 3.5 Output — parallel to Method A

`attn_probe_temporal_scores.h5`, schema deliberately parallel to `mtrf_scores.h5` /
`predictions__<layer>.h5`:
```
attrs: layer, readout_level {individual|group}, lookback_ms, highpass_hz, fs, nc_threshold,
       loss, probe_cfg_json, n_params, equivalent_linear_params
parcels      [4]    parcel_nc_r [4]
heldout_r    [4]    heldout_r_nc [4]          ← SAME metric as mTRF/insilico
per-subject (3a):  heldout_r_persubj [n_subj, 4]   (then averaged → heldout_r)
+ summary JSON: {layer, readout_level, parcel, heldout_r, heldout_r_nc}
```
Deliverable = a **third/fourth column** beside Method A in the handover tables: mTRF r vs.
probe-individual r vs. probe-group r, per parcel × layer. That comparison is the scientific
payoff ("does the learned temporal encoder beat ridge, as MIRAGE claims").

### 3.6 Method selectability (a first-class requirement)

The eventually-selected method (mTRF / probe-individual / probe-group) must be trivially
switchable for everything downstream (handover tables, and later the in-silico MMN). Design:
- **Within the probe:** a single `--readout_level {individual, group}` flag toggles
  `SubjectHeadBank` over 19 subjects vs. one `"group"` head — same code path, same output schema.
- **Across methods:** keep the **output contract identical** (`heldout_r`/`heldout_r_nc` per the 4
  parcels, same parcel order). Then a thin `--method {mtrf, probe_individual, probe_group}`
  selector at the *consumer* layer (handover/figure scripts now; `insilico_mmn.py` later) reads
  whichever h5 without branching on internals. No unified fit-API needed (closed-form vs gradient
  genuinely differ) — unify the **interface and the output**, not the solver.

### 3.7 MMN integration (PLANNED, deferred — do not build yet)

Once a method is selected, wire it into `insilico_mmn.py` as an alternative **mapping** behind the
existing parcel/forward-model interface:
- `insilico_mmn.py` currently calls `fit_mapping()` → a RidgeCV; refactor so the mapping is one of
  `{mtrf_ridge, probe}` chosen by `--method`. The probe path loads a trained
  `SingleRoiProbeSystem` (per layer) and, at inference, slides the lookback window over each MMN
  stimulus to produce the predicted parcel-EEG time course (`standard`, `deviant_mean`, `deviants`)
  exactly as the ridge does now — the downstream MMN metric (baseline-normalised negative peak,
  deviant−standard) is unchanged.
- Deliverable then: the 8-frequency-pair in-silico MMN re-run through the selected probe, compared
  to the mTRF MMN. **Gate:** only worth doing if the probe clearly beats/ties the mTRF in §3.4.
- Keep the probe **causal at MMN inference** (lookback only) — same guardrail as training.

### 3.8 Phasing

- **B-data:** add per-subject EEG to the h5 (`--store_subjects`). Gates 3a only.
- **B0 (smoke):** synthetic lag-recovery test + blocks.2 overfit-a-tiny-subset on GPU. Confirms
  window/trunk/head wiring and that 1−r loss trains.
- **B1 = 3a (Kadir individual):** per-subject heads, 4 parcels, all 6 layers → held-out r.
- **B2 = 3b (Kadir group):** single group head, 4 parcels, all 6 layers → held-out r. Direct
  apples-to-apples vs mTRF (both on group EEG).
- **B3:** comparison table {mTRF, probe-individual, probe-group} × 6 layers × 4 parcels into the
  handover; pick the method.
- **B4 (later, gated):** MMN integration (§3.7).

### 3.9 Compute

Gradient-trained → **GPU** (Hannes can switch clusters; the CPU `standard` partition has no GRES).
One small probe per (layer, readout_level); 252 train windows × ~200 sampled times is tiny.
Expect minutes/layer on a single GPU. `scripts/slurm_attn_probe_temporal.sh` requests 1 GPU,
`--array=0-5`.

---

## 4. Noise ceiling

- ⚠️ **FLAG (2026-06-12, raised by Hannes): the exact NC definition still needs to be discussed.**
  The current implementation ([format_eeg_hdf5.py:223](../src/mbs/data_prep/format_eeg_hdf5.py#L223))
  splits the **19 subjects** (not stimulus repeats) into two halves, averages EEG within each half,
  then correlates the two half-group averages **across the 30 s windows at each fixed within-window
  time-bin** (per channel), Spearman-Brown corrected. Two points to settle later: (a) the unit being
  halved is subjects — fine for 1-rep continuous speech, but worth stating explicitly vs. the visual
  benchmarks' (likely within-subject) ceiling; (b) correlating *across windows at a fixed latency*
  is a defensible but specific choice — confirm it's the quantity we want. **For now this is fine;
  we proceed with it and revisit before any NC-corrected numbers are reported.**
- **Keep** the current per-time-point, per-channel **cross-subject** split-half ceiling — it is
  the correct fallback for 1-rep-per-subject data (Kadir confirmed).
- **Caveat for comparability:** the visual benchmarks may use a *within-subject* (repeats)
  ceiling. A cross-subject ceiling is a different (generally more conservative) quantity, so
  NC-corrected auditory numbers are **not** on the same scale as the visual ones. Report **raw r
  alongside NC-corrected r**, and state the ceiling type explicitly.
- **Next step:** add one dataset *with repeats* (e.g. the natural-sounds 43–50-rep set, or
  Di Liberto Bach 3-rep) to obtain a proper within-subject ceiling and cross-check.
- Average NC over the **scored latency window** when collapsing to a scalar — not over all bins.

---

## 5. Data / leakage guardrails

- Train/test split **by run** (already done) prevents overlapping-window leakage across splits.
- With overlapping 30 s / 10 s windows, the same absolute time appears at 3 offsets — fine
  within a split, but never let the same absolute time land in both train and test (run-level
  split guarantees this).
- `delta_t` features are causal — preserve that; do not introduce future-leaking lags
  (lags should look **back**, l ≥ 0, matching the forward TRF and Kadir's caution).

---

## 6. Known gaps to close before scaling

- **Non-Whisper backbones.** `extract_features_delta_t.py` is Whisper-only. wav2vec2 (~49 Hz),
  AST (patch grid), VGGish (~0.96 s frames) have **different native time resolutions** → each
  needs resampling to the EEG 50 Hz grid (MIRAGE resamples all streams to a common grid). Plan
  a per-model time-alignment step before B scales to all models.
- **Multi-dataset pooling** (separate, partly orthogonal track). The D1+D2 confound (model
  predicts "which dataset" from amplitude scale) needs: per-dataset per-channel z-scoring on
  train stats, within-dataset test scoring, and a shared-ROI/channel harmonization story across
  montages. Defer until single-dataset temporal method is settled.

---

## 7. Open decisions

**Resolved (A):** lags 0–800 ms @ 20 ms; per-dim feature z-score on train stats; NC-mask before
fit; 0.5 Hz high-pass; uncorrected r reported (Kadir-sanctioned). **Resolved (B, 2026-06-13):**
4 NC-parcels; per-subject heads first then group; loss = 1−r; all 6 layers; encoding-only first;
selectable methods (§3.6).

**Still open:**
1. **Probe capacity (B):** `d_model`, `num_latents`, `cross_attn_layers`, `pos_mode`
   (`learned` vs `sin`), lookback `L`. Start small (the data is tiny: 252 windows); tune in B0.
2. **Per-subject NC (3a):** no cross-subject NC exists per individual — use that subject's
   own within-subject split-half if runs allow, else group NC for masking only (§3.2/§4).
3. **Exact NC definition** (⚠️ §4): across-window-at-fixed-latency correlation vs alternatives;
   comparability vs visual benchmarks. *Fine for now; settle before reporting NC-corrected numbers.*

---

## 8. Milestones

- **M1 ✅** Workstream A mTRF baseline on Broderick/Whisper-base — de-confounded r-vs-latency +
  layer ranking (mid-depth ~blocks-3 raw; blocks.2 adopted for downstream after the in-silico
  MMN + held-out eval, 2026-06-13).
- **M2 ✅** `evaluate_features_temporal.py` re-scoped to ERP/MMN-only; mTRF is the default
  continuous-speech metric; held-out validation added.
- **M3 ✅ (2026-06-13)** Workstream B (§3): probe built + 17 tests + B0 sweep + B2 group probe on
  all 6 layers. Comparison done — **mTRF beats the probe in 23/24 cells; method selected = mTRF.**
  B1 (probe-individual) and B4 (probe MMN) deferred (probe doesn't beat ridge at this data scale).
  See §11 (2026-06-13 GPU-node entry). **Encoding only.**
- **M4** MMN integration of the selected method (§3.7); then scale B across whisper sizes /
  wav2vec2 (after §6 backbone time-alignment).

---

## 9. Risks

- Learned probe (B) may not beat ridge at this data scale (256 train segments) — mTRF (A) is the
  safe fallback and may be the reportable method.
- Cross-subject NC inflates NC-corrected scores vs visual benchmarks — mitigate by reporting raw
  r and adding a repeats dataset (§4).
- Time-alignment across non-Whisper backbones (§6) is fiddly and gates multi-model scaling.

---

## 10. References

- `docs/literature/Gokce_2024.txt` — vision pipeline this repo adapts.
- `docs/literature/Gokce_AlKhamissi_2026.txt` — MIRAGE; the temporal-encoder + subject-head
  method that Workstream B targets.
- `ambitious-brain-model/neurips_26/` — Kadir's current implementation of MIRAGE.
- Crosse et al. (mTRF toolbox); Di Liberto/Lalor/Broderick — forward-TRF methodology for the
  Broderick dataset used here.

Repo relationship clarified 2026-06-11: `ambitious-brain-model` (ABM) is the big working
monorepo (888 tracked files); `mirage` (github.com/epflneuroailab/mirage, 97 files) is the
trimmed **public release** of the same work and is embedded in ABM as a (currently unfetched)
submodule. The core model architecture is identical in both `models/` dirs — read `mirage` for
the clean model definition. ABM is a strict superset: mine its
`neurips_26/src/brain_enc/analysis/noise_ceiling.py`, `cli/compute_noise_ceiling.py`,
`cli/linear_baseline.py`, `cli/make_linear_probe_staircase_figure.py` for the NC and
linear-vs-learned-readout methodology (directly relevant to our §4 NC question and the
Workstream A-vs-B comparison).

---

## 11. Progress log

### 2026-06-11 — Workstream A built + first run submitted

**Implemented** (committed to working tree, not yet git-committed):
- `src/mbs/evaluation/evaluate_features_mtrf.py` — lagged shared-weight Ridge (mTRF).
  Two modes: `single_lag` (default; r-vs-lag curve, the minimal correct fix to the per-bin
  method) and `fir` (full multi-lag mTRF, one r/channel, optional `--feature_pca`).
- `tests/test_evaluate_features_mtrf.py` — 11 tests, all pass. Includes a **synthetic
  lag-recovery** test (curve peaks at a planted stimulus→EEG lag) and an **alpha-per-target
  == separate-fit** regression test.
- `scripts/slurm_mtrf.sh` — array job (one task per layer + 1 diagnostic).

**Key design decisions locked** (all CLI flags):
- Lags 0–800 ms @ 20 ms (Kadir's 50–800 is the narrower fitting window).
- Features z-scored per dim on train stats.
- **Random time-point subsampling per segment** for training (`--n_train_time_samples 200`) —
  the *fitting*-side autocorrelation fix; kept DISTINCT from the significance-test n_eff
  correction in `scripts/plot_score_distributions.py`.
- **Fit all electrodes at once** per (layer, lag) with `RidgeCV(alpha_per_target=True)`. Proven
  (test + numeric check) identical per-channel to separate fits → no low-SNR contamination of
  high-SNR channels; ~67× fewer SVDs than the per-ROI loop. ROIs = channel subsets at reporting.
  NOTE: this no-pooling guarantee holds for Workstream A's independent ridge ONLY; the
  Workstream B shared trunk genuinely pools across channels — SNR-mixing concern returns there.
- Channels with mean NC ≤ `--nc_threshold` dropped before fitting; scores NC-corrected by
  dividing r by per-channel time-mean NC.

**Pilot finding (blocks.2, 5 electrodes, 0–400 ms @ 40 ms) — IMPORTANT:**
- Encoding magnitude healthy: NC-corrected peak r Fz 0.68, FT7 0.55, AF3 0.52 (raw ≈ 0.46–0.52).
- **The r-vs-lag curves are essentially FLAT** (raw span ≈ 0.02–0.03 over 0–400 ms). The
  reported "best_lag" is the argmax of a flat noisy curve → **not meaningful yet.**
- Likely cause: (a) delta_T features are temporally smeared (Whisper's wide receptive field —
  `feature[t]` already integrates seconds of past), and (b) features + group-averaged EEG are
  dominated by slow envelope structure autocorrelated over >400 ms, so every lag predicts
  similarly. Classic TRF pitfall with slow regressors.
- **Implication:** scaling a flat curve to 67 ROIs × 6 layers just gives flat curves; the
  encoding-vs-latency / MMN-latency story is NOT resolvable from continuous speech as-is until
  we test the slow-autocorrelation confound.

**Running now — SLURM job 55037406 (array 0–6), submitted 2026-06-11 ~16:10:**
- tasks 0–5: full `single_lag` sweep, layer_id = task, ALL electrodes, lags 0–800 @ 20 ms,
  no high-pass → `outputs/results/whisper-base-mtrf-full/layer_<t>/`
- task 6: **diagnostic** — layer 2, `--highpass_hz 1.0`, same electrodes/lags →
  `outputs/results/whisper-base-mtrf-hp1/layer_2/` (does high-passing un-flatten the curve?)
- Each task writes its OWN dir (concurrent writers to one HDF5 corrupt it). ~5–10 min/task.

**Pick up here (next session):**
1. Check job: `sacct -j 55037406 --format=JobID,State,Elapsed` and `logs/mtrf_55037406_*.{out,err}`.
2. **Compare flat (mtrf-full/layer_2) vs high-passed (mtrf-hp1/layer_2) curves at Fz/T7/AF3.**
   - If high-pass produces a lag PEAK → slow-autocorrelation was the culprit; adopt high-pass
     (and/or feature derivative) and the latency story is recoverable on speech.
   - If still flat → latency is not resolvable from continuous speech; defer the latency/MMN
     question to the discrete MMN paradigm (Phase 6), and use the `fir` single-r number as the
     Workstream A deliverable for the dataset-size/encoding questions.
3. Add a `scripts/plot_mtrf_scores.py` (r-vs-lag per layer × electrode; layer ranking table).
4. Then decide: scale to whisper sizes / wav2vec2, or move to Workstream B.

**Open risk:** the flat-curve result may mean Workstream A's *latency* output is uninformative
on speech even though its *magnitude* (fir mode) is fine. This does not block the
dataset-size/encoding questions, but it does affect how we select layers and the MMN claim.

### 2026-06-12 — Job 55037406 results: the flatness was the slow-autocorrelation confound

All 7 tasks COMPLETED clean (~6 min each, ~5–6 GB, 66 electrodes scored/layer). The
whole-brain `alpha_per_target` refactor works — all electrodes × 41 lags in ~6 min.

**Headline (figure: `outputs/figures/mtrf_highpass_diagnostic.png`):**
- **No high-pass:** raw r flat ~0.45–0.52 across 0–800 ms (span ~0.02) → lag uninformative.
- **High-pass 1 Hz:** clean **unimodal TRF peaks at ~120–200 ms**, decaying to ~0 by ~700 ms.
  Temporal electrodes (TP7/T7/FT7) peak earliest (~120 ms) and strongest (r ≈ 0.10–0.13);
  frontal/parietal (Fz/Pz) later/lower. Physiologically correct auditory response.
- **Conclusion: latency IS recoverable on continuous speech, but only after high-passing.**
  Removing slow drift is mandatory for any latency/MMN analysis.

**Two consequences that block naive scaling:**
1. **Layer selection on raw (no-HP) r is invalid.** Raw r decreases monotonically with depth
   (blocks-0 ≈ 0.50–0.55 → blocks-5 ≈ 0.30) — it just rewards the most envelope-like (earliest)
   layer via the slow confound. This both contradicts the old per-bin "blocks.2" result and is
   not trustworthy. **Redo layer selection on high-passed data** (only layer 2 has HP so far).
2. **NC-correction is inconsistent under high-pass.** The stored NC was computed on raw EEG; a
   high-passed signal has a different noise ceiling. All 2026-06-12 numbers above are RAW r.
   Before reporting NC-corrected high-passed scores, **recompute NC on high-passed group-halves**
   (re-run format_eeg_hdf5 with a high-pass, or compute NC post-filter).

**Pick up here (2026-06-12):**
1. **Re-run the full sweep WITH high-pass for all 6 layers** (`slurm_mtrf.sh`: set HIGHPASS=1.0
   for tasks 0–5, new out dir). Then layer-select on the de-confounded peak r → the real
   "which layer at which latency" table.
2. **Sensitivity check on the cutoff** (0.5 / 1 / 2 Hz): 1 Hz is aggressive (eats the delta band
   where envelope tracking lives); a gentler cutoff may retain more genuine signal.
3. **Recompute NC on high-passed EEG** so NC-corrected latency scores are valid.
4. Plotter added: `scripts/plot_mtrf_scores.py`.

### 2026-06-12 (later) — Job 55039297: high-pass cutoff sweep (6 layers × 0.5/1/2 Hz)

All 18 tasks COMPLETED clean (~6 min each). Uncorrected r (Kadir-sanctioned; sidesteps
NC-on-filtered). `slurm_mtrf.sh` now does the 18-task cutoff×layer sweep. Outputs in
`outputs/results/whisper-base-mtrf-hp{0p5,1p0,2p0}/layer_<0..5>/`.
Figure: `outputs/figures/mtrf_cutoff_layer_summary.png`.

**Peak uncorrected r @ lag, mean over auditory electrodes [TP7,T7,FT7,T8,AF3,Fz,Pz]:**
```
cutoff |   L0          L1          L2          L3          L4          L5
0.5Hz  | .104@100   .114@120   .119@120   .124@120   .119@140   .120@180
1.0Hz  | .079@120   .080@120   .084@140   .089@140   .084@140   .082@160
2.0Hz  | .077@120   .076@120   .076@140   .077@140   .074@120   .072@160
```

**Findings:**
1. **Cutoff:** the latency peak is ROBUST (~120-140 ms) across all three cutoffs — only the
   magnitude changes. 0.5 Hz retains the most signal (peak ~0.12 mean / up to 0.18 at FT7);
   2 Hz is over-aggressive. **Adopt 0.5 Hz** (de-confounds while keeping the delta band).
2. **Layer (de-confounded):** mid-to-deep layers (L2-L5) are ~tied and clearly beat the early
   layers (L0/L1); **blocks-3 is marginally but consistently best across all three cutoffs.**
   This **overturns** both the no-HP raw metric (which favored L0 via the slow confound) and the
   old per-bin "blocks-2". Honest statement: mid-depth > early; within mid-depth it's flat.
3. **Latency is physiologically coherent:** temporal electrodes (TP7/T7/FT7/T8) peak ~120-140 ms
   (N1-like), strong (r up to 0.16-0.18 at 0.5 Hz); frontal (AF3/Fz) later and weaker.

**Status:** Workstream A is now a working, de-confounded, literature-coherent encoding model
with recoverable auditory latencies. Recommended config: 0.5 Hz high-pass, uncorrected r,
single_lag.

### 2026-06-13 — in-silico MMN + held-out validation done; Workstream B planned

- **In-silico MMN** built (`scripts/insilico_mmn.py`): 8 identity-MMN frequency pairs ×
  6 layers, 4 NC-parcels, predicted parcel-EEG time courses for Sophie
  (`outputs/insilico_mmn_predictions/predictions__<layer>.h5`).
- **Held-out validation** added (`insilico_mmn.py:evaluate_heldout`): fits on the built-in
  run-level train split, scores the held-out TEST runs (62 windows) → per-parcel out-of-sample
  Pearson r. All parcels/layers positive (generalises); temporal best raw r, frontal best
  NC-normalised; layer curve flat. Did **not** overturn layer choice — **blocks.2 adopted**
  downstream (best-justified early layer; frontal = where MMN lives).
- **Workstream B plan locked** (§3, this doc): per-subject probe first → group probe →
  selectable {mTRF, probe-ind, probe-grp}; same 4 parcels; all 6 layers; encoding-only;
  MMN integration deferred (§3.7). GPU. **Next: implement B-data → B0 → B1 → B2.**

### 2026-06-13 (later) — ⭐ HANDOVER to a GPU node: Workstream B code written, TDD

**Why this note:** the probe is gradient-trained → wants a GPU. Development started on the
**CPU login node**, where the full-size probe training test got CPU-starved (>11 min, killed).
All pure/unit tests pass on CPU; the one training (integration) test needs a real compute/GPU
node. A fresh session on the GPU server should pick up from here.

**Approach = test-driven.** Tests were written FIRST (they define the API), then the modules.

**Files created (committed to working tree, NOT git-committed yet):**
- `tests/test_attn_probe_temporal.py` — 17 tests. **16 pass on CPU** (`pytest -m "not slow"`,
  ~66 s, dominated by torch import + model builds). 1 is `@pytest.mark.slow`
  (`test_probe_learns_planted_lag`) — the end-to-end learning/lag-recovery proof; **pending a
  GPU/compute node.** (`slow` marker is unregistered → harmless pytest warning; optionally add
  to `pyproject.toml [tool.pytest.ini_options] markers`.)
- `src/mbs/evaluation/attn_probe/dataset_temporal.py` — windowed dataset + parcel loading.
  Key fns: `build_windowed_design` (lookback window kept as token axis; token `L-1-lag` = lag
  `lag`, consistent with mTRF `build_lagged_design`), `sampled_windowed_design`,
  `load_parcel_eeg` (per-subject parcel mean), `WindowedTemporalDataset`, and the shared
  `CLUSTERS`/`channel_r`/`build_parcels` (defines the SAME 4 parcels from group NC).
- `src/mbs/evaluation/attn_probe/engine_temporal.py` — `corr_loss` (1−Pearson, differentiable,
  matches `pearson_along_time`), `build_probe_system` (readout selectable purely by `subjects`
  list: `["group"]` → 1 head; 19 ids → 19 heads, shared trunk), `predict_concat` / `score_heldout`
  (along-time r = the Method-A-comparable metric), `train_temporal_probe`, `TemporalTrainConfig`.
  **Reuses `SingleRoiProbeSystem`/`LatentAttentionTrunk` UNCHANGED** — the window feeds straight
  through `TokenAdapter`'s 3-D path, so there is *no new model code*.
- `src/mbs/evaluation/evaluate_features_attn_probe_temporal.py` — the CLI driver, mirrors
  `evaluate_features_mtrf.py:main`. `--readout_level {group,individual}`, all 6 layers via
  `--layer_id`, writes `attn_probe_temporal_scores.h5` (parallel schema:
  `<layer>/heldout_r,heldout_r_nc,parcels,parcel_nc_r`, +`heldout_r_persubj` for individual)
  + summary JSON. Imports verified clean on the login node.

**What the GPU instance should do, in order:**
1. **Run the unit suite on the node** to confirm the env: `pytest tests/test_attn_probe_temporal.py
   -m "not slow" -q` → expect 16 passed.
2. **Run the learning test (B0 smoke):** `pytest tests/test_attn_probe_temporal.py -m "slow" -q`
   → must pass (trained held-out r ≫ init, and >0.5). If it's flaky, the knobs are in the test
   (epochs/lr/n_train_time_samples) — it uses the DEFAULT-size probe; fine on GPU. This proves
   the window captures the planted lag and `corr_loss` training works end-to-end.
3. **B2 group probe (do FIRST end-to-end — needs no new data):** run the driver on the existing
   `outputs/neural_data/broderick2018_30s.h5` (only has `group`):
   ```
   python -m mbs.evaluation.evaluate_features_attn_probe_temporal \
     --model_id whisper-base \
     --target_feature_layers configs/extraction/audio/whisper_base_layers.json \
     --features_dir outputs/features/whisper-base-delta-t/merged/ \
     --data_hdf5_path outputs/neural_data/broderick2018_30s.h5 \
     --output_dir outputs/results/whisper-base-probe-group/ \
     --readout_level group --layer_id 2 --device cuda
   ```
   Smoke on blocks.2 first, then loop `--layer_id 0..5`. Compare `heldout_r` to Method A's
   blocks.2 row (frontal +0.139, temporal +0.181, parietal +0.088, occipital +0.106).
4. **B-data → B1 individual (NOT yet implemented):** the `individual` variant needs per-subject
   EEG in the h5, which `format_eeg_hdf5.py` does NOT yet write (only `group`). **TODO: add a
   `--store_subjects` mode** to `src/mbs/data_prep/format_eeg_hdf5.py` that writes
   `<split>/neural_data/sub-XXX/<roi>` for all 19 subjects (+`noise_ceilings/sub-XXX/<roi>`),
   keeping `group`. The driver already consumes that layout (`load_parcel_eeg(subject=…)`), and
   the synthetic-h5 fixture in the test documents the exact contract. Per-subject NC fallback is
   still open (§4/§7-2): the parcel *definition* uses group NC (fine); for masking/NC-norm of an
   individual, use that subject's own split-half if available, else group NC. Then run the driver
   with `--readout_level individual`.
5. **TODO: `scripts/slurm_attn_probe_temporal.sh`** — GPU array `--array=0-5` (one layer/task),
   `--gres=gpu:1`, mirror `scripts/slurm_mtrf.sh` header but on the GPU partition. Not written yet.
6. **B3:** drop the three columns {mTRF, probe-group, probe-individual} × 6 layers × 4 parcels
   into `aux/XX_handover_for_Sophie.md` and pick the method. MMN integration stays deferred (§3.7).

**Gotchas:** (a) don't develop training on the login node — CPU-starved; (b) `train_temporal_probe`
auto-falls back to CPU if `cuda` unavailable; (c) features are identical across subjects (all heard
all runs) so the driver preprocesses features ONCE and only varies the EEG target per subject;
(d) EEG is left raw+high-passed (not z-scored) because `corr_loss` is scale-invariant.

### 2026-06-13 (GPU node) — ⭐ Workstream B EXECUTED: B0 + B2 done; **ridge wins, mTRF is the method**

Picked up the handover above on an L40S node (Sinteract, `l40s` partition). Workstream B is now
run and benchmarked. **Conclusion: the learned temporal probe does NOT beat the linear mTRF on
Broderick at this data scale; M3's method selection resolves to mTRF (Workstream A).**

**Env fix first (logged for reuse, also in the Sophie handover "⚠️ ENV NOTE"):** the venv torch was
a `+cu130` build but the L40S driver is 560.35.03 = CUDA 12.6 → `torch.cuda.is_available()` was
False (silent CPU fallback) even on a good GPU. Fixed by reinstalling matched cu126 builds via
`uv pip` (the venv is uv-managed, no internal pip): `torch==2.12.0+cu126`,
`torchvision==0.27.0+cu126`, `--extra-index-url .../whl/cu126 --index-strategy unsafe-best-match`.
Then `cuda True | NVIDIA L40S`.

**Tests:** all 17 in `tests/test_attn_probe_temporal.py` pass on GPU (the `@slow` planted-lag
learning proof reaches held-out r > 0.5 in ~24 s, vs >11 min CPU-starved on the login node).

**B0 capacity/regularization sweep (group head, blocks.2)** — the default probe overfit hard
(train r ≈ 0.75, held-out r ≈ 0.06; ~0.69 gap). Sweeping capacity showed a clean monotonic
relationship — smaller model → higher train loss (less memorization) → **higher** held-out r:

| config (d_model/latents/layers, wd, dropout) | train loss | heldout r (front/temp/par/occ) |
|---|---|---|
| 256/16/2, 1e-4, 0.1 (default) | 0.246 | .056 / .094 / .056 / .048 |
| 128/8/1, 1e-3, 0.2 | 0.352 | .050 / .126 / .080 / .070 |
| **64/4/1, 1e-2, 0.3 (adopted)** | **0.426** | **.079 / .141 / .084 / .070** |
| 256/16/2, 1e-2, 0.3 | 0.197 | .043 / .073 / .046 / .040 |
| 32/2/1, 3e-2, 0.3 | (higher) | .088 / .122 / .077 / .059 |

**Capacity, not weight decay, is the dominant lever** (full-size + heavy reg = worst). 64/4/1 is the
sweet spot; 32/2/1 starts underfitting the high-SNR temporal/parietal parcels (but helps low-SNR
frontal). Even the best probe keeps a large train/val gap and lands below ridge. AMP off
(`--amp false`) throughout the sweep to remove fp16 as a confound.

**B2 — group probe vs mTRF, all 6 layers × 4 parcels (held-out r, probe / mTRF; bold = winner):**

| layer | frontal | temporal | parietal | occipital |
|---|---|---|---|---|
| blocks.0 | .065 / **.134** | .119 / **.186** | .075 / **.099** | .060 / **.109** |
| blocks.1 | .062 / **.131** | .119 / **.179** | .069 / **.088** | .058 / **.104** |
| blocks.2 | .079 / **.139** | .141 / **.181** | **.084 / .088** | .070 / **.106** |
| blocks.3 | .053 / **.123** | .119 / **.167** | **.074 / .070** | .064 / **.086** |
| blocks.4 | .056 / **.134** | .116 / **.174** | .067 / **.071** | .063 / **.090** |
| blocks.5 | .066 / **.140** | .116 / **.172** | .060 / **.073** | .056 / **.089** |

**mTRF ≥ probe in 23/24 cells.** Probe ties only on parietal (blocks.2/3); trails ~25–35 % on
temporal, ~50 % on frontal. Both layer-flat; probe best at blocks.2 (matches adopted mTRF layer).
Consistent with §9 risk: RidgeCV's closed-form L2 out-regularizes a learned nonlinear model at
252 train windows. MIRAGE's gains over ridge need much more data + multi-stream fusion.

**Decisions:**
- **M3 resolved → method = mTRF.** Probe built/validated/benchmarked but not the reportable encoder.
- **B1 (per-subject `individual`) deferred** — needs the unbuilt `format_eeg_hdf5.py --store_subjects`
  and only multiplies a method that already loses.
- **B4 (MMN integration of the probe) dropped** — §3.7 gated it on "probe clearly beats mTRF"; it
  doesn't. The in-silico MMN stays on the mTRF mapping (2026-06-13 entry above).
- **Re-open only with more data** — the harness is apples-to-apples and the code is ready, so it's a
  cheap re-run if/when we pool datasets or add multi-stream features (§6).

**Artifacts:** `outputs/results/whisper-base-probe-group-r2-all/attn_probe_temporal_scores.h5`
(6 layers, group); sweep dirs `whisper-base-probe-group{,-r1,-r2,-r3,-r4}/`.

**Minor TODO (non-blocking):** `engine_temporal.py:144,167` use deprecated `torch.cuda.amp.{GradScaler,
autocast}` (FutureWarning under torch 2.12) — should migrate to `torch.amp.*('cuda', ...)`.
`scripts/slurm_attn_probe_temporal.sh` was never needed (interactive node sufficed) — skip unless B reopens.

---

## 12. Discussion — would the learned probe (Workstream B) benefit from multi-source data?

*This is a discussion, not a committed plan. Prompted by Hannes (2026-06-13): Sophie reported that
combining data from multiple sources "didn't work well." The question is whether the learned
temporal probe — which lost to the mTRF (§11) — would be the thing that benefits from more/combined
data, and under what conditions. Captured here so a future "if B reopens" decision is informed.*

### The asymmetry that makes this worth considering
The probe didn't lose because the architecture is wrong; it lost to **overfitting at 252 windows**
(train r ≈ 0.75, held-out ≈ 0.06). That gap is a data-starvation signature. The mTRF (RidgeCV) is
already near its regularized ceiling, so extra data buys it little; a high-capacity gradient model's
ceiling, by contrast, rises with data. So the **marginal value of more data is asymmetric — it
favours the probe**, and this is precisely MIRAGE's premise (its gains over ridge appear at scale,
not at n=252). The probe also has the architecture pooling wants — a **shared trunk + per-source
heads**: the trunk can learn the dataset-invariant feature→neural mapping while each head absorbs
source-specific scale/montage. Ridge is a single linear map and cannot pool this way without either
no sharing (fit separately) or inheriting the confound (stack).

### Two very different senses of "multiple sources"
1. **Multi-*stream* fusion (same dataset, several model backbones — whisper + wav2vec2 + AST over
   the same Broderick EEG).** This is a large part of where MIRAGE's gains actually came from, and it
   **sidesteps Sophie's problem entirely**: same subjects, montage, paradigm → no cross-dataset
   confound, just complementary input signal + nonlinear cross-stream structure for the trunk to
   exploit that a per-stream ridge can't. **Lower risk; the better first test if B reopens.** (Gated
   on §6's per-model time-alignment, since wav2vec2/AST have different native rates.)
2. **Multi-*dataset* pooling (several EEG datasets).** This is what bit Sophie, and the failure mode
   is exactly §6's **dataset-identity confound**: different amplifiers/references/units → amplitude
   scale differs → a flexible model learns "which dataset" instead of "feature→brain." The
   uncomfortable corollary: **a learned probe is *more* vulnerable to this than ridge, not less** —
   its flexibility is also what lets it take the shortcut. Naive pooling could hurt the probe *worse*
   than it hurt Sophie's linear analysis. More rows help only if they carry *transferable* signal;
   pooling incompatible montages/paradigms mostly adds nuisance variance.

### Conditions under which multi-dataset pooling would actually help the probe
Essentially §6's harmonization checklist, all of which the probe needs:
- **Per-dataset, per-channel z-scoring on train stats** — neutralizes the amplitude confound (most
  likely culprit in Sophie's case).
- **A common time grid** — resample all streams + EEG to one rate, or windows aren't comparable.
- **Harmonized parcels/channels** — map heterogeneous montages onto canonical ROIs (or source space)
  so the heads share a target space.
- **Shared trunk + per-dataset/per-subject heads** — push source-specific scale into the head, keep
  the trunk invariant; optionally a dataset-confusion/adversarial penalty to actively block shortcut
  learning.
- **Within-dataset scoring** — score each dataset on its own held-out runs, never pool the metric, or
  the confound inflates the number (a likely reason naive pooling can "look" fine then fail).
- **Same modality/paradigm ideally** (auditory continuous speech) — pooling speech with oddball-MMN
  shares little of the mapping.

### Honest expectation
Even done correctly, multi-*dataset* pooling would plausibly bring the probe **to parity** with the
mTRF rather than to a decisive win, *unless* the added data carries nonlinear/temporal-context
structure ridge fundamentally can't capture. Multi-*stream* fusion has the better shot at a real win
because it adds complementary signal without the confound. Separately, adding a **repeats dataset**
(§4) is worth doing regardless — not to beat ridge, but for a proper within-subject noise ceiling on
the visual-benchmark scale. **Bottom line: the probe *is* the method that benefits from more data and
its shared-trunk/per-head design is purpose-built for pooling — but Sophie's experience is the correct
prior that the benefit is conditional on harmonization, and the flexible model is the one most likely
to cheat if harmonization is skipped. If B reopens, test multi-stream fusion on Broderick before
multi-dataset pooling.**

---

## 13. Plan — train the encoder on D2 and D3, both methods, and compare (committed 2026-06-13)

*Operationalizes §12 for the two datasets that already exist. Goal = the two questions Hannes wants
answered: (Q1) does the **mTRF** encoding **replicate** on an independent dataset (D2)? (Q2) does the
**probe** close the gap to / beat the mTRF when given **more data** (D3, 409 train windows vs D1's
252)? Encoding-r only; no MMN. Decisions locked with Hannes 2026-06-13 — see end of section.*

### 13.1 Datasets (verified to exist + readable in Sophie's tree, 2026-06-13)
| set | neural h5 | train / test windows | note |
|---|---|---|---|
| D1 Broderick | `outputs/neural_data/broderick2018_30s.h5` (ours) | 252 / 62 | done (mTRF + probe) |
| D2 Surprisal | `…/sigfstea/…/surprisal_30s.h5` (139 MB) | 157 / 43 | Weissbart Cortical Surprisal, 13 subj, 63-ch native 10-20 |
| D3 combined | `…/sigfstea/…/d3_combined_30s.h5` (233 MB) | 409 / `test_d1`=62 + `test_d2`=43 | D1∪D2, **already per-channel z-scored on train stats** (Fz std=1) |

All 10 parcel channels (Fz,F3,F4,T7,T8,Pz,P7,P8,Oz,O2) are present in D2 and D3 → the **same 4
parcels** transfer; only `parcel_nc_r` is recomputed per dataset. D2 whisper-base features exist at
`…/whisper-base-delta-t-surprisal/merged/` (1.6 GB), config **identical** to ours (whisper-base,
delta_t, blocks.0–5, [n,1500,512]) → reuse, do not re-extract.

### 13.2 The three confounds and their three (orthogonal) fixes
- **Amplitude/scale leak** (D1 std≈1357 vs D2 std≈0.0003): fixed by **per-dataset z-scoring on train
  stats** — already baked into `d3_combined_30s.h5`. Verify on load; do NOT re-standardize.
- **Pooled-test inflation**: fixed by **scoring `test_d1` and `test_d2` separately, never pooled**
  (§12). This is the one real code change (13.3).
- **Montage mismatch** (BioSemi128→10-20 vs native 63-ch): mitigated by **parcels** (region averages,
  same 4). Not the same problem as the other two.

### 13.3 Required code change — per-dataset held-out scoring (both drivers)
Both `insilico_mmn.py`/`evaluate_features_mtrf` (held-out path) and
`evaluate_features_attn_probe_temporal.py` currently score a single `test` split. Add: when the
neural h5 has `test_d1`/`test_d2` groups, score **each separately** and write
`heldout_r__d1` / `heldout_r__d2` (+ `_nc`). For D1/D2-alone runs the existing single-`test` path is
unchanged. TDD: extend `tests/test_attn_probe_temporal.py` with a synthetic 2-subgroup h5 asserting
the per-subgroup r is computed within-group only (a planted between-group scale offset must NOT
inflate it). Minor: D3 features = D1∪D2; D1/D2 chunk filenames collide → rename-on-copy into a single
`whisper-base-delta-t-d3/merged/` OR teach `load_layer_features` to accept multiple folders.

### 13.4 Steps
1. **Copy (~2 GB)** Sophie's `surprisal_30s.h5`, `d3_combined_30s.h5`, and
   `whisper-base-delta-t-surprisal/merged/` into our `outputs/` (copy, not symlink). Assemble the D3
   feature dir from D1∪D2 (handle filename collision).
2. **Verify torch/CUDA on the GPU node** (handover ENV NOTE): `cuda True | NVIDIA L40S` before any
   probe run. mTRF is CPU.
3. **Q1 — D2 standalone:** mTRF (6 layers, 4 parcels, held-out r on D2 test) + probe (best config
   `d_model=64,num_latents=4,1 layer`, group). Compare to D1. *Expectation: mTRF replicates if
   parcel-level r is positive and temporal-led; probe likely worse than on D1 (157 < 252 windows).*
4. **Q2 — D3 pooled:** mTRF + probe trained on the 409-window pool, **scored on `test_d1` and
   `test_d2` separately**. Key contrasts: (a) does D3-trained beat D1-trained on D1-test and
   D2-trained on D2-test (does pooling help)? (b) **does the probe's gap to the mTRF shrink vs the
   23/24-cell loss at D1's 252 windows** — the actual "more data" re-test.
5. **Comparison tables** into `aux/XX_handover_for_Sophie.md`: train{D1,D2,D3} × method{mTRF,probe} ×
   test{D1,D2} × 4 parcels × 6 layers, held-out r (+ NC-norm). Headline = Q1 (replication) and Q2
   (probe gap vs data scale).

### 13.5 Compute / method notes
- mTRF: CPU, minutes. Probe: GPU (Kuma L40S, cu126 torch), minutes/run.
- Probe `corr_loss` is scale-invariant per batch, but a **mixed D1+D2 batch with residual scale gap
  could still let it predict dataset identity** → rely on D3's pre-z-scoring AND per-dataset scoring;
  if any inflation persists, fall back to per-dataset batches or a shared-trunk + per-dataset-head
  setup (the `individual`-style mechanism, here keyed on dataset).
- Keep everything else identical to the D1 runs (0.5 Hz HP, 0–800 ms lookback, same parcels) so the
  only moving part is the dataset.

### 13.6 Decisions locked (Hannes, 2026-06-13)
(1) Score `test_d1`/`test_d2` separately, never pooled. (2) Copy data into our repo (~2 GB; reuse
features = whisper activations, NOT her mapping/results — re-extraction would reproduce identical
numbers). (3) Same 4 parcels, per-dataset NC. (4) Clean slate vs Sophie's per-bin numbers. Scope =
the two questions above; D1 already in hand, not re-run.

### 13.7 Risks
- D2's lower reliability / fewer windows may make even the mTRF weak → Q1 could read "partial
  replication"; report honestly. Per §12, the realistic Q2 outcome is the probe reaching **parity**,
  not a decisive win — a clean parity-at-2×-data result is still informative.

---

## 14. ⚠️ Verification agenda — the "probe wins on D2" result is suspicious (2026-06-14)

**The result.** whisper-base, blocks.2, held-out r (full table in `XX_handover_for_Sophie.md`):
the learned probe **loses** to the mTRF on D1 (temporal 0.141 vs 0.162) but **beats it ~1.9× on D2**
(temporal **0.363** vs 0.195), and holds high on D2-test under pooling (0.300). This **flips** the
earlier "ridge wins" conclusion (which was D1-only) and is **not** a data-quantity effect — D2 has
*fewer* train windows than D1 (157 vs 252). Either D2 has genuinely more nonlinear/temporal
structure the probe can exploit, or it's an artifact. **Verify before reporting.**

**Leading hypothesis = split-structure asymmetry.** D1 is split **by run** (`test_runs`, entirely
separate ~3-min audiobook segments → zero acoustic overlap between train and test). D2
(`format_eeg_hdf5_surprisal.py`) is split **by story part, holding out one part per audiobook** — so
each test part shares the *same audiobook* (speaker, recording, prosody) as the train parts. With
30 s / 10 s overlapping windows, D2's held-out set is plausibly "closer" to train than D1's, and a
high-capacity probe can exploit that proximity far more than a rigid ridge → inflates the probe's D2
edge specifically. (Note the mTRF *also* scores higher on D2 than D1, consistent with an easier
split, but the flexible probe amplifies it.)

**Tests, in priority order:**

1. **Inspect the D2 split (cheap, do first).** Read `format_eeg_hdf5_surprisal.py` split logic and the
   actual train vs test stimulus IDs in `surprisal_30s.h5`: do test parts come from the **same
   audiobooks** as train parts? Are any test windows temporally adjacent to / overlapping train
   windows (same audiobook, neighbouring offsets)? If yes → the split is "easier" than D1's and the
   confound is real. Also confirm **no identical window** appears in both splits.
2. **Disjoint-audiobook re-split of D2 (the decisive test).** Re-format D2 holding out **entire
   audiobooks** (not parts within an audiobook) so train/test are fully disjoint recordings — the D1
   analogue. Re-run mTRF + probe. **If the probe's D2 advantage collapses → it was split structure;
   if it survives → it's real D2 structure.** This is the single most informative check.
3. **Seed stability of the probe on D2.** Re-run probe D2 with 2–3 seeds (`--seed`). Is temporal
   ≈0.36 stable, or is it a high-variance draw? (The probe is gradient-trained; one run only so far.)
4. **Model-size consistency (already in flight).** tiny/small/medium probe-D2 jobs are running. If
   "probe ≫ mTRF on D2" holds across sizes, it's far less likely a fluke; if it's base-only, suspect
   the run.
5. **NC sanity.** D2 temporal NC ≈ 0.50, so probe r 0.363 ⇒ r/NC ≈ 0.73 — high. Sanity-check against
   a within-subject ceiling if obtainable; an r/NC near or above 1 would flag leakage.
6. **(lower priority) Confirm the mTRF transfer near-failure is real**, not a feature-standardisation
   / parcel-alignment bug across datasets: spot-check that D1→D2 transfer uses D1 train mu/sd on D2
   features and the same 4 parcels (it should, via `cross_score_dataset`).

**Decision rule.** If test 2 shows the probe's D2 win survives a disjoint-audiobook split → the probe
becomes a real candidate method (at least for D2-like data) and Workstream B reopens for the MMN.
If it collapses → keep the mTRF as the reportable method and treat D2's by-part split as the cause;
document it and move on. Until test 2 is done, **report the mTRF as primary and flag the probe-D2
result as preliminary.**

## 15. mTRF segfault on wide models — eigen fix + PCA option (2026-06-14)

**Symptom.** The whisper-family mTRF sweep: `tiny`/`base` completed all of D1/D2/D3+transfer, but
`small` and `medium` **segfaulted** (`Segmentation fault (core dumped)`) on every task that *fits on
D1 or D3* — D1 mtrf, D3 mtrf, and the d1→d2 transfer all produced empty (96-byte / 1176-byte) h5s.
The D2-side tasks (`_1`) ran fine. Deterministic, not OOM (`sacct` = FAILED, not OUT_OF_MEMORY).

**Root cause — LAPACK int32 overflow in the GCV SVD.** `RidgeCV` (GCV) chooses its solver by shape:
`n_samples > n_features` → **SVD of the full design X**. The D1 design is 252 windows × 200 samples
= **50,400 rows**; columns = 41 lags × `d_model`. The `gesdd` workspace integer scales as ~`4·min²`
(2³¹ ≈ 2.1e9):
- tiny `d=384` → 15,744 cols → 4·min² ≈ 9.9e8 ✅
- base `d=512` → 20,992 cols → ≈ 1.76e9 ✅ (just under)
- small `d=768` → 31,488 cols → ≈ 3.96e9 ❌ overflow → segfault
- medium `d=1024` → 41,984 cols → ≈ 7.05e9 ❌ overflow → segfault

D2 has fewer rows (31,400) than features → RidgeCV auto-switches to the *eigen* (Gram) path → never
hits `gesdd` → survives. Exactly why every `_1` task ran and every D1/D3 task crashed.

**Fix applied (Hannes' choice).** Force `gcv_mode='eigen'` in `fit_parcel_mtrf`
(`evaluate_features_mtrf_parcels.py`; reused by the cross-dataset driver). Eigen computes the
**numerically identical** GCV ridge solution via the n×n Gram matrix `X Xᵀ` — so **tiny/base results
are unchanged** (no re-run needed) and only `small`/`medium` are re-run. SLURM bumped to
`--cpus-per-task=32`, `--time=48:00:00` (jed `standard` is infinite-wall, 72c / 504 GB) because eigen
is heavier than svd: cost is an O(n³) eigendecomposition of the Gram, `n = n_windows·200` (D1 ≈
50,400; **D3 ≈ 81,800**, Gram ≈ 50 GB in float64). D3 for the deep models is the slowest case and the
most likely to need the full 48 h — if a D3 task still times out, that is the trigger to switch to PCA.

**Why eigen is not enough, and PCA (IMPLEMENTED 2026-06-14).** The eigen fix dodges the *width*
overflow (svd of a wide design) but **not** the *n* overflow: pooled **D3 has n ≈ 81,800 rows**, so
the eigen path's n×n Gram (81,800²) overflows int32 in the LAPACK symmetric eigensolver workspace
(~2n²) and in 32-bit BLAS indexing of the 6.7e9-element Gram → small/medium **D3 segfaulted in ~90 s
even with eigen**. Both exact-GCV paths are therefore dead for D3 at wide feature dims.

**Fix: PCA the features before lagging (`--pca_var`, Hannes' choice = 0.95).** Fit `PCA(n_components=
0.95)` on the standardised TRAIN features per layer (PC count is **variance-driven → varies by
model/layer**, e.g. medium keeps however many PCs reach 95 %), lag the projection, store the PCA and
re-apply it to held-out/transfer features. The design width becomes `n_PCs · n_lags` —
**model-independent and small** — so `n_samples > n_features` and the cheap `svd` GCV path is used (no
width overflow, and no n×n Gram → no D3 overflow). Bonus: arguably a **fairer scaling comparison**
(comparable mapping capacity per layer regardless of raw width). Selectable + separate outputs so it
doesn't clobber the raw-feature runs:
- code: `fit_parcel_mtrf(..., pca_var=0.95)` returns `{model, mu, sd, pca}`; `score_parcel_mtrf`
  applies the stored PCA; `--pca_var` on both drivers; `gcv_mode='auto'` when PCA is on, else `'eigen'`.
  Provenance: `pca_var` h5 attr + per-layer `n_pcs`. Tests:
  `test_fit_score_parcel_mtrf_with_pca_reduces_and_recovers`, `test_fit_parcel_mtrf_no_pca_by_default`.
- run: `PCA_VAR=0.95 MODEL_ID=... sbatch --export=ALL scripts/slurm_mtrf_parcels.sh` → writes to
  `…-mtrf-parcels-pca-<tag>/` (raw-feature dirs untouched). Same `PCA_VAR` knob on `slurm_cross_mtrf.sh`.

The raw-feature (eigen) D1/D2 numbers stay as the non-PCA reference; the PCA set is the consistent
all-sizes pipeline that *also* covers D3. If we ever want to revisit the variance fraction, 256 fixed
PCs or a different % are one-flag changes.

## 16. Redo the attention encoder with MSE, not 1−Pearson (2026-06-17)

**Decision: retrain the temporal attention encoder with MSE loss.** The temporal probe currently
trains on `corr_loss` (1−Pearson, `attn_probe/engine_temporal.py:29`), and its docstring claims this
"matches MIRAGE." **That claim is wrong.** MIRAGE (Gokce & AlKhamissi 2026, App. A.1) trains with
**mean-squared error** and only uses Pearson for *checkpoint selection*; the paper explicitly reports
that augmenting MSE with a Pearson-correlation term **did not outperform MSE alone**. Our own
non-temporal probe (`attn_probe/engine.py:120`) already uses MSE — only the temporal path diverged.

**Why it matters for the MMN.** `corr_loss` is scale- and shift-invariant, so the encoder's predicted
amplitude is arbitrary — only the sign/shape of a deviant−standard deflection is interpretable, and
magnitudes are not comparable across parcels or models. That blocks an amplitude-based MMN criterion
for Method B and forces a sign-only rule. Training on **MSE fits the real EEG amplitude**, so:
(1) predicted amplitudes become calibrated and the MMN criterion can use magnitude for *both* methods,
(2) we faithfully match MIRAGE, (3) per the paper this costs nothing in fit quality.

**Scope.** Swap `corr_loss` → `F.mse_loss` in the temporal training step (`engine_temporal.py`),
re-standardise EEG targets as needed (note (d) above — raw+high-pass was chosen *because* corr_loss
was scale-invariant; MSE wants z-scored or consistently-scaled targets), retrain the checkpoint
(`outputs/results/whisper-small-probe-group-d2-mmn/model__blocks.10.pt`), and re-run the in-silico MMN.
Until retrained, the mTRF stays the primary/reportable method (it already preserves amplitude).

### 16.1 IMPLEMENTED + SUBMITTED (2026-06-17 PM) ✅

All four scope items done, plus electrode-level targets to match the §20 mTRF sweep. Tests green
(22/22 on a jed CPU node, job 56269622, incl. the two slow learning/selection tests that the
login node timed out). Encoder sweep submitted to Kuma.

**Code changes (4 files):**
- `attn_probe/engine_temporal.py` — training loss `corr_loss` → `F.mse_loss`; added MIRAGE-style
  **best-checkpoint selection** (keep the weights with best validation Pearson, restore at end;
  gated by `TemporalTrainConfig.eval_every`). `corr_loss` kept (still unit-tested), no longer the
  objective.
- `attn_probe/dataset_temporal.py` — added `build_electrodes()` + `montage_pos()`, **mirroring
  `scripts/eeg_targets.py`** so the encoder's electrode set is byte-for-byte the same as the §20
  mTRF sweep (verified at run time: **47 electrodes**, **5 parcels** — identical counts).
- `attn_probe/checkpoint.py` — persist per-subject `eeg_mu`/`eeg_sd`; new `predictions_to_units()`
  inverts z-unit predictions back to real EEG amplitude (the additive μ cancels in deviant−standard,
  so an MMN difference is exactly `z_diff * sd`). Legacy (1−Pearson) checkpoints raise on inversion.
- `evaluate_features_attn_probe_temporal.py` — `--target_level {parcels,electrodes}`; D2
  (`surprisal_30s.h5`) is now the default NC source; **EEG targets z-scored per target on
  train-portion stats** and the scaling stored in the checkpoint; **validation split carved from
  TRAIN** (`--val_frac 0.2`, `--eval_every 5`) for checkpoint selection so the **test split is
  never touched during selection** (the §20 CV-on-train discipline). Also fixed a pre-existing
  crash where the driver read `args.save_model` that the unit-test namespace lacked.

**Decisions locked here:**
- *Validation = carved from train, not the test split.* Selecting the checkpoint on the reported
  test split would bias it; 20 % of train is held out for selection instead.
- *`heldout_r` stays comparable.* Pearson is scale-invariant, so z-scoring the EEG targets does not
  change the reported r — new numbers sit beside the old mTRF / 1−Pearson ones, while the stored
  predictions are now amplitude-calibrated.
- *Did NOT touch* `scripts/eeg_targets.py` or any mTRF code — the concurrently-running mTRF sweep
  (jed `eeg_sweep` 56149224) is unaffected.

**New run scripts:**
- `scripts/run_probe_d2_levels.sh` — one model, both target levels sequentially (D2 only).
- `scripts/kuma_probe_d2_levels.sh` — sbatch wrapper of the above (single model).
- `scripts/kuma_probe_d2_sweep.sh` — **array** twin of `slurm_eeg_mapping_sweep.sh`: 8 tasks =
  {tiny,base,small,medium} × {parcels,electrodes}, one GPU each (`TASK/2`=model, `TASK%2`=level).
- `scripts/jed_probe_tests.sh` — CPU sbatch running the full `test_attn_probe_temporal.py`.

**Submitted (Kuma, job `3654194`, `--array=0-7`):** all 4 models × both target levels, group
readout, layers 0..N each, MSE + selection, `--epochs 200 --d_model 64 --num_latents 4
--cross_attn_layers 1 --dropout 0.3 --weight_decay 1e-2`. Outputs →
`outputs/results/<model>-probe-group-d2-{parcels,electrodes}/attn_probe_temporal_scores.h5`
(+ per-layer `model__<layer>.pt` checkpoints now carrying `eeg_mu`/`eeg_sd`).

### 16.2 HOW TO ANALYZE TOMORROW (2026-06-18)

1. **Confirm the sweep finished cleanly:**
   ```bash
   sacct -j 3654194 --format=JobID,State,ExitCode | grep -E "_[0-7] "   # want COMPLETED 0:0 ×8
   ls outputs/results/whisper-*-probe-group-d2-*/attn_probe_temporal_scores.h5
   ```
   Each `attn_probe_temporal_scores.h5` has, per layer key: `heldout_r__test` [n_target],
   `heldout_r_nc__test`, `parcels`/`parcel_nc_r`, and `final_train_loss`.

2. **Encoder vs. mTRF, same (model × level).** The mTRF sweep wrote
   `outputs/results/eeg_mapping/<model>__<level>__D2.json` (`test_r_chosen`, `cv_score_by_layer`,
   `chosen_layer`). Compare the encoder's best-layer mean held-out r against the mTRF `test_r_chosen`
   for each model × level. Headline question: **does MSE + selection close (or beat) the gap that
   made "ridge wins" (log 2026-06-13)?** Plot the mTRF side with
   `python scripts/plot_eeg_mapping.py --target_level {parcels,electrodes}`.

3. **⚠️ Layer-selection caveat (do this honestly).** The mTRF picks its layer by **CV-on-train**;
   the encoder run records only per-layer **test** `heldout_r` + `final_train_loss` in the h5 — the
   per-layer *validation* score is NOT persisted (best-val selection happens *within* a layer, over
   epochs, via `eval_every`). So picking the encoder's layer by `heldout_r__test` is a mild peek.
   Two clean options before quoting a single number: (a) re-derive the per-layer val r offline (the
   train-carved val split is reproducible from `--seed 42` + `--val_frac 0.2`) and select on that;
   or (b) cheap follow-up — write `best_val_r` per layer into the h5 and re-run. For a first look,
   report the **whole per-layer test curve** (not just the max) so the choice is transparent.

4. **Amplitude calibration sanity (unblocks the MMN).** Confirm the checkpoints carry the EEG stats
   and that inversion works:
   ```python
   from mbs.evaluation.attn_probe.checkpoint import load_probe_checkpoint
   _, c = load_probe_checkpoint("outputs/results/whisper-small-probe-group-d2-parcels/model__blocks.10.pt")
   print(c["eeg_mu"]["group"], c["eeg_sd"]["group"])   # finite, per-target; sd>0
   ```
   With this in place the in-silico MMN can use a **magnitude** criterion (deviant−standard in real
   units = `z_diff * sd`) for both Def-1 and Def-2 stimuli — the original motivation for §16.

5. **Next step after the read-out (not tonight):** point the encoder MMN driver
   (`scripts/insilico_mmn_attn.py`) at the chosen calibrated checkpoint and re-run the in-silico MMN
   (§3.7 / §17), now using `predictions_to_units` for the amplitude verdict.

## 17. The two ways to construct the MMN — physical control of the eliciting tone (2026-06-17)

**The distinction (this is the one that matters, and it is NOT counterbalancing).** The MMN is always
read **time-locked to the last (eliciting) tone** — the tone after whose onset you measure the
deflection. The only question is what you subtract from it. There are two constructions:

- **Definition 1 — uncontrolled.**
  `standard = {1000, 1000, 1000, 1000}` vs `deviant = {1000, 1000, 1000, 1200}`.
  The sequences are identical except for the last tone. The surprise in the deviant comes from the
  1000→1200 shift; the standard has no shift, no surprise. **But the eliciting tones differ
  physically** (1000 Hz in the standard, 1200 Hz in the deviant), so `deviant − standard` confounds
  *surprise* with the *acoustic difference of the probe tone itself* (a 1200 Hz tone simply evokes a
  different response than a 1000 Hz tone, surprise aside).

- **Definition 2 — physically controlled.**
  `standard = {1200, 1200, 1200, 1200}` vs `deviant = {1000, 1000, 1000, 1200}`.
  Again no change (no surprise) in the standard, and the deviant's surprise is the 1000→1200 shift at
  the final tone. **Now the eliciting tone is physically identical (1200 Hz) in both conditions** — so
  `deviant − standard` isolates the surprise/prediction-error and holds the probe-tone acoustics fixed.
  This is the clean, modern design.

Counterbalancing is **one way to achieve Definition 2** (averaging each physical tone equally across
the standard and deviant roles), not the distinction itself.

**Who does which:**

| | Construction | How |
|---|---|---|
| **Early MMN papers** (Sams 1985, Tiitinen 1994) | **Definition 1 (uncontrolled)** | Classic oddball: standard ERP = response to the frequent tone (1000 Hz), deviant ERP = response to the rarer *different* tone (1032 Hz etc.). Verbatim Tiitinen: *"MMN was obtained by subtracting the response to the standard tone from the response to the deviant tone."* The two eliciting tones differ physically. |
| **Weber 2022** (eLife, TNU) | **Definition 2 (controlled)** | Counterbalanced roving/volatility oddball (440/528 Hz). Each physical tone serves equally as standard and as deviant, so for a given tone they compare it-as-deviant vs it-as-standard — *same physical probe tone*, only the preceding context differs. *"Both stimulus categories have, on average, the same physical properties."* |
| **Us** (`insilico_mmn.py` `method_09`, identity-MMN) | **Definition 2 (controlled)** | The strictest form: the final/critical tone is **literally physically identical** in the standard and deviant clip (trial-level identity, not just average-matched); the deviance lives entirely in the preceding context frequency (1000→600 Hz). |

**Bottom line: we and Weber both use the controlled Definition 2; the founding MMN papers used the
uncontrolled Definition 1.** So our identity-MMN design matches modern best practice (Weber), *not*
the classic Sams/Tiitinen oddball — which is the looser construction.

**⚠️ Consequence for `mmn_screening_plan.md`.** The Sams/Tiitinen frequency *pairs* now listed in that
plan's Step 2 come from a **Definition-1 classic oddball** (different standard vs deviant tones).
Adopting those stimuli in a literal oddball would switch the screen to Definition 1 and reintroduce the
physical confound — inconsistent with our Definition-2 `method_09`. The frequencies are still useful as
a **deviance-size axis** for the parameterized check, but they must be embedded in a Definition-2
(physically controlled / counterbalanced) paradigm, not a raw oddball. This is the "Paradigm match"
flag in the screening plan; resolve it toward Definition 2.

## 18. Durable cu126 torch pin (2026-06-17)

**Problem.** The working `torch==2.12.0+cu126` build was only ever a **venv-level override** (installed
via `uv pip install` — the §11 L40S fix — which never touches `uv.lock`). The lock pinned the generic
`torch==2.12.0` from PyPI. So any `uv add`/`uv sync` re-synced the venv to the lock and silently
clobbered the GPU build with the generic wheel, which fails at import (`libtorch_cuda.so: undefined
symbol: ncclCommResume`). This bit us when `uv add PyPDF2` was run (the lock diff touched *only* pypdf2
— torch wasn't in it — yet the venv torch broke, which is exactly the venv-override-lost signature).

**Fix (in `pyproject.toml`).** Declare an explicit pytorch-cu126 index and route torch/torchvision to it
via `[tool.uv.sources]`, so the **lock itself carries the `+cu126` build** and future `uv add`/`uv sync`
keep it:
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true                         # only used by packages that name it in [tool.uv.sources]

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux'" }]
```
`explicit = true` keeps the cu126 index from polluting resolution of any other package. The
`sys_platform == 'linux'` marker is required because the cu126 index has **no macOS wheels** — without
it, the universal `uv lock` (the repo supports Darwin, cf. the `scikit-learn-intelex` Darwin marker)
would fail to resolve torch for macOS; with it, non-Linux falls back to the default PyPI index.

**Apply:** `uv lock` regenerates the lock with torch/torchvision sourced from `download.pytorch.org/whl/
cu126` (version unchanged, 2.12.0+cu126). The venv already has that build, so no `uv sync` is needed now;
the win is that the lock is now self-consistent and the build survives future dependency changes.

## 19. Quick-run reference — fit-quality bars + electrode-level MMN (2026-06-17)

Two ready-to-run drivers added this session. **Login node OOM-kills both — submit to a compute node.**
All paths relative to the repo root; `source env.sh` is done inside each slurm script.

**(a) Held-out fit-quality bars across the model size ladder (D2).** One figure, 4 panels
(tiny/base/small/medium), 5 bars each (frontal/central/temporal/parietal/occipital), height = held-out
TEST Pearson r, each model at its best D2 layer (tiny=blocks.1, base=blocks.5, small=blocks.10,
medium=blocks.22). Parcels (incl. central) built from `surprisal_30s.h5`, NC floor r>0.2.
```bash
sbatch scripts/slurm_fit_quality_bars.sh
# -> outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png
grep -E "r=\+|DONE" logs/fq_bars_*.out
```
Generator: `scripts/plot_fit_quality_bars.py` (refits each model at its best layer; reuses
`insilico_mmn.fit_mapping`). Best layers were read from `outputs/results/<model>-mtrf-parcels-d2/`;
note base only ever had `blocks.5` evaluated (not a real sweep).

> **In flight (submitted 2026-06-17 ~16:47, still running ~17:03, ≥16 min):** job **55914970**
> (`fq_bars`, node jst184) is exactly run (a) above. It refits all four models sequentially
> (tiny→base→small→medium; medium is slow — 13 G feature tree). When it finishes the figure lands at
> `outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png` and the per-parcel `r=...`
> values + `DONE` line are in `logs/fq_bars_55914970.out`. Check: `squeue -j 55914970` (empty = done),
> then `ls -la outputs/figures/fit_quality/fit_quality_bars__all_models__d2__mTRF.png`.

**(b) Electrode-level in-silico MMN — the MMN screen for Sophie.** Feeds MMN stimulus pairs through the
frozen mTRF (per-electrode targets, ~30–47 channels passing NC r>0.2), writes **one 10-20 montage figure
per pair** (each electrode's deviant−standard trace at its scalp position, fronto-central ROI in red,
100–240 ms band shaded) plus a **simple auto-verdict** (ROI mean amp in 100–240 ms < 0 ⇒ MMN present)
and an `N/M pairs show an MMN` summary.

⚠️ **Train the mapping on D2, NOT Broderick.** Broderick/D1 has ~zero held-out r on the central
electrodes (its central NC floor fails), which kills the fronto-central MMN ROI. D2 (Cortical
Surprisal — also a human-speech audiobook EEG) has healthy fronto-central channels (FCz=0.99, C3=0.92,
Cz=0.78), so it's the right training set for the MMN screen. **D2 is now the script default.**
```bash
sbatch scripts/slurm_insilico_mmn_electrodes.sh          # D2, whisper-small, blocks.10, all methods
# -> outputs/figures/insilico_mmn_electrodes/insilico_mmn_electrodes__<method>__<layer>.png
grep -E "MMN|pairs show|ROI used" logs/insilico_mmn_elec_*.out
```
Driver: `scripts/insilico_mmn_electrodes.py` (electrode counterpart of `insilico_mmn.py` — each
electrode is a single-member "parcel", so `fit_mapping`/`analyze_method`/held-out eval are reused
unchanged). The dataset to fit on is `--train_neural`/`--train_features` (the dataset-agnostic names;
old `--broderick_*` kept as aliases). Runs against the 8 existing identity-MMN pairs (method_09/12/37/44/55 ±counter).

> **TODO (long-term cleanup):** the original `scripts/insilico_mmn.py` (the parcel-level driver) still
> carries the misnamed `--broderick_neural`/`--broderick_features` flags AND defaults to Broderick/D1.
> Rename them to `--train_*` and switch its default to D2 there too, for the same central-electrode
> reason — left untouched for now so the existing parcel figures keep reproducing. The electrode driver
> already does the right thing; `insilico_mmn.py` is the remaining one.
**Adding a new pair (Sophie):** drop the stimuli in `outputs/mmn_stimuli/<name>` and delta_T features in
`outputs/features/mmn-<name>-delta-t`, then `--methods <name1>,<name2>,...`. Criterion knobs:
`--mmn_lo_ms/--mmn_hi_ms/--mmn_thresh`. The auto-verdict is a first pass — keep the call visual until a
few are eyeballed. Pair selection: ~10 from recent Def-2 studies (same eliciting tone), see
`aux/mmn_screening_plan.md` and §17.

## 20. ⭐ PICK UP HERE (2026-06-18 AM) — clean D2-only model→EEG mapping, from scratch

**Decision (2026-06-17 PM).** Redo the model→EEG mapping cleanly. Supersedes the ad-hoc fq_bars +
best-layer-from-old-results approach (§19a) and the scattered `*-mtrf-*` results. One honest pipeline:

- **D2 only** (Cortical Surprisal; Broderick/D1 had ~zero held-out on central electrodes).
- **Per model** (tiny/base/small/medium): **full layer sweep**; **k-fold CV *within the train split***
  picks the layer (test split never touched during selection).
- At the chosen layer, score the held-out **~20% test split** → per-target test r; **visualize like
  fq_bars**.
- **Two granularities**: the 5 parcels AND all electrodes passing NC r>0.2 (~47 on D2).

**New files (built 2026-06-17, syntax-checked, not yet run):**
- `scripts/eeg_targets.py` — shared, **broderick-free** target builders (parcels, electrodes, montage,
  `load_split_targets`). The clean home the MMN drivers should import from too.
- `scripts/eeg_mapping_sweep.py` — per (model × level): sweep layers → CV-on-train select → refit →
  held-out test r. RidgeCV `gcv_mode='eigen'`; optional `--pca_var`. Writes one JSON per run to
  `outputs/results/eeg_mapping/<model>__<level>__D2.json`.
- `scripts/plot_eeg_mapping.py` — (A) layer-selection curves (CV solid, test dashed, chosen layer
  circled) + (B) test-r bars (fq_bars-style, one panel/model), per level.
- `scripts/slurm_eeg_mapping_sweep.sh` — array, 8 tasks = {tiny,base,small,medium}×{parcels,electrodes}.

**RUN ORDER (compute node — login node OOMs):**
```bash
sbatch --array=1 scripts/slurm_eeg_mapping_sweep.sh            # smoke: base/parcels, local feats
sbatch --array=0-7 scripts/slurm_eeg_mapping_sweep.sh          # full grid
# when all 8 JSONs exist in outputs/results/eeg_mapping/:
python scripts/plot_eeg_mapping.py --target_level parcels      # -> outputs/figures/eeg_mapping/*parcels*D2.png
python scripts/plot_eeg_mapping.py --target_level electrodes
```
Long pole = **medium/electrodes** (24 layers × 5-fold CV, wide design). Escape hatch if it drags:
`PCA_VAR=0.95 sbatch --array=6,7 --export=ALL scripts/slurm_eeg_mapping_sweep.sh`.

**STILL TO DO when you pick up (in order):**
1. **[done once fq_bars finished]** Hard-delete the superseded results — run:
   `rm -rf outputs/results/*-mtrf-* outputs/results/mtrf_fitquality_d2_blocks10.json`
   (matches only the old mTRF dirs; **keeps** `*-probe-group-*` and `*-delta-t*`). _If I already ran it,
   skip._ Decide separately whether the probe-group (Workstream B) dirs also go.
2. **broderick→`train_*`/D2 code rename** of the two kept MMN drivers (`insilico_mmn.py`,
   `insilico_mmn_electrodes.py`) + dedupe them onto `eeg_targets.py`. The sweep code is already clean;
   these are the remaining offenders (cf. §19 TODO).
3. Submit the sweep (run order above); eyeball the two figures per level.
4. Feed the **CV-chosen layer per model** into the electrode-MMN screen (the chosen layer replaces the
   inherited `blocks.10` — that was the "isn't the layer assumed?" point; now it's derived on D2).

**OPEN DECISION (default chosen, change if you disagree):** layer selection uses **5-fold CV** over
train stimuli (mean r over targets). Robust but ~6× fits/layer = the main cost. Swap to a single
held-out validation split for speed via fewer folds if the sweep is too slow.

**Status at handoff (2026-06-17 ~18:00):** fq_bars (§19a, job 55914970) still running — 3/4 models done
(tiny/base/small results in its log), grinding medium; figure not yet written. That run is the OLD
approach and is now mostly superseded by this §20 pipeline — keep it only as a cross-check.
