# Temporal EEG Encoding — Project Plan (2026-06-11)

Author: drafted with Claude for H. Mehrer, after the 2026-06-11 conversation with Kadir Gökce.
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
| **(1) Per-bin Ridge** | separate per time bin | none (zero-lag) | closed-form | across stimuli @ fixed offset | current `evaluate_features_temporal.py` |
| **(2) mTRF** | one set, shared across time | window (e.g. 50–800 ms) | closed-form ridge | along time | **target baseline** |
| **(3) Learned temporal probe** | shared trunk + subject heads | window | gradient (SGD) | along time | `attn_probe/` (vision-shaped today) = **target method** |

**Decision:** retire (1) as a primary metric; build (2) as a fast, literature-comparable
baseline; then adapt the existing `attn_probe/` into (3), which is Kadir's current method
(MIRAGE, Gökce/Al-Khamissi 2026) and reportedly beats ridge.

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

## 2. Workstream A — closed-form mTRF baseline (do first)

Goal: a fast, defensible, literature-comparable number this week. New file
`src/mbs/evaluation/evaluate_features_mtrf.py` (leave `evaluate_features_temporal.py` in place,
re-scoped to ERP/MMN only).

**Design matrix (lagged / FIR).** For each stimulus segment and each output time `t`, stack
features over a lag window `L = {l_min … l_max}` (start 50–800 ms; at 50 Hz that's lags ~3–40
bins). Predict `EEG[t]` from `[feature[t−l] for l in L]`. One **shared** `RidgeCV` across all
`(segment, t)` rows → one weight set per (layer, subject, ROI). This is the mTRF / Crosse-style
forward model.

**Scoring.** Predict held-out runs, **concatenate predictions over time**, correlate vs actual
EEG **along the time axis**, per channel. Optionally NC-correct via `sqrt(nc/100)` after
averaging NC over the scored window.

**Housekeeping (apply here and in B):**
- Mask channels with NC below a threshold *before* fitting (mirror Kadir's `> 0.1`; expose as a
  flag). Operate on the curated high-NC electrode set.
- Restrict lags/latencies to the 50–800 ms window.
- Collapse `score` to a scalar for layer ranking by **window-average or peak over latency**, not
  a flat mean over all 1500 bins.
- Keep the existing run-level train/test split.

**Decisions to confirm before coding (see §7):** exact lag window and step; whether to z-score
features per dimension; ridge alpha grid (reuse `ALPHA_LIST_SHORT`); single-output-per-channel
vs joint multi-output ridge.

**Deliverable:** `temporal_mtrf_scores.h5` (`score[lag-or-window, n_ch]` per layer/subject/ROI)
+ summary JSON; one figure: held-out r vs latency for Fz, one curve per layer; layer ranking
table vs the Phase-4a (mean-pooled) ranking.

---

## 3. Workstream B — learned temporal probe (target method)

Adapt the existing `attn_probe/` from vision (one vector per stimulus) to time-resolved EEG.
This is method (3) / MIRAGE-style. Three concrete edits, smallest-first:

**B1. Time-resolved target in `dataset.py`.**
- `SingleRoiSubjectDataset` currently returns `y = [N_subject]`. Change to return the EEG over a
  short output window `y = [T_out, N_subject]` (or per-time-point samples; see B3).
- Keep stimulus→feature alignment by ID (`align_feature_indices`) unchanged.
- Carry per-(time, channel) NC `[T, N_subject]` through for masking + NC-correction.

**B2. Tokens = time, in `model.py`.**
- Feed `feats = [T_window, C]` (already produced by `delta_t`) so `TokenAdapter` (ndim==3 path)
  tokenizes over **time**; enable a temporal position encoding (`pos_mode="sin"` or `"learned"`
  already exist; MIRAGE uses learned absolute + rotary).
- Trunk attends over the lookback window → reads out across time. `SubjectHeadBank` stays
  (one linear head per subject; "one head per subject" = as-is).

**B3. Loss + autocorrelation handling in `engine.py`.**
- Switch loss from plain MSE to **Pearson-over-time** (or `1 − corr`) to match the scoring
  objective and MIRAGE (paper line 113). `RunningPearson` already gives the eval metric.
- **Randomly sample time points** per batch (Kadir's key trick) instead of feeding contiguous,
  highly autocorrelated frames — sample a set of output times `t` and their lag windows.
- Evaluate by correlating predicted vs actual **along time** on held-out runs, NC-corrected,
  averaged over channels then subjects (the engine already averages r over subjects).

**Decisions to confirm (see §7):** output window length `T_out` and lookback length;
shared-trunk-across-subjects vs per-subject; probe capacity (`d_model`, `num_latents`,
`cross_attn_layers`); whether to also share one trunk across ROIs.

**Deliverable:** trained probe per (layer, ROI); held-out r-vs-latency and scalar ranking,
compared against the Workstream A mTRF baseline (this comparison is the scientific payoff:
does the learned temporal encoder beat ridge, as MIRAGE claims).

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

## 7. Open decisions (resolve before coding each workstream)

1. **Lag window & resolution** for the mTRF / lookback: 50–800 ms confirmed by Kadir; step size
   (every bin = 20 ms? every 2nd?) and whether to taper.
2. **Feature standardization:** per-dimension z-score features before ridge? (likely yes.)
3. **Channel set:** NC threshold value and whether to fit only the curated high-NC electrodes.
4. **Scalar collapse for ranking:** peak vs window-average over 50–800 ms.
5. **Probe scope (B):** one trunk across subjects (yes) and across ROIs (?); probe capacity.
6. **Loss (B):** Pearson-over-time vs MSE; output window length.
7. **Exact NC definition** (⚠️ flagged in §4): subject-split vs other unit; across-window-at-fixed-
   latency correlation vs alternatives; comparability scale vs visual benchmarks. *Fine for now;
   discuss before reporting NC-corrected numbers.*

---

## 8. Milestones

- **M1 (this week):** Workstream A mTRF baseline runs on Broderick/Whisper-base, produces
  r-vs-latency for Fz + layer ranking. Sanity vs literature (Broderick/Lalor numbers).
- **M2:** `evaluate_features_temporal.py` re-scoped/documented as ERP/MMN-only; A is the default
  continuous-speech metric.
- **M3:** Workstream B B1–B3 implemented; probe trains on one (layer, ROI); beats or matches A.
- **M4:** scale B across layers/models (after §6 backbone time-alignment); MMN endpoint.

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
with recoverable auditory latencies. Layer-selection question answered for whisper-base
(mid-depth, ~blocks-3). Recommended config: 0.5 Hz high-pass, uncorrected r, single_lag.

**Pick up here:**
1. Optional rigor: recompute NC on 0.5 Hz-high-passed EEG → NC-corrected latency numbers
   (or keep uncorrected r per Kadir). The `fir` mode for a single literature-comparable r.
2. **In-silico MMN** (the payoff): feed Sophie's MMN stimuli through whisper blocks-3, apply the
   trained ridge, test for a deviant-minus-standard response at ~120-200 ms at Fz/FCz.
3. Scale: other whisper sizes / wav2vec2 (needs the §6 time-alignment for non-whisper).
4. Or start Workstream B (learned probe) and compare against this mTRF baseline.
