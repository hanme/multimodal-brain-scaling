# Handover: Auditory EEG Encoding Models — Sophie

_Living doc, newest material on top. Last updated 2026-06-15._
Feeds the MMN pipeline (`00_schizophrenia_pipeline_Sophie_2026.md`); full technical log in
`aux/project_plan_20260611.md`.

## 📍 Quick map

**Current state (2026-06-15):** mTRF EEG-encoders across the Whisper family. PCA@95 % pipeline
**complete for tiny/base/small** (D1/D2/D3 + transfer); **medium re-running**. Headline: **scaling is
flat** — a bigger Whisper does *not* predict EEG better. The **attention encoder** (a learned
non-linear readout; the code calls it `attn_probe`/"probe", Workstream B) is done; "linear mTRF vs
attention encoder" flips by dataset (D2 looks suspiciously easy — see §14 of the plan).

**Where to find what:**
- **Latest results** → right here: *Family results* (scaling table) + *Results (whisper-base)* (per-parcel detail).
- **For your MMN** → *in-silico MMN + per-parcel predictions* — the HDF5 deliverable and how to score it.
- **GPU gotcha** → *ENV NOTE* (torch cu126 build) just below.
- **Why the method is what it is** → *method change (per-bin → mTRF)* and *Workstream B (learned probe)*.
- **Overflow / PCA saga** → the ⚠️ box in this section (+ `project_plan §15`).
- **Reference manual** (repo layout, datasets, how to run) → numbered §1–§6 at the bottom.

---

## ⚠️ ENV NOTE — GPU / PyTorch CUDA build (read before running anything on a GPU node)

**Workstream B (the learned temporal probe) is gradient-trained and wants a GPU.** When you move
onto a GPU node you may have to fix the PyTorch install first — this bit me on 2026-06-13:

- **The venv is `uv`-managed and has NO `pip` inside it.** `pip list` / `python -m pip list` show
  the *base module's* environment, not the venv (they'll look empty / wrong for torch). Use
  **`uv pip ...`** for everything, and query installed versions with
  `python -c "import importlib.metadata as m; print([f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions() if 'torch' in d.metadata['Name'].lower()])"`.
- **The torch in the venv was a `+cu130` build (compiled for CUDA 13.0).** The GPU nodes here are
  **NVIDIA L40S with driver 560.35.03 = CUDA 12.6**, which is *too old* for a cu130 build. Symptom:
  `torch.cuda.is_available()` returns **False on a node with a perfectly good idle GPU**, and torch
  silently trains on CPU (you'll see a `UserWarning: CUDA initialization: The NVIDIA driver on your
  system is too old (found version 12060)`). The unit tests still pass on CPU, so this is easy to miss.
- **Fix — reinstall torch + torchvision with the matching cu126 build** (pin the same versions so
  nothing else churns; torchvision has its own compiled CUDA ops so it must match torch's CUDA):
  ```bash
  source env.sh
  uv pip install "torch==2.12.0+cu126" "torchvision==0.27.0+cu126" \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    --reinstall-package torch --reinstall-package torchvision
  python -c "import torch, torchvision; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
  ```
  Expected: `cuda True | NVIDIA L40S`. (Adjust the version pins to whatever is currently in the
  venv; the cu126 index also serves cu124/cu121 if 12.6 ever changes — any cu12x build works with
  the 12.6 driver.) **Sanity rule: if `torch.cuda.is_available()` is False on a GPU node, check the
  torch build tag vs `nvidia-smi`'s CUDA version before debugging anything else.**

---

## ⭐ RESULTS (2026-06-14) — multi-dataset encoding D1/D2/D3 + transfer (whisper-base)

The two encoders we compare:
- **mTRF** — linear lagged ridge (closed-form, CPU).
- **attention encoder** — a learned, non-linear attention readout over the lookback window (GPU,
  gradient-trained, best config d_model=64). *(This is what the code/output dirs call `attn_probe` /
  "probe" — same thing, clearer name.)*

Parcel-level held-out r, **blocks.2**, same canonical 4 parcels everywhere (D1 membership, NC
recomputed per dataset), 0.5 Hz HP, 0–800 ms lags. D3 scored on `test_d1`/`test_d2` **separately**
(never pooled).

| trained → tested | frontal | temporal | parietal | occipital |
|---|---|---|---|---|
| mTRF D1→D1 | +0.122 | +0.162 | +0.070 | +0.084 |
| mTRF D2→D2 | +0.158 | +0.195 | +0.120 | +0.127 |
| mTRF D3→D1 test (pooled) | +0.065 | +0.040 | +0.030 | +0.030 |
| mTRF D3→D2 test (pooled) | +0.201 | +0.243 | +0.142 | +0.168 |
| mTRF D1→D2 transfer | +0.037 | +0.017 | +0.008 | +0.005 |
| mTRF D2→D1 transfer | +0.069 | +0.053 | +0.018 | +0.015 |
| **attn-enc D1→D1** | +0.079 | +0.141 | +0.084 | +0.070 |
| **attn-enc D2→D2** | **+0.295** | **+0.363** | +0.246 | +0.227 |
| attn-enc D3→D1 test | +0.068 | +0.065 | +0.033 | +0.039 |
| attn-enc D3→D2 test | +0.270 | +0.300 | +0.225 | +0.261 |

**Findings (whisper-base):**
1. **mTRF replicates** on the independent dataset (D2 ≥ D1 on every parcel). The encoder is not a
   Broderick artifact.
2. **Cross-dataset transfer essentially fails** (D1→D2 temporal +0.017, D2→D1 +0.053). The
   speech→EEG mapping is largely **not shared** across corpora → pooling (D3) is zero-sum (helps D2,
   craters D1). ⚠️ This is also a warning for the MMN: if speech→speech transfer breaks, speech→tones
   (the in-silico MMN) is an even bigger extrapolation.
3. **The "ridge wins" conclusion FLIPS by dataset.** On D1 the attention encoder loses to the mTRF
   (0.141 vs 0.162, the old 23/24 result); on **D2 it crushes the mTRF** (temporal 0.363 vs 0.195,
   ~1.9×) and holds under pooling (D3→D2 0.300). It is **not a data-quantity effect** — D2 has *fewer*
   train windows than D1 (157 vs 252). Something about D2's structure suits the non-linear readout.

> ⚠️ **#3 is suspicious — do not over-claim until verified.** Likely confound: the **splits differ
> structurally**. D1 = split **by run** (separate audiobook segments, zero acoustic overlap). D2 =
> split **by story part, one part held out per audiobook** → test shares the *same audiobook* as
> train; with 10 s-stride overlapping windows the held-out set may be "closer" to training, which the
> high-capacity attention encoder exploits far more than a rigid ridge. **Verification agenda in
> `project_plan_20260611.md` §14.**

**Whisper-family status (2026-06-15):**
- **attention encoder d2/d3** (GPU; dirs `…-probe-group-…`): complete for tiny/base/small/medium.
- **PCA mTRF** (the consistent all-sizes pipeline, incl. D3): **tiny/base/small COMPLETE**
  (d1/d2/d3 + both transfers); **medium re-running** with the covariance_eigh fix (see ⚠️ c).
- **Raw-feature mTRF** (eigen reference): tiny/base complete incl. d3; small d1/d2 done (raw d3
  impossible); medium d2 done, raw d1+xfer still grinding (slow — redundant now, PCA≈raw).
- All `MODEL_ID`-parameterized (large needs extraction): `slurm_mtrf_parcels.sh`,
  `slurm_cross_mtrf.sh` (CPU), `kuma_probe_d2d3.sh` (GPU); features resolve local-else-Sophie's-tree
  via `scripts/_whisper_features.sh`; reader `mbs.analysis.compare_encoders`. (Run details in §5.)

### ⭐ FAMILY RESULTS — does a bigger audio model predict EEG better? (best-layer mean parcel r)

| size | D1 | D2 | D3·test_d1 | D3·test_d2 |
|------|------|------|------|------|
| tiny   | 0.132 | **0.232** | 0.068 | **0.223** |
| base   | 0.136 | 0.164 | 0.079 | 0.182 |
| small  | 0.134 | 0.165 | 0.077 | 0.181 |
| medium\*| 0.135 | 0.163 | 0.071 | 0.148 |

\*medium = partial pre-fix run; winning D1/D2 layers are inside the computed range, full re-run in flight.

**Findings (PCA@95%, all sizes):**
1. **Scaling is FLAT.** D1 ≈ **0.135 for every size** (tiny = medium). A bigger Whisper does **not**
   predict EEG better — the headline for the scaling question, and it's clean.
2. **PCA ≈ raw.** Where both exist they match (small D2: PCA 0.165 vs raw 0.167; base D2 0.164 vs
   0.161) — PCA@95% is a faithful, fairer substitute, not a confound. (So the slow raw-eigen medium
   jobs are redundant.)
3. **Best layer scales with depth** — D1 winner blocks-0(tiny)→4(small)→8(medium); D2 1→10→19. Always
   ~the early third of the network.
4. **tiny is the D2 outlier** (0.232 vs ~0.164) → *inverse* scaling on D2 only. With the
   attention-encoder-wins-on-D2 result, this again points at D2's split being "easy" (§14) — treat
   D2 cautiously.
5. **Pooling (D3) doesn't help.** D3·test_d1 (~0.07) ≪ D1-alone (~0.135) — pooling *hurts* D1
   prediction; D3·test_d2 ≈ D2-alone. (Consistent with the transfer near-failure: the two datasets
   don't reinforce each other — a ⚠️ for any plan to pool speech datasets for the MMN.)

> ⚠️ **mTRF segfault on wide models — three-stage fix (eigen → PCA → covariance_eigh PCA).** All the
> same root cause: a **LAPACK workspace integer overflowing int32** at large dimensions.
> (a) **Wide features** (41 lags × d_model: small 768→31k cols, medium 1024→42k) overflow the
> *svd-of-X* GCV path → first fix `gcv_mode='eigen'` (Gram path; numerically identical, tiny/base
> unchanged). (b) But **pooled D3** has n≈81,800 rows, so eigen's n×n Gram *also* overflows →
> D3 still segfaulted → real fix `--pca_var 0.95`: PCA features to 95 % variance *before* lagging
> (PC count varies by model/layer), shrinking the design to `n_PCs·lags` (model-independent, small) →
> cheap svd path, no overflow on either axis, D3 works, **and a fairer cross-size comparison**.
> (c) But PCA's own default `svd_solver='full'` SVDs the tall [n·T, d] matrix → **overflowed again for
> medium** (d=1024, 378k–613k rows). Final fix: **`svd_solver='covariance_eigh'`** — eigendecompose
> the d×d covariance instead (n-independent, can't overflow, faster). tiny/base/small unaffected
> (identical result); only medium re-runs. Selectable via `PCA_VAR=0.95 sbatch …`; raw vs PCA in
> separate dirs (`-mtrf-parcels[-pca]-<tag>`), each layer records `n_pcs`. Full write-up:
> `project_plan §15`.

**Code (built test-driven this session, NOT git-committed):**
`evaluate_features_mtrf_parcels.py` (now with `--pca_var`; `fit_parcel_mtrf` returns
`{model, mu, sd, pca}` and `score_parcel_mtrf` re-applies the stored PCA),
`evaluate_cross_dataset_mtrf.py` (`--pca_var`, source PCA re-applied to target),
`evaluate_features_attn_probe_temporal.py` (per-dataset), `analysis/compare_encoders.py`; helpers in
`attn_probe/dataset_temporal.py`. Tests: `test_mtrf_parcels.py` (incl. 2 new PCA tests:
reduces-&-recovers, no-PCA-by-default), `test_cross_dataset_mtrf.py`, `test_compare_encoders.py`,
per-dataset probe test — all pass. ⚠️ disk quota bit us once (`errno 122`); drivers truncate the
output h5 on `--overwrite`.

---

## What this is

This repo is the computational backbone for turning audio models (Whisper, wav2vec2, AST, VGGish)
into **EEG encoding models**: linear mappings from a model layer's activations to recorded EEG
electrode signals. Once trained, such a model can tell us:

- Which layer of which audio model best predicts human EEG responses to sound
- Whether that model-to-brain alignment differs between healthy controls and schizophrenia patients
- Which layer Sophie's MMN unit selection should target, grounded in actual neural data

---

## ⭐ UPDATE 2026-06-12 — in-silico MMN sweep + per-PARCEL prediction data for your MMN metric

**Read this first if you want the model's predicted EEG to run your MMN-detection metric on.**
This is the in-silico MMN: we feed your literature MMN stimuli through whisper-base, apply the
Broderick→EEG mTRF mapping (from the update below), and get a **predicted EEG time course** for
each stimulus. The deliverable for you is a set of HDF5 files with those predictions at the
**coarse 10-20 parcel level** (Kadir's clusters), spanning the full clip — i.e. a long flat
pre-stimulus baseline plus the post-onset response — so you can compute "does this show an MMN?"
exactly as you did at the model-unit level (baseline-normalised negative peak on deviant−standard,
flat before onset, dip ~200 ms after).

### What we ran
- **8 MMN methods, all identity-design** (the final/critical tone is *physically identical* in
  standard and deviant; the deviance is in the preceding **context** frequency). They span
  deviance size and BOTH directions, with three matched up/down mirror pairs:

  | method dir | context→final | direction | |Δ| |
  |---|---|---|---|
  | `method_37` | 1050→1000 Hz | DOWN | ~4% |
  | `method_12` | 1200→1000 Hz | DOWN | ~17% |
  | `method_44` | 1000→633 Hz | DOWN | ~37% |
  | `method_09` | 1000→600 Hz | DOWN | ~40% |
  | `method_55` | 2000→1000 Hz | DOWN | ~50% |
  | `method_12_counter` | 1000→1200 Hz | UP | ~20% |
  | `method_44_counter` | 633→1000 Hz | UP | ~58% |
  | `method_55_counter` | 1000→2000 Hz | UP | octave |

  (m12/m44/m55 appear in both directions = artifact control: does the MMN scale with deviance
  and flip/behave consistently across direction?) Stimuli are copies (not symlinks) of your
  pre-generated whisper set under `…/scz_updated_pipeline_071226/data/audio_outputs_literature/`.
- **6 whisper-base layers** (blocks.0–5). Mapping fit once per layer, applied to all 8 methods.

### Parcels, not single electrodes — and NC-based channel exclusion (important)
We aggregate electrodes to **Kadir's coarse clusters** and exclude unreliable channels first
(precedent: Kadir's `evaluate_features_committed_layers.py` drops units with reliability `r ≤ 0.1`
before fitting). We use **r > 0.2** — but it doesn't matter exactly, because Broderick has a hard
reliability gap (every channel is either r ≥ 0.34 or r ≤ 0.16). Each parcel = **raw average of its
surviving channels**:

| parcel | channels kept (r>0.2) | dropped |
|---|---|---|
| frontal | Fz, F3, F4 | FCz |
| temporal | T7, T8 | — |
| parietal | Pz, P7, P8 | P3, P4 |
| occipital | Oz, O2 | O1 |
| ~~central~~ | **none** (Cz=0.16, C3=0.09, C4=0.00) | **parcel dropped** |

So you get **4 parcels**. Central drops out entirely — Broderick has no reliable central EEG under
passive audiobook listening. (Note: the earlier huge "Cz MMN" was a junk channel, raw std ~19,000,
r ~0.16 — NOT an NC-normalisation artifact; the EEG is not NC-normalised.) ⚠️ The exact NC
definition for our cross-subject (no-repetition) data is still an open question — see your Q2 to
Kadir in `aux/sophies_questions_20260611.txt`. For now this is fine; selection is robust to it here.

### ⭐ Where the prediction data is (this is what you asked for)
```
outputs/insilico_mmn_predictions/predictions__blocks.{0,1,2,3,4,5}.h5
```
One file per layer. Each file (RAW = NOT baseline-corrected; you choose the baseline window):
```
attrs: layer, highpass_hz (0.5), lag_max_ms, fs (50), time_step_ms (20), nc_r_threshold (0.2), note,
       heldout_test_windows, heldout_n_samples           (held-out eval, see section below)
parcels          [4]            (e.g. b'frontal' b'temporal' b'parietal' b'occipital')
parcel_members   [4]            (e.g. b'Fz+F3+F4')
parcel_nc_r      [4]            (cross-subject reliability of the parcel, r-scale)
heldout_r        [4]            out-of-sample Pearson r of the mapping on held-out TEST runs
heldout_r_nc     [4]            same, divided by parcel_nc_r (noise-ceiling-normalised)
<method>/                       (one group per method, names as in the table above)
   time_ms       [n_t]          0 = final/critical-tone onset; NEGATIVE = before onset (the baseline)
   standard      [n_t, 4]       raw predicted EEG, standard clip,  columns = parcels
   deviant_mean  [n_t, 4]       mean over the 15 deviant variants
   deviants      [15, n_t, 4]   per-variant (N3/N5/N7 × 5), if you want trial-level stats
   deviant_ids   [15]           variant ids
   attrs: context_final, direction, final_tone_onset_s, n_deviants
```
`n_t` ≈ 1474 bins (the whole ~29.4 s clip minus the lag margin), so `time_ms` runs from roughly
−29,000 ms up to ~+200 ms — i.e. a very long flat baseline before onset, then the response.

**Compute the MMN metric (parcel × method):**
```python
import h5py, numpy as np
with h5py.File("outputs/insilico_mmn_predictions/predictions__blocks.3.h5") as h:
    parcels = [p.decode() for p in h["parcels"][:]]          # ['frontal','temporal',...]
    g = h["method_09"]
    t   = g["time_ms"][:]                                    # [n_t]
    mmn = g["deviant_mean"][:] - g["standard"][:]            # [n_t, 4]  (= your D_A)
# baseline-normalise on the pre-onset window, then negative-peak in the MMN band
base = (t >= -150) & (t < 0)
mmn_bc = mmn - mmn[base].mean(0, keepdims=True)
band = (t >= 100) & (t <= 250)
neg_peak = mmn_bc[band].min(0)                               # most-negative value per parcel
# 'shows an MMN' ≈ neg_peak << 0 with a flat (mmn_bc[base] ≈ 0) baseline
for p, v in zip(parcels, neg_peak):
    print(f"{p:9s} 100–250 ms negative peak = {v:+.3f}")
```
This mirrors your unit-level pipeline: `D_A = Y_dev − Y_std`, baseline-normalised negative-peak
metric per row, then (across the 8 methods / 24 deviants) a one-sample t-test per parcel if you
want significance. With only 4 parcels the old "top-5% of units" selection becomes "which parcels
show the effect" — and the high-NC ones (temporal r≈0.86, parietal/frontal) are the ones to trust.

### ⭐ NEW (2026-06-13) — held-out validation: is the mapping actually any good?

Until now we *fit* the Broderick→EEG mapping but never reported an out-of-sample score, so there
was no number telling us the predicted EEG is faithful rather than overfit. We now do exactly what
Kadir does in `multimodal-brain-scaling_Kadir_orig/.../evaluate_features_committed_layers.py`
(fit on `train`, predict on a held-out `test`, report Pearson r).

- **The split is built into the data, by audiobook run.** `broderick2018_30s.h5` already carries
  `splits=['train','test']` with `test_runs=[2,9,13,14]`: 252 train windows (16 runs) vs **62 test
  windows from 4 entirely separate runs** (audio02/09/13/14). Because train/test are different runs,
  there is **zero acoustic overlap** — no leakage, and no need to shave a buffer around a cut point.
  (Windows are 30 s with 10 s stride, so they overlap *within* the train set, but never reach test.)
- **No leakage in preprocessing:** test features are standardised with the **train** mean/std, and
  the MMN forward model is still fit on the **full** train set (the eval doesn't shrink it).
- **What we report:** per-parcel out-of-sample Pearson r between predicted and recorded EEG, pooled
  along time over the test windows (`heldout_r`), plus the noise-ceiling-normalised version
  `heldout_r_nc = heldout_r / parcel_nc_r`. Stored as top-level datasets in each `predictions__<layer>.h5`.

**Results — out-of-sample r per parcel × layer** (62 test windows, 24,800 pooled time samples;
all 6 layers complete, job 55104689). Each cell is `heldout_r` (and `heldout_r_nc` below it):

| layer | frontal | temporal | parietal | occipital |
|---|---|---|---|---|
| blocks.0 | +0.134 | +0.186 | +0.099 | +0.109 |
| blocks.1 | +0.131 | +0.179 | +0.088 | +0.104 |
| blocks.2 | +0.139 | +0.181 | +0.088 | +0.106 |
| blocks.3 | +0.123 | +0.167 | +0.070 | +0.086 |
| blocks.4 | +0.134 | +0.174 | +0.071 | +0.090 |
| blocks.5 | +0.140 | +0.172 | +0.073 | +0.089 |

NC-normalised (`heldout_r_nc = r / parcel_nc_r`), same layout:

| layer | frontal | temporal | parietal | occipital |
|---|---|---|---|---|
| blocks.0 | +0.245 | +0.216 | +0.142 | +0.156 |
| blocks.1 | +0.239 | +0.207 | +0.126 | +0.149 |
| blocks.2 | +0.254 | +0.210 | +0.126 | +0.151 |
| blocks.3 | +0.224 | +0.193 | +0.100 | +0.123 |
| blocks.4 | +0.245 | +0.202 | +0.101 | +0.128 |
| blocks.5 | +0.255 | +0.199 | +0.105 | +0.127 |

All values are **positive across every parcel and layer** — the mapping generalises to unseen runs
rather than overfitting — and modest, as expected for group-level EEG encoding. Auditory-sensible
ordering: **temporal** strongest in raw r (≈0.17–0.19), **frontal** strongest after NC-normalisation
(≈0.24–0.26, since frontal's noise ceiling is lower so its r counts for more); parietal/occipital
weakest. Layer differences are small and flat across blocks.0–5 (slight early-layer edge in raw r) —
consistent with the encoding-quality curve being shallow for whisper-base on this data. Treat these
as the credibility weight on each parcel's MMN: an MMN in **temporal** (high NC r≈0.86 *and* the best
raw held-out r) or **frontal** is far more trustworthy than one in a parcel that barely predicts
held-out EEG.

To re-read them yourself: `h5["heldout_r"][:]` / `h5["heldout_r_nc"][:]` (same parcel order as
`h5["parcels"]`). Toggle with `--eval_heldout false`; sampling density via `--n_eval_time_samples`.

### Figures (visual version of the same data)
`outputs/figures/insilico_mmn/insilico_mmn__<method>__blocks.<L>.png` — 48 figures (8 methods × 6
layers), rows = the 4 parcels (each its **own y-scale**, annotated with NC + member channels),
columns = deviant / standard / deviant−standard, shaded 100–250 ms band, x = time from final-tone
onset. Some already show the expected ~200 ms negative dip. **Read the high-NC parcels** (temporal,
parietal, frontal); ignore raw amplitude on low-NC ones.

### Code / how to regenerate
- `scripts/insilico_mmn.py` — fits NC-masked parcels, loops methods, writes both the figures and
  the `predictions__<layer>.h5`. Mapping is fit once per layer (depends only on layer+parcels, not
  on method) and applied to all 8 methods.
- `scripts/slurm_insilico_mmn.sh` — `sbatch --array=0-5` = the 6-layer scan.
- `scripts/slurm_mmn_extract.sh` — `sbatch --export=ALL,MMN_METHOD=method_XX --array=0-15` extracts
  the whisper-base delta_T features for a method's 16 stimuli.
- The `.h5`/`.wav`/`.png` outputs are gitignored (data, not code); they live on disk at the paths above.

---

## ⭐ UPDATE 2026-06-11/12 — method change (per-bin Ridge → mTRF) + first de-confounded results

**Read this first — it supersedes the Phase 4b per-time-bin approach (§2b, §3e below) for
continuous speech, and overturns the old "blocks.2 is best" conclusion.** Triggered by a
conversation with Kadir (2026-06-11) and a clean re-analysis. Full running log:
`aux/project_plan_20260611.md`.

### Why the method changed
The old temporal evaluator (`evaluate_features_temporal.py`) fit an *independent* Ridge at each
20 ms bin, `feature[t] → EEG[t]`, scored by correlating across stimuli at a fixed within-window
offset. Three problems for continuous speech:
1. **Zero lag** — predicts EEG[t] from the model feature at the *same* instant, but auditory
   cortex lags the stimulus ~50–200 ms.
2. **No weight sharing** — 1500 separate fits; noisy and wasteful.
3. **Wrong score axis** — correlating across overlapping 30 s segments at a fixed offset mixes
   unrelated moments; the field standard correlates *along time*.

### New method: mTRF (lagged shared-weight Ridge) — "Workstream A"
`src/mbs/evaluation/evaluate_features_mtrf.py`
(`python -m mbs.evaluation.evaluate_features_mtrf`). One Ridge shared across all (stimulus, time),
predicting `EEG[t]` from features at an explicit lag `feature[t-lag]`, scored *along time*.
`single_lag` mode sweeps lags → an encoding-vs-latency curve; `fir` mode is the full multi-lag
mTRF (one r/channel, literature-comparable). This is the standard method of the Broderick/Lalor
lab whose data we use. All electrodes fit at once with `alpha_per_target=True` (per-channel
regularization — no low-SNR contamination — and ~67× faster). Tests:
`tests/test_evaluate_features_mtrf.py` (11, incl. a synthetic lag-recovery test).

These three are points on ONE spectrum: per-bin (old) → **mTRF (linear, Workstream A)** →
learned temporal probe (Kadir's MIRAGE / the `attn_probe/` package = Workstream B target;
gradient-trained shared trunk + per-subject heads). See `aux/project_plan_20260611.md` §0–§3.

### Key finding: the lag curve was FLAT — a slow-drift confound, fixed by high-pass
On raw EEG the encoding-vs-lag curve is flat (~0.48 at every lag): both the model features and
the group-averaged EEG are dominated by slow envelope structure autocorrelated over >400 ms, so
any lag predicts equally. **High-passing the EEG and features removes the drift and reveals a
proper auditory TRF.** ⚠️ This *fitting/scoring* high-pass is a DIFFERENT issue from the
significance-test n_eff autocorrelation correction in §"Phase 4b results" (`plot_score_distributions.py`)
— keep them distinct.

### De-confounded results (whisper-base × Broderick, uncorrected r; jobs 55037406 + 55039297)
High-pass cutoff sweep 0.5 / 1 / 2 Hz × 6 layers (all electrodes):
- **Latency ~120–140 ms** at temporal electrodes (TP7/T7/FT7/T8) — classic N1-like auditory
  response — robust across all cutoffs. Frontal (Fz/AF3) later and weaker.
- **0.5 Hz** retains the most signal (peak r ~0.12 mean over auditory electrodes, up to 0.18 at
  FT7); 2 Hz over-filters.
- **Best layer = mid-depth (blocks-3), robust across cutoffs.** Layers 2–5 are ~tied and clearly
  beat the early layers; this **overturns** the old per-bin "blocks-2" and the confounded
  no-highpass metric (which favored the earliest layer).
- Figures: `outputs/figures/mtrf_highpass_diagnostic.png` (flat vs high-passed),
  `outputs/figures/mtrf_cutoff_layer_summary.png` (layer comparison + spatial latency).

### Recommended config, caveats, open items
- **Config:** 0.5 Hz high-pass, `single_lag`, uncorrected r, mid-depth (~blocks-3).
- **NC under high-pass:** the stored NC was computed on raw EEG and is invalid for high-passed
  data, so we report **uncorrected r** (Kadir explicitly OK'd this). Recomputing NC on
  high-passed EEG is an optional later refinement.
- **Caveats:** whisper-base only; Broderick only (not yet replicated on a 2nd dataset); the
  mid-depth layer differences are small.
- **Next:** in-silico MMN (feed Sophie's stimuli through whisper blocks-3, apply the ridge, look
  for a deviant−standard response at ~120–200 ms); scale to other whisper sizes / wav2vec2; or
  start Workstream B (learned probe) and compare to this mTRF baseline.

### New files (this update)
- `src/mbs/evaluation/evaluate_features_mtrf.py`, `tests/test_evaluate_features_mtrf.py`
- `scripts/slurm_mtrf.sh` (cutoff×layer sweep), `scripts/plot_mtrf_scores.py`

---

## ⭐ UPDATE 2026-06-13 — Workstream B: the attention encoder (MIRAGE-style): **ridge wins at this data scale**

> _Terminology: "attention encoder" = the learned non-linear readout this section calls the
> **probe** (its code package is `attn_probe`). Renamed for clarity; "probe" below = attention encoder._

**Bottom line: the gradient-trained attention encoder does NOT beat the linear mTRF on Broderick
(this was the D1-only conclusion — later flipped on D2, see the RESULTS section up top).
The mTRF (Workstream A) stays the reportable method.** This closes the encoding-method comparison.

### What we built and ran
The probe is the MIRAGE-style counterpart to the mTRF: instead of a closed-form lagged Ridge, a
shared `LatentAttentionTrunk` attends over the lookback window (kept as a token sequence) and a
readout head predicts the same 4 NC-parcels. Trained with `1 − Pearson`, scored **along time on
the same held-out runs** — i.e. **the identical `heldout_r` metric the mTRF reports**, so the two
sit side by side. Code: `src/mbs/evaluation/evaluate_features_attn_probe_temporal.py` +
`attn_probe/dataset_temporal.py` + `attn_probe/engine_temporal.py`; tests
`tests/test_attn_probe_temporal.py` (17, incl. a synthetic planted-lag learning proof — all pass
on GPU). The `--readout_level {group,individual}` flag toggles a single group head vs one head per
subject over a shared trunk. We ran the **group** variant (group-averaged EEG, fully comparable to
the mTRF).

### Head-to-head — held-out r, probe (best config) vs mTRF, all 6 layers × 4 parcels
Probe config after a small capacity/regularization sweep (see below): `d_model=64, num_latents=4,
1 cross-attn layer, weight_decay=1e-2, dropout=0.3, 0.5 Hz high-pass, 0–800 ms lookback`. Each cell
is **probe / mTRF**; **bold = winner**:

| layer | frontal | temporal | parietal | occipital |
|---|---|---|---|---|
| blocks.0 | 0.065 / **0.134** | 0.119 / **0.186** | 0.075 / **0.099** | 0.060 / **0.109** |
| blocks.1 | 0.062 / **0.131** | 0.119 / **0.179** | 0.069 / **0.088** | 0.058 / **0.104** |
| blocks.2 | 0.079 / **0.139** | 0.141 / **0.181** | **0.084 / 0.088** (tie) | 0.070 / **0.106** |
| blocks.3 | 0.053 / **0.123** | 0.119 / **0.167** | **0.074 / 0.070** | 0.064 / **0.086** |
| blocks.4 | 0.056 / **0.134** | 0.116 / **0.174** | 0.067 / **0.071** | 0.063 / **0.090** |
| blocks.5 | 0.066 / **0.140** | 0.116 / **0.172** | 0.060 / **0.073** | 0.056 / **0.089** |

**mTRF ≥ probe in 23 of 24 cells.** The probe only reaches parity on **parietal** (tied at
blocks.2, a hair ahead at blocks.3); it trails by ~25–35 % on temporal and ~50 % on frontal.
Both methods are layer-flat; the probe's best layer is blocks.2, matching the adopted mTRF choice.

### Why — it's overfitting, and it's a hard ceiling at 252 windows
The default-size probe (`d_model=256, 16 latents, 2 layers`) reached **train r ≈ 0.75 but held-out
r ≈ 0.06** — a ~0.69 train/val gap. A capacity×regularization sweep on blocks.2 showed a clean
monotonic relationship (smaller model → higher train loss → **higher** held-out r):

| config | train loss | held-out temporal r |
|---|---|---|
| 256/16/2 (default) | 0.20–0.25 | 0.07–0.09 |
| 128/8/1 | 0.35 | 0.126 |
| **64/4/1 (adopted)** | **0.43** | **0.141** |
| 32/2/1 | higher | 0.122 (starts underfitting temporal/parietal) |

Capacity — not weight decay — is the dominant lever (the full-size model with heavy reg was the
*worst*). Even the best, smallest probe still has a large train/val gap and lands below ridge.
This is the predicted outcome at this data scale (252 train windows): RidgeCV's strong closed-form
L2 regularizes better than a learned nonlinear model can with this little data. MIRAGE's reported
gains over ridge come with far more data + multi-stream fusion than we have on single-dataset
Broderick.

### Decision / scope
- **Method selected: mTRF (Workstream A).** The probe is implemented, validated, and benchmarked,
  but is not the reportable encoder here.
- **Deferred (not worth it at this scale):** the per-subject `individual` variant (needs a
  `--store_subjects` mode in `format_eeg_hdf5.py`, not built) and the MMN integration of the probe
  (§3.7 of `project_plan_20260611.md` gated this on "probe clearly beats mTRF" — it doesn't).
- **Re-open the probe only if** we add much more data (more datasets / multi-stream features) — the
  code is ready and the comparison harness is apples-to-apples, so it's a cheap re-run later.
- Results: `outputs/results/whisper-base-probe-group-r2-all/attn_probe_temporal_scores.h5`
  (all 6 layers, group). See also `aux/project_plan_20260611.md` §11 (2026-06-13) for the full log.

---

<!-- ════════════ REFERENCE MANUAL (stable background; §1–§6) ════════════ -->
_Everything above is the dated work log (newest first). Everything below is the stable reference:
repo layout, datasets, the hook architecture, and how to run the pipeline._

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

> ⚠️ **SUPERSEDED for continuous speech (2026-06-12).** This independent-per-bin, zero-lag,
> across-stimulus-at-fixed-offset scoring is replaced by the mTRF (`evaluate_features_mtrf.py`).
> See the "UPDATE 2026-06-11/12" section above.

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
*(Phase 4b, status: **Done**, but **SUPERSEDED for continuous speech (2026-06-12)** by the mTRF
— `src/mbs/evaluation/evaluate_features_mtrf.py`; see "UPDATE 2026-06-11/12" above. Still valid
only for discrete ERP/MMN epochs, where per-latency-across-stimuli IS the evoked waveform.)*

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
| WA | mTRF evaluator `evaluate_features_mtrf.py` + 11 tests | **Done** (2026-06-12) — replaces per-bin eval for continuous speech |
| WA | mTRF full sweep + high-pass cutoff sweep (whisper-base) | **Done** (2026-06-12) — jobs 55037406, 55039297; mid-depth (~blocks-3) best, latency ~120–140 ms at 0.5 Hz HP |
| WA | `scripts/slurm_mtrf.sh`, `scripts/plot_mtrf_scores.py` | **Done** (2026-06-12) |
| WB | Learned temporal probe (adapt `attn_probe/` to time) | **Done (group)** (2026-06-13) — built + 17 tests + 6-layer benchmark; **ridge wins, mTRF is the method** (see ⭐ UPDATE 2026-06-13 above). Individual variant + MMN integration deferred. |
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
