# Handoff — CP2: per-model MMN waveform comparison (own top-5 literature vs own top-5 Phase-2)

**Status:** complete, 2026-08-17. **Nothing committed was modified.** All artefacts are new files
under `aux/analysis_presentation/`.
**Entry point:** `aux/analysis_presentation/per_model_top5.py`.
**Full argued write-up:** `aux/analysis_presentation/per_model_top5_recommendation.md` (this file is
the machine-readable summary of it; that file is the one to read for the reasoning).

---

## 1. What was asked, and the constraints it came with

Replace the main-deck head-to-head slide, which averaged the MMN difference wave across models.
The objection driving it, verbatim: *"I think we have to show it more on the per model basis, at
least like not not show it like the average here. I think the average is confusing in the sense
that like people want to know how this is composed."* The averaged version moves to the appendix.

Two decisions were pre-made and are **not open**:

1. **Ranking key = each model's OWN response**, not cross-model consensus: *"you pick the
   frequencies that … you do the literature stuff, and then you look those take those that look
   best or have the largest trough … for that specific model."*
2. **Panel size = top FIVE per row** (written on screen during the call).

Framing was **sequencing, not rejection**: the consensus-set-viewed-per-model version was
*deferred*, not ruled out — *"that I think we can do later, but first we need to pick basically two
models or do two slides."* **The consensus figures must not be deleted.** They are intact.

Presentation-driven model choice was explicitly sanctioned (*"we can cherry pick here … present
those that look more like an MMN"*) but **no model was named**. wav2vec2-large came up only as a
worked example and was **not** to be assumed. It was evaluated and is **not** recommended (§6).

---

## 2. Repository finding that motivated new code (confirmed, not assumed)

`aux/analysis_novel_search/plots/` already emits `literature_waveforms__<model>.png` and
`phase2_waveforms__<model>.png`. **These show the CONSENSUS ranking de-overlaid by model — the same
ten stimuli in every model's figure.** `plot_instance_waveforms_per_model()`
(`phase1_results.py:837`) takes `ranked.head(n)` from a single shared ordering produced by
`novel_search_common.rank_directions()`, which sorts by `n_agree` (how many of the six models pass
S7) then by `mean_uv` over the agreeing models only. A model that dislikes the consensus set is
shown its own weak responses there.

That is a different question from the one asked, so a new figure was built. **Both views are kept.**

---

## 3. Fixed data definitions (inherited from `novel_search_common`, never restated)

| choice | value |
|---|---|
| site | **FCz electrode only** — no parcels at any stage |
| read-out | **mTRF only** — the attention encoder is out of scope |
| models | 6: whisper-{tiny,base,small,medium}, wav2vec2-{medium,large}. **whisper-large excluded**, filtered at load in `load_scored`, so it cannot reach a panel |
| LIT set | 24 literature methods × {regular, counter} = **48 direction-instances** |
| P2 set | 127 pairs (those reaching `n_agree ≥ 5` in Phase 1) × 2 directions = **254 direction-instances** |
| unit | one (model, direction-instance) cell; regular and counter are **separate rows**, never collapsed |
| ranking key | that model's own `trough_uv` at FCz, ascending (most negative wins) |
| criteria | committed **S7** = trough in 100–240 ms **and** ≤ −0.75 µV **and** ≥ 50 % recovery within 120 ms |
| deviance restriction | **none** — this is a head-to-head of best responders, not a dose–response |
| one trace | mean(15 deviants) − standard, mean-baseline-corrected over [−3×SOA, 0), in µV. **No averaging across models anywhere.** |

**Sources on disk** (the only prediction roots that exist in this checkout — the `_pertrial` and
`_figs` siblings named in the overview are **not** present):

- LIT: `outputs/results_soafix_full/mmn_s7_roi.csv` + `outputs/insilico_mmn_predictions_soafix/<model>/electrode_predictions__<layer>.h5`
- P2: `outputs/results_novel_search/phase2_mmn_s7_roi.csv` + `outputs/insilico_mmn_predictions_novel_phase2/<model>/electrode_predictions__<layer>.h5`
- Verified: the `(FCz, mtrf, X=0.75)` slices of `results_soafix_full` and `results_soafix` are
  **byte-identical**, so this analysis and `literature_results.py` read the same numbers.

---

## 4. What was built

`aux/analysis_presentation/per_model_top5.py` (~600 lines). **Nothing is re-scored.** It imports
everything that must not drift and restates none of it:

| imported from | what |
|---|---|
| `novel_search_common` | `SEARCH_MODELS`, `ROI`, `MAPPING`, `DIP_UV_THRESHOLD`, `load_scored`, `rank_directions` |
| `phase1_results` | `load_fcz_waves` (trace construction), `LAYER`, `WAVE_XLIM`, `MMN_WINDOW`, `WAVE_MODEL_STYLE`, `load_literature` + `_with_std_dev` (the direction convention) |
| `analyze_mmn_criteria` | `trace_stats`, `decide` |
| `analyze_mmn_s7_roi` | `WINDOW`, `RECOVERY_MS`, `RECOVERY_FRAC`, `CTRL_WINDOW`, `EDGE_GUARD_BINS` |
| `analyze_mmn_screen_24freq` | `load_semitones` |
| `novel_search_plots` | `MLABEL`, `INK`, `MUTED` |

Shape statistics are computed on the **z** trace (the trace S2 is defined on); `trough_uv` is then
read off the **µV** trace at that argmin — exactly what `analyze_mmn_s7_roi.collect_rows` does.

**Why a new module rather than editing `phase1_results.py`:** that module is phase-scoped (one
source per invocation, under a `phase1_`/`phase2_` prefix) and its figures are cited by path from
`novel_stimulus_search_results.md`. This figure needs *both* sources in one render and a different
ranking key. It sits beside the other presentation deliverable (`trough_distributions.py`) and
follows that directory's `plots/<subdir>/` + `svgs/<subdir>/` tree convention.

---

## 5. Decisions made during the task (each was open; each is now settled)

### 5.1 Literature de-duplication — **5 distinct ORDERED frequency pairs**

The 24 literature methods collapse onto far fewer stimuli: **ten** are 1000→1200 Hz, **five** are
1000→1850 Hz, **two** are 1000→1500 Hz — distinct papers sharing a frequency pair, differing only
in SOA, tone duration or intensity. Across both directions the 48 instances cover only **20 distinct
ordered pairs**.

Undeduplicated this is severe, not theoretical: whisper-tiny's raw literature top 5 puts **three of
five slots** on 1850→1000 Hz; whisper-base's raw top 5 is **all five** the 1000↔1850 pair.

**Rule: five distinct ordered (standard → deviant) pairs, keeping the deepest-troughing instance of
each.** *Ordered*, so 1000→1850 and 1850→1000 remain two stimuli — collapsing them would erase the
direction asymmetry the counterbalancing exists to measure. Applied to the **literature row only**;
the Phase-2 grid has no duplicate pairs by construction, so the flag is left off there to keep that
visible. The un-deduplicated ranking is written in full to `per_model_full_ranking.csv`.

> The existing across-model figure does **not** dedupe — that is why 1000→1850 Hz appears three
> times on it.

### 5.2 y-axis — shared **within** a figure, **never across** models

Each figure shares one y-axis across its ten panels so rows A and B are directly comparable.
A shared axis **across** models is not legible: best troughs span −9.61 µV (whisper-medium) to
−2.65 µV (whisper-tiny), a **3.6× range**; medians span −1.24 µV (whisper-base) to −7.63 µV
(wav2vec2-large), a **6.2× range**. One common axis would compress whisper-base into ~16 % of the
axis height. Per-model scaling is required and each figure says so on its face. `--free_y`
autoscales every panel for inspection only.

### 5.3 Shape-quality quantification (answering the "it slopes down afterwards" objection)

Three quantities, none invented for this figure:

1. **`recovery_frac`** — the S2 quantity: fraction of its own depth the trace climbs back within
   120 ms of the trough. ≥ 0.50 required. Higher = cleaner dip-and-recover.
2. **`s4_specificity`** — the committed S4 criterion: is the scored trough more negative than the
   deepest point of the 300–440 ms control window? **False ⇒ the trace's real minimum is later**, so
   what got scored is the shoulder of a ramp still descending.
3. **`post_slope_uv_per_100ms`** — OLS slope of the µV trace from trough to 360 ms. Positive =
   returns to baseline; **negative = the down-sloping shape that drew the objection**.

**Only 3 of all 60 panels are negative**: whisper-base LIT-1 (−0.06, effectively flat),
whisper-medium LIT-2 (−0.47), **wav2vec2-large LIT-1 (−1.54, the one real offender)**. All 30
Phase-2 panels are positive.

"Clean panel" = passes S7 **and** S4 **and** positive post-trough slope.

---

## 6. Results

### 6.1 Model-choice evidence (`model_choice_evidence.csv`)

| model | LIT best/median µV | P2 best/median µV | **gap (median)** | S7 rate LIT (n=48) | S7 rate P2 (n=254) | P2 med. recovery | clean /10 | held-out r |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny | −2.65 / −2.17 | −2.63 / −2.41 | −0.24 | 0.333 | 0.492 | 1.31 | **10** | +0.212 |
| whisper-base | −1.34 / −1.24 | −2.65 / −2.39 | −1.16 | 0.271 | 0.480 | 0.72 | 9 | +0.173 |
| **whisper-small** | −1.49 / −1.32 | −3.33 / −3.05 | **−1.73** | 0.312 | 0.496 | 1.59 | 9 | +0.118 |
| whisper-medium | −9.61 / −2.30 | −4.67 / −3.99 | −1.69 | 0.542 | 0.610 | 2.50 | 9 | +0.079 |
| **wav2vec2-medium** | −2.08 / −1.88 | −3.79 / −2.93 | −1.05 | **0.604** | **0.736** | 1.07 | 9 | +0.223 |
| wav2vec2-large | −7.90 / −3.76 | −8.05 / −7.63 | −3.87 | 0.542 | 0.555 | 0.84 | **7** | **+0.244** |

### 6.2 The 60 panels (`per_model_top5_comparison.csv`)

```
model            set         panels: std→dev Hz, trough µV  (S7✗ = fails the criterion)
whisper-tiny     literature  1850→1000 -2.65 | 1500→1000 -2.51 | 2000→1000 -2.17 | 1000→2000 -1.51 | 1000→1850 -1.34
whisper-tiny     phase2      6979→800  -2.63 | 2934→1467 -2.60 | 6979→734  -2.41 | 3200→1345 -2.38 | 3200→673  -2.33
whisper-base     literature  1850→1000 -1.34 | 1000→1850 -1.26 | 1000→2000 -1.23 | 1500→1000 -1.02 | 1200→1000 -0.80
whisper-base     phase2      1600→7611 -2.65 | 200→7611  -2.39 | 1745→7611 -2.39 | 1345→7611 -2.38 | 1903→7611 -2.24
whisper-small    literature  1000→1850 -1.49 | 1000→1500 -1.32 | 1000→2000 -1.32 | 1850→1000 -1.19 | 600→1000  -1.08
whisper-small    phase2      2263→7611 -3.33 | 800→7611  -3.13 | 673→7611  -3.05 | 4150→7611 -2.97 | 734→7611  -2.93
whisper-medium   literature  633→700   -9.61 | 1000→633  -5.69 | 1000→1850 -2.30 | 633→1000  -2.28 | 1000→1200 -2.19
whisper-medium   phase2      1467→3200 -4.67 | 800→3200  -4.00 | 1467→2934 -3.99 | 1467→5869 -3.83 | 1467→4935 -3.62
wav2vec2-medium  literature  1000→1200 -2.08 | 2000→1000 -2.05 | 1200→1000 -1.88 | 1850→1000 -1.64 | 1000→633  -1.63
wav2vec2-medium  phase2      2691→6400 -3.79 | 4150→6400 -3.40 | 951→6979  -2.93 | 1345→2691 -2.92 | 951→7611  -2.92
wav2vec2-large   literature  1850→1000 -7.90 | 1200→1000 -4.04 | 2000→1000 -3.76 S7✗ | 1000→1500 -3.40 | 1500→1000 -3.12
wav2vec2-large   phase2      1600→3200 -8.05 | 1745→6400 -7.83 | 1745→7611 -7.63 S7✗ | 1745→4935 -7.01 | 673→3200  -6.35
```

**wav2vec2-large is the only model with S7 failures in its own top 5** (2 of 10 panels), both for
want of recovery (fractions 0.29 and 0.19 against a 0.50 requirement).

### 6.3 Recommendation

> **Main slide 1 — `whisper-small`. Main slide 2 — `wav2vec2-medium`. Runner-up — `whisper-base`.**

- **whisper-small** — the clearest *visual* story. Largest clean gap: median −1.32 → −3.05 µV
  (**2.3×**) with no outlier carrying it (its P2 rank 5 is still −2.93 µV). All 10 panels pass S7;
  P2 row is 5/5 S4-specific, 0/5 down-sloping, median post-trough slope **+3.95 µV/100 ms**.
- **wav2vec2-medium** — the strongest *statistical* story and the other architecture family.
  **Highest S7 pass rate of any model on both sets** (60.4 % LIT, **73.6 % P2**) — strictly
  like-for-like (same stimuli, criteria, electrode). Second-highest held-out r (+0.223). 9/10 clean.
  Smallest gap of the viable models (−1.05 µV) — that is the trade.
- **whisper-base (runner-up)** — same shape as whisper-small, slightly larger gap than
  wav2vec2-medium (−1.16 vs −1.05), all 10 pass S7. Displaced on pass rate (.271/.480 vs
  .604/.736), fit (+0.173 vs +0.223) and P2 recovery (0.72 vs 1.07); also, two adjacent whisper
  models make a narrower pair than one per family.

**Rejected, with reasons:**

- **wav2vec2-large** — highest r (+0.244) and biggest median gap (−3.87 µV), so it is the tempting
  pick, but it is **the one model that fails on the objection already raised** and r does not fix
  shape: 2/10 panels fail S7, LIT-1 is still descending at −1.54 µV/100 ms, only 7/10 clean, and its
  best LIT stimulus (−7.90) essentially ties its best P2 one (−8.05) so the head-to-head has no
  headline.
- **whisper-medium** — excellent P2 row, but its LIT row is led by a **−9.61 µV outlier**
  (`method_43`, 633→700 Hz) deeper than anything in P2, which **inverts the slide's message**
  (literature wins best-single-stimulus by +4.94 µV) and sets the shared axis. Lowest r (+0.079).
- **whisper-tiny** — cleanest panels (10/10) but **no story**: median gap −0.24 µV, best-trough gap
  **+0.02 µV**. A real appendix result; a dead main slide.

**Two known blemishes in the recommended 20 panels**, both in row A, neither in row B:
whisper-small LIT-4 (1850→1000, `method_19_counter`) and wav2vec2-medium LIT-4 (1850→1000,
`method_21_counter`) each pass S7 but **fail S4**. For whisper-small, promoting the next distinct
pair `method_44` (633→1000 Hz, −0.99 µV, S4 ✓, slope +0.59) leaves the row median **unchanged at
−1.32 µV**. **Deliberately not done automatically** — silently dropping a panel for how it looks is
what this analysis exists to avoid.

---

## 7. The held-out-r episode (relevant if fit quality is ever re-examined)

**Initially,** `outputs/results/eeg_mapping/wav2vec2-{medium,large}__*__D2.json` were absent, so both
wav2vec2 models reported *unavailable*. Root cause, established from git rather than guessed:

- The path is **deliberately not gitignored** — `.gitignore` excludes `outputs/features/`,
  `outputs/neural_data/`, `outputs/mmn_stimuli/`, `outputs/results/*-cv/` and `*.h5` under the
  comment *"keep figures and JSON summaries"*.
- The four whisper JSONs **are tracked**, committed 2026-06-18 in `456f2b9`.
- The wav2vec2 sweeps ran in the later 2026-07-13 round and their JSONs were **never committed to
  any branch** (`git log --all --diff-filter=A -- 'outputs/results/eeg_mapping/wav2vec2*'` empty).
  Only the *derived* half returned: `1267c72` hardcodes `encoder.layers.2` / `encoder.layers.12`
  into `scripts/slurm_insilico_mmn_electrodes.sh:50`. `chosen_layer` survived as a shell constant;
  `test_r_chosen`, which lives only in the JSON, did not.
- whisper-large is the control case: its JSON is also absent locally, but its r *is* in the handoff
  table because it finished before doc time. Same gap, different timing.

**Resolved 2026-08-17** — the user pulled the four files:

```bash
rsync -av "jed:/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/results/eeg_mapping/"'wav2vec2-*__*__D2.json' \
      outputs/results/eeg_mapping/
```
(`jed` = `~/.ssh/config` alias for `sigfstea@jed.hpc.epfl.ch`.)

**Values now in hand** (mean `test_r_chosen`, n=47 electrodes / n=5 parcels):

| model | layer | electrodes | parcels |
|---|---|---|---|
| wav2vec2-large | `encoder.layers.12` | **+0.2439** | +0.2119 |
| wav2vec2-medium | `encoder.layers.2` | **+0.2231** | +0.1996 |
| whisper-tiny | `blocks.0` | +0.2118 | +0.1870 |
| whisper-base | `blocks.0` | +0.1731 | +0.1530 |
| whisper-small | `blocks.1` | +0.1178 | +0.0734 |
| whisper-medium | `blocks.12` | +0.0793 | +0.0782 |

**⚠️ THE MOST IMPORTANT CAVEAT IN THIS DOCUMENT.** Do **not** quote r across architectures.
wav2vec2 was mapped on **10 s/10 s** windows against whisper's **30 s/10 s** (both `pca_var=None`);
both handoffs call the whisper ballpark a *loose sanity gate* for wav2vec2, not an exact comparison.
"wav2vec2 has the best held-out r" is **not** evidence that wav2vec2 predicts EEG better than
whisper. Its legitimate uses here: removing a "we don't know whether that model fits at all"
objection, and ranking within a family. The like-for-like argument for wav2vec2-medium is the **S7
pass rate**, which shares stimuli, criteria and site with every other model.

**This changed the recommendation.** The first version was whisper-small + whisper-base, with
wav2vec2-medium as runner-up **solely because its r was unknown**. At +0.223 that objection is gone
and wav2vec2-medium beats whisper-base on every column but a marginally smaller gap. **Nothing else
moved** — no figure, no trough, no S7 verdict; the six SVGs are byte-identical before and after
(md5-verified), because r is not plotted.

**A new validation became possible** and is now enforced: `heldout_test_r()` asserts each JSON's
`chosen_layer` equals the layer the prediction HDF5s were written at (`LAYER`). Those two travelled
separately, so a mismatch would mean the reported fit quality described a different mapping than the
plotted waveforms. **All six match** — which independently confirms the hardcoded shell constants
against the sweep's own output.

---

## 8. Validation — every check passed

| check | result |
|---|---|
| exactly 6 models, whisper-large absent | ✅ filtered in `load_scored`, asserted |
| 48 LIT + 254 P2 direction-instances | ✅ |
| every regular condition has its counter; direction carried explicitly | ✅ asserted per set |
| trough sign convention — every top-5 trough negative | ✅ (a positive one raises `SystemExit`) |
| derived `trough_uv` reproduces the committed scored CSVs | ✅ max deviation **5.0 × 10⁻⁶ µV** over 1,812 rows (CSVs round to 5 dp) |
| top-5 reproduces `literature_top48.csv` / `phase2_top50.csv`, ≥ 2 models | ✅ **48 values, MATCH** (whisper-tiny, whisper-base; against `mean/median/min/max_uv_all6`, the only µV columns those tables carry) |
| every panel is a mean of 15 deviants | ✅ all 1,812 cells; **none excluded** (short cells would be dropped and named) |
| each sweep's `chosen_layer` == the prediction layer | ✅ all 6 |
| continuity with the across-model figure | ✅ 2263→7611 Hz in whisper-small's P2 top 5; 1600→3200 Hz in wav2vec2-large's |
| `results_soafix_full` vs `results_soafix` at (FCz, mTRF, X=0.75) | ✅ byte-identical slices |
| SVG reproducibility | ✅ byte-identical across re-runs (fixed `svg.hashsalt`, `metadata={"Date": None}`) |

---

## 9. File manifest

**Created (all new, all under `aux/analysis_presentation/`):**

| path | contents |
|---|---|
| `per_model_top5.py` | the analysis + figures |
| `per_model_top5_comparison.csv` | 60 rows — the panels. Columns: `model, set, rank, method_id, method, direction, f_standard, f_deviant, semitones, trough_uv, latency_ms, recovery_frac, s7_pass, s2_pass, s4_specificity, post_slope_uv_per_100ms, ctrl_min_uv, n_deviants, consensus_rank, n_agree` |
| `per_model_full_ranking.csv` | 1,812 rows — the un-truncated per-model ranking (what de-dup removed is auditable here) |
| `model_choice_evidence.csv` | 6 rows — the §6.1 table plus `heldout_r_source` |
| `per_model_top5_recommendation.md` | the argued write-up + one caption paragraph per figure |
| `handoff_cp2_per_model_top5.md` | this file |
| `plots/per_model_top5/<model>.png` × 6 | presentation-resolution rasters (200 dpi) |
| `svgs/per_model_top5/<model>.svg` × 6 | vector twins — **the deck needs these**; the current head-to-head raster is soft when projected |

**Modified:** no committed script, figure or CSV, at any point. Within this task's own new module,
`per_model_top5.py` gained the `chosen_layer` assertion on 2026-08-17.

**Figure anatomy:** 2 rows × 5 columns. Row A = that model's own top-5 literature; row B = its own
top-5 Phase-2. Shaded band = 100–240 ms scoring window; ▾ marks the scored trough; each panel titled
with standard→deviant Hz, direction, trough µV @ latency, and that model's own S7 verdict (**red
title = fail**). Shared y within the figure.

---

## 10. Reproducibility

```bash
cd /Users/sophiesigfstead/Documents/multimodal-brain-scaling-2
PYTHONPATH="$PWD/src:$PWD/scripts" conda run -n mbs-env \
    python aux/analysis_presentation/per_model_top5.py

# inspection variant, every panel autoscaled (NOT the deck figures):
PYTHONPATH="$PWD/src:$PWD/scripts" conda run -n mbs-env \
    python aux/analysis_presentation/per_model_top5.py --free_y --out_dir /tmp/freey
```

`--n_top` changes panels per row (default 5). `--skip_figures` runs tables only.
**Bare `python` will not work** — base has no h5py/pandas here, and `mbs-env`'s editable `mbs`
install resolves to a *sibling* checkout, so the `PYTHONPATH` prefix is required or imports silently
test the wrong code.

Runtime ≈ 30 s; reads ~12 HDF5s and 2 CSVs, writes 12 figures + 3 CSVs.

---

## 11. Open items / things not to re-litigate

- **Not an output of this task, still stale:** `aux/handoff_enable_large_wav2vec2_models.md`
  § Results reads *"_pending_"* for all four wav2vec2 rows (written mid-sweep in `7a0e225`, never
  revised). Values to paste are in §7 above. It is a **tracked** doc, so it was left alone.
- `outputs/results/eeg_mapping/whisper-large__*__D2.json` is still absent locally. Irrelevant here —
  whisper-large is excluded from this analysis — and its r is already in that handoff table.
- **The consensus per-model figures were not touched and must not be deleted** — that view was
  deferred, not rejected (§1).
- **Deviance is deliberately unrestricted.** Do not add a semitone cap to this figure; it is a
  best-responder head-to-head, and the dose–response question lives in
  `aux/analysis_MMN_realness_checks/`.
- **The Phase-2 set is selected on the outcome** (`n_agree ≥ 5` at FCz in Phase 1). This figure
  bounds what the best model-selected stimuli look like; it does **not** estimate an effect size,
  and no statistic is computed on it.
- **Predicted µV are not literature EEG µV.** Ridge shrinks amplitude and the 0.75 µV floor is
  calibrated to the models' own distribution. Only direction and monotonicity transfer.
- Unrelated to this task: `aux/analysis_MMN_realness_checks/lit_2st_bins_and_n_change.py` shows as
  modified in the working tree (a `set_key` parameter added for an `n_effect_pairwise_stats.py`).
  **Not written by this task** — flagged only so it is not mistaken for part of it.
