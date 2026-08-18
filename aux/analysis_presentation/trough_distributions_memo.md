# CP3 — Per-model trough distributions: literature vs model-selected

**Slide this fills:** "Results: Comparing selected vs. literature responses" (main deck, currently blank).
**Script:** `aux/analysis_presentation/trough_distributions.py` (read-only over committed CSVs).
**Date:** 2026-08-17. **Site:** FCz electrode. **Read-out:** mTRF. **Units:** µV, predicted difference
wave (deviant − standard) sampled at the trough latency — the same signed `trough_uv` column the
head-to-head and dose-response figures use.

---

## The claim, stated so it cannot drift

The claim is **a count in the left tail**, not a group difference. The two distributions overlap
heavily and are expected to. What matters is that for a given model there exist *some* — it does not
have to be many — model-selected frequency pairs whose predicted MMN trough goes deeper than **any**
literature stimulus that model was given.

The group tests (Welch t, Cohen's d, Mann–Whitney) were requested and are reported below as
secondary. They are unimpressive at the primary population, which is fine and expected: a small *d*
does not weaken the tail count, because the two are answering different questions.

## Headline result (primary population)

**4 of 6 models have model-selected stimuli that beat every literature condition.**

| model | n LIT | n P2 | LIT median | P2 median | LIT deepest | P2 deepest | **n deeper than LIT min** | **%** | n > LIT p90 | n > LIT p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| whisper-tiny    | 48 | 254 | −0.42 | −0.74 | −2.65 | −2.63 | **0**  | 0.0 %  | 22 | 2 |
| whisper-base    | 48 | 254 | −0.48 | −0.64 | −1.34 | −2.65 | **58** | 22.8 % | 65 | 61 |
| whisper-small   | 48 | 254 | −0.44 | −0.75 | −1.49 | −3.33 | **36** | 14.2 % | 59 | 45 |
| whisper-medium  | 48 | 254 | −0.95 | −1.01 | −9.61 | −4.67 | **0**  | 0.0 %  | 53 | 46 |
| wav2vec2-medium | 48 | 254 | −0.89 | −1.25 | −2.08 | −3.79 | **40** | 15.7 % | 52 | 50 |
| wav2vec2-large  | 48 | 254 | −1.11 | −1.28 | −7.90 | −8.05 | **1**  | 0.4 %  | 39 | 5 |

"LIT p90 / p95" = the 90th and 95th **percentile of depth**, i.e. the 10th and 5th percentile of the
signed µV. They exist because the strict count rests on a single literature condition.

### One sentence per model

- **whisper-tiny** — no model-selected stimulus goes deeper than this model's deepest literature
  stimulus (−2.65 µV), but it misses by 0.018 µV: the deepest selected pair reaches −2.63 µV, and 22
  selected stimuli beat the literature's 90th-percentile depth.
- **whisper-base** — 58 of 254 model-selected stimuli (22.8 %) drive a deeper trough than **any** of
  the 48 literature conditions; the deepest reaches −2.65 µV against the literature's −1.34 µV.
- **whisper-small** — 36 of 254 (14.2 %) go deeper than every literature condition; the deepest
  reaches −3.33 µV against −1.49 µV.
- **whisper-medium** — no model-selected stimulus beats the literature maximum, but that maximum is a
  lone outlier at −9.61 µV (next deepest literature condition −5.69 µV, median −0.95 µV); against the
  literature's 90th-percentile depth, 53 of 254 selected stimuli go deeper.
- **wav2vec2-medium** — 40 of 254 (15.7 %) go deeper than every literature condition; the deepest
  reaches −3.79 µV against −2.08 µV.
- **wav2vec2-large** — 1 of 254 goes deeper than every literature condition (−8.05 µV against
  −7.90 µV), and 39 beat the literature's 90th-percentile depth.

### The two zeros are outlier-driven, and differently so

- **whisper-medium**: the literature bar is one freak condition, `method_43` at −9.61 µV, with the
  next literature condition at −5.69 µV against a −0.95 µV median. The strict count of 0 is a
  statement about that one stimulus, not about the literature set.
- **whisper-tiny**: not an outlier artefact. Its three deepest literature conditions cluster at
  −2.646 / −2.641 / −2.634 µV (methods 17–19 counter, all the same 10.65-semitone deviance), and the
  best selected pair lands at −2.628 µV. This model genuinely was not beaten — by 0.018 µV.

## The tail count is gate-invariant, which is the useful robustness result

| model | n deeper than LIT min @ **all** | @ **S2** | @ **S7** |
|---|---|---|---|
| whisper-tiny    | 0  | 0  | 0  |
| whisper-base    | 58 | 56 | 56 |
| whisper-small   | 36 | 36 | 36 |
| whisper-medium  | 0  | 0  | 0  |
| wav2vec2-medium | 40 | 39 | 39 |
| wav2vec2-large  | 1  | 1  | 1  |

The µV floor sits nowhere near the tail, so gating barely touches the headline count. It moves the
*medians* substantially — that is what the floor is for — but not the claim.

## Secondary: group comparison, as requested

Contrast is **NOVEL-P2 minus LIT** throughout; negative = P2 deeper. Cohen's *d* uses the **classic
pooled SD**, `sqrt(((n₁−1)s₁² + (n₂−1)s₂²)/(n₁+n₂−2))` — stated because with n = 48 against n = 254
and unequal variances, Glass's Δ or Hedges' *g* would each give a different number on the same data.
The sets are **independent** (different stimuli, no pairing), so Welch and Mann–Whitney are the right
forms — unlike the paired N-effect contrast in the sibling deliverable.

Primary population (all traces):

| model | Welch t | df | p | mean diff (µV) | 95 % CI | Cohen's d | MWU p | rank-biserial r |
|---|---|---|---|---|---|---|---|---|
| whisper-tiny    | −0.74 | 60.7  | 0.462 | −0.093 | [−0.344, +0.158] | −0.13 | 0.148 | −0.13 |
| whisper-base    | +0.28 | 143.0 | 0.782 | +0.026 | [−0.158, +0.209] | +0.03 | 0.692 | −0.04 |
| whisper-small   | −0.40 | 147.5 | 0.689 | −0.038 | [−0.228, +0.151] | −0.04 | 0.356 | −0.08 |
| whisper-medium  | +0.53 | 58.3  | 0.600 | +0.134 | [−0.373, +0.640] | +0.10 | 0.831 | −0.02 |
| wav2vec2-medium | −2.98 | 83.9  | **0.004** | −0.299 | [−0.499, −0.100] | **−0.38** | **0.004** | −0.26 |
| wav2vec2-large  | +0.30 | 65.7  | 0.767 | +0.097 | [−0.555, +0.750] | +0.05 | 0.916 | +0.01 |

**Read plainly: five of six models show no group difference at all**, and the sign is not even
consistent. Only wav2vec2-medium separates, at *d* = −0.38 — a small effect. This is exactly what was
expected, and it does not bear on the tail claim.

A rank-based companion is reported alongside the t-test because these distributions are skewed — the
project memo documents 0 of 31 per-bin trough distributions consistent with normality — so the
t-test's own assumption is not met and its p is the optimistic one of the two.

**Note, and do not overstate it:** at the **S2** population the group comparison gets much stronger —
5 of 6 models show NOVEL-P2 deeper with p < 0.05 (whisper-base *d* = −0.48, p = 2e-6; whisper-small
*d* = −0.47, p = 1e-6; wav2vec2-medium *d* = −0.38). The reason is mechanical: the `all` population
includes traces whose argmin is not an MMN latency, which adds roughly symmetric noise to both sets
and dilutes any difference. S2 keeps only traces whose trough latency is meaningful. This is a real
result but it is a *secondary* population, and the headline should stay the tail count.

## Which stimuli clear the bar

135 (model, direction-instance) cells beat their model's literature maximum at the primary
population — full list in `tail_winners.csv` with frequency pair, direction, semitones, µV, margin,
and `n_agree`.

- **Direction is lopsided: 95 of 135 winners are `regular`, 40 are `counter`.**
- **They are not obscure stimuli:** 116 of 135 winners have `n_agree ≥ 4` and 81 have `n_agree ≥ 5` — at least four of the
  six models independently showed an S7-passing MMN for that pair.
- **They are mostly wide-deviance and high-frequency:** winner deviances run 7.5–63 semitones, and
  42 of 135 involve f_high = 7611 Hz (the grid's top frequency). This is the **frequency-stripe**
  pattern §17 already documented — the models respond to particular frequencies more than to the
  change between two tones — so the tail should be read as "these stimuli drive deeper predicted
  troughs", not as "these are better *oddball contrasts*".
- Deepest single winners: whisper-small `method_1812` (2263→7611 Hz, 21.0 st, −3.33 µV, 6/6 agree);
  wav2vec2-medium `method_1835` (2691→6400 Hz, 15.0 st, −3.79 µV); whisper-base `method_1750`
  (1600→7611 Hz, 27.0 st, −2.65 µV, 6/6 agree).

## Populations and gates

`trough_uv` is **finite for every row** — 0 NaN across 288 LIT and 1524 NOVEL-P2 rows. It is the µV
difference-wave value sampled at the argmin latency of the z-scored trace inside 100–240 ms
(`scripts/analyze_mmn_s7_roi.py`), and that argmin exists whether or not the trace passes any
criterion. **So a trough is defined for all traces, and the widest population is every row.**

| gate | role | definition | what it costs |
|---|---|---|---|
| `all` | **PRIMARY** | every trace, no shape criterion, no µV floor | includes traces whose argmin is not an MMN latency — 11.8 % of LIT and 21.2 % of NOVEL-P2 rows have a *positive* trough, i.e. a peak |
| `s2` | companion | shape only, no µV floor | the project's own definition of "ungated"; every trough measured at a defensible latency |
| `s7` | companion | S2 **and** trough ≤ −0.75 µV | censors the deep tail on one side and the shallow end on the other — gating first and counting the tail after would be measuring the gate |

All three are produced and filed by gate, mirroring the realness-checks deliverable. `all` is primary
per the brief; the S7 companion is clearly labelled on its own figures.

**Deliberate inclusion of all 254 Phase-2 instances.** The sub-threshold reversals carried along only
for counterbalancing are *not* dropped. Excluding them would narrow the distribution on the selected
side and make the tail look like a property of the filter rather than of the stimuli.

## Figure form: KDE, not histogram

Both were produced (`__kde` and `__hist` for every model and both panels). **KDE + rug is the one for
the slide.** With n = 48 across 30 common bins the literature histogram is spiky and gappy — at
projection size the gaps read as structure that is not there, and the stepped NOVEL-P2 outline over
it is busy. The KDE gives two clean readable shapes; the **rug of actual observations** underneath is
what keeps it honest, because a KDE smooths mass past the extremes and the extremes are the entire
claim. Both are areas normalised to **density, not counts**: on a count axis NOVEL-P2 would look five
times more prevalent at every depth purely from n.

## Caveats

- **Each model's µV scale is its own.** mTRF predictions are amplitude-shrunk and the shrinkage
  differs by model and layer. Compare each panel against its **own** literature rule; never one
  model's depth against another's. `whisper-large` is excluded throughout.
- **NOVEL-P2 is selected on a closely related outcome** (`n_agree ≥ 5` at FCz in Phase 1). The tail
  count answers "can a search over the frequency grid find stimuli that beat the literature for this
  model?" — which is the project's motivating question — not "are novel stimuli better in general".
- **Predicted µV are not literature EEG µV.** Only direction and monotonicity transfer.
- The x-axis on the whisper-medium and wav2vec2-large panels is stretched by literature outliers.
  That is kept rather than clipped, because the dashed rule *is* the reference line and clipping it
  out would hide why those counts are what they are.

## Validation (all asserted in the script, re-run every time)

- 48 LIT cells × 6 models = 288 rows; 254 NOVEL-P2 cells × 6 models = 1524 rows. **Exact.**
- No duplicate `(model, method_id, direction)` rows in either set.
- Both directions present for every pair: 24 × 2 = 48 and 127 × 2 = 254.
- All six models present; `whisper-large` absent (dropped at load time in `load_fcz`).
- `trough_uv` finite for all 1812 rows.
- Deepest trough negative for every (model, set) — sign convention holds. S7 ⊆ S2 everywhere.
- Cross-check against `phase2_model_uv_box.csv`: that table covers the **n_agree ≥ 5 tier (91 of the
  254 instances)**, not the full Phase-2 population, so its medians are not expected to match. Its
  `min_uv_all` for wav2vec2-large (−8.0532) reproduces this analysis's Phase-2 deepest value exactly,
  confirming the same column and the same vintage.

## Reproducibility

```bash
conda activate mbs-env
PYTHONPATH=$PWD python aux/analysis_presentation/trough_distributions.py
```

**Reads** (read-only): `outputs/results_soafix_full/mmn_s7_roi.csv`,
`outputs/results_novel_search/phase2_mmn_s7_roi.csv`, `outputs/results_novel_search/grid_index.csv`,
`outputs/results_novel_search/phase2_final_ranking.csv`,
`data/metadata/literature_frequency_intensity_duration_metadata.csv`.

**Writes**, all under `aux/analysis_presentation/`:

| path | contents |
|---|---|
| `trough_distributions.py` | the analysis |
| `trough_distributions_by_model.csv` | 36 rows — descriptives per (gate, model, set) + tail counts |
| `trough_distribution_stats.csv` | 54 rows — Welch t / Cohen's d / Mann–Whitney per (model, gate) |
| `tail_winners.csv` | 399 rows — instances beating the literature max, per (gate, model) |
| `trough_distribution_summary.csv` | 18 rows — the cross-model slide table, per gate |
| `plots/{all,s7,s2}/trough_dist__<model>__{kde,hist}.png` | 36 per-model figures |
| `plots/{all,s7,s2}/trough_dist__panel6__{kde,hist}.png` | 6 multi-model 2×3 panels |
| `svgs/{all,s7,s2}/…svg` | vector mirror of every PNG, same render |

**For the slide:** `plots/all/trough_dist__panel6__kde.png` (primary), with
`plots/s7/trough_dist__panel6__kde.png` as the gated companion in backup. Each figure states its own
population and gate in the title.
