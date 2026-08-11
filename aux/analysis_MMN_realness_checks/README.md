# How to read these figures

Two dose-response checks on whether the in-silico MMN behaves like a human MMN, at the **FCz
electrode**, **mTRF** mapping, **six models**. 22 figures = 11 views × 2 gates.

Read §1 first — it is the vocabulary every figure uses. Then find your figure in §3.

---

## 1. Vocabulary shared by every figure

**The y-axis is always `trough_uv`** — the deviant−standard difference wave in µV, sampled at the
trough latency, plotted conventionally (**negative downward**). **More negative = a deeper MMN**, so
a line that *descends* left-to-right is the MMN-like direction. Every panel is labelled `↓ deeper`.

**The x-axis is the manipulation** — either deviance size in semitones, or N (standards between
deviants). Both are things the human MMN is known to respond to.

**The two gates** decide which traces are allowed into an amplitude figure:

| gate | keeps | lives under |
|---|---|---|
| **S7@0.75** | shape **and** trough ≤ −0.75 µV | `plots/s7/` |
| **S2** | shape only — **no µV floor** | `plots/s2/` |

"Shape" (S2) = an interior trough in 100–240 ms that recovers ≥50% within 120 ms. S7 adds a depth
requirement on top. **S7 is the headline; S2 is the uncensored companion.** The floor removes the
shallow tail *by construction* — exactly the traces a weak dose-response moves — so **S7 numbers
are a lower bound** and the gap between the two gates is what the floor was hiding. It is not a
small gap: S7 keeps only 48% of LIT's S2 traces and 73% of NOVEL-P2's.

**The three sets** appear as panels of every multi-panel figure, always in this order:

| panel | set | what it is |
|---|---|---|
| 1 | `lit` | 48 literature conditions. Unselected, but timing varies and coverage is lumpy. |
| 2 | `lit_p2` | both sources pooled (302 conditions). ~87% NOVEL-P2 rows. |
| 3 | `p2` | 254 selected conditions. Timing fixed, but **selected on the outcome**. |
| 4 | `p2_top100` | **pooled figures only** — the top 100 of the phase-2 ranking. |

`p2_top100` is the first 100 rows of the search's own committed ranking
(`phase2_final_ranking.csv`, ordered by `n_agree` then `mean_uv`): 9 direction-instances at 6/6
model agreement, 82 at 5/6, 9 at 4/6. **It is selected on the outcome twice over** — NOVEL-P2 is
already the subset of the 903-pair grid that reached 5/6 agreement in phase 1, and ranking within
it by agreement and trough depth selects again on the quantity the figure plots. Its S7 pass rate
is 83% against p2's 56%, and its troughs are deeper by construction (median −1.57 vs −1.49 µV). It
answers *"among the stimuli that most reliably evoke a model MMN, does the trough still track the
manipulation?"* — never *"how big is the effect?"*. It is deliberately **not** one of the three
analysis sets: no statistic, rate figure or per-model panel is computed on it.

**Colour = model identity**, always the same six colours (Okabe–Ito, CVD-checked). Every line is
solid; linestyle encodes nothing. Dotted grey is the y = 0 reference, not data. The dark grey
series is the **pooled** result over the six models.

**One summary statistic everywhere: the median with a 95% bootstrap CI of the median**, with a
mean ± SEM companion of the two pooled figures so the choice can be audited rather than trusted. Earlier
versions mixed mean ± SEM (the `lit` panel) with median + IQR (the `lit_p2` and `p2` panels) on a
*shared* axis, which was misleading three ways: the panels were not the same statistic; SEM measures
precision of the centre while IQR measures spread of the data, and on these bins the IQR runs ~10×
wider, so `p2` looked an order of magnitude noisier than `lit` when it actually has ~90–100
conditions per bin against `lit`'s 2–48; and the small-n robustness argument for the median was
being applied to the panel with the *large* bins. Median because the trough distributions really are
skewed (mean − median ≈ 0.2 µV on a ~1.5 µV signal); bootstrap CI so that every error bar in the
deliverable means the same thing and whisker lengths are comparable across panels.

The evidence for that choice, and its limits: **0 of 31 bins are consistent with normality**
(Shapiro–Wilk), with skewness around −2 to −3 — a tight body near −1.3 µV and a long tail to
−9.6 µV. A mean under that shape is dragged toward the tail and stops describing a typical
condition (in LIT it sits 0.4 µV deeper than the median). There is also no closed-form standard
error for a median, and its sampling distribution is not normal here, so ±1.96·SE would be
unsupported — the bootstrap assumes nothing about shape. **The choice does not manufacture a
result:** the N=3→7 change agrees to ≤0.02 µV between mean- and median-based versions in every
set, so only the level moves, not the trend. Compare `n_effect_pooled.png` against
`n_effect_pooled__mean_sem.png` — same axis, same shape, ~0.35 µV apart. What the median does
cost you is the tail, which is real data: for depth extremes read the per-model scatter.

---

## 2. The two questions

- **`deviance_*`** — does the trough deepen as the standard→deviant frequency gap grows?
- **`n_effect_*`** — does the trough deepen as more standards separate the deviants?

Both are among the best-replicated properties of the human MMN, which is why they are the test.

---

## 3. The four figure families

### A. `*_pooled_*` — the headline amplitude result

`<gate>/all_sets/deviance_pooled.png`, `<gate>/all_sets/n_effect_pooled.png`
…and a `__mean_sem` companion of each, **on an identical axis** so the two can be flipped
between directly.

Three panels, one per set, pooled over the six models. **One series per panel** — the middle panel
pools both sources into a single line. Grey numbers above each point are the n behind it. The
y-axis is shared across all three panels so the sets are directly comparable, and is scaled to the
plotted **bin summaries** (not to the individual troughs behind them, which run ~4× deeper and
would leave most of each panel empty). Bins under n = 10 do not set that scale — a bootstrap CI
widens as n shrinks, and LIT's 2-condition bins would otherwise drag every panel's axis down — so
their CIs may run off the bottom; any bin whose *centre* is off-axis is marked with its true value.

*How to read it:* a line descending left-to-right = deeper with more deviance/more standards =
the MMN-like direction. Flat = no amplitude effect.

*Two things to watch:*
- **The N-effect panels use the BALANCED set** — only stimuli producing a criterion-passing MMN at
  N=3 *and* 5 *and* 7. That makes N a within-stimulus manipulation, so a change across N is a
  change in the same stimuli rather than a change in which stimuli are being averaged. (The
  earlier unbalanced version was misleading for exactly that reason: LIT's line rose while no
  individual stimulus deepened.) The panel subtitle reports how many stimuli survived. The
  deviance panels are not balanced — there is no repeated within-stimulus factor to balance on.
- The deviance panels clip the deepest 2% of troughs so one outlier bin cannot set the scale.
  Clipped bins are drawn **on the boundary with their true value labelled**, never dropped.

### B. `*_per_model_*` — is the effect in every model, or one of them?

`<gate>/<set>/deviance_per_model__<set>.png`, `<gate>/<set>/n_effect_per_model__<set>.png`

Six panels, one per model. Deviance versions show raw points plus a per-model OLS fit and ρ on a
**shared y-axis** (their scatter spans the full range in every panel, so one scale is readable and
lets you compare amplitude across models). N versions show the median with its bootstrap CI at each N on **per-panel y-scales** — the models sit ~2 µV apart while each one's change across N is ~0.05–0.35 µV, so a
shared axis flattens every trend to a few percent of its height. In the N files compare the SHAPE
of each panel, not its height; the pooled figure carries the cross-model amplitude comparison.

*How to read it:* look for consistency of **sign** across the six panels, not just significance.
Six same-signed weak effects are stronger evidence than one significant model. In the `lit_p2`
files, marker shape distinguishes the two sources (● LIT, △ NOVEL-P2).

*Watch:* `▽ k off-scale` means k points are deeper than the axis and are drawn as hollow carets on
the bottom edge. They are **included** in ρ and the fit — only their drawn position is clipped. The
same convention appears on the pooled and overlap figures.

### C. `*_pass_rate*` — the uncensored view, and often the real result

`<gate>/all_sets/deviance_pass_rate.png`, `<gate>/all_sets/n_effect_pass_rate.png`

Three panels; y = **fraction of all conditions that pass the gate**, denominator = every condition,
not just the passing ones. One faint line per model plus the heavy pooled line.

*Why this matters:* a count outcome **cannot be floored by an amplitude gate**, so a dose-response
can appear here that the amplitude axis cannot show. For the N-effect this *is* the result — the
rate climbs 54% → 59% (p = 2e-10) while the gated trough barely moves. **A monotone rate beside a
flat gated trough is a real positive, not a null.**

*Watch:* LIT's per-model lines swing between 0 and 1 because most of its semitone values carry only
2 conditions. Read the pooled line there.

### D. `<gate>/lit_p2/deviance_overlap_lit_vs_p2.png` — is the combined set trustworthy?

The diagnostic for `lit_p2`. LIT spans 0.84–12 semitones and NOVEL-P2 spans 4.50–63, so a slope
fitted across the union is substantially a **between-source contrast wearing a deviance label** —
the sources also differ in SOA, tone duration, deviant probability and selection. This figure
restricts to the **4.50–12 st window where both have conditions** and fits each separately.

*How to read it:* if the two fits agree there, the combined slope is credible as a deviance effect;
if they diverge, it is a source effect. They agree in sign and magnitude (−0.089 vs −0.094) but
**both are non-significant** — agreement on a null. That rules out a source artefact driving the
combined slope; it does not establish a deviance effect inside the overlap.

*This is the only figure where two lines on one axis is intentional.* Source is marked by the
marker riding each fit.

---

## 4. The statistics CSVs

| file | contents |
|---|---|
| `deviance_scaling_s7gated_stats{,__s2}.csv` | ρ per (set × model), pooled rows, the per-source overlap ρ, and a ≤24 st subset |
| `n_effect_stats*.csv` | ρ per (set × model × unit); `unit=cell` is primary, `unit=trial` is the optimistic bound |
| `n_effect_paired_n3_vs_n7*.csv` | the paired within-condition N3-vs-N7 test — settles composition vs real deepening |
| `n_effect_source_agreement*.csv` | per model, do LIT and NOVEL-P2 agree in sign? This licenses (or refuses) reading `lit_p2` as one experiment |
| `n_effect_change_table*.csv` | **the within-stimulus µV change table**: mean trough at each N, and the average change N3→N5, N5→N7, N3→N7, per model and pooled, over balanced stimuli only |

**On `unit` in the N stats:** each (model, condition, N) cell holds 5 variations of the same
paradigm re-rolled, so its trials are not independent. The primary ρ collapses each cell to one
value first; the raw-trial ρ inflates n ~5× and understates p. Quote the cell-level one.

---

## 5. Five ways to misread these figures

1. **Taking an S7 amplitude null as "no effect."** The floor censors the shallow tail. The N-effect
   is null at S7 and significant at S2 — the gate was hiding about half of it.
2. **Reading a change across N as deepening when the contributing stimuli changed.** The N
   amplitude figures are balanced across N precisely to remove this; `n_effect_change_table*.csv`
   and `n_effect_paired_n3_vs_n7*.csv` are the within-stimulus confirmations.
3. **Reading `lit_p2` as a bigger LIT.** For deviance it is ~87% NOVEL-P2 and mostly a
   between-source contrast; always read it beside the overlap diagnostic. (For the N-effect the
   sources span the identical N ∈ {3,5,7}, so the objection does not apply — but they only agree
   in sign 3/6 of the time, so it is still reported per source.)
4. **Treating LIT's ρ = −0.31 as the strongest deviance result.** It decomposes into a step between
   two dense method clusters that also differ in SOA and duration; within each cluster it is flat.
5. **Reading µV as clinical µV.** Ridge shrinks the mTRF's predicted amplitude and the 0.75 µV
   floor is calibrated to the models' own distribution, not a clinical scale. Only *direction* and
   *monotonicity* transfer.

---

## 6. Where the files live

Figures are filed by **gate**, then by **set**, and `svgs/` mirrors `plots/` exactly:

```
plots/<gate>/<set>/<figure>.png        svgs/<gate>/<set>/<figure>.svg
      s7|s2   all_sets | lit | lit_p2 | p2
```

`all_sets/` holds the three-panel figures (pooled, pass-rate) that show all three sets side by side
and so belong to none of them. The overlap diagnostic lives under `lit_p2/`, the set it qualifies.
The gate is now carried by the directory rather than by an `__s2` filename suffix, which also
retires the old misleading `s7gated` / `s7_rate` stems — a file under `s2/` is the S2 figure, and
every figure still states its own gate in the title.

## Regenerating

```bash
conda activate mbs-env
python aux/analysis_MMN_realness_checks/deviance_scaling_s7gated.py --gate s7   # and --gate s2
python aux/analysis_MMN_realness_checks/n_effect_plots.py           --gate s7   # and --gate s2
```

PNGs land in `plots/`, vector twins in `svgs/`, statistics beside this file. Output is
reproducible: identical input gives byte-identical figures.
