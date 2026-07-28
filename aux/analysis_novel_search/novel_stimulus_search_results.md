# Novel tone-pair stimulus search — results

> **STATUS (2026-07-27): Phase 1 complete.** All 903 pairs × 2 directions were extracted for
> 6 models and scored through the electrode-level in-silico MMN. Every number and figure below
> comes from that run. This memo reports **results only** — extraction cost, budget and Phase-2
> sizing live in `aux/sophies_repository_overview.md` §17.4.

> **Extends** `aux/analysis_with_counter/results_analysis_with_counter.md`, which screened the
> **24 literature frequency methods** × 7 models. This memo screens a **903-pair novel frequency
> grid** × 6 models (whisper-large excluded), mTRF only, at the FCz electrode, at a fixed µV floor
> of X = 0.75. 903 unordered pairs × {regular, counter} = **1,806 direction-instances**.

> **The question.** The literature screen tested whatever pairs published MMN studies happened to
> use. This asks whether pairs *absent* from that literature drive stronger and more
> model-consistent responses.

> **The short answer, up front.** The novel grid's top tier does beat the literature set's — 17
> direction-instances at 6/6 where the literature never exceeds 5/6 — but **that count is not
> distinguishable from chance** (12.0 expected from the models' own marginal rates, p = 0.103), and
> **not one pair of 903 reaches the top tier in both directions.** Three independent views of the
> data — the empty both-directions column, the row-and-column striping of the µV heatmap, and the
> mirror-image direction waveforms — all say the same thing: the models are responding to
> **which frequencies are present**, not to the change between them.

---

## Caveats — all load-bearing, all apply to every section below

**1 — This is one deviant draw.** Every condition is scored from a single realization (N7/var1),
not a mean over the 15 available deviants. The models are deterministic, so this is **not
measurement noise** — it is one sample from the stochastic-prefix distribution rather than its
mean. Rankings could legitimately shift on a fuller draw. **No µV figure here is final.**

**2 — Amplitude shrinkage: these are not human-scale microvolts.** mTRF predictions are ~4×
amplitude-shrunk. X = 0.75 µV is calibrated to the *model's own* trough distribution
(median ≈ −0.8 µV), not to literature EEG scale (human MMN ≈ 1–5 µV). **Do not read `trough_uv` or
`mean_uv` as a literature-comparable amplitude**, and never compare `trough_uv` across models. The
scales genuinely differ: on the strong set, wav2vec2-large's median trough is −2.53 µV against
whisper-tiny's −1.12 µV, which says nothing about which model detects deviance better. This
constrains how Figures 3, 4, 5 and 7 may be read, and each says so on its face.

**3 — Out-of-domain mapping.** Speech-trained models with a speech-fit mapping are being applied to
pure tones. This caveat applies equally to the literature screen, so the *comparison* in Section 3
is fair, but it limits any absolute claim about these stimuli.

**4 — The grid is uniformly coarse.** Every step is 1.500 semitones, so **8.8%** is the finest
deviance anywhere in the grid — coarser than 8 of the 24 literature frequency methods (the finest
is 0.84 st). Treat this as a coarse-to-fine search of a wide space, **not a threshold study**;
nothing here speaks to near-threshold deviance, and a follow-up would need a finer grid over
whatever region this one flags.

**5 — wav2vec2 comparability.** Mapped on 10 s/10 s windows, vs whisper's 30 s/10 s. Both used
`pca_var=None` — no PCA on the features. Its test r is a loose sanity gate only, not a
like-for-like quality figure.

---

## Section 1 — What the grid covers

> **Code:** `scripts/build_novel_grid_csv.py`
> **Data:** `data/metadata/novel_grid_frequency_metadata.csv`,
> `outputs/results_novel_search/grid_index.csv`

**Design.** 43 frequencies: a uniform ladder from 200 Hz in exact 1.500-semitone steps, to 7611 Hz.

```
 200   218   238   259   283   308   336   367
 400   436   476   519   566   617   673   734
 800   872   951  1037  1131  1234  1345  1467
1600  1745  1903  2075  2263  2468  2691  2934
3200  3490  3805  4150  4525  4935  5382  5869
6400  6979  7611
```

Semitone spacing keeps every step the same perceptual size; an equal-Hz grid over this range would
put a 13.5-semitone gap at the bottom and 0.55-semitone steps at the top. 12/1.5 = 8 exactly, so
eight steps is an octave and 200/400/800/1600/3200/6400 are all grid points — true 2:1 pairs exist.
The top overshoots the nominal 7500 by one step because 62.746/1.5 is not an integer; 7611 Hz is
still under the 8 kHz Nyquist, and every step stays identical.

43 frequencies → **903 unordered pairs**. Both directions of each pair are synthesized by the
existing counterbalancing, so the 903 rows cover all **1,806 ordered** combinations. The diagonal
is excluded by construction: a same-frequency "deviant" synthesizes to a waveform byte-identical to
the standard, so the difference is exactly zero. The graded control is the smallest-Δf pairs.

**What was run.** Each of the 1,806 direction-instances was extracted for 6 models and pushed
through the frozen mTRF mapping and the committed per-model electrode layer, then scored by the
electrode-level in-silico MMN at FCz.

| model | committed electrode layer |
|---|---|
| whisper-tiny | `blocks.0` |
| whisper-base | `blocks.0` |
| whisper-small | `blocks.1` |
| whisper-medium | `blocks.12` |
| wav2vec2-medium | `encoder.layers.2` |
| wav2vec2-large | `encoder.layers.12` |

No mapping, layer selection or model was re-fit for this search; whisper-large was excluded.

### Section 1 summary
- 43 frequencies, 903 pairs, **1,806 direction-instances**, 6 models, one deviant draw each.
- The grid's floor is 8.8% deviance and its ceiling is 3705%, so it is wide and coarse — a search
  of a large space, not a threshold study.

---

## Section 2 — Does the novel grid produce cross-model agreement, and does it track deviance?

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py`,
> `aux/analysis_novel_search/plots/novel_search_plots.py`
> **Data:** `outputs/results_novel_search/phase1_ranked_directions.csv`,
> `plots/phase1_agreement_tiers.csv`, `plots/phase1_mean_uv_grid.csv`,
> `plots/phase1_frequency_stripes.csv`
> **Definitions.** **S2** = negative trough in 100–240 ms that recovers ≥50% of its depth within
> 120 ms. **S7@X** = `S2 AND (trough_uv ≤ −X µV)`; here X = 0.75, at FCz, mTRF.
> **n_agree** = how many of the 6 models show S7 (0–6). **mean_uv** = mean `trough_uv` across
> *only* the agreeing models — averaging in models that failed S2 would mix in latencies that are
> not MMN latencies. Undefined when `n_agree = 0`. **mean_uv_all6** = mean `trough_uv` across all
> six regardless of S7; a different quantity, used only in Figure 3.

**Table 1. Agreement tiers at both µV floors.** Direction-instances are out of 1,806; pairs out of
903. "both" = pairs whose two directions are *both* at that tier; "either" = pairs with at least
one direction there.

| n_agree | X = 0.75: dirs | % | both | either | X = 0.50: dirs | % | both | either |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 17 | 0.9% | **0** | **17** | 51 | 2.8% | **0** | **51** |
| 5 | 114 | 6.3% | 4 | 110 | 239 | 13.2% | 11 | 228 |
| 4 | 289 | 16.0% | 21 | 268 | 427 | 23.6% | 52 | 375 |
| 3 | 540 | 29.9% | 84 | 456 | 578 | 32.0% | 94 | 484 |
| 2 | 562 | 31.1% | 77 | 485 | 400 | 22.1% | 40 | 360 |
| 1 | 249 | 13.8% | 10 | 239 | 95 | 5.3% | 5 | 90 |
| 0 | 35 | 1.9% | 0 | 35 | 16 | 0.9% | 0 | 16 |

**The gap between "both" and "either" is the headline of this table.** At the top tier it is total:
17 direction-instances reach 6/6, they belong to 17 *different* pairs, and **not one pair reaches
6/6 in both directions** — at either µV floor. At 5/6 only 4 of 110 pairs manage it. If the models
were responding to the *change* between two tones, reversing the pair should preserve the response;
instead, agreement almost never survives reversal.

**Lowering the floor to X = 0.50 moves the counts but not the conclusion.** Every tier inflates —
6/6 triples from 17 to 51, ≥5 roughly doubles from 131 to 290 — because a lower floor admits
shallower troughs at every level. The `pairs_both` column stays at **0** at the top tier and rises
only to 11 at 5/6. The floor is a sensitivity knob on how many pairs make the cut, not on whether
the effect is directional. X = 0.75 is retained.

**Table 2. n_agree vs deviance, by octile of semitone distance.** Spearman ρ(semitones, n_agree)
= **+0.114** (p = 1.2 × 10⁻⁶, n = 1806).

| deviance octile (st) | mean st | mean % deviance | n | mean n_agree | % at n_agree ≥ 5 |
|---|---:|---:|---:|---:|---:|
| 1.5 – 4.5 | 2.84 | 18 | 226 | **1.88** | 0.4% |
| 4.5 – 9.0 | 7.06 | 51 | 226 | 2.64 | 6.6% |
| 9.0 – 13.5 | 11.64 | 97 | 226 | **2.95** | 10.2% |
| 13.5 – 19.5 | 16.77 | 165 | 226 | 2.85 | 9.7% |
| 19.5 – 25.5 | 22.39 | 267 | 226 | 2.83 | 11.1% |
| 25.5 – 33.0 | 28.92 | 435 | 224 | 2.85 | 5.8% |
| 33.0 – 42.0 | 37.01 | 759 | 226 | 2.84 | 8.9% |
| 42.0 – 63.0 | 49.44 | 1727 | 226 | **2.52** | 5.3% |

The expected monotone-rise-then-plateau signature is **present but weak and short**. Agreement
climbs over the first two octiles — from 1.88 to 2.95 mean n_agree between 2.8 and 11.6 semitones —
then goes completely flat (2.83–2.85 across four consecutive octiles spanning 13 to 37 semitones)
and finally *declines* at the largest deviances. So the metric is not flat, and it is not measuring
nothing: it separates a 3-semitone change from a 12-semitone one. But **it carries no information
above roughly one octave**, which is where almost the entire grid lives. ρ = +0.114 is a real
correlation on 1,806 points and a negligible effect size.

![Two panels. Left: per-model S7 rate against deviance in semitones, six Okabe-Ito series each with a distinct marker; all six rise steeply from the smallest deviance bin and then run nearly flat, with wav2vec2-medium highest at around 60 percent and whisper-base lowest at around 33 percent. Right: mean n_agree against deviance with standard-error bars, rising from 1.9 to about 2.95 over the first two bins, then flat near 2.85 for four bins, then falling to 2.52 in the largest-deviance bin. Spearman rho = +0.114.](plots/novel_deviance_scaling.png)

**Figure 1. Deviance scaling: a short rise, then nothing.** Per-model S7 rate (left) and mean
n_agree (right) against deviance. All six models rise over the first two bins and then run flat;
Spearman ρ(semitones, n_agree) = +0.114 — significant at n = 1806, negligible in size.

![Heatmap of cross-model agreement over the 43x43 frequency grid, standard frequency on the vertical axis and deviant frequency on the horizontal, on a single-hue light-to-dark Blues ramp from 0 to 6 agreeing models with the excluded diagonal in grey. The dark cells do not form bands parallel to the diagonal, which is what a deviance-driven response would produce; instead they concentrate in horizontal and vertical stripes at particular frequencies, and the pattern is visibly asymmetric about the diagonal.](plots/novel_n_agree_heatmap.png)

**Figure 2. Cross-model agreement over the 43×43 grid.** Row = standard, column = deviant, so
distance from the grey diagonal is deviance. The dark cells form **stripes at particular
frequencies, not bands parallel to the diagonal** — the signature of a frequency preference
rather than a deviance response. Asymmetry about the diagonal is the direction effect.

![Heatmap of the mean trough depth across all six models over the 43x43 frequency grid, on a red-white-blue diverging ramp whose neutral midpoint is exactly zero microvolts, with the excluded diagonal in grey. Values run from -3.44 to +1.11 microvolts. The dominant structure is horizontal and vertical striping rather than diagonal banding: the row at standard = 1745 Hz is uniformly dark red across all 42 of its deviants, the column at deviant = 7611 Hz is uniformly dark red down all 42 of its standards, and an isolated blue patch sits where standards of 2075 to 2691 Hz meet deviants of 336 to 400 Hz. Deviance, which is distance from the diagonal, is not the organising variable.](plots/phase1_mean_uv_heatmap.png)

**Figure 3. Mean trough depth over the same grid, and the clearest evidence in this memo.**
Cell value is `mean_uv_all6`; diverging ramp with its neutral midpoint pinned at exactly 0 µV, so
sign is readable directly. The row at standard = 1745 Hz and the column at deviant = 7611 Hz are
uniformly dark across all 42 of their partners, against a grand mean of −0.726 µV.

**Why this is the diagnostic panel.** A deviance-driven response would appear as bands running
*parallel to the diagonal*, since distance from the diagonal is deviance. That is not what the grid
shows. The structure is **rows and columns**: standard =
1745 Hz averages −1.505 µV across all 42 of its deviants, and deviant = 7611 Hz averages −1.768 µV
across all 42 of its standards, against a grand mean of −0.726 µV. A row stripe means "this
standard produces a trough regardless of what follows it"; a column stripe means "this deviant
produces a trough regardless of what preceded it". Both are frequency preferences. The diverging
ramp is pinned to 0 so the sign is readable directly — a fifth of the grid is positive (no trough
at all), and those cells are also organised by frequency, not by deviance.

![Small-multiple grid of 35 panels, the first of four figures covering 127 pairs. Each panel plots the FCz microvolt difference wave against time from -120 to 460 ms for one pair, with all six models overlaid in Okabe-Ito colours except wav2vec2-large, which is drawn in a mid grey so it does not dominate a twelve-trace panel; the regular direction is solid and the counter direction dashed, the 100-240 ms scoring window is shaded and the zero line marked. Each panel autoscales because the models' microvolt scales differ by roughly fivefold. The dominant visual pattern is that the regular and counter traces of the same model are near mirror images of each other about zero rather than both dipping in the scoring window, and that wav2vec2-large and whisper-medium carry excursions several times larger than the four whisper models.](plots/phase1_strong_waveforms_1.png)

**Figure 4. FCz µV difference waves, per model** (first of four; 127 pairs total). Regular solid,
counter dashed, 100–240 ms scoring window shaded. Each panel autoscales because the models' µV
scales differ ~5× — read shape, never one trace's depth against another's.

The full set is `phase1_strong_waveforms_1..4.png` — 35 + 35 + 35 + 22 panels. Traces are **per
model, not a mean over models**: Caveat 2 makes a cross-model µV mean an average over six
differently-shrunk axes, and each panel autoscales for the same reason.

![The same 35 panels with the six model traces collapsed to one line per direction: regular in blue solid, counter in orange dashed, each with a shaded plus-or-minus-one-standard-deviation band across the six models, on a shared vertical axis in z units fixed from -2 to 2. In panel after panel the two lines are near mirror images about zero — where the regular line dips inside the shaded 100 to 240 ms window the counter line rises by a similar amount, and vice versa — rather than both dipping, which is what a deviance response would look like. The two lines also overlap within each other's one-standard-deviation band for most of the trace, so the six models rarely separate the directions as cleanly as the mean lines suggest.](plots/phase1_direction_waveforms_1.png)

**Figure 5. The same panels collapsed to one line per direction** (first of four), in z units so
the six models can be averaged at all. **The two directions are near mirror images about zero**
rather than both dipping in the window — a deviance response would dip both ways. Band = ±1 SD
across models; the two lines sit inside each other's band more often than the means suggest.

The full set is `phase1_direction_waveforms_1..4.png`. It averages the **baseline-z-scored**
difference wave, not µV: z is normalised per model by its own pre-onset baseline, so the mean is
not simply whichever model has the least amplitude shrinkage, and it is the trace the S2 verdict is
actually computed on. The y-window is fixed at −2 to 2 across all 127 panels so they can be read
against each other; traces outside it are clipped, not rescaled. (A raw-µV variant is written to
`phase1_direction_waveforms_uv_*.png` by `--wave_units uv`; it shows the same pattern but is
dominated by wav2vec2-large and whisper-medium, and clips more at the same window, so it should
not be cited.)

**Why the mirror image matters.** A deviance response would dip both ways: reversing which tone is
standard and which is deviant should not flip the sign of the response. An anti-symmetric pair of
traces is instead what a response driven by *which tone arrives last* looks like — the same
conclusion Table 1's empty `pairs_both` column and Figure 3's row-and-column striping reach by
other routes.

### Section 2 summary
- **Agreement rises with deviance only over the first octave and then stops.** Mean n_agree goes
  1.88 → 2.95 between 2.8 and 11.6 semitones, plateaus at ~2.85 from 13 to 37 semitones, and falls
  to 2.52 beyond that. ρ = +0.114 — significant, negligible.
- **The grid is organised by frequency, not by deviance.** The mean-µV heatmap stripes by row and
  column, not in bands parallel to the diagonal.
- **Zero of 903 pairs reach 6/6 in both directions**, at either µV floor. The direction-collapsed
  waveforms show why: the two directions are near mirror images, not two dips.

---

## Section 3 — Do novel pairs beat the literature pairs?

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py`
> (`compare_to_literature`, `chance_baseline`)
> **Data:** `outputs/results_novel_search/phase1_mmn_s7_roi.csv` vs
> `outputs/results_24freq_7models/mmn_s7_roi.csv`, **scored identically** — same S7@0.75, same
> FCz, same mTRF, same 6 models (whisper-large dropped from the literature side too, so the
> comparison is like-for-like). Tables: `plots/phase1_novel_vs_literature.csv`,
> `plots/phase1_chance_baseline.csv`.

**Table 3. n_agree distribution, novel vs literature.** Percentages of each set's
direction-instances. "Chance" is the exact Poisson-binomial over the six models' own marginal S7
rates on the novel grid (0.325 to 0.599), i.e. what independent models would produce.

| n_agree | novel (n = 1806) | literature (n = 48) | chance |
|---:|---:|---:|---:|
| 6 | **0.9%** (17) | **0.0%** (0) | 0.7% |
| 5 | 6.3% (114) | 4.2% (2) | 5.5% |
| 4 | 16.0% (289) | 8.3% (4) | 18.0% |
| 3 | 29.9% (540) | 22.9% (11) | 30.8% |
| 2 | 31.1% (562) | 45.8% (22) | 28.6% |
| 1 | 13.8% (249) | 10.4% (5) | 13.7% |
| 0 | 1.9% (35) | 8.3% (4) | 2.6% |

**Table 4. Best pair on each side.** Ranked by n_agree, then mean_uv.

| | method | standard → deviant | semitones | % deviance | n_agree | mean_uv |
|---|---|---|---:|---:|---:|---:|
| novel | `method_1767` | 1745 → 7611 Hz | 25.50 | 336% | **6/6** | −3.183 µV |
| literature | `method_20` | 1000 → 1850 Hz | 10.66 | 85% | 5/6 | −1.365 µV |

**Table 5. Where the best novel pairs sit relative to the literature set.**

| | literature (24 methods) | novel 6/6 tier (17 instances) | novel grid overall |
|---|---:|---:|---:|
| deviance, median | 3.16 st (20%) | **25.50 st (336%)** | 19.50 st (206%) |
| deviance, range | 0.84 – 12.00 st | 9.00 – 63.00 st | 1.49 – 63.00 st |
| higher tone, median | 1200 Hz | **7611 Hz** | 2691 Hz |
| higher tone, max | 2000 Hz | 7611 Hz | 7611 Hz |
| share with higher tone ≥ 5869 Hz | **0 of 24** | **12 of 17** | 14% of grid |

**The novel grid's top tier does beat the literature's, and it does so entirely outside the
literature's range.** No published pair in the 24-method set exceeds 12 semitones or puts either
tone above 2000 Hz; the novel 6/6 tier has a median deviance of 25.5 semitones and 12 of its 17
members have a tone at or above 5869 Hz, with 9 of 17 involving the grid's top rung, 7611 Hz.
This is consistent with Section 2's finding that 7611 Hz is a dark column in the heatmap regardless
of what it is paired with — the "winning" pairs are largely the pairs that happen to contain the
frequencies these models respond to.

**Table 6. The chance baseline.** Observed vs expected direction-instances under independent
models, exact Poisson-binomial over the marginal S7 rates (whisper-tiny 0.423, whisper-base 0.325,
whisper-small 0.331, whisper-medium 0.531, wav2vec2-medium 0.599, wav2vec2-large 0.461).

| tier | observed | expected under independence | ratio | p (binomial, ≥ observed) |
|---|---:|---:|---:|---:|
| exactly 6/6 | 17 | 12.0 | 1.41× | **0.103** |
| ≥ 5/6 | 131 | 110.6 | 1.18× | 0.028 |

![Grouped bar chart of the n_agree distribution for the novel grid and the literature set, as percentages of each set's direction-instances, with the independent-models chance curve overlaid as a black dashed line with circular markers. The novel bars track the chance curve closely at every tier, sitting slightly above it at 4, 5 and 6 and slightly below at 3; the literature bars are shifted toward lower agreement, peaking at 46 percent at n_agree = 2 and reaching zero at 6.](plots/phase1_novel_vs_literature.png)

**Figure 6. Novel grid vs literature set vs chance**, scored identically. **The novel bars track
the chance curve at every tier.** The literature bars sit lower, peaking at n_agree = 2 and never
reaching 6 — but the novel grid's advantage over them is not an advantage over chance.

**The novel grid's agreement is not clearly above chance, and this is the most important result in
the memo.** 17 direction-instances at 6/6 against 12.0 expected is a 1.41× enrichment with
p = 0.103 — the same near-miss the literature screen reported (10 of 48 pairs at 7/7 against 6.9
expected, p = 0.087; see Section 11 of `results_analysis_with_counter.md`). The ≥5 tier does clear
p = 0.05 at 1.18×, but a 18% enrichment over chance on a one-sided test at n = 1806 is a thin
result to build a stimulus set on. And the baseline itself is *generous* to the observed counts:
six models sharing an architecture family, a training corpus and a frozen mTRF mapping are not
independent, so the true null is above the dashed line, not on it.

**A wider search space did not create agreement.** Going from 24 literature pairs to 903 novel ones
— a 37× larger space, covering deviances from 8.8% to 3705% — moved the top tier from 5/6 to 6/6
and moved the enrichment from 1.45× to 1.41×. The grid found more *instances* of near-chance
agreement, which is what a larger sample does.

### Section 3 summary
- **Yes, a novel pair beats the best literature pair**: `method_1767` (1745 → 7611 Hz) at 6/6 and
  −3.18 µV, against `method_20` (1000 → 1850 Hz) at 5/6 and −1.37 µV.
- **No, that does not mean the search worked.** 17 at 6/6 against 12.0 expected by chance,
  p = 0.103. The literature screen's near-chance result reproduces on a 37× larger space.
- The winners sit far outside the literature's frequency and deviance range, and cluster on the
  frequencies the heatmap already flags as preferred — 9 of 17 contain 7611 Hz.

---

## Section 4 — The consensus set, and which models drive it

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py`
> **Data:** `plots/phase1_top30.csv`, `plots/phase1_model_uv_box.csv`,
> `plots/phase1_direction_asymmetry.csv`
> **Definition.** Pairs where **both directions** reach n_agree = 6/6, sorted by the mean of the
> two directions' `mean_uv`. Requiring both directions is what distinguishes a deviance response
> from a **frequency preference** — a pair that only works one way is telling you the models like
> one of its two tones, not that they detect the change.

**Table 7. Consensus set (both directions at 6/6): EMPTY — 0 pairs of 903.**

No pair in the grid meets the definition, at X = 0.75 or at X = 0.50. This is a real outcome, not a
pipeline fault (Decision 0 in
`aux/analysis_with_counter/MMN_pipeline_analysis_decisions_notes_062226_with_counter.md`).

**Table 8. The nearest misses** — the only 4 pairs reaching n_agree ≥ 5 in *both* directions.

| pair | f_low → f_high | semitones | % deviance | mean_uv regular | mean_uv counter | mean of the two |
|---|---|---:|---:|---:|---:|---:|
| `method_1740` | 1600 → 3200 Hz (exact octave) | 12.00 | 100.0 | −3.995 | −1.491 | **−2.743** |
| `method_1744` | 1600 → 4525 Hz | 18.00 | 182.8 | −1.507 | −2.282 | −1.894 |
| `method_1679` | 1234 → 2263 Hz | 10.50 | 83.4 | −0.926 | −2.171 | −1.549 |
| `method_1718` | 1467 → 2263 Hz | 7.50 | 54.3 | −1.452 | −1.248 | −1.350 |

These four are the closest thing the search produced to a symmetric deviance response, and they are
tightly clustered: every f_low is between 1234 and 1600 Hz, every f_high between 2263 and 4525 Hz,
every deviance between 7.5 and 18 semitones. That is a *different* region from the 6/6 tier's
high-frequency winners, and it sits on the 1600 Hz row the heatmap flags. Even here the two
directions disagree substantially in depth — `method_1740` is 2.7× deeper one way than the other.

**Table 9. Top-30 direction-instances** (`plots/phase1_top30.csv`). Ranked by n_agree desc, then
mean_uv asc. **mean/median/max/min are across all six models' `trough_uv`, agreeing or not** — a
spread computed only over the agreeing models would narrow itself by construction as n_agree falls,
making tier-5 rows look tighter than tier-6 rows for a purely mechanical reason. `mean_uv` (the
ranking key) remains the agreeing-models-only mean, so at 6/6 the two columns coincide.

| rank | method | f_low | f_high | st | % dev | n_agree | mean_uv | mean all6 | median all6 | max all6 | min all6 | did not agree |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `method_1767` | 1745 | 7611 | 25.50 | 336 | 6 | −3.183 | −3.183 | −2.434 | −0.996 | −7.457 | — |
| 2 | `method_1783` | 1903 | 7611 | 24.00 | 300 | 6 | −2.514 | −2.514 | −2.369 | −1.582 | −3.724 | — |
| 3 | `method_1750` | 1600 | 7611 | 27.00 | 376 | 6 | −2.273 | −2.273 | −2.478 | −0.797 | −2.886 | — |
| 4 | `method_1882` | 4150 | 7611 | 10.50 | 83 | 6 | −2.223 | −2.223 | −1.973 | −1.280 | −3.671 | — |
| 5 | `method_1812` | 2263 | 7611 | 21.00 | 236 | 6 | −2.137 | −2.137 | −1.917 | −1.443 | −3.324 | — |
| 6 | `method_1627` | 951 | 7611 | 36.01 | 700 | 6 | −2.053 | −2.053 | −1.950 | −1.062 | −3.034 | — |
| 7 | `method_1603` | 872 | 7611 | 37.51 | 773 | 6 | −2.027 | −2.027 | −1.864 | −1.084 | −3.248 | — |
| 8 | `method_1066_counter` | 218 | 1745 | 36.01 | 700 | 6 | −1.987 | −1.987 | −1.399 | −0.904 | −3.658 | — |
| 9 | `method_1807` | 2263 | 4935 | 13.50 | 118 | 6 | −1.869 | −1.869 | −1.216 | −0.895 | −4.855 | — |
| 10 | `method_1881` | 4150 | 6979 | 9.00 | 68 | 6 | −1.822 | −1.822 | −1.188 | −0.764 | −4.926 | — |
| 11 | `method_1766` | 1745 | 6979 | 24.00 | 300 | 6 | −1.697 | −1.697 | −1.502 | −0.914 | −3.300 | — |
| 12 | `method_1525` | 673 | 7611 | 41.99 | 1031 | 6 | −1.644 | −1.644 | −1.364 | −1.049 | −3.020 | — |
| 13 | `method_1024_counter` | 200 | 1600 | 36.00 | 700 | 6 | −1.607 | −1.607 | −1.445 | −0.870 | −2.815 | — |
| 14 | `method_1809` | 2263 | 5869 | 16.50 | 159 | 6 | −1.389 | −1.389 | −1.287 | −0.847 | −2.351 | — |
| 15 | `method_1318_counter` | 400 | 951 | 14.99 | 138 | 6 | −1.358 | −1.358 | −1.269 | −1.014 | −1.717 | — |
| 16 | `method_1042` | 200 | 7611 | 63.00 | 3706 | 6 | −1.322 | −1.322 | −1.213 | −0.765 | −2.250 | — |
| 17 | `method_1106_counter` | 238 | 1745 | 34.49 | 633 | 6 | −1.201 | −1.201 | −1.123 | −1.017 | −1.635 | — |
| 18 | `method_1740` | 1600 | 3200 | 12.00 | 100 | 5 | −3.995 | −3.438 | −1.973 | −0.655 | −10.813 | whisper-tiny |
| 19 | `method_1515` | 673 | 3200 | 26.99 | 375 | 5 | −3.201 | −2.678 | −1.574 | −0.066 | −9.677 | whisper-base |
| 20 | `method_1762` | 1745 | 4935 | 18.00 | 183 | 5 | −2.966 | −2.579 | −1.794 | −0.641 | −8.303 | whisper-small |
| 21 | `method_1765` | 1745 | 6400 | 22.50 | 267 | 5 | −2.956 | −2.553 | −1.831 | −0.536 | −7.880 | whisper-small |
| 22 | `method_1730` | 1467 | 6400 | 25.50 | 336 | 5 | −2.693 | −2.157 | −1.564 | +0.525 | −6.325 | whisper-tiny |
| 23 | `method_1032` | 200 | 3200 | 48.00 | 1500 | 5 | −2.626 | −2.137 | −0.943 | +0.304 | −7.844 | whisper-small |
| 24 | `method_1601` | 872 | 6400 | 34.51 | 634 | 5 | −2.601 | −2.240 | −1.206 | −0.434 | −8.233 | whisper-tiny |
| 25 | `method_1737` | 1600 | 2468 | 7.50 | 54 | 5 | −2.496 | −2.147 | −1.634 | −0.407 | −6.023 | whisper-tiny |
| 26 | `method_1701` | 1345 | 2691 | 12.01 | 100 | 5 | −2.487 | −2.067 | −1.920 | +0.035 | −4.336 | whisper-tiny |
| 27 | `method_1710` | 1345 | 5869 | 25.51 | 336 | 5 | −2.458 | −2.055 | −1.809 | −0.041 | −5.454 | whisper-tiny |
| 28 | `method_1722` | 1467 | 3200 | 13.50 | 118 | 5 | −2.415 | −2.053 | −1.721 | −0.241 | −4.399 | whisper-tiny |
| 29 | `method_1835` | 2691 | 6400 | 15.00 | 138 | 5 | −2.321 | −1.943 | −0.905 | −0.054 | −5.283 | whisper-tiny |
| 30 | `method_1825` | 2468 | 7611 | 19.50 | 208 | 5 | −2.304 | −1.987 | −1.536 | −0.401 | −4.625 | wav2vec2-medium |

**whisper-tiny is the model that most often breaks a 6/6.** It is the sole dissenter in 8 of the 13
five-model rows above, and it has the smallest strong-set n of any model (85, against 127 for
wav2vec2-medium). At 0.423 its marginal S7 rate is mid-pack, so this is not simply a stricter
model — it is a model that disagrees on *these particular* pairs, which are exactly the ones the
other five rank highest.

![Horizontal boxplot with one row per model, showing the distribution of trough depth in microvolts at FCz over the direction-instances at n_agree >= 5, with individual points jittered over each box and the n annotated per row. The X = -0.75 microvolt floor is marked as a dashed vertical line at the right edge, and every distribution is bounded by it. The four whisper models sit in a narrow band between about -0.8 and -3 microvolts while wav2vec2-large spreads much wider, reaching below -10 microvolts, which reflects differing amplitude shrinkage across models and not differing MMN strength.](plots/phase1_model_uv_box.png)

**Figure 7. Trough depth per model on the n_agree ≥ 5 set.** Agreeing direction-instances only, n
annotated per row. **Read each row against its own −0.75 µV floor, never against another row** —
wav2vec2-large's wider spread is its mapping's amplitude scale, not a stronger MMN (Caveat 2).

Read Figure 7 **row by row, against each row's own X = −0.75 µV floor**. The rows are not on a
common scale: mTRF amplitude shrinkage differs by model and by committed layer, so wav2vec2-large's
wider spread is a property of its mapping, not evidence that it detects deviance more strongly
(Caveat 2). What *is* comparable within a row is how far each distribution clears its own floor —
and for every model the mass sits close to it, with medians between −1.12 µV (whisper-tiny) and
−2.53 µV (wav2vec2-large) against a −0.75 µV threshold.

**Table 10. Direction asymmetry across the whole grid** (`plots/phase1_direction_asymmetry.csv`).

| | value |
|---|---|
| pairs whose two directions land in the same n_agree tier | **196 of 903 (21.7%)** |
| mean \|n_agree(regular) − n_agree(counter)\| | **1.39 tiers** |
| pairs with a direction at ≥5 | 127 |
| …of which reach ≥5 both ways | **4 (3.1%)** |
| 6/6 instances that are `regular` (f_low → f_high, ascending) | 13 of 17 |
| mean n_agree, regular vs counter | 2.856 vs 2.483 |

**Asymmetry is the rule, not the exception.** Four fifths of pairs put their two directions in
different tiers, and the average gap is 1.4 tiers out of 6. There is also a systematic *direction*
to it: ascending pairs average 0.37 more agreeing models than descending ones, and 13 of the 17
6/6 instances are ascending. A deviance detector should be indifferent to the sign of the change.
A system that responds more to a high tone than a low one would produce exactly this.

### Section 4 summary
- **The consensus set is empty.** Zero pairs of 903 reach 6/6 in both directions; four reach ≥5
  both ways, all in a narrow 1234–1600 → 2263–4525 Hz region.
- **The direction asymmetry is systematic, not noisy**: 78% of pairs split their tiers, mean gap
  1.39, and ascending pairs beat descending ones by 0.37 models on average.
- **whisper-tiny is the usual dissenter** — sole holdout in 8 of the 13 five-model rows in the
  top 30.
- **No pair from this grid should be carried forward as a validated deviance stimulus on this
  evidence.**

---

## Section 5 — The structure of the ranking

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py` (`plot_ranking_structure`)
> **Data:** `plots/phase1_cutoff_curve.csv`, `plots/phase1_direction_asymmetry.csv`
> **Why this section exists.** Sections 2–4 ask whether the grid found anything. This one asks
> what the ranked list *looks like* — how many pairs survive any given cutoff, whether the ranking
> has a natural break, and whether the survivors occupy a particular region of the grid. It is the
> section to read before selecting any subset for further work.

![Four panels. Top left: pairs qualifying against the mean_uv cutoff, one curve per n_agree threshold on a symlog axis, with dashed reference lines marking the 127 pairs at n_agree >= 5 and the 17 at 6/6. Every curve is flat out to a cutoff of about 1.25 microvolts and then falls steeply, so the pair count is insensitive to the cutoff over the range where it would plausibly be set. Top right: absolute mean_uv against rank for the top 300 direction-instances, coloured by n_agree tier, showing a smooth continuous decay from 3.2 to 1.5 microvolts with no elbow; the only discontinuities are the tier boundaries at rank 17 and rank 131, which are artifacts of the sort key. Bottom left: f_low against f_high on log axes with all 903 pairs in grey and the 127 pairs having a direction at n_agree >= 5 in blue, showing the strong pairs spread broadly across the grid rather than clustered. Bottom right: a 7x7 matrix of pair counts by regular-direction n_agree against counter-direction n_agree, with the mass concentrated off the diagonal around (2,3) and (3,2), and only four pairs in the region where both directions reach 5 or more.](plots/phase1_ranking_structure.png)

**Figure 8. The shape of the ranked list.** Clockwise from top left: yield at every cutoff (flat
until ~1.25 µV, so the cutoff barely matters); |mean_uv| vs rank over the top 300 (**a smooth
decay with no elbow** — the only breaks are the tier boundaries at ranks 17 and 131, artifacts of
the sort key); where the 127 strong pairs sit in (f_low, f_high) space (**spread, not clustered**);
and each pair's two directions cross-tabulated (**mass off the diagonal; the 6/6 cell is empty**).

**Table 11. Yield at each agreement threshold.** Pairs with at least one direction at the given
tier. (Pairs, not direction-instances, because both directions of a pair travel together.)

| tier | direction-instances | distinct pairs | % of the 903 |
|---|---:|---:|---:|
| ≥ 6/6 | 17 | **17** | 1.9% |
| ≥ 5/6 | 131 | **127** | 14.1% |
| ≥ 4/6 | 420 | 387 | 42.9% |
| ≥ 3/6 | 960 | 720 | 79.7% |
| ≥ 2/6 | 1522 | 886 | 98.1% |
| ≥ 1/6 | 1771 | 903 | 100.0% |

Two features of this table are worth noticing. **The 6/6 and 5/6 tiers do not overlap in pairs** —
no pair has one direction at 6 and the other at 5 — so 17 + 110 = 127 exactly, with no
double-counting. And **the tiers grow very fast below 5/6**: dropping from ≥5 to ≥4 triples the
pair count, and by ≥2 the criterion admits 98% of the grid and has stopped selecting anything.

**There is no elbow in the ranking.** Figure 8's top-right panel plots |mean_uv| against rank for the top
300 direction-instances. It decays smoothly from 3.2 µV at rank 1 to 1.5 µV at rank 300 with no
break. The two visible discontinuities are at ranks 17 and 131, which are exactly the n_agree tier
boundaries, i.e. artifacts of the sort key rather than structure in the data. Within a tier the
decay is continuous. The same is true of the µV cutoff: in Figure 8's top-left panel every curve is flat
out to about |mean_uv| = 1.25 µV before falling, so any cutoff in the plausible range returns
essentially the tier's full membership. **Nothing in the ranking itself tells you where to stop** —
a cutoff has to come from the tier definition or from outside the data.

**The strong pairs are spread, not clustered.** Figure 8's bottom-left panel puts the 127 pairs with a
direction at ≥5 against all 903 in (f_low, f_high) space. They cover most of the grid, shifted
toward the top-left — median deviance 21.0 st against 19.5 grid-wide, median f_high 4150 Hz against
2691 Hz — but they do not concentrate in a region a narrower, finer follow-up grid could target.
The concentration that *does* exist is on individual frequencies (the 1745 Hz row, the 7611 Hz
column), which Section 2 identifies as the artifact rather than the signal.

**Direction asymmetry, seen as a matrix.** Figure 8's bottom-right panel cross-tabulates each pair's two
directions. The mass sits off the diagonal, clustered around (2,3) and (3,2); only 4 pairs land in
the region where both directions reach 5 or more, and the (6,6) cell is empty. This is Table 10 in
picture form, and it is the clearest single statement of why the consensus set is empty.

### Section 5 summary
- **The tiers are the only natural cut points.** 17 pairs at 6/6, 127 with a direction at ≥5, then
  387 at ≥4 — after which the criterion stops selecting.
- **There is no elbow**: |mean_uv| decays smoothly over the top 300, and the only breaks are tier
  boundaries, which are artifacts of the sort key.
- **The survivors are spread across the grid**, so there is no obvious sub-region for a finer
  follow-up — the only concentration is on individual preferred frequencies.

---

## Reproducing this memo

Full run guide: `aux/sophies_repository_overview.md` §17.5 (Stages A–J). After the cluster stages
and the rsync back:

```bash
python scripts/analyze_mmn_s7_roi.py --predictions_root outputs/insilico_mmn_predictions_novel \
    --dip_uv_threshold 0.75 --out outputs/results_novel_search/phase1_mmn_s7_roi.csv
python scripts/rank_novel_phase1.py                            # the ranking CSVs
python aux/analysis_novel_search/plots/novel_search_plots.py   # Section 2 heatmap + scaling
python aux/analysis_novel_search/plots/phase1_results.py       # Sections 2–5
```

`phase1_results.py` takes `--results_dir` / `--out_dir` / `--predictions_root` /
`--literature_csv`, so a re-run can be rendered anywhere without clobbering the committed figures,
and writes every table in this memo as a CSV beside its figure. `--wave_units uv` renders the
direction waveforms in raw microvolts; `--wave_ylim LO HI` changes their shared y-window.

*Every section is filled from the console output of the script named in its provenance header; the
ranking CSVs are the source of record. Extraction cost, budget and any selection of pairs for
further work are out of scope here and live in `aux/sophies_repository_overview.md` §17.4.*
