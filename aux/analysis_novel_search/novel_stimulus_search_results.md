# Novel tone-pair stimulus search — results

> **STATUS (2026-07-31): Phase 1 and Phase 2 both complete.** **Phase 1** extracted all 903
> pairs × 2 directions for 6 models and scored them from **one** deviant realization (N7/var1).
> **Phase 2** re-scored a selected **127 pairs × 2 directions = 254 direction-instances** from
> the **full 15-deviant grid**, so each condition's `deviant_mean` is now an average over 15
> draws rather than a single one. Same models, same frozen mTRF mapping, same S7 metric, same
> FCz electrode, same ranking code — the *only* thing that changed is how many deviants each
> mean is built from. The two runs keep separate prediction roots
> (`outputs/insilico_mmn_predictions_novel` and `..._novel_phase2`), so every waveform figure is
> traceable to the deviant count its caption claims. **Section 0 is the literature screen on its own; Sections 1–5 are Phase 1; Sections 6–7 are Phase 2 and the comparison.**
> This memo reports **results only** — run scale and disk requirements live in
> `aux/sophies_repository_overview.md` §17.4 (*Scale of the run*).

> **Extends** `aux/analysis_with_counter/results_analysis_with_counter.md`, which screened the
> **24 literature frequency methods** × 7 models. This memo screens a **903-pair novel frequency
> grid** × 6 models (whisper-large excluded), mTRF only, at the FCz electrode, at a fixed µV floor
> of X = 0.75. 903 unordered pairs × {regular, counter} = **1,806 direction-instances**.

> **The question.** The literature screen tested whatever pairs published MMN studies happened to
> use. This asks whether pairs *absent* from that literature drive stronger and more
> model-consistent responses.

> **The short answer, up front.** The novel grid's top tier beats the literature set's **on
> cross-model consistency, and essentially not at all on amplitude** — its top 10 averages 6.0
> agreeing models against the literature top 10's 4.0, but −2.21 µV against −2.07 µV, a gap of
> 0.14 µV. And **not one pair of 903 reaches the top tier in both directions.** Three independent
> views of the data — the empty both-directions column, the row-and-column striping of the µV
> heatmap, and the mirror-image direction waveforms — all say the same thing: the models are
> responding to **which frequencies are present**, not to the change between them. Table 3 puts a
> number on it: 7611 Hz averages 4.55 of 6 agreeing models when it is the deviant and 2.41 when it
> is the standard, over the same 42 partner frequencies each way.

> **What Phase 2 changed: nothing that mattered, and that is the result.** Fifteen deviants
> instead of one left the ranking nearly intact — ρ(Phase-1 rank, Phase-2 rank) = **+0.923**
> (p = 2.3 × 10⁻¹⁰⁶, n = 254) — and left the consensus set **still empty**: 0 of 127 pairs reach
> 6/6 in both directions, and **all four** pairs that managed ≥5 both ways on a single draw fail
> to hold on fifteen. The frequency preference is sharper on the fuller draw, not weaker: over
> the same 20 instances each way, **7611 Hz averages 5.10 of 6 agreeing models when it is the
> deviant and 2.35 when it is the standard**, and 1745 Hz does the mirror image (4.92 as the
> standard, 1.54 as the deviant). The high ρ is a good result *about the screen* — a 4-clip
> screen predicts a 32-clip evaluation well — and not a result about the stimuli.

---

## Caveats — all load-bearing, all apply to every section below

**1 — Which draw a number comes from, and which supersedes which.** The two halves of this memo
are scored from different numbers of deviants, and every figure has to be read against its own.

- **Sections 1–5 (Phase 1) are one deviant draw.** Every condition there is scored from a single
  realization (N7/var1), not a mean over the 15 available deviants. The models are deterministic,
  so this is **not measurement noise** — it is one sample from the stochastic-prefix distribution
  rather than its mean. **No µV figure in Sections 1–5 is final.**
- **Sections 6–7 (Phase 2) are the 15-deviant mean.** `deviant_mean` there averages all 15
  realizations, which is the quantity the design was meant to score, so those µV and `n_agree`
  values are not subject to the caveat above.
- **Phase-2 values supersede Phase-1 values for the 127 pairs Phase 2 covers.** Where the two
  disagree about a pair in that set — and they disagree about 78 of 254 direction-instances —
  Section 6 is the number to quote. For the **other 776 pairs the grid holds, Phase 1 is all
  there is**, and it is still one draw.

**2 — Amplitude shrinkage: these are not human-scale microvolts.** mTRF predictions are ~4×
amplitude-shrunk. X = 0.75 µV is calibrated to the *model's own* trough distribution
(median ≈ −0.8 µV), not to literature EEG scale (human MMN ≈ 1–5 µV). **Do not read `trough_uv` or
`mean_uv` as a literature-comparable amplitude**, and never compare `trough_uv` across models. The
scales genuinely differ: on the strong set, wav2vec2-large's median trough is −2.53 µV against
whisper-tiny's −1.12 µV, which says nothing about which model detects deviance better. This
constrains how Figures 3, 4, 5 and 7 may be read, and each says so on its face. It is also why
the cross-model mean in Figure 5 is drawn but not leaned on.

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

**6 — The Phase-2 set is selected, so it cannot support a population claim.** *(Sections 6–7
only.)* The 254 direction-instances there were chosen for having scored `n_agree ≥ 5` in Phase 1.
Any statistic whose meaning depends on the population being unselected is therefore confounded
there, and each is flagged where it appears: the **deviance correlation is not computed for
Phase 2 at all** (the selection removed the small-deviance instances it lived in), and the
tier-distribution comparison against the unselected literature set is not repeated either.
ρ = +0.923 likewise says the screen ranks reliably **among pairs it already ranked highly**; it
does not establish that the screen would have ordered the other 776 pairs correctly.

**7 — Phase 2 has a blind spot, by construction.** *(Sections 6–7 only.)* Only pairs that cleared
`n_agree ≥ 5` on their single Phase-1 draw were re-measured. A pair that would reach 6/6 in both
directions under 15 deviants but happened to score 4 or below on its one Phase-1 draw is
**invisible to this design** — it was never extracted. The rate is not zero: 8 of the 254
instances that *were* re-measured rose a tier (Table 21), all from the 3/6-or-below tiers, which
is the region the screen cut at. This is a limitation of the two-phase design, not a defect in
either run; closing it would mean evaluating pairs the screen rejected.

---

## Section 0 — The literature screen on its own terms

> **Code:** `aux/analysis_novel_search/plots/literature_results.py`
> **Data:** `outputs/results_soafix/mmn_s7_roi.csv` — the (FCz, mTRF, X = 0.75) slice over the same
> six models, so every count here is commensurable with every Phase-1 and Phase-2 number in this
> memo. Frequencies from `data/metadata/literature_frequency_intensity_duration_metadata.csv`;
> waveforms from `outputs/insilico_mmn_predictions_soafix` (15 deviant realizations per condition).
> Tables: `plots/literature_agreement_tiers.csv`, `plots/literature_top48.csv`,
> `plots/literature_consensus_heatmap.csv`.

Sections 1–7 use the 24 published frequency methods only as a *comparison arm* — a tier
distribution and a top 10 held up against the novel grid in Section 3. This section reports that
same set the way Sections 2–5 report the novel grid: its own agreement tiers, its own consensus
yield, its full ranking, and the predicted waveforms behind its top 10. Nothing here is re-scored.
**24 methods × {regular, counter} = 48 direction-instances**, and `n_agree` runs 0–6 throughout.

**Table 0.1. n_agree × number of stimuli, at X = 0.75.** `direction-instances` counts the 48
ordered instances; the last two columns count the 24 *methods*, split by whether both of a
method's directions land in the tier or only one. Both-vs-either is the direction-asymmetry
signal: a tier where the two columns nearly agree is measuring a deviance response, one where
`both` collapses is measuring a frequency preference.

| n_agree | direction-instances | % of 48 | methods with both directions | methods with either |
|---:|---:|---:|---:|---:|
| 6 | 0 | 0.0% | 0 | 0 |
| 5 | 2 | 4.2% | 0 | 2 |
| 4 | 6 | 12.5% | 0 | 6 |
| 3 | 17 | 35.4% | 1 | 16 |
| 2 | 18 | 37.5% | 3 | 15 |
| 1 | 4 | 8.3% | 0 | 4 |
| 0 | 1 | 2.1% | 0 | 1 |

- **No literature instance reaches 6/6, and only two reach 5/6** (`method_20` and `method_21`,
  both 1000 → 1850 Hz). The published set has no unanimous stimulus at this floor.
- **The mass sits at 2 and 3** — 35 of 48 instances, 73% of the set.
- **Agreement almost never survives reversal.** Of the 24 methods, only one has both directions at
  3/6 and three have both at 2/6; no method has both directions at 4/6 or better. This is the same
  frequency-preference result Section 2 finds on the novel grid, and it is already visible in the
  published set.

![Heatmap with the number of models agreeing on the vertical axis, 1 at the bottom to 6 at the top, and the top-X rank threshold on the horizontal axis running 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 48. Each cell is annotated with the number of the 48 literature direction-instances that sit in the top X of at least that many models, on a yellow-green-blue sequential ramp. The bottom row rises from 9 at top-2 to 48 by top-30. The 2-model row runs 2, 4, 9, 12, 16 across the first five columns; the 3-model row 1, 3, 3, 6, 10. The 4-model row is zero until top-6 and reaches only 5 at top-15. The 5-model row is zero across the whole 2-to-20 range and first appears at top-25 with 7, and the 6-model row is zero until a single stimulus appears at top-30, reaching 20 at top-40. The strongly coloured region is confined to the bottom-left-to-right sweep of the 1- and 2-model rows and the two right-hand columns, where the cut is so loose it admits most of the set.](plots/literature_consensus_heatmap.png)

**Figure 0.1. Consensus yield over the literature set**, built exactly as the Phase-1 and Phase-2
versions (Figures 11 and 20): each model ranks the instances by its *own* `trough_uv` among its S2
responses, and the cell counts instances in the top X of at least Y models.

- **The strict corner is empty.** No instance is in the top 10 of even 4 of the 6 models — the
  4-model row reaches 5 at top-15, and the 5- and 6-model rows are flat zero until top-25 and
  top-30 respectively.
- **The right-hand columns are degenerate, not encouraging.** Top-48 of a 48-instance set is every
  instance a model has an S2 response for, so the 30 at (6 models, top-48) is bounded by
  wav2vec2-large's 38 S2 responses rather than by any agreement.
- **This is the same shape the novel grid shows** (Section 5): the models order these stimuli
  largely independently, and cross-model consensus only appears once the cut is loose enough to be
  uninformative.

**Table 0.2. All 48 literature direction-instances, ranked.** Same columns and same keys as
Table 10 — n_agree descending, then `mean_uv` ascending. **mean/median/max/min are across all six
models' `trough_uv`, agreeing or not**; `mean_uv` (the ranking key) stays the agreeing-models-only
mean, and is undefined at `n_agree = 0`. Model names in the last column are abbreviated
(`tiny`/`base`/`small`/`medium` are the whisper models, `w2v-med`/`w2v-lg` the wav2vec2 pair).

| rank | method | stimulus | n_agree | mean_uv | mean all6 | median all6 | max all6 | min all6 | did not agree |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `method_20` | 1000 → 1850 Hz | 5 | −1.365 | −1.581 | −1.410 | −0.800 | −2.660 | w2v-lg |
| 2 | `method_21` | 1000 → 1850 Hz | 5 | −1.348 | −1.579 | −1.413 | −0.808 | −2.735 | w2v-lg |
| 3 | `method_19_counter` | 1850 → 1000 Hz | 4 | −3.270 | −2.164 | −1.266 | 0.777 | −7.902 | medium, w2v-med |
| 4 | `method_17_counter` | 1850 → 1000 Hz | 4 | −3.147 | −2.102 | −1.250 | 0.621 | −7.456 | medium, w2v-med |
| 5 | `method_18_counter` | 1850 → 1000 Hz | 4 | −3.132 | −2.098 | −1.255 | 0.625 | −7.377 | medium, w2v-med |
| 6 | `method_60` | 1000 → 1500 Hz | 4 | −1.968 | −1.350 | −1.187 | 0.102 | −3.404 | tiny, base |
| 7 | `method_75` | 1000 → 1200 Hz | 4 | −1.128 | −0.862 | −0.881 | 0.024 | −1.773 | base, small |
| 8 | `method_72` | 1000 → 1200 Hz | 4 | −1.115 | −0.854 | −0.848 | 0.022 | −1.784 | base, small |
| 9 | `method_60_counter` | 1500 → 1000 Hz | 3 | −2.216 | −1.303 | −0.809 | −0.165 | −3.119 | small, medium, w2v-med |
| 10 | `method_17` | 1000 → 1850 Hz | 3 | −2.010 | −1.029 | −0.805 | 0.443 | −2.546 | tiny, small, w2v-med |
| 11 | `method_18` | 1000 → 1850 Hz | 3 | −1.994 | −1.022 | −0.808 | 0.431 | −2.715 | tiny, small, w2v-med |
| 12 | `method_53_counter` | 1200 → 1000 Hz | 3 | −1.971 | −1.306 | −0.797 | −0.584 | −4.035 | tiny, base, medium |
| 13 | `method_19` | 1000 → 1850 Hz | 3 | −1.915 | −0.991 | −0.813 | 0.414 | −2.476 | tiny, small, w2v-med |
| 14 | `method_9` | 600 → 1000 Hz | 3 | −1.733 | −0.902 | −0.829 | 0.387 | −2.728 | tiny, base, w2v-med |
| 15 | `method_55_counter` | 2000 → 1000 Hz | 3 | −1.663 | −1.556 | −1.410 | 0.080 | −3.761 | small, medium, w2v-lg |
| 16 | `method_21_counter` | 1850 → 1000 Hz | 3 | −1.537 | −0.953 | −0.889 | −0.208 | −1.677 | base, small, medium |
| 17 | `method_20_counter` | 1850 → 1000 Hz | 3 | −1.530 | −0.977 | −0.930 | −0.200 | −1.682 | base, small, medium |
| 18 | `method_44` | 633 → 1000 Hz | 3 | −1.457 | −0.709 | −0.573 | 0.318 | −2.279 | base, w2v-med, w2v-lg |
| 19 | `method_55` | 1000 → 2000 Hz | 3 | −1.356 | −0.878 | −0.930 | −0.262 | −1.513 | medium, w2v-med, w2v-lg |
| 20 | `method_30_counter` | 1200 → 1000 Hz | 3 | −1.162 | −0.859 | −0.843 | −0.415 | −1.261 | tiny, base, medium |
| 21 | `method_29_counter` | 1200 → 1000 Hz | 3 | −1.130 | −0.848 | −0.838 | −0.418 | −1.221 | tiny, base, medium |
| 22 | `method_74` | 1000 → 1500 Hz | 3 | −1.112 | −0.484 | −0.491 | 0.277 | −1.220 | tiny, base, small |
| 23 | `method_31_counter` | 1200 → 1000 Hz | 3 | −1.102 | −0.833 | −0.843 | −0.424 | −1.192 | tiny, base, medium |
| 24 | `method_28_counter` | 1200 → 1000 Hz | 3 | −1.088 | −0.822 | −0.840 | −0.420 | −1.193 | tiny, base, medium |
| 25 | `method_32_counter` | 1200 → 1000 Hz | 3 | −1.070 | −0.818 | −0.838 | −0.421 | −1.181 | tiny, base, medium |
| 26 | `method_43` | 633 → 700 Hz | 2 | −5.361 | −1.838 | −0.321 | 0.267 | −9.609 | tiny, base, small, w2v-med |
| 27 | `method_44_counter` | 1000 → 633 Hz | 2 | −3.658 | −1.376 | −0.677 | 0.880 | −5.686 | tiny, base, small, w2v-lg |
| 28 | `method_31` | 1000 → 1200 Hz | 2 | −1.812 | −0.829 | −0.474 | −0.190 | −1.864 | tiny, base, small, w2v-lg |
| 29 | `method_28` | 1000 → 1200 Hz | 2 | −1.804 | −0.830 | −0.484 | −0.194 | −1.875 | tiny, base, small, w2v-lg |
| 30 | `method_29` | 1000 → 1200 Hz | 2 | −1.801 | −0.845 | −0.526 | −0.202 | −1.902 | tiny, base, small, w2v-lg |
| 31 | `method_30` | 1000 → 1200 Hz | 2 | −1.785 | −0.840 | −0.534 | −0.194 | −1.874 | tiny, base, small, w2v-lg |
| 32 | `method_32` | 1000 → 1200 Hz | 2 | −1.773 | −0.832 | −0.518 | −0.196 | −1.868 | tiny, base, small, w2v-lg |
| 33 | `method_33_counter` | 1200 → 1000 Hz | 2 | −1.766 | −0.610 | −0.264 | 0.365 | −2.595 | tiny, base, small, w2v-med |
| 34 | `method_53` | 1000 → 1200 Hz | 2 | −1.645 | −0.752 | −0.502 | −0.058 | −2.193 | tiny, base, small, w2v-lg |
| 35 | `method_27` | 1000 → 1064 Hz | 2 | −1.541 | −0.849 | −0.811 | 0.041 | −2.082 | tiny, base, small, medium |
| 36 | `method_33` | 1000 → 1200 Hz | 2 | −1.435 | −0.825 | −0.670 | −0.297 | −2.081 | tiny, small, medium, w2v-lg |
| 37 | `method_12_counter` | 1200 → 1000 Hz | 2 | −1.341 | −0.481 | −0.225 | 0.156 | −1.877 | tiny, small, medium, w2v-lg |
| 38 | `method_9_counter` | 1000 → 600 Hz | 2 | −1.156 | 0.106 | −0.366 | 3.674 | −1.357 | base, small, medium, w2v-lg |
| 39 | `method_74_counter` | 1500 → 1000 Hz | 2 | −1.145 | −0.551 | −0.457 | −0.038 | −1.251 | base, small, w2v-med, w2v-lg |
| 40 | `method_12` | 1000 → 1200 Hz | 2 | −1.031 | −0.458 | −0.389 | 0.356 | −1.277 | tiny, base, small, w2v-lg |
| 41 | `method_37` | 1000 → 1050 Hz | 2 | −0.961 | −0.476 | −0.269 | −0.188 | −1.145 | tiny, base, small, w2v-med |
| 42 | `method_10` | 1000 → 1122 Hz | 2 | −0.941 | −0.320 | −0.130 | 0.191 | −1.097 | base, small, w2v-med, w2v-lg |
| 43 | `method_37_counter` | 1050 → 1000 Hz | 2 | −0.935 | −0.557 | −0.528 | −0.057 | −0.981 | tiny, base, small, w2v-lg |
| 44 | `method_10_counter` | 1122 → 1000 Hz | 1 | −1.987 | −0.780 | −0.592 | −0.387 | −1.987 | tiny, base, small, medium, w2v-med |
| 45 | `method_27_counter` | 1064 → 1000 Hz | 1 | −0.966 | 0.063 | 0.034 | 1.282 | −0.966 | tiny, base, small, w2v-med, w2v-lg |
| 46 | `method_75_counter` | 1200 → 1000 Hz | 1 | −0.906 | −0.345 | −0.283 | 0.102 | −0.906 | tiny, base, small, medium, w2v-med |
| 47 | `method_72_counter` | 1200 → 1000 Hz | 1 | −0.820 | −0.340 | −0.270 | 0.107 | −0.820 | tiny, base, small, medium, w2v-lg |
| 48 | `method_43_counter` | 700 → 633 Hz | 0 | — | −0.126 | −0.383 | 1.387 | −0.582 | tiny, base, small, medium, w2v-med, w2v-lg |


- **The ranking is led by depth only after agreement.** `method_19/17/18_counter` (1850 → 1000 Hz)
  carry by far the deepest agreeing-model troughs in the set at −3.27 to −3.13 µV, but sit at
  ranks 3–5 because two models do not clear the floor on them; the two 5/6 instances above them
  are less than half as deep.
- **`min all6` shows how much of that depth is one model.** On those three instances the minimum
  across models is −7.9 to −7.4 µV against a median near −1.26 — wav2vec2-large alone, whose µV
  scale is not comparable with the others (Caveat 2).
- **What fails is the µV floor, not the shape.** Across the 288 model-instance cells, 262 satisfy
  S2 — an MMN-shaped trough with a recovery — but only 125 reach S7. So 137 of the 163 failures
  are traces that dipped and recovered in the window and simply did not reach 0.75 µV; only 26
  lack the shape at all.
- **Both directions of the same method routinely land far apart** — `method_10` at rank 42 against
  `method_10_counter` at 44, but `method_19` at 13 against `method_19_counter` at 3, and
  `method_43` at 26 against `method_43_counter` at 48, last in the set.

![Small-multiple grid of ten panels in two rows of five, one per instance in the literature top 10, each plotting the FCz microvolt difference wave against time from -120 to 360 ms after the final tone's onset with the six models overlaid in Okabe-Ito colours and wav2vec2-large re-hued violet. The 100-240 ms scoring window is shaded and the zero line marked, and each panel autoscales because the models' microvolt scales differ by roughly fivefold. Panels 1 and 2, method_20 and method_21, span about -3 to +3 microvolts with the traces crossing zero repeatedly and no common trough. Panels 3, 4 and 5, the three 1850 to 1000 Hz counter instances, are dominated by a single violet wav2vec2-large trace that descends to about -8 to -11 microvolts while the other five stay within about plus or minus 2.5, so the panel's vertical range is set by one model. Panels 6 through 10 span roughly -4 to +4 with whisper medium in orange swinging widest. Across all ten panels the traces are visibly jagged from sample to sample and no panel shows the six models dipping together inside the shaded window.](plots/literature_waveforms.png)

**Figure 0.2. FCz µV difference wave for the literature top 10**, ranked by `n_agree` then
`mean_uv` — the same order as Table 0.2's first ten rows. Five of the ten are `_counter`
instances, so the figure is drawn per *direction-instance* rather than per method: plotting
`method_19`'s regular trace under a panel titled by `method_19_counter`'s rank would show a trace
the ranking never scored. The axis runs to **360 ms**, the full span the criteria read out over —
S2's recovery search follows a trough that may itself sit as late as 240 ms.

- **The deepest instances are one model, not six.** Panels 3–5 (`method_19/17/18_counter`) look
  dramatic, but the excursion is wav2vec2-large alone, while the other five stay inside ±2.5 µV.
  Its scored trough is the −7.90/−7.46/−7.38 µV `min all6` of Table 0.2, against a `median all6`
  near −1.26; the trace then keeps descending past −10 µV *after* the scoring window, which is
  drift the criteria never read. This is why `mean_uv` at 4/6 can be deeper than `mean_uv` at 5/6
  without meaning a stronger response.
- **No panel shows six models dipping together.** Even at the top of the ranking the traces cross
  zero independently through the shaded window — the visual counterpart of no instance reaching
  6/6.
- **The traces are jagged at the 20 ms sample grid.** These are single-condition in-silico
  predictions averaged over 15 deviant realizations, not trial-averaged EEG; read the presence and
  timing of a trough, not its fine shape.

### Section 0 summary
- **The published set produces no unanimous stimulus.** 0 of 48 instances reach 6/6 and 2 reach
  5/6; 73% sit at 2 or 3 of 6.
- **Its failures are amplitude, not morphology** — 262 of 288 model-instance cells satisfy S2 and
  only 125 clear the 0.75 µV floor.
- **Agreement does not survive reversal**: no method has both directions at 4/6 or better, which
  is the frequency-preference signature Section 2 finds across the whole novel grid.
- **This is the baseline Section 3's comparison is against**, and it is why that comparison turns
  on consistency rather than on trough depth.

---

## Section 1 — What does the grid cover?

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
> `plots/phase1_agreement_tiers.csv`, `plots/phase1_deviance_octiles.csv`,
> `plots/phase1_frequency_involvement.csv`, `plots/phase1_mean_uv_grid.csv`,
> `plots/phase1_frequency_stripes.csv`
> **Definitions.** **S2** = negative trough in 100–240 ms that recovers ≥50% of its depth within
> 120 ms. **S7@X** = `S2 AND (trough_uv ≤ −X µV)`; here X = 0.75, at FCz, mTRF.
> **n_agree** = how many of the 6 models show S7 (0–6). **mean_uv** = mean `trough_uv` across
> *only* the agreeing models — averaging in models that failed S2 would mix in latencies that are
> not MMN latencies. Undefined when `n_agree = 0`. **mean_uv_all6** = mean `trough_uv` across all
> six regardless of S7; a different quantity, used in Figures 3 and 4 and in Table 3.

**Table 1. Agreement tiers.** Direction-instances out of 1,806, at the single reporting floor
X = 0.75 µV (`plots/phase1_agreement_tiers.csv`). The pairs view of the same tiers — how often a
pair reaches a tier in *both* directions — is Table 8, where it does its work.

| n_agree | direction-instances | % of 1,806 |
|---:|---:|---:|
| 6 | 17 | 0.9% |
| 5 | 114 | 6.3% |
| 4 | 289 | 16.0% |
| 3 | 540 | 29.9% |
| 2 | 562 | 31.1% |
| 1 | 249 | 13.8% |
| 0 | 35 | 1.9% |

- **The top tier is thin: 17 of 1,806 direction-instances, 0.9%.**
- Two thirds of the grid sits at 2/6 or 3/6 — the models mostly half-agree.

**Table 2. n_agree vs deviance, by octile of semitone distance**
(`plots/phase1_deviance_octiles.csv`). Spearman ρ(semitones, n_agree) = **+0.114**
(p = 1.2 × 10⁻⁶, n = 1806).

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

- **The expected monotone-rise-then-plateau signature is present but weak and short.** Agreement
  climbs from 1.88 to 2.95 mean n_agree between 2.8 and 11.6 semitones, then goes completely flat
  (2.83–2.85 across four consecutive octiles spanning 13 to 37 semitones) and finally *declines*
  at the largest deviances.
- **The metric is not measuring nothing** — it separates a 3-semitone change from a 12-semitone
  one — but **it carries no information above roughly one octave**, which is where almost the
  entire grid lives.
- ρ = +0.114 is a real correlation on 1,806 points and a negligible effect size.

**Table 3. What a frequency's role is worth** (`plots/phase1_frequency_involvement.csv`). Every
frequency appears as the standard 42 times and as the deviant 42 times, so the two halves of each
row are directly comparable. A representative selection; the CSV carries all 43.

| Hz | mean n_agree **as standard** | mean n_agree **as deviant** | mean_uv_all6 as standard | mean_uv_all6 as deviant |
|---:|---:|---:|---:|---:|
| 7611 | 2.41 | **4.55** | −0.438 | **−1.768** |
| 6979 | 1.38 | **3.62** | −0.251 | −1.041 |
| 6400 | 1.74 | **3.05** | −0.528 | −1.085 |
| 4150 | 3.74 | 2.50 | −1.115 | −0.888 |
| 2263 | 3.12 | 2.91 | −0.857 | −1.105 |
| 1745 | 3.81 | 2.05 | **−1.505** | −0.287 |
| 1600 | **4.07** | 2.12 | −1.414 | −0.423 |
| 200 | 2.48 | 2.93 | −0.510 | −0.885 |

- **A frequency's score depends on which side of the pair it sits on.** 7611 Hz averages 4.55 of 6
  agreeing models as the deviant and 2.41 as the standard, over 42 instances each way; 1600 Hz
  does the mirror image (4.07 as the standard, 2.12 as the deviant).
- **Nothing about deviance can explain this** — the n is identical in both roles, and the partner
  frequencies are the same 42 in both.
- The two extremes here are exactly the row and column Figure 3 shows as stripes.

![Two panels. Left: per-model S7 rate against deviance in semitones, six Okabe-Ito series each with a distinct marker; all six rise steeply from the smallest deviance bin and then run nearly flat, with wav2vec2-medium highest at around 60 percent and whisper-base lowest at around 33 percent. Right: mean n_agree against deviance with standard-error bars, rising from 1.9 to about 2.95 over the first two bins, then flat near 2.85 for four bins, then falling to 2.52 in the largest-deviance bin. Spearman rho = +0.114.](plots/novel_deviance_scaling.png)

**Figure 1. Deviance scaling.** Per-model S7 rate (left) and mean n_agree (right) against deviance.

- All six models rise over the first two bins and then run flat: the rise is real and short.

![Heatmap of cross-model agreement over the 43x43 frequency grid, standard frequency on the vertical axis and deviant frequency on the horizontal, on a single-hue light-to-dark Blues ramp from 0 to 6 agreeing models with the excluded diagonal in grey. The dark cells do not form bands parallel to the diagonal, which is what a deviance-driven response would produce; instead they concentrate in horizontal and vertical stripes at particular frequencies, and the pattern is visibly asymmetric about the diagonal.](plots/novel_n_agree_heatmap.png)

**Figure 2. Cross-model agreement over the 43×43 grid.** Row = standard, column = deviant, so
distance from the grey diagonal is deviance.

- **The dark cells form stripes at particular frequencies, not bands parallel to the diagonal** —
  the signature of a frequency preference rather than a deviance response.
- Asymmetry about the diagonal is the direction effect.

![Heatmap of the mean trough depth across all six models over the 43x43 frequency grid, on a diverging blue-white-red ramp whose neutral midpoint is exactly zero microvolts, with blue for negative values, an MMN-like trough, and red for positive values, no trough, and the excluded diagonal in grey. Values run from -3.44 to +1.11 microvolts. The dominant structure is horizontal and vertical striping rather than diagonal banding: the row at standard = 1745 Hz is uniformly dark blue across all 42 of its deviants, the column at deviant = 7611 Hz is uniformly dark blue down all 42 of its standards, and an isolated red patch sits where standards of 2075 to 2691 Hz meet deviants of 336 to 400 Hz. Deviance, which is distance from the diagonal, is not the organising variable.](plots/phase1_mean_uv_heatmap.png)

**Figure 3. Mean trough depth over the same grid — the clearest evidence in this memo.** Cell
value is `mean_uv_all6`; **blue = negative = an MMN-like trough, red = positive = no trough**, with
the ramp's neutral midpoint pinned at exactly 0 µV so sign is readable directly.

- **A deviance-driven response would appear as bands parallel to the diagonal**, since distance
  from the diagonal is deviance. That is not what the grid shows.
- **The structure is rows and columns.** Standard = 1745 Hz averages −1.505 µV across all 42 of its
  deviants; deviant = 7611 Hz averages −1.768 µV across all 42 of its standards; the grand mean is
  −0.726 µV. A row stripe means "this standard produces a trough regardless of what follows it"; a
  column stripe, "this deviant produces a trough regardless of what preceded it". Both are
  frequency preferences.
- **A fifth of the grid is positive** (no trough at all), and those cells are also organised by
  frequency, not by deviance.

![Two panels in standard-frequency by deviant-frequency space on logarithmic axes labelled from 200 to 8000 Hz, with a grey identity line. Circles are the 48 literature direction-instances; triangles are the top 24 novel direction-instances. Left panel colours both by n_agree on the same discrete light-to-dark Blues ramp as Figure 2: the literature circles are mid-ramp steps, 35 of the 48 at 2 or 3, with six at 4, two at 5, four pale ones at 1 and a single near-white one at 0, while the novel triangles are uniformly the darkest step at 6, a step no circle reaches. Right panel colours both by mean trough depth across all six models on the same blue-white-red diverging ramp and the same zero-centred normalisation as Figure 3: the literature circles run from near-white through mid blue, a few of them faintly warm, and the novel triangles are a deeper and much more uniform blue, though the darkest circles and the lighter triangles now overlap in shade rather than separating cleanly. The two sets barely overlap in position — every circle sits in a tight cluster around a 1000 Hz standard with deviants from 600 to 2000 Hz, while the triangles sit mostly along the top edge at a 7611 Hz deviant, with a few at a 3100 Hz deviant and a group at low deviants of 200 to 250 Hz against standards near 1500 to 1750 Hz.](plots/phase1_literature_heatmap.png)

**Figure 4. The 24 literature stimuli (circles) and the top 24 novel instances (triangles) on one
pair of colour scales**, identical to Figures 2 and 3, so a marker here and a cell there of the
same colour mean the same number.

- **The two sets barely overlap in position.** Every literature pair lives in 600–2000 Hz, which
  the novel ladder covers in 14 of its 43 rungs; the novel winners sit outside it, most of them
  against a 7611 Hz deviant.
- **They separate on agreement, not on depth.** On the left panel the novel triangles are uniformly
  the darkest step and no literature circle reaches it. On the right panel the two sets overlap:
  the deepest literature circles are as dark as, and in places darker than, the lighter novel
  triangles.
- Because the scales are shared and not renormalised, that is a statement about magnitude rather
  than about contrast stretching — and it is the same split Tables 5 and 6 make numerically, where
  the sets differ by two agreeing models but only 0.139 µV in depth.
- **Nothing on the µV panel is clipped.** Every literature `mean_uv_all6` falls inside the novel
  grid's −3.44 to +1.11 µV range, so the shared normalisation shows the literature set at full
  contrast rather than saturating it at the ramp end.

![Small-multiple grid of 30 panels, one per pair in the top 30 of the ranking. Each panel plots the FCz microvolt difference wave against time from -120 to 460 ms for the regular direction only, with the six models overlaid in Okabe-Ito colours and wav2vec2-large re-hued violet. The 100-240 ms scoring window is shaded and the zero line marked, and each panel autoscales because the models' microvolt scales differ by roughly fivefold. Whisper medium in orange and wav2vec2 large in violet swing several times wider than the four other models, both above and below zero.](plots/phase1_strong_waveforms_1.png)

**Figure 5. FCz µV difference waves for the top 30 pairs**, regular direction only, six per-model
traces.

- **Each panel autoscales** because the models' µV scales differ ~5× — read shape, never one
  trace's depth against another's. No cross-model mean is drawn: it would be an average over six
  incomparable scales, dominated by whisper-medium and wav2vec2-large.
- The de-overlaid per-model versions of this chunk are `phase1_strong_waveforms_1__<model>.png`,
  which share a y-axis within a model and so make pairs comparable to each other.

![The same 30 panels with the six model traces collapsed to one line per direction: regular in blue solid, counter in orange dashed, on a shared vertical axis in z units fixed from -2 to 2, with no shaded band. In panel after panel the two lines are near mirror images about zero — where the regular line dips inside the shaded 100 to 240 ms window the counter line rises by a similar amount, and vice versa — rather than both dipping, which is what a deviance response would look like.](plots/phase1_direction_waveforms_1.png)

**Figure 6. The top 30 pairs collapsed to one line per direction**, in z units so the six models
can be averaged at all.

- **The two directions are near mirror images about zero** rather than both dipping in the window.
  A deviance response would dip both ways: reversing which tone is standard and which is deviant
  should not flip the sign of the response.
- **An anti-symmetric pair of traces is what a response driven by which tone arrives last looks
  like** — the same conclusion Table 8's empty `pairs_both` column and Figure 3's striping reach by
  other routes.
- z is normalised per model by its own pre-onset baseline, so the mean is not simply whichever
  model has the least amplitude shrinkage, and it is the trace the S2 verdict is computed on. The
  raw-µV variant is `phase1_direction_waveforms_uv_1.png`; it shows the same pattern but is
  dominated by wav2vec2-large and whisper-medium.

### Section 2 summary
- **Agreement rises with deviance only over the first octave and then stops.** Mean n_agree goes
  1.88 → 2.95 between 2.8 and 11.6 semitones, plateaus at ~2.85 from 13 to 37 semitones, and falls
  to 2.52 beyond that. ρ = +0.114 — significant, negligible.
- **The grid is organised by frequency, not by deviance.** 7611 Hz scores 4.55 of 6 as a deviant
  and 2.41 as a standard over the same 42 partners; the mean-µV heatmap stripes by row and column,
  not in bands parallel to the diagonal.
- **Zero of 903 pairs reach 6/6 in both directions** (Table 8). The direction-collapsed waveforms
  show why: the two directions are near mirror images, not two dips.

---

## Section 3 — Do novel pairs beat the literature pairs?

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py` (`load_literature`,
> `compare_to_literature`, `table_novel_vs_literature_top`, `plot_literature_heatmaps`)
> **Data:** `outputs/results_novel_search/phase1_mmn_s7_roi.csv` vs
> `outputs/results_soafix/mmn_s7_roi.csv`, **scored identically** — same S7@0.75, same
> FCz, same mTRF, same 6 models (whisper-large dropped from the literature side too, so the
> comparison is like-for-like). Frequencies from
> `data/metadata/literature_frequency_intensity_duration_metadata.csv`. Tables:
> `plots/phase1_novel_vs_literature.csv`, `plots/phase1_top10_vs_literature.csv`.

**Table 4. n_agree distribution, novel vs literature.** Percentages of each set's
direction-instances.

| n_agree | novel (n = 1806) | literature (n = 48) |
|---:|---:|---:|
| 6 | **0.9%** (17) | **0.0%** (0) |
| 5 | 6.3% (114) | 4.2% (2) |
| 4 | 16.0% (289) | 12.5% (6) |
| 3 | 29.9% (540) | 35.4% (17) |
| 2 | 31.1% (562) | 37.5% (18) |
| 1 | 13.8% (249) | 8.3% (4) |
| 0 | 1.9% (35) | 2.1% (1) |

**Table 5. The novel grid's top 10** (`plots/phase1_top10_vs_literature.csv`).

| rank | method | stimulus | n_agree | mean_uv |
|---:|---|---|---:|---:|
| 1 | `method_1767` | 1745 → 7611 Hz | 6 | −3.183 |
| 2 | `method_1783` | 1903 → 7611 Hz | 6 | −2.514 |
| 3 | `method_1750` | 1600 → 7611 Hz | 6 | −2.273 |
| 4 | `method_1882` | 4150 → 7611 Hz | 6 | −2.223 |
| 5 | `method_1812` | 2263 → 7611 Hz | 6 | −2.137 |
| 6 | `method_1627` | 951 → 7611 Hz | 6 | −2.054 |
| 7 | `method_1603` | 872 → 7611 Hz | 6 | −2.027 |
| 8 | `method_1066_counter` | 1745 → 218 Hz | 6 | −1.987 |
| 9 | `method_1807` | 2263 → 4935 Hz | 6 | −1.869 |
| 10 | `method_1881` | 4150 → 6979 Hz | 6 | −1.822 |
| | **mean of the top 10** | | **6.0** | **-2.209** |

**Table 6. The literature set's top 10** (`plots/phase1_top10_vs_literature.csv`), ranked on the
same two keys.

| rank | method | stimulus | n_agree | mean_uv |
|---:|---|---|---:|---:|
| 1 | `method_20` | 1000 → 1850 Hz | 5 | −1.365 |
| 2 | `method_21` | 1000 → 1850 Hz | 5 | −1.348 |
| 3 | `method_19_counter` | 1850 → 1000 Hz | 4 | −3.270 |
| 4 | `method_17_counter` | 1850 → 1000 Hz | 4 | −3.147 |
| 5 | `method_18_counter` | 1850 → 1000 Hz | 4 | −3.132 |
| 6 | `method_60` | 1000 → 1500 Hz | 4 | −1.968 |
| 7 | `method_75` | 1000 → 1200 Hz | 4 | −1.128 |
| 8 | `method_72` | 1000 → 1200 Hz | 4 | −1.115 |
| 9 | `method_60_counter` | 1500 → 1000 Hz | 3 | −2.216 |
| 10 | `method_17` | 1000 → 1850 Hz | 3 | −2.010 |
| | **mean of the top 10** | | **4.0** | **-2.070** |

- **The novel top 10 is unanimous and the literature top 10 is not.** All ten novel instances reach
  6/6; the literature's best two reach 5/6, no literature instance reaches 6/6 at all, and its
  tenth reaches 3/6. Mean n_agree **6.0 against 4.0**.
- **On depth the two sets are close to level**: mean `mean_uv` **−2.209 µV against −2.070 µV**, a
  gap of 0.139 µV, or 7%. **Depth does not separate the novel grid from the literature set** — the
  deepest instance in the literature top 10 (`method_19_counter`, 1850 → 1000 Hz, −3.270 µV) is
  deeper than every one of the novel top 10, and the two behind it (`method_17_counter` −3.147,
  `method_18_counter` −3.132) are deeper than nine of the ten.
- **The whole of the margin is in agreement, not in depth.** A literature pair can drive a trough
  as deep as anything on the novel grid; what no literature pair does is get all six models to
  register it. That is the one axis on which the novel grid is clearly ahead, and it is the axis
  the search was built to optimise.

**Table 7. Where the best novel pairs sit relative to the literature set.**

| | literature (24 methods) | novel 6/6 tier (17 instances) | novel grid overall |
|---|---:|---:|---:|
| deviance, median | 3.16 st (20%) | **25.50 st (336%)** | 19.50 st (206%) |
| deviance, range | 0.84 – 12.00 st | 9.00 – 63.00 st | 1.49 – 63.00 st |
| higher tone, median | 1200 Hz | **7611 Hz** | 2691 Hz |
| higher tone, max | 2000 Hz | 7611 Hz | 7611 Hz |
| share with higher tone ≥ 5869 Hz | **0 of 24** | **12 of 17** | 14% of grid |

- **The novel grid's top tier out-agrees the literature's from entirely outside the literature's
  range.** No published pair in the 24-method set exceeds 12 semitones or puts either tone above
  2000 Hz; the novel 6/6 tier has a median deviance of 25.5 semitones and 12 of its 17 members have
  a tone at or above 5869 Hz, 9 of 17 involving the grid's top rung, 7611 Hz. The stimuli that get
  all six models to agree are ones the published literature has not tested.
- **This is consistent with Section 2's finding, not independent of it.** 7611 Hz is a dark column
  in the heatmap regardless of what it is paired with, so the "winning" pairs are largely the pairs
  that happen to contain the frequencies these models respond to.
- **A wider search space produced more-agreed pairs, not deeper ones and not a different kind of
  pair.** Going from 24 literature pairs to 903 novel ones — a 37× larger space covering 8.8% to
  3705% deviance — moved the top tier from 5/6 to 6/6 but left mean top-10 depth essentially where
  it was, and the winners are concentrated on individual preferred frequencies.

### Section 3 summary
- **On consistency yes; on amplitude no.** The novel top 10 averages 6.0 agreeing models against
  the literature top 10's 4.0, but −2.209 µV against −2.070 µV — a 0.139 µV difference in depth
  that carries no weight next to a two-model difference in agreement.
- **The margin is in agreement alone** — the deepest instance in the literature top 10 (−3.270 µV)
  beats every one of the novel top 10, so any claim that novel pairs drive *deeper* MMN is not
  supported.
- **The winners sit far outside the literature's frequency and deviance range**, and cluster on the
  frequencies Section 2 identifies as preferred: 9 of 17 6/6 instances contain 7611 Hz.

---

## Section 4 — Is there a consensus set, and which models decide it?

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py`
> **Data:** `plots/phase1_agreement_tiers.csv`, `plots/phase1_top50.csv`,
> `plots/phase1_model_uv_box.csv`, `plots/phase1_model_dissent.csv`,
> `plots/phase1_direction_asymmetry.csv`
> **Definition.** The **consensus set** is the pairs where **both directions** reach n_agree = 6/6.
> Requiring both directions is what distinguishes a deviance response from a **frequency
> preference** — a pair that only works one way is telling you the models like one of its two
> tones, not that they detect the change.

**Table 8. Agreement tiers, in pairs** (`plots/phase1_agreement_tiers.csv`). The same tiers as
Table 1, counted in pairs. "both" = pairs whose two directions are *both* at that tier; "single" =
pairs with exactly one direction there. Out of 903 pairs.

| n_agree | pairs, **both** directions | pairs, **single** direction | pairs, either | % of 903, either |
|---:|---:|---:|---:|---:|
| 6 | **0** | 17 | 17 | 1.9% |
| 5 | 4 | 106 | 110 | 12.2% |
| 4 | 21 | 247 | 268 | 29.7% |
| 3 | 84 | 372 | 456 | 50.5% |
| 2 | 77 | 408 | 485 | 53.7% |
| 1 | 10 | 229 | 239 | 26.5% |
| 0 | 0 | 35 | 35 | 3.9% |

- **The consensus set is empty: 0 of 903 pairs reach 6/6 in both directions.** The 17 6/6
  direction-instances belong to 17 *different* pairs. This is a real outcome, not a pipeline fault
  (Decision 0 in
  `aux/analysis_with_counter/MMN_pipeline_analysis_decisions_notes_062226_with_counter.md`).
- **The both/single split is the result.** At 6/6 it is 0 against 17; at 5/6, 4 against 106. If the
  models were responding to the *change* between two tones, reversing the pair should preserve the
  response; instead agreement almost never survives reversal.
- The `both` column peaks at 3/6 and 2/6, the tiers where agreement is weak enough that two
  directions landing together carries no information.

**Table 9. Stimuli showing agreement in both directions** — the only 4 pairs reaching n_agree ≥ 5
in *both* directions anywhere in the grid.

| pair | f_low → f_high | semitones | % deviance | mean_uv regular | mean_uv counter | mean of the two |
|---|---|---:|---:|---:|---:|---:|
| `method_1740` | 1600 → 3200 Hz (exact octave) | 12.00 | 100.0 | −3.995 | −1.491 | **−2.743** |
| `method_1744` | 1600 → 4525 Hz | 18.00 | 182.8 | −1.507 | −2.282 | −1.894 |
| `method_1679` | 1234 → 2263 Hz | 10.50 | 83.4 | −0.926 | −2.171 | −1.549 |
| `method_1718` | 1467 → 2263 Hz | 7.50 | 54.3 | −1.452 | −1.248 | −1.350 |

- **These four are the closest thing the search produced to a symmetric deviance response**, and
  they are tightly clustered: every f_low between 1234 and 1600 Hz, every f_high between 2263 and
  4525 Hz, every deviance between 7.5 and 18 semitones.
- That is a *different* region from the 6/6 tier's high-frequency winners, and it sits on the
  1600 Hz row Table 3 flags.
- **Even here the two directions disagree substantially in depth** — `method_1740` is 2.7× deeper
  one way than the other.

**Table 10. Top-50 direction-instances** (`plots/phase1_top50.csv`). Ranked by n_agree desc, then
mean_uv asc. **P2 rank** is the same instance's position in the 254-long Phase-2 ranking
(Section 6). **mean/median/max/min are across all six models' `trough_uv`, agreeing or not** — a
spread computed only over the agreeing models would narrow itself by construction as n_agree
falls. `mean_uv` (the ranking key) remains the agreeing-models-only mean, so at 6/6 the two
coincide.

| rank | method | P2 rank | stimulus | n_agree | mean_uv | mean all6 | median all6 | max all6 | min all6 | did not agree |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `method_1767` | 94 | 1745 → 7611 Hz | 6 | −3.183 | −3.183 | −2.434 | −0.996 | −7.457 | — |
| 2 | `method_1783` | 29 | 1903 → 7611 Hz | 6 | −2.514 | −2.514 | −2.369 | −1.582 | −3.724 | — |
| 3 | `method_1750` | 2 | 1600 → 7611 Hz | 6 | −2.273 | −2.273 | −2.478 | −0.797 | −2.886 | — |
| 4 | `method_1882` | 5 | 4150 → 7611 Hz | 6 | −2.223 | −2.223 | −1.973 | −1.280 | −3.671 | — |
| 5 | `method_1812` | 1 | 2263 → 7611 Hz | 6 | −2.137 | −2.137 | −1.917 | −1.443 | −3.324 | — |
| 6 | `method_1627` | 3 | 951 → 7611 Hz | 6 | −2.053 | −2.053 | −1.950 | −1.062 | −3.034 | — |
| 7 | `method_1603` | 4 | 872 → 7611 Hz | 6 | −2.027 | −2.027 | −1.864 | −1.084 | −3.248 | — |
| 8 | `method_1066_counter` | 8 | 1745 → 218 Hz | 6 | −1.987 | −1.987 | −1.399 | −0.904 | −3.658 | — |
| 9 | `method_1807` | 42 | 2263 → 4935 Hz | 6 | −1.869 | −1.869 | −1.216 | −0.895 | −4.855 | — |
| 10 | `method_1881` | 31 | 4150 → 6979 Hz | 6 | −1.822 | −1.822 | −1.188 | −0.764 | −4.926 | — |
| 11 | `method_1766` | 7 | 1745 → 6979 Hz | 6 | −1.697 | −1.697 | −1.502 | −0.914 | −3.300 | — |
| 12 | `method_1525` | 6 | 673 → 7611 Hz | 6 | −1.644 | −1.644 | −1.364 | −1.049 | −3.020 | — |
| 13 | `method_1024_counter` | 51 | 1600 → 200 Hz | 6 | −1.607 | −1.607 | −1.445 | −0.870 | −2.815 | — |
| 14 | `method_1809` | 9 | 2263 → 5869 Hz | 6 | −1.389 | −1.389 | −1.287 | −0.847 | −2.351 | — |
| 15 | `method_1318_counter` | 87 | 951 → 400 Hz | 6 | −1.358 | −1.358 | −1.269 | −1.014 | −1.717 | — |
| 16 | `method_1042` | 74 | 200 → 7611 Hz | 6 | −1.322 | −1.322 | −1.213 | −0.765 | −2.250 | — |
| 17 | `method_1106_counter` | 88 | 1745 → 238 Hz | 6 | −1.201 | −1.201 | −1.123 | −1.017 | −1.635 | — |
| 18 | `method_1740` | 10 | 1600 → 3200 Hz | 5 | −3.995 | −3.438 | −1.973 | −0.655 | −10.813 | whisper-tiny |
| 19 | `method_1515` | 15 | 673 → 3200 Hz | 5 | −3.201 | −2.678 | −1.574 | −0.066 | −9.677 | whisper-base |
| 20 | `method_1762` | 12 | 1745 → 4935 Hz | 5 | −2.966 | −2.579 | −1.794 | −0.641 | −8.303 | whisper-small |
| 21 | `method_1765` | 11 | 1745 → 6400 Hz | 5 | −2.956 | −2.553 | −1.831 | −0.536 | −7.880 | whisper-small |
| 22 | `method_1730` | 25 | 1467 → 6400 Hz | 5 | −2.693 | −2.157 | −1.564 | +0.525 | −6.325 | whisper-tiny |
| 23 | `method_1032` | 49 | 200 → 3200 Hz | 5 | −2.626 | −2.137 | −0.943 | +0.304 | −7.844 | whisper-small |
| 24 | `method_1601` | 96 | 872 → 6400 Hz | 5 | −2.601 | −2.240 | −1.206 | −0.434 | −8.233 | whisper-tiny |
| 25 | `method_1737` | 13 | 1600 → 2468 Hz | 5 | −2.496 | −2.147 | −1.634 | −0.407 | −6.023 | whisper-tiny |
| 26 | `method_1701` | 20 | 1345 → 2691 Hz | 5 | −2.487 | −2.067 | −1.920 | +0.035 | −4.336 | whisper-tiny |
| 27 | `method_1710` | 16 | 1345 → 5869 Hz | 5 | −2.458 | −2.055 | −1.809 | −0.041 | −5.454 | whisper-tiny |
| 28 | `method_1722` | 14 | 1467 → 3200 Hz | 5 | −2.415 | −2.053 | −1.721 | −0.241 | −4.399 | whisper-tiny |
| 29 | `method_1835` | 115 | 2691 → 6400 Hz | 5 | −2.321 | −1.943 | −0.905 | −0.054 | −5.283 | whisper-tiny |
| 30 | `method_1825` | 21 | 2468 → 7611 Hz | 5 | −2.304 | −1.987 | −1.536 | −0.401 | −4.625 | wav2vec2-medium |
| 31 | `method_1744_counter` | 100 | 4525 → 1600 Hz | 5 | −2.282 | −1.773 | −1.062 | +0.771 | −5.384 | whisper-base |
| 32 | `method_1747` | 23 | 1600 → 5869 Hz | 5 | −2.248 | −1.984 | −1.710 | −0.667 | −4.287 | whisper-tiny |
| 33 | `method_1748` | 97 | 1600 → 6400 Hz | 5 | −2.247 | −1.937 | −1.530 | −0.389 | −5.077 | whisper-tiny |
| 34 | `method_1745` | 17 | 1600 → 4935 Hz | 5 | −2.238 | −1.961 | −1.821 | −0.577 | −4.401 | whisper-tiny |
| 35 | `method_1749` | 101 | 1600 → 6979 Hz | 5 | −2.184 | −1.912 | −1.769 | −0.551 | −3.506 | whisper-tiny |
| 36 | `method_1679_counter` | 27 | 2263 → 1234 Hz | 5 | −2.171 | −1.851 | −1.890 | −0.250 | −3.499 | whisper-base |
| 37 | `method_1848` | 106 | 2934 → 7611 Hz | 5 | −2.157 | −1.900 | −2.011 | −0.620 | −2.686 | whisper-medium |
| 38 | `method_1713` | 93 | 1345 → 7611 Hz | 5 | −2.154 | −1.875 | −2.294 | −0.480 | −2.960 | whisper-tiny |
| 39 | `method_1727` | 32 | 1467 → 4935 Hz | 5 | −2.148 | −1.831 | −1.736 | −0.247 | −3.214 | whisper-tiny |
| 40 | `method_1578` | 24 | 800 → 7611 Hz | 5 | −2.141 | −1.908 | −1.956 | −0.745 | −3.133 | whisper-tiny |
| 41 | `method_1676_counter` | 18 | 1745 → 1234 Hz | 5 | −2.134 | −1.788 | −1.554 | −0.056 | −4.389 | whisper-small |
| 42 | `method_1650` | 30 | 1037 → 7611 Hz | 5 | −2.127 | −1.839 | −1.999 | −0.395 | −3.218 | whisper-tiny |
| 43 | `method_1739` | 28 | 1600 → 2934 Hz | 5 | −2.109 | −1.733 | −1.429 | +0.149 | −4.242 | whisper-tiny |
| 44 | `method_1266_counter` | 39 | 4150 → 336 Hz | 5 | −2.066 | −1.596 | −1.469 | +0.755 | −3.455 | whisper-medium |
| 45 | `method_1798` | 33 | 2075 → 7611 Hz | 5 | −2.053 | −2.322 | −2.240 | −1.022 | −3.668 | wav2vec2-large |
| 46 | `method_1324_counter` | 102 | 1600 → 400 Hz | 5 | −2.028 | −1.550 | −1.767 | +0.841 | −2.688 | whisper-medium |
| 47 | `method_1703` | 34 | 1345 → 3200 Hz | 5 | −2.027 | −1.724 | −1.573 | −0.212 | −3.362 | whisper-tiny |
| 48 | `method_1887` | 36 | 4525 → 6979 Hz | 5 | −2.024 | −1.759 | −0.983 | −0.438 | −5.320 | whisper-base |
| 49 | `method_1778` | 146 | 1903 → 4935 Hz | 5 | −2.000 | −1.755 | −1.494 | −0.528 | −3.672 | whisper-small |
| 50 | `method_1720` | 40 | 1467 → 2691 Hz | 5 | −1.973 | −1.616 | −1.135 | +0.171 | −4.381 | whisper-tiny |

- **whisper-tiny is the model that most often breaks a 6/6.** It is the sole dissenter in 21 of the
  33 five-model rows above, and over the whole grid it accounts for **46 of the 114** rows at 5/6
  (`plots/phase1_model_dissent.csv`). At 0.423 its marginal S7 rate is mid-pack, so this is not
  simply a stricter model — it is a model that disagrees on *these particular* pairs, which are
  exactly the ones the other five rank highest.
- **The P2 rank column moves most at the very top.** Rank 1 `method_1767` returns at Phase-2 rank
  94; rank 2 `method_1783` at 29. Below about rank 10 most instances stay within a few dozen
  places. Section 7 quantifies this.
- **The 6/6 tier is a high-frequency tier**: 9 of its 17 members contain 7611 Hz, and 13 of 17 are
  ascending.

![Horizontal boxplot with one row per model over the 131 direction-instances at n_agree at least 5. Each model contributes one continuous distribution of trough depth in microvolts at FCz across all 131 instances, whether or not that model agreed, with individual points jittered over each box and the count that cleared S7 annotated in the row label. The X = -0.75 microvolt floor is a dashed vertical line. The four whisper models sit in a narrow band with medians between -0.91 and -1.78 microvolts, wav2vec2 medium at -1.71, and wav2vec2 large spreads much wider with a median of -2.26 and a tail past -10 microvolts, which reflects differing amplitude shrinkage across models rather than differing MMN strength. whisper tiny's distribution straddles the floor most, matching its 85 of 131 S7 count.](plots/phase1_model_uv_box.png)

**Figure 7. Trough depth per model over the n_agree ≥ 5 set.** Every direction-instance
contributes its value for that model, agreeing or not; the row label gives how many cleared S7.

- **whisper-tiny's distribution straddles the floor.** It clears S7 on 85 of 131 against
  wav2vec2-medium's 127 of 131, and its median sits at −0.91 µV against a −0.75 µV floor — it is a
  model applying the same criterion to a smaller predicted amplitude, not one seeing nothing.
- **Read each row against its own −0.75 µV floor, never against another row** (Caveat 2).
  wav2vec2-large's wider spread is its mapping's amplitude scale.
- What *is* comparable within a row is how far each distribution clears its own floor, and for
  every model the mass sits close to it.

**Table 11. Direction asymmetry across the whole grid** (`plots/phase1_direction_asymmetry.csv`).

| | value |
|---|---|
| pairs whose two directions land in the same n_agree tier | **196 of 903 (21.7%)** |
| mean \|n_agree(regular) − n_agree(counter)\| | **1.39 tiers** |
| pairs with a direction at ≥5 | 127 |
| …of which reach ≥5 both ways | **4 (3.1%)** |
| 6/6 instances that are `regular` (f_low → f_high, ascending) | 13 of 17 |
| mean n_agree, regular vs counter | 2.856 vs 2.483 |

- **Asymmetry is the rule, not the exception.** Four fifths of pairs put their two directions in
  different tiers, and the average gap is 1.4 tiers out of 6.
- **There is a systematic direction to it**: ascending pairs average 0.37 more agreeing models than
  descending ones, and 13 of the 17 6/6 instances are ascending.
- A deviance detector should be indifferent to the sign of the change. A system that responds more
  to a high tone than a low one would produce exactly this.

### Section 4 summary
- **No: the consensus set is empty.** Zero pairs of 903 reach 6/6 in both directions; four reach ≥5
  both ways, all in a narrow 1234–1600 → 2263–4525 Hz region.
- **The direction asymmetry is systematic, not noisy**: 78% of pairs split their tiers, mean gap
  1.39, and ascending pairs beat descending ones by 0.37 models on average.
- **whisper-tiny decides most of the near-misses** — 46 of the 114 grid-wide 5/6 rows — and its
  distribution straddles the µV floor rather than sitting far from it.
- **No pair from this grid should be carried forward as a validated deviance stimulus on this
  evidence.**

---

## Section 5 — Where does the ranking stop selecting, and where do the survivors sit?

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py` (`plot_ranking_structure`,
> `table_yield`)
> **Data:** `plots/phase1_cutoff_curve.csv`, `plots/phase1_yield.csv`,
> `plots/phase1_direction_asymmetry.csv`
> **Why this section exists.** Sections 2–4 ask whether the grid found anything. This one asks
> what the ranked list *looks like* — how many pairs survive any given cutoff, and whether the
> survivors occupy a particular region of the grid. It is the section to read before selecting any
> subset for further work.

![Line chart of pairs qualifying against the mean_uv cutoff on a symlog vertical axis, one curve per n_agree threshold from 1 to 6 on a light-to-dark Blues ramp, with dashed reference lines marking the 127 pairs at n_agree at least 5 and the 17 at 6 of 6. The horizontal axis runs from 0 to 3.5 microvolts. Every curve is flat from 0 out to a cutoff of about 1.25 microvolts and then falls steeply, the 6 of 6 curve reaching zero by about 3.25 microvolts and the others by 3.5.](plots/phase1_yield_curve.png)

**Figure 8. Pairs qualifying at each agreement and µV cutoff, all 903 pairs.**

- **The µV cutoff barely matters.** Every curve is flat from 0 out to about 1.25 µV, so any cutoff
  in the plausible range returns essentially the tier's full membership.
- **Nothing in the ranking itself tells you where to stop** — a cutoff has to come from the tier
  definition or from outside the data.

![Scatter of f_low against f_high on logarithmic axes labelled at 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000 and 8000 Hz. All 903 grid pairs are drawn in light grey forming a dense triangular lattice above the diagonal, the 127 pairs with a direction at n_agree at least 5 are overlaid in blue, and the 24 literature pairs are drawn as larger orange diamonds. The blue points cover most of the triangle, shifted toward the upper left. The orange diamonds crowd into a single small cluster around f_low 600 to 1000 Hz and f_high 700 to 2000 Hz, occupying a corner the novel grid extends far beyond in both directions.](plots/phase1_grid_position.png)

**Figure 9. Where the strong pairs sit in the frequency grid**, with the literature pairs overlaid.

- **The survivors are spread, not clustered.** The 127 pairs with a direction at ≥5 cover most of
  the grid, shifted toward the top-left — median deviance 21.0 st against 19.5 grid-wide, median
  f_high 4150 Hz against 2691 Hz.
- **The literature pairs occupy one corner.** All 24 sit inside 600–2000 Hz on both axes, a region
  the novel grid extends well past in both directions — which is why Table 7's ranges do not
  overlap.
- **There is no sub-region a narrower, finer follow-up grid could target.** The concentration that
  does exist is on individual frequencies (the 1600–1745 Hz rows, the 7611 Hz column), which
  Section 2 identifies as the artifact rather than the signal.

![Seven by seven matrix of pair counts, regular-direction n_agree on the vertical axis against counter-direction n_agree on the horizontal, cells shaded on a Blues ramp with the count printed in each. The mass concentrates off the diagonal around the (3,2) and (2,3) cells, which hold 104 and 77 pairs, the (3,3) cell holds 84, and the corner where both directions reach 6 is empty. Only four pairs sit in the region where both directions reach 5 or more.](plots/phase1_direction_matrix.png)

**Figure 10. Each pair's two directions, cross-tabulated, all 903 pairs.**

- **The mass sits off the diagonal**, clustered around (2,3) and (3,2).
- **The (6,6) cell is empty and only 4 pairs reach ≥5 both ways.** This is Table 11 in picture
  form, and the clearest single statement of why the consensus set is empty.

**Table 12. Yield at each agreement threshold** (`plots/phase1_yield.csv`). Direction-instances at
or above the tier, and the distinct pairs they belong to. **Shares are of the 1,806
direction-instances in the grid** — a share of the 903 pairs would compare an instance count to a
pair count and read as roughly twice the yield it is.

| tier | direction-instances | % of 1,806 | distinct pairs |
|---|---:|---:|---:|
| ≥ 6/6 | 17 | **0.9%** | 17 |
| ≥ 5/6 | 131 | **7.3%** | 127 |
| ≥ 4/6 | 420 | 23.3% | 387 |
| ≥ 3/6 | 960 | 53.2% | 720 |
| ≥ 2/6 | 1522 | 84.3% | 886 |
| ≥ 1/6 | 1771 | 98.1% | 903 |

- **The 6/6 and 5/6 tiers do not overlap in pairs** — no pair has one direction at 6 and the other
  at 5 — so 17 + 110 = 127 exactly, with no double-counting.
- **The tiers grow very fast below 5/6**: dropping from ≥5 to ≥4 triples the pair count, and by ≥2
  the criterion admits 84% of all direction-instances and has stopped selecting anything.

![Heatmap with the number of models agreeing on the vertical axis, 1 at the bottom to 6 at the top, and the top-X rank threshold on the horizontal axis running 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300. Each cell is annotated with the number of direction-instances that sit in the top X of at least that many models, on a yellow-green-blue sequential ramp. The bottom row rises from 59 at top-10 to 1164 at top-300 and is the only strongly coloured band. The row for 2 models runs 1, 5, 11, 18, 27 across the first five columns and reaches 459 at top-300. The rows for 3 or more models stay near zero across most of the ladder: 3 models first reaches 2 at top-30, 4 models reaches 1 at top-40, 5 models reaches 1 at top-125, and the row for all 6 models is zero everywhere until a single stimulus appears at top-300.](plots/phase1_consensus_heatmap.png)

**Figure 11. Consensus yield: direction-instances in the top X of at least Y models.** Each model
ranks the instances by its own `trough_uv`, most negative first, among the instances where that
model satisfies S2 — so this figure uses no µV threshold and no S7 verdict at all, only rank
order. Cumulative in Y, so every column is non-increasing upward.

- **Exactly one stimulus of the 1,806 is in the top 300 of all six models**, and none is at any
  cut below that. The 6/6 row is zero for eleven of the twelve columns.
- **Consensus has to be bought with a very loose cut.** Three models first share a stimulus at
  top-30, four at top-40, five at top-125 — and top-125 is already more instances per model than
  the entire n_agree ≥ 5 tier of Table 1 contains.
- **Even pairwise overlap is scarce at the sharp end**: at top-10, 59 distinct instances fill the
  60 available slots, so just one stimulus is picked by two models and none by three.
- **This corroborates Sections 2 and 4 by an independent route.** Those rest on the S7 threshold
  and the S2 shape criterion; this uses neither, and reaches the same place.

### Section 5 summary
- **The tiers are the only natural cut points.** 17 pairs at 6/6, 127 with a direction at ≥5, then
  387 at ≥4 — after which the criterion stops selecting.
- **The µV cutoff is not a lever**: every yield curve is flat from 0 to 1.25 µV.
- **The survivors are spread across the grid** and the literature pairs sit in one corner of it, so
  there is no obvious sub-region for a finer follow-up.
- **On rank order alone, the models barely overlap at all.** Exactly one instance of the 1,806 is
  in the top 300 of all six models, and none at any tighter cut (Figure 11) — a result that uses
  neither the µV floor nor the S7 verdict.

---

## Section 6 — Do the Phase-1 findings hold on 15 deviants?

> **Code:** `aux/analysis_novel_search/plots/phase1_results.py --phase 2 --skip_deviance`,
> `aux/analysis_novel_search/plots/novel_search_plots.py`, `scripts/rank_novel_phase2.py`,
> `scripts/novel_search_common.py`. The script keeps its Phase-1 name because Sections 2–5 cite it
> by that path; `--phase 2` reads the Phase-2 slice and writes every output under a `phase2_`
> prefix, so neither phase clobbers the other.
> **Data:** `outputs/results_novel_search/phase2_mmn_s7_roi.csv` (74,676 rows),
> `phase2_final_ranking.csv` (254), `consensus_set.csv` (**empty**), `phase2_selected_pairs.csv`
> (127). Tables: `plots/phase2_agreement_tiers.csv`, `phase2_frequency_involvement.csv`,
> `phase2_top50.csv`, `phase2_model_uv_box.csv`, `phase2_model_dissent.csv`,
> `phase2_direction_asymmetry.csv`, `phase2_yield.csv`, `phase2_top10_vs_literature.csv`.
> **What changed.** Only the deviant count: `deviant_mean` averages **15** realizations
> (`--trial_levels 3,5,7 --num_variations 5`) where Phase 1 used **1**. Same models, same frozen
> mTRF mapping, same S7@0.75 at FCz, same ranking code.
> **What was selected.** Every pair with at least one direction at `n_agree ≥ 5` on the Phase-1
> ranking — 127 of 903, via `select_pairs_by_agreement` with `MIN_AGREE_PHASE2 = 5`. **Both**
> directions of each went forward, including **123 reversals that did not clear the bar**, because
> a direction asymmetry is the artifact the counterbalanced design exists to measure.
> 127 × 2 = **254 direction-instances**.
> **Waveform provenance.** The Phase-2 in-silico run wrote to its own predictions root,
> `outputs/insilico_mmn_predictions_novel_phase2` (602 MB, 254 method groups per model, all
> carrying `n_deviants = 15`), kept separate from the Phase-1 root so neither overwrites the
> other. Figures 14 and 15 are drawn from it. The loader checks each file's `n_deviants`
> attribute against the phase being rendered and refuses rather than pass a 1-deviant trace off
> as Phase 2, so these figures cannot silently be the Phase-1 ones.
> **What has no Phase-2 counterpart.** Table 2 (deviance octiles) is not reproduced: the selection
> removed the small-deviance octiles the correlation lived in, so a Phase-2 version would measure
> the selection rather than the stimuli (Caveat 6).

**Table 13. Phase-2 agreement tiers, in direction-instances and in pairs**
(`plots/phase2_agreement_tiers.csv`; combines Tables 1 and 8 for the 254 instances / 127 pairs).

| n_agree | direction-instances | % of 254 | pairs, **both** directions | pairs, **single** direction |
|---:|---:|---:|---:|---:|
| 6 | 9 | 3.5% | **0** | 9 |
| 5 | 82 | 32.3% | **0** | 82 |
| 4 | 39 | 15.4% | 1 | 37 |
| 3 | 33 | 13.0% | 0 | 33 |
| 2 | 51 | 20.1% | 0 | 51 |
| 1 | 35 | 13.8% | 0 | 35 |
| 0 | 5 | 2.0% | 0 | 5 |

- **The consensus set is still empty: 0 of 127 pairs reach 6/6 in both directions on 15 deviants.**
  `outputs/results_novel_search/consensus_set.csv` is a header and nothing else.
- **`pairs_both` is now zero at 5/6 as well**, where Phase 1 had 4 (Table 9). The single non-zero
  entry anywhere in the column is one pair at 4/6 both ways — **126 of 127 pairs are carried by a
  single direction**.
- **The percentages are not comparable to Table 1's.** Table 1 is over 1,806 unselected instances;
  this is over 254 chosen for scoring highly. The comparable quantity is Section 7.

![Heatmap of cross-model agreement over the 43 by 43 frequency grid restricted to the 254 direction-instances Phase 2 evaluated, on a single-hue light-to-dark Blues ramp from 0 to 6 agreeing models. Only 254 of the 1806 off-diagonal cells carry colour; the rest are a pale unmeasured background and the excluded diagonal is a distinct mid grey. The filled cells do not form bands parallel to the diagonal. Instead the right-hand region where the deviant is 5869 Hz or above is filled with the darkest steps while the matching top rows, where those same frequencies are the standard, are near-white, and the reverse holds around 1600 to 1745 Hz.](plots/novel_n_agree_heatmap_phase2.png)

**Figure 12. Cross-model agreement over the grid, Phase 2** (mirrors Figure 2). 254 of 1,806
off-diagonal cells, 14% coverage; unmeasured cells are a pale background distinct from the
mid-grey excluded diagonal.

- **Asymmetry about the diagonal survives partial coverage.** (i,j) and (j,i) are either both
  present or both absent, since both directions of every selected pair were carried.
- **The dark cells are a high-deviant-frequency region, not a band parallel to the diagonal.**

![Heatmap of the mean trough depth across all six models over the same restricted 43 by 43 grid, on a diverging blue-white-red ramp whose neutral midpoint is exactly zero microvolts, with blue for negative values, an MMN-like trough, and red for positive. Values run from -3.04 to +0.45 microvolts against a grand mean of -0.937. The right-hand column at deviant = 7611 Hz is filled top to bottom in dark blue while the top row at standard = 7611 Hz is near-white and in places red; the 1600 and 1745 Hz rows are dark blue across their deviants while the matching columns are pale.](plots/phase2_mean_uv_heatmap.png)

**Figure 13. Mean trough depth over the grid, Phase 2** (mirrors Figure 3). **Blue = negative = an
MMN-like trough, red = positive = no trough**, neutral midpoint at exactly 0 µV.

- **The two darkest marginals are the two Phase 1 found**: standard = 1745 Hz at −1.722 µV over 13
  deviants, deviant = 7611 Hz at −1.846 µV over 20 standards.
- **The row/column marginals are NOT comparable to Figure 3's.** A Phase-1 stripe averaged 42
  cells; here a row holds between 1 and 20. `plots/phase2_frequency_stripes.csv` carries the
  denominator behind every value; Table 14 is the presentation to quote.

**Table 14. What a frequency's role is worth, Phase 2**
(`plots/phase2_frequency_involvement.csv`; mirrors Table 3). Each frequency appears the same
number of times in each role, so the two halves of a row are comparable.

| Hz | n each role | mean n_agree **as standard** | mean n_agree **as deviant** | mean_uv_all6 as standard | mean_uv_all6 as deviant |
|---:|---:|---:|---:|---:|---:|
| 7611 | 20 | 2.35 | **5.10** | −0.341 | **−1.846** |
| 6979 | 12 | 1.17 | **4.67** | −0.006 | −1.456 |
| 6400 | 8 | 1.25 | **4.38** | −0.147 | −1.654 |
| 4150 | 14 | 4.36 | 2.14 | −1.283 | −0.606 |
| 2263 | 10 | 4.90 | 3.20 | −1.432 | −0.928 |
| 1745 | 13 | **4.92** | 1.54 | **−1.722** | −0.223 |
| 1600 | 17 | 4.71 | 1.82 | −1.713 | −0.339 |

- **The frequency preference reproduces and sharpens.** 7611 Hz: 5.10 of 6 as the deviant against
  2.35 as the standard, where Phase 1 gave 4.55 against 2.41. 1745 Hz: 4.92 as the standard against
  1.54 as the deviant, where Phase 1 gave 3.81 against 2.05.
- **Fifteen deviants push the two roles further apart, not closer** — the effect is not an artifact
  of a single noisy draw.

![Small-multiple grid of 30 panels, one per pair in the Phase-2 top 30. Each panel plots the FCz microvolt difference wave against time from -120 to 460 ms for the regular direction only, with the six models overlaid in Okabe-Ito colours and wav2vec2-large re-hued violet. The 100-240 ms scoring window is shaded and each panel autoscales. Compared with the Phase-1 version the per-model traces are visibly smoother, the 15-deviant mean having removed much of the sample-to-sample jitter, though whisper medium in orange still swings to large positive excursions just after the window and wav2vec2 large in violet still carries the widest range.](plots/phase2_strong_waveforms_1.png)

**Figure 14. FCz µV difference waves for the Phase-2 top 30 pairs**, regular direction only, six
per-model traces (mirrors Figure 5).

- **The traces are visibly smoother than Phase 1's.** Averaging 15 deviants instead of 1 removes
  most of the sample-to-sample jitter, which is why the µV correlation between phases is +0.957
  (Table 22) while the tier assignments move more.
- **Each panel still autoscales** — the models' µV scales differ ~5×, so read shape, never one
  trace's depth against another's.
- The de-overlaid per-model versions are `phase2_strong_waveforms_1__<model>.png`, which share a
  y-axis within a model and annotate each panel with that model's own n_agree for both directions.

![The same 30 Phase-2 panels with the six model traces collapsed to one line per direction: regular in blue solid, counter in orange dashed, on a shared vertical axis in z units fixed from -2 to 2, with no shaded band. The mirror-image pattern is stark and near-universal: in essentially every panel the blue regular line dips below zero inside the shaded 100 to 240 ms window while the orange counter line rises above zero across the same window by a comparable amount, and the two cross near the window's start. Very few panels show both lines dipping together.](plots/phase2_direction_waveforms_1.png)

**Figure 15. The Phase-2 top 30 collapsed to one line per direction**, in z units (mirrors
Figure 6). The raw-µV variant is `phase2_direction_waveforms_uv_1.png`.

- **This is the sharpest single statement of the frequency-preference result in the memo.** Across
  the top 30 pairs the two directions' in-window z traces correlate at **r = −0.48 on average**
  (median −0.61, negative for 26 of 30), and **not one of the 30 pairs has both directions
  mean-negative inside the scoring window** — all 30 have opposite signs.
- **Fifteen deviants made this cleaner, not weaker.** The same measurement on Phase 1's top 30
  gives mean r = −0.42 with 1 of 30 pairs dipping both ways and 28 of 30 with opposite signs. The
  anti-symmetry was not an artifact of the single draw; averaging removed the noise that partly
  masked it.
- **A deviance response would dip both ways**: reversing which tone is standard and which is
  deviant should not flip the sign of the response. An anti-symmetric pair of traces is what a
  response driven by *which tone arrives last* looks like.

**Table 15. Phase-2 top-50 direction-instances** (`plots/phase2_top50.csv`; mirrors Table 10).
**P1 rank** is the same instance's position in the 1,806-long Phase-1 ranking.

| rank | method | P1 rank | stimulus | n_agree | mean_uv | mean all6 | median all6 | max all6 | min all6 | did not agree |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `method_1812` | 5 | 2263 → 7611 Hz | 6 | −2.245 | −2.245 | −2.042 | −1.318 | −3.327 | — |
| 2 | `method_1750` | 3 | 1600 → 7611 Hz | 6 | −2.173 | −2.173 | −2.477 | −0.763 | −2.646 | — |
| 3 | `method_1627` | 6 | 951 → 7611 Hz | 6 | −2.137 | −2.137 | −2.118 | −1.020 | −2.918 | — |
| 4 | `method_1603` | 7 | 872 → 7611 Hz | 6 | −2.074 | −2.074 | −2.213 | −1.056 | −2.892 | — |
| 5 | `method_1882` | 4 | 4150 → 7611 Hz | 6 | −1.995 | −1.995 | −1.741 | −1.127 | −2.965 | — |
| 6 | `method_1525` | 12 | 673 → 7611 Hz | 6 | −1.601 | −1.601 | −1.268 | −1.056 | −3.054 | — |
| 7 | `method_1766` | 11 | 1745 → 6979 Hz | 6 | −1.563 | −1.563 | −1.265 | −0.908 | −3.308 | — |
| 8 | `method_1066_counter` | 8 | 1745 → 218 Hz | 6 | −1.422 | −1.422 | −1.129 | −0.784 | −2.920 | — |
| 9 | `method_1809` | 14 | 2263 → 5869 Hz | 6 | −1.329 | −1.329 | −1.213 | −0.779 | −2.293 | — |
| 10 | `method_1740` | 18 | 1600 → 3200 Hz | 5 | −3.143 | −2.729 | −1.808 | −0.661 | −8.053 | whisper-tiny |
| 11 | `method_1765` | 21 | 1745 → 6400 Hz | 5 | −2.911 | −2.516 | −1.807 | −0.545 | −7.834 | whisper-small |
| 12 | `method_1762` | 20 | 1745 → 4935 Hz | 5 | −2.815 | −2.455 | −1.768 | −0.654 | −7.006 | whisper-small |
| 13 | `method_1737` | 25 | 1600 → 2468 Hz | 5 | −2.566 | −2.206 | −1.684 | −0.407 | −6.187 | whisper-tiny |
| 14 | `method_1722` | 28 | 1467 → 3200 Hz | 5 | −2.469 | −2.128 | −1.751 | −0.423 | −4.672 | whisper-tiny |
| 15 | `method_1515` | 19 | 673 → 3200 Hz | 5 | −2.422 | −2.032 | −1.439 | −0.086 | −6.347 | whisper-base |
| 16 | `method_1710` | 27 | 1345 → 5869 Hz | 5 | −2.416 | −2.009 | −1.939 | +0.029 | −4.874 | whisper-tiny |
| 17 | `method_1745` | 34 | 1600 → 4935 Hz | 5 | −2.245 | −1.967 | −1.782 | −0.577 | −4.606 | whisper-tiny |
| 18 | `method_1676_counter` | 41 | 1745 → 1234 Hz | 5 | −2.230 | −1.872 | −1.509 | −0.080 | −4.939 | whisper-small |
| 19 | `method_1568` | 84 | 800 → 3200 Hz | 5 | −2.228 | −1.884 | −1.465 | −0.167 | −3.999 | whisper-tiny |
| 20 | `method_1701` | 26 | 1345 → 2691 Hz | 5 | −2.200 | −1.824 | −1.861 | +0.055 | −3.584 | whisper-tiny |
| 21 | `method_1825` | 30 | 2468 → 7611 Hz | 5 | −2.173 | −1.782 | −1.465 | +0.171 | −4.078 | wav2vec2-medium |
| 22 | `method_1738` | 51 | 1600 → 2691 Hz | 5 | −2.172 | −1.856 | −1.775 | −0.274 | −3.708 | whisper-tiny |
| 23 | `method_1747` | 32 | 1600 → 5869 Hz | 5 | −2.158 | −1.866 | −1.753 | −0.405 | −4.159 | whisper-tiny |
| 24 | `method_1578` | 40 | 800 → 7611 Hz | 5 | −2.151 | −1.916 | −1.927 | −0.740 | −3.132 | whisper-tiny |
| 25 | `method_1730` | 22 | 1467 → 6400 Hz | 5 | −2.144 | −1.709 | −1.464 | +0.470 | −3.547 | whisper-tiny |
| 26 | `method_1729` | 78 | 1467 → 5869 Hz | 5 | −2.127 | −1.671 | −1.831 | +0.609 | −3.828 | whisper-tiny |
| 27 | `method_1679_counter` | 36 | 2263 → 1234 Hz | 5 | −2.113 | −1.804 | −1.797 | −0.260 | −3.962 | whisper-base |
| 28 | `method_1739` | 43 | 1600 → 2934 Hz | 5 | −2.107 | −1.714 | −1.500 | +0.255 | −4.378 | whisper-tiny |
| 29 | `method_1783` | 2 | 1903 → 7611 Hz | 5 | −2.106 | −2.016 | −1.939 | −1.522 | −2.892 | wav2vec2-large |
| 30 | `method_1650` | 42 | 1037 → 7611 Hz | 5 | −2.096 | −1.819 | −1.938 | −0.432 | −2.764 | whisper-tiny |
| 31 | `method_1881` | 10 | 4150 → 6979 Hz | 5 | −2.020 | −1.805 | −1.203 | −0.733 | −5.161 | whisper-tiny |
| 32 | `method_1727` | 39 | 1467 → 4935 Hz | 5 | −2.018 | −1.729 | −1.598 | −0.287 | −3.621 | whisper-tiny |
| 33 | `method_1798` | 45 | 2075 → 7611 Hz | 5 | −1.972 | −1.872 | −1.828 | −1.189 | −2.919 | wav2vec2-large |
| 34 | `method_1703` | 47 | 1345 → 3200 Hz | 5 | −1.969 | −1.675 | −1.325 | −0.206 | −3.246 | whisper-tiny |
| 35 | `method_1487` | 89 | 617 → 3200 Hz | 5 | −1.933 | −1.484 | −1.387 | +0.762 | −4.020 | whisper-base |
| 36 | `method_1887` | 48 | 4525 → 6979 Hz | 5 | −1.930 | −1.685 | −0.948 | −0.459 | −4.873 | whisper-base |
| 37 | `method_1144_counter` | 61 | 1600 → 259 Hz | 5 | −1.913 | −1.674 | −1.579 | −0.479 | −3.841 | whisper-tiny |
| 38 | `method_1764` | 55 | 1745 → 5869 Hz | 5 | −1.882 | −1.665 | −1.304 | −0.584 | −3.160 | whisper-small |
| 39 | `method_1266_counter` | 44 | 4150 → 336 Hz | 5 | −1.873 | −1.429 | −1.480 | +0.788 | −2.776 | whisper-medium |
| 40 | `method_1720` | 50 | 1467 → 2691 Hz | 5 | −1.849 | −1.514 | −1.127 | +0.161 | −3.602 | whisper-tiny |
| 41 | `method_1712` | 56 | 1345 → 6979 Hz | 5 | −1.841 | −1.547 | −1.432 | −0.080 | −2.621 | whisper-tiny |
| 42 | `method_1807` | 9 | 2263 → 4935 Hz | 5 | −1.837 | −1.644 | −1.126 | −0.676 | −4.384 | wav2vec2-medium |
| 43 | `method_1746` | 63 | 1600 → 5382 Hz | 5 | −1.829 | −1.483 | −1.526 | +0.248 | −2.993 | whisper-tiny |
| 44 | `method_1755` | 73 | 1745 → 2691 Hz | 5 | −1.826 | −1.611 | −1.339 | −0.536 | −3.188 | whisper-base |
| 45 | `method_1744` | 97 | 1600 → 4525 Hz | 5 | −1.765 | −1.511 | −1.863 | −0.241 | −2.268 | whisper-small |
| 46 | `method_1875` | 71 | 3805 → 7611 Hz | 5 | −1.763 | −1.580 | −1.572 | −0.662 | −2.521 | wav2vec2-large |
| 47 | `method_1708` | 64 | 1345 → 4935 Hz | 5 | −1.762 | −1.410 | −1.702 | +0.353 | −2.389 | whisper-tiny |
| 48 | `method_1699_counter` | 52 | 2263 → 1345 Hz | 5 | −1.760 | −1.454 | −1.546 | +0.072 | −2.453 | whisper-base |
| 49 | `method_1032` | 23 | 200 → 3200 Hz | 5 | −1.748 | −1.409 | −0.941 | +0.286 | −3.510 | whisper-small |
| 50 | `method_1273` | 67 | 336 → 7611 Hz | 5 | −1.731 | −1.532 | −1.356 | −0.540 | −2.763 | whisper-tiny |

- **The top of the list is recognisably the same list.** Six of Phase 1's top nine instances are
  among Phase 2's nine 6/6; nine of Phase 1's seventeen 6/6 instances hold the tier.
- **All five of the Phase-2 top 5 contain 7611 Hz**, six of the nine 6/6 instances do, and eight of
  the nine are ascending.
- **Phase 1's rank 1 is the conspicuous casualty.** `method_1767` does not appear in this table at
  all: it comes back at 4/6 and rank 94 of 254, losing both wav2vec2 models. **Section 4's best
  pair in the grid does not survive its own re-measurement.**

**Table 16. Which model breaks a would-be unanimous row** (`plots/phase2_model_dissent.csv`).
"Sole dissenter" counts rows at exactly 5/6 where that model is the one holdout: 82 such rows in
Phase 2, 114 in Phase 1. All 114 of Phase 1's sit inside the 254 by construction, so the Phase-1
count is both the whole-grid and the same-instances figure.

| model | P2 S7 rate | **P2 sole dissenter** | P1 S7 rate (same 254) | P1 S7 rate (all 1806) | P1 sole dissenter |
|---|---:|---:|---:|---:|---:|
| whisper-tiny | 0.492 | **33 (40.2%)** | 0.516 | 0.422 | **46 (40.4%)** |
| whisper-medium | 0.610 | 17 (20.7%) | 0.673 | 0.531 | 21 (18.4%) |
| whisper-base | 0.480 | 11 (13.4%) | 0.492 | 0.325 | 14 (12.3%) |
| whisper-small | 0.496 | 9 (11.0%) | 0.504 | 0.331 | 16 (14.0%) |
| wav2vec2-large | 0.555 | 9 (11.0%) | 0.669 | 0.461 | 13 (11.4%) |
| wav2vec2-medium | 0.736 | 3 (3.7%) | 0.807 | 0.599 | 4 (3.5%) |

- **whisper-tiny is still the usual dissenter: 40.2% against 40.4% in Phase 1** — unchanged to a
  tenth of a point.
- Its marginal rate (0.492) is mid-pack, so it remains a model that disagrees on *these particular*
  pairs rather than a stricter model overall.
- **Every model's rate falls from Phase 1 to Phase 2 on the same instances.** That is Section 7's
  regression to the mean, not a property of the stimuli.

![Horizontal boxplot with one row per model over the 91 Phase-2 direction-instances at n_agree at least 5. Each model contributes one continuous distribution of trough depth in microvolts at FCz across all 91 instances, whether or not it agreed, with points jittered over each box and the S7 count annotated in the row label. The X = -0.75 microvolt floor is a dashed vertical line. Medians run from -0.89 for whisper tiny to -2.45 for wav2vec2 large, whose distribution again spreads widest, past -8 microvolts.](plots/phase2_model_uv_box.png)

**Figure 16. Trough depth per model over the Phase-2 n_agree ≥ 5 set** (mirrors Figure 7).

- **The picture is Phase 1's.** whisper-tiny again straddles the floor with a median of −0.89 µV
  and 58 of 91 clearing S7; wav2vec2-medium clears 88 of 91.
- Medians move by less than 0.2 µV per model between the phases, which is Section 7's finding in a
  different presentation.

**Table 17. Direction asymmetry across the evaluated set**
(`plots/phase2_direction_asymmetry.csv`; mirrors Table 11). **The 127 pairs were selected on one
direction's score, so this set is enriched for asymmetric pairs by construction** — the Phase-1
column for the same 127 pairs is the fair comparison, not Table 11's grid-wide figures.

| | Phase 2 (127 pairs) | Phase 1, same 127 pairs | Phase 1, all 903 (Table 11) |
|---|---:|---:|---:|
| pairs whose two directions land in the same n_agree tier | **1 (0.8%)** | 4 (3.1%) | 196 (21.7%) |
| mean \|n_agree(regular) − n_agree(counter)\| | **2.74 tiers** | 2.95 | 1.39 |
| pairs with a direction at ≥5 | 91 | 127 | 127 |
| …of which reach ≥5 both ways | **0** | 4 | 4 |
| 6/6 instances that are ascending | **8 of 9** | 13 of 17 | 13 of 17 |
| mean n_agree, regular vs counter | **3.98 vs 2.76** | 4.32 vs 3.01 | 2.86 vs 2.48 |

- **The ascending preference reproduces at almost exactly the same size.** The regular-minus-counter
  gap is 1.23 models in Phase 2 against 1.31 in Phase 1 on the same pairs.
- **A deviance detector should be indifferent to the sign of the change.** A system responding more
  to a high tone than a low one produces exactly this.

![Line chart of pairs qualifying against the mean_uv cutoff on a symlog vertical axis, one curve per n_agree threshold on a light-to-dark Blues ramp, horizontal axis from 0 to 3.5 microvolts, with dashed reference lines at the 91 pairs with a direction at n_agree at least 5 and the 9 at 6 of 6. Every curve is flat from 0 to about 1.25 microvolts before falling steeply toward zero.](plots/phase2_yield_curve.png)

**Figure 17. Pairs qualifying at each agreement and µV cutoff, the 127 evaluated pairs.**

- **The µV cutoff is no more of a lever on 15 deviants than on 1** — flat from 0 to about 1.25 µV,
  exactly as in Figure 8.

![Scatter of f_low against f_high on logarithmic axes with dense frequency labels, all 903 grid pairs in light grey, and the 91 Phase-2 pairs having a direction at n_agree at least 5 in blue. The blue points spread across the upper-left half of the triangle rather than clustering in any one region.](plots/phase2_grid_position.png)

**Figure 18. Where the Phase-2 strong pairs sit in the frequency grid.**

- **Plotted against the full 903-pair grid, not the 127 they were drawn from**, so "spread, not
  clustered" remains a claim about the grid rather than about the subset.
- The 91 survivors keep Phase 1's shift toward larger deviance and higher f_high. Where the
  literature set sits in this same space is settled once, in Figure 9.

![Seven by seven matrix of pair counts, regular-direction n_agree on the vertical axis against counter-direction n_agree on the horizontal, cells shaded on a Blues ramp with counts printed. The mass sits far off the diagonal at regular = 5 against counter = 1, 2 and 3, holding 17, 18 and 13 pairs, and the region where both directions reach 5 or more is completely empty.](plots/phase2_direction_matrix.png)

**Figure 19. Each pair's two directions, cross-tabulated, the 127 evaluated pairs.**

- **The both-directions-strong region is empty.** The mass sits at (regular 5, counter 1–3) —
  17 pairs at (5,1), 18 at (5,2), 13 at (5,3).
- **Only 1 of 127 pairs lands its two directions in the same tier at all**, against 4 for the same
  pairs in Phase 1.

**Table 18. Yield at each agreement threshold, Phase 2** (`plots/phase2_yield.csv`; mirrors
Table 12). **Shares are of the 254 direction-instances Phase 2 evaluated** — the within-phase
yield, matching Table 12's within-grid one. Because the 254 were selected on their Phase-1 score,
these percentages are much higher than Table 12's and are not a coverage statistic.

| tier | direction-instances | % of 254 | distinct pairs |
|---|---:|---:|---:|
| ≥ 6/6 | 9 | **3.5%** | 9 |
| ≥ 5/6 | 91 | **35.8%** | 91 |
| ≥ 4/6 | 130 | 51.2% | 121 |
| ≥ 3/6 | 163 | 64.2% | 127 |
| ≥ 2/6 | 214 | 84.3% | 127 |
| ≥ 1/6 | 249 | 98.0% | 127 |

- **A third of the evaluated set still has a direction at ≥5/6** (35.8%), against 7.3% of the whole
  grid in Phase 1 — which is the selection working, not a change in the stimuli.
- **36 of the 127 pairs the screen selected no longer have a direction at 5** on 15 deviants: the
  ≥5/6 tier holds 91 pairs where Phase 1 gave it 127.
- The bottom two rows match Phase 1's almost exactly (84.3% vs 84.3%, 98.0% vs 98.1%), because by
  ≥2/6 the criterion has stopped selecting in either phase.

![Heatmap with the number of models agreeing on the vertical axis, 1 at the bottom to 6 at the top, and the top-X rank threshold on the horizontal axis running 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250 — tens from 10 to 100, then coarsening; it stops at 250 because only 254 instances were evaluated. Each cell is annotated with the number of Phase-2 direction-instances in the top X of at least that many models, on a yellow-green-blue sequential ramp. The bottom row rises from 30 at top-5 to 254 at top-200. The 2-model row is zero at top-5 and first reaches 5 at top-10; 3 models first reach 3 at top-20; 4 models reach 2 at top-30. The 5-model and 6-model rows are zero across the whole 5-to-50 range and first appear at top-60 and top-70 respectively, each with a single-digit count. The right-hand columns are strongly coloured across every row, but 200 is already 79 percent of the evaluated set.](plots/phase2_consensus_heatmap.png)

**Figure 20. Consensus yield, Phase 2** (same construction as Figure 11). The ladder steps by tens
from 10 to 100 — the range where the models actually begin to overlap — and then coarsens to 250.
Over 254 instances a top-10 cut is already 4% of the set, where over Phase 1's 1,806 it is 0.6%,
so the two figures need different resolutions to show the same region.

- **At top-5 the six models pick 30 different stimuli** — the bottom-left cell is 30, filling all
  30 available slots, so there is no overlap whatsoever between the models' five strongest
  responses.
- **Agreement arrives late and one model at a time.** Two models first share a stimulus at
  top-10, three at top-20, four at top-30, five at top-60 and all six at top-70 — and the
  five- and six-model rows are **empty across the entire 5-to-50 range**.
- **The strong right-hand corner is an artifact of the smaller set.** Top-200 is 79% of the 254
  instances, so "in the top 200 of all six models" is close to "not last" — read the left half of
  this figure, not the right.
- **Column for column at the left, this is Phase 1's picture.** The 15-deviant scores reproduce
  the same rank-based disagreement the single draw showed, on a set already selected for scoring
  highly.

**Table 19. The stimuli behind the 5-model row of Figure 20** — every Phase-2 direction-instance
that reaches at least 5 of 6 models at a top-70, top-80, top-90 or top-100 cut
(`plots/phase2_consensus_members_n5.csv`). **The four sets are nested**, so a tick can only turn
on as the cut widens and never off. "P2 rank" is the instance's place in the Phase-2 ranking
(Table 15); `mean_uv` is the agreeing-models-only mean.

| P2 rank | method | stimulus | direction | n_agree | mean_uv | top-70 | top-80 | top-90 | top-100 |
|---:|---|---|---|---:|---:|:---:|:---:|:---:|:---:|
| 1 | `method_1812` | 2263 → 7611 Hz | regular | 6 | −2.245 | ✓ | ✓ | ✓ | ✓ |
| 2 | `method_1750` | 1600 → 7611 Hz | regular | 6 | −2.173 | ✓ | ✓ | ✓ | ✓ |
| 5 | `method_1882` | 4150 → 7611 Hz | regular | 6 | −1.995 | ✓ | ✓ | ✓ | ✓ |
| 10 | `method_1740` | 1600 → 3200 Hz | regular | 5 | −3.143 | ✓ | ✓ | ✓ | ✓ |
| 14 | `method_1722` | 1467 → 3200 Hz | regular | 5 | −2.469 | ✓ | ✓ | ✓ | ✓ |
| 39 | `method_1266_counter` | 4150 → 336 Hz | counter | 5 | −1.873 | ✓ | ✓ | ✓ | ✓ |
| 29 | `method_1783` | 1903 → 7611 Hz | regular | 5 | −2.106 | ✗ | ✓ | ✓ | ✓ |
| 3 | `method_1627` | 951 → 7611 Hz | regular | 6 | −2.137 | ✗ | ✗ | ✓ | ✓ |
| 4 | `method_1603` | 872 → 7611 Hz | regular | 6 | −2.074 | ✗ | ✗ | ✓ | ✓ |
| 15 | `method_1515` | 673 → 3200 Hz | regular | 5 | −2.422 | ✗ | ✗ | ✓ | ✓ |
| 17 | `method_1745` | 1600 → 4935 Hz | regular | 5 | −2.244 | ✗ | ✗ | ✓ | ✓ |
| 22 | `method_1738` | 1600 → 2691 Hz | regular | 5 | −2.172 | ✗ | ✗ | ✓ | ✓ |
| 27 | `method_1679_counter` | 2263 → 1234 Hz | counter | 5 | −2.113 | ✗ | ✗ | ✓ | ✓ |
| 48 | `method_1699_counter` | 2263 → 1345 Hz | counter | 5 | −1.760 | ✗ | ✗ | ✓ | ✓ |
| 12 | `method_1762` | 1745 → 4935 Hz | regular | 5 | −2.815 | ✗ | ✗ | ✗ | ✓ |
| 26 | `method_1729` | 1467 → 5869 Hz | regular | 5 | −2.127 | ✗ | ✗ | ✗ | ✓ |
| 28 | `method_1739` | 1600 → 2934 Hz | regular | 5 | −2.107 | ✗ | ✗ | ✗ | ✓ |
| 35 | `method_1487` | 617 → 3200 Hz | regular | 5 | −1.933 | ✗ | ✗ | ✗ | ✓ |
| 47 | `method_1708` | 1345 → 4935 Hz | regular | 5 | −1.762 | ✗ | ✗ | ✗ | ✓ |
| 51 | `method_1024_counter` | 1600 → 200 Hz | counter | 5 | −1.706 | ✗ | ✗ | ✗ | ✓ |
| 58 | `method_1613` | 951 → 2263 Hz | regular | 5 | −1.663 | ✗ | ✗ | ✗ | ✓ |
| 61 | `method_1230_counter` | 4150 → 308 Hz | counter | 5 | −1.613 | ✗ | ✗ | ✗ | ✓ |
| 69 | `method_1355_counter` | 1345 → 436 Hz | counter | 5 | −1.508 | ✗ | ✗ | ✗ | ✓ |
| | **count** | | | | | **6** | **7** | **14** | **23** |

- **The strictest cut that yields anything is top-70, and it yields 6 instances** — six pairs, all
  distinct, none of them a pair's two directions.
- **Only 3 of the 23 are 6/6 on the S7 criterion** (`method_1812`, `method_1750`, `method_1882`,
  all against a 7611 Hz deviant); the rest are 5/6. The rank-based cut and the threshold-based one
  select overlapping but not identical sets, which is the point of showing both.
- **Widening from top-70 to top-100 nearly quadruples the yield**, 6 → 23, without adding a single
  6/6 instance beyond the two that arrive at top-90 (`method_1627`, `method_1603`). Past top-70
  the cut is buying quantity, not consensus.
- **Descending pairs appear from top-70 onward** — `method_1266_counter` (4150 → 336 Hz) is in the
  strictest set, and five more counter instances arrive by top-100. On a rank-based criterion the
  ascending preference of Table 17 is present but not absolute.

### Section 6 summary
- **Yes, every structural finding holds.** The consensus set is still empty (0 of 127 both ways),
  `pairs_both` is now zero at 5/6 as well, and 126 of 127 pairs are carried by a single direction.
- **The frequency preference holds and sharpens**: 7611 Hz goes from 4.55-vs-2.41 on one deviant
  to 5.10-vs-2.35 on fifteen; 1745 Hz from 3.81-vs-2.05 to 4.92-vs-1.54.
- **whisper-tiny is still the usual dissenter** (40.2% against 40.4%) and the ascending preference
  is unchanged in size (1.23 models against 1.31).
- **The direction waveforms make the preference starker on 15 deviants than on 1**: none of the
  top 30 pairs dips both ways inside the scoring window, against 1 of 30 in Phase 1, and the two
  directions' in-window traces anti-correlate at mean r = −0.48.
- **What does not hold is any individual pair's exact standing**: Phase 1's best pair,
  `method_1767`, falls to 4/6 and rank 94 of 254, and the ≥5/6 yield halves.
- **The rank-order consensus picture is Phase 1's too** (Figure 20): at top-5 the six models pick
  30 different stimuli, no two overlap at all until top-10, and nothing reaches 6/6 agreement
  until the cut is relaxed to top-70 of 254 — with the five- and six-model rows empty across the
  whole 5-to-50 range.

---

## Section 7 — Do the two phases' responses correlate?

> **Code:** `aux/analysis_novel_search/plots/novel_search_plots.py`
> **Data:** `outputs/results_novel_search/phase2_final_ranking.csv`, which already carries
> `phase1_rank`, `phase1_n_agree`, `phase1_mean_uv` and `rank_shift` merged in — no join needed.
> Tables: `plots/phase2_rank_shift.csv`, `plots/phase2_tier_migration.csv`,
> `plots/phase2_correlation_summary.csv`.
> **Population.** All 254 shared direction-instances and only those. Every number is
> within-instance: the same (pair, direction) scored twice, once from 1 deviant and once from 15.

![Scatter of Phase-2 rank against Phase-1 rank for the 254 shared direction-instances, with a dashed identity line. The cloud hugs the line over the whole range with a scatter of roughly plus or minus 30 ranks and a handful of outliers reaching about 95 ranks. Spearman rho = +0.923 at p = 2.3e-106.](plots/novel_rank_stability.png)

**Figure 21. Phase-1 rank against Phase-2 rank**, Phase-1 ranks re-ranked within the shared subset
so both axes count the same population.

- **ρ = +0.923** (p = 2.3 × 10⁻¹⁰⁶, n = 254). The order is largely preserved.
- The scatter is tightest at both ends and loosest in the middle, which is where the 5/6 tier sits.

**Table 20. Rank stability and the distribution of movement** (`plots/phase2_rank_shift.csv`).
`rank_shift` = Phase-1 rank within the subset − Phase-2 rank; positive = moved up.

| | value |
|---|---|
| shared direction-instances | **254** |
| ρ(phase1_rank, phase2_rank) | **+0.923** (p = 2.3 × 10⁻¹⁰⁶) |
| ρ(phase1_n_agree, phase2_n_agree) | **+0.908** (p = 5.6 × 10⁻⁹⁷) |
| median \|rank shift\| | **14** places of 254 |
| mean \|rank shift\| | **20.8** places of 254 |
| 90th percentile \|rank shift\| | 51 |
| max \|rank shift\| | 97 |
| moved ≤ 10 places | 100 of 254 (39%) |
| moved ≤ 25 places | 176 of 254 (69%) |
| moved ≤ 50 places | 228 of 254 (90%) |
| kept their n_agree tier | **176 (69%)** |

- **90% of instances land within 50 places** of where the screen put them in a 254-long list.
- **69% keep their exact agreement tier.**

![Seven by seven heatmap with Phase-1 n_agree on the vertical axis and Phase-2 n_agree on the horizontal, counts printed in each cell. 69 percent of instances sit on the diagonal, the largest cell being 75 instances that stayed at 5 of 6. Essentially all off-diagonal mass sits to the left of the diagonal, a lower Phase-2 tier than Phase-1 tier.](plots/novel_tier_migration.png)

**Figure 22. n_agree tier, Phase 1 against Phase 2.**

- **The off-diagonal mass is almost entirely to the left of the diagonal** — instances fell rather
  than rose.
- **The Phase-2 6/6 column has a single non-zero cell, on the diagonal**: nothing rose into the top
  tier from below.

**Table 21. Phase-1 × Phase-2 n_agree, all 254 shared direction-instances**
(`plots/phase2_tier_migration.csv`). Rows are Phase 1, columns Phase 2.

| P1 \ P2 | 6 | 5 | 4 | 3 | 2 | 1 | 0 | **n** | **mean P2 n_agree** | held | fell | rose |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **6** | **9** | 7 | 1 | 0 | 0 | 0 | 0 | 17 | 5.47 | 9 | 8 | **0** |
| **5** | 0 | **75** | 30 | 7 | 2 | 0 | 0 | 114 | 4.56 | 75 | 39 | **0** |
| **4** | 0 | 0 | **6** | 2 | 0 | 0 | 0 | 8 | 3.75 | 6 | 2 | **0** |
| **3** | 0 | 0 | 1 | **21** | 11 | 2 | 0 | 35 | 2.60 | 21 | 13 | 1 |
| **2** | 0 | 0 | 1 | 3 | **35** | 7 | 0 | 46 | 1.96 | 35 | 7 | 4 |
| **1** | 0 | 0 | 0 | 0 | 2 | **26** | 1 | 29 | 1.03 | 26 | 1 | 2 |
| **0** | 0 | 0 | 0 | 0 | 1 | 0 | **4** | 5 | 0.40 | 4 | 0 | 1 |
| **total** | 9 | 82 | 39 | 33 | 51 | 35 | 5 | **254** | **3.37** | **176** | **70** | **8** |

- **The high tiers leak downward only.** Rows 6, 5 and 4 have **zero** in `rose`; 49 instances fell
  out of them. All 8 rises come from rows 3 and below — the carried reversals.
- **The 5/6 tier is where the churn is**: 114 entered, 75 stayed, 30 fell to 4.

![Scatter of Phase-2 against Phase-1 mean_uv_all6 in microvolts with a dashed identity line, 254 points coloured by Phase-2 n_agree on a light-to-dark Blues ramp. The cloud sits tightly along the identity line from about -3.5 to +0.5 microvolts with only slight scatter, and the darkest points, the high-agreement instances, sit at the deep negative end. Spearman rho = +0.957.](plots/phase2_uv_scatter.png)

**Figure 23. Predicted trough depth, Phase 1 against Phase 2**, on `mean_uv_all6` — the mean over
all six models, defined for every instance and over the same six models in both phases.

- **ρ = +0.957 on all 254.** The underlying µV response is reproduced almost exactly.
- The high-agreement instances sit at the deep negative end, as they must.

![Ten Spearman correlations between the phases drawn as points with bootstrap 95 percent confidence interval bars, sorted strongest first and coloured by group. Per-model trough_uv for whisper base and whisper small sit at +0.997 and +0.996, whisper tiny at +0.986, mean_uv_all6 at +0.957, whisper medium at +0.955, rank at +0.923, n_agree at +0.908, wav2vec2 medium and large at +0.858 and +0.849, and mean_uv over agreeing models only lowest at +0.768 with the widest interval.](plots/phase2_correlation_summary.png)

**Figure 24. Phase-1 vs Phase-2 agreement, by quantity**, with 2,000-sample percentile bootstrap
95% CIs.

- **The µV quantities sit above the ranking quantities.** ρ(mean_uv_all6) = +0.957 exceeds
  ρ(rank) = +0.923 with non-overlapping CIs.
- **What moves between the phases is mostly the discretization** — instances crossing the −0.75 µV
  S7 threshold, which flips `n_agree` and so the rank — not the underlying predicted response.
- **The two wav2vec2 models are the least reproducible** (+0.858, +0.849), with CIs that do not
  overlap the whisper models'.

**Table 22. Every Phase-1 vs Phase-2 correlation, ranked**
(`plots/phase2_correlation_summary.csv`). n = 254 (248 for `mean_uv`, undefined at n_agree = 0).

| quantity | ρ | 95% CI |
|---|---:|---|
| `trough_uv`, whisper-base | **+0.997** | +0.994 to +0.998 |
| `trough_uv`, whisper-small | **+0.996** | +0.993 to +0.998 |
| `trough_uv`, whisper-tiny | +0.986 | +0.976 to +0.993 |
| **`mean_uv_all6` (all 6 models)** | **+0.957** | +0.941 to +0.968 |
| `trough_uv`, whisper-medium | +0.955 | +0.935 to +0.967 |
| **rank (within the shared subset)** | **+0.923** | +0.894 to +0.943 |
| **n_agree** | **+0.908** | +0.879 to +0.931 |
| `trough_uv`, wav2vec2-medium | +0.858 | +0.809 to +0.897 |
| `trough_uv`, wav2vec2-large | +0.849 | +0.792 to +0.894 |
| `mean_uv` (agreeing models only) | +0.768 | +0.688 to +0.835 |

- **Four of six models reproduce their own µV almost exactly** across a 15-fold change in deviant
  count (ρ ≥ 0.955).
- **Do not read the +0.768 row as "µV is the noisiest quantity."** `mean_uv` averages only the
  agreeing models, so once `n_agree` changes the two phases average different model sets and that
  correlation is partly measuring the change of composition. The comparable µV quantity is
  `mean_uv_all6` at +0.957.

![Bland-Altman plot of the Phase-2 minus Phase-1 microvolt difference against the mean of the two phases, with a solid red mean-bias line at +0.087 microvolts and dashed plus and minus 1.96 SD lines at about +0.51 and -0.34. The cloud is centred slightly above zero with no trend against the mean.](plots/phase2_uv_bland_altman.png)

**Figure 25. Bland-Altman: how the two phases' µV differ in absolute terms.**

- **Phase-2 troughs are +0.087 µV shallower on average** (SD 0.215); 83 of 254 got deeper.
- **No trend against magnitude** — a small uniform shift, not a scale change.

![Histogram of the rank shift between phases over the 254 direction-instances in bins of 10 ranks, sharply peaked just above zero with 73 instances in the 0 to 10 bin and 55 in the 10 to 20 bin against 18 in the -10 to 0 bin, and strongly left-skewed with a long thin tail running out to -100 and a short right tail ending near +65.](plots/phase2_rank_shift.png)

**Figure 26. How far instances moved between the rankings.**

- Rank shift sums to zero by construction, so the shape is the informative part: **168 instances
  rose a little and 80 fell a lot.**
- **The long left tail is a handful of highly-ranked instances collapsing** (`method_1767` at −93 is
  in it); the mass just above zero is everything else drifting up to fill the space.

![Mean Phase-2 n_agree plotted against Phase-1 n_agree tier with standard-error bars, a dashed identity line, and the n annotated at each point. The curve sits clearly below the identity line at Phase-1 tiers 3 through 6, essentially on it at tier 2, and at or above it at tiers 1 and 0.](plots/phase2_tier_regression.png)

**Figure 27. Mean Phase-2 agreement by Phase-1 tier.**

- **The high tiers fall and the low tiers rise** — 6/6 loses 0.53 models on average, 0/6 gains 0.40.
- **This is regression to the mean, guaranteed by the selection rule.** The 254 were chosen for
  scoring high on a single draw containing a sampling component, so their second measurement moves
  toward their true value. A real degradation would push every row down; regression pushes the ends
  inward, which is what the figure shows.
- The 123 carried reversals make the bottom rows visible: they entered on their partner's score,
  not their own, so they have nothing to regress from.

### Section 7 summary — do Phase 1 and Phase 2 correlate reasonably well?

**Yes, and more strongly than the headline rank figure suggests.**

- **The underlying signal is reproduced almost exactly.** ρ(mean_uv_all6) = **+0.957**, and four of
  the six models reproduce their own `trough_uv` at ρ ≥ 0.955. Fifteen deviants and one deviant
  give nearly the same predicted waveform depth.
- **The ranking built on that signal is a little looser but still strong**: ρ(rank) = **+0.923**,
  ρ(n_agree) = **+0.908**, 69% of instances keep their exact tier and 90% move ≤ 50 of 254 places.
  The gap between +0.957 and +0.923 is the cost of discretizing a continuous µV at a threshold.
- **The disagreement that does exist is systematic and explained, not noise.** Mean n_agree fell
  3.661 → 3.370 with 70 instances falling against 8 rising — the signature of regression to the
  mean on a set selected for a high score on a single draw, confirmed by Figure 27's tier-by-tier
  pattern. It is evidence about the selection, not about the stimuli.
- **Two limits on that conclusion.** ρ = +0.923 is measured on a **selected** subset drawn from the
  top of the Phase-1 ranking, so it says the screen ranks reliably *among pairs it already ranked
  highly*; it does not establish that the 1,552 rejected instances were ordered correctly. And 8 of
  254 instances rose a tier on re-measurement, all from tiers 3 and below, so a pair that would
  reach 6/6 both ways on 15 deviants but scored 4 or below on its one Phase-1 draw is invisible by
  construction (Caveats 6 and 7).
- **Practical read: a one-deviant, 4-clip screen is a sound way to choose which pairs to evaluate
  properly, and an unsound way to decide a pair's final tier.** Every conclusion in this memo that
  depends on a tier boundary — the empty consensus set above all — is quoted from Phase 2.

---

## Reproducing this memo

Full run guide: `aux/sophies_repository_overview.md` §17.5 (Stages A–J). After the cluster stages
and the rsync back:

```bash
# Section 0 — the literature screen on its own. Reads the same scored CSV Section 3 compares
# against and the soafix predictions; touches no novel-grid output, so it can run standalone.
python aux/analysis_novel_search/plots/literature_results.py

# Phase 1 — Sections 1–5
python scripts/analyze_mmn_s7_roi.py --predictions_root outputs/insilico_mmn_predictions_novel \
    --dip_uv_threshold 0.75 --out outputs/results_novel_search/phase1_mmn_s7_roi.csv
python scripts/rank_novel_phase1.py                            # the ranking CSVs
python aux/analysis_novel_search/plots/novel_search_plots.py   # Section 2 heatmap + scaling
python aux/analysis_novel_search/plots/phase1_results.py \
    --literature_csv outputs/results_soafix/mmn_s7_roi.csv     # Sections 2–5
python aux/analysis_novel_search/plots/phase1_results.py --only_waveforms --wave_units uv

# Phase 2 — Sections 6–7. Same scorer, same ranker; only the deviant count differs.
python scripts/analyze_mmn_s7_roi.py --predictions_root outputs/insilico_mmn_predictions_novel \
    --dip_uv_threshold 0.75 --out outputs/results_novel_search/phase2_mmn_s7_roi.csv
python scripts/rank_novel_phase2.py                            # ranking + consensus set + rho
# Phase-2 waveform figures need the Phase-2 predictions, which live in their OWN root on the
# cluster (Stage J sets PREDICTIONS_ROOT) and must not overwrite the Phase-1 root locally:
for m in whisper-tiny whisper-base whisper-small whisper-medium wav2vec2-medium wav2vec2-large; do
  mkdir -p outputs/insilico_mmn_predictions_novel_phase2/$m
  rsync -av "<cluster>:<repo>/outputs/insilico_mmn_predictions_novel_phase2/$m/"'*.h5' \
        "outputs/insilico_mmn_predictions_novel_phase2/$m/"
done

python aux/analysis_novel_search/plots/phase1_results.py --phase 2 --skip_deviance \
    --literature_csv outputs/results_soafix/mmn_s7_roi.csv \
    --predictions_root outputs/insilico_mmn_predictions_novel_phase2   # Section 6, incl. Figs 13-14
python aux/analysis_novel_search/plots/phase1_results.py --phase 2 --only_waveforms \
    --wave_units uv \
    --predictions_root outputs/insilico_mmn_predictions_novel_phase2   # the raw-µV waveform variant
python aux/analysis_novel_search/plots/novel_search_plots.py --skip_phase1_figures
    # ^ Phase-2 counterpart of Figure 2, plus the Section-7 cross-phase outputs

# PDF of this memo, figures inlined. Needs `markdown` and a Chromium-family browser, neither of
# which is a dependency of mbs-env, so run it from any interpreter that has markdown installed.
python scripts/md_to_pdf.py aux/analysis_novel_search/novel_stimulus_search_results.md \
    --title "Novel tone-pair stimulus search — results (Phases 1 and 2)"
```

`phase1_results.py` keeps its Phase-1 name because Sections 2–5 cite it by that path; **`--phase 2`
reads `phase2_mmn_s7_roi.csv` and writes every output under a `phase2_` prefix**, so the two phases
cannot overwrite each other. It also takes `--results_dir` / `--out_dir` / `--predictions_root` /
`--literature_csv`, so any re-run can be rendered anywhere, and it writes every table in this memo
as a CSV beside its figure. **`--literature_csv` has to be passed explicitly** — its built-in
default is `outputs/results_24freq_7models/mmn_s7_roi.csv`, not the screen Section 3 is computed
from, so a bare re-run would rebuild Tables 4 and 6 and Figure 4 off the wrong CSV.
**`--skip_deviance` is what Phase 2 runs with**: the deviance
correlation is confounded by the selection (Caveat 6), so it is not computed rather than computed
and disclaimed. `--skip_literature` suppresses the literature comparison; note it also drops the
literature overlay from the grid-position figure, which is why the raw-µV pass must carry
**`--only_waveforms`** — that flag stops the run after the waveform figures, so a pass configured
for the waveforms cannot rewrite unrelated figures from a different configuration. `--wave_ylim
LO HI` changes the shared y-window. The waveform panels read the prediction HDF5s and check each
file's `n_deviants` attribute against the phase being rendered, so pointing a Phase-2 run at
Phase-1 predictions skips the figure loudly instead of mislabelling it. `novel_search_plots.py --skip_phase1_figures` writes
the Phase-2 counterpart of Figure 2 (`novel_n_agree_heatmap_phase2.png`) and the Section-7
cross-phase outputs, leaving the committed Phase-1 figures untouched.

### Figure outputs: PNG and SVG

Both scripts write **every** figure twice — the PNG into `--out_dir`
(`aux/analysis_novel_search/plots/`, which is what this memo embeds and what the PDF build reads),
and a vector SVG of the same figure, same stem, into `--svg_dir`. That defaults to `svgs/` beside
the plots directory — `aux/analysis_novel_search/svgs/` for the commands above, and `/tmp/svgs`
for a `--out_dir /tmp/figs` test run — so the SVGs never land inside the tree the PNGs live in.
Both formats are written with `bbox_inches="tight"`, so the two crop identically and are
interchangeable. `--no_svg` writes only the PNGs. Nothing needs re-running to get the SVGs: they
come out of the same commands, on every pass, including the `--only_waveforms --wave_units uv`
ones.

SVG text is embedded as glyph outlines rather than font references. These figures carry `µ`, `→`,
`≥`, `−`, `✓` and `ρ`, and a font-referencing SVG renders those through whatever the viewer has
installed — outlines cost file size but render identically everywhere. Heatmap colour grids are
drawn with `imshow`, so the cell grid itself arrives as an embedded raster inside the SVG; axes,
labels, annotations and colourbars are vector.
