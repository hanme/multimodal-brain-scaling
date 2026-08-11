# How to read these figures

Two dose-response checks on whether the in-silico MMN behaves like a human MMN, at the **FCz
electrode**, **mTRF** mapping, **six models**. 22 figures = 11 views × 2 gates.

Read §1 first — it is the vocabulary every figure uses. Then find your figure in §3.

---

## 1. Vocabulary shared by every figure

**The y-axis is always `trough_uv`** — the deviant−standard difference wave in µV, sampled at the
trough latency. **More negative = a deeper MMN.** The axis is **inverted everywhere**, so
*up = deeper = more MMN-like*. A line that rises left-to-right is the MMN-like direction.

**The x-axis is the manipulation** — either deviance size in semitones, or N (standards between
deviants). Both are things the human MMN is known to respond to.

**The two gates** decide which traces are allowed into an amplitude figure:

| gate | keeps | filename |
|---|---|---|
| **S7@0.75** | shape **and** trough ≤ −0.75 µV | bare name |
| **S2** | shape only — **no µV floor** | `__s2` suffix |

"Shape" (S2) = an interior trough in 100–240 ms that recovers ≥50% within 120 ms. S7 adds a depth
requirement on top. **S7 is the headline; S2 is the uncensored companion.** The floor removes the
shallow tail *by construction* — exactly the traces a weak dose-response moves — so **S7 numbers
are a lower bound** and the gap between the two gates is what the floor was hiding. It is not a
small gap: S7 keeps only 48% of LIT's S2 traces and 73% of NOVEL-P2's.

**The three sets** appear as the three panels of every multi-panel figure, always in this order:

| panel | set | what it is |
|---|---|---|
| left | `lit` | 48 literature conditions. Unselected, but timing varies and coverage is lumpy. |
| middle | `lit_p2` | both sources pooled (302 conditions). ~87% NOVEL-P2 rows. |
| right | `p2` | 254 selected conditions. Timing fixed, but **selected on the outcome**. |

**Colour = model identity**, always the same six colours (Okabe–Ito, CVD-checked). Every line is
solid; linestyle encodes nothing. Dotted grey is the y = 0 reference, not data. The dark grey
series is the **pooled** mean over the six models.

---

## 2. The two questions

- **`deviance_*`** — does the trough deepen as the standard→deviant frequency gap grows?
- **`n_effect_*`** — does the trough deepen as more standards separate the deviants?

Both are among the best-replicated properties of the human MMN, which is why they are the test.

---

## 3. The four figure families

### A. `*_pooled_*` — the headline amplitude result

`deviance_pooled_s7gated.png`, `n_effect_pooled.png` (+ `__s2`)

Three panels, one per set, pooled over the six models. **One series per panel** — the middle panel
pools both sources into a single line. Grey numbers above each point are the n behind it. The
y-axis is shared across all three panels so the sets are directly comparable.

*How to read it:* a line rising left-to-right = deeper with more deviance/more standards = the
MMN-like direction. Flat = no amplitude effect.

*Two things to watch:*
- **These means are UNPAIRED.** The gate admits a different *mix* of conditions at each x, so a
  rising line can be composition rather than deepening. On LIT's N-effect panel it is exactly
  that — the line rises but the paired test is null (p = 0.23, 50% of conditions, a coin flip).
  `n_effect_paired_n3_vs_n7*.csv` is what settles it.
- The deviance panels clip the deepest 2% of troughs so one outlier bin cannot set the scale.
  Clipped bins are drawn **on the boundary with their true value labelled**, never dropped.

### B. `*_per_model_*` — is the effect in every model, or one of them?

`deviance_per_model_s7gated__{lit,lit_p2,p2}.png`, `n_effect_per_model__{lit,lit_p2,p2}.png` (+ `__s2`)

Six panels, one per model, **shared y-axis** within a file. Deviance versions show raw points plus
a per-model OLS fit and ρ; N versions show mean ± SEM at each N. The stat block gives Spearman ρ,
p, and n.

*How to read it:* look for consistency of **sign** across the six panels, not just significance.
Six same-signed weak effects are stronger evidence than one significant model. In the `lit_p2`
files, marker shape distinguishes the two sources (● LIT, △ NOVEL-P2).

*Watch:* `△ k off-scale` in a panel means k points are deeper than the clip and are drawn as hollow
carets on the top edge. They are **included** in ρ and the fit — only their position is clipped.

### C. `*_rate*` — the uncensored view, and often the real result

`deviance_s7_rate.png`, `n_effect_s7_rate.png` (+ `__s2`)

Three panels; y = **fraction of all conditions that pass the gate**, denominator = every condition,
not just the passing ones. One faint line per model plus the heavy pooled line.

*Why this matters:* a count outcome **cannot be floored by an amplitude gate**, so a dose-response
can appear here that the amplitude axis cannot show. For the N-effect this *is* the result — the
rate climbs 54% → 59% (p = 2e-10) while the gated trough barely moves. **A monotone rate beside a
flat gated trough is a real positive, not a null.**

*Watch:* LIT's per-model lines swing between 0 and 1 because most of its semitone values carry only
2 conditions. Read the pooled line there.

### D. `deviance_overlap_lit_vs_p2*` — is the combined set trustworthy?

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
| `deviance_scaling_s7gated_stats*.csv` | ρ per (set × model), pooled rows, the per-source overlap ρ, and a ≤24 st subset |
| `n_effect_stats*.csv` | ρ per (set × model × unit); `unit=cell` is primary, `unit=trial` is the optimistic bound |
| `n_effect_paired_n3_vs_n7*.csv` | the paired within-condition N3-vs-N7 test — settles composition vs real deepening |
| `n_effect_source_agreement*.csv` | per model, do LIT and NOVEL-P2 agree in sign? This licenses (or refuses) reading `lit_p2` as one experiment |

**On `unit` in the N stats:** each (model, condition, N) cell holds 5 variations of the same
paradigm re-rolled, so its trials are not independent. The primary ρ collapses each cell to one
value first; the raw-trial ρ inflates n ~5× and understates p. Quote the cell-level one.

---

## 5. Five ways to misread these figures

1. **Taking an S7 amplitude null as "no effect."** The floor censors the shallow tail. The N-effect
   is null at S7 and significant at S2 — the gate was hiding about half of it.
2. **Reading a rising unpaired line as deepening.** See §3A. Check the paired CSV.
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

## 6. A note on the filenames

The `s7gated` / `s7_rate` stems date from when S7 was the only gate, so the S2 variants read
awkwardly — `deviance_pooled_s7gated__s2.png` is the **S2** figure despite `s7gated` in the name.
The `__s2` suffix and the in-figure title are authoritative; every figure states its own gate in the
title and subtitle. Renaming to a gate-neutral stem (`deviance_pooled__{s7,s2}.png`) would be
cleaner and is a one-line change, but the current names are referenced from the memo.

## Regenerating

```bash
conda activate mbs-env
python aux/analysis_MMN_realness_checks/deviance_scaling_s7gated.py --gate s7   # and --gate s2
python aux/analysis_MMN_realness_checks/n_effect_plots.py           --gate s7   # and --gate s2
```

PNGs land in `plots/`, vector twins in `svgs/`, statistics beside this file. Output is
reproducible: identical input gives byte-identical figures.
