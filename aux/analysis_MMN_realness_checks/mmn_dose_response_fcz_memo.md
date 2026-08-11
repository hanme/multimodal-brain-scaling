# MMN "realness" at FCz — the N-effect and deviance scaling of the S7 trough

Two dose-response checks on whether the in-silico MMN behaves like a human/clinical MMN. Both ask
the same question — **does the trough deepen as the paradigm makes the oddball more surprising?** —
along two axes the MMN literature has strong priors about.

**Site:** the FCz electrode only, mTRF only. **Gate:** S7@0.75 for every amplitude figure and
statistic. **Models:** six — whisper tiny/base/small/medium, wav2vec2 medium/large.

**Sources**, reported side by side and never pooled into one regression:

| | LIT | NOVEL-P2 |
|---|---|---|
| conditions | 48 = 24 literature Frequency methods × {regular, counter} | 254 = 127 selected pairs × {regular, counter} |
| semitone range | 0.84 – 12.0 (10 values) | 4.50 – 63.0 (122 values) |
| timing | SOA 200–1000 ms, duration 50–200 ms — **varies** | 580 ms / 80 ms / 10% — **fixed** |
| selection | none | **selected on the outcome** |
| S7@0.75 at FCz | 125/288 | 856/1524 |

Three analysis sets throughout: `lit` (48), `lit_p2` (302), `p2` (254).

**Status.** Part C (deviance) is complete. **Part B (N-effect) is blocked** on a cluster re-run —
see §2; the scripts are written and validated, only the input is missing.

---

## 1. Deviance scaling — the headline

Figures: `plots/*.png` (vector twins in `svgs/*.svg`) —
`deviance_pooled_s7gated`, `deviance_per_model_s7gated__{lit,lit_p2,p2}`, `deviance_s7_rate`,
`deviance_overlap_lit_vs_p2`. Statistics: `deviance_scaling_s7gated_stats.csv`.

Spearman ρ of `trough_uv` vs semitones, S7@0.75-gated, pooled over the 6 models. **Negative ρ =
deeper with more deviance = the MMN-like direction.**

| set | full range | ≤ 24 st | > 24 st |
|---|---|---|---|
| `lit` | **−0.31** (p = 4.2e-4, n = 125) | −0.31 (same; LIT tops out at 12 st) | — |
| `lit_p2` | −0.10 (p = 2.5e-3, n = 981) | **−0.17** (p = 2.6e-5, n = 644) | +0.10 (p = 0.068, n = 337) |
| `p2` | −0.06 (p = 0.056, n = 856) | **−0.15** (p = 9.1e-4, n = 519) | +0.10 (p = 0.068, n = 337) |

Direction is MMN-like in every set, and **all 6 models are negative** in `lit` and `lit_p2`
(4 and 2 individually significant). But three qualifications change how much of this is a deviance
effect:

### 1a. LIT's large ρ is a step between two clusters, not a gradient

LIT's −0.31 does **not** survive being split at the point where NOVEL-P2 begins:

| LIT sub-range | ρ | p | n | median trough |
|---|---|---|---|---|
| 0.84 – 3.16 st (low cluster) | −0.09 | 0.51 | 60 | −1.13 µV |
| 4.50 – 12 st (overlap) | −0.09 | 0.48 | 65 | −1.36 µV |
| full range | **−0.31** | 4.2e-4 | 125 | −1.27 µV |

Within each cluster the trend is flat; the whole correlation is the **−1.13 → −1.36 µV step
between them**. Of 24 methods, 10 sit at 3.16 st and 5 at 10.65 st, and SOA (200–1000 ms) and
duration (50–200 ms) vary across those methods. So LIT's headline number is a two-cluster contrast
confounded with timing — structurally the same weakness as the combined slope, and it should not be
quoted on its own.

### 1b. The overlap diagnostic: the sources agree, on a null

Within 4.50–12 st, where both sources have conditions and the range confound is removed:

| source | ρ | p | S7 rows | median |
|---|---|---|---|---|
| LIT | −0.089 | 0.48 | 65 / 120 | −1.36 µV |
| NOVEL-P2 | −0.094 | 0.20 | 189 / 336 | −1.29 µV |

Same sign, near-identical magnitude, near-identical median — but **both non-significant**. This is
agreement on a *null*. It does the useful negative job of ruling out a systematic µV offset between
the two sources driving the combined slope, so the `lit_p2` number is not a source artefact. It
does **not** establish a deviance effect inside the overlap. The `lit_p2` ρ must always be read
with this pair beside it.

### 1c. The effect exists below 2 octaves and reverses above

NOVEL-P2 is the only source with timing controlled, and it splits sharply at 24 st (2 octaves):

- **≤ 24 st: ρ = −0.15 (p = 9.1e-4, n = 519)** — deeper with more deviance, 5/6 models negative.
- **> 24 st: ρ = +0.10 (p = 0.068, n = 337)** — the trend flattens and turns.

Human MMN amplitude saturates with large deviance, so saturation is itself MMN-like; but a
separation above two octaves is far outside the classical MMN range and is arguably a *different
sound* rather than a deviant. **The ≤ 24 st estimate is the cleanest number in this deliverable**:
timing fixed, 519 gated rows, and it is the one that is not a between-cluster or between-source
contrast. It is also small.

### 1d. The uncensored view — S7@0.75 rate

The rate (S7 count / **all** conditions) is not floored by the gate, so it can show a dose-response
the amplitude axis cannot:

| set | first bin → last bin | trial-level ρ |
|---|---|---|
| `lit` | 0.333 (4/12) → 0.500 (6/12) | **+0.215** (p = 2.4e-4) |
| `lit_p2` | 0.357 (60/168) → 0.589 (99/168) | +0.069 (p = 3.2e-3) |
| `p2` | 0.511 (92/180) → 0.589 (99/168) | +0.005 (p = 0.86) |

More deviance → more likely to evoke a model MMN, clearly in LIT and `lit_p2`, **flat in NOVEL-P2
alone**. The rate effect is carried by the low-deviance end that only LIT covers, and NOVEL-P2's
rate is both high (0.51–0.59 everywhere) and compressed — exactly what selecting on S7 agreement
would produce. This is a positive result, but it is a positive result about the range LIT uniquely
covers, not about NOVEL-P2's.

### Verdict on deviance

A **real but small** MMN-like deviance effect, confined to ≤ 24 st, best estimated at ρ ≈ −0.15
from NOVEL-P2. The much larger LIT number (−0.31) is a confounded two-cluster step and overstates
it. The two sources fail in opposite ways — LIT has timing confounds and lumpy coverage but an
unselected sample and low-deviance reach; NOVEL-P2 has fixed timing and dense coverage but is
selected on the outcome and truncated from below — and they agree on **direction** everywhere and
on **magnitude** where they overlap. Given those opposite weaknesses, that convergence is the
strongest evidence available here, and it supports a weak deviance effect rather than a strong one.

---

## 2. The N-effect — blocked, not null

**No N result is reported, because the data to compute it does not exist yet.** This is a missing
input, not a null finding, and must not be read as one.

The 15 deviants of each method are a 3 × 5 grid of `N ∈ {3,5,7}` × `variation ∈ {1..5}`, so N lives
entirely inside the average the pipeline takes. `insilico_mmn_electrodes.py` wrote `deviant_mean`
and discarded the per-trial stack — verified absent in every committed electrode h5 for both
sources, though `n_deviants = 15` confirms the trials were computed. One driver patch plus two
prediction re-runs fixes it; see **`aux/analysis_MMN_realness_checks/handoff_per_trial_deviants_n_effect.md`** for the patch, the
exact cluster commands and three verification gates.

Both analysis scripts are written and **validated end-to-end on real data**. The soafix *parcel*
h5s do retain the legacy full-axis per-trial stack, so the whole path was exercised at the frontal
and central parcels before any cluster time was requested:

- `analyze_mmn_per_trial_n.py` reproduced the pipeline's own stored `n7v1_peak` to **2.4 × 10⁻⁷**
  across all 288 (model × method) cells — the per-trial scoring is the committed criterion exactly;
- it emitted exactly **4,320 rows** = 6 × 48 × 15, all assertions passing;
- `n_effect_plots.py` rendered all 5 figures and both stats CSVs from that input.

Those parcel numbers are a plumbing check, **not** a result, and appear in no figure or table.

When the re-run lands, the N analysis is the better-founded of the two: N is produced by the same
code path in both sources (`generate_deviant_sequence`, the same 1/(N+1) prefix rule, the same
3 × 5 grid) and both span the identical `N ∈ {3,5,7}`, so `lit_p2` is a straightforward n-boost on
an identical manipulation rather than a between-source contrast. It needs no overlap diagnostic —
only the per-model sign-agreement check, which is emitted.

---

## 3. Interpretation limits

1. **N is confounded with oddball probability by construction, in both sources.** The generator
   sets rare-tone probability in the prefix to 1/(N+1) (`00aa_generate_audio_stimuli.py:339`), so
   N = 3/5/7 means 25% / 16.7% / 12.5%. A trough deepening across N would be genuinely MMN-like but
   **cannot separate local spacing from global rarity**. Both are real mechanisms in humans; this
   design cannot attribute the effect to one. The figures label the x-axis with both.

2. **NOVEL-P2 is selected on the outcome variable.** Phase 2 admitted only pairs with at least one
   direction reaching `n_agree ≥ 5` of 6 models showing S7 at FCz in phase 1 (`MIN_AGREE_PHASE2`).
   The Part C outcome is the S7-gated FCz trough — essentially the selection variable. NOVEL-P2
   therefore answers the narrower question *"among stimuli that reliably evoke a model MMN, does
   amplitude scale with deviance?"*, **not** *"does deviance drive MMN amplitude?"*. Its larger n
   does not make it the stronger claim.

3. **NOVEL-P2's range is truncated from below by that selection.** Of the full 903-pair grid, 0 of
   42 pairs below 2 st and 0 of 41 between 2–4 st survived into phase 2; only 2 of 61 in 4–6 st.
   The minimum selected separation is 4.50 st. The small-deviance end — where the effect should be
   weakest and the trend most diagnostic — is **absent**, and LIT is the only coverage there.

4. **NOVEL-P2 trades a timing confound for a frequency-region one.** Fixing SOA/duration/probability
   across all conditions is a genuine advantage over LIT. But `f_low` ranges 200–4,525 Hz and
   `f_high` up to 7,611 Hz, so large separations necessarily involve different absolute frequency
   regions, where model and cochlear sensitivity differ. The ≤ 24 st subset is reported alongside
   the full range for this reason.

5. **LIT's coverage is lumpy and confounded with timing.** 10 of 24 methods at 3.16 st, 5 at
   10.65 st; the other 8 values carry a handful of S7-passing rows each, with SOA 200–1000 ms and
   duration 50–200 ms varying across them. §1a shows this is not a cosmetic caveat — it is the
   whole of LIT's −0.31.

6. **Six models, not seven.** whisper-large is excluded from every figure, table and statistic, for
   two independent reasons: its predicted µV run ~20–35× every other model (median S7-passing LIT
   trough at FCz −37.3 µV, min −160 µV, against a −1.08 to −1.75 µV band for the other six), so it
   would dominate any raw-µV mean and force a symlog axis that makes the rest unreadable; and it was
   never run in the novel search, so including it would make `lit` and `p2` incomparable. **It was
   not tested here — it did not quietly fail the criteria.** Excluding it is what buys back real µV
   units in the pooled panels.

7. **The amplitude axis is floored at −0.75 µV by construction.** Every plotted trough passes
   `trough_uv ≤ −0.75`, so each bin's shallow tail is absent and the gated slope is a **lower bound**
   on any true amplitude effect — most so at small deviance, where shallow troughs are expected and
   are exactly what the gate removes. This deliverable contains no ungated view by design; the
   S7@0.75 **rate** figures are the only uncensored evidence and should be read together with the
   amplitude ones, not as a footnote.

8. **Predicted µV are not literature EEG µV.** Ridge shrinks the mTRF's predicted amplitude, and the
   0.75 µV floor is calibrated to each model's own predicted-trough distribution, not to a clinical
   scale. Absolute depths (medians −1.27 to −1.49 µV here) are **not** comparable to published MMN
   amplitudes. Only direction and monotonicity across deviance and N are.

9. **The `lit_p2` deviance slope is partly a between-source contrast.** LIT supplies nearly all
   conditions below 4.5 st and NOVEL-P2 nearly all above 12 st, so a slope across the union moves
   with any systematic µV difference between two sources that differ in SOA, duration, deviant
   probability and selection. `lit_p2` is 87% NOVEL-P2 rows (856 of 981). This is why figure 4
   exists, and the `lit_p2` ρ is never quoted without the overlap result beside it (§1b). **The same
   caution does not apply to the `lit_p2` N-effect**, where both sources span the identical
   `N ∈ {3,5,7}`.

10. **A null here would be a real finding.** These dose-response effects are among the
    best-replicated properties of the human MMN; a model trough that does not track them bounds how
    MMN-like the in-silico response is. What the deviance data actually shows is neither a clean
    positive nor a null: a **small, range-limited, direction-consistent** effect, with the largest
    single number in the analysis (LIT's −0.31) explained away as a confounded step once decomposed.
