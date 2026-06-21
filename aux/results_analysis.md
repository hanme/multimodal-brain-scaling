# In-Silico MMN Results Analysis

**Scope:** 4 Whisper models (tiny/base/small/medium) × 2 target levels (parcels, electrodes) ×
10 literature classic-oddball stimulus methods × 2 independent mapping methods (mTRF, encoder)
= 160 (model, level, mapping, method) combinations, all complete (`outputs/results/mmn_results_table.csv`).

## What MMN means

**Mismatch negativity (MMN)** is an auditory event-related potential: a negative-going EEG
deflection, typically ~100–250 ms after an unexpected ("deviant") sound embedded among repeated
("standard") sounds, largest at fronto-central scalp sites (Näätänen et al., 1978; Näätänen &
Picton, 1987; peak ≈170 ms per Sams et al., 1985; window per Umbricht & Krljes, 2005). It reflects
automatic pre-attentive detection of acoustic deviance and is reduced or absent in some clinical
populations (e.g. schizophrenia) — the motivation for using it as an in-silico screening signal
here.

**How this document operationalizes it.** For each (model, level, mapping, method) run:

1. Predict EEG for the deviant-train average and the standard, time-locked to the final/critical
  tone.
2. Z-score each trace within a pre-onset baseline window (3× that stimulus method's SOA).
3. `diff = z(deviant) − z(standard)`; `baseline_normalized_peak` = the most-negative point of
  `diff` within the 100–240 ms post-onset window.
4. **MMN present** (per run, per target) := `baseline_normalized_peak < 0` — the deviant response
  dips below the standard's in that window. "Mean MMN" in the tables below averages that value
   over a fronto-central ROI:
  - **Electrodes:** ROI = `{Fz, FCz, Cz, FC1, FC2, F1, F2}` — the exact ROI already used by
  `insilico_mmn_electrodes.py`'s built-in verdict.
  - **Parcels:** ROI = `{frontal, central}` — the natural parcel-level analog; no parcel-level
  verdict previously existed in the pipeline, so this generalizes the electrode criterion.

This is a **magnitude/negativity criterion only** — it checks that the deviant response dips below
the standard's somewhere in the window, but not that the dip has the characteristic MMN *shape*
(a trough that rises back toward baseline within the window, rather than e.g. the tail of an
unrelated ascending trend). See the shape-criteria caveat below.

This document deliberately does **not** discuss fit quality (encoder probe accuracy on held-out
EEG) — only MMN presence/absence and response magnitude are covered here.

---

## Section 1 — Method A (mTRF)

**Table 1a. Mean MMN per model × method**


| Model  | Method    | Stimulus name | MMN (parcels) | MMN (electrodes) |
| ------ | --------- | ------------- | ------------- | ---------------- |
| tiny   | method_27 | 1000→1064 Hz  | -0.87         | -0.78            |
| tiny   | method_37 | 1000→1050 Hz  | -0.25         | -0.30            |
| tiny   | method_43 | 633→700 Hz    | -0.61         | -0.49            |
| tiny   | method_44 | 633→1000 Hz   | -1.46         | -1.63            |
| tiny   | method_53 | 1000→1200 Hz  | -1.04         | -0.83            |
| tiny   | method_55 | 1000→2000 Hz  | -1.43         | -1.28            |
| tiny   | method_60 | 1000→1500 Hz  | -0.56         | -0.40            |
| tiny   | method_72 | 1000→1200 Hz  | -0.96         | -0.78            |
| tiny   | method_74 | 1000→1500 Hz  | -0.58         | -0.53            |
| tiny   | method_75 | 1000→1200 Hz  | -0.95         | -0.77            |
| base   | method_27 | 1000→1064 Hz  | -0.01         | +0.02            |
| base   | method_37 | 1000→1050 Hz  | -0.48         | -0.38            |
| base   | method_43 | 633→700 Hz    | +0.02         | +0.05            |
| base   | method_44 | 633→1000 Hz   | +0.16         | +0.11            |
| base   | method_53 | 1000→1200 Hz  | -0.31         | -0.36            |
| base   | method_55 | 1000→2000 Hz  | -1.44         | -1.42            |
| base   | method_60 | 1000→1500 Hz  | +0.43         | -0.12            |
| base   | method_72 | 1000→1200 Hz  | -0.40         | -0.38            |
| base   | method_74 | 1000→1500 Hz  | -0.06         | -0.06            |
| base   | method_75 | 1000→1200 Hz  | -0.40         | -0.38            |
| small  | method_27 | 1000→1064 Hz  | +0.30         | -0.21            |
| small  | method_37 | 1000→1050 Hz  | -0.21         | -0.20            |
| small  | method_43 | 633→700 Hz    | -0.22         | -0.17            |
| small  | method_44 | 633→1000 Hz   | -0.57         | -0.64            |
| small  | method_53 | 1000→1200 Hz  | -0.48         | -0.18            |
| small  | method_55 | 1000→2000 Hz  | -0.68         | -0.81            |
| small  | method_60 | 1000→1500 Hz  | -0.64         | -0.42            |
| small  | method_72 | 1000→1200 Hz  | -0.20         | -0.13            |
| small  | method_74 | 1000→1500 Hz  | -0.40         | -0.25            |
| small  | method_75 | 1000→1200 Hz  | -0.21         | -0.14            |
| medium | method_27 | 1000→1064 Hz  | -0.29         | -0.28            |
| medium | method_37 | 1000→1050 Hz  | -0.30         | -0.14            |
| medium | method_43 | 633→700 Hz    | -3.26         | -3.76            |
| medium | method_44 | 633→1000 Hz   | -0.41         | -0.74            |
| medium | method_53 | 1000→1200 Hz  | -0.55         | -0.32            |
| medium | method_55 | 1000→2000 Hz  | +0.00         | -0.26            |
| medium | method_60 | 1000→1500 Hz  | -1.33         | -0.61            |
| medium | method_72 | 1000→1200 Hz  | -0.66         | -0.50            |
| medium | method_74 | 1000→1500 Hz  | -0.59         | -0.34            |
| medium | method_75 | 1000→1200 Hz  | -0.63         | -0.49            |


**Table 1b. Per-model MMN count summary** (rows ordered tiny → medium)


| Model  | Parcels (n/10) | Electrodes (n/10) | Total (n/20) |
| ------ | -------------- | ----------------- | ------------ |
| tiny   | 10/10          | 10/10             | 20/20        |
| base   | 7/10           | 7/10              | 14/20        |
| small  | 9/10           | 10/10             | 19/20        |
| medium | 9/10           | 10/10             | 19/20        |


**Table 1c. Per-method average across models** (parcels + electrodes pooled, i.e. averaged over
the 8 model×level cells for that method)


| Method    | Stimulus name | Average MMN/response across models |
| --------- | ------------- | ---------------------------------- |
| method_27 | 1000→1064 Hz  | -0.26                              |
| method_37 | 1000→1050 Hz  | -0.28                              |
| method_43 | 633→700 Hz    | -1.06                              |
| method_44 | 633→1000 Hz   | -0.65                              |
| method_53 | 1000→1200 Hz  | -0.51                              |
| method_55 | 1000→2000 Hz  | -0.91                              |
| method_60 | 1000→1500 Hz  | -0.46                              |
| method_72 | 1000→1200 Hz  | -0.50                              |
| method_74 | 1000→1500 Hz  | -0.35                              |
| method_75 | 1000→1200 Hz  | -0.50                              |


**Table 1d. Per-model average across stimuli** (each method's parcels/electrodes mean pooled
into one value first, then averaged over the 10 methods)


| Model  | Average MMN/response across stimuli | Number of stimuli showing MMN |
| ------ | ----------------------------------- | ----------------------------- |
| tiny   | -0.83                               | 10/10                         |
| base   | -0.27                               | 6/10                          |
| small  | -0.32                               | 9/10                          |
| medium | -0.77                               | 10/10                         |


**Table 1e. Mean MMN per model, averaged across stimuli** (parcels and electrodes kept separate,
unlike Table 1d — each column is that level's mean over the 10 methods)


| Model  | Mean MMN in parcels (avg across stimuli) | Mean MMN in electrodes (avg across stimuli) |
| ------ | ---------------------------------------- | ------------------------------------------- |
| tiny   | -0.87                                    | -0.78                                       |
| base   | -0.25                                    | -0.29                                       |
| small  | -0.33                                    | -0.32                                       |
| medium | -0.80                                    | -0.74                                       |


---

## Section 2 — Method B (encoder)

**Table 2a. Mean MMN per model × method**


| Model  | Method    | Stimulus name | MMN (parcels) | MMN (electrodes) |
| ------ | --------- | ------------- | ------------- | ---------------- |
| tiny   | method_27 | 1000→1064 Hz  | -0.78         | -0.11            |
| tiny   | method_37 | 1000→1050 Hz  | -0.42         | -0.89            |
| tiny   | method_43 | 633→700 Hz    | +0.28         | +0.06            |
| tiny   | method_44 | 633→1000 Hz   | +0.26         | -0.12            |
| tiny   | method_53 | 1000→1200 Hz  | +0.51         | -1.49            |
| tiny   | method_55 | 1000→2000 Hz  | +1.79         | -0.04            |
| tiny   | method_60 | 1000→1500 Hz  | -2.53         | +1.07            |
| tiny   | method_72 | 1000→1200 Hz  | -6.53         | +0.41            |
| tiny   | method_74 | 1000→1500 Hz  | -0.06         | +0.25            |
| tiny   | method_75 | 1000→1200 Hz  | -6.49         | +0.41            |
| base   | method_27 | 1000→1064 Hz  | +1.63         | -0.48            |
| base   | method_37 | 1000→1050 Hz  | +2.33         | -0.41            |
| base   | method_43 | 633→700 Hz    | -0.17         | -0.45            |
| base   | method_44 | 633→1000 Hz   | -0.77         | +1.15            |
| base   | method_53 | 1000→1200 Hz  | -1.64         | -2.87            |
| base   | method_55 | 1000→2000 Hz  | -2.41         | -0.30            |
| base   | method_60 | 1000→1500 Hz  | +0.72         | -1.17            |
| base   | method_72 | 1000→1200 Hz  | -2.89         | -0.36            |
| base   | method_74 | 1000→1500 Hz  | +1.65         | +0.57            |
| base   | method_75 | 1000→1200 Hz  | -2.91         | -0.37            |
| small  | method_27 | 1000→1064 Hz  | -0.42         | +0.02            |
| small  | method_37 | 1000→1050 Hz  | +0.22         | +0.60            |
| small  | method_43 | 633→700 Hz    | +0.28         | +0.12            |
| small  | method_44 | 633→1000 Hz   | +0.87         | +1.38            |
| small  | method_53 | 1000→1200 Hz  | +0.77         | +0.15            |
| small  | method_55 | 1000→2000 Hz  | -0.30         | -1.25            |
| small  | method_60 | 1000→1500 Hz  | +0.19         | -1.74            |
| small  | method_72 | 1000→1200 Hz  | -0.19         | +1.37            |
| small  | method_74 | 1000→1500 Hz  | +0.34         | +0.04            |
| small  | method_75 | 1000→1200 Hz  | -0.20         | +1.35            |
| medium | method_27 | 1000→1064 Hz  | -0.18         | +0.01            |
| medium | method_37 | 1000→1050 Hz  | -0.30         | +0.48            |
| medium | method_43 | 633→700 Hz    | +0.25         | +0.07            |
| medium | method_44 | 633→1000 Hz   | -1.59         | +0.02            |
| medium | method_53 | 1000→1200 Hz  | -1.02         | -0.97            |
| medium | method_55 | 1000→2000 Hz  | +2.61         | +0.25            |
| medium | method_60 | 1000→1500 Hz  | +1.29         | -2.76            |
| medium | method_72 | 1000→1200 Hz  | -1.06         | -0.10            |
| medium | method_74 | 1000→1500 Hz  | -0.21         | -0.24            |
| medium | method_75 | 1000→1200 Hz  | -1.04         | -0.10            |


**Table 2b. Per-model MMN count summary** (rows ordered tiny → medium)


| Model  | Parcels (n/10) | Electrodes (n/10) | Total (n/20) |
| ------ | -------------- | ----------------- | ------------ |
| tiny   | 6/10           | 5/10              | 11/20        |
| base   | 6/10           | 8/10              | 14/20        |
| small  | 4/10           | 2/10              | 6/20         |
| medium | 7/10           | 5/10              | 12/20        |


**Table 2c. Per-method average across models** (parcels + electrodes pooled, i.e. averaged over
the 8 model×level cells for that method)


| Method    | Stimulus name | Average MMN/response across models |
| --------- | ------------- | ---------------------------------- |
| method_27 | 1000→1064 Hz  | -0.04                              |
| method_37 | 1000→1050 Hz  | +0.20                              |
| method_43 | 633→700 Hz    | +0.06                              |
| method_44 | 633→1000 Hz   | +0.15                              |
| method_53 | 1000→1200 Hz  | -0.82                              |
| method_55 | 1000→2000 Hz  | +0.04                              |
| method_60 | 1000→1500 Hz  | -0.62                              |
| method_72 | 1000→1200 Hz  | -1.17                              |
| method_74 | 1000→1500 Hz  | +0.29                              |
| method_75 | 1000→1200 Hz  | -1.17                              |


**Table 2d. Per-model average across stimuli** (each method's parcels/electrodes mean pooled
into one value first, then averaged over the 10 methods)


| Model  | Average MMN/response across stimuli | Number of stimuli showing MMN |
| ------ | ----------------------------------- | ----------------------------- |
| tiny   | -0.72                               | 6/10                          |
| base   | -0.46                               | 6/10                          |
| small  | +0.18                               | 3/10                          |
| medium | -0.23                               | 7/10                          |


**Table 2e. Mean MMN per model, averaged across stimuli** (parcels and electrodes kept separate,
unlike Table 2d — each column is that level's mean over the 10 methods)


| Model  | Mean MMN in parcels (avg across stimuli) | Mean MMN in electrodes (avg across stimuli) |
| ------ | ---------------------------------------- | ------------------------------------------- |
| tiny   | -1.40                                    | -0.05                                       |
| base   | -0.45                                    | -0.47                                       |
| small  | +0.16                                    | +0.20                                       |
| medium | -0.13                                    | -0.33                                       |


---

## Cross-Method Comparisons

**Table 3. mTRF vs. encoder agreement** (per model × level, over the 10 methods)


| Model  | Level      | Both MMN | Both no-MMN | Agree | Disagree |
| ------ | ---------- | -------- | ----------- | ----- | -------- |
| tiny   | parcels    | 6        | 0           | 6/10  | 4/10     |
| tiny   | electrodes | 5        | 0           | 5/10  | 5/10     |
| base   | parcels    | 4        | 1           | 5/10  | 5/10     |
| base   | electrodes | 6        | 1           | 7/10  | 3/10     |
| small  | parcels    | 3        | 0           | 3/10  | 7/10     |
| small  | electrodes | 2        | 0           | 2/10  | 8/10     |
| medium | parcels    | 7        | 1           | 8/10  | 2/10     |
| medium | electrodes | 5        | 0           | 5/10  | 5/10     |


**Table 4. Stimulus-method consistency** (MMN count across all 16 model × level × mapping combinations)


| Method    | Stimulus (source)              | mTRF (n/8) | Encoder (n/8) | Total (n/16) |
| --------- | ------------------------------ | ---------- | ------------- | ------------ |
| method_27 | 1000→1064 Hz (Schall_1999a)    | 6/8        | 5/8           | 11/16        |
| method_37 | 1000→1050 Hz (Javitt_2000a)    | 8/8        | 4/8           | 12/16        |
| method_43 | 633→700 Hz (Michie_2000b)      | 6/8        | 2/8           | 8/16         |
| method_44 | 633→1000 Hz (Michie_2000c)     | 6/8        | 3/8           | 9/16         |
| method_53 | 1000→1200 Hz (Salisbury_2002a) | 8/8        | 5/8           | 13/16        |
| method_55 | 1000→2000 Hz (Shinozaki_2002a) | 7/8        | 5/8           | 12/16        |
| method_60 | 1000→1500 Hz (Umbricht_2003a)  | 7/8        | 4/8           | 11/16        |
| method_72 | 1000→1200 Hz (Bodatsch_2011)   | 8/8        | 6/8           | 14/16        |
| method_74 | 1000→1500 Hz (Domjan_2012)     | 8/8        | 3/8           | 11/16        |
| method_75 | 1000→1200 Hz (Karger_2014)     | 8/8        | 6/8           | 14/16        |


---

## Notes & caveats

- `**whisper-tiny` under mTRF is 100% MMN-positive (20/20)** — every single stimulus method
triggers the criterion at both levels. That is unusually clean compared to every other
model/mapping combination and is worth treating with some skepticism rather than as a strong
positive finding: it may reflect a systematic negative drift in the predicted timecourse
(e.g. baseline-window variance sensitivity, the same mechanism below) rather than 10
independent genuine effects.
- **A handful of encoder runs show extreme mean MMN values** (e.g. `whisper-tiny`/parcels/method_72:
-6.53, method_75: -6.49; `whisper-base`/parcels/method_72: -2.89, method_75: -2.91). Direct
inspection of the underlying figures shows the predicted deviant/standard traces for these
particular (model, method) pairs are smooth monotonic ramps rather than oscillatory
ERP-like shapes; the pre-onset baseline sits on the near-flat part of that ramp, so its variance
is tiny and the z-scored peak metric is inflated by dividing by a near-zero baseline std. This
is a property of those specific predicted-response shapes interacting with the z-scoring
metric, not a data-pipeline bug — but it means these particular cells shouldn't be read as
"10x stronger MMN" in a literal sense.
  - `whisper-medium`/electrodes/mTRF/method_43 (-3.76) is the analogous case on the mTRF side.
- **The threshold is exactly 0**, so several verdicts are marginal (e.g. `base`/method_27/parcels
mTRF: -0.01; `base`/method_43/parcels mTRF: +0.02). Treat rows within roughly ±0.1 of zero as
"no clear evidence either way" rather than a confident verdict.
- **mTRF is far more uniformly MMN-positive than the encoder** (Table 1b/2b totals: mTRF ranges
14–20/20 per model; encoder ranges 6–14/20). Table 3 shows agreement between the two mappings
is generally weak-to-moderate (2–8 out of 10 per model/level), so the two independent mapping
methods are not strongly corroborating each other's verdicts at the individual-method level —
worth investigating further before treating either mapping's verdict as ground truth.
- Fit quality (encoder probe accuracy on held-out EEG) is intentionally excluded from this
document per request; these tables describe MMN presence/absence and response magnitude only.
- **Stimulus sources are provisional, not yet fully vetted.** The 10 methods were selected to get
the pipeline running end-to-end, with the expectation that 1–2 may be swapped in/out later once
the literature list is reviewed more carefully. Current inclusion criteria: most recent
publication year per paper, only one method permitted per paper, frequency variants only
(no duration/intensity deviants in this round). Treat the per-method rows (Tables 1a/1c/2a/2c,
Table 4) as more provisional than the per-model rollups (1b/1d/1e/2b/2d/2e) — a future swap could
shift individual method rows without necessarily changing the overall per-model picture much, or
could change it a lot; this hasn't been stress-tested.
- **The magnitude criterion can be satisfied by curve shapes that aren't a real MMN trough.**
Because `baseline_normalized_peak` is just "most negative point in the 100–240 ms window," it
doesn't distinguish a genuine dip-and-recover trough from e.g. a monotonically ascending (or
descending) curve that simply happens to be most negative right at the window's edge — in which
case the "peak" is really just wherever the window cut off an unrelated trend, not a deviance
response. The smooth-ramp cases already flagged above (`whisper-tiny`/`whisper-base` encoder
parcels, several methods) are examples of this: visually, those curves don't dip-and-recover, they
ramp through the window. **Open question: should a shape-based criterion be added** (e.g. require
the window's minimum to fall strictly inside the window rather than at either edge, and/or require
the curve to rise back toward baseline after the trough by some margin)? This would need the full
per-time-point curves (saved in the prediction h5s on jed, not in the local CSV) rather than just
the scalar `peak`/`n7v1_peak` already extracted — not yet implemented, flagging for discussion
before deciding whether to add it.

---

## Section 3 — Pre-tone baseline window comparison (excluding the N-1 tone window to prevent leakage)

### Rationale

The baseline window used for z-scoring (`finalize_method()`) originally spanned the 4th-, 3rd-,
and 2nd-to-last tone before the critical tone (`[-3×SOA, 0)`). The 2nd-to-last tone (N-1) sits
immediately before the critical-tone onset, and EEG responses to a tone often include a large
negative dip of their own — so including N-1 in the "quiet" baseline risks leaking some of that
response's amplitude into the baseline statistics. Because the MMN verdict is itself a *trough*
(a negative deflection), a baseline that is already partly dipping could systematically work
against detecting the deviant's trough (i.e. disadvantage the observed MMN). This comparison
tests whether dropping the N-1 tone from the baseline changes the results.

### Setup

Both runs are otherwise identical (same models, stimuli, mapping methods, lag/window settings) —
the only change is the baseline window passed to `finalize_method()`:


| Run                                    | Baseline window    | Tones included                |
| -------------------------------------- | ------------------ | ----------------------------- |
| `outputs/` (original)                  | `[-3×SOA, 0)`      | N-2, N-3, N-4 (3 tones)       |
| `outputs_new_pre_tone_baseline/` (new) | `[-3×SOA, -1×SOA)` | N-2, N-3 (2 tones, drops N-1) |


Both result tables (`outputs/results/mmn_results_table.csv`,
`outputs_new_pre_tone_baseline/results/mmn_results_table.csv`) have identical shape — 160 rows
(4 models × {parcels, electrodes} × {mTRF, encoder} × 10 methods) — so every run lines up 1:1
for a paired comparison.

### Headline numbers

**Table 5. Overall agreement between the two baseline windows**


| Comparison                                                         | Value     |
| ------------------------------------------------------------------ | --------- |
| Paired electrode/parcel peak values compared                       | 4160      |
| Pearson r, all paired peak values (raw)                            | 0.32      |
| Pearson r, all paired peak values (excl. 1 outlier run, see below) | 0.79      |
| Spearman ρ, all paired peak values                                 | 0.82      |
| Sign agreement (MMN present/absent), electrode/parcel level (raw)  | 92.8%     |
| Sign agreement, electrode/parcel level (excl. outlier run)         | 93.7%     |
| ROI-level verdict flips (frontal+central mean, 80 runs)            | 4/80 (5%) |
| MMN-present count, ROI level — original baseline                   | 58/80     |
| MMN-present count, ROI level — new baseline                        | 58/80     |


**Table 6. Per-mapping agreement (ROI-mean peak, excluding the 1 outlier run)**

| Mapping | n runs | Pearson r | Mean |diff| |
| ------- | ------ | --------- | ----------- |
| mTRF    | 40     | 0.97      | 0.10        |
| Encoder | 39     | 0.76      | 0.90        |

**Table 7. The 4 runs whose ROI-level verdict flips**


| Model          | Level   | Mapping | Method    | Peak (original) | Peak (new) |
| -------------- | ------- | ------- | --------- | --------------- | ---------- |
| whisper-base   | parcels | encoder | method_55 | -2.41           | +3.61      |
| whisper-medium | parcels | mtrf    | method_55 | +0.00           | -0.10      |
| whisper-tiny   | parcels | encoder | method_44 | +0.26           | -0.66      |
| whisper-tiny   | parcels | encoder | method_60 | -2.53           | +1.24      |


**Table 8. Per-method sensitivity to the baseline window** (mean/max |diff| across all
electrode+parcel peak columns, pooled over models and mappings, n=416 values per method)

| Method    | Stimulus name | SOA (ms) | Mean |diff| | Max |diff| |
| --------- | ------------- | -------- | ----------- | ---------- |
| method_53 | 1000→1200 Hz  | 333      | 3.13        | 37.50      |
| method_60 | 1000→1500 Hz  | 300      | 1.02        | 5.90       |
| method_37 | 1000→1050 Hz  | 310      | 0.63        | 7.53       |
| method_55 | 1000→2000 Hz  | 500      | 0.38        | 6.50       |
| method_27 | 1000→1064 Hz  | 900      | 0.36        | 3.42       |
| method_72 | 1000→1200 Hz  | 500      | 0.25        | 1.55       |
| method_75 | 1000→1200 Hz  | 500      | 0.24        | 1.47       |
| method_74 | 1000→1500 Hz  | 1000     | 0.23        | 1.83       |
| method_44 | 633→1000 Hz   | 510      | 0.23        | 1.75       |
| method_43 | 633→700 Hz    | 510      | 0.20        | 1.89       |

### Per-mapping breakdown (parcels/electrodes × mTRF/encoder)

Same "before" (original 3-tone baseline) vs. "now" (new 2-tone baseline excluding N-1) comparison
as Table 5, but broken out per mapping method instead of pooled, with explicit delta columns
(`now − before`). Mean MMN is the ROI-mean `baseline_normalized_peak` (frontal+central for
parcels; Fz/FCz/Cz/FC2/F1/F2 for electrodes — FC1 isn't present in this montage), averaged over
the 10 stimulus methods; the count column is how many of those 10 methods are MMN-present
(ROI-mean peak < 0).

**Table 9. Parcels — mTRF**


| Model  | Mean MMN (before) | Mean MMN (now) | Δ mean MMN | MMN count (before, /10) | MMN count (now, /10) | Δ count |
| ------ | ----------------- | -------------- | ---------- | ----------------------- | -------------------- | ------- |
| tiny   | -0.87             | -0.83          | +0.04      | 10/10                   | 10/10                | 0       |
| base   | -0.25             | -0.25          | -0.00      | 7/10                    | 7/10                 | 0       |
| small  | -0.33             | -0.35          | -0.02      | 9/10                    | 9/10                 | 0       |
| medium | -0.80             | -1.02          | -0.21      | 9/10                    | 10/10                | +1      |


**Table 10. Parcels — encoder**


| Model  | Mean MMN (before) | Mean MMN (now) | Δ mean MMN | MMN count (before, /10) | MMN count (now, /10) | Δ count |
| ------ | ----------------- | -------------- | ---------- | ----------------------- | -------------------- | ------- |
| tiny   | -1.40             | -0.81          | +0.59      | 6/10                    | 6/10                 | 0       |
| base   | -0.45             | +0.58          | +1.03      | 6/10                    | 5/10                 | -1      |
| small  | +0.16             | +0.66          | +0.50      | 4/10                    | 4/10                 | 0       |
| medium | -0.13             | -0.20          | -0.07      | 7/10                    | 7/10                 | 0       |


**Table 11. Electrodes — mTRF**


| Model  | Mean MMN (before) | Mean MMN (now) | Δ mean MMN | MMN count (before, /10) | MMN count (now, /10) | Δ count |
| ------ | ----------------- | -------------- | ---------- | ----------------------- | -------------------- | ------- |
| tiny   | -0.78             | -0.78          | +0.00      | 10/10                   | 10/10                | 0       |
| base   | -0.29             | -0.33          | -0.04      | 7/10                    | 7/10                 | 0       |
| small  | -0.32             | -0.34          | -0.03      | 10/10                   | 10/10                | 0       |
| medium | -0.74             | -0.97          | -0.22      | 10/10                   | 10/10                | 0       |


**Table 12. Electrodes — encoder**


| Model  | Mean MMN (before) | Mean MMN (now) | Δ mean MMN | MMN count (before, /10) | MMN count (now, /10) | Δ count |
| ------ | ----------------- | -------------- | ---------- | ----------------------- | -------------------- | ------- |
| tiny   | -0.05             | +1.17          | +1.22      | 5/10                    | 4/10                 | -1      |
| base   | -0.47             | -0.30          | +0.17      | 8/10                    | 7/10                 | -1      |
| small  | +0.20             | +0.37          | +0.17      | 2/10                    | 2/10                 | 0       |
| medium | -0.33             | -0.14          | +0.19      | 5/10                    | 7/10                 | +2      |


Mirrors the pooled finding: **mTRF is essentially baseline-window-invariant** (Tables 9/11 — all
Δ count = 0 except one +1, Δ mean MMN mostly under 0.05 in magnitude except `medium`), while
**encoder shows larger swings in mean magnitude and more count flips** (Tables 10/12), consistent
with the z-score-denominator instability described below being concentrated in the encoder mapping.

### Structural risk

- Dropping the N-1 tone shrinks the baseline to 2 tone-cycles instead of 3, so the z-score's
baseline SD is estimated from fewer samples — for one run (`whisper-tiny`/encoder/`method_53`)
this denominator collapsed ~5× (confirmed directly in the raw prediction h5), flipping the sign
and inflating the peak by an order of magnitude; short-SOA, encoder-mapped runs are the most
exposed to this failure mode.

---

## Section 4 — Comparison of different possible MMN shape metrics using existing ROI definitions

### Metric definitions

All five criteria share the same magnitude gate (the windowed trough must be `< 0`) and are
evaluated on the same ROI-mean z-scored `deviant − standard` curve over the current 100–240 ms
window (`{frontal, central}` for parcels, `{Fz, FCz, Cz, FC1, FC2, F1, F2}` for electrodes); C0
is the metric already used everywhere else in this document.


| ID     | Name                       | Definition                                                                                                                                                                                                                                    |
| ------ | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C0** | current (magnitude only)   | The most negative point of the curve in the 100–240 ms window is below 0. No shape requirement.                                                                                                                                               |
| **S1** | interior argmin            | Like C0, plus the trough may not sit at either edge of the window — it must be at least one time-bin inside both the 100 ms and 240 ms boundaries.                                                                                            |
| **S2** | trough + recovery          | Like C0, plus after the trough the curve must climb back up by at least 50% of the trough's depth within 120 ms (a genuine dip-and-recover, not a curve still descending).                                                                    |
| **S3** | interior & recovery        | S1 **and** S2 both required.                                                                                                                                                                                                                  |
| **S4** | control-window specificity | Like C0, plus the in-window trough must be more negative than the lowest point of a later window (300–440 ms) where no MMN is expected — rules out cases where the "trough" is really the edge of a deeper, unrelated dip further downstream. |


### Rationale

The magnitude-only criterion (C0) used everywhere above is satisfied by any curve whose
lowest point in the window is negative — it cannot distinguish a genuine MMN dip-and-recover
from a curve that is simply still ramping downward when the window cuts off (see the
"magnitude criterion can be satisfied by curve shapes that aren't a real MMN trough" caveat
above). A handful of runs already flagged as extreme outliers (e.g. `whisper-tiny`/encoder/
parcels/`method_72`/`method_75`, peaks ≈ −6.5) turned out on inspection to be smooth monotonic
ramps rather than oscillatory MMN-like responses, with their true (much later) trough sitting
outside the scored window. `aux/mmn_criterion_investigation.md` worked through candidate
shape-aware criteria to catch this systematically; this section reports the actual counts from
running those criteria (`scripts/analyze_mmn_criteria.py`) over all 160 runs, on the existing
100–240 ms window and ROI definitions (the window itself is not varied — see that document's
Q2 for why it was kept as-is).

### Results — mTRF

**Table 13. MMN-present counts per criterion, by model** (n/20 = parcels + electrodes pooled, 10 methods each)


| Model            | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) |
| ---------------- | --------- | --------- | --------- | --------- | --------- |
| tiny             | 20/20     | 8/20      | 17/20     | 8/20      | 16/20     |
| base             | 12/20     | 4/20      | 12/20     | 4/20      | 6/20      |
| small            | 19/20     | 11/20     | 19/20     | 11/20     | 14/20     |
| medium           | 19/20     | 17/20     | 17/20     | 15/20     | 10/20     |
| **Total (n/80)** | **70/80** | **40/80** | **65/80** | **38/80** | **46/80** |


### Results — Encoder

**Table 14. MMN-present counts per criterion, by model** (n/20 = parcels + electrodes pooled, 10 methods each)


| Model            | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) |
| ---------------- | --------- | --------- | --------- | --------- | --------- |
| tiny             | 11/20     | 4/20      | 3/20      | 2/20      | 2/20      |
| base             | 13/20     | 6/20      | 9/20      | 4/20      | 7/20      |
| small            | 6/20      | 3/20      | 4/20      | 2/20      | 3/20      |
| medium           | 12/20     | 2/20      | 0/20      | 0/20      | 1/20      |
| **Total (n/80)** | **42/80** | **15/80** | **16/80** | **8/80**  | **13/80** |


### Summary

- **Encoder counts collapse far more under shape gating than mTRF.** Under the recommended
S2 criterion, mTRF retains 65/70 (93%) of its C0-positive runs, but encoder retains only
16/42 (38%) — consistent with the earlier suspicion that a larger share of the encoder's
magnitude-only "MMN present" verdicts are non-MMN-shaped curves (ramps), not genuine troughs.
- **S1/S3 (interior-only) penalize mTRF disproportionately** (70→40 and 70→38) compared to
S2/S4 (70→65 and 70→46) — because several genuine mTRF troughs sit right at the window's
100 ms edge and get wrongly rejected by an interior-only test. This is the concrete
case for preferring S2 and/or S4 over S1/S3, matching the recommendation in
`aux/mmn_criterion_investigation.md`.
- **4 of the 5 previously flagged smooth-ramp cases behave exactly as predicted** — all four
`whisper-{tiny,base}`/parcels/encoder/`method_{72,75}` runs flip to absent under every shape
criterion. The fifth, `whisper-medium`/electrodes/mTRF/`method_43`, unexpectedly stays
**present under all four shape criteria** (interior trough at 171 ms, 95% recovery) —
contradicting the earlier figure-based read of it as an edge-spike artifact and worth a
second look at that specific trace.
- **All 3 clear-trough sanity checks stay present under S2/S4**, including  
`whisper-base`/mTRF/parcels/`method_55`, whose trough sits at the window's left (100 ms)  
edge — the concrete real-data example that rules out S1/S3 as a standalone criterion.

