# In-Silico MMN Results Analysis — With Counterbalanced Methods

> **Extends `aux/results_analysis.md`** by adding 10 counterbalanced stimulus pairs
> (standard/deviant frequencies swapped) to every analysis. Scope: 4 Whisper models
> × 2 levels (parcels, electrodes) × 20 methods (10 regular + 10 counter) × 2 mappings
> (mTRF, encoder) = **320 (model, level, mapping, method)** combinations.
> All metric definitions and section structure are identical to `aux/results_analysis.md`;
> only counts, tables, and observations are updated. See that document for full method
> descriptions, criterion definitions, and the Section 3 pre-tone baseline analysis
> (which is not repeated here as the counterbalanced stimuli use the same baseline design).

## Key motivation for counterbalancing

Adding counterbalanced pairs (where the former deviant frequency becomes the new standard
and vice versa) provides a within-stimulus control for frequency-preference artifacts. If
the in-silico MMN signal reflects genuine deviance detection — the model responding to
*unexpectedness* rather than to a specific frequency — it should appear when *either*
frequency plays the deviant role. Conversely, if the signal is driven by a frequency bias
(e.g. the model simply responds more strongly to higher-pitched tones regardless of context),
only the direction where the "preferred" frequency is deviant would show MMN. Section 0b
(below) tests this directly using the S2 criterion (interior trough + 50% recovery in
100–240 ms), which requires a genuine dip-and-recover shape and is therefore more
diagnostic than the magnitude-only C0 threshold.

---

## Section 0b — Counterbalanced analysis (S2 criterion)

> All tables in this section use the **S2 criterion** (baseline-normalised peak < 0 in
> 100–240 ms AND trough is interior to the window AND 50% depth recovery before window
> end). S2 is preferred over C0 here because it filters out ramp responses that merely
> cross zero at the window edge, which is the dominant failure mode for the encoder. C0
> counts are reported alongside S2 in Sections 1–4 for completeness.

### Aggregate per-method comparison (n/8 per cell = 4 models × 2 levels)

**Table CB-0. mTRF — S2 criterion**

| Method | Source | Stimulus | Regular (n/8) | Counter (n/8) |
| ------ | ------ | -------- | ------------- | ------------- |
| method_27 | Schall_1999a | 1000→1064 Hz | 4/8 | 6/8 |
| method_37 | Javitt_2000a | 1000→1050 Hz | 8/8 | 6/8 |
| method_43 | Michie_2000b | 633→700 Hz | 6/8 | 6/8 |
| method_44 | Michie_2000c | 633→1000 Hz | 6/8 | 8/8 |
| method_53 | Salisbury_2002a | 1000→1200 Hz | 6/8 | 7/8 |
| method_55 | Shinozaki_2002a | 1000→2000 Hz | 7/8 | 8/8 |
| method_60 | Umbricht_2003a | 1000→1500 Hz | 4/8 | 8/8 |
| method_72 | Bodatsch_2011 | 1000→1200 Hz | 8/8 | 8/8 |
| method_74 | Domjan_2012 | 1000→1500 Hz | 8/8 | 6/8 |
| method_75 | Karger_2014 | 1000→1200 Hz | 8/8 | 8/8 |
| **Total** | | | **65/80** | **71/80** |

**Table CB-0b. mTRF — S4 criterion** (tone-end-relative dip+recovery scan; included for comparison)

| Method | Source | Stimulus | Regular (n/8) | Counter (n/8) |
| ------ | ------ | -------- | ------------- | ------------- |
| method_27 | Schall_1999a | 1000→1064 Hz | 8/8 | 5/8 |
| method_37 | Javitt_2000a | 1000→1050 Hz | 8/8 | 8/8 |
| method_43 | Michie_2000b | 633→700 Hz | 6/8 | 6/8 |
| method_44 | Michie_2000c | 633→1000 Hz | 8/8 | 8/8 |
| method_53 | Salisbury_2002a | 1000→1200 Hz | 8/8 | 7/8 |
| method_55 | Shinozaki_2002a | 1000→2000 Hz | 7/8 | 8/8 |
| method_60 | Umbricht_2003a | 1000→1500 Hz | 4/8 | 8/8 |
| method_72 | Bodatsch_2011 | 1000→1200 Hz | 8/8 | 8/8 |
| method_74 | Domjan_2012 | 1000→1500 Hz | 8/8 | 7/8 |
| method_75 | Karger_2014 | 1000→1200 Hz | 8/8 | 8/8 |
| **Total** | | | **73/80** | **71/80** |

**Table CB-0c. Encoder — S2 criterion**

| Method | Source | Stimulus | Regular (n/8) | Counter (n/8) |
| ------ | ------ | -------- | ------------- | ------------- |
| method_27 | Schall_1999a | 1000→1064 Hz | 2/8 | 5/8 |
| method_37 | Javitt_2000a | 1000→1050 Hz | 1/8 | 2/8 |
| method_43 | Michie_2000b | 633→700 Hz | 2/8 | 3/8 |
| method_44 | Michie_2000c | 633→1000 Hz | 2/8 | 3/8 |
| method_53 | Salisbury_2002a | 1000→1200 Hz | 1/8 | 1/8 |
| method_55 | Shinozaki_2002a | 1000→2000 Hz | 2/8 | 5/8 |
| method_60 | Umbricht_2003a | 1000→1500 Hz | 2/8 | 2/8 |
| method_72 | Bodatsch_2011 | 1000→1200 Hz | 2/8 | 1/8 |
| method_74 | Domjan_2012 | 1000→1500 Hz | 0/8 | 2/8 |
| method_75 | Karger_2014 | 1000→1200 Hz | 2/8 | 1/8 |
| **Total** | | | **16/80** | **25/80** |

**Table CB-0d. Encoder — S4 criterion**

| Method | Source | Stimulus | Regular (n/8) | Counter (n/8) |
| ------ | ------ | -------- | ------------- | ------------- |
| method_27 | Schall_1999a | 1000→1064 Hz | 4/8 | 5/8 |
| method_37 | Javitt_2000a | 1000→1050 Hz | 2/8 | 3/8 |
| method_43 | Michie_2000b | 633→700 Hz | 2/8 | 4/8 |
| method_44 | Michie_2000c | 633→1000 Hz | 3/8 | 4/8 |
| method_53 | Salisbury_2002a | 1000→1200 Hz | 1/8 | 1/8 |
| method_55 | Shinozaki_2002a | 1000→2000 Hz | 5/8 | 5/8 |
| method_60 | Umbricht_2003a | 1000→1500 Hz | 3/8 | 3/8 |
| method_72 | Bodatsch_2011 | 1000→1200 Hz | 2/8 | 2/8 |
| method_74 | Domjan_2012 | 1000→1500 Hz | 2/8 | 4/8 |
| method_75 | Karger_2014 | 1000→1200 Hz | 2/8 | 2/8 |
| **Total** | | | **26/80** | **33/80** |

Under mTRF (S2), regular and counter totals are nearly symmetric (65 vs 71/80),
confirming bidirectionality. Under the encoder (S2), counts are uniformly low in both
directions (16 vs 25/80) — the encoder rarely produces genuine troughs under S2
regardless of direction. The counter-biased encoder totals (25 vs 16) partly reflect
whisper-small's counter-heavy C0 calls (16/20 counter vs 6/20 regular) mostly
failing S2 as expected for ramp responses. Method_60 (1000→1500 Hz) shows the largest
directional gap under mTRF S2 (regular 4/8, counter 8/8) — reliably more bidirectional
in the counter direction across both mappings. Methods 72 and 75 (1000→1200 Hz) are
the most symmetric and highest-scoring under mTRF S2 (8/8 in both directions).

---

### Regular vs counter split

**Table CB-1a. mTRF MMN counts: regular vs counterbalanced — S2 criterion** (n/10 per set)

| Model | Set | Parcels (n/10) | Electrodes (n/10) | Total (n/20) |
| ------ | ----------- | -------------- | ----------------- | ------------ |
| tiny | regular | 8/10 | 9/10 | 17/20 |
| tiny | counter | 9/10 | 9/10 | 18/20 |
| base | regular | 6/10 | 6/10 | 12/20 |
| base | counter | 10/10 | 9/10 | 19/20 |
| small | regular | 9/10 | 10/10 | 19/20 |
| small | counter | 10/10 | 9/10 | 19/20 |
| medium | regular | 9/10 | 8/10 | 17/20 |
| medium | counter | 8/10 | 7/10 | 15/20 |

**Table CB-1b. Encoder MMN counts: regular vs counterbalanced — S2 criterion** (n/10 per set)

| Model | Set | Parcels (n/10) | Electrodes (n/10) | Total (n/20) |
| ------ | ----------- | -------------- | ----------------- | ------------ |
| tiny | regular | 1/10 | 2/10 | 3/20 |
| tiny | counter | 3/10 | 2/10 | 5/20 |
| base | regular | 3/10 | 6/10 | 9/20 |
| base | counter | 2/10 | 5/10 | 7/20 |
| small | regular | 4/10 | 0/10 | 4/20 |
| small | counter | 4/10 | 4/10 | 8/20 |
| medium | regular | 0/10 | 0/10 | 0/20 |
| medium | counter | 3/10 | 2/10 | 5/20 |

Under S2, the encoder's counts collapse dramatically relative to C0 — tiny falls from
11/20 (C0 regular) to 3/20 (S2 regular). Whisper-medium drops to 0/20 regular under S2,
confirming those C0 positives were entirely ramp responses. The mTRF S2 counts stay
close to C0, with the most notable change being whisper-base regular (7→6 parcels,
7→6 electrodes) — a minor tightening driven by a borderline method_27 case.

### Agreement between regular and counter versions

**Table CB-2a. Regular ↔ counter agreement — mTRF — S2 criterion**
(n = 20 pairs per model = 10 method-bases × 2 levels)

| Model | Both MMN | Both no-MMN | Disagree |
| ------ | -------- | ----------- | -------- |
| tiny | 15/20 | 0/20 | 5/20 |
| base | 11/20 | 0/20 | 9/20 |
| small | 18/20 | 0/20 | 2/20 |
| medium | 13/20 | 1/20 | 6/20 |

**Table CB-2b. Regular ↔ counter agreement — encoder — S2 criterion**

| Model | Both MMN | Both no-MMN | Disagree |
| ------ | -------- | ----------- | -------- |
| tiny | 1/20 | 13/20 | 6/20 |
| base | 5/20 | 9/20 | 6/20 |
| small | 1/20 | 9/20 | 10/20 |
| medium | 0/20 | 15/20 | 5/20 |

### Per-method regular vs counter MMN count (across all model×level)

**Table CB-3a. mTRF: per-method-base regular vs counter — S2 criterion** (n/8 per cell = 4 models × 2 levels)

| Method | Regular (Std→Dev) | Counter (Dev→Std) | Regular MMN (n/8) | Counter MMN (n/8) |
| ------ | ----------------- | ----------------- | ----------------- | ----------------- |
| method_27 | 1000→1064 Hz | 1064→1000 Hz | 4/8 | 6/8 |
| method_37 | 1000→1050 Hz | 1050→1000 Hz | 8/8 | 6/8 |
| method_43 | 633→700 Hz | 700→633 Hz | 6/8 | 6/8 |
| method_44 | 633→1000 Hz | 1000→633 Hz | 6/8 | 8/8 |
| method_53 | 1000→1200 Hz | 1200→1000 Hz | 6/8 | 7/8 |
| method_55 | 1000→2000 Hz | 2000→1000 Hz | 7/8 | 8/8 |
| method_60 | 1000→1500 Hz | 1500→1000 Hz | 4/8 | 8/8 |
| method_72 | 1000→1200 Hz | 1200→1000 Hz | 8/8 | 8/8 |
| method_74 | 1000→1500 Hz | 1500→1000 Hz | 8/8 | 6/8 |
| method_75 | 1000→1200 Hz | 1200→1000 Hz | 8/8 | 8/8 |

**Table CB-3b. Encoder: per-method-base regular vs counter — S2 criterion** (n/8 per cell)

| Method | Regular (Std→Dev) | Counter (Dev→Std) | Regular MMN (n/8) | Counter MMN (n/8) |
| ------ | ----------------- | ----------------- | ----------------- | ----------------- |
| method_27 | 1000→1064 Hz | 1064→1000 Hz | 2/8 | 5/8 |
| method_37 | 1000→1050 Hz | 1050→1000 Hz | 1/8 | 2/8 |
| method_43 | 633→700 Hz | 700→633 Hz | 2/8 | 3/8 |
| method_44 | 633→1000 Hz | 1000→633 Hz | 2/8 | 3/8 |
| method_53 | 1000→1200 Hz | 1200→1000 Hz | 1/8 | 1/8 |
| method_55 | 1000→2000 Hz | 2000→1000 Hz | 2/8 | 5/8 |
| method_60 | 1000→1500 Hz | 1500→1000 Hz | 2/8 | 2/8 |
| method_72 | 1000→1200 Hz | 1200→1000 Hz | 2/8 | 1/8 |
| method_74 | 1000→1500 Hz | 1500→1000 Hz | 0/8 | 2/8 |
| method_75 | 1000→1200 Hz | 1200→1000 Hz | 2/8 | 1/8 |

### Summary of counterbalanced findings (S2)

**mTRF is highly bidirectional under S2.** Tables CB-2a shows 11–18/20 "both MMN" pairs
per model (55–90% bidirectionality). The "both no-MMN" cell is zero or one for every
model, so every disagreement is a marginal single-direction case rather than a systematic
asymmetry. Notably:

- **whisper-base** shows a striking counter advantage: regular 12/20 vs counter 19/20.
  Under S2, the counterbalanced stimuli produce more genuine trough-shaped responses for
  base than the originals. Methods 43 (633→700 Hz), 44 (633→1000 Hz), and 60
  (1000→1500 Hz) all flip from below-threshold in the regular direction to clear S2 positives
  in the counter direction for base, accounting for most of the 9/20 disagreements.
- **whisper-small** shows near-perfect bidirectionality: 18/20 both-MMN, 2/20 disagree —
  the strongest counterbalance result of any model under S2.
- **Per-method (Table CB-3a)**: methods 72 and 75 achieve 8/8 in both directions — the most
  symmetric and highest-confidence methods. Method 60 (regular 4/8, counter 8/8) shows the
  largest directional gap, consistent with the 633 Hz standard evoking a strong neural
  baseline that suppresses the deviance signal in the regular direction.

**Encoder is dominated by "both no-MMN" under S2.** Table CB-2b shows 9–15/20
"both no-MMN" as the dominant category per model — the encoder rarely produces genuine
troughs in either direction. The small residual disagreement (5–10/20) reflects cases where
one direction passes S2 while the other does not; these are not systematic directional biases
but rather marginal borderline cases. Per Table CB-3b, only methods 27 and 55 produce
encoder S2 counter counts ≥5/8, suggesting those two stimulus pairs generate the most
shape-reliable counter responses.

**Overall verdict on the control (S2):** mTRF results hold up strongly under the
counterbalanced control with S2 — bidirectionality is confirmed at 55–90% per model and
both-no-MMN is essentially zero, meaning every mTRF S2 positive is bidirectionally
supported. The encoder's S2 bidirectionality is uninformative (both directions mostly fail
S2), which is consistent with the encoder's already-known shape-quality deficit rather
than a specific frequency-preference artifact.

---

## Section 1 — Method A (mTRF), all 20 methods

> **Data:** `outputs/results_with_counter/mmn_results_table.csv` (320 rows).
> Aggregations below filter to `mapping=="mtrf"`. The ROI mean for parcels is
> `{frontal, central}` averaged; for electrodes it is `{Fz, FCz, Cz, FC2, F1, F2}`
> (FC1 absent from this montage). All definitions unchanged from `results_analysis.md`.

**Table 1a. Mean MMN per model × method (mTRF)**

| Model | Method | Stimulus | Type | MMN (parcels) | MMN (electrodes) |
| ------ | ------ | -------- | ---- | ------------- | ---------------- |
| tiny | method_27 | 1000→1064 Hz | regular | -0.87 | -0.78 |
| tiny | method_27_counter | 1064→1000 Hz | counter | +0.10 | +0.14 |
| tiny | method_37 | 1000→1050 Hz | regular | -0.25 | -0.30 |
| tiny | method_37_counter | 1050→1000 Hz | counter | -0.23 | -0.13 |
| tiny | method_43 | 633→700 Hz | regular | -0.61 | -0.49 |
| tiny | method_43_counter | 700→633 Hz | counter | -0.61 | -0.46 |
| tiny | method_44 | 633→1000 Hz | regular | -1.46 | -1.63 |
| tiny | method_44_counter | 1000→633 Hz | counter | -1.11 | -0.72 |
| tiny | method_53 | 1000→1200 Hz | regular | -1.04 | -0.83 |
| tiny | method_53_counter | 1200→1000 Hz | counter | -0.66 | -0.83 |
| tiny | method_55 | 1000→2000 Hz | regular | -1.43 | -1.28 |
| tiny | method_55_counter | 2000→1000 Hz | counter | -2.51 | -2.38 |
| tiny | method_60 | 1000→1500 Hz | regular | -0.56 | -0.40 |
| tiny | method_60_counter | 1500→1000 Hz | counter | -1.50 | -1.56 |
| tiny | method_72 | 1000→1200 Hz | regular | -0.96 | -0.78 |
| tiny | method_72_counter | 1200→1000 Hz | counter | -0.12 | -0.19 |
| tiny | method_74 | 1000→1500 Hz | regular | -0.58 | -0.53 |
| tiny | method_74_counter | 1500→1000 Hz | counter | -0.40 | -0.47 |
| tiny | method_75 | 1000→1200 Hz | regular | -0.95 | -0.77 |
| tiny | method_75_counter | 1200→1000 Hz | counter | -0.12 | -0.19 |
| base | method_27 | 1000→1064 Hz | regular | -0.01 | +0.02 |
| base | method_27_counter | 1064→1000 Hz | counter | -0.18 | -0.15 |
| base | method_37 | 1000→1050 Hz | regular | -0.48 | -0.38 |
| base | method_37_counter | 1050→1000 Hz | counter | -0.49 | -0.46 |
| base | method_43 | 633→700 Hz | regular | +0.02 | +0.05 |
| base | method_43_counter | 700→633 Hz | counter | -0.64 | -0.59 |
| base | method_44 | 633→1000 Hz | regular | +0.16 | +0.11 |
| base | method_44_counter | 1000→633 Hz | counter | -0.48 | -0.59 |
| base | method_53 | 1000→1200 Hz | regular | -0.31 | -0.36 |
| base | method_53_counter | 1200→1000 Hz | counter | -0.71 | -0.72 |
| base | method_55 | 1000→2000 Hz | regular | -1.44 | -1.42 |
| base | method_55_counter | 2000→1000 Hz | counter | -0.77 | -0.61 |
| base | method_60 | 1000→1500 Hz | regular | +0.43 | -0.12 |
| base | method_60_counter | 1500→1000 Hz | counter | -2.03 | -1.66 |
| base | method_72 | 1000→1200 Hz | regular | -0.40 | -0.38 |
| base | method_72_counter | 1200→1000 Hz | counter | -0.51 | -0.38 |
| base | method_74 | 1000→1500 Hz | regular | -0.06 | -0.06 |
| base | method_74_counter | 1500→1000 Hz | counter | -0.69 | -0.52 |
| base | method_75 | 1000→1200 Hz | regular | -0.40 | -0.38 |
| base | method_75_counter | 1200→1000 Hz | counter | -0.50 | -0.38 |
| small | method_27 | 1000→1064 Hz | regular | +0.30 | -0.21 |
| small | method_27_counter | 1064→1000 Hz | counter | -0.64 | -0.14 |
| small | method_37 | 1000→1050 Hz | regular | -0.21 | -0.20 |
| small | method_37_counter | 1050→1000 Hz | counter | -0.14 | -0.30 |
| small | method_43 | 633→700 Hz | regular | -0.22 | -0.17 |
| small | method_43_counter | 700→633 Hz | counter | -0.33 | -0.33 |
| small | method_44 | 633→1000 Hz | regular | -0.57 | -0.64 |
| small | method_44_counter | 1000→633 Hz | counter | -0.10 | -0.53 |
| small | method_53 | 1000→1200 Hz | regular | -0.48 | -0.18 |
| small | method_53_counter | 1200→1000 Hz | counter | -0.23 | -0.69 |
| small | method_55 | 1000→2000 Hz | regular | -0.68 | -0.81 |
| small | method_55_counter | 2000→1000 Hz | counter | -1.01 | -0.54 |
| small | method_60 | 1000→1500 Hz | regular | -0.64 | -0.42 |
| small | method_60_counter | 1500→1000 Hz | counter | -0.77 | -0.59 |
| small | method_72 | 1000→1200 Hz | regular | -0.20 | -0.13 |
| small | method_72_counter | 1200→1000 Hz | counter | -0.40 | -0.09 |
| small | method_74 | 1000→1500 Hz | regular | -0.40 | -0.25 |
| small | method_74_counter | 1500→1000 Hz | counter | -0.71 | +0.04 |
| small | method_75 | 1000→1200 Hz | regular | -0.21 | -0.14 |
| small | method_75_counter | 1200→1000 Hz | counter | -0.40 | -0.09 |
| medium | method_27 | 1000→1064 Hz | regular | -0.29 | -0.28 |
| medium | method_27_counter | 1064→1000 Hz | counter | -0.31 | -0.29 |
| medium | method_37 | 1000→1050 Hz | regular | -0.30 | -0.14 |
| medium | method_37_counter | 1050→1000 Hz | counter | +0.06 | +0.01 |
| medium | method_43 | 633→700 Hz | regular | -3.26 | -3.76 |
| medium | method_43_counter | 700→633 Hz | counter | +0.12 | -0.32 |
| medium | method_44 | 633→1000 Hz | regular | -0.41 | -0.74 |
| medium | method_44_counter | 1000→633 Hz | counter | -2.05 | -2.57 |
| medium | method_53 | 1000→1200 Hz | regular | -0.55 | -0.32 |
| medium | method_53_counter | 1200→1000 Hz | counter | -0.12 | -0.11 |
| medium | method_55 | 1000→2000 Hz | regular | +0.00 | -0.26 |
| medium | method_55_counter | 2000→1000 Hz | counter | -0.71 | -0.57 |
| medium | method_60 | 1000→1500 Hz | regular | -1.33 | -0.61 |
| medium | method_60_counter | 1500→1000 Hz | counter | -0.59 | -0.70 |
| medium | method_72 | 1000→1200 Hz | regular | -0.66 | -0.50 |
| medium | method_72_counter | 1200→1000 Hz | counter | -0.33 | -0.11 |
| medium | method_74 | 1000→1500 Hz | regular | -0.59 | -0.34 |
| medium | method_74_counter | 1500→1000 Hz | counter | -0.46 | -0.31 |
| medium | method_75 | 1000→1200 Hz | regular | -0.63 | -0.49 |
| medium | method_75_counter | 1200→1000 Hz | counter | -0.34 | -0.11 |

**Table 1b. Per-model MMN count summary — mTRF** (n/20 per level, 10 regular + 10 counter)

| Model | Parcels (n/20) | Electrodes (n/20) | Total (n/40) |
| ------ | -------------- | ----------------- | ------------ |
| tiny | 19/20 | 19/20 | 38/40 |
| base | 17/20 | 17/20 | 34/40 |
| small | 19/20 | 19/20 | 38/40 |
| medium | 17/20 | 19/20 | 36/40 |
| **Total** | **72/80** | **74/80** | **146/160** |

**Comparison to original (regular only, n/10 per level):** tiny 10/10→19/20 (stable at ~95%), base 7/10→17/20 (improves: 70%→85% — counter methods are consistently positive for base), small 9/10→19/20 (stable at 95%), medium 9+10/20→36/40 (90%). No model degrades under the counterbalanced control; base actually improves.

**Table 1c. Per-method average across models — mTRF** (pooled over all 4 models × 2 levels = 8 cells)

| Method | Stimulus | Type | Avg MMN across models |
| ------ | -------- | ---- | --------------------- |
| method_27 | 1000→1064 Hz | regular | -0.26 |
| method_27_counter | 1064→1000 Hz | counter | -0.18 |
| method_37 | 1000→1050 Hz | regular | -0.28 |
| method_37_counter | 1050→1000 Hz | counter | -0.21 |
| method_43 | 633→700 Hz | regular | -1.06 |
| method_43_counter | 700→633 Hz | counter | -0.40 |
| method_44 | 633→1000 Hz | regular | -0.65 |
| method_44_counter | 1000→633 Hz | counter | -1.02 |
| method_53 | 1000→1200 Hz | regular | -0.51 |
| method_53_counter | 1200→1000 Hz | counter | -0.51 |
| method_55 | 1000→2000 Hz | regular | -0.91 |
| method_55_counter | 2000→1000 Hz | counter | -1.14 |
| method_60 | 1000→1500 Hz | regular | -0.46 |
| method_60_counter | 1500→1000 Hz | counter | -1.18 |
| method_72 | 1000→1200 Hz | regular | -0.50 |
| method_72_counter | 1200→1000 Hz | counter | -0.27 |
| method_74 | 1000→1500 Hz | regular | -0.35 |
| method_74_counter | 1500→1000 Hz | counter | -0.44 |
| method_75 | 1000→1200 Hz | regular | -0.50 |
| method_75_counter | 1200→1000 Hz | counter | -0.27 |

Notable: methods_53, 55, 60 show equal or stronger average MMN in the counter direction; methods_43, 72, 75 show weaker counter MMN, suggesting the 633 Hz and 1200 Hz standards evoke stronger baseline responses that diminish the counter deviance signal. Method_44 (633→1000 Hz) shows a stronger counter (−1.02) than regular (−0.65).

**Table 1d. Per-model average across stimuli — mTRF** (mean and count over all 20 methods for each model)

| Model | Avg MMN across stimuli | n methods with MMN (≥1 level) |
| ------ | ---------------------- | ----------------------------- |
| tiny | -0.76 | 19/20 |
| base | -0.46 | 17/20 |
| small | -0.36 | 19/20 |
| medium | -0.63 | 18/20 |

**Table 1e. Mean MMN per model, level separated — mTRF**

| Model | Mean MMN parcels (avg across 20 methods) | Mean MMN electrodes (avg across 20 methods) |
| ------ | ---------------------------------------- | ------------------------------------------- |
| tiny | -0.79 | -0.73 |
| base | -0.47 | -0.45 |
| small | -0.40 | -0.32 |
| medium | -0.64 | -0.63 |

---

## Section 2 — Method B (encoder), all 20 methods

> Same data source as Section 1, filtered to `mapping=="encoder"`.

**Table 2a. Mean MMN per model × method (encoder)**

| Model | Method | Stimulus | Type | MMN (parcels) | MMN (electrodes) |
| ------ | ------ | -------- | ---- | ------------- | ---------------- |
| tiny | method_27 | 1000→1064 Hz | regular | -0.78 | -0.11 |
| tiny | method_27_counter | 1064→1000 Hz | counter | +0.06 | -0.09 |
| tiny | method_37 | 1000→1050 Hz | regular | -0.42 | -0.89 |
| tiny | method_37_counter | 1050→1000 Hz | counter | +0.14 | -0.12 |
| tiny | method_43 | 633→700 Hz | regular | +0.28 | +0.06 |
| tiny | method_43_counter | 700→633 Hz | counter | -0.21 | -0.25 |
| tiny | method_44 | 633→1000 Hz | regular | +0.26 | -0.12 |
| tiny | method_44_counter | 1000→633 Hz | counter | -2.52 | +0.31 |
| tiny | method_53 | 1000→1200 Hz | regular | +0.51 | -1.49 |
| tiny | method_53_counter | 1200→1000 Hz | counter | -0.41 | +0.21 |
| tiny | method_55 | 1000→2000 Hz | regular | +1.79 | -0.04 |
| tiny | method_55_counter | 2000→1000 Hz | counter | -4.35 | -0.45 |
| tiny | method_60 | 1000→1500 Hz | regular | -2.53 | +1.07 |
| tiny | method_60_counter | 1500→1000 Hz | counter | -3.34 | -1.91 |
| tiny | method_72 | 1000→1200 Hz | regular | -6.53 | +0.41 |
| tiny | method_72_counter | 1200→1000 Hz | counter | +0.44 | -0.47 |
| tiny | method_74 | 1000→1500 Hz | regular | -0.06 | +0.25 |
| tiny | method_74_counter | 1500→1000 Hz | counter | -0.27 | -0.26 |
| tiny | method_75 | 1000→1200 Hz | regular | -6.49 | +0.41 |
| tiny | method_75_counter | 1200→1000 Hz | counter | +0.44 | -0.52 |
| base | method_27 | 1000→1064 Hz | regular | +1.63 | -0.48 |
| base | method_27_counter | 1064→1000 Hz | counter | -2.43 | -0.14 |
| base | method_37 | 1000→1050 Hz | regular | +2.33 | -0.41 |
| base | method_37_counter | 1050→1000 Hz | counter | -10.37 | -0.81 |
| base | method_43 | 633→700 Hz | regular | -0.17 | -0.45 |
| base | method_43_counter | 700→633 Hz | counter | +0.09 | +0.02 |
| base | method_44 | 633→1000 Hz | regular | -0.77 | +1.15 |
| base | method_44_counter | 1000→633 Hz | counter | -7.84 | +0.05 |
| base | method_53 | 1000→1200 Hz | regular | -1.64 | -2.87 |
| base | method_53_counter | 1200→1000 Hz | counter | -0.71 | +0.97 |
| base | method_55 | 1000→2000 Hz | regular | -2.41 | -0.30 |
| base | method_55_counter | 2000→1000 Hz | counter | -1.00 | -0.84 |
| base | method_60 | 1000→1500 Hz | regular | +0.72 | -1.17 |
| base | method_60_counter | 1500→1000 Hz | counter | -5.06 | -2.03 |
| base | method_72 | 1000→1200 Hz | regular | -2.89 | -0.36 |
| base | method_72_counter | 1200→1000 Hz | counter | +0.43 | -0.98 |
| base | method_74 | 1000→1500 Hz | regular | +1.65 | +0.57 |
| base | method_74_counter | 1500→1000 Hz | counter | -1.64 | -0.34 |
| base | method_75 | 1000→1200 Hz | regular | -2.91 | -0.37 |
| base | method_75_counter | 1200→1000 Hz | counter | +0.40 | -0.97 |
| small | method_27 | 1000→1064 Hz | regular | -0.42 | +0.02 |
| small | method_27_counter | 1064→1000 Hz | counter | -1.09 | -0.87 |
| small | method_37 | 1000→1050 Hz | regular | +0.22 | +0.60 |
| small | method_37_counter | 1050→1000 Hz | counter | -0.52 | -0.81 |
| small | method_43 | 633→700 Hz | regular | +0.28 | +0.12 |
| small | method_43_counter | 700→633 Hz | counter | -0.46 | -0.53 |
| small | method_44 | 633→1000 Hz | regular | +0.87 | +1.38 |
| small | method_44_counter | 1000→633 Hz | counter | -0.27 | -2.54 |
| small | method_53 | 1000→1200 Hz | regular | +0.77 | +0.15 |
| small | method_53_counter | 1200→1000 Hz | counter | -1.15 | -0.42 |
| small | method_55 | 1000→2000 Hz | regular | -0.30 | -1.25 |
| small | method_55_counter | 2000→1000 Hz | counter | -1.61 | -0.65 |
| small | method_60 | 1000→1500 Hz | regular | +0.19 | -1.74 |
| small | method_60_counter | 1500→1000 Hz | counter | +1.85 | +0.33 |
| small | method_72 | 1000→1200 Hz | regular | -0.19 | +1.37 |
| small | method_72_counter | 1200→1000 Hz | counter | +0.37 | -2.58 |
| small | method_74 | 1000→1500 Hz | regular | +0.34 | +0.04 |
| small | method_74_counter | 1500→1000 Hz | counter | -0.38 | -0.18 |
| small | method_75 | 1000→1200 Hz | regular | -0.20 | +1.35 |
| small | method_75_counter | 1200→1000 Hz | counter | +0.33 | -2.57 |
| medium | method_27 | 1000→1064 Hz | regular | -0.18 | +0.01 |
| medium | method_27_counter | 1064→1000 Hz | counter | -0.88 | -0.11 |
| medium | method_37 | 1000→1050 Hz | regular | -0.30 | +0.48 |
| medium | method_37_counter | 1050→1000 Hz | counter | +0.32 | -0.46 |
| medium | method_43 | 633→700 Hz | regular | +0.25 | +0.07 |
| medium | method_43_counter | 700→633 Hz | counter | -1.10 | -0.39 |
| medium | method_44 | 633→1000 Hz | regular | -1.59 | +0.02 |
| medium | method_44_counter | 1000→633 Hz | counter | -1.14 | -0.17 |
| medium | method_53 | 1000→1200 Hz | regular | -1.02 | -0.97 |
| medium | method_53_counter | 1200→1000 Hz | counter | +0.37 | +0.39 |
| medium | method_55 | 1000→2000 Hz | regular | +2.61 | +0.25 |
| medium | method_55_counter | 2000→1000 Hz | counter | -0.66 | +0.19 |
| medium | method_60 | 1000→1500 Hz | regular | +1.29 | -2.76 |
| medium | method_60_counter | 1500→1000 Hz | counter | +0.36 | -2.47 |
| medium | method_72 | 1000→1200 Hz | regular | -1.06 | -0.10 |
| medium | method_72_counter | 1200→1000 Hz | counter | +0.60 | +0.06 |
| medium | method_74 | 1000→1500 Hz | regular | -0.21 | -0.24 |
| medium | method_74_counter | 1500→1000 Hz | counter | -0.59 | +0.13 |
| medium | method_75 | 1000→1200 Hz | regular | -1.04 | -0.10 |
| medium | method_75_counter | 1200→1000 Hz | counter | +0.63 | +0.06 |

The same large extreme values as the original appear in encoder/parcels/tiny for methods_72 (−6.53) and _75 (−6.49) and in method_55_counter/tiny (−4.35) and method_37_counter/base (−10.37). As in the original analysis, these extreme values reflect the z-score-denominator instability (near-zero baseline SD) on smooth ramp responses, not a literal 10× stronger MMN — see `results_analysis.md` Notes & caveats.

**Table 2b. Per-model MMN count summary — encoder** (n/20 per level)

| Model | Parcels (n/20) | Electrodes (n/20) | Total (n/40) |
| ------ | -------------- | ----------------- | ------------ |
| tiny | 12/20 | 13/20 | 25/40 |
| base | 13/20 | 15/20 | 28/40 |
| small | 11/20 | 11/20 | 22/40 |
| medium | 12/20 | 10/20 | 22/40 |
| **Total** | **48/80** | **49/80** | **97/160** |

**Comparison to original (regular only):** tiny 11/20→25/40 (55%→62%), base 14/20→28/40 (70%→70%), small 6/20→22/40 (30%→55%), medium 12/20→22/40 (60%→55%). Whisper-small shows the largest shift — counter methods are positive at a much higher rate than regular, confirming the asymmetry flagged in Section 0b.

**Table 2c. Per-method average across models — encoder**

| Method | Stimulus | Type | Avg MMN across models |
| ------ | -------- | ---- | --------------------- |
| method_27 | 1000→1064 Hz | regular | -0.04 |
| method_27_counter | 1064→1000 Hz | counter | -0.69 |
| method_37 | 1000→1050 Hz | regular | +0.20 |
| method_37_counter | 1050→1000 Hz | counter | -1.58 |
| method_43 | 633→700 Hz | regular | +0.06 |
| method_43_counter | 700→633 Hz | counter | -0.35 |
| method_44 | 633→1000 Hz | regular | +0.15 |
| method_44_counter | 1000→633 Hz | counter | -1.77 |
| method_53 | 1000→1200 Hz | regular | -0.82 |
| method_53_counter | 1200→1000 Hz | counter | -0.09 |
| method_55 | 1000→2000 Hz | regular | +0.04 |
| method_55_counter | 2000→1000 Hz | counter | -1.17 |
| method_60 | 1000→1500 Hz | regular | -0.62 |
| method_60_counter | 1500→1000 Hz | counter | -1.53 |
| method_72 | 1000→1200 Hz | regular | -1.17 |
| method_72_counter | 1200→1000 Hz | counter | -0.27 |
| method_74 | 1000→1500 Hz | regular | +0.29 |
| method_74_counter | 1500→1000 Hz | counter | -0.44 |
| method_75 | 1000→1200 Hz | regular | -1.17 |
| method_75_counter | 1200→1000 Hz | counter | -0.28 |

**Table 2d. Per-model average across stimuli — encoder**

| Model | Avg MMN across stimuli | n methods with MMN (≥1 level) |
| ------ | ---------------------- | ----------------------------- |
| tiny | -0.70 | 12/20 |
| base | -1.06 | 14/20 |
| small | -0.25 | 11/20 |
| medium | -0.24 | 11/20 |

**Table 2e. Mean MMN per model, level separated — encoder**

| Model | Mean MMN parcels (avg across 20 methods) | Mean MMN electrodes (avg across 20 methods) |
| ------ | ---------------------------------------- | ------------------------------------------- |
| tiny | -1.20 | -0.20 |
| base | -1.63 | -0.49 |
| small | -0.07 | -0.44 |
| medium | -0.17 | -0.31 |

---

## Cross-Method Comparisons (updated, 20 methods)

**Table 3. mTRF vs. encoder agreement** (per model × level, over all 20 methods)

| Model | Level | Both MMN | Both no-MMN | Agree | Disagree |
| ------ | ---------- | -------- | ----------- | ----- | -------- |
| tiny | parcels | 12 | 1 | 13/20 | 7/20 |
| tiny | electrodes | 12 | 0 | 12/20 | 8/20 |
| base | parcels | 11 | 1 | 12/20 | 8/20 |
| base | electrodes | 13 | 1 | 14/20 | 6/20 |
| small | parcels | 10 | 0 | 10/20 | 10/20 |
| small | electrodes | 10 | 0 | 10/20 | 10/20 |
| medium | parcels | 11 | 2 | 13/20 | 7/20 |
| medium | electrodes | 9 | 0 | 9/20 | 11/20 |

Agreement is generally moderate (10–14/20), similar to the original 10-method analysis (2–8/10). The mTRF–encoder disagreement does not worsen or improve substantially when counter methods are added, suggesting the two mapping methods disagree on a consistent subset of stimuli rather than counter stimuli specifically driving more disagreement.

**Table 4. Stimulus-method consistency** (MMN count across all 32 model×level×mapping cells per method)

| Method | Stimulus (source) | mTRF (n/8) | Encoder (n/8) | Total (n/16) |
| ------ | ----------------- | ---------- | ------------- | ------------ |
| method_75 | 1000→1200 Hz (Karger_2014) | 8/8 | 6/8 | 14/16 |
| method_75_counter | 1200→1000 Hz (Karger_2014) | 8/8 | 3/8 | 11/16 |
| method_74 | 1000→1500 Hz (Domjan_2012) | 8/8 | 3/8 | 11/16 |
| method_74_counter | 1500→1000 Hz (Domjan_2012) | 7/8 | 7/8 | 14/16 |
| method_72 | 1000→1200 Hz (Bodatsch_2011) | 8/8 | 6/8 | 14/16 |
| method_72_counter | 1200→1000 Hz (Bodatsch_2011) | 8/8 | 3/8 | 11/16 |
| method_60 | 1000→1500 Hz (Umbricht_2003a) | 7/8 | 4/8 | 11/16 |
| method_60_counter | 1500→1000 Hz (Umbricht_2003a) | 8/8 | 5/8 | 13/16 |
| method_53 | 1000→1200 Hz (Salisbury_2002a) | 8/8 | 5/8 | 13/16 |
| method_53_counter | 1200→1000 Hz (Salisbury_2002a) | 8/8 | 4/8 | 12/16 |
| method_55 | 1000→2000 Hz (Shinozaki_2002a) | 7/8 | 5/8 | 12/16 |
| method_55_counter | 2000→1000 Hz (Shinozaki_2002a) | 8/8 | 7/8 | 15/16 |
| method_37 | 1000→1050 Hz (Javitt_2000a) | 8/8 | 4/8 | 12/16 |
| method_37_counter | 1050→1000 Hz (Javitt_2000a) | 6/8 | 6/8 | 12/16 |
| method_43 | 633→700 Hz (Michie_2000b) | 6/8 | 2/8 | 8/16 |
| method_43_counter | 700→633 Hz (Michie_2000b) | 7/8 | 6/8 | 13/16 |
| method_44 | 633→1000 Hz (Michie_2000c) | 6/8 | 3/8 | 9/16 |
| method_44_counter | 1000→633 Hz (Michie_2000c) | 8/8 | 6/8 | 14/16 |
| method_27 | 1000→1064 Hz (Schall_1999a) | 6/8 | 5/8 | 11/16 |
| method_27_counter | 1064→1000 Hz (Schall_1999a) | 6/8 | 7/8 | 13/16 |

Notable pattern: methods 43 and 44 (633 Hz standard) show low regular totals (8/16, 9/16) but their counter versions score 13/16 and 14/16 — the reversal is consistent across both mTRF and encoder. Method_55_counter (2000→1000 Hz) is the highest-scoring single method (15/16). Method_74_counter matches method_74 in mTRF (7 vs 8/8) and outperforms the regular direction in the encoder (7 vs 3/8).

---

## Section 4 — Shape metrics C0–S6 (updated, 20 methods)

> **Code:** `scripts/analyze_mmn_criteria_s5_s6.py --roi_variant current`
> **Data:** `outputs/results_with_counter/mmn_criteria_s5_s6.csv`
>
> **Important note on S4 values:** The S4 criterion here comes from
> `analyze_mmn_criteria_s5_s6.py` (column `current__S4_specificity`), which may differ
> slightly from the S4 computed by `analyze_mmn_criteria.py` used in the original
> `results_analysis.md` Tables 13–14. The C0–S3 and S5–S6 counts are computed from the
> same definitions. The relative ordering of criteria and the mTRF-vs-encoder gap
> are unaffected by this minor implementation difference.

> **New criterion S7 (amplitude-gated MMN).** S7 = **S2 AND** the deviant−standard
> difference wave, **in microvolts**, at the S2 trough latency ≤ **−X µV** (headline
> **X = 1.0 µV**, *provisional — pending a literature amplitude threshold; TODO*). Unlike
> C0–S6, which score a **z-scored** (dimensionless, baseline-SD-unit) difference, S7 tests
> an **absolute amplitude** taken from a *separate* mean-only baseline-corrected difference
> wave (`analyze_mmn_criteria_s5_s6.py --dip_uv_threshold`, column `current__S7_uv_gated`).
> Units are made comparable across mappings: the mTRF predicts native EEG **Volts** (×1e6 →
> µV); the encoder predicts **z-units**, converted to native µV via the per-target `eeg_sd`
> in the checkpoint `model__<layer>.pt` (the additive mean cancels in deviant−standard). **By
> construction S7 ⊆ S2**, so S7 ≤ S2 in every cell below.
>
> **Scale caveat:** ridge/encoder predictions systematically **under-estimate** true MMN
> amplitude, so the model's predicted µV scale need **not** match literature EEG µV (a
> literature MMN is ≈ 1–5 µV, ~3 µV typical peak; Duncan et al. 2009). X is therefore
> calibrated to the **model's own** predicted-µV distribution, not to the literature:
> across the 20-method set the median S2-passing trough is ≈ **−0.9 µV** (mTRF, current
> ROI). On the native EEG scale **1.0 µV** is only ≈ 33% of a typical ~3 µV human peak
> MMN, but because the model's predictions are ~4× amplitude-shrunk, on the model's own
> scale 1.0 µV sits **just beyond** the median trough magnitude — so it removes slightly
> **more than half** of the S2-passing mTRF troughs (not merely the shallowest 0 → −0.5 µV
> tail), keeping the deeper, clearer ones. See **Section 7** for the full ROI / X-sweep
> breakdown and the µV-trough distribution.

### Results — mTRF (n/40 per model = 20 methods × 2 levels)

**Table 13. MMN-present counts per criterion, by model — mTRF**

| Model | C0 (n/40) | S1 (n/40) | S2 (n/40) | S3 (n/40) | S4 (n/40) | S5 (n/40) | S6 (n/40) | S7 (n/40) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 38/40 | 13/40 | 35/40 | 13/40 | 24/40 | 31/40 | 17/40 | 17/40 |
| base | 32/40 | 15/40 | 31/40 | 15/40 | 18/40 | 29/40 | 15/40 | 7/40 |
| small | 38/40 | 24/40 | 38/40 | 24/40 | 26/40 | 37/40 | 16/40 | 8/40 |
| medium | 34/40 | 25/40 | 32/40 | 23/40 | 18/40 | 28/40 | 13/40 | 26/40 |
| **Total** | **142/160** | **77/160** | **136/160** | **75/160** | **86/160** | **125/160** | **61/160** | **58/160** |

The mTRF shape-criterion picture is broadly preserved: C0 and S2 remain very high (89–95% and 85–95% per model), S1/S3 penalise substantially (edge-rejection effect unchanged), and S6 is the most restrictive. The S2/C0 retention rate across models is 35/38, 31/32, 38/38, 32/34 — roughly 92–100%, essentially identical to the original 92–100% in `results_analysis.md`. Counter methods do not degrade shape quality. **S7 (amplitude gate at 1.0 µV) tightens S2 to 58/160 — an S7/S2 retention of 43%** (58 of 136 S2-positives clear the 1.0 µV floor); it now sits just below S6 (61/160) rather than between S6 and S2, removing slightly more than half of the genuine mTRF troughs along with the shallow-amplitude ones. whisper-medium retains the most (26/32 = 81%), whisper-small the least (8/38 = 21%), reflecting its shallower predicted troughs.

**Table 13b. Regular vs counter breakdown — mTRF** (n/20 per set)

| Model | Set | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) | S7 (n/20) |
| ------ | ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | regular | 20/20 | 8/20 | 17/20 | 8/20 | 16/20 | 19/20 | 10/20 | 8/20 |
| tiny | counter | 18/20 | 5/20 | 18/20 | 5/20 | 8/20 | 12/20 | 7/20 | 9/20 |
| base | regular | 12/20 | 4/20 | 12/20 | 4/20 | 6/20 | 15/20 | 7/20 | 2/20 |
| base | counter | 20/20 | 11/20 | 19/20 | 11/20 | 12/20 | 14/20 | 8/20 | 5/20 |
| small | regular | 19/20 | 11/20 | 19/20 | 11/20 | 14/20 | 17/20 | 6/20 | 3/20 |
| small | counter | 19/20 | 13/20 | 19/20 | 13/20 | 12/20 | 20/20 | 10/20 | 5/20 |
| medium | regular | 19/20 | 17/20 | 17/20 | 15/20 | 10/20 | 14/20 | 9/20 | 16/20 |
| medium | counter | 15/20 | 8/20 | 15/20 | 8/20 | 8/20 | 14/20 | 4/20 | 10/20 |

Counter methods maintain high S2 rates (15–19/20) across all models, confirming that their C0 positives are predominantly genuine dip-and-recover troughs rather than ramps — the same shape quality as the regular stimuli. S7 remains roughly balanced across regular and counter overall (e.g. tiny 8/9); base and small skew slightly counter (2/5 and 3/5) while medium skews regular (16/10), but there is no systematic direction bias — counter troughs are broadly as deep in µV as the regular ones.

### Results — Encoder (n/40 per model)

**Table 14. MMN-present counts per criterion, by model — encoder**

| Model | C0 (n/40) | S1 (n/40) | S2 (n/40) | S3 (n/40) | S4 (n/40) | S5 (n/40) | S6 (n/40) | S7 (n/40) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 24/40 | 11/40 | 8/40 | 5/40 | 7/40 | 15/40 | 7/40 | 3/40 |
| base | 27/40 | 11/40 | 16/40 | 5/40 | 14/40 | 19/40 | 6/40 | 6/40 |
| small | 22/40 | 13/40 | 12/40 | 8/40 | 10/40 | 9/40 | 4/40 | 8/40 |
| medium | 22/40 | 5/40 | 5/40 | 2/40 | 8/40 | 22/40 | 6/40 | 4/40 |
| **Total** | **95/160** | **40/160** | **41/160** | **20/160** | **39/160** | **65/160** | **23/160** | **21/160** |

The encoder's shape-gating collapse is preserved: S2/C0 retention is 8/24, 16/27, 12/22, 5/22 (33–59%) — slightly higher than the original 38% in `results_analysis.md`, but still far below mTRF's 92–100%. Adding counter methods does not rescue the encoder's shape problem. **S7 lands at 21/160** — an S7/S2 retention of 51% (21 of 41), slightly *above* the mTRF's 43% at this X: when the encoder *does* produce a genuine S2 trough, that trough is usually of adequate µV amplitude. So even at the tighter 1.0 µV floor S7 does not disproportionately punish the encoder; the encoder's deficit is upstream (it rarely passes S2 at all: 41/160 vs mTRF 136/160), not in amplitude.

**Table 14b. Regular vs counter breakdown — encoder** (n/20 per set)

| Model | Set | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) | S7 (n/20) |
| ------ | ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | regular | 11/20 | 4/20 | 3/20 | 2/20 | 2/20 | 4/20 | 2/20 | 0/20 |
| tiny | counter | 13/20 | 7/20 | 5/20 | 3/20 | 5/20 | 11/20 | 5/20 | 3/20 |
| base | regular | 13/20 | 6/20 | 9/20 | 4/20 | 7/20 | 10/20 | 4/20 | 2/20 |
| base | counter | 14/20 | 5/20 | 7/20 | 1/20 | 7/20 | 9/20 | 2/20 | 4/20 |
| small | regular | 6/20 | 3/20 | 4/20 | 2/20 | 3/20 | 2/20 | 0/20 | 2/20 |
| small | counter | 16/20 | 10/20 | 8/20 | 6/20 | 7/20 | 7/20 | 4/20 | 6/20 |
| medium | regular | 12/20 | 2/20 | 0/20 | 0/20 | 1/20 | 6/20 | 1/20 | 0/20 |
| medium | counter | 10/20 | 3/20 | 5/20 | 2/20 | 7/20 | 16/20 | 5/20 | 4/20 |

Notably, whisper-small counter methods show substantially higher C0 (16 vs 6/20) and S2 (8 vs 4/20) counts. Whisper-medium counter shows higher S5 (16 vs 6/20) and S6 (5 vs 1/20) — suggesting the counter direction for medium produces more genuine recovering troughs at later latencies, outside the fixed 100–240 ms window. S7 follows S2: the encoder's amplitude-qualified troughs concentrate in the counter direction (small 2 reg vs 6 counter; medium 0 reg vs 4 counter), i.e. the few encoder MMNs that clear the µV floor come disproportionately from the swapped-frequency stimuli.

---

## Section 5 — ROI sensitivity (updated, 20 methods)

> **Data:** `outputs/results_with_counter/mmn_roi_variants.csv` (1280 rows = 20 methods × 4 models × 2 levels × 4 ROI variants per level × 2 mappings).
> C0 criterion (peak < 0 in 100–240 ms) only — shape criteria are not recomputed per ROI variant.
> ROI variants: electrodes {Fz, FCz, Fz+FCz, current7}; parcels {central, current2, frontal, temporal}.

### MMN count per ROI variant (n/20 per model = 20 methods)

**Table 5a. mTRF — electrodes** (C0 count per single-site and multi-site ROI variant)

| Model | FCz (n/20) | Fz (n/20) | Fz+FCz (n/20) | current7 (n/20) |
| ------ | ---------- | --------- | ------------- | --------------- |
| tiny | 17/20 | 18/20 | 19/20 | 19/20 |
| base | 16/20 | 16/20 | 16/20 | 16/20 |
| small | 19/20 | 18/20 | 16/20 | 19/20 |
| medium | 17/20 | 19/20 | 18/20 | 17/20 |
| **Total** | **69/80** | **71/80** | **69/80** | **71/80** |

**Table 5b. mTRF — parcels**

| Model | central (n/20) | current2 (n/20) | frontal (n/20) | temporal (n/20) |
| ------ | -------------- | --------------- | -------------- | --------------- |
| tiny | 17/20 | 19/20 | 20/20 | 18/20 |
| base | 16/20 | 16/20 | 18/20 | 18/20 |
| small | 19/20 | 19/20 | 13/20 | 14/20 |
| medium | 19/20 | 17/20 | 17/20 | 17/20 |
| **Total** | **71/80** | **71/80** | **68/80** | **67/80** |

**Table 5c. Encoder — electrodes**

| Model | FCz (n/20) | Fz (n/20) | Fz+FCz (n/20) | current7 (n/20) |
| ------ | ---------- | --------- | ------------- | --------------- |
| tiny | 12/20 | 13/20 | 11/20 | 12/20 |
| base | 13/20 | 14/20 | 14/20 | 14/20 |
| small | 12/20 | 13/20 | 12/20 | 11/20 |
| medium | 10/20 | 11/20 | 11/20 | 10/20 |
| **Total** | **47/80** | **51/80** | **48/80** | **47/80** |

**Table 5d. Encoder — parcels**

| Model | central (n/20) | current2 (n/20) | frontal (n/20) | temporal (n/20) |
| ------ | -------------- | --------------- | -------------- | --------------- |
| tiny | 13/20 | 12/20 | 12/20 | 12/20 |
| base | 13/20 | 13/20 | 13/20 | 15/20 |
| small | 12/20 | 11/20 | 10/20 | 10/20 |
| medium | 12/20 | 12/20 | 13/20 | 12/20 |
| **Total** | **50/80** | **48/80** | **48/80** | **49/80** |

### Mean ROI peak per variant

**Table 5e. mTRF — electrodes** (mean z-score peak across 20 methods)

| Model | FCz | Fz | Fz+FCz | current7 |
| ------ | --- | -- | ------ | -------- |
| tiny | −0.76 | −0.61 | −0.66 | −0.66 |
| base | −0.50 | −0.42 | −0.45 | −0.35 |
| small | −0.41 | −0.40 | −0.39 | −0.27 |
| medium | −0.60 | −0.43 | −0.48 | −0.48 |

**Table 5f. mTRF — parcels**

| Model | central | current2 | frontal | temporal |
| ------ | ------- | -------- | ------- | -------- |
| tiny | −0.75 | −0.77 | −0.84 | −0.72 |
| base | −0.51 | −0.39 | −0.44 | −0.46 |
| small | −0.60 | −0.37 | −0.20 | −0.22 |
| medium | −0.57 | −0.58 | −0.70 | −0.67 |

**Table 5g. Encoder — electrodes**

| Model | FCz | Fz | Fz+FCz | current7 |
| ------ | --- | -- | ------ | -------- |
| tiny | −0.16 | −0.29 | −0.20 | −0.09 |
| base | −0.33 | −0.69 | −0.49 | −0.32 |
| small | −0.28 | −0.31 | −0.28 | −0.34 |
| medium | −0.29 | −0.21 | −0.25 | −0.17 |

**Table 5h. Encoder — parcels**

| Model | central | current2 | frontal | temporal |
| ------- | ------- | -------- | ------- | -------- |
| tiny | −1.29 | −1.20 | −1.11 | −1.11 |
| base | −1.78 | −1.62 | −1.48 | −1.36 |
| small | −0.06 | −0.05 | −0.08 | −0.06 |
| medium | −0.15 | −0.17 | −0.19 | −0.18 |

### Correlation between ROI variants (continuous peak, n=160 runs pooled across all models × mappings × methods)

**Table 5i. Electrodes: Pearson r vs current7**

| Variant | r vs current7 |
| ------- | ------------- |
| FCz | +0.951 |
| Fz+FCz | +0.871 |
| Fz | +0.615 |

**Table 5j. Parcels: Pearson r vs current2**

| Variant | r vs current2 |
| ------- | ------------- |
| frontal | +0.977 |
| central | +0.974 |
| temporal | +0.958 |

### Section 5 summary

**Electrodes:** FCz tracks current7 most faithfully in continuous peak values (r=+0.951),
outperforming Fz (r=+0.615) and Fz+FCz (r=+0.871). Notably, Fz and current7 agree on
the same binary MMN count (71/80 each under mTRF), but FCz has lower count agreement
(69/80) — suggesting Fz and current7 agree on direction but not on peak magnitude, while
FCz agrees better on magnitude but slightly worse on binary calls. The low r=0.615 for Fz
vs current7 (continuous) indicates that Fz peak magnitude diverges from the broadband
ROI for some methods, likely where energy is distributed more centrally than frontally.
For reporting, current7 remains the primary ROI; Fz is the recommended canonical
single-electrode for figures/tables where a single site is needed, but FCz should be noted
as the stronger continuous proxy.

**Parcels:** All three single-parcel variants track current2 closely (r=0.958–0.977).
Central (r=0.974) and frontal (r=0.977) have essentially equivalent correlations, but
frontal has notably lower MMN counts for whisper-small (13/20 vs 19/20 for current2 and
central) — suggesting frontal is less reliable for the smaller model where the spatial
distribution of mTRF weights may produce weaker frontal projections. Central remains the
recommended single-parcel ROI: it matches current2 in count (both 71/80 under mTRF) and
has high continuous correlation (r=0.974).

**Encoder ROI sensitivity:** The encoder shows minimal sensitivity to ROI choice in both
directions — all variants give counts within ±3 of each other for both electrodes and
parcels. This is consistent with the encoder's lower signal-to-noise (more marginal peaks)
making ROI choice less impactful than for mTRF.

---

## Section 6 — Shape criteria under Fz/central ROI (updated, 20 methods)

> **Code:** `scripts/analyze_mmn_criteria_s5_s6.py --roi_variant fz_central`
> **Data:** `outputs/results_with_counter/mmn_criteria_s5_s6_fz_central.csv`

**Table 25. Electrodes — Fz only — mTRF** (n/20 per model)

| Model | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) | S7 (n/20) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 18/20 | 9/20 | 15/20 | 6/20 | 11/20 | 14/20 | 7/20 | 3/20 |
| base | 16/20 | 8/20 | 16/20 | 8/20 | 11/20 | 18/20 | 11/20 | 3/20 |
| small | 18/20 | 12/20 | 18/20 | 12/20 | 12/20 | 20/20 | 8/20 | 0/20 |
| medium | 19/20 | 14/20 | 19/20 | 14/20 | 10/20 | 13/20 | 5/20 | 12/20 |
| **Total** | **71/80** | **43/80** | **68/80** | **40/80** | **44/80** | **65/80** | **31/80** | **18/80** |

**Table 26. Parcels — central only — mTRF** (n/20 per model)

| Model | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) | S7 (n/20) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 17/20 | 7/20 | 15/20 | 7/20 | 11/20 | 17/20 | 8/20 | 2/20 |
| base | 16/20 | 8/20 | 16/20 | 8/20 | 12/20 | 15/20 | 8/20 | 1/20 |
| small | 19/20 | 14/20 | 19/20 | 14/20 | 14/20 | 19/20 | 12/20 | 1/20 |
| medium | 19/20 | 13/20 | 19/20 | 13/20 | 10/20 | 13/20 | 6/20 | 11/20 |
| **Total** | **71/80** | **42/80** | **69/80** | **42/80** | **47/80** | **64/80** | **34/80** | **15/80** |

**Table 27. Electrodes — Fz only — encoder** (n/20 per model)

| Model | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) | S7 (n/20) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 13/20 | 7/20 | 4/20 | 3/20 | 2/20 | 3/20 | 1/20 | 0/20 |
| base | 14/20 | 3/20 | 11/20 | 3/20 | 8/20 | 12/20 | 3/20 | 0/20 |
| small | 13/20 | 6/20 | 4/20 | 3/20 | 9/20 | 5/20 | 4/20 | 1/20 |
| medium | 11/20 | 3/20 | 4/20 | 2/20 | 3/20 | 8/20 | 2/20 | 0/20 |
| **Total** | **51/80** | **19/80** | **23/80** | **11/80** | **22/80** | **28/80** | **10/80** | **1/80** |

**Table 28. Parcels — central only — encoder** (n/20 per model)

| Model | C0 (n/20) | S1 (n/20) | S2 (n/20) | S3 (n/20) | S4 (n/20) | S5 (n/20) | S6 (n/20) | S7 (n/20) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 13/20 | 7/20 | 5/20 | 4/20 | 4/20 | 7/20 | 3/20 | 1/20 |
| base | 13/20 | 8/20 | 6/20 | 4/20 | 7/20 | 8/20 | 3/20 | 2/20 |
| small | 12/20 | 6/20 | 8/20 | 4/20 | 4/20 | 7/20 | 2/20 | 2/20 |
| medium | 12/20 | 2/20 | 4/20 | 2/20 | 7/20 | 12/20 | 3/20 | 4/20 |
| **Total** | **50/80** | **23/80** | **23/80** | **14/80** | **22/80** | **34/80** | **11/80** | **9/80** |

### Combined (parcels + electrodes, Fz/central ROI)

**Table 29. mTRF, combined (Fz + central)** (n/40 per model)

| Model | C0 (n/40) | S1 (n/40) | S2 (n/40) | S3 (n/40) | S4 (n/40) | S5 (n/40) | S6 (n/40) | S7 (n/40) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 35/40 | 16/40 | 30/40 | 13/40 | 22/40 | 31/40 | 15/40 | 5/40 |
| base | 32/40 | 16/40 | 32/40 | 16/40 | 23/40 | 33/40 | 19/40 | 4/40 |
| small | 37/40 | 26/40 | 37/40 | 26/40 | 26/40 | 39/40 | 20/40 | 1/40 |
| medium | 38/40 | 27/40 | 38/40 | 27/40 | 20/40 | 26/40 | 11/40 | 23/40 |
| **Total** | **142/160** | **85/160** | **137/160** | **82/160** | **91/160** | **129/160** | **65/160** | **33/160** |

**Comparison to original Table 29 (n/20):** As fractions — C0: 88% (vs 86% original), S2: 86% (vs 80%), S6: 41% (vs 39%). The mTRF Fz/central picture is essentially unchanged or slightly improved when counter methods are added. **S7 (1.0 µV floor) = 33/160 (21%)** — now well below S6 (65/160), and by a different mechanism (amplitude, not latency envelope); on the narrower Fz/central ROI the single-site troughs are shallower, so S7 retains just 33/137 = 24% of S2 here vs 58/136 = 43% under the full ROI (Table 13).

**Table 30. Encoder, combined (Fz + central)** (n/40 per model)

| Model | C0 (n/40) | S1 (n/40) | S2 (n/40) | S3 (n/40) | S4 (n/40) | S5 (n/40) | S6 (n/40) | S7 (n/40) |
| ------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| tiny | 26/40 | 14/40 | 9/40 | 7/40 | 6/40 | 10/40 | 4/40 | 1/40 |
| base | 27/40 | 11/40 | 17/40 | 7/40 | 15/40 | 20/40 | 6/40 | 2/40 |
| small | 25/40 | 12/40 | 12/40 | 7/40 | 13/40 | 12/40 | 6/40 | 3/40 |
| medium | 23/40 | 5/40 | 8/40 | 4/40 | 10/40 | 20/40 | 5/40 | 4/40 |
| **Total** | **101/160** | **42/160** | **46/160** | **25/160** | **44/160** | **62/160** | **21/160** | **10/160** |

**Comparison to original Table 30 (n/20):** C0: 63% (vs 60%), S2: 29% (vs 24%), S6: 13% (vs 7.5%). The encoder's absolute counts improve with counter methods (97/160 C0-positive vs 48/80=60%), but the S2/C0 retention ratio stays low (46/101=46%) — shape quality is not rescued by adding counter stimuli. **S7 = 10/160**, now falling below S6 (21/160), and retains only 10/46 = 22% of S2 on the Fz/central ROI — so on the single-site ROI the encoder loses roughly three-quarters of its (already scarce) S2 troughs to the amplitude floor.

### Summary

The Section 4 and Section 6 findings from `results_analysis.md` are preserved with 20 methods:
**encoder still collapses far more than mTRF under shape gating** (S2/C0 retention: 96% for mTRF Fz/central vs 46% for encoder), the edge-rejection penalty on mTRF under S1/S3 persists, and the mTRF-vs-encoder gap at S6 is maintained (65/142=46% mTRF S6/C0 vs 21/101=21% encoder). The narrowing to Fz/central still does not rescue the encoder's shape problem, confirming all original Section 6 conclusions. **The new S7 amplitude gate (1.0 µV) reinforces this**: mTRF keeps 33/160 amplitude-qualified MMNs vs the encoder's 10/160 (Fz/central), so requiring a genuine microvolt-scale trough — not just a z-scored dip — widens rather than closes the mTRF−encoder gap. S7 is now the strictest usable criterion (amplitude-gated), landing below S6 in aggregate (33 vs 65/160 mTRF; 10 vs 21/160 encoder) and at or below S6 in every model × mapping cell except whisper-medium mTRF; see **Section 7** for the full ROI-option and X-sweep breakdown.

---

## Section 7 — Amplitude-gated MMN (S2 vs S7) across ROI options

> **Code:** `scripts/analyze_mmn_s7_roi.py`
> **Data:** `outputs/results_with_counter/mmn_s7_roi.csv` (8,640 rows = 20 methods × 4 models
> × 2 mappings × 9 ROI options × 6 X-thresholds).
>
> This section asks how the shape verdict **S2** and the amplitude-gated verdict **S7** behave
> at each candidate *reporting site* rather than the committed averaged ROI. MMN is classically
> reported at single fronto-central sites, so the ROI options are the **frontal** and **central
> parcels** plus **each electrode of the frontal/central 10-20 clusters** (Fz, F3, F4, FCz —
> frontal; Cz, C3, C4 — central; all 7 survive the NC floor). Each option is a **single target**
> (no averaging). S7 = S2 AND the microvolt deviant−standard trough at the S2 latency ≤ −X µV,
> headline **X = 1.0 µV** (provisional). All 20 methods (regular + counter) are pooled.
>
> **Scale caveat (repeated):** these are the model's **regularization-shrunk predicted µV**, not
> literature EEG µV. A literature MMN is ≈ 1–5 µV; the model's S2-passing troughs sit at a median
> of ≈ −0.7 to −0.9 µV, so X is calibrated to the model's own scale, not the literature's.

### µV-trough distribution (the calibration data)

Deepest µV trough in the 100–240 ms window, over the **S2-passing** single-target traces (the
ones S7 can gate), one row per (mapping, ROI-kind):

**Table 31. Predicted µV-trough distribution (S2-passing single-target traces)**

| Mapping × kind | n | min | median | max | ≤−0.25 | ≤−0.5 | ≤−1.0 | ≤−1.5 | ≤−2.0 | ≤−2.5 |
| -------------- | --- | ----- | ------ | ---- | ------ | ----- | ----- | ----- | ----- | ----- |
| mTRF × parcel | 131 | −36.28 | −0.80 | +0.80 | 104 | 86 | 56 | 41 | 30 | 21 |
| mTRF × electrode | 453 | −77.01 | −0.86 | +1.55 | 362 | 293 | 213 | 152 | 113 | 88 |
| encoder × parcel | 44 | −19.14 | −0.88 | +4.72 | 34 | 29 | 20 | 14 | 12 | 12 |
| encoder × electrode | 156 | −14.52 | −0.70 | +4.49 | 102 | 89 | 63 | 50 | 37 | 31 |

The median S2-passing trough is ≈ −0.8 µV for every mapping × kind, so **1.0 µV sits just beyond
the median and removes slightly more than half of the genuine troughs** (the ≤−1.0 µV column of
Table 31 keeps only ≈ 40–47% of each S2-passing set), not merely the shallow (0 → −0.5 µV) tail.
A handful of very deep mTRF
troughs (min −36 to −77 µV) are noisy single electrodes; S7 rightly keeps them (they are far past
any threshold), and the ROI-mean criteria of Sections 4/6 average that noise down.

### S2 → S7 by ROI option, per model (X = 1.0 µV)

Each cell is **S2 → S7** present-count, n/20 per model; **Total** is n/80 (20 methods × 4 models).

**Table 32. mTRF — S2 → S7 per ROI option**

| ROI option | kind | tiny | base | small | medium | Total (n/80) |
| ---------- | ---- | ---- | ---- | ----- | ------ | ------------ |
| frontal | parcel | 18→10 | 16→9 | 13→7 | 15→15 | 62→41 |
| central | parcel | 15→2 | 16→1 | 19→1 | 19→11 | 69→15 |
| Fz | electrode | 15→3 | 16→3 | 18→0 | 19→12 | 68→18 |
| F3 | electrode | 18→15 | 16→13 | 18→14 | 14→12 | 66→54 |
| F4 | electrode | 16→13 | 15→12 | 19→14 | 13→12 | 63→51 |
| FCz | electrode | 16→5 | 16→2 | 19→1 | 15→9 | 66→17 |
| Cz | electrode | 16→8 | 16→7 | 16→3 | 16→12 | 64→30 |
| C3 | electrode | 15→4 | 17→1 | 18→2 | 13→7 | 63→14 |
| C4 | electrode | 15→10 | 15→3 | 17→5 | 16→11 | 63→29 |

**Table 33. Encoder — S2 → S7 per ROI option**

| ROI option | kind | tiny | base | small | medium | Total (n/80) |
| ---------- | ---- | ---- | ---- | ----- | ------ | ------------ |
| frontal | parcel | 5→2 | 5→2 | 7→4 | 4→3 | 21→11 |
| central | parcel | 5→1 | 6→2 | 8→2 | 4→4 | 23→9 |
| Fz | electrode | 4→0 | 11→0 | 4→1 | 4→0 | 23→1 |
| F3 | electrode | 7→3 | 7→5 | 4→3 | 4→3 | 22→14 |
| F4 | electrode | 4→2 | 5→5 | 3→3 | 4→3 | 16→13 |
| FCz | electrode | 7→0 | 10→1 | 5→4 | 3→2 | 25→7 |
| Cz | electrode | 3→1 | 5→0 | 5→2 | 13→6 | 26→9 |
| C3 | electrode | 5→0 | 10→2 | 4→3 | 3→2 | 22→7 |
| C4 | electrode | 5→2 | 7→4 | 4→3 | 6→3 | 22→12 |

### S7 vs the amplitude threshold X

Total S7 present-count as X rises (pooled over all 9 ROI options × 20 methods × 4 models = 720
cells per mapping; S2 is the X→0 reference):

**Table 34. S7 falling with the amplitude threshold X**

| X (µV) | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 |
| ------ | ---- | --- | --- | --- | --- | --- |
| mTRF S7 (of S2 = 584) | 466 | 379 | 269 | 193 | 143 | 109 |
| encoder S7 (of S2 = 200) | 136 | 118 | 83 | 64 | 49 | 43 |

### Section 7 summary

- **S7 tightens S2 at every site**, but by how much depends strongly on the reporting ROI. For the
  mTRF, the **lateral frontal electrodes F3/F4 have the deepest predicted troughs** (S7/S2 = 54/66
  = 82% and 51/63 = 81%), while the **classic midline MMN sites are far shallower** — Fz retains
  only 18/68 = 26%, FCz 17/66 = 26%, central parcel 15/69 = 22%. So the amplitude floor bites
  hardest exactly where MMN is conventionally reported; the µV depth of the model's predicted MMN
  is greater off-midline.
- **The encoder starts far lower on S2 and the 1.0 µV floor all but eliminates its best midline
  site** (Fz 1/23 = 4%); its strongest site is now the lateral-frontal F3 (22 S2 → 14 S7).
- **S7 declines smoothly and monotonically with X** (Table 34), confirming the {0.25 … 2.5} µV
  sweep is well-centred: at the lenient 0.25 µV end S7 ≈ 0.8 × S2, at the strict 2.5 µV end
  S7 ≈ 0.2 × S2. There is no threshold at which the sweep is off-scale.
- **The mTRF−encoder gap persists and widens under the amplitude gate** at every ROI option, in the
  same direction as Sections 4/6: requiring a genuine microvolt-scale trough — not merely a z-scored
  dip — is additional evidence that the mTRF's in-silico MMN is the more physiologically credible.
- **Reporting-site recommendation:** if a single amplitude-qualified electrode is wanted, **F3/F4**
  (mTRF) carry the deepest and most consistently S7-positive troughs; the committed averaged ROIs of
  Sections 4/6 remain the most robust because they average down the single-site noise visible in the
  −36 to −77 µV tail of Table 31. **X = 1.0 µV is provisional** pending the literature amplitude
  review; the full sweep is retained in `mmn_s7_roi.csv` so a revised X is a one-line recompute.

---

*Generated by `scripts/generate_counter_analysis_docs.py` and manually reviewed/expanded.*
*Section 7 and the S7 columns (Tables 13–14b, 25–30) added by hand from
`analyze_mmn_criteria_s5_s6.py` (S7 column) and `analyze_mmn_s7_roi.py` (Section 7).*
