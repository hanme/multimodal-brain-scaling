# Does `baseline_normalized_peak` need a shape-aware criterion?

**Status: investigation + recommendation, not a committed change.** This is the "second
pair of eyes" before we touch the metric, since it changes the paper's core
presence/absence claims. It addresses the shape-constraint weakness flagged in
`aux/results_analysis.md`. The scoring window itself is settled at the current
absolute 100–240 ms (see Q2 below for the literature cross-check that confirmed it) and
is not revisited or varied anywhere in this analysis.

> **Constraint on this pass.** The full per-time-point curves live only in the prediction
> HDF5s on `jed`; this machine has no cluster access and the local `mmn_results_table.csv`
> has only the reduced scalars. So the *quantitative recount across all 160 runs has not
> been produced here* — it needs one cluster run of the script below. What this pass
> *did* use: (a) the locally-rendered prediction figures (which plot the exact z-scored
> `deviant − standard` curve the metric scores, with the 100–240 ms band shaded) to
> inspect the flagged cases and controls; (b) a literature cross-reference of the 10
> source papers' own MMN windows; (c) a synthetic validation of each candidate criterion.

---

## TL;DR recommendation

1. **Add a shape constraint — yes.** The flagged smooth-ramp cases are real and the
   current "min in window < 0" rule cannot tell them from a genuine MMN. Of the candidate
   shape tests, **a post-trough recovery requirement (S2)** and/or **a control-window
   specificity check (S4)** are the right ones. **Avoid a strict "interior argmin" test
   (S1) as the sole criterion** — it correctly kills ramps but *also* kills genuine MMNs
   whose trough sits at the early (100 ms) window edge, which we observe in real runs
   (e.g. `whisper-base`/mtrf/parcels/`method_55`).
2. **Window width — settled, not revisited.** The current absolute 100–240 ms window was
   cross-checked against the 10 source papers' own MMN windows (Q2 below) and confirmed to
   be a defensible choice. This investigation, and `scripts/analyze_mmn_criteria.py`, score
   only this window throughout — no widening, tightening, or SOA-relative scaling.

**Net:** keep the current 100–240 ms window, and gate the verdict with a recovery
**and/or** specificity shape test. Run `scripts/analyze_mmn_criteria.py` on the cluster to
get the exact new counts before committing a formula.

---

## Q1 — shape-aware criterion

### Candidates implemented (`scripts/analyze_mmn_criteria.py`)

All candidates keep the magnitude gate (`peak < 0`) and add a shape condition, evaluated on
the **ROI-mean z_diff trace** (the actual verdict signal: `{frontal, central}` for parcels,
`{Fz,FCz,Cz,FC1,FC2,F1,F2}` for electrodes).

| ID | Name | Extra condition on the windowed z_diff |
|----|------|----------------------------------------|
| **C0** | current | none (most-negative point `< 0`) |
| **S1** | interior argmin | the minimum is not at either window edge (≥1 bin in) |
| **S2** | trough + recovery | after the trough, z_diff climbs back ≥50% of the trough depth within 120 ms |
| **S3** | interior **&** recovery | S1 **and** S2 |
| **S4** | control-window specificity | in-window min is *more negative* than the min of a no-MMN control window (300–440 ms) |

### Synthetic validation (confirms the criteria do what we claim)

Run on four canonical 20 ms-grid traces (`python scripts/analyze_mmn_criteria.py` ships the
same `trace_stats`/`decide` used here):

| trace | peak | argmin | C0 | S1 | S2 | S3 | S4 |
|-------|------|--------|----|----|----|----|----|
| genuine trough (dip ~170 ms, recovers) | −1.88 | 160 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **monotonic ramp** (min pinned at 240 ms, true trough later) | −2.27 | 240 | **✓** | ✗ | ✗ | ✗ | ✗ |
| early trough at 100 ms edge (recovers) | −2.00 | 100 | ✓ | **✗** | ✓ | **✗** | ✓ |
| flat / positive | +0.20 | — | ✗ | ✗ | ✗ | ✗ | ✗ |

Read-out:
- **Every shape criterion correctly rejects the monotonic ramp** while keeping the genuine
  trough. That is the whole point.
- **S1/S3 (interior) wrongly reject the early-edge trough**; **S2 (recovery) and S4
  (specificity) keep it.** This is the decisive trade-off — see the real example below.

### Figure evidence on the actually-flagged cases

(Read directly off the local prediction figures; the 3rd-column trace is the scored z_diff.)

- **`whisper-tiny`/encoder/parcels/`method_72`** (peak ≈ −6.5) and **`method_75`** (≈ −6.5):
  z_diff sits ~0 until ~150 ms, then **descends monotonically through the window with its
  minimum at the right edge (~240 ms)**, and the true trough is at ~290–300 ms (outside the
  window). Classic ramp / window-clip. → would flip to **absent** under S1/S2/S3/S4.
- **`whisper-base`/encoder/parcels/`method_72`** (≈ −2.9), **`method_75`** (≈ −2.9): same
  shape — descends through the window, trough at ~300 ms, only partial recovery after. →
  flips to **absent** under the shape criteria.
- **`whisper-medium`/mtrf/electrodes/`method_43`** (ROI ≈ −3.76): the frontal ROI channels
  (F1/Fz/F2/FCz/FC2) show **narrow 1–2-sample spikes near the right window edge**, not broad
  troughs (`Cz` is actually slightly positive). The extreme value is an edge/variance-inflation
  artifact, not an MMN. → flips to **absent** under S1/S3 (edge) and S4 (the spike isn't more
  negative than later activity); S2 depends on the exact recovery slope.

### Clear-trough sanity checks (must stay present)

- **`whisper-tiny`/mtrf/parcels/`method_44`** (≈ −1.5): oscillatory, with a clear in-window
  dip (~180–230 ms) that **rebounds to +2.5 by ~300 ms**. Stays present under S2/S4 (strong
  recovery, in-window-specific). S1 borderline (argmin near the right edge).
- **`whisper-base`/mtrf/parcels/`method_55`** (≈ −1.4): genuine dip-and-recover, but its
  **trough is near the *left* (100–120 ms) edge**. Stays present under **S2/S4** but **S1/S3
  would wrongly reject it** — the concrete false-negative that rules out interior-only.

**Conclusion (Q1):** a shape gate is warranted and cleanly separates the ramp cases from
genuine troughs. **Prefer S2 (recovery) or S4 (specificity), or their AND** as the gate.
Do **not** ship S1/S3 alone.

---

## Q2 — window width / placement (settled)

### What the 10 source papers actually used

Reported MMN measurement windows (frequency deviants), from the source papers / same-group
conventions. "≈/conv." = the paper itself wasn't pinned down this pass; value is the group's
standard window or the field convention — verify before citing.

| method | source | std→dev | SOA (ms) | reported MMN window (ms) |
|--------|--------|---------|----------|--------------------------|
| method_27 | Schall_1999a | 1000→1064 | 900 | ≈100–200 (not pinned) |
| method_37 | Javitt_2000a | 1000→1050 | 310 | peak ~180–200 (narrow; latency ≈185) |
| method_43 | Michie_2000b | 633→700 | 510 | ~90–170 (freq) / peak-pick 90–250 |
| method_44 | Michie_2000c | 633→1000 | 510 | as method_43 |
| method_53 | Salisbury_2002a | 1000→1200 | 333 | **mean voltage 100–200** (confirmed) |
| method_55 | Shinozaki_2002a | 1000→2000 | 500 | ≈100–200 (large deviant → earlier; not pinned) |
| method_60 | Umbricht_2003a | 1000→1500 | 300 | **freq peak 100–200** (confirmed, group conv.) |
| method_72 | Bodatsch_2011 | 1000→1200 | 500 | **150–250** (fMMN; review-confirmed) |
| method_74 | Domjan_2012 | 1000→1500 | 1000 | ~100–250 (latency study) |
| method_75 | Karger_2014 | 1000→1200 | 500 | ~100–250 (latency study) |

**Union ≈ 90–250 ms; modal ≈ 100–200 ms.** The current **100–240 ms** window is a defensible
common window: it covers the early-edge papers (≥90–100) and reaches the right edge of the
150–250 convention.

**Conclusion (Q2):** keep the current absolute **100–240 ms** window. This question is
closed — `scripts/analyze_mmn_criteria.py` scores only this window, and no alternative
window (wider, tighter, or SOA-relative) is considered anywhere else in this analysis.

---

## How to produce the actual numbers (cluster)

```bash
# on jed, where outputs/insilico_mmn_predictions/<model>[-<level>]/...h5 live:
python scripts/analyze_mmn_criteria.py \
  --predictions_root outputs/insilico_mmn_predictions \
  --out outputs/results/mmn_criteria_comparison.csv
```

The script:
- recomputes `z_diff` from the stored raw `standard`/`deviant_mean` and **asserts it
  reproduces the committed `peak`** (so it can't silently diverge from the pipeline);
- writes a per-run CSV (peak, argmin latency, interior/recovery/control stats and the
  present/absent verdict under C0/S1/S2/S3/S4, all on the current 100–240 ms window) plus
  a per-target CSV;
- prints headline **MMN-present counts per criterion**, the list of runs that flip
  `present → absent` under each candidate, the five flagged ramp cases, the clear-trough
  sanity cases, and all borderline (|peak|<0.1) runs.

Deliverable to finish after that run: drop the printed C0-vs-S2-vs-S4 count table into
`results_analysis.md`, confirm the five flagged ramp cases flipped to absent and the
clear-trough cases stayed present, then pick the gate (recommend **S2**, or **S2∧S4** if you
want to be conservative) and wire it into `finalize_method` as an extra returned flag
(`mmn_present`) without changing `peak` itself.
