#!/usr/bin/env python
"""Build the novel tone-pair grid: every unordered pair of a log-spaced frequency grid.

The literature screen (24 methods) tests whatever pairs published MMN studies happened to use.
This builds the search space for the complementary question -- do pairs that DON'T appear in the
literature drive stronger, more model-consistent responses? -- as a dense grid over
[200, 7500] Hz.

Grid: N frequencies log-spaced from F_LO to F_HI, so every adjacent step is the same ratio
(32 points over 200->7500 Hz = 1.1240x = 2.02 semitones = 12.5% deviance per step). Log spacing
matters because pitch perception is ratio-based: a linear grid would be perceptually dense at the
top and sparse at the bottom.

Unordered pairs only: {f_i -> f_j} and {f_j -> f_i} are the SAME pair, because the generator
synthesizes both directions of every row (audio_outputs_regular / audio_outputs_counter). So
n frequencies give n(n-1)/2 rows and 2 x n(n-1)/2 = n(n-1) direction-instances, covering every
ordered combination exactly once. Emitting reversed rows too would double the cost for nothing.

The diagonal (f_i == f_j) is excluded deliberately: the "deviant" sequence would synthesize to a
waveform byte-identical to the standard, so deviant - standard is exactly zero by construction --
a degenerate control, not an informative one. The graded control is the smallest-Deltaf pairs
(adjacent grid steps).

method_id starts at 1001 so it can never collide with the literature ids (1-76), which share the
method_{id} stimulus-directory namespace.

  python scripts/build_novel_grid_csv.py            # -> data/metadata/novel_grid_frequency_metadata.csv
  python scripts/build_novel_grid_csv.py --n_freqs 8 --out /tmp/small.csv --index_out /tmp/i.csv
"""

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

# The literature sheet's column order, reproduced exactly: 00aa_generate_audio_stimuli.py and
# insilico_mmn.build_methods_from_csv/load_soa_table/load_duration_map all read this schema by
# name, so the grid CSV is a drop-in for --metadata_csv.
# NOTE the asymmetry between the two halves -- deviant_int precedes deviant_dur, while
# standard_int comes last. It is not a mirror image; do not "tidy" it.
COLUMNS = [
    "source", "method_id", "paradigm", "change_type", "standard_id", "standard_freq",
    "standard_dur", "standard_isi", "standard_soa", "standard_int", "deviant_id",
    "deviant_freq", "deviant_int", "deviant_dur", "deviant_isi", "deviant_soa", "p_deviant_pc",
]

# Fixed tone parameters for every grid pair (ms / dB / %).
TONE_DUR_MS = 80
ISI_MS = 500
SOA_MS = 580          # = TONE_DUR_MS + ISI_MS
INTENSITY_DB = 80
P_DEVIANT_PC = 10

F_LO, F_HI, N_FREQS = 200.0, 7500.0, 32
FIRST_METHOD_ID = 1001


def build_grid(n_freqs=N_FREQS, f_lo=F_LO, f_hi=F_HI):
    """n log-spaced frequencies, rounded to integer Hz.

    Rounded values are what gets written, so the synthesized tone and the metadata agree; the
    ratio between adjacent rounded steps is therefore only approximately constant (the rounding
    is at most 0.5 Hz, i.e. <0.25% at the bottom of the range and negligible at the top).
    """
    return [round(f_lo * (f_hi / f_lo) ** (i / (n_freqs - 1))) for i in range(n_freqs)]


def build_rows(freqs, first_id=FIRST_METHOD_ID):
    """One row per unordered pair, lower frequency as the standard."""
    rows, index = [], []
    for mid, (f_lo, f_hi) in enumerate(combinations(freqs, 2), start=first_id):
        rows.append({
            "source": "novel_grid",
            "method_id": mid,
            "paradigm": "oddball",
            # 00aa_generate_audio_stimuli.py and build_methods_from_csv both filter on exactly
            # this string; anything else is silently dropped by both.
            "change_type": "Frequency",
            "standard_id": mid,
            "standard_freq": f_lo,
            "standard_dur": TONE_DUR_MS,
            "standard_isi": ISI_MS,
            "standard_soa": SOA_MS,
            "standard_int": INTENSITY_DB,
            "deviant_id": mid,
            "deviant_freq": f_hi,
            "deviant_int": INTENSITY_DB,
            "deviant_dur": TONE_DUR_MS,
            "deviant_isi": ISI_MS,
            "deviant_soa": SOA_MS,
            "p_deviant_pc": P_DEVIANT_PC,
        })
        ratio = f_hi / f_lo
        index.append({
            "method_id": mid,
            "f_low": f_lo,
            "f_high": f_hi,
            "ratio": round(ratio, 6),
            "semitones": round(12.0 * math.log2(ratio), 4),
            "pct_deviance": round(100.0 * (f_hi - f_lo) / f_lo, 4),
        })
    return rows, index


def verify(freqs, rows):
    """Fail loudly rather than let a malformed grid reach a 100+ CHF extraction."""
    n = len(freqs)
    assert len(set(freqs)) == n, (
        f"grid has duplicate frequencies after rounding: {n - len(set(freqs))} collisions. "
        f"Widen the range or reduce --n_freqs.")
    assert len(rows) == n * (n - 1) // 2, f"expected {n * (n - 1) // 2} rows, got {len(rows)}"

    valid = set(freqs)
    seen = set()
    for r in rows:
        lo, hi = r["standard_freq"], r["deviant_freq"]
        assert lo in valid and hi in valid, f"method {r['method_id']}: {lo}/{hi} off-grid"
        assert lo != hi, f"method {r['method_id']}: degenerate pair {lo}=={hi}"
        assert lo < hi, f"method {r['method_id']}: standard {lo} not below deviant {hi}"
        key = frozenset((lo, hi))
        assert key not in seen, f"method {r['method_id']}: duplicate unordered pair {lo}/{hi}"
        seen.add(key)

    ids = [r["method_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate method_id"
    assert min(ids) > 76, f"method_id {min(ids)} collides with the literature range (1-76)"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_freqs", type=int, default=N_FREQS,
                   help=f"grid points, log-spaced (default {N_FREQS} -> "
                        f"{N_FREQS * (N_FREQS - 1) // 2} pairs)")
    p.add_argument("--f_lo", type=float, default=F_LO, help=f"lowest frequency, Hz (default {F_LO:g})")
    p.add_argument("--f_hi", type=float, default=F_HI, help=f"highest frequency, Hz (default {F_HI:g})")
    p.add_argument("--first_method_id", type=int, default=FIRST_METHOD_ID,
                   help=f"first method_id (default {FIRST_METHOD_ID}; must exceed the "
                        f"literature range 1-76)")
    p.add_argument("--out", default="data/metadata/novel_grid_frequency_metadata.csv",
                   help="stimulus metadata CSV, in the literature schema")
    p.add_argument("--index_out", default="outputs/results_novel_search/grid_index.csv",
                   help="per-pair frequency/deviance index for the ranking step to join against")
    args = p.parse_args()

    assert args.n_freqs >= 2, "--n_freqs must be at least 2"
    assert 0 < args.f_lo < args.f_hi, "need 0 < --f_lo < --f_hi"

    freqs = build_grid(args.n_freqs, args.f_lo, args.f_hi)
    rows, index = build_rows(freqs, args.first_method_id)
    verify(freqs, rows)

    for path, cols, data in ((args.out, COLUMNS, rows),
                             (args.index_out, list(index[0]), index)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            # csv defaults to CRLF; the repo's metadata sheets are LF and a stray \r would ride
            # along on the final field of every row.
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(data)

    step_ratio = (args.f_hi / args.f_lo) ** (1 / (args.n_freqs - 1))
    print(f"grid: {len(freqs)} frequencies, {args.f_lo:g}-{args.f_hi:g} Hz, log-spaced")
    print(f"  step ratio {step_ratio:.4f} = {12 * math.log2(step_ratio):.3f} semitones "
          f"= {100 * (step_ratio - 1):.1f}% deviance (the finest the grid can resolve)")
    print(f"  {freqs}")
    print(f"pairs: {len(rows)} unordered -> {2 * len(rows)} direction-instances "
          f"(method_id {rows[0]['method_id']}-{rows[-1]['method_id']}, "
          f"plus a _counter dir for each)")
    print(f"  deviance range {index[0]['pct_deviance']:.1f}% - "
          f"{max(i['pct_deviance'] for i in index):.1f}%")
    print(f"tone: {TONE_DUR_MS} ms, ISI {ISI_MS} ms, SOA {SOA_MS} ms, {INTENSITY_DB} dB, "
          f"p_deviant {P_DEVIANT_PC}%")
    print(f"wrote {args.out}")
    print(f"wrote {args.index_out}")


if __name__ == "__main__":
    main()
