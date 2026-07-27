#!/usr/bin/env python
"""Build the novel tone-pair grid: every unordered pair of a semitone-spaced frequency grid.

The literature screen (24 methods) tests whatever pairs published MMN studies happened to use.
This builds the search space for the complementary question -- do pairs that DON'T appear in the
literature drive stronger, more model-consistent responses? -- as a dense grid over
[200, 7500] Hz. The literature's own frequency methods all sit between 600 and 1000 Hz, so this
deliberately overshoots them in both directions.

The grid is a single uniform LADDER: 43 rungs from 200 Hz in exact 1.500-semitone steps, giving
200 ... 7611 Hz at 8.8% deviance per step.

Why semitones rather than hertz. Pitch is ratio-based, so an equal-Hz grid over this range would
put a 13.5-semitone gap at the bottom and 0.55-semitone steps at the top -- a 24x difference in
perceptual step size, most of the points wasted in the treble. Equal ratio steps also mean "k
steps apart" denotes the same deviance everywhere, and that the two directions of a pair carry
equal deviance magnitude (+1.500 st one way, -1.500 st the other), which is what makes the
frequency-preference test interpretable.

Why 1.5 specifically. 12 / 1.5 = 8 exactly, so eight steps is one octave and 200/400/800/1600/
3200/6400 are all grid points -- true 2:1 pairs exist. Any step that does not divide 12 evenly
loses that: at the 1.494 st needed to land 43 points exactly on 7500 Hz, 12/1.494 = 8.03 and NO
pair anywhere in the grid would be an exact octave.

Why the top is 7611 and not 7500. 200 -> 7500 Hz is 62.746 semitones, and 62.746 / 1.5 = 41.83
steps -- not an integer, so a strict 1.5-semitone ladder cannot land on 7500. The choice is to
stop short (6979 Hz), to append 7500 as an irregular rung, or to overshoot by one step. This
overshoots: 7611 Hz is still comfortably under the 8 kHz Nyquist of the 16 kHz sample rate every
model takes (above which a tone aliases to a frequency the metadata does not claim), and it keeps
every step identical with no irregular rung to caveat in the analysis.

The cost of a uniform 1.5-semitone grid is resolution: the finest deviance it can express is
8.8%, coarser than 8 of the 24 literature frequency methods. This is a coarse-to-fine search of a
wide space, not a threshold study; a follow-up would need a finer grid over whatever region this
one flags.

Unordered pairs only: {f_i -> f_j} and {f_j -> f_i} are the SAME pair, because the generator
synthesizes both directions of every row (audio_outputs_regular / audio_outputs_counter). So
n frequencies give n(n-1)/2 rows and 2 x n(n-1)/2 = n(n-1) direction-instances, covering every
ordered combination exactly once. Emitting reversed rows too would double the cost for nothing.

The diagonal (f_i == f_j) is excluded deliberately: the "deviant" sequence would synthesize to a
waveform byte-identical to the standard, so deviant - standard is exactly zero by construction --
a degenerate control, not an informative one. The graded control is the smallest-Deltaf pairs.

method_id starts at 1001 so it can never collide with the literature ids (1-76), which share the
method_{id} stimulus-directory namespace.

  python scripts/build_novel_grid_csv.py          # -> data/metadata/novel_grid_frequency_metadata.csv
  python scripts/build_novel_grid_csv.py --n_ladder 5 --out /tmp/small.csv --index_out /tmp/i.csv
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

F_LO = 200.0             # ladder start, Hz
SEMITONE_STEP = 1.5      # 12 / 1.5 = 8 steps per octave exactly, so octaves land on the grid
N_LADDER = 43            # 200 ... 7611 Hz (one step past 7500; see the module docstring)
EXTRA_FREQS = ()         # none: a uniform ladder, no irregular rung
FIRST_METHOD_ID = 1001


def build_grid(f_lo=F_LO, semitone_step=SEMITONE_STEP, n_ladder=N_LADDER,
               extra_freqs=EXTRA_FREQS):
    """The ladder (rounded to integer Hz) plus any extras, sorted and de-duplicated.

    Anchoring every rung to f_lo -- f_lo * 2**(step*i/12) -- rather than interpolating between
    two endpoints keeps the step exact and the octaves exact. Interpolating instead between 200
    and a rounded 7184 would accumulate endpoint-rounding error and shift rung 27 from 4525 to
    4526.

    Rounded values are what gets written, so the synthesized tone and the metadata agree; the
    ratio between adjacent ROUNDED rungs is therefore only approximately constant (at most half
    a hertz, i.e. ~0.25% at the bottom of the range and negligible at the top).
    """
    ladder = [round(f_lo * 2 ** (semitone_step * i / 12)) for i in range(n_ladder)]
    return sorted(set(ladder) | {int(f) for f in extra_freqs})


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
        f"Widen the range or reduce --n_ladder / --semitone_step.")
    assert freqs == sorted(freqs), "grid must be ascending"
    assert len(rows) == n * (n - 1) // 2, f"expected {n * (n - 1) // 2} rows, got {len(rows)}"
    # 8 kHz Nyquist at the 16 kHz sample rate every model takes; a higher tone would alias to a
    # different frequency than the metadata claims.
    assert max(freqs) < 8000, f"{max(freqs)} Hz is at or above the 8 kHz Nyquist limit"

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
    p.add_argument("--f_lo", type=float, default=F_LO,
                   help=f"ladder start, Hz (default {F_LO:g})")
    p.add_argument("--semitone_step", type=float, default=SEMITONE_STEP,
                   help=f"ladder step in semitones (default {SEMITONE_STEP:g}; must divide 12 "
                        f"evenly for octaves to land on the grid -- 1.5 gives 8 steps/octave)")
    p.add_argument("--n_ladder", type=int, default=N_LADDER,
                   help=f"ladder rungs (default {N_LADDER})")
    p.add_argument("--extra_freqs", type=str,
                   default=",".join(str(f) for f in EXTRA_FREQS),
                   help=f"comma-separated extra frequencies merged into the ladder, in Hz "
                        f"(default {','.join(str(f) for f in EXTRA_FREQS)}). Pass an empty "
                        f"string for a pure ladder.")
    p.add_argument("--first_method_id", type=int, default=FIRST_METHOD_ID,
                   help=f"first method_id (default {FIRST_METHOD_ID}; must exceed the "
                        f"literature range 1-76)")
    p.add_argument("--out", default="data/metadata/novel_grid_frequency_metadata.csv",
                   help="stimulus metadata CSV, in the literature schema")
    p.add_argument("--index_out", default="outputs/results_novel_search/grid_index.csv",
                   help="per-pair frequency/deviance index for the ranking step to join against")
    args = p.parse_args()

    assert args.n_ladder >= 1, "--n_ladder must be at least 1"
    assert args.f_lo > 0, "--f_lo must be positive"
    assert args.semitone_step > 0, "--semitone_step must be positive"
    extras = tuple(float(x) for x in args.extra_freqs.split(",") if x.strip())

    freqs = build_grid(args.f_lo, args.semitone_step, args.n_ladder, extras)
    assert len(freqs) >= 2, "need at least 2 distinct frequencies to form a pair"
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

    steps = [12.0 * math.log2(b / a) for a, b in zip(freqs, freqs[1:])]
    finest = min(i["semitones"] for i in index)
    print(f"grid: {len(freqs)} frequencies, {freqs[0]}-{freqs[-1]} Hz")
    print(f"  ladder: {args.n_ladder} rungs from {args.f_lo:g} Hz in {args.semitone_step:g}-semitone "
          f"steps -> {round(args.f_lo * 2 ** (args.semitone_step * (args.n_ladder - 1) / 12))} Hz")
    if extras:
        print(f"  extras: {', '.join(f'{e:g}' for e in extras)} Hz")
    # Adjacent steps differ only by integer-Hz rounding UNLESS an extra sits off the ladder, in
    # which case the irregular rung is the interesting one -- it is the grid's finest deviance.
    irregular = [(a, b, s) for a, b, s in zip(freqs, freqs[1:], steps)
                 if abs(s - args.semitone_step) > 0.1]
    print(f"  adjacent steps {min(steps):.3f}-{max(steps):.3f} semitones; "
          f"finest pair {finest:.3f} st = {100 * (2 ** (finest / 12) - 1):.2f}% "
          f"(the finest deviance the grid can resolve)")
    for a, b, s in irregular:
        print(f"    irregular rung {a} -> {b} = {s:.3f} st = {100 * (b / a - 1):.2f}%")
    octaves = [f for f in freqs if abs(round(12 * math.log2(f / freqs[0])) % 12) < 1e-6
               and abs(12 * math.log2(f / freqs[0]) - round(12 * math.log2(f / freqs[0]))) < 1e-6]
    print(f"  exact octaves of {freqs[0]} Hz on the grid: "
          f"{', '.join(str(f) for f in octaves) if len(octaves) > 1 else 'none'}")
    print(f"  {freqs}")
    print(f"pairs: {len(rows)} unordered -> {2 * len(rows)} direction-instances "
          f"(method_id {rows[0]['method_id']}-{rows[-1]['method_id']}, "
          f"plus a _counter dir for each)")
    print(f"  deviance range {min(i['pct_deviance'] for i in index):.1f}% - "
          f"{max(i['pct_deviance'] for i in index):.1f}%")
    print(f"tone: {TONE_DUR_MS} ms, ISI {ISI_MS} ms, SOA {SOA_MS} ms, {INTENSITY_DB} dB, "
          f"p_deviant {P_DEVIANT_PC}%")
    print(f"wrote {args.out}")
    print(f"wrote {args.index_out}")


if __name__ == "__main__":
    main()
