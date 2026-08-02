#!/usr/bin/env python3
"""Verify a trailing-floor re-screen: the epoch is long enough, and the unchanged half didn't move.

The bug this checks for is silent. S2's recovery test in analyze_mmn_criteria.trace_stats is
`post = z[imin+1 : imin+1+rb]` -- a bare Python slice with no end guard. The trough may sit as late
as 240 ms and the search runs 120 ms past it, so the criterion reads out to 360 ms; when the epoch
ends first it is evaluated on whatever samples happen to exist, and when the trough lands on the
final sample `post` is empty and S2 returns False without ever being tested. Nothing errors.

After regenerating the short-SOA stimuli with a 400 ms trailing floor, every instance should have a
complete epoch. This script proves it, over all 7 models x 48 conditions at FCz/mtrf, and separately
proves that the 24 conditions whose audio was byte-identical scored identically too.

    python scripts/verify_soafix_predictions.py \
        --predictions_root outputs/insilico_mmn_predictions_soafix \
        --new_s7_csv outputs/results_soafix/mmn_s7_roi.csv

Pass --baseline_predictions_root to print the before-picture alongside the after.
Exit code is the contract: 0 all good, 1 something is wrong.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import h5py
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_mmn_criteria import (                                       # noqa: E402
    iter_prediction_files, compute_z_diff, window_mask)
from analyze_mmn_s7_roi import WINDOW, RECOVERY_MS                       # noqa: E402

ROI = "FCz"
MAPPING = "mtrf"
LEVEL = "electrodes"

WHISPER_MODELS = ["whisper-tiny", "whisper-base", "whisper-small",
                  "whisper-medium", "whisper-large"]
WAV2VEC2_MODELS = ["wav2vec2-medium", "wav2vec2-large"]
EXPECTED_MODELS = set(WHISPER_MODELS + WAV2VEC2_MODELS)

# Ids whose SOA >= 400: their audio is byte-identical under the floor, so their scores must be too.
UNCHANGED_IDS = [9, 12, 20, 21, 27, 33, 43, 44, 55, 72, 74, 75]
CHANGED_IDS = [10, 17, 18, 19, 28, 29, 30, 31, 32, 37, 53, 60]

TRAILING_FLOOR_MS = 400.0
MIN_TAIL_MS = 360.0          # the full extent of the data any S7 verdict depends on

# max(time_ms) = reserved_tail - EDGE, where the epoch loses one feature frame relative to the
# audio: the last frame starts 20 ms before the clip ends, and wav2vec2's conv stack drops a
# further 20 ms (stride 320, receptive field 400 samples => 499 frames for a 10 s clip, not 500).
# NB the reserved tail is max(floor, SOA), NOT the SOA -- for the 12 short-SOA ids those differ,
# so `soa_ms - max(time_ms)` reads as low as -180.9 post-fix by design. It is the reserved tail
# that is invariant.
EDGE_RANGE = {"whisper": (19.0, 19.25), "wav2vec2": (38.875, 39.25)}
EXPECTED_N_T = {"whisper": 1460, "wav2vec2": 459}


def family_of(model):
    return "wav2vec2" if model.startswith("wav2vec2") else "whisper"


def condition_names(ids):
    return ([f"method_{i:02d}" for i in ids] + [f"method_{i:02d}_counter" for i in ids])


def scan_file(h5_path):
    """Per-condition epoch + S2-reach measurements for one model's prediction file."""
    rows = []
    with h5py.File(h5_path, "r") as h:
        names = [e.decode() if isinstance(e, bytes) else str(e) for e in h["electrodes"][:]]
        if ROI not in names:
            raise KeyError(f"{ROI} not among the {len(names)} electrodes in {h5_path}")
        # Electrode order differs between families (the NC floor is computed on different neural
        # files), so index by name -- never by a hardcoded position.
        idx = names.index(ROI)
        groups = [k for k in h if isinstance(h[k], h5py.Group)]
        for method in sorted(groups):
            g = h[method]
            t = g["time_ms"][:].astype(float)
            soa = float(g.attrs["soa_ms"])
            z = compute_z_diff(t, g["standard"][:].astype(float),
                               g["deviant_mean"][:].astype(float), soa)[:, idx]
            dt = float(t[1] - t[0])
            win_idx = np.where(window_mask(t, *WINDOW))[0]
            imin = int(win_idx[int(np.argmin(z[win_idx]))]) if win_idx.size else -1
            rb = max(1, int(round(RECOVERY_MS / dt)))
            n_post = z[imin + 1: imin + 1 + rb].size if imin >= 0 else 0
            rows.append(dict(
                method=method, soa_ms=soa, n_t=len(t), dt=dt, max_t=float(t.max()),
                reserved=max(TRAILING_FLOOR_MS, soa),
                n_win=int(win_idx.size), imin=imin, argmin_ms=float(t[imin]) if imin >= 0 else np.nan,
                trough_on_last=(imin == len(t) - 1), rb=rb, n_post=n_post,
                truncated=(n_post < rb)))
    return rows


def collect(root, label):
    """{model: [row, ...]} for every mtrf/electrode prediction file under root.

    iter_prediction_files strips a -parcels/-electrodes suffix off the directory name, so two
    directories can map to the same model. In the committed baseline the <model>-electrodes ones
    hold only encoder (__attn.h5) files and are filtered out here -- but a second mtrf file would
    otherwise overwrite the first and be scored silently, so refuse instead.
    """
    found, seen = {}, {}
    for h5_path, model, level, mapping in iter_prediction_files(Path(root)):
        if mapping != MAPPING or level != LEVEL:
            continue
        if model in found:
            raise RuntimeError(f"two {MAPPING}/{LEVEL} files map to model '{model}' under {root}: "
                               f"{seen[model]} and {h5_path}")
        seen[model] = h5_path
        found[model] = scan_file(h5_path)
    if not found:
        print(f"WARNING: no {MAPPING}/{LEVEL} prediction files under {root} ({label})")
    return found


def report(per_model, floor_expected, failures, label):
    print(f"\n{label}")
    print(f"{'model':<17} {'n':>4} {'reserved-max_t':>16} {'min max_t':>10} {'n_win':>10} "
          f"{'trough@end':>11} {'truncated':>10}")
    for model in sorted(per_model, key=lambda m: (family_of(m), m)):
        rows = per_model[model]
        fam = family_of(model)
        gaps = [r["reserved"] - r["max_t"] for r in rows]
        max_ts = [r["max_t"] for r in rows]
        n_wins = sorted({r["n_win"] for r in rows})
        n_last = sum(r["trough_on_last"] for r in rows)
        n_trunc = sum(r["truncated"] for r in rows)
        print(f"{model:<17} {len(rows):>4} {min(gaps):>7.3f}-{max(gaps):<8.3f} "
              f"{min(max_ts):>10.3f} {str(n_wins):>10} {n_last:>11} {n_trunc:>10}")

        if not floor_expected:
            continue

        if len(rows) != 48:
            failures.append(f"{model}: {len(rows)} conditions, expected 48")
        lo, hi = EDGE_RANGE[fam]
        bad = [r for r in rows if not (lo <= r["reserved"] - r["max_t"] <= hi)]
        if bad:
            failures.append(f"{model}: {len(bad)} conditions outside the {lo}-{hi} ms edge for "
                            f"{fam} (e.g. {bad[0]['method']}: reserved {bad[0]['reserved']:.0f} - "
                            f"max_t {bad[0]['max_t']:.3f} = {bad[0]['reserved']-bad[0]['max_t']:.3f})")
        short = [r for r in rows if r["max_t"] < MIN_TAIL_MS]
        if short:
            failures.append(f"{model}: {len(short)} conditions end before {MIN_TAIL_MS:g} ms "
                            f"(shortest {min(r['max_t'] for r in short):.3f} at "
                            f"{min(short, key=lambda r: r['max_t'])['method']})")
        on_last = [r["method"] for r in rows if r["trough_on_last"]]
        if on_last:
            failures.append(f"{model}: trough on the final sample -- S2 untested -- for {on_last}")
        trunc = [(r["method"], r["imin"], r["n_t"], r["n_post"]) for r in rows if r["truncated"]]
        if trunc:
            failures.append(f"{model}: {len(trunc)} truncated recovery searches "
                            f"(method, imin, n_t, n_post): {trunc[:5]}")
        wrong_nt = {r["n_t"] for r in rows} - {EXPECTED_N_T[fam]}
        if wrong_nt:
            failures.append(f"{model}: unexpected frame counts {wrong_nt}, expected "
                            f"{EXPECTED_N_T[fam]}")


def diff_scores(new_csv, baseline_csv, failures):
    """The 24 byte-identical conditions must score bit-identically; the other 24 must not."""
    print(f"\nScore diff vs {baseline_csv}")
    new = pd.read_csv(new_csv)
    base = pd.read_csv(baseline_csv)
    keys = ["model", "mapping", "method", "roi_kind", "roi", "dip_uv_threshold"]
    cols = ["s2", "s7", "trough_uv", "min_uv", "is_counter"]

    def prep(df, methods):
        d = df[(df.mapping == MAPPING) & (df.roi_kind == "electrode")
               & (df.method.isin(methods))].copy()
        # Float join key: identical decimal text parses to identical doubles, but stringifying
        # removes the whole class of bug.
        d["dip_uv_threshold"] = d["dip_uv_threshold"].map(lambda x: f"{x:.3f}")
        return d.set_index(keys).sort_index()

    unchanged = condition_names(UNCHANGED_IDS)
    a, b = prep(new, unchanged), prep(base, unchanged)
    n_models = len(EXPECTED_MODELS)
    expected_rows = len(unchanged) * n_models * 7 * 7          # 7 electrodes x 7 thresholds
    print(f"  unchanged conditions: {len(a)} new rows, {len(b)} baseline rows "
          f"(expect {expected_rows})")
    if len(a) != expected_rows or len(b) != expected_rows:
        failures.append(f"unchanged-condition row counts {len(a)}/{len(b)} != {expected_rows}; "
                        f"the join below is not comparing what it claims to")

    only_new, only_base = a.index.difference(b.index), b.index.difference(a.index)
    if len(only_new) or len(only_base):
        failures.append(f"unchanged conditions: {len(only_new)} keys only in the new run, "
                        f"{len(only_base)} only in the baseline")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    for c in cols:
        if a[c].dtype == bool or b[c].dtype == bool:
            n_diff = int((a[c].astype(bool) != b[c].astype(bool)).sum())
            if n_diff:
                failures.append(f"unchanged conditions: {n_diff} rows differ in '{c}'. With "
                                f"byte-identical audio this is arithmetically impossible -- "
                                f"something leaked between the runs.")
        else:
            d = (a[c].astype(float) - b[c].astype(float)).abs()
            n_exact = int((d == 0).sum())
            print(f"  {c:<10}: {n_exact}/{len(d)} exact, max|delta| {d.max():.3e}")
            if n_exact != len(d):
                # The CSV carries round(x, 5), so identical arrays give identical text. A handful
                # of 1e-5 flips is a rounding cliff, not a real change; anything larger is real.
                if d.max() > 1e-3:
                    failures.append(f"unchanged conditions: '{c}' differs by up to {d.max():.3e}")
                else:
                    print(f"    note: {len(d) - n_exact} rows differ within the 5-dp rounding "
                          f"cliff (max {d.max():.3e}) -- negligible, but not bit-identical")

    # A no-op run would sail through the check above; make sure the other half actually moved.
    changed = condition_names(CHANGED_IDS)
    ca, cb = prep(new, changed), prep(base, changed)
    common_c = ca.index.intersection(cb.index)
    if len(common_c):
        moved = sum(int((ca.loc[common_c, c].astype(float)
                         != cb.loc[common_c, c].astype(float)).sum())
                    for c in ["trough_uv", "min_uv"])
        print(f"  changed conditions:   {moved} differing cells across trough_uv/min_uv "
              f"(over {len(common_c)} rows)")
        if moved == 0:
            failures.append("the 24 changed conditions scored identically to the baseline -- the "
                            "regenerated audio never reached the screen")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_root", default="outputs/insilico_mmn_predictions_soafix")
    p.add_argument("--baseline_predictions_root", default=None,
                   help="optional; prints the pre-fix picture alongside for the report")
    p.add_argument("--new_s7_csv", default=None,
                   help="output of analyze_mmn_s7_roi.py on the new predictions")
    p.add_argument("--baseline_s7_csv", default="outputs/results_24freq_7models/mmn_s7_roi.csv")
    args = p.parse_args()

    failures = []
    print("=" * 88)
    print(f"Trailing-floor prediction check -- {ROI}, {MAPPING}, {LEVEL}")
    print("=" * 88)

    if args.baseline_predictions_root:
        before = collect(args.baseline_predictions_root, "baseline")
        report(before, False, failures, "BEFORE (committed baseline)")

    after = collect(args.predictions_root, "new")
    if not after:
        print(f"\nERROR: nothing to check under {args.predictions_root}")
        return 1
    report(after, True, failures, "AFTER (trailing floor)")

    missing = EXPECTED_MODELS - set(after)
    if missing:
        failures.append(f"missing models {sorted(missing)} -- MODELS was probably left at the six "
                        f"SEARCH_MODELS, silently dropping whisper-large")

    total = sum(len(v) for v in after.values())
    print(f"\n{total} cells checked ({len(after)} models x 48 conditions)")

    if args.new_s7_csv:
        if not Path(args.baseline_s7_csv).exists():
            failures.append(f"baseline score CSV not found: {args.baseline_s7_csv}")
        else:
            diff_scores(args.new_s7_csv, args.baseline_s7_csv, failures)
    else:
        print("\n(skipping the score diff -- pass --new_s7_csv to compare against the baseline)")

    print()
    print("=" * 88)
    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS -- every epoch reaches the criteria window, no trough on a final sample, no "
          "truncated recovery search, and the unchanged half is untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
