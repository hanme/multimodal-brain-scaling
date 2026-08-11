#!/usr/bin/env python
"""Per-TRIAL MMN scoring at a single site -- the input to the N-effect (Plot 1).

WHY THIS EXISTS
---------------
Every committed MMN verdict is scored on `deviant_mean`, the average of a method's 15 deviant
trials. But the 15 trials are not interchangeable: they are a 3 x 5 grid of
N in {3,5,7} standards-between-deviants x variation in {1..5}, so the N-effect -- more standards
between deviants => deeper MMN, one of the best-replicated properties of the human MMN -- lives
ENTIRELY inside the average the committed pipeline takes. This script re-scores each deviant on
its own, using the identical criterion, so N becomes a variable instead of being averaged away.

Scoring is the committed one, imported rather than reimplemented, with the single trial standing
in for `deviant_mean` everywhere:

  z_diff_i   = zscore_baseline(dev_i, base) - zscore_baseline(standard, base)    [compute_z_diff]
  S2_i       = interior trough in 100-240 ms with >=50% recovery within 120 ms   [trace_stats/decide]
  trough_uv_i= the mean-only baseline-corrected dev_i - standard difference wave in uV,
               sampled at TRIAL i's OWN S2 trough latency                        [uv_diff_wave]
  S7_i       = S2_i AND trough_uv_i <= -0.75 uV

The knobs match analyze_mmn_s7_roi.py exactly (WINDOW, RECOVERY_MS, RECOVERY_FRAC, CTRL_WINDOW,
EDGE_GUARD_BINS), so a per-trial verdict differs from a committed one only because the input is one
trial rather than the 15-trial mean.

INPUT REQUIREMENT -- read this before running
---------------------------------------------
Needs the per-trial deviant stack in the prediction h5. The ELECTRODE driver historically wrote
only `deviant_mean` and discarded the stack, so the committed electrode h5s CANNOT be scored here;
scripts/insilico_mmn_electrodes.py now also writes `deviants_fc` (+ `deviants_fc_electrodes`,
`deviant_ids`), and the two prediction roots must be re-run once for those datasets to exist.
Until then this script exits with that message rather than silently producing nothing.

The legacy `deviants` dataset (full target axis, written by the PARCEL driver and by
insilico_mmn_attn.py) is also accepted, which is what lets the scoring path be validated against
real data -- see --roi -- without waiting for the re-run.

SANITY CHECK (always run, never optional): for N7/var1 the per-trial z-trough must reproduce the
`n7v1_peak` dataset the pipeline already stores (insilico_mmn.py computes exactly this quantity).
A mismatch means this path has diverged from the pipeline's and the run aborts.

mTRF only, one site, one floor. whisper-large is excluded (its predicted uV run ~20-35x every
other model and it was never run in the novel search).

Usage:
    python scripts/analyze_mmn_per_trial_n.py \
        --predictions_root outputs/insilico_mmn_predictions_soafix --dataset lit \
        --out outputs/results_soafix_full/mmn_per_trial_n_fcz.csv
    python scripts/analyze_mmn_per_trial_n.py \
        --predictions_root outputs/insilico_mmn_predictions_novel_phase2 --dataset p2 \
        --out outputs/results_novel_search/phase2_mmn_per_trial_n_fcz.csv
"""
from pathlib import Path
import argparse
import re
import sys

import numpy as np
import h5py
import pandas as pd

# This script lives with the analysis it feeds, not in scripts/, but the scoring primitives it
# MUST reuse (rather than reimplement) do live there -- so scripts/ goes on the path explicitly.
# Found by marker, not by a fixed parents[k] index, so relocating this file cannot break it.
_HERE = Path(__file__).resolve().parent
REPO = next(d for d in (_HERE, *_HERE.parents) if (d / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "scripts"))

from analyze_mmn_criteria import (  # noqa: E402
    iter_prediction_files, zscore_baseline, trace_stats, decide,
    BASELINE_START_MULT, BASELINE_END_MULT,
)
from analyze_mmn_criteria_s5_s6 import uv_diff_wave  # noqa: E402
from novel_search_common import SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD  # noqa: E402

# Scoring knobs -- identical to analyze_mmn_s7_roi.py:55-60. Not exposed as flags: a per-trial
# verdict scored differently from the committed one would not be comparable to it.
WINDOW = (100.0, 240.0)
RECOVERY_MS = 120.0
RECOVERY_FRAC = 0.5
CTRL_WINDOW = (300.0, 440.0)
EDGE_GUARD_BINS = 1

# "method_09_N3_var1_deviant_0000000" -> N=3, variation=1. NOTE: a counter method's deviant ids are
# byte-identical to its regular twin's, so DIRECTION comes from the h5 group name, never from here.
DEV_ID_RE = re.compile(r"_N(?P<n>\d+)_var(?P<var>\d+)_", re.IGNORECASE)

EXPECTED_ROWS = {"lit": 6 * 48 * 15, "p2": 6 * 254 * 15}      # 4,320 and 22,860
EXPECTED_CONDITIONS = {"lit": 48, "p2": 254}
N_LEVELS, N_VARIATIONS = (3, 5, 7), 5


def parse_deviant_id(sid):
    m = DEV_ID_RE.search(sid)
    if not m:
        raise SystemExit(f"deviant id {sid!r} does not carry _N<k>_var<v>_; cannot assign N")
    return int(m.group("n")), int(m.group("var"))


def stack_axis(h5, g):
    """(per-trial stack [n_dev, n_t, k], names of its k target columns), or (None, None).

    Two layouts are accepted:
      deviants_fc + deviants_fc_electrodes  the fronto-central slice the electrode driver now
                                            writes -- k = 7, its own name list
      deviants                              the legacy full-axis stack (parcel driver /
                                            insilico_mmn_attn.py) -- k = every target in the file
    Only the STACK's axis varies. `standard`, `deviant_mean` and `n7v1_peak` are always stored over
    the file's full target list, so they are indexed separately -- see file_targets().
    """
    if "deviants_fc" in g:
        names = [n.decode() if hasattr(n, "decode") else n for n in g["deviants_fc_electrodes"][:]]
        return g["deviants_fc"], names
    if "deviants" in g:
        return g["deviants"], file_targets(h5)
    return None, None


def file_targets(h5):
    """The file's full target list -- the axis of `standard`, `deviant_mean` and `n7v1_peak`."""
    tkey = "parcels" if "parcels" in h5 else "electrodes"
    return [n.decode() if hasattr(n, "decode") else n for n in h5[tkey][:]]


def score_file(h5_path, model, roi, x, native_to_uv):
    """One row per (model, method, deviant trial) at `roi`, plus the n7v1 sanity residuals."""
    rows, residuals = [], []
    lo, hi = WINDOW
    with h5py.File(h5_path, "r") as h5:
        groups = [k for k, v in h5.items() if isinstance(v, h5py.Group)]
        # Whether this vintage carries the per-trial stack is decided BEFORE anything else, so a
        # stackless file is reported as such rather than as a target-name error.
        if not groups or stack_axis(h5, h5[groups[0]])[0] is None:
            return None, None

        all_names = file_targets(h5)
        if roi not in all_names:
            raise SystemExit(f"{h5_path}: {roi!r} is not among its targets {all_names}")
        roi_full = all_names.index(roi)

        for method, g in h5.items():
            if not isinstance(g, h5py.Group):
                continue
            stack, names = stack_axis(h5, g)
            if stack is None:
                raise SystemExit(f"{h5_path}/{method}: per-trial stack missing from this group "
                                 f"but present in others -- the file is inconsistent")
            if roi not in names:
                raise SystemExit(
                    f"{h5_path}: {roi!r} is not on the stored per-trial axis {names}; the stack "
                    f"was sliced to a cluster that excludes it. Use --roi to pick one of those.")
            roi_stack = names.index(roi)

            t = g["time_ms"][:].astype(float)
            std_col = g["standard"][:, roi_full].astype(float)
            soa = float(g.attrs.get("soa_ms", np.nan))
            ids = [i.decode() if hasattr(i, "decode") else i for i in g["deviant_ids"][:]]
            assert stack.shape[0] == len(ids), \
                f"{h5_path}/{method}: {stack.shape[0]} trials but {len(ids)} deviant ids"

            base = (t >= BASELINE_START_MULT * soa) & (t < BASELINE_END_MULT * soa)
            if not base.any():
                base = t < 0
            z_std = zscore_baseline(std_col[:, None], base)[:, 0]

            # method_09_counter -> pair 9, counter. A counter method's deviant ids are identical
            # to its regular twin's, so direction is the GROUP name's business, never the id's.
            is_counter = method.endswith("_counter")
            method_id = int(re.search(r"method_(\d+)", method).group(1))

            for i, sid in enumerate(ids):
                n_std, variation = parse_deviant_id(sid)
                dev_i = stack[i, :, roi_stack].astype(float)
                z_i = zscore_baseline(dev_i[:, None], base)[:, 0] - z_std
                s = trace_stats(t, z_i, lo, hi, RECOVERY_MS, RECOVERY_FRAC,
                                CTRL_WINDOW[0], CTRL_WINDOW[1], EDGE_GUARD_BINS)
                s2 = bool(decide(s).get("S2_recovery", False))
                argmin_ms = s.get("argmin_ms", float("nan"))

                # uV at THIS trial's own S2 trough latency (not the method-mean's).
                uv = uv_diff_wave(t, std_col[:, None], dev_i[:, None], soa, [roi], MAPPING,
                                  None, native_to_uv)[:, 0]
                trough_uv = (float("nan") if np.isnan(argmin_ms)
                             else float(uv[int(np.argmin(np.abs(t - argmin_ms)))]))
                s7 = bool(s2 and not np.isnan(trough_uv) and trough_uv <= -x)
                assert not (s7 and not s2), "S7 must be a subset of S2"

                rows.append(dict(model=model, method=method, method_id=method_id,
                                 direction="counter" if is_counter else "regular",
                                 N=n_std, variation=variation, s2=s2,
                                 trough_uv=round(trough_uv, 5) if np.isfinite(trough_uv)
                                 else float("nan"), s7=s7))

                # Sanity: the pipeline already stored this exact quantity for N7/var1.
                if n_std == 7 and variation == 1 and "n7v1_peak" in g:
                    win = (t >= lo) & (t <= hi)
                    stored = float(g["n7v1_peak"][roi_full])
                    if np.isfinite(stored):
                        residuals.append(abs(float(z_i[win].min()) - stored))
    return rows, residuals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_root", required=True,
                   help="one of outputs/insilico_mmn_predictions_soafix (LIT) or "
                        "outputs/insilico_mmn_predictions_novel_phase2 (NOVEL-P2)")
    p.add_argument("--dataset", required=True, choices=("lit", "p2"),
                   help="tag written into the CSV's `dataset` column; also selects the expected "
                        "row count that is asserted at the end")
    p.add_argument("--out", required=True)
    p.add_argument("--roi", default=ROI, help=f"single reporting target (default {ROI})")
    p.add_argument("--dip_uv_threshold", type=float, default=DIP_UV_THRESHOLD)
    p.add_argument("--native_to_uv", type=float, default=1e6,
                   help="native EEG unit -> uV (Volts->uV = 1e6); mTRF predictions are native V")
    p.add_argument("--n7v1_tol", type=float, default=1e-4,
                   help="max |per-trial N7/var1 z-trough - stored n7v1_peak| before aborting")
    p.add_argument("--skip_count_assert", action="store_true",
                   help="for partial/validation runs over a subset of models or methods")
    args = p.parse_args()

    root = Path(args.predictions_root)
    rows, residuals, seen_models, no_stack = [], [], set(), []
    for h5_path, model, level, mapping in iter_prediction_files(root):
        if mapping != MAPPING or model not in SEARCH_MODELS:
            continue                                    # mTRF only; whisper-large excluded HERE
        r, res = score_file(h5_path, model, args.roi, args.dip_uv_threshold, args.native_to_uv)
        if r is None:
            no_stack.append(h5_path)
            continue
        rows.extend(r)
        residuals.extend(res)
        seen_models.add(model)

    if not rows:
        raise SystemExit(
            f"No per-trial deviant stack under {root}.\n"
            f"  Scanned {len(no_stack)} mTRF file(s), none carrying `deviants_fc` or `deviants`.\n"
            f"  The electrode driver discarded the stack until the deviants_fc patch; this root "
            f"must be re-run once (Part A2) before the N-effect can be built.")

    # The sanity gate: this path must reproduce the pipeline's own N7/var1 number.
    if residuals:
        worst = max(residuals)
        print(f"n7v1 sanity: max |per-trial z-trough - stored n7v1_peak| = {worst:.3e} "
              f"over {len(residuals)} methods (tol {args.n7v1_tol:g})")
        if worst > args.n7v1_tol:
            raise SystemExit(
                f"per-trial scoring DIVERGED from the pipeline (residual {worst:.3e} > "
                f"{args.n7v1_tol:g}). Fix before plotting -- the trials are not the same traces.")
    else:
        print("WARNING: no N7/var1 trial found, so the pipeline sanity check did not run")

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", args.dataset)

    assert "whisper-large" not in set(df["model"]), "whisper-large leaked into the per-trial CSV"
    if not args.skip_count_assert:
        n_cond, n_rows = EXPECTED_CONDITIONS[args.dataset], EXPECTED_ROWS[args.dataset]
        assert df["method"].nunique() == n_cond, \
            f"{df['method'].nunique()} conditions, expected {n_cond}"
        assert len(seen_models) == len(SEARCH_MODELS), \
            f"models {sorted(seen_models)}, expected {SEARCH_MODELS}"
        assert len(df) == n_rows, f"{len(df)} rows, expected {n_rows}"
        # 15 trials = 3 N-levels x 5 variations, in every (model, condition).
        per_cond = df.groupby(["model", "method"], observed=True).size()
        assert (per_cond == 15).all(), \
            f"{int((per_cond != 15).sum())} (model, condition) cells do not have 15 trials"
        assert sorted(df["N"].unique()) == list(N_LEVELS), f"N levels {sorted(df['N'].unique())}"
        assert df["variation"].nunique() == N_VARIATIONS

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df['model'].nunique()} models x "
          f"{df['method'].nunique()} conditions x 15 trials) -> {out}")

    print(f"\nS2 / S7@{args.dip_uv_threshold:g} per N at {args.roi} (denominator = all trials):")
    print(f"  {'N':<5}{'n':>7}{'S2':>8}{'S7':>8}{'S7 rate':>10}{'med trough_uv (S7)':>22}")
    for n in N_LEVELS:
        d = df[df["N"] == n]
        med = d.loc[d["s7"], "trough_uv"].median() if d["s7"].any() else float("nan")
        print(f"  {n:<5}{len(d):>7}{int(d['s2'].sum()):>8}{int(d['s7'].sum()):>8}"
              f"{d['s7'].mean():>10.3f}{med:>22.3f}")


if __name__ == "__main__":
    main()
