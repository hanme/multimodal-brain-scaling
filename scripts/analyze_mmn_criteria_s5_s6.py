"""Retired old S4 (control-window specificity) and added a redefined S4, plus
deadline-bound S5/S6, on top of analyze_mmn_criteria's C0-S3.

WHY THIS EXISTS
---------------
C0-S3 (scripts/analyze_mmn_criteria.py) pre-clip the trough search to the fixed
100-240 ms scoring window. A genuine dip-and-recover trough that happens to sit outside
that window (confirmed in real runs -- see the flagged smooth-ramp cases in
aux/mmn_criterion_investigation.md, whose true troughs sit near 290-300 ms) is invisible
to every C0-S3 criterion, regardless of which ROI variant scores it.

This script adds:
  S4  -- tone-end-relative dip+recovery, ANY qualifying sample (not just a window's
         argmin). For each method's final/critical tone (time_ms==0 is its onset; its
         duration is looked up per-method from --duration_csv and used as
         tone_end_ms), scan every sample in [tone_end_ms, tone_end_ms+250ms]: a sample
         qualifies if it is < 0 and recovers >=recovery_frac of its own depth by the
         ABSOLUTE deadline tone_end_ms+400ms. S4 is true iff any sample qualifies. This
         supersedes the old S4 (control-window specificity, amc.decide()'s
         S4_specificity), which required the in-window argmin itself to be the
         recovering point -- amc.decide()'s S4_specificity is still computed here for
         the "current" window as a side effect of reusing decide(), but only as a
         debug-only current__S4_specificity column; it is retired and must not be
         reported as "S4" in any table.
  S5  -- trough<0 found by an UNBOUND search over the entire available post-onset
         trace [0, t.max()] (unchanged from before -- still via amc.trace_stats), but
         recovery is now checked against the same ABSOLUTE deadline as S4
         (tone_end_ms+400ms) instead of a fixed 120ms-after-trough window. t.max()
         varies per run (mTRF vs encoder lag windows differ), so both the trough search
         and the deadline are computed fresh per run.
  S6  -- S5 AND the located trough's latency falls inside a literature-derived envelope
         (default 90-250 ms, --envelope; see aux/mmn_criterion_investigation.md's
         union-across-10-papers cross-check). Definition unchanged; inherits the new S5.

It reuses analyze_mmn_criteria.py's trace_stats/decide/window_mask/compute_z_diff/
iter_prediction_files/roi_indices verbatim (imported, not duplicated) for the "current"
window and for locating the global trough, and -- following the precedent set by
analyze_mmn_criteria_fz_central.py -- supports a --roi_variant flag that monkey-patches
ELECTRODE_ROI/PARCEL_ROI on the imported module for the fz_central case, so one script
produces both deliverables. New S4 and the deadline-bound S5 recovery test are NOT
implemented via trace_stats/decide -- see dip_recovery_scan()/deadline_bound_recovery()
below.

decide() also returns C0_current/S1_interior/S3_interior_recovery/S4_specificity for
the global window as a side effect of reusing the same function. These are NOT
physically meaningful on an unbounded window (interior-edge-guard is nearly vacuous;
specificity's fixed 300-440 ms control window has no well-defined relationship to a
freely-located global trough) -- they are written as debug-only global__debug_* columns
and must not be reported as S5/S6 in any table.

READ-ONLY over the prediction h5s and --duration_csv. Does not modify
analyze_mmn_criteria.py, analyze_mmn_criteria_fz_central.py, trace_stats, or decide.

Usage:
    python scripts/analyze_mmn_criteria_s5_s6.py \
        --predictions_root outputs/insilico_mmn_predictions \
        --roi_variant current \
        --out outputs/results/mmn_criteria_s5_s6.csv

    python scripts/analyze_mmn_criteria_s5_s6.py \
        --predictions_root outputs/insilico_mmn_predictions \
        --roi_variant fz_central \
        --out outputs/results/mmn_criteria_s5_s6_fz_central.csv
"""

from pathlib import Path
import argparse
import csv

import numpy as np
import h5py

import analyze_mmn_criteria as amc

CRITS = ["C0_current", "S1_interior", "S2_recovery", "S3_interior_recovery", "S4_specificity"]

# How far past tone-end the new S4 dip window and the S4/S5 recovery deadline extend.
S4_DIP_SPAN_MS = 250.0
RECOVERY_DEADLINE_SPAN_MS = 400.0


def load_duration_map(duration_csv):
    """{method_id (int): (standard_dur, deviant_dur)} from the literature metadata CSV."""
    out = {}
    with open(duration_csv, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["method_id"])] = (float(row["standard_dur"]), float(row["deviant_dur"]))
    return out


def parse_method_id(method_str):
    """'method_27' -> 27, 'method_27_counter' -> 27."""
    return int(method_str.split("_")[1])


def tone_end_ms_for(method_id, duration_map):
    """Final/critical tone's duration (== its end, since time_ms==0 is its onset)."""
    std_dur, dev_dur = duration_map[method_id]
    assert std_dur == dev_dur, (
        f"method_id {method_id}: standard_dur ({std_dur}) != deviant_dur ({dev_dur}) -- "
        "new S4/S5 assume a single tone_end_ms per method; this method needs a real fix, "
        "not a silent pick of one duration over the other.")
    return dev_dur


def dip_recovery_scan(t, z, tone_end_ms, dip_span_ms=S4_DIP_SPAN_MS,
                       deadline_span_ms=RECOVERY_DEADLINE_SPAN_MS, recovery_frac=0.5):
    """New S4: scan every sample in [tone_end_ms, tone_end_ms+dip_span_ms] (not just the
    window's argmin) for one that dips below 0 and recovers >=recovery_frac of its own
    depth by the absolute deadline tone_end_ms+deadline_span_ms. Returns the FIRST
    (earliest) qualifying sample, if any.
    """
    tmax = float(t.max())
    dip_hi = tone_end_ms + dip_span_ms
    if dip_hi > tmax:
        print(f"  CLIP dip window: tone_end_ms+{dip_span_ms:.0f}={dip_hi:.1f}ms > "
              f"t.max()={tmax:.1f}ms")
        dip_hi = tmax
    deadline = tone_end_ms + deadline_span_ms
    if deadline > tmax:
        print(f"  CLIP recovery deadline: tone_end_ms+{deadline_span_ms:.0f}="
              f"{deadline:.1f}ms > t.max()={tmax:.1f}ms")
        deadline = tmax

    qualifies = False
    qualifying_ms = float("nan")
    idxs = np.where((t >= tone_end_ms) & (t <= dip_hi))[0]
    for i in idxs:
        depth = float(z[i])
        if depth >= 0.0:
            continue
        post = z[(t > t[i]) & (t <= deadline)]
        if post.size == 0:
            continue
        rise = float(post.max()) - depth
        frac = rise / abs(depth)
        if frac >= recovery_frac:
            qualifies = True
            qualifying_ms = float(t[i])
            break

    return dict(qualifies=qualifies, qualifying_argmin_ms=qualifying_ms,
                tone_end_ms=tone_end_ms, dip_window_hi_ms=dip_hi, deadline_ms=deadline)


def deadline_bound_recovery(t, z, imin_ms, depth, deadline_ms, recovery_frac):
    """S5's recovery test: has the global trough recovered by an ABSOLUTE deadline,
    rather than by a fixed duration after the trough?
    """
    post = z[(t > imin_ms) & (t <= deadline_ms)]
    if post.size == 0:
        return False, 0.0
    rise = float(post.max()) - depth
    frac = rise / abs(depth)
    return frac >= recovery_frac, frac


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_root", default="outputs/insilico_mmn_predictions")
    p.add_argument("--out", default="outputs/results/mmn_criteria_s5_s6.csv")
    p.add_argument("--roi_variant", choices=["current", "fz_central"], default="current")
    p.add_argument("--window", type=float, nargs=2, default=(100.0, 240.0))
    p.add_argument("--envelope", type=float, nargs=2, default=(90.0, 250.0),
                   help="literature-derived plausible MMN-latency envelope for S6 "
                        "(aux/mmn_criterion_investigation.md union of 10 source papers)")
    p.add_argument("--recovery_ms", type=float, default=120.0)
    p.add_argument("--recovery_frac", type=float, default=0.5)
    p.add_argument("--ctrl_window", type=float, nargs=2, default=(300.0, 440.0))
    p.add_argument("--edge_guard_bins", type=int, default=1)
    p.add_argument("--duration_csv",
                   default="data/metadata/literature_frequency_intensity_duration_metadata.csv",
                   help="per-method tone-duration lookup for the tone-end-relative S4/S5")
    args = p.parse_args()

    if args.roi_variant == "fz_central":
        amc.ELECTRODE_ROI = {"Fz"}
        amc.PARCEL_ROI = {"central"}

    cur_lo, cur_hi = args.window
    env_lo, env_hi = args.envelope
    duration_map = load_duration_map(args.duration_csv)

    rows = []
    n_files = 0
    tmax_by_mapping = {}

    for h5_path, model, level, mapping in amc.iter_prediction_files(Path(args.predictions_root)):
        n_files += 1
        with h5py.File(h5_path, "r") as h5:
            tkey = "parcels" if "parcels" in h5 else "electrodes"
            tnames = [t.decode() if hasattr(t, "decode") else t for t in h5[tkey][:]]
            roi = amc.roi_indices(tnames, level)
            for method, g in h5.items():
                if not isinstance(g, h5py.Group):
                    continue
                t = g["time_ms"][:].astype(float)
                std = g["standard"][:].astype(float)
                dev = g["deviant_mean"][:].astype(float)
                soa = float(g.attrs.get("soa_ms", np.nan))
                z = amc.compute_z_diff(t, std, dev, soa)

                roi_use = roi or list(range(len(tnames)))
                z_roi = z[:, roi_use].mean(1)

                tmax = float(t.max())
                tmax_by_mapping.setdefault(mapping, []).append(tmax)

                method_id = parse_method_id(method)
                tone_end_ms = tone_end_ms_for(method_id, duration_map)

                row = dict(model=model, level=level, mapping=mapping, method=method,
                           source=g.attrs.get("source", ""),
                           label=g.attrs.get("context_final", ""),
                           soa_ms=soa, n_roi_targets=len(roi_use))

                # ---- "current" window: reproduces analyze_mmn_criteria.py exactly ----
                s_cur = amc.trace_stats(t, z_roi, cur_lo, cur_hi, args.recovery_ms,
                                        args.recovery_frac, args.ctrl_window[0],
                                        args.ctrl_window[1], args.edge_guard_bins)
                v_cur = amc.decide(s_cur)
                row["current_peak"] = round(s_cur.get("peak", float("nan")), 4)
                row["current_argmin_ms"] = s_cur.get("argmin_ms", float("nan"))
                row["current_interior"] = s_cur.get("interior", False)
                row["current_recovery_frac"] = round(s_cur.get("recovery_frac", 0.0), 3)
                row["current_recovered"] = s_cur.get("recovered", False)
                row["current_ctrl_peak"] = round(s_cur.get("ctrl_peak", float("nan")), 4)
                for cname, val in v_cur.items():
                    # current__S4_specificity is written here too (old, retired definition);
                    # debug-only -- never surfaced as "S4" downstream.
                    row[f"current__{cname}"] = val

                # ---- new S4: tone-end-relative dip+recovery, any qualifying sample ----
                s4 = dip_recovery_scan(t, z_roi, tone_end_ms, recovery_frac=args.recovery_frac)
                row["s4__dip_recovery"] = s4["qualifies"]
                row["s4__qualifying_argmin_ms"] = s4["qualifying_argmin_ms"]
                row["s4_tone_end_ms"] = s4["tone_end_ms"]
                row["s4_dip_window_hi_ms"] = s4["dip_window_hi_ms"]

                # ---- "global" window: [0, t.max()] computed fresh per run ----
                s_glob = amc.trace_stats(t, z_roi, 0.0, tmax, args.recovery_ms,
                                         args.recovery_frac, args.ctrl_window[0],
                                         args.ctrl_window[1], args.edge_guard_bins)
                v_glob = amc.decide(s_glob)
                global_argmin_ms = s_glob.get("argmin_ms", float("nan"))
                global_peak = s_glob.get("peak", 0.0)

                # S5: deadline-bound recovery from the global trough, NOT trace_stats's
                # own fixed-120ms-after-trough "recovered"/"recovery_frac" fields.
                recovery_deadline = min(tone_end_ms + RECOVERY_DEADLINE_SPAN_MS, tmax)
                recovered5, frac5 = deadline_bound_recovery(
                    t, z_roi, global_argmin_ms, global_peak, recovery_deadline,
                    args.recovery_frac)
                s5 = bool(global_peak < 0.0 and recovered5)
                s6 = bool(s5 and not np.isnan(global_argmin_ms)
                         and env_lo <= global_argmin_ms <= env_hi)

                row["global_peak"] = round(global_peak, 4)
                row["global_argmin_ms"] = global_argmin_ms
                row["global_recovery_frac"] = round(frac5, 3)
                row["global_recovery_deadline_ms"] = recovery_deadline
                row["global__S5"] = s5
                row["global__S6_envelope_recovery"] = s6
                # debug-only, not physically meaningful on an unbounded window -- see docstring
                for cname, val in v_glob.items():
                    if cname == "S2_recovery":
                        continue
                    row[f"global__debug_{cname}"] = val

                rows.append(row)

    if not rows:
        print(f"No prediction HDF5s found under {args.predictions_root} -- nothing to do.")
        return

    print("\n" + "=" * 78)
    print("t.max() sanity check (post-onset extent of the 'global' window), by mapping")
    print("=" * 78)
    for mapping, vals in sorted(tmax_by_mapping.items()):
        vals = np.array(vals)
        print(f"  {mapping:<10} n={len(vals):>4}  min={vals.min():.1f}ms  "
              f"max={vals.max():.1f}ms  mean={vals.mean():.1f}ms")

    lead = ["model", "level", "mapping", "method", "source", "label", "soa_ms",
            "n_roi_targets"]
    fieldnames = lead + sorted({k for r in rows for k in r} - set(lead))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nROI variant: {args.roi_variant}  "
          f"(ELECTRODE_ROI={amc.ELECTRODE_ROI}, PARCEL_ROI={amc.PARCEL_ROI})")
    print(f"Wrote {len(rows)} rows -> {args.out}")

    total = len(rows)
    print(f"\nS5 present: {sum(1 for r in rows if r['global__S5'])}/{total}")
    print(f"S6 present: {sum(1 for r in rows if r['global__S6_envelope_recovery'])}/{total}")


if __name__ == "__main__":
    main()
