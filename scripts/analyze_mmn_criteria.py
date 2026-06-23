"""Compare the current MMN presence/absence criterion against shape-aware
candidates, directly on the per-run prediction HDF5s.

WHY THIS EXISTS
---------------
The committed verdict (insilico_mmn.finalize_method) is:

    baseline_normalized_peak = min( z_diff[100..240 ms] )        # MMN present := < 0

where z_diff = z(deviant_mean) - z(standard), each z-scored within a per-method
baseline window of [-3*SOA, 0) ms. One structural weakness is flagged in
aux/results_analysis.md: no shape constraint -- a monotonic ramp whose minimum
happens to land at the window's right edge scores the same as a genuine
dip-and-recover trough.

The 100-240 ms scoring window itself was cross-checked against the 10 source papers'
own MMN windows (see aux/mmn_criterion_investigation.md) and kept as-is; this script
always scores that single window and does not vary it.

This script recomputes z_diff from the stored RAW `standard`/`deviant_mean` (so it
reproduces `peak` exactly -- it asserts this) and then scores several shape-aware
criteria against that window, with special attention to the already-flagged
smooth-ramp cases and the borderline (|peak| < 0.1) cases.

It is READ-ONLY over the prediction h5s and writes one tidy CSV + a console summary.
Nothing here changes the pipeline; it is a decision aid.

The reduced local outputs/results/mmn_results_table.csv does NOT contain the per-time
curves, so this must run where the prediction h5s live (the cluster). Directory layout
matches scripts/build_mmn_results_table.py:

    <root>/<model>/predictions__<layer>.h5                       mTRF, parcels
    <root>/<model>/electrode_predictions__<layer>.h5             mTRF, electrodes
    <root>/<model>-<level>/<method>/predictions__<layer>__attn.h5 encoder

Usage:
    python scripts/analyze_mmn_criteria.py \
        --predictions_root outputs/insilico_mmn_predictions \
        --out outputs/results/mmn_criteria_comparison.csv
"""

from pathlib import Path
import argparse
import csv

import numpy as np
import h5py

# Verdict ROIs -- identical to results_analysis.md / insilico_mmn_electrodes.FC_ROI.
PARCEL_ROI = {"frontal", "central"}
ELECTRODE_ROI = {"Fz", "FCz", "Cz", "FC1", "FC2", "F1", "F2"}

# Baseline window multipliers (must match insilico_mmn.finalize_method defaults).
BASELINE_START_MULT = -3.0
BASELINE_END_MULT = 0.0

# "Borderline" band around the threshold, per the caveats note (treat |peak|<0.1 as
# "no clear evidence either way").
BORDERLINE = 0.1


# --------------------------------------------------------------------------------------
# z-scoring -- a faithful re-implementation of insilico_mmn.finalize_method's z()/peak.
# --------------------------------------------------------------------------------------
def zscore_baseline(sig, base_mask):
    """Full z-score within base_mask (mean & std), matching finalize_method.z()."""
    mu = sig[base_mask].mean(0, keepdims=True)
    sd = sig[base_mask].std(0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (sig - mu) / sd


def compute_z_diff(time_ms, standard, deviant_mean, soa_ms):
    """Reconstruct z_diff [n_t, n_target] exactly as the pipeline does for the verdict."""
    base = (time_ms >= BASELINE_START_MULT * soa_ms) & (time_ms < BASELINE_END_MULT * soa_ms)
    if not base.any():  # extremely short pre-onset baseline -- shouldn't happen, but be safe
        base = time_ms < 0
    z_dev = zscore_baseline(deviant_mean, base)
    z_std = zscore_baseline(standard, base)
    return z_dev - z_std


# --------------------------------------------------------------------------------------
# Per-target / per-trace shape statistics within a scoring window.
# --------------------------------------------------------------------------------------
def window_mask(time_ms, lo, hi):
    return (time_ms >= lo) & (time_ms <= hi)


def trace_stats(time_ms, z, lo, hi, recovery_ms, recovery_frac,
                ctrl_lo, ctrl_hi, edge_guard_bins):
    """All shape statistics for a single 1-D z_diff trace `z` over scoring window [lo,hi].

    Returns a dict of raw measurements; the present/absent decisions are derived from
    these in `decide()` so every criterion reads the same underlying numbers.
    """
    win = window_mask(time_ms, lo, hi)
    win_idx = np.where(win)[0]
    out = dict(n_win=int(win.sum()))
    if win_idx.size == 0:
        return out

    zw = z[win_idx]
    k = int(np.argmin(zw))            # position of the minimum *within* the window
    imin = int(win_idx[k])            # ...as an index into the full trace
    depth = float(zw[k])              # the baseline_normalized_peak for this window
    out.update(peak=depth, argmin_ms=float(time_ms[imin]),
               argmin_pos=k, n_win_idx=int(win_idx.size))

    # (S1) interior: minimum is not within `edge_guard_bins` of either window edge.
    out["interior"] = (k >= edge_guard_bins) and (k <= win_idx.size - 1 - edge_guard_bins)

    # (S2) recovery: after the trough, does z rise back toward 0 by >= frac*|depth|
    # within recovery_ms? Searched over the FULL trace after imin (not clipped to window),
    # so a genuine in-window trough that recovers just past the edge still counts, while a
    # ramp that keeps falling does not.
    rb = max(1, int(round(recovery_ms / float(time_ms[1] - time_ms[0]))))
    post = z[imin + 1: imin + 1 + rb]
    if post.size and depth < 0:
        rise = float(post.max()) - depth            # how far it climbed back up
        out["recovery_frac"] = rise / abs(depth)
        out["recovered"] = out["recovery_frac"] >= recovery_frac
    else:
        out["recovery_frac"] = 0.0
        out["recovered"] = False

    # (S4) control-window specificity: is the in-window min more negative than the min of
    # a later window with no expected MMN? A ramp's true trough is later, so the control
    # window is at least as negative -> fails specificity.
    cwin = window_mask(time_ms, ctrl_lo, ctrl_hi)
    out["ctrl_peak"] = float(z[cwin].min()) if cwin.any() else float("nan")
    out["more_neg_than_ctrl"] = (depth < out["ctrl_peak"]) if cwin.any() else False
    return out


def decide(stats):
    """Boolean MMN-present verdict for each criterion, from one trace's stats dict.

    All criteria share the magnitude gate (peak < 0); the shape criteria add conditions.
    Missing keys (degenerate/empty window) -> False.
    """
    peak = stats.get("peak", 0.0)
    neg = peak < 0.0
    d = dict(
        C0_current=neg,                                              # min<0 only
        S1_interior=neg and stats.get("interior", False),            # min not at edge
        S2_recovery=neg and stats.get("recovered", False),           # dip-and-recover
        S3_interior_recovery=neg and stats.get("interior", False)
        and stats.get("recovered", False),                           # both
        S4_specificity=neg and stats.get("more_neg_than_ctrl", False),  # vs control win
    )
    return d


# --------------------------------------------------------------------------------------
# h5 discovery (mirrors build_mmn_results_table.py) and ROI assembly.
# --------------------------------------------------------------------------------------
def iter_prediction_files(root):
    """Yield (h5_path, model, level, mapping) for every prediction file under root."""
    if not root.exists():
        return
    for model_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if model_dir.name.endswith("-parcels") or model_dir.name.endswith("-electrodes"):
            model, _, dir_level = model_dir.name.rpartition("-")
        else:
            model, dir_level = model_dir.name, None
        for h5_path in sorted(model_dir.rglob("*predictions__*.h5")):
            mapping = "encoder" if h5_path.name.endswith("__attn.h5") else "mtrf"
            level = dir_level or ("electrodes"
                                  if h5_path.name.startswith("electrode_predictions__")
                                  else "parcels")
            yield h5_path, model, level, mapping


def roi_indices(target_names, level):
    roi = ELECTRODE_ROI if level == "electrodes" else PARCEL_ROI
    return [i for i, t in enumerate(target_names) if t in roi]


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_root", default="outputs/insilico_mmn_predictions")
    p.add_argument("--out", default="outputs/results/mmn_criteria_comparison.csv")
    # the single scoring window (ms); settled at 100-240 per
    # aux/mmn_criterion_investigation.md -- not varied.
    p.add_argument("--window", type=float, nargs=2, default=(100.0, 240.0))
    # shape-criterion knobs
    p.add_argument("--recovery_ms", type=float, default=120.0,
                   help="how long after the trough to look for recovery")
    p.add_argument("--recovery_frac", type=float, default=0.5,
                   help="fraction of trough depth that must be recovered")
    p.add_argument("--ctrl_window", type=float, nargs=2, default=(300.0, 440.0),
                   help="control window with no expected MMN (specificity check)")
    p.add_argument("--edge_guard_bins", type=int, default=1,
                   help="how many bins from each window edge count as 'edge' for S1")
    args = p.parse_args()

    windows = {"current": tuple(args.window)}

    rows = []           # one row per (model, level, mapping, method) -- ROI verdict
    target_rows = []    # one row per (..., target) -- per-target detail
    n_files = 0

    for h5_path, model, level, mapping in iter_prediction_files(Path(args.predictions_root)):
        n_files += 1
        with h5py.File(h5_path, "r") as h5:
            tkey = "parcels" if "parcels" in h5 else "electrodes"
            tnames = [t.decode() if hasattr(t, "decode") else t for t in h5[tkey][:]]
            roi = roi_indices(tnames, level)
            for method, g in h5.items():
                if not isinstance(g, h5py.Group):
                    continue
                t = g["time_ms"][:].astype(float)
                std = g["standard"][:].astype(float)
                dev = g["deviant_mean"][:].astype(float)
                soa = float(g.attrs.get("soa_ms", np.nan))
                z = compute_z_diff(t, std, dev, soa)        # [n_t, n_target]

                # ---- validation: recomputed current peak must match the stored one ----
                stored_peak = g["peak"][:] if "peak" in g else None
                cur_lo, cur_hi = windows["current"]
                recomputed = z[window_mask(t, cur_lo, cur_hi)].min(0)
                if stored_peak is not None:
                    bad = ~np.isclose(recomputed, stored_peak, atol=1e-3, equal_nan=True)
                    if bad.any():
                        print(f"  WARN {model}/{level}/{mapping}/{method}: recomputed peak "
                              f"differs from stored at {int(bad.sum())} target(s) "
                              f"(max |Δ|={np.nanmax(np.abs(recomputed-stored_peak)):.3g})")

                # ---- ROI-mean trace = the actual verdict signal ----
                roi_use = roi or list(range(len(tnames)))
                z_roi = z[:, roi_use].mean(1)

                row = dict(model=model, level=level, mapping=mapping, method=method,
                           source=g.attrs.get("source", ""),
                           label=g.attrs.get("context_final", ""),
                           soa_ms=soa, n_roi_targets=len(roi_use))

                for wname, (lo, hi) in windows.items():
                    s = trace_stats(t, z_roi, lo, hi, args.recovery_ms, args.recovery_frac,
                                    args.ctrl_window[0], args.ctrl_window[1],
                                    args.edge_guard_bins)
                    verdicts = decide(s)
                    row[f"{wname}_peak"] = round(s.get("peak", float("nan")), 4)
                    row[f"{wname}_argmin_ms"] = s.get("argmin_ms", float("nan"))
                    row[f"{wname}_interior"] = s.get("interior", False)
                    row[f"{wname}_recovery_frac"] = round(s.get("recovery_frac", 0.0), 3)
                    row[f"{wname}_recovered"] = s.get("recovered", False)
                    row[f"{wname}_ctrl_peak"] = round(s.get("ctrl_peak", float("nan")), 4)
                    for cname, v in verdicts.items():
                        row[f"{wname}__{cname}"] = v
                rows.append(row)

                # per-target detail (current window only) for auditing edge cases
                for ti in roi_use:
                    st = trace_stats(t, z[:, ti], cur_lo, cur_hi, args.recovery_ms,
                                     args.recovery_frac, args.ctrl_window[0],
                                     args.ctrl_window[1], args.edge_guard_bins)
                    target_rows.append(dict(
                        model=model, level=level, mapping=mapping, method=method,
                        target=tnames[ti], peak=round(st.get("peak", float("nan")), 4),
                        argmin_ms=st.get("argmin_ms", float("nan")),
                        interior=st.get("interior", False),
                        recovered=st.get("recovered", False),
                        ctrl_peak=round(st.get("ctrl_peak", float("nan")), 4)))

    if not rows:
        print(f"No prediction HDF5s found under {args.predictions_root} -- nothing to do.\n"
              f"(This script must run where the prediction h5s live, e.g. the cluster.)")
        return

    # ---- write the tidy per-run CSV ----
    lead = ["model", "level", "mapping", "method", "source", "label", "soa_ms",
            "n_roi_targets"]
    fieldnames = lead + sorted({k for r in rows for k in r} - set(lead))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    tgt_out = str(Path(args.out).with_name(Path(args.out).stem + "__per_target.csv"))
    with open(tgt_out, "w", newline="") as f:
        tlead = ["model", "level", "mapping", "method", "target", "peak", "argmin_ms",
                 "interior", "recovered", "ctrl_peak"]
        w = csv.DictWriter(f, fieldnames=tlead)
        w.writeheader()
        w.writerows(target_rows)

    print_summary(rows)
    print(f"\nWrote {len(rows)} ROI rows -> {args.out}")
    print(f"Wrote {len(target_rows)} per-target rows -> {tgt_out}")


def print_summary(rows):
    """Headline counts per criterion, flip table, and the flagged/sanity/borderline cases."""
    crits = ["C0_current", "S1_interior", "S2_recovery", "S3_interior_recovery",
             "S4_specificity"]
    total = len(rows)

    print("\n" + "=" * 78)
    print(f"MMN-present counts (ROI verdict) over {total} (model,level,mapping,method) runs")
    print("=" * 78)
    for c in crits:
        key = f"current__{c}"
        print(f"  {c:<24} {sum(1 for r in rows if r.get(key)):>4}")
    print("legend: C0=current(min<0)  S1=interior-argmin  S2=trough+recovery  "
          "S3=interior&recovery  S4=control-window specificity")

    # flips vs current/C0 -- the runs whose verdict would change under each shape criterion
    print("\n" + "-" * 78)
    print("Runs whose verdict FLIPS off 'present' under each candidate:")
    print("-" * 78)
    for c in crits[1:]:
        flipped = [r for r in rows
                   if r.get("current__C0_current") and not r.get(f"current__{c}")]
        print(f"\n  {c}: {len(flipped)} run(s) reclassified present -> absent")
        for r in sorted(flipped, key=lambda r: r["current_peak"]):
            print(f"    {r['model']:<14} {r['level']:<10} {r['mapping']:<8} "
                  f"{r['method']:<10} peak={r['current_peak']:+.2f} "
                  f"argmin={r['current_argmin_ms']:.0f}ms "
                  f"interior={r['current_interior']} recov={r['current_recovered']}")

    # explicitly track the cases the caveats flagged
    print("\n" + "-" * 78)
    print("Flagged smooth-ramp cases (expect: should become ABSENT under shape criteria):")
    print("-" * 78)
    flagged = [("whisper-tiny", "parcels", "encoder", "method_72"),
               ("whisper-tiny", "parcels", "encoder", "method_75"),
               ("whisper-base", "parcels", "encoder", "method_72"),
               ("whisper-base", "parcels", "encoder", "method_75"),
               ("whisper-medium", "electrodes", "mtrf", "method_43")]
    _print_named(rows, flagged, crits)

    print("\n" + "-" * 78)
    print("Clear-trough sanity checks (expect: should STAY present under shape criteria):")
    print("-" * 78)
    troughs = [("whisper-tiny", "parcels", "mtrf", "method_44"),
               ("whisper-base", "parcels", "mtrf", "method_55"),
               ("whisper-medium", "parcels", "mtrf", "method_43")]
    _print_named(rows, troughs, crits)

    # borderline runs (|current peak| < BORDERLINE) -- closest to the present/absent threshold
    print("\n" + "-" * 78)
    print(f"Borderline runs (|current peak| < {BORDERLINE}):")
    print("-" * 78)
    for r in sorted((r for r in rows if abs(r.get("current_peak", 9)) < BORDERLINE),
                    key=lambda r: r["current_peak"]):
        print(f"  {r['model']:<14} {r['level']:<10} {r['mapping']:<8} {r['method']:<10} "
              f"peak={r['current_peak']:+.2f}")


def _print_named(rows, names, crits):
    index = {(r["model"], r["level"], r["mapping"], r["method"]): r for r in rows}
    for key in names:
        r = index.get(key)
        if r is None:
            print(f"  {key} -- not found in predictions")
            continue
        verds = " ".join(f"{c.split('_')[0]}={'Y' if r.get(f'current__{c}') else '.'}"
                         for c in crits)
        print(f"  {key[0]:<14} {key[1]:<10} {key[2]:<8} {key[3]:<10} "
              f"peak={r['current_peak']:+.2f} argmin={r['current_argmin_ms']:.0f}ms "
              f"recov_frac={r['current_recovery_frac']:.2f} | {verds}")


if __name__ == "__main__":
    main()
