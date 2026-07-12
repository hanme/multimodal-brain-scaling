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
  S7  -- the S2 ("current" window) criterion PLUS an ABSOLUTE microvolt amplitude floor:
         TRUE iff current__S2_recovery passes AND the mean-baseline-corrected deviant-standard
         difference wave, IN MICROVOLTS, at the S2 trough latency (current_argmin_ms, nearest
         sample, ROI-averaged) is <= -X uV. X = --dip_uv_threshold (default 1.0 uV, PROVISIONAL
         -- literature value pending). By construction S7 is a subset of S2, so S7 counts can
         only be <= S2 counts.
         CRITICAL: C0-S6 all run on compute_z_diff (z-scored, dimensionless baseline-SD units).
         S7 needs an ABSOLUTE uV amplitude, which lives only in the raw standard/deviant_mean
         traces, so uv_diff_wave() computes a SEPARATE mean-only baseline-corrected difference
         wave (mirroring insilico_mmn.finalize_method's diff_b), NOT the z-scored trace. mTRF
         predictions are in native EEG units (Volts ~1e-6) -> uV via --native_to_uv (default
         1e6). Encoder predictions are z-units -> multiplied per-target by eeg_sd from the
         checkpoint model__<layer>.pt (the +mean cancels in dev-std) to recover native units,
         then the same native->uV factor. The model's predicted uV scale is shrunk by
         regularization and need NOT match literature EEG uV -- X is calibrated to the model's
         own predicted-uV distribution (printed at the end of a run).

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


# --------------------------------------------------------------------------------------
# S7 microvolt amplitude gate -- unit handling (see the S7 bullet in the module docstring).
# The z-scored compute_z_diff is dimensionless; S7 needs an absolute uV amplitude, which only
# the raw standard/deviant_mean traces carry. uv_diff_wave() mirrors insilico_mmn.finalize_method's
# mean-only baseline-corrected diff_b, converting BOTH mappings onto a comparable uV scale.
# --------------------------------------------------------------------------------------
def load_encoder_sd(model, level, checkpoint_root, layer=None):
    """{target_name: eeg_sd (native/Volts)} for the group subject, from the encoder checkpoint
    outputs/results/<model>-probe-group-d2-<level>/model__<layer>.pt. Returns None (with no
    error) if no checkpoint is found -- the caller then leaves S7 unavailable for those rows.

    ck["parcels"]["names"] holds the target names (electrode names at electrode level, too) and
    is aligned to ck["eeg_sd"]["group"]; predictions_to_units is native = z*sd + mu, and the
    additive mu cancels in dev-std, so only sd is needed.
    """
    d = Path(checkpoint_root) / f"{model}-probe-group-d2-{level}"
    pts = []
    if layer:
        cand = d / f"model__{layer}.pt"
        if cand.exists():
            pts = [cand]
    if not pts:
        pts = sorted(d.glob("model__*.pt"))
    if not pts:
        return None
    import torch  # lazy: only needed for encoder rows
    ck = torch.load(pts[0], map_location="cpu", weights_only=False)
    names = list(ck["parcels"]["names"])
    sd = np.asarray(ck["eeg_sd"]["group"], dtype=float).ravel()
    return {n: float(sd[i]) for i, n in enumerate(names)}


def uv_diff_wave(time_ms, standard, deviant_mean, soa_ms, target_names, mapping,
                 name2sd=None, native_to_uv=1e6):
    """Mean-only baseline-corrected deviant-standard difference wave IN MICROVOLTS, per target.

    Mirrors insilico_mmn.finalize_method's diff_b (mean-only baseline over [-3*SOA, 0)), NOT the
    z-scored verdict trace. For the encoder (z-unit predictions) each target column is multiplied
    by its native EEG sd (name2sd) to recover native units -- the additive mean cancels in
    dev-std. Then native (Volts) -> uV via native_to_uv (default 1e6). Returns [n_t, n_target] uV,
    or None if mapping=="encoder" and no per-target sd is available.
    """
    base = (time_ms >= amc.BASELINE_START_MULT * soa_ms) & (time_ms < amc.BASELINE_END_MULT * soa_ms)
    if not base.any():
        base = time_ms < 0
    diff = deviant_mean - standard
    diff_b = diff - diff[base].mean(0, keepdims=True)          # mean-only baseline correction
    if mapping == "encoder":
        if name2sd is None:
            return None
        sd_col = np.array([name2sd.get(n, np.nan) for n in target_names], dtype=float)
        diff_b = diff_b * sd_col[None, :]                      # z-units -> native (Volts)
    return diff_b * native_to_uv                               # native -> uV


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


# Candidate X values printed alongside the empirical uV distribution, so the placeholder
# --dip_uv_threshold can be calibrated to the model's own (regularization-shrunk) uV scale.
UV_SWEEP = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5)


def print_uv_distribution(uv_dist_rows, chosen_x):
    """Empirical microvolt-trough distribution (min/median/max per mapping x level) plus how many
    runs each candidate X in UV_SWEEP would gate in. The trough (min_uv, deepest dip in the 100-240
    ms window) is the amplitude S7 tests; the S2-passing subset is the one S7 can actually gate."""
    print("\n" + "=" * 78)
    print("S7 uV-trough distribution (min_uv = deepest dip in 100-240 ms, ROI-averaged), by "
          "mapping x level")
    print(f"chosen X = {chosen_x:g} uV; SCALE CAVEAT: predicted uV is shrunk by regularization "
          "and need NOT match literature EEG uV.")
    print("=" * 78)
    for subset, label in ((False, "ALL runs"), (True, "S2-pass only")):
        print(f"\n-- {label} --")
        hdr = (f"  {'mapping x level':<22}{'n':>5}{'min':>9}{'med':>9}{'max':>9}   "
               + "".join(f"<=-{x:<4g}" for x in UV_SWEEP))
        print(hdr)
        for level in ("parcels", "electrodes"):
            for mapping in ("mtrf", "encoder"):
                vals = np.array([r[2] for r in uv_dist_rows
                                 if r[0] == mapping and r[1] == level and (not subset or r[4])
                                 and not np.isnan(r[2])])
                if vals.size == 0:
                    print(f"  {mapping+' x '+level:<22}{0:>5}   (none)")
                    continue
                counts = "".join(f"{int((vals <= -x).sum()):>7}" for x in UV_SWEEP)
                print(f"  {mapping+' x '+level:<22}{vals.size:>5}{vals.min():>9.3f}"
                      f"{np.median(vals):>9.3f}{vals.max():>9.3f}   {counts}")


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
    # ---- S7 microvolt amplitude gate ----
    p.add_argument("--dip_uv_threshold", type=float, default=1.0,
                   help="S7 amplitude floor X (uV): S7 = S2 AND (uV diff wave at the S2 trough "
                        "latency <= -X uV). Default 1.0 uV is PROVISIONAL (literature value pending).")
    p.add_argument("--native_to_uv", type=float, default=1e6,
                   help="native EEG unit -> uV factor. mTRF predicts in the EEG HDF5's native unit "
                        "(MNE loads BrainVision in Volts ~1e-6), so 1e6 converts Volts->uV; encoder "
                        "eeg_sd is in the same native unit.")
    p.add_argument("--checkpoint_root", default="outputs/results",
                   help="root holding <model>-probe-group-d2-<level>/model__<layer>.pt, used to "
                        "convert encoder z-unit predictions to native uV for S7. READ-ONLY.")
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
    enc_sd_cache = {}          # (model, level) -> {target_name: eeg_sd} or None (encoder S7)
    enc_missing = set()        # (model, level) with no checkpoint -> S7 unavailable
    uv_dist_rows = []          # (mapping, level, min_uv, trough_uv, s2, s7) for the distribution print

    for h5_path, model, level, mapping in amc.iter_prediction_files(Path(args.predictions_root)):
        n_files += 1
        with h5py.File(h5_path, "r") as h5:
            tkey = "parcels" if "parcels" in h5 else "electrodes"
            tnames = [t.decode() if hasattr(t, "decode") else t for t in h5[tkey][:]]
            roi = amc.roi_indices(tnames, level)

            # S7 encoder unit conversion needs per-target eeg_sd from the checkpoint (once per file).
            name2sd = None
            if mapping == "encoder":
                key = (model, level)
                if key not in enc_sd_cache:
                    nm = h5_path.name
                    layer = (nm.split("predictions__")[1].split("__attn")[0]
                             if "predictions__" in nm else None)
                    enc_sd_cache[key] = load_encoder_sd(model, level, args.checkpoint_root, layer)
                    if enc_sd_cache[key] is None:
                        enc_missing.add(key)
                        print(f"  WARN no encoder checkpoint under "
                              f"{args.checkpoint_root}/{model}-probe-group-d2-{level}/ "
                              f"-> S7 unavailable (trough_uv=NaN, S7=False) for these rows")
                name2sd = enc_sd_cache[key]

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

                # ---- S7: S2 PLUS an absolute-microvolt amplitude floor (SEPARATE uV wave,
                # NOT the z-scored trace). uV wave is ROI-averaged over the same roi_use as z_roi.
                uv = uv_diff_wave(t, std, dev, soa, tnames, mapping, name2sd, args.native_to_uv)
                argmin_ms = row["current_argmin_ms"]
                if uv is None:                       # encoder row with no checkpoint
                    trough_uv = float("nan")
                    min_uv = float("nan")
                else:
                    uv_roi = uv[:, roi_use].mean(1)
                    win = (t >= cur_lo) & (t <= cur_hi)
                    min_uv = float(uv_roi[win].min()) if win.any() else float("nan")
                    if np.isnan(argmin_ms):
                        trough_uv = float("nan")
                    else:
                        j = int(np.argmin(np.abs(t - argmin_ms)))   # nearest sample to S2 trough
                        trough_uv = float(uv_roi[j])
                s2 = bool(v_cur.get("S2_recovery", False))
                s7 = bool(s2 and not np.isnan(trough_uv)
                          and trough_uv <= -args.dip_uv_threshold)
                row["trough_uv"] = round(trough_uv, 5) if not np.isnan(trough_uv) else float("nan")
                row["min_uv"] = round(min_uv, 5) if not np.isnan(min_uv) else float("nan")
                row["current__S7_uv_gated"] = s7
                assert not (s7 and not s2), "S7 must be a subset of S2"   # hard S7 <= S2 check
                uv_dist_rows.append((mapping, level, min_uv, trough_uv, s2, s7))

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

    s2_n = sum(1 for r in rows if r.get("current__S2_recovery"))
    s7_n = sum(1 for r in rows if r["current__S7_uv_gated"])
    print(f"S2 present: {s2_n}/{total}")
    print(f"S7 present: {s7_n}/{total}  (X = {args.dip_uv_threshold:g} uV; S7 subset of S2 -> "
          f"S7 <= S2: {s7_n} <= {s2_n})")
    assert s7_n <= s2_n, "S7 count exceeds S2 count -- S7 must be a subset of S2"
    if enc_missing:
        print(f"  NOTE encoder S7 unavailable (no checkpoint) for "
              f"{sorted(enc_missing)} -> those rows have trough_uv=NaN, S7=False")
    print_uv_distribution(uv_dist_rows, args.dip_uv_threshold)


if __name__ == "__main__":
    main()
