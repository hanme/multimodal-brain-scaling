"""Amplitude-gated MMN (S2 vs S7) across ROI options -- the Section 7 deliverable.

WHY THIS EXISTS
---------------
Section 4/6's S7 (analyze_mmn_criteria_s5_s6.py) gates the committed ROI-mean verdict on an
absolute microvolt trough. This script instead asks: how does S2 (shape) vs S7 (shape + uV
floor) look at EACH candidate reporting site, rather than the averaged ROI? MMN is classically
reported at single fronto-central sites (Fz/FCz), so the ROI options here are:

  * the frontal parcel and the central parcel (from the PARCEL prediction HDF5s), and
  * each electrode that is a member of the frontal or central 10-20 cluster
    (eeg_targets.CLUSTERS['frontal'] + ['central'] = Fz, F3, F4, FCz, Cz, C3, C4),
    intersected with the electrodes that survived the NC floor (from the ELECTRODE HDF5s).

Each ROI option is a SINGLE target (no averaging). For each (model, mapping, method, roi_option)
S2 is the interior-trough + 50%-recovery verdict on that target's z-scored deviant-standard
difference (identical to analyze_mmn_criteria's S2), and S7 = S2 AND the microvolt difference
wave at the S2 trough latency <= -X uV, swept over X in {0.25, 0.5, 1.0, 1.5, 2.0, 2.5} uV.

Units: the uV difference wave is the mean-only baseline-corrected dev-std (NOT the z-scored
verdict trace), with the encoder's z-unit predictions converted to native uV via the checkpoint
eeg_sd -- reused verbatim from analyze_mmn_criteria_s5_s6 (uv_diff_wave / load_encoder_sd). The
model's predicted uV scale is shrunk by regularization and need not match literature EEG uV; the
X-sweep is calibrated to the model's own predicted-uV distribution, printed at the end.

READ-ONLY over the prediction h5s; reads the encoder checkpoints (outputs/results/...) for the
uV conversion. Writes one long-format CSV + a console summary. Does not modify the pipeline or
analyze_mmn_criteria*.py. S7 is a subset of S2 by construction (asserted per row).

Usage:
    python scripts/analyze_mmn_s7_roi.py \
        --predictions_root outputs/insilico_mmn_predictions \
        --out outputs/results_with_counter/mmn_s7_roi.csv
"""

from pathlib import Path
import argparse

import numpy as np
import h5py
import pandas as pd

from analyze_mmn_criteria import (
    iter_prediction_files, compute_z_diff, trace_stats, decide,
)
from analyze_mmn_criteria_s5_s6 import uv_diff_wave, load_encoder_sd, UV_SWEEP
from eeg_targets import CLUSTERS

# ROI options: the two fronto-central parcels + every frontal/central cluster electrode.
PARCEL_OPTIONS = ["frontal", "central"]
FC_ELECTRODES = CLUSTERS["frontal"] + CLUSTERS["central"]     # [Fz, F3, F4, FCz, Cz, C3, C4]

MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]

# Scoring knobs -- identical to analyze_mmn_criteria / analyze_mmn_criteria_s5_s6 defaults.
WINDOW = (100.0, 240.0)
RECOVERY_MS = 120.0
RECOVERY_FRAC = 0.5
CTRL_WINDOW = (300.0, 440.0)
EDGE_GUARD_BINS = 1


def collect_rows(root, checkpoint_root, native_to_uv):
    """One row per (model, mapping, method, roi_option, X) with S2, trough_uv, min_uv, S7."""
    rows = []
    enc_sd_cache = {}
    lo, hi = WINDOW
    for h5_path, model, level, mapping in iter_prediction_files(root):
        with h5py.File(h5_path, "r") as h5:
            tkey = "parcels" if "parcels" in h5 else "electrodes"
            tnames = [t.decode() if hasattr(t, "decode") else t for t in h5[tkey][:]]

            if level == "parcels":
                roi_kind, wanted = "parcel", PARCEL_OPTIONS
            else:
                roi_kind, wanted = "electrode", FC_ELECTRODES
            options = [(n, tnames.index(n)) for n in wanted if n in tnames]

            name2sd = None
            if mapping == "encoder":
                key = (model, level)
                if key not in enc_sd_cache:
                    nm = h5_path.name
                    layer = (nm.split("predictions__")[1].split("__attn")[0]
                             if "predictions__" in nm else None)
                    enc_sd_cache[key] = load_encoder_sd(model, level, checkpoint_root, layer)
                    if enc_sd_cache[key] is None:
                        print(f"  WARN no encoder checkpoint under "
                              f"{checkpoint_root}/{model}-probe-group-d2-{level}/ -> "
                              f"S7 unavailable (trough_uv=NaN) for these rows")
                name2sd = enc_sd_cache[key]

            for method, g in h5.items():
                if not isinstance(g, h5py.Group):
                    continue
                t = g["time_ms"][:].astype(float)
                std = g["standard"][:].astype(float)
                dev = g["deviant_mean"][:].astype(float)
                soa = float(g.attrs.get("soa_ms", np.nan))
                z = compute_z_diff(t, std, dev, soa)                       # [n_t, n_target]
                uv = uv_diff_wave(t, std, dev, soa, tnames, mapping, name2sd, native_to_uv)
                is_counter = method.endswith("_counter")
                win = (t >= lo) & (t <= hi)

                for roi_name, idx in options:
                    s = trace_stats(t, z[:, idx], lo, hi, RECOVERY_MS, RECOVERY_FRAC,
                                    CTRL_WINDOW[0], CTRL_WINDOW[1], EDGE_GUARD_BINS)
                    s2 = bool(decide(s).get("S2_recovery", False))
                    argmin_ms = s.get("argmin_ms", float("nan"))
                    if uv is None:
                        trough_uv = min_uv = float("nan")
                    else:
                        uv_i = uv[:, idx]
                        min_uv = float(uv_i[win].min()) if win.any() else float("nan")
                        trough_uv = (float("nan") if np.isnan(argmin_ms)
                                     else float(uv_i[int(np.argmin(np.abs(t - argmin_ms)))]))
                    common = dict(model=model, mapping=mapping, method=method,
                                  is_counter=is_counter, roi_kind=roi_kind, roi=roi_name,
                                  s2=s2, trough_uv=round(trough_uv, 5) if not np.isnan(trough_uv)
                                  else float("nan"),
                                  min_uv=round(min_uv, 5) if not np.isnan(min_uv) else float("nan"))
                    for x in UV_SWEEP:
                        s7 = bool(s2 and not np.isnan(trough_uv) and trough_uv <= -x)
                        assert not (s7 and not s2), "S7 must be a subset of S2"
                        rows.append({**common, "dip_uv_threshold": x, "s7": s7})
    return rows


def _order_options(df):
    """ROI options in a readable order: the two parcels, then the cluster electrodes."""
    present = df["roi"].unique().tolist()
    ordered = [r for r in PARCEL_OPTIONS + FC_ELECTRODES if r in present]
    return ordered + [r for r in present if r not in ordered]


def print_s2_vs_s7_by_roi(df, ref_x):
    """S2 and S7(@ref_x) counts per ROI option, pooled over all methods x models, per mapping.
    n per cell = n_methods x n_models (80 for the 20-method set, 40 for the 10-method set)."""
    n_cell = df["model"].nunique() * df["method"].nunique()
    print("\n" + "=" * 78)
    print(f"S2 vs S7 present-count by ROI option (pooled over {df['method'].nunique()} methods x "
          f"{df['model'].nunique()} models = n/{n_cell}), S7 at X = {ref_x:g} uV")
    print("=" * 78)
    options = _order_options(df)
    at_x = df[np.isclose(df["dip_uv_threshold"], ref_x)]
    for mapping in ("mtrf", "encoder"):
        sub = at_x[at_x["mapping"] == mapping]
        print(f"\n-- {mapping} --")
        print(f"  {'ROI option':<12}{'kind':<11}{'S2 (n/'+str(n_cell)+')':>12}"
              f"{'S7 (n/'+str(n_cell)+')':>12}{'S7/S2':>9}")
        for roi in options:
            rs = sub[sub["roi"] == roi]
            if rs.empty:
                continue
            n2 = int(rs["s2"].sum()); n7 = int(rs["s7"].sum())
            kind = rs["roi_kind"].iloc[0]
            ratio = f"{(n7/n2*100):.0f}%" if n2 else "--"
            print(f"  {roi:<12}{kind:<11}{n2:>8}/{n_cell:<3}{n7:>8}/{n_cell:<3}{ratio:>9}")


def print_x_sweep(df):
    """Total S7 count per X in the sweep (per mapping), showing S7 falling as X rises; S2 is the
    X=0 reference (all rows share the same S2)."""
    print("\n" + "=" * 78)
    print("S7 present-count vs amplitude threshold X (pooled over all ROI options x methods x "
          "models)")
    print("=" * 78)
    for mapping in ("mtrf", "encoder"):
        sub = df[df["mapping"] == mapping]
        n_cells = sub[np.isclose(sub["dip_uv_threshold"], UV_SWEEP[0])].shape[0]
        s2_total = int(sub[np.isclose(sub["dip_uv_threshold"], UV_SWEEP[0])]["s2"].sum())
        print(f"\n-- {mapping} (n={n_cells} ROI-option cells; S2={s2_total}) --")
        header = f"  {'X (uV)':<8}" + "".join(f"{x:>8g}" for x in UV_SWEEP)
        counts = f"  {'S7':<8}" + "".join(
            f"{int(sub[np.isclose(sub['dip_uv_threshold'], x)]['s7'].sum()):>8}" for x in UV_SWEEP)
        print(header)
        print(counts)


def print_uv_distribution(df):
    """min_uv (deepest dip in 100-240 ms) distribution per mapping x roi_kind, over the S2-passing
    single-target traces (the ones S7 can gate)."""
    print("\n" + "=" * 78)
    print("uV-trough distribution per single ROI-option target (S2-pass only), by mapping x kind")
    print("SCALE CAVEAT: predicted uV is regularization-shrunk and need NOT match literature uV.")
    print("=" * 78)
    base = df[np.isclose(df["dip_uv_threshold"], UV_SWEEP[0])]      # one row per option
    print(f"  {'mapping x kind':<22}{'n':>5}{'min':>9}{'med':>9}{'max':>9}   "
          + "".join(f"<=-{x:<4g}" for x in UV_SWEEP))
    for kind in ("parcel", "electrode"):
        for mapping in ("mtrf", "encoder"):
            vals = base[(base["mapping"] == mapping) & (base["roi_kind"] == kind)
                        & (base["s2"])]["trough_uv"].dropna().to_numpy()
            if vals.size == 0:
                print(f"  {mapping+' x '+kind:<22}{0:>5}   (none)")
                continue
            counts = "".join(f"{int((vals <= -x).sum()):>7}" for x in UV_SWEEP)
            print(f"  {mapping+' x '+kind:<22}{vals.size:>5}{vals.min():>9.3f}"
                  f"{np.median(vals):>9.3f}{vals.max():>9.3f}   {counts}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_root", default="outputs/insilico_mmn_predictions")
    p.add_argument("--out", default="outputs/results_with_counter/mmn_s7_roi.csv")
    p.add_argument("--dip_uv_threshold", type=float, default=0.5,
                   help="reference X (uV) for the per-ROI S2-vs-S7 count table; the CSV always "
                        "carries the full sweep {0.25..2.5}. Default 0.5 uV is PROVISIONAL.")
    p.add_argument("--native_to_uv", type=float, default=1e6,
                   help="native EEG unit -> uV factor (Volts->uV = 1e6). See "
                        "analyze_mmn_criteria_s5_s6.py.")
    p.add_argument("--checkpoint_root", default="outputs/results",
                   help="root holding <model>-probe-group-d2-<level>/model__<layer>.pt for the "
                        "encoder uV conversion. READ-ONLY.")
    args = p.parse_args()

    rows = collect_rows(Path(args.predictions_root), args.checkpoint_root, args.native_to_uv)
    if not rows:
        print(f"No prediction HDF5s found under {args.predictions_root} -- nothing to do.")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")

    # Hard invariant: S7 is a subset of S2 in every cell.
    viol = int(((df["s7"]) & (~df["s2"])).sum())
    assert viol == 0, f"{viol} rows violate S7 <= S2"
    print(f"S7 <= S2 check: {viol} violations (must be 0)")

    print_uv_distribution(df)
    print_s2_vs_s7_by_roi(df, args.dip_uv_threshold)
    print_x_sweep(df)


if __name__ == "__main__":
    main()
