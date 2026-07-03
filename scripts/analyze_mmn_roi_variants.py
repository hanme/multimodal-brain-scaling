"""ROI sensitivity check for the in-silico MMN verdict: how much does the
present/absent call (and the continuous peak magnitude) depend on which
electrodes/parcels get averaged into the ROI, versus single canonical sites?

WHY THIS EXISTS
---------------
The committed verdict averages the z-scored deviant-standard difference over a
fixed ROI before scoring (`{Fz, FCz, Cz, FC1, FC2, F1, F2}` for electrodes,
`{frontal, central}` for parcels -- see aux/results_analysis.md, Section 4).
The MMN literature more often reports single canonical sites (Fz/FCz) rather
than an averaged scalp ROI, and the intracranial generators include
auditory/temporal cortex despite the classic fronto-central scalp topography.
This script recomputes the magnitude-only (C0) verdict -- peak = min(z_diff in
[100,240]ms), present iff < 0, no shape criteria -- under single-site/single-
parcel ROI variants alongside the current multi-site ROI, on the same 160 runs.

READ-ONLY over the prediction h5s; does not modify analyze_mmn_criteria.py or
the pipeline. Reuses iter_prediction_files/compute_z_diff/window_mask and the
ELECTRODE_ROI/PARCEL_ROI constants from that script.

Usage:
    python scripts/analyze_mmn_roi_variants.py \
        --predictions_root outputs/insilico_mmn_predictions \
        --out outputs/results/mmn_roi_variants.csv
"""

from pathlib import Path
import argparse

import numpy as np
import h5py
import pandas as pd

from analyze_mmn_criteria import (
    iter_prediction_files, compute_z_diff, window_mask,
    ELECTRODE_ROI, PARCEL_ROI,
)

ROI_VARIANTS = {
    "electrodes": {
        "Fz": {"Fz"},
        "FCz": {"FCz"},
        "Fz_FCz": {"Fz", "FCz"},
        "current7": ELECTRODE_ROI,
    },
    "parcels": {
        "frontal": {"frontal"},
        "temporal": {"temporal"},
        "central": {"central"},
        "current2": PARCEL_ROI,
    },
}

MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]


def roi_variant_indices(target_names, level):
    """{variant_name: [indices into target_names]} for every variant applicable to `level`."""
    return {vname: [i for i, t in enumerate(target_names) if t in roi_set]
            for vname, roi_set in ROI_VARIANTS[level].items()}


def collect_rows(root, lo, hi):
    rows = []
    for h5_path, model, level, mapping in iter_prediction_files(root):
        with h5py.File(h5_path, "r") as h5:
            tkey = "parcels" if "parcels" in h5 else "electrodes"
            tnames = [t.decode() if hasattr(t, "decode") else t for t in h5[tkey][:]]
            variant_idx = roi_variant_indices(tnames, level)

            for method, g in h5.items():
                if not isinstance(g, h5py.Group):
                    continue
                t = g["time_ms"][:].astype(float)
                std = g["standard"][:].astype(float)
                dev = g["deviant_mean"][:].astype(float)
                soa = float(g.attrs.get("soa_ms", np.nan))
                z = compute_z_diff(t, std, dev, soa)        # [n_t, n_target] -- once per run

                win = window_mask(t, lo, hi)
                common = dict(model=model, level=level, mapping=mapping, method=method,
                              source=g.attrs.get("source", ""),
                              label=g.attrs.get("context_final", ""), soa_ms=soa)

                for vname, idx in variant_idx.items():
                    if not idx:
                        continue
                    z_roi = z[:, idx].mean(1)
                    peak = float(z_roi[win].min())
                    rows.append({**common, "roi_variant": vname, "n_roi_targets": len(idx),
                                 "peak": round(peak, 4), "present": bool(peak < 0)})
    return rows


def print_count_tables(df):
    print("\n" + "=" * 78)
    print("Deliverable 2 -- present-count (n/10 methods) by ROI variant")
    print("=" * 78)
    for level in ("electrodes", "parcels"):
        variants = list(ROI_VARIANTS[level].keys())
        for mapping in ("mtrf", "encoder"):
            sub = df[(df.level == level) & (df.mapping == mapping)]
            pivot = sub.pivot_table(index="model", columns="roi_variant",
                                     values="present", aggfunc="sum")
            pivot = pivot.reindex(index=MODEL_ORDER, columns=variants).fillna(0).astype(int)
            print(f"\n-- {level} x {mapping} --")
            header = f"{'model':<16}" + "".join(f"{v:>12}" for v in variants)
            print(header)
            for m in MODEL_ORDER:
                short = m.replace("whisper-", "")
                print(f"{short:<16}" + "".join(f"{pivot.loc[m, v]:>9}/10" for v in variants))
            total = pivot.sum(axis=0)
            print(f"{'total (n/40)':<16}" + "".join(f"{total[v]:>9}/40" for v in variants))

    print("\n" + "-" * 78)
    print("Combined (mTRF + Encoder) present-count by ROI variant")
    print("-" * 78)
    for level in ("electrodes", "parcels"):
        variants = list(ROI_VARIANTS[level].keys())
        sub = df[df.level == level]
        pivot = sub.pivot_table(index="model", columns="roi_variant",
                                 values="present", aggfunc="sum")
        pivot = pivot.reindex(index=MODEL_ORDER, columns=variants).fillna(0).astype(int)
        print(f"\n-- {level}, combined --")
        header = f"{'model':<16}" + "".join(f"{v:>12}" for v in variants)
        print(header)
        for m in MODEL_ORDER:
            short = m.replace("whisper-", "")
            print(f"{short:<16}" + "".join(f"{pivot.loc[m, v]:>9}/20" for v in variants))
        total = pivot.sum(axis=0)
        print(f"{'total (n/80)':<16}" + "".join(f"{total[v]:>9}/80" for v in variants))


def print_continuous_table(df):
    print("\n" + "=" * 78)
    print("Deliverable 3 -- mean peak by ROI variant (markdown-pastable)")
    print("=" * 78)
    for level in ("electrodes", "parcels"):
        variants = list(ROI_VARIANTS[level].keys())
        for mapping in ("mtrf", "encoder"):
            sub = df[(df.level == level) & (df.mapping == mapping)]
            pivot = sub.pivot_table(index="model", columns="roi_variant",
                                     values="peak", aggfunc="mean")
            pivot = pivot.reindex(index=MODEL_ORDER, columns=variants)
            print(f"\n-- {level} x {mapping} --")
            print("| Model  | " + " | ".join(variants) + " |")
            print("| ------ | " + " | ".join("-" * len(v) for v in variants) + " |")
            for m in MODEL_ORDER:
                short = m.replace("whisper-", "")
                vals = " | ".join(f"{pivot.loc[m, v]:+.2f}" for v in variants)
                print(f"| {short:<6} | {vals} |")
            avg = pivot.mean(axis=0)
            avg_vals = " | ".join(f"{avg[v]:+.2f}" for v in variants)
            print(f"| **avg** | {avg_vals} |")


def _wide(df, level):
    return df[df.level == level].pivot_table(
        index=["model", "mapping", "method"], columns="roi_variant", values="peak"
    ).reset_index()


def print_agreement(df):
    print("\n" + "=" * 78)
    print("Deliverable 4 -- Pearson r vs current ROI, pooled across applicable runs")
    print("=" * 78)
    elec = _wide(df, "electrodes")
    print(f"\n-- electrodes (n={len(elec)}) --")
    for variant in ("Fz", "FCz", "Fz_FCz"):
        r = elec["current7"].corr(elec[variant])
        print(f"  current7 vs {variant:<8} r={r:+.3f}")

    parc = _wide(df, "parcels")
    print(f"\n-- parcels (n={len(parc)}) --")
    for variant in ("frontal", "temporal", "central"):
        r = parc["current2"].corr(parc[variant])
        print(f"  current2 vs {variant:<8} r={r:+.3f}")


def print_disagreements(df):
    print("\n" + "=" * 78)
    print("Deliverable 5 -- sign disagreement lists")
    print("=" * 78)

    elec = _wide(df, "electrodes")
    elec_dis = elec[np.sign(elec["Fz"]) != np.sign(elec["FCz"])]
    print(f"\n-- Fz vs FCz sign disagreement: {len(elec_dis)} run(s) --")
    for _, r in elec_dis.iterrows():
        print(f"  {r['model']:<14} {r['mapping']:<8} {r['method']:<10} "
              f"Fz={r['Fz']:+.3f} FCz={r['FCz']:+.3f}")

    parc = _wide(df, "parcels")
    signs = np.sign(parc[["frontal", "temporal", "central"]])
    parc_dis = parc[signs.nunique(axis=1) > 1]
    print(f"\n-- frontal/temporal/central sign disagreement: {len(parc_dis)} run(s) --")
    for _, r in parc_dis.iterrows():
        print(f"  {r['model']:<14} {r['mapping']:<8} {r['method']:<10} "
              f"frontal={r['frontal']:+.3f} temporal={r['temporal']:+.3f} "
              f"central={r['central']:+.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions_root", default="outputs/insilico_mmn_predictions")
    p.add_argument("--out", default="outputs/results/mmn_roi_variants.csv")
    p.add_argument("--window", type=float, nargs=2, default=(100.0, 240.0))
    args = p.parse_args()

    rows = collect_rows(Path(args.predictions_root), args.window[0], args.window[1])
    if not rows:
        print(f"No prediction HDF5s found under {args.predictions_root} -- nothing to do.")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")

    print_count_tables(df)
    print_continuous_table(df)
    print_agreement(df)
    print_disagreements(df)


if __name__ == "__main__":
    main()
