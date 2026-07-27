"""Re-run analyze_mmn_criteria's shape-aware comparison (C0/S1/S2/S3/S4) under
single-site ROI definitions instead of the current averaged ROI: Fz only for
electrodes, central only for parcels (the best-correlated single-site stand-ins
per Section 5 of aux/results_analysis.md, r=+0.764 and r=+0.971 respectively).

Monkey-patches ELECTRODE_ROI/PARCEL_ROI on the imported analyze_mmn_criteria
module before calling its main(). Every shape-criteria function in that module
(trace_stats, decide, iter_prediction_files, compute_z_diff, window_mask) only
ever sees the already-ROI-averaged trace; the ROI constants are read solely
inside roi_indices(), which is called at run time -- so patching them on the
module object before main() runs changes the ROI for the whole loop, with no
need to touch analyze_mmn_criteria.py or duplicate its loop/CSV-writer/summary
logic. The per-target peak-reconstruction validation inside main() compares
against the stored per-target `peak` dataset and is independent of ROI choice,
so it is unaffected by the patch.

READ-ONLY over the prediction h5s.

Usage:
    python scripts/analyze_mmn_criteria_fz_central.py \
        --predictions_root outputs/insilico_mmn_predictions \
        --out outputs/results/mmn_criteria_fz_central.csv
"""
import sys

import pandas as pd

import analyze_mmn_criteria as amc

amc.ELECTRODE_ROI = {"Fz"}
amc.PARCEL_ROI = {"central"}

DEFAULT_OUT = "outputs/results/mmn_criteria_fz_central.csv"


def print_model_tables(out_csv):
    """Table-13/14-style present-count tables (model x criterion), mirrored
    for the Fz/central ROI so they're directly comparable to the existing
    current7/current2 tables in aux/results_analysis.md."""
    df = pd.read_csv(out_csv)
    crits = ["C0_current", "S1_interior", "S2_recovery", "S3_interior_recovery",
              "S4_specificity"]
    model_order = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]
    for mapping in ("mtrf", "encoder"):
        sub = df[df.mapping == mapping]
        print(f"\n-- {mapping}: MMN-present counts per criterion (n/20 per model, "
              f"parcels+electrodes pooled) --")
        header = f"{'model':<10}" + "".join(f"{c.split('_')[0]:>8}" for c in crits)
        print(header)
        totals = [0] * len(crits)
        for m in model_order:
            row = sub[sub.model == m]
            counts = [int(row[f"current__{c}"].sum()) for c in crits]
            totals = [t + c for t, c in zip(totals, counts)]
            print(f"{m.replace('whisper-', ''):<10}"
                  + "".join(f"{c:>5}/20" for c in counts))
        print(f"{'Total':<10}" + "".join(f"{t:>5}/80" for t in totals))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv += ["--out", DEFAULT_OUT]
    sys.argv = [sys.argv[0]] + argv
    amc.main()
    out_path = argv[argv.index("--out") + 1]
    print_model_tables(out_path)
