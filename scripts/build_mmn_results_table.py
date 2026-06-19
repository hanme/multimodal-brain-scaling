"""Combined MMN results table: glob the per-model/method prediction HDF5s written by
insilico_mmn.py (mTRF) and insilico_mmn_attn.py (encoder), and assemble one CSV.

Rows = (model, mapping, stimulus method); columns = baseline_normalized_peak per parcel
(frontal/central/temporal) plus the matching n7v1_peak diagnostic per parcel.

Expects the model-namespaced directory convention from the cluster runbook:
  outputs/insilico_mmn_predictions/<model>/predictions__<layer>.h5         (mTRF)
  outputs/insilico_mmn_predictions/<model>/predictions__<layer>__attn.h5  (encoder)

Usage:
  python scripts/build_mmn_results_table.py \
    --predictions_root outputs/insilico_mmn_predictions --out outputs/results/mmn_results_table.csv
"""

from pathlib import Path
import argparse
import csv

import h5py


def rows_from_h5(h5_path, model, mapping):
    rows = []
    with h5py.File(h5_path, "r") as h5:
        parcels = [p.decode() if hasattr(p, "decode") else p for p in h5["parcels"][:]]
        for method, g in h5.items():
            if not isinstance(g, h5py.Group):
                continue
            row = dict(model=model, mapping=mapping, method=method, layer=h5.attrs.get("layer", ""),
                       source=g.attrs.get("source", ""), label=g.attrs.get("context_final", ""))
            for pname, peak, n7v1_peak in zip(parcels, g["peak"][:], g["n7v1_peak"][:]):
                row[f"{pname}_peak"] = float(peak)
                row[f"{pname}_n7v1_peak"] = float(n7v1_peak)
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_root", default="outputs/insilico_mmn_predictions")
    p.add_argument("--out", default="outputs/results/mmn_results_table.csv")
    args = p.parse_args()

    root = Path(args.predictions_root)
    rows = []
    for model_dir in sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []:
        for h5_path in sorted(model_dir.glob("predictions__*.h5")):
            mapping = "encoder" if h5_path.name.endswith("__attn.h5") else "mtrf"
            rows.extend(rows_from_h5(h5_path, model_dir.name, mapping))

    if not rows:
        print(f"No prediction HDF5s found under {root} -- nothing to write.")
        return

    lead = ["model", "mapping", "method", "layer", "source", "label"]
    fieldnames = lead + sorted({k for r in rows for k in r} - set(lead))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
