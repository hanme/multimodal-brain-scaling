"""Compare held-out parcel encoding r across datasets and methods (plan §13).

Reads any number of result HDF5s (``mtrf_parcel_scores.h5`` from the mTRF parcel encoder,
``attn_probe_temporal_scores.h5`` from the learned probe) and assembles the D1/D2/D3 ×
mTRF/probe comparison. Held-out splits are kept SEPARATE (D3's ``test_d1`` / ``test_d2`` are
never pooled — plan §13.2).

Usage:
  python -m mbs.analysis.compare_encoders \
    "D1-mTRF=outputs/results/whisper-base-mtrf-parcels-d1/mtrf_parcel_scores.h5" \
    "D2-mTRF=outputs/results/whisper-base-mtrf-parcels-d2/mtrf_parcel_scores.h5" \
    "D3-mTRF=outputs/results/whisper-base-mtrf-parcels-d3/mtrf_parcel_scores.h5" \
    --value r --layer blocks.2
"""

from pathlib import Path
import argparse

import h5py
import numpy as np


def read_parcel_heldout(h5_path):
    """{layer -> {split -> {parcels:[str], r:[float], r_nc:[float]|None}}}.

    Handles both schemas: per-split ``heldout_r__<split>`` and legacy ``heldout_r`` (mapped
    to the canonical ``"test"`` split)."""
    out = {}
    with h5py.File(Path(h5_path), "r") as f:
        for layer_key in f.keys():
            g = f[layer_key]
            if not isinstance(g, h5py.Group) or "parcels" not in g:
                continue
            layer = layer_key.replace("-", ".")
            parcels = [p.decode() if hasattr(p, "decode") else str(p) for p in g["parcels"][:]]
            splits = {}
            for k in g.keys():
                if k.startswith("heldout_r__"):                # per-split (not the _nc__ ones)
                    split = k[len("heldout_r__"):]
                    nc_key = f"heldout_r_nc__{split}"
                    splits[split] = {"parcels": parcels, "r": g[k][:],
                                     "r_nc": g[nc_key][:] if nc_key in g else None}
            if not splits and "heldout_r" in g:                # legacy single-split schema
                splits["test"] = {"parcels": parcels, "r": g["heldout_r"][:],
                                  "r_nc": g["heldout_r_nc"][:] if "heldout_r_nc" in g else None}
            out[layer] = splits
    return out


def to_records(run_label, data):
    """Flatten read_parcel_heldout output into long-form records tagged with a run label."""
    recs = []
    for layer, splits in data.items():
        for split, cell in splits.items():
            r_nc = cell["r_nc"]
            for i, parcel in enumerate(cell["parcels"]):
                recs.append({
                    "run": run_label, "layer": layer, "split": split, "parcel": parcel,
                    "r": float(cell["r"][i]),
                    "r_nc": (float(r_nc[i]) if r_nc is not None else float("nan")),
                })
    return recs


def _layer_sort_key(layer):
    try:
        return (0, int(str(layer).split(".")[-1]))
    except ValueError:
        return (1, str(layer))


def format_markdown(records, value="r", layer=None):
    """Wide markdown table: rows = (layer, parcel), columns = run/split, cells = `value`.

    If `layer` is given, restrict to that layer (rows become just parcels)."""
    if layer is not None:
        records = [r for r in records if r["layer"] == layer]
    # column = "run · split" (drop split label when a run has only the single "test" split)
    run_splits = {}
    for r in records:
        run_splits.setdefault(r["run"], set()).add(r["split"])
    def col_name(rec):
        sp = run_splits[rec["run"]]
        return rec["run"] if sp == {"test"} else f"{rec['run']}·{rec['split']}"

    cols, layers, parcels = [], [], []
    cell = {}
    for rec in records:
        c = col_name(rec)
        if c not in cols:
            cols.append(c)
        if rec["layer"] not in layers:
            layers.append(rec["layer"])
        if rec["parcel"] not in parcels:
            parcels.append(rec["parcel"])
        cell[(rec["layer"], rec["parcel"], c)] = rec[value]
    layers = sorted(layers, key=_layer_sort_key)

    head = (["layer", "parcel"] if layer is None else ["parcel"]) + cols
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for lyr in layers:
        for p in parcels:
            row = ([] if layer is not None else [lyr]) + [p]
            for c in cols:
                v = cell.get((lyr, p, c))
                row.append("" if v is None else f"{v:+.3f}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description="Compare held-out parcel encoding r across runs.")
    ap.add_argument("runs", nargs="+", help='LABEL=path/to/scores.h5 (repeatable)')
    ap.add_argument("--value", choices=["r", "r_nc"], default="r")
    ap.add_argument("--layer", default=None, help="restrict the table to one layer (e.g. blocks.2)")
    return ap.parse_args()


def main(args):
    records = []
    for spec in args.runs:
        assert "=" in spec, f"expected LABEL=path, got {spec!r}"
        label, path = spec.split("=", 1)
        records += to_records(label, read_parcel_heldout(path))
    print(format_markdown(records, value=args.value, layer=args.layer))


if __name__ == "__main__":
    main(parse_args())
