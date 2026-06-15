"""Learned temporal attention probe vs temporal EEG — Workstream B (MIRAGE-style).

The gradient-trained counterpart to ``evaluate_features_mtrf.py``: instead of a closed-form
lagged Ridge, a shared ``LatentAttentionTrunk`` attends over the lookback window (kept as a
token sequence) and a subject head reads out the parcel EEG. Trained with ``1 - Pearson`` and
random time-point sampling; scored along time on the built-in held-out runs — the SAME
``heldout_r`` metric Method A reports, so the two sit side by side.

Readout level is selectable and is the only structural difference between the two MIRAGE
variants:
  --readout_level individual   one head per subject over a shared trunk (Kadir individual)
  --readout_level group        a single 'group' head on the group-averaged EEG (Kadir group)

Targets are the SAME 4 NC-parcels as Method A / the in-silico MMN (frontal/temporal/parietal/
occipital), defined once from the group noise ceiling.

Output (parallel to mtrf_scores.h5 / predictions__<layer>.h5):
  outputs/.../attn_probe_temporal_scores.h5
    attrs: readout_level, lookback_ms, highpass_hz, fs, nc_threshold, probe/train config
    <layer>/parcels, parcel_nc_r, heldout_r [P], heldout_r_nc [P]
            (+ heldout_r_persubj [n_subj, P] for the individual variant)
  + attn_probe_temporal_summary.json
"""

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import torch

from mbs.core import str2bool
from mbs.evaluation.utils.evaluation_helpers import load_layer_features, load_neural_metadata
from mbs.evaluation.evaluate_features_mtrf import highpass_along_time
from mbs.evaluation.attn_probe.dataset_temporal import (
    build_parcels, recompute_parcel_nc, load_parcel_eeg, parcel_nc_vector, list_test_splits,
)
from mbs.evaluation.attn_probe.engine_temporal import (
    TemporalTrainConfig, train_temporal_probe, score_heldout,
)
from mbs.evaluation.attn_probe.model import ProbeConfig

FS = 50.0
TIME_STEP_MS = 20.0


def parse_args():
    p = argparse.ArgumentParser(description="Learned temporal attention probe (Workstream B).")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--target_feature_layers", type=str, required=True)
    p.add_argument("--data_hdf5_path", type=str, required=True)
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--readout_level", choices=["individual", "group"], default="group")
    p.add_argument("--parcels_from", type=str, default="outputs/neural_data/broderick2018_30s.h5",
                   help="dataset whose NC defines the canonical parcel membership (same parcels "
                        "across D1/D2/D3 — decision C); NC is recomputed on --data_hdf5_path")
    p.add_argument("--lookback_ms", type=float, default=800.0)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--nc_threshold", type=float, default=0.2)

    # probe capacity
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_latents", type=int, default=16)
    p.add_argument("--cross_attn_layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos_mode", choices=["learned", "sin", "none"], default="learned")

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_train_time_samples", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str2bool, default=True)

    p.add_argument("--layer_id", type=int, default=None, help="run only this layer index")
    p.add_argument("--overwrite", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _decode(xs):
    return [x.decode() if hasattr(x, "decode") else x for x in xs]


def _aligned_feats(neural_h5, subject, parcels, split, layer_feats, id_map, highpass_hz):
    """High-passed parcel EEG + id-aligned, high-passed features for one (subject, split)."""
    ids, eeg = load_parcel_eeg(neural_h5, subject, parcels, split)
    raw = [id_map.get(s) for s in ids]
    keep = [i for i, v in enumerate(raw) if v is not None]
    fi = [raw[i] for i in keep]
    feats = np.asarray(layer_feats[fi], dtype=np.float32)
    eeg = eeg[keep].astype(np.float32)
    feats = highpass_along_time(feats, FS, highpass_hz)
    eeg = highpass_along_time(eeg, FS, highpass_hz)
    return feats, eeg


def run_layer(args, layer_name, parcels, subjects, lookback, out_h5):
    layer_feats, id_map = load_layer_features(layer_name, features_folder=Path(args.features_dir))
    if layer_feats.ndim != 3:
        print(f"Layer {layer_name}: features {layer_feats.ndim}D, expected 3D — skip.")
        return None
    layer_feats = layer_feats.astype(np.float32)
    d = layer_feats.shape[2]

    test_splits = list_test_splits(args.data_hdf5_path)   # ["test"] or ["test_d1","test_d2"]

    # Features are identical across subjects (all heard all runs); preprocess ONCE.
    feats_tr, _ = _aligned_feats(args.data_hdf5_path, subjects[0], parcels, "train",
                                 layer_feats, id_map, args.highpass_hz)
    fd = feats_tr.shape[-1]
    mu = feats_tr.reshape(-1, fd).mean(0)
    sd = feats_tr.reshape(-1, fd).std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    feats_tr = (feats_tr - mu) / sd                       # train-stat z-score
    # held-out features per split, standardized with the SAME train stats
    feats_te_by_split = {}
    for split in test_splits:
        f_te, _ = _aligned_feats(args.data_hdf5_path, subjects[0], parcels, split,
                                 layer_feats, id_map, args.highpass_hz)
        feats_te_by_split[split] = (f_te - mu) / sd

    # Per-subject parcel EEG (only the target varies across subjects).
    eeg_tr, eeg_te_by_split = {}, {split: {} for split in test_splits}
    for s in subjects:
        _, e_tr = _aligned_feats(args.data_hdf5_path, s, parcels, "train",
                                 layer_feats, id_map, args.highpass_hz)
        eeg_tr[s] = e_tr
        for split in test_splits:
            _, e_te = _aligned_feats(args.data_hdf5_path, s, parcels, split,
                                     layer_feats, id_map, args.highpass_hz)
            eeg_te_by_split[split][s] = e_te
    feats_train = {s: feats_tr for s in subjects}
    # training-monitor val = the first held-out split (eval_every defaults off)
    val_split = test_splits[0]
    feats_val = {s: feats_te_by_split[val_split] for s in subjects}
    eeg_te = eeg_te_by_split[val_split]

    probe_cfg = ProbeConfig(
        in_dim=d, d_model=args.d_model, nhead=args.nhead, num_latents=args.num_latents,
        cross_attn_layers=args.cross_attn_layers, dropout=args.dropout, pos_mode=args.pos_mode,
        max_tokens=max(4096, lookback + 1),
    )
    train_cfg = TemporalTrainConfig(
        device=args.device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, n_train_time_samples=args.n_train_time_samples,
        amp=args.amp, seed=args.seed,
    )

    P = len(parcels)
    model, history = train_temporal_probe(
        feats_train=feats_train, eeg_train=eeg_tr, feats_val=feats_val, eeg_val=eeg_te,
        subjects=subjects, in_dim=d, n_parcel=P, lookback=lookback,
        train_cfg=train_cfg, probe_cfg=probe_cfg,
    )

    dev = train_cfg.device if torch.cuda.is_available() else "cpu"
    nc_r = parcel_nc_vector(parcels)
    names = [p[0] for p in parcels]

    key = layer_name.replace(".", "-")
    if key in out_h5:
        del out_h5[key]
    g = out_h5.create_group(key)
    g.create_dataset("parcels", data=np.array(names, dtype="S"))
    g.create_dataset("parcel_nc_r", data=nc_r)
    g.attrs["final_train_loss"] = float(history[-1]["train_loss"])

    entry = {"layer": layer_name, "parcels": names, "n_subjects": len(subjects), "splits": {}}
    # score each held-out split SEPARATELY (never pool — plan §13.2)
    for split in test_splits:
        feats_te = feats_te_by_split[split]
        eeg_te_s = eeg_te_by_split[split]
        r_persubj = np.stack([
            score_heldout(model, feats_te, eeg_te_s[s], lookback, s, device=dev) for s in subjects
        ], axis=0)                                              # [n_subj, P]
        heldout_r = r_persubj.mean(axis=0).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            heldout_r_nc = np.where(nc_r > 0, heldout_r / nc_r, np.nan).astype(np.float32)
        g.create_dataset(f"heldout_r__{split}", data=heldout_r)
        g.create_dataset(f"heldout_r_nc__{split}", data=heldout_r_nc)
        if args.readout_level == "individual":
            g.create_dataset(f"heldout_r_persubj__{split}", data=r_persubj.astype(np.float32))
        entry["splits"][split] = {"heldout_r": heldout_r.tolist(),
                                  "heldout_r_nc": heldout_r_nc.tolist()}
        print(f"  [{layer_name}] {args.readout_level} held-out r [{split}] over {len(subjects)} subj:")
        for nm, rr, rn in zip(names, heldout_r, heldout_r_nc):
            print(f"    {nm:<10} r={rr:+.3f}   r/NC={rn:+.3f}")

    if args.readout_level == "individual":
        g.create_dataset("subjects", data=np.array(subjects, dtype="S"))
    return entry


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.target_feature_layers) as f:
        layer_list = [e["name"] if isinstance(e, dict) else e for e in json.load(f)]

    print(f"Canonical parcels from {args.parcels_from} (NC r > {args.nc_threshold}), "
          f"NC recomputed on {args.data_hdf5_path}:")
    canonical = build_parcels(args.parcels_from, args.nc_threshold)
    parcels = recompute_parcel_nc(canonical, args.data_hdf5_path)   # same members, dataset NC
    assert parcels, "No parcels survived the NC threshold."

    subjects_all, _, _, _ = load_neural_metadata(Path(args.data_hdf5_path))
    subjects_all = _decode(subjects_all)
    if args.readout_level == "group":
        assert "group" in subjects_all, "group-averaged EEG not in the neural HDF5."
        subjects = ["group"]
    else:
        subjects = [s for s in subjects_all if s != "group"]
        assert subjects, ("--readout_level individual needs per-subject EEG; re-run "
                          "format_eeg_hdf5.py with --store_subjects.")
    print(f"Readout level: {args.readout_level} | subjects: {len(subjects)} | "
          f"parcels: {[p[0] for p in parcels]}")

    lookback = int(round(args.lookback_ms / TIME_STEP_MS)) + 1   # lags 0..N -> N+1 tokens

    scores_path = out_dir / "attn_probe_temporal_scores.h5"
    summary_path = out_dir / "attn_probe_temporal_summary.json"
    summary = {"model_id": args.model_id, "readout_level": args.readout_level,
               "lookback_ms": args.lookback_ms, "config": vars(args), "entries": []}

    file_mode = "w" if args.overwrite else "a"      # truncate on overwrite (avoids corrupt leftovers)
    with h5py.File(scores_path, file_mode) as out_h5:
        out_h5.attrs["readout_level"] = args.readout_level
        out_h5.attrs["lookback_ms"] = args.lookback_ms
        out_h5.attrs["lookback_bins"] = lookback
        out_h5.attrs["highpass_hz"] = args.highpass_hz
        out_h5.attrs["fs"] = FS
        out_h5.attrs["nc_threshold"] = args.nc_threshold
        for layer_idx, layer_name in enumerate(layer_list):
            if args.layer_id is not None and layer_idx != args.layer_id:
                continue
            key = layer_name.replace(".", "-")
            if key in out_h5 and not args.overwrite:
                print(f"Layer {layer_name} already present — skip (use --overwrite).")
                continue
            entry = run_layer(args, layer_name, parcels, subjects, lookback, out_h5)
            if entry is not None:
                summary["entries"].append(entry)
                out_h5.flush()
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

    print(f"Scores written to {scores_path}")
    print(f"Summary written to {summary_path}")


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
