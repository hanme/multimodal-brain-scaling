"""Learned temporal attention probe vs temporal EEG — Workstream B (MIRAGE-style).

The gradient-trained counterpart to ``evaluate_features_mtrf.py``: instead of a closed-form
lagged Ridge, a shared ``LatentAttentionTrunk`` attends over the lookback window (kept as a
token sequence) and a subject head reads out the target EEG. Trained with **MSE** (MIRAGE) and
random time-point sampling, with MIRAGE-style checkpoint selection (best validation Pearson on
a split carved from TRAIN); scored along time on the built-in held-out runs — the SAME
``heldout_r`` metric Method A reports, so the two sit side by side.

Readout level selects the head structure:
  --readout_level individual   one head per subject over a shared trunk (Kadir individual)
  --readout_level group        a single 'group' head on the group-averaged EEG (Kadir group)

Target level mirrors the §20 mTRF sweep (run each separately):
  --target_level parcels       the 5 coarse 10-20 parcels (frontal/central/temporal/parietal/occipital)
  --target_level electrodes    every electrode passing the NC floor (each its own target)

EEG targets are z-scored per target on train stats (so MSE fits real amplitude); the scaling is
stored in the checkpoint (``eeg_mu``/``eeg_sd``) so predictions invert to real units — see §16.

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
    build_parcels, build_electrodes, recompute_parcel_nc, load_parcel_eeg, parcel_nc_vector,
    list_test_splits, grouped_kfold,
)
from mbs.evaluation.attn_probe.engine_temporal import (
    TemporalTrainConfig, train_temporal_probe, score_heldout,
)
from mbs.evaluation.attn_probe.model import ProbeConfig
from mbs.evaluation.attn_probe.checkpoint import save_probe_checkpoint

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
    p.add_argument("--target_level", choices=["parcels", "electrodes"], default="parcels",
                   help="prediction targets: the 5 coarse parcels, or every electrode passing the "
                        "NC floor (each its own target). Mirrors the §20 mTRF sweep; run each "
                        "level separately.")
    p.add_argument("--parcels_from", type=str, default="outputs/neural_data/surprisal_30s.h5",
                   help="dataset whose NC defines the canonical target membership; NC is "
                        "recomputed on --data_hdf5_path. Default D2 (Cortical Surprisal).")
    p.add_argument("--lookback_ms", type=float, default=800.0)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--nc_threshold", type=float, default=0.2)
    p.add_argument("--val_mode", choices=["grouped", "random"], default="grouped",
                   help="validation split for layer/epoch selection. 'grouped' = a held-out "
                        "audiobook-part fold (non-overlapping with train — no 20s window leak); "
                        "'random' = a random fraction of train windows (legacy, overlapping).")
    p.add_argument("--n_folds", type=int, default=4,
                   help="number of group-by-part folds (grouped val_mode).")
    p.add_argument("--fold_idx", type=int, default=0,
                   help="which grouped fold is the held-out validation set (0..n_folds-1).")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="random val_mode only: fraction of TRAIN windows held out for selection.")
    p.add_argument("--eval_every", type=int, default=5,
                   help="evaluate validation Pearson every N epochs and keep the best-scoring "
                        "weights (MIRAGE-style). 0 disables selection (keep final epoch).")

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
    p.add_argument("--save_model", type=str2bool, default=True,
                   help="save a reusable checkpoint (state_dict + mu/sd + parcels + lookback) per "
                        "layer as model__<layer>.pt, so the trained mapping can be applied to "
                        "arbitrary new stimuli (in-silico MMN figure, Sophie's own stimuli).")
    return p.parse_args()


def _decode(xs):
    return [x.decode() if hasattr(x, "decode") else x for x in xs]


def _aligned_feats(neural_h5, subject, parcels, split, layer_feats, id_map, highpass_hz):
    """High-passed parcel EEG + id-aligned, high-passed features for one (subject, split).

    Returns (feats [n,T,d], eeg [n,T,P], stimulus_ids [n]) — ids aligned to the array rows, used
    for group-by-part CV folds."""
    ids, eeg = load_parcel_eeg(neural_h5, subject, parcels, split)
    raw = [id_map.get(s) for s in ids]
    keep = [i for i, v in enumerate(raw) if v is not None]
    fi = [raw[i] for i in keep]
    feats = np.asarray(layer_feats[fi], dtype=np.float32)
    eeg = eeg[keep].astype(np.float32)
    feats = highpass_along_time(feats, FS, highpass_hz)
    eeg = highpass_along_time(eeg, FS, highpass_hz)
    return feats, eeg, [ids[i] for i in keep]


def run_layer(args, layer_name, parcels, subjects, lookback, out_h5):
    layer_feats, id_map = load_layer_features(layer_name, features_folder=Path(args.features_dir))
    if layer_feats.ndim != 3:
        print(f"Layer {layer_name}: features {layer_feats.ndim}D, expected 3D — skip.")
        return None
    layer_feats = layer_feats.astype(np.float32)
    d = layer_feats.shape[2]

    test_splits = list_test_splits(args.data_hdf5_path)   # ["test"] or ["test_d1","test_d2"]

    # Features are identical across subjects (all heard all runs); preprocess ONCE.
    feats_tr_all, _, train_ids = _aligned_feats(args.data_hdf5_path, subjects[0], parcels, "train",
                                                layer_feats, id_map, args.highpass_hz)
    n_tr = feats_tr_all.shape[0]

    # Validation split out of TRAIN for MIRAGE-style layer/epoch selection; the held-out TEST
    # split(s) are never touched during selection. 'grouped' = a held-out audiobook-part fold, so
    # val windows never overlap train windows (separate .wav files) — no 20 s window leak; 'random'
    # = legacy random-window carve (overlapping, inflated). Default grouped.
    val_mode = getattr(args, "val_mode", "grouped")
    if val_mode == "grouped":
        fold_id = grouped_kfold(train_ids, k=int(getattr(args, "n_folds", 4)), seed=args.seed)
        fi_sel = int(getattr(args, "fold_idx", 0))
        val_idx = np.where(fold_id == fi_sel)[0]
        tr_idx = np.where(fold_id != fi_sel)[0]
    else:
        val_frac = float(getattr(args, "val_frac", 0.2))
        perm = np.random.default_rng(args.seed).permutation(n_tr)
        n_val = int(round(val_frac * n_tr)) if n_tr > 1 else 0
        val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    if tr_idx.size == 0:                                  # degenerate tiny-data guard
        tr_idx, val_idx = np.arange(n_tr), np.array([], dtype=int)
    has_val = val_idx.size > 0

    # Feature z-score stats from the TRAIN PORTION only (no val/test leakage).
    fd = feats_tr_all.shape[-1]
    mu = feats_tr_all[tr_idx].reshape(-1, fd).mean(0)
    sd = feats_tr_all[tr_idx].reshape(-1, fd).std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    feats_tr_all = (feats_tr_all - mu) / sd
    feats_te_by_split = {}
    for split in test_splits:
        f_te, _, _ = _aligned_feats(args.data_hdf5_path, subjects[0], parcels, split,
                                    layer_feats, id_map, args.highpass_hz)
        feats_te_by_split[split] = (f_te - mu) / sd

    # Per-subject EEG targets, z-scored per target on the TRAIN-PORTION stats. We REMEMBER the
    # scaling (eeg_mu/eeg_sd) so predictions can be read back in real units for the magnitude
    # MMN criterion (plan §16). Pearson heldout_r is scale-invariant -> numbers stay comparable
    # to the mTRF / 1-Pearson runs.
    eeg_tr_all, eeg_te_by_split = {}, {split: {} for split in test_splits}
    eeg_mu, eeg_sd = {}, {}
    for s in subjects:
        _, e_tr, _ = _aligned_feats(args.data_hdf5_path, s, parcels, "train",
                                    layer_feats, id_map, args.highpass_hz)
        n_tgt = e_tr.shape[-1]
        emu = e_tr[tr_idx].reshape(-1, n_tgt).mean(0)
        esd = e_tr[tr_idx].reshape(-1, n_tgt).std(0)
        esd = np.where(esd > 1e-6, esd, 1.0)
        eeg_mu[s], eeg_sd[s] = emu.astype(np.float32), esd.astype(np.float32)
        eeg_tr_all[s] = ((e_tr - emu) / esd).astype(np.float32)
        for split in test_splits:
            _, e_te, _ = _aligned_feats(args.data_hdf5_path, s, parcels, split,
                                        layer_feats, id_map, args.highpass_hz)
            eeg_te_by_split[split][s] = ((e_te - emu) / esd).astype(np.float32)

    feats_train = {s: feats_tr_all[tr_idx] for s in subjects}
    eeg_train = {s: eeg_tr_all[s][tr_idx] for s in subjects}
    feats_val = {s: feats_tr_all[val_idx] for s in subjects} if has_val else None
    eeg_val = {s: eeg_tr_all[s][val_idx] for s in subjects} if has_val else None

    probe_cfg = ProbeConfig(
        in_dim=d, d_model=args.d_model, nhead=args.nhead, num_latents=args.num_latents,
        cross_attn_layers=args.cross_attn_layers, dropout=args.dropout, pos_mode=args.pos_mode,
        max_tokens=max(4096, lookback + 1),
    )
    train_cfg = TemporalTrainConfig(
        device=args.device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, n_train_time_samples=args.n_train_time_samples,
        amp=args.amp, seed=args.seed,
        eval_every=int(getattr(args, "eval_every", 0)) if has_val else 0,
    )

    P = len(parcels)
    model, history = train_temporal_probe(
        feats_train=feats_train, eeg_train=eeg_train, feats_val=feats_val, eeg_val=eeg_val,
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

    # Per-layer VALIDATION r (the held-out selection split). For grouped val_mode this is the
    # held-out audiobook-part fold (non-overlapping) — the honest layer-selection signal that the
    # CV aggregation averages over folds. NOT used as a reported test number.
    if has_val:
        val_feats = feats_tr_all[val_idx]
        r_persubj_val = np.stack([
            score_heldout(model, val_feats, eeg_tr_all[s][val_idx], lookback, s, device=dev)
            for s in subjects
        ], axis=0)
        heldout_r_val = r_persubj_val.mean(axis=0).astype(np.float32)
        g.create_dataset("heldout_r__val", data=heldout_r_val)
        g.attrs["val_mode"] = val_mode
        g.attrs["fold_idx"] = int(getattr(args, "fold_idx", 0))
        entry["val_r"] = heldout_r_val.tolist()
        print(f"  [{layer_name}] val r ({val_mode} fold {getattr(args,'fold_idx',0)}) "
              f"mean={float(np.nanmean(heldout_r_val)):+.3f}")

    if args.readout_level == "individual":
        g.create_dataset("subjects", data=np.array(subjects, dtype="S"))

    if getattr(args, "save_model", True):
        ckpt_path = Path(args.output_dir) / f"model__{layer_name}.pt"
        save_probe_checkpoint(
            ckpt_path, model=model, cfg=probe_cfg, mu=mu, sd=sd, lookback=lookback,
            parcel_names=names, parcel_members=["+".join(p[1]) for p in parcels],
            parcel_nc=[p[2] for p in parcels], subjects=subjects,
            highpass_hz=args.highpass_hz, fs=FS, layer=layer_name,
            eeg_mu=eeg_mu, eeg_sd=eeg_sd,
            meta={"data_hdf5_path": args.data_hdf5_path, "features_dir": args.features_dir,
                  "parcels_from": getattr(args, "parcels_from", ""),
                  "target_level": getattr(args, "target_level", "parcels"),
                  "readout_level": args.readout_level},
        )
        print(f"  saved checkpoint -> {ckpt_path}")
    return entry


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.target_feature_layers) as f:
        layer_list = [e["name"] if isinstance(e, dict) else e for e in json.load(f)]

    print(f"Canonical {args.target_level} from {args.parcels_from} (NC r > {args.nc_threshold}), "
          f"NC recomputed on {args.data_hdf5_path}:")
    if args.target_level == "electrodes":
        canonical = build_electrodes(args.parcels_from, args.nc_threshold)
    else:
        canonical = build_parcels(args.parcels_from, args.nc_threshold)
    parcels = recompute_parcel_nc(canonical, args.data_hdf5_path)   # same members, dataset NC
    assert parcels, f"No {args.target_level} survived the NC threshold."

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
        out_h5.attrs["target_level"] = args.target_level
        out_h5.attrs["val_mode"] = args.val_mode
        out_h5.attrs["n_folds"] = args.n_folds
        out_h5.attrs["fold_idx"] = args.fold_idx
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
