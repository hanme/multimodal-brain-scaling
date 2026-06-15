"""Parcel-level mTRF encoder with per-dataset held-out scoring (Workstream A, D2/D3).

The encoding-only, MMN-free counterpart of the parcel fit inside ``scripts/insilico_mmn.py``,
refactored as importable, tested functions so it sits beside the learned-probe driver
(``evaluate_features_attn_probe_temporal.py``) and produces the SAME ``heldout_r`` schema. It
fits one lagged shared-weight RidgeCV mapping (audio features -> 4 NC-parcels) on the train
split and scores each held-out split **separately** — for D3 that means ``test_d1`` and
``test_d2`` are never pooled (plan §13.2, the dataset-identity leak).

Same 4 parcels across datasets (members fixed from D1; NC recomputed per dataset, decision C).

Output (parallel to mtrf_scores.h5 / attn_probe_temporal_scores.h5):
  <output_dir>/mtrf_parcel_scores.h5
    attrs: lookback/lag config, highpass_hz, fs, nc_threshold, parcels_from
    <layer>/parcels, parcel_nc_r, heldout_r__<split>, heldout_r_nc__<split>
  + mtrf_parcel_summary.json
"""

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV

from mbs.core import str2bool
from mbs.evaluation.utils.evaluation_helpers import load_layer_features, ALPHA_LIST_SHORT, ALPHA_LIST
from mbs.evaluation.evaluate_features_mtrf import (
    lags_in_bins, sample_time_indices, build_lagged_design,
    pearson_along_time, highpass_along_time,
)
from mbs.evaluation.attn_probe.dataset_temporal import (
    build_parcels, recompute_parcel_nc, list_test_splits,
    load_parcel_eeg, parcel_nc_vector,
)

FS = 50.0
TIME_STEP_MS = 20.0


# ---------------------------------------------------------------------------
# Fit / score (pure, tested)
# ---------------------------------------------------------------------------

def fit_parcel_mtrf(feats_tr, eeg_tr, lags, *, highpass_hz, n_train_time_samples,
                    fs, rng, alphas=None, standardize=True, pca_var=None):
    """Fit one shared lagged RidgeCV: features -> parcel EEG. Returns {model, mu, sd, pca}.

    pca_var: if set (e.g. 0.95), fit a PCA on the standardised TRAIN features keeping that
    fraction of variance and lag the PCA *projection* instead of the raw features. The PC count is
    variance-driven, so it varies by model/layer. This shrinks the design width to
    n_PCs · n_lags — model-independent and small — so neither LAPACK overflow is hit: the SVD path
    (which overflows on feature width for wide models) AND the eigen path (which overflows on the
    n×n Gram for pooled D3, n≈81.8k) are both avoided. The fitted PCA is stored and re-applied to
    held-out / transfer features. See project_plan §15.
    """
    alphas = ALPHA_LIST_SHORT if alphas is None else alphas
    feats = highpass_along_time(feats_tr, fs, highpass_hz)
    eeg = highpass_along_time(eeg_tr, fs, highpass_hz)
    mu = sd = None
    if standardize:
        d = feats.shape[-1]
        mu = feats.reshape(-1, d).mean(0)
        sd = feats.reshape(-1, d).std(0)
        sd = np.where(sd > 1e-6, sd, 1.0)
        feats = (feats - mu) / sd
    pca = None
    if pca_var is not None and pca_var > 0:
        from sklearn.decomposition import PCA
        d = feats.shape[-1]
        # svd_solver='covariance_eigh' decomposes the d×d covariance (eigh), NOT an SVD of the tall
        # [n·T, d] feature matrix. The latter ('full') overflows 32-bit LAPACK gesdd workspace at
        # d=1024 on the big fitting sets (medium D1/D3: 378k–613k rows) — the very crash PCA was
        # meant to cure, resurfacing inside PCA. covariance_eigh is n-independent so it never
        # overflows, and is much faster when n·T ≫ d (always our case). Same PCA result.
        pca = PCA(n_components=pca_var, svd_solver="covariance_eigh")
        feats = pca.fit_transform(feats.reshape(-1, d)).reshape(
            feats.shape[0], feats.shape[1], -1).astype(np.float32)
    t_idx = sample_time_indices(feats.shape[1], int(lags.max()), n_train_time_samples, rng)
    X, Y = build_lagged_design(feats, eeg, lags, t_idx)
    # No PCA: force gcv_mode='eigen' (Gram path) so the default-'auto' SVD of a very wide design
    # (small/medium: 41 lags × d ≈ 31k–42k cols → LAPACK gesdd workspace ~4·min² overflows int32)
    # doesn't segfault. 'eigen' is the numerically identical GCV solution, so tiny/base are
    # unchanged. WITH PCA the design is narrow (n_PCs·lags), so n_samples > n_features and the cheap
    # 'auto' (→ svd) path is both safe and correct — and crucially it avoids eigen's n×n Gram, which
    # is what blocks pooled D3. (project_plan §15)
    gcv_mode = "auto" if pca is not None else "eigen"
    model = RidgeCV(alphas=alphas, alpha_per_target=True, gcv_mode=gcv_mode)
    model.fit(X.astype(np.float32), Y.astype(np.float32))
    return {"model": model, "mu": mu, "sd": sd, "pca": pca}


def score_parcel_mtrf(fit, feats_te, eeg_te, lags, *, highpass_hz, fs):
    """Per-parcel out-of-sample Pearson r along time on one held-out split. Returns [P]."""
    feats = highpass_along_time(feats_te, fs, highpass_hz)
    eeg = highpass_along_time(eeg_te, fs, highpass_hz)
    if fit["mu"] is not None:
        feats = (feats - fit["mu"]) / fit["sd"]
    if fit.get("pca") is not None:
        d = feats.shape[-1]
        feats = fit["pca"].transform(feats.reshape(-1, d)).reshape(
            feats.shape[0], feats.shape[1], -1).astype(np.float32)
    t_idx = np.arange(int(lags.max()), feats.shape[1])
    X, Y = build_lagged_design(feats, eeg, lags, t_idx)
    Yhat = fit["model"].predict(X.astype(np.float32))
    return pearson_along_time(Y, Yhat)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _aligned(neural_h5, parcels, split, layer_feats, id_map):
    """(features [n,T,d], parcel EEG [n,T,P]) aligned by stimulus id for one split."""
    ids, eeg = load_parcel_eeg(neural_h5, "group", parcels, split)
    raw = [id_map.get(s) for s in ids]
    keep = [i for i, v in enumerate(raw) if v is not None]
    fi = [raw[i] for i in keep]
    return np.asarray(layer_feats[fi], dtype=np.float32), eeg[keep].astype(np.float32)


def cross_score_dataset(fit, neural_h5, features_dir, parcels, layer_name, lags, *,
                        highpass_hz, fs):
    """Score a fitted parcel-mTRF on ANOTHER dataset's held-out splits (out-of-domain transfer).

    Returns {split: r[P]}. The SOURCE-domain feature standardisation (fit['mu']/['sd']) is applied
    to the target features — no peeking at target statistics. Same 4 parcels (members shared)."""
    layer_feats, id_map = load_layer_features(layer_name, features_folder=Path(features_dir))
    layer_feats = layer_feats.astype(np.float32)
    out = {}
    for split in list_test_splits(neural_h5):
        feats_te, eeg_te = _aligned(neural_h5, parcels, split, layer_feats, id_map)
        out[split] = score_parcel_mtrf(fit, feats_te, eeg_te, lags, highpass_hz=highpass_hz, fs=fs)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Parcel-level mTRF encoder (D2/D3, per-dataset scoring).")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--target_feature_layers", type=str, required=True)
    p.add_argument("--data_hdf5_path", type=str, required=True)
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--parcels_from", type=str, default="outputs/neural_data/broderick2018_30s.h5",
                   help="dataset whose NC defines the canonical parcel membership (same parcels rule)")
    p.add_argument("--lag_min_ms", type=float, default=0.0)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--lag_step_ms", type=float, default=20.0)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--nc_threshold", type=float, default=0.2)
    p.add_argument("--n_train_time_samples", type=int, default=200)
    p.add_argument("--standardize_features", type=str2bool, default=True)
    p.add_argument("--pca_var", type=float, default=None,
                   help="if set (e.g. 0.95), PCA the features to this variance fraction before "
                        "lagging; PC count varies by model/layer. Shrinks the design so neither "
                        "LAPACK overflow is hit (replaces the eigen workaround; needed for D3).")
    p.add_argument("--use_wide_range_alphas", type=str2bool, default=False)
    p.add_argument("--layer_id", type=int, default=None)
    p.add_argument("--overwrite", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    alphas = ALPHA_LIST if args.use_wide_range_alphas else ALPHA_LIST_SHORT

    with open(args.target_feature_layers) as f:
        layer_list = [e["name"] if isinstance(e, dict) else e for e in json.load(f)]

    lags = lags_in_bins(args.lag_min_ms, args.lag_max_ms, args.lag_step_ms, TIME_STEP_MS)
    splits = list_test_splits(args.data_hdf5_path)

    print(f"Canonical parcels from {args.parcels_from} (NC r > {args.nc_threshold}):")
    canonical = build_parcels(args.parcels_from, args.nc_threshold)
    parcels = recompute_parcel_nc(canonical, args.data_hdf5_path)
    names = [p[0] for p in parcels]
    nc_r = parcel_nc_vector(parcels)
    print(f"Parcels: {[(n, m, round(r,3)) for n, m, r in parcels]}")
    print(f"Held-out splits (scored separately): {splits}")

    scores_path = out_dir / "mtrf_parcel_scores.h5"
    summary_path = out_dir / "mtrf_parcel_summary.json"
    summary = {"model_id": args.model_id, "data_hdf5_path": args.data_hdf5_path,
               "splits": splits, "config": vars(args), "entries": []}

    # overwrite -> truncate (also recovers from a corrupt/half-written leftover file)
    file_mode = "w" if args.overwrite else "a"
    with h5py.File(scores_path, file_mode) as out_h5:
        out_h5.attrs["highpass_hz"] = args.highpass_hz
        out_h5.attrs["fs"] = FS
        out_h5.attrs["lag_max_ms"] = args.lag_max_ms
        out_h5.attrs["nc_threshold"] = args.nc_threshold
        out_h5.attrs["pca_var"] = float(args.pca_var) if args.pca_var else 0.0
        out_h5.attrs["splits"] = np.array(splits, dtype="S")
        for layer_idx, layer_name in enumerate(layer_list):
            if args.layer_id is not None and layer_idx != args.layer_id:
                continue
            key = layer_name.replace(".", "-")
            if key in out_h5 and not args.overwrite:
                print(f"Layer {layer_name} present — skip (use --overwrite)."); continue

            layer_feats, id_map = load_layer_features(layer_name, features_folder=Path(args.features_dir))
            if layer_feats.ndim != 3:
                print(f"Layer {layer_name}: {layer_feats.ndim}D features, expected 3D — skip."); continue
            layer_feats = layer_feats.astype(np.float32)

            feats_tr, eeg_tr = _aligned(args.data_hdf5_path, parcels, "train", layer_feats, id_map)
            fit = fit_parcel_mtrf(feats_tr, eeg_tr, lags, highpass_hz=args.highpass_hz,
                                  n_train_time_samples=args.n_train_time_samples, fs=FS, rng=rng,
                                  alphas=alphas, standardize=args.standardize_features,
                                  pca_var=args.pca_var)
            n_pcs = int(fit["pca"].n_components_) if fit["pca"] is not None else None
            if n_pcs is not None:
                print(f"  [{layer_name}] PCA: {n_pcs} comps "
                      f"({fit['pca'].explained_variance_ratio_.sum():.3f} var) from d={layer_feats.shape[-1]}")

            if key in out_h5:
                del out_h5[key]
            g = out_h5.create_group(key)
            g.create_dataset("parcels", data=np.array(names, dtype="S"))
            g.create_dataset("parcel_nc_r", data=nc_r)
            if n_pcs is not None:
                g.attrs["n_pcs"] = n_pcs
            entry = {"layer": layer_name, "parcels": names, "n_pcs": n_pcs, "splits": {}}
            print(f"  [{layer_name}] held-out r (parcels {names}):")
            for split in splits:
                feats_te, eeg_te = _aligned(args.data_hdf5_path, parcels, split, layer_feats, id_map)
                r = score_parcel_mtrf(fit, feats_te, eeg_te, lags, highpass_hz=args.highpass_hz, fs=FS)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r_nc = np.where(nc_r > 0, r / nc_r, np.nan).astype(np.float32)
                g.create_dataset(f"heldout_r__{split}", data=r.astype(np.float32))
                g.create_dataset(f"heldout_r_nc__{split}", data=r_nc)
                entry["splits"][split] = {"heldout_r": r.tolist(), "heldout_r_nc": r_nc.tolist(),
                                          "n_test": int(eeg_te.shape[0])}
                print(f"    {split:<9} " + "  ".join(f"{n}={rr:+.3f}" for n, rr in zip(names, r)))
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
