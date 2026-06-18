"""Model->EEG mTRF layer sweep on ONE dataset (default D2), with CV-on-train layer selection.

For one model and one target level (parcels | electrodes):
  1. For every layer: fit the FIR mTRF on the TRAIN split and score it by k-fold CV *within train*
     (mean Pearson r over targets+folds). Selection never touches the test split.
  2. Pick the layer with the best CV score.
  3. At that layer, refit on all of TRAIN and score the held-out ~20% TEST split -> per-target test r.
     (Test r is also computed at every layer for the figure, but selection uses CV only.)

Writes one JSON (per-layer cv/test scores + the chosen layer + per-target test r) that
plot_eeg_mapping.py turns into the layer curve + the test-r bars. Dataset-agnostic: pass any
EEG file via --neural / --features_dir; default is D2 (Cortical Surprisal).

  python scripts/eeg_mapping_sweep.py --model_id whisper-small --target_level electrodes \
    --features_dir <whisper-small D2 features> --out outputs/results/eeg_mapping/...json
"""

import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

from mbs.evaluation.utils.evaluation_helpers import load_layer_features
from mbs.evaluation.evaluate_features_mtrf import (
    lags_in_bins, build_lagged_design, highpass_along_time, sample_time_indices, pearson_along_time,
)
from eeg_targets import FS, TIME_STEP_MS, build_targets, load_split_targets, grouped_kfold


def layers_for(model_id, layers_config):
    cfg = layers_config or f"configs/extraction/audio/{model_id.replace('-', '_')}_layers.json"
    spec = json.load(open(cfg))
    return [e["name"] for e in spec], [float(e["position"]) for e in spec], cfg


def standardize(feats):
    """Per-feature z-score stats from [n,T,d] -> (mu, sd) with sd floored."""
    flat = feats.reshape(-1, feats.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    return mu, np.where(sd > 1e-6, sd, 1.0)


def fit_predict(feats_tr, eeg_tr, feats_va, eeg_va, lags, n_samples, alphas, seed):
    """Fit RidgeCV on (feats_tr, eeg_tr), return per-target Pearson r on (feats_va, eeg_va).

    feats already high-passed; standardized here with TRAIN stats (no leakage). RidgeCV picks
    alpha per target; gcv_mode='eigen' avoids the wide-design SVD int32 overflow (plan §15).
    """
    mu, sd = standardize(feats_tr)
    ftr, fva = (feats_tr - mu) / sd, (feats_va - mu) / sd
    rng = np.random.default_rng(seed)
    t_tr = sample_time_indices(ftr.shape[1], int(lags.max()), n_samples, rng)
    X, Y = build_lagged_design(ftr, eeg_tr, lags, t_tr)
    model = RidgeCV(alphas=alphas, alpha_per_target=True, gcv_mode="eigen")
    model.fit(X.astype(np.float32), Y.astype(np.float32))
    t_va = sample_time_indices(fva.shape[1], int(lags.max()), n_samples, np.random.default_rng(seed + 1))
    Xv, Yv = build_lagged_design(fva, eeg_va, lags, t_va)
    return pearson_along_time(Yv, model.predict(Xv.astype(np.float32)))      # [n_target]


def maybe_pca(feats_tr_list, var):
    """Fit PCA on stacked TRAIN features (over time) -> transform fn, or identity if var is None."""
    if not var:
        return (lambda f: f), None
    stack = np.concatenate([f.reshape(-1, f.shape[-1]) for f in feats_tr_list], 0)
    pca = PCA(n_components=var, svd_solver="full").fit(stack)
    k = pca.n_components_

    def tf(f):
        n, T, _ = f.shape
        return pca.transform(f.reshape(-1, f.shape[-1])).reshape(n, T, k).astype(np.float32)
    return tf, k


def score_layer(layer, targets, args, lags, alphas):
    """CV-on-train score + held-out TEST score for one layer. Returns (cv_r, test_r) per target."""
    feats_all, id_map = load_layer_features(layer, features_folder=Path(args.features_dir))
    feats_all = feats_all.astype(np.float32)
    eeg_tr, feats_tr, train_ids = load_split_targets(args.neural, feats_all, id_map, targets,
                                                     "train", return_ids=True)
    eeg_te, feats_te = load_split_targets(args.neural, feats_all, id_map, targets, "test")
    feats_tr = highpass_along_time(feats_tr, FS, args.highpass_hz)
    feats_te = highpass_along_time(feats_te, FS, args.highpass_hz)
    eeg_tr = highpass_along_time(eeg_tr, FS, args.highpass_hz)
    eeg_te = highpass_along_time(eeg_te, FS, args.highpass_hz)

    pca_tf, n_pc = maybe_pca([feats_tr], args.pca_var)
    feats_tr, feats_te = pca_tf(feats_tr), pca_tf(feats_te)

    # k-fold CV over TRAIN stimuli, GROUPED BY AUDIOBOOK PART so val windows never overlap the
    # train windows (separate .wav files) — the selection signal is non-overlapping; test untouched.
    n = eeg_tr.shape[0]
    fold_id = grouped_kfold(train_ids, k=args.n_folds, seed=args.seed)
    cv = []
    for f in range(args.n_folds):
        vi = np.where(fold_id == f)[0]
        ti = np.where(fold_id != f)[0]
        if vi.size == 0 or ti.size == 0:
            continue
        cv.append(fit_predict(feats_tr[ti], eeg_tr[ti], feats_tr[vi], eeg_tr[vi],
                              lags, args.n_train_time_samples, alphas, args.seed))
    cv_r = np.nanmean(np.stack(cv, 0), 0)                                    # [n_target]

    # refit on ALL train, score the held-out TEST split
    test_r = fit_predict(feats_tr, eeg_tr, feats_te, eeg_te,
                         lags, args.n_train_time_samples, alphas, args.seed)
    print(f"  {layer:10s} CV r={np.nanmean(cv_r):+.3f}  TEST r={np.nanmean(test_r):+.3f}"
          + (f"  (PCA k={n_pc})" if n_pc else ""))
    return cv_r, test_r


def main():
    p = argparse.ArgumentParser(description="Model->EEG mTRF layer sweep with CV-on-train selection.")
    p.add_argument("--model_id", required=True)
    p.add_argument("--target_level", choices=["parcels", "electrodes"], required=True)
    p.add_argument("--features_dir", required=True, help="D2 (or other) features for this model")
    p.add_argument("--neural", default="outputs/neural_data/surprisal_30s.h5")
    p.add_argument("--layers_config", default="", help="override; else configs/.../<model>_layers.json")
    p.add_argument("--n_folds", type=int, default=4,
                   help="group-by-part CV folds (grouped_kfold); test split never touched")
    p.add_argument("--nc_r_threshold", type=float, default=0.2)
    p.add_argument("--highpass_hz", type=float, default=0.5)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--n_train_time_samples", type=int, default=120)
    p.add_argument("--alpha_log_min", type=float, default=1.0)
    p.add_argument("--alpha_log_max", type=float, default=7.0)
    p.add_argument("--alpha_n", type=int, default=25)
    p.add_argument("--pca_var", type=float, default=None,
                   help="if set (e.g. 0.95), PCA features to this variance before lagging (speeds wide models)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lags = lags_in_bins(0.0, args.lag_max_ms, TIME_STEP_MS, TIME_STEP_MS)
    alphas = np.logspace(args.alpha_log_min, args.alpha_log_max, args.alpha_n)
    layers, positions, cfg = layers_for(args.model_id, args.layers_config)
    print(f"{args.model_id} / {args.target_level}: {len(layers)} layers from {cfg}")

    print(f"Building {args.target_level} (NC floor r>{args.nc_r_threshold}) from {args.neural}:")
    targets = build_targets(args.neural, args.target_level, args.nc_r_threshold)
    names = [t[0] for t in targets]
    nc_r = [t[2] for t in targets]

    cv_by, test_by = [], []
    for layer in layers:
        cv_r, test_r = score_layer(layer, targets, args, lags, alphas)
        cv_by.append(cv_r.tolist())
        test_by.append(test_r.tolist())

    cv_score = [float(np.nanmean(c)) for c in cv_by]
    chosen_idx = int(np.nanargmax(cv_score))
    chosen = layers[chosen_idx]
    print(f"\nCHOSEN layer (max CV) = {chosen}  "
          f"CV r={cv_score[chosen_idx]:+.3f}  TEST r={np.nanmean(test_by[chosen_idx]):+.3f}")

    out = dict(
        model_id=args.model_id, target_level=args.target_level, neural=args.neural,
        features_dir=args.features_dir, n_folds=args.n_folds, nc_r_threshold=args.nc_r_threshold,
        highpass_hz=args.highpass_hz, lag_max_ms=args.lag_max_ms, pca_var=args.pca_var,
        layers=layers, positions=positions, targets=names, nc_r=nc_r,
        cv_r_by_layer=cv_by, test_r_by_layer=test_by, cv_score_by_layer=cv_score,
        chosen_idx=chosen_idx, chosen_layer=chosen,
        test_r_chosen=test_by[chosen_idx], cv_score_chosen=cv_score[chosen_idx])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
