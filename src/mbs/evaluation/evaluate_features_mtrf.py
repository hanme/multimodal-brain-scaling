"""mTRF / lagged shared-weight Ridge for audio features vs temporal EEG.

Workstream-A replacement for the per-time-bin Ridge in ``evaluate_features_temporal.py``
for *continuous* stimuli (e.g. Broderick speech). It fixes the three problems of the
per-bin method at once: zero stimulus->EEG lag, no weight sharing across time, and the
(dubious) across-stimulus-at-fixed-offset score axis.

Two modes (one shared design-matrix builder):

  single_lag (default)
      For each stimulus->EEG lag ``l`` (bins), fit ONE Ridge shared across all
      ``(stimulus, time)`` samples:  ``feature[t-l] -> EEG[t]``. Score by Pearson r
      *along the concatenated time axis* of the held-out runs. Output per
      (layer, subject, roi):  ``scores[n_lags, n_ch]`` — the encoding-vs-latency curve.

  fir
      A single multivariate-FIR mTRF over the whole lag window: one shared Ridge whose
      input stacks the features at every lag, scored along time. Output: ``scores[1, n_ch]``
      — one encoding r per channel (the literature mTRF number). Optional PCA for big models.

Efficiency: all electrodes of all selected ROIs are fit in ONE multi-output RidgeCV per
(layer, subject, lag) with ``alpha_per_target=True``. Ridge is separable across output
channels, so per-channel weights/predictions are identical to fitting each channel alone;
per-target alpha selection removes the only coupling (a shared alpha that low-SNR channels
would drag). ROIs are recovered as channel subsets of this single fit.

Design choices (all CLI-configurable; defaults stated here for the record):
  * Lag window 0-800 ms, 20 ms step (Kadir suggested 50-800 ms as the *fitting* window).
  * Features z-scored per dimension using TRAIN statistics (--standardize_features).
  * Optional high-pass of EEG and features along time (--highpass_hz) — a diagnostic for
    the slow-autocorrelation confound that flattens the lag curve.
  * Training uses RANDOM time-point subsampling per segment (Kadir's note) to bound memory
    and reduce overfitting on the strong temporal autocorrelation of EEG. NOTE: this is the
    *fitting*-side autocorrelation fix, distinct from the significance-testing n_eff
    correction in scripts/plot_score_distributions.py.
  * Channels with mean NC <= --nc_threshold are dropped before fitting.
  * Scores are NC-corrected by dividing r by the per-channel NC averaged over time.
"""

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
from sklearn.linear_model import RidgeCV
from tqdm.auto import tqdm

from mbs.core import str2bool
from .utils.evaluation_helpers import (
    ALPHA_LIST_SHORT,
    ALPHA_LIST,
    load_layer_features,
    load_neural_metadata,
    load_neural_data,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Lagged shared-weight Ridge (mTRF) for temporal audio features vs EEG."
    )
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--target_feature_layers", type=str, required=True)
    p.add_argument("--data_hdf5_path", type=str, required=True)
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--mode", choices=["single_lag", "fir"], default="single_lag")
    p.add_argument("--lag_min_ms", type=float, default=0.0)
    p.add_argument("--lag_max_ms", type=float, default=800.0)
    p.add_argument("--lag_step_ms", type=float, default=20.0)

    p.add_argument("--highpass_hz", type=float, default=0.0,
                   help="High-pass EEG and features along time before fitting (0 = off). "
                        "Diagnostic for the slow-autocorrelation confound.")
    p.add_argument("--n_train_time_samples", type=int, default=200,
                   help="Random time points sampled per training segment (autocorrelation fix).")
    p.add_argument("--test_time_stride", type=int, default=1,
                   help="Stride over valid output time bins when scoring (1 = all).")
    p.add_argument("--feature_pca", type=int, default=0,
                   help="If > 0, PCA-reduce features to this many components (fit on train).")
    p.add_argument("--standardize_features", type=str2bool, default=True)
    p.add_argument("--nc_threshold", type=float, default=0.0,
                   help="Drop channels whose mean NC (r-scale) is <= this before fitting.")
    p.add_argument("--noise_ceiling_correct", type=str2bool, default=True)
    p.add_argument("--use_wide_range_alphas", type=str2bool, default=False)
    p.add_argument("--roi_allowlist", type=str, default="",
                   help="Comma-separated ROI names to evaluate (default: all in the HDF5).")
    p.add_argument("--exclude_whole_brain", type=str2bool, default=True)
    p.add_argument("--layer_id", type=int, default=None)
    p.add_argument("--overwrite", type=str2bool, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core helpers (pure functions — unit tested)
# ---------------------------------------------------------------------------

def lags_in_bins(lag_min_ms: float, lag_max_ms: float, lag_step_ms: float,
                 time_step_ms: float) -> np.ndarray:
    """Sorted, unique non-negative lag grid in EEG time bins (look-back: EEG[t] <- feat[t-lag])."""
    assert lag_max_ms >= lag_min_ms >= 0.0
    ms = np.arange(lag_min_ms, lag_max_ms + 1e-9, lag_step_ms)
    bins = np.round(ms / time_step_ms).astype(int)
    return np.unique(np.clip(bins, 0, None))


def sample_time_indices(T: int, max_lag: int, n_samples: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Random output-time indices in [max_lag, T) (decorrelates autocorrelated rows)."""
    valid = np.arange(max_lag, T)
    if n_samples <= 0 or n_samples >= valid.size:
        return valid
    return np.sort(rng.choice(valid, size=n_samples, replace=False))


def build_lagged_design(feats: np.ndarray, eeg: np.ndarray,
                        lags: np.ndarray, time_idx: np.ndarray):
    """Lagged design matrix.

    feats [n_stim, T, d], eeg [n_stim, T, n_ch], lags [n_lags] (>=0), time_idx [n_t] (>= max lag).
    Returns X [n_stim*n_t, n_lags*d], Y [n_stim*n_t, n_ch].
    """
    lags = np.asarray(lags)
    time_idx = np.asarray(time_idx)
    assert time_idx.min() >= int(lags.max()), "time_idx must be >= max lag"
    idx = time_idx[:, None] - lags[None, :]          # [n_t, n_lags]
    Xg = feats[:, idx, :]                              # [n_stim, n_t, n_lags, d]
    n_stim, n_t, n_lags, d = Xg.shape
    X = Xg.reshape(n_stim * n_t, n_lags * d)
    Y = eeg[:, time_idx, :].reshape(n_stim * n_t, eeg.shape[2])
    return X, Y


def pearson_along_time(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson r per channel along axis 0 (the pooled time axis). Returns [n_ch]."""
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    yt = y_true - y_true.mean(0, keepdims=True)
    yp = y_pred - y_pred.mean(0, keepdims=True)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r.astype(np.float32)


def mask_channels(nc_2d: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of channels whose time-mean NC (r-scale) exceeds threshold."""
    nc_mean = nc_2d.mean(axis=0) if nc_2d.ndim == 2 else nc_2d
    return np.where(nc_mean > threshold)[0]


def highpass_along_time(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass along axis=1 (the per-stimulus time axis).

    x: [n_stim, T, k]. Used as a diagnostic to remove slow drift that flattens lag tuning.
    """
    if cutoff_hz <= 0:
        return x
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="highpass")
    return filtfilt(b, a, x, axis=1).astype(np.float32)


def _standardize(feat_train, feat_test):
    mu = feat_train.reshape(-1, feat_train.shape[-1]).mean(0)
    sd = feat_train.reshape(-1, feat_train.shape[-1]).std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (feat_train - mu) / sd, (feat_test - mu) / sd


def _pca_fit_transform(feat_train, feat_test, n_comp):
    from sklearn.decomposition import PCA
    n_tr, T, d = feat_train.shape
    pca = PCA(n_components=min(n_comp, d), random_state=0)
    pca.fit(feat_train.reshape(-1, d))
    ft = pca.transform(feat_train.reshape(-1, d)).reshape(n_tr, T, -1)
    fe = pca.transform(feat_test.reshape(-1, d)).reshape(feat_test.shape[0], feat_test.shape[1], -1)
    return ft.astype(np.float32), fe.astype(np.float32)


def fit_score_block(feat_train, feat_test, eeg_train, eeg_test, nc_vec,
                    lags, args, rng, fs):
    """Fit and score one pre-assembled, pre-masked channel block (all ROIs at once).

    feat_*: [n_stim, T, d]   eeg_*: [n_stim, T, C]   nc_vec: [C] (r-scale, for correction)
    Returns scores_raw, scores_nc, each [n_rows, C] (n_rows = len(lags) | 1).
    """
    T = feat_train.shape[1]

    if args.highpass_hz and args.highpass_hz > 0:
        feat_train = highpass_along_time(feat_train, fs, args.highpass_hz)
        feat_test = highpass_along_time(feat_test, fs, args.highpass_hz)
        eeg_train = highpass_along_time(eeg_train, fs, args.highpass_hz)
        eeg_test = highpass_along_time(eeg_test, fs, args.highpass_hz)

    if args.standardize_features:
        feat_train, feat_test = _standardize(feat_train, feat_test)
    if args.feature_pca and args.feature_pca > 0:
        feat_train, feat_test = _pca_fit_transform(feat_train, feat_test, args.feature_pca)

    alphas = ALPHA_LIST if args.use_wide_range_alphas else ALPHA_LIST_SHORT
    max_lag = int(lags.max())
    t_train = sample_time_indices(T, max_lag, args.n_train_time_samples, rng)
    t_test = np.arange(max_lag, T, max(1, args.test_time_stride))

    lag_sets = [np.array([l]) for l in lags] if args.mode == "single_lag" else [lags]

    C = eeg_train.shape[2]
    scores_raw = np.full((len(lag_sets), C), np.nan, dtype=np.float32)
    for i, lag_set in enumerate(lag_sets):
        Xtr, Ytr = build_lagged_design(feat_train, eeg_train, lag_set, t_train)
        Xte, Yte = build_lagged_design(feat_test, eeg_test, lag_set, t_test)
        model = RidgeCV(alphas=alphas, alpha_per_target=True)
        model.fit(Xtr.astype(np.float32), Ytr.astype(np.float32))
        Ypred = model.predict(Xte.astype(np.float32))
        scores_raw[i] = pearson_along_time(Yte, Ypred)

    if args.noise_ceiling_correct:
        nc_safe = np.where(nc_vec > 0, nc_vec, 1e-6)
        scores_nc = scores_raw / nc_safe[None, :]
    else:
        scores_nc = scores_raw.copy()
    return scores_raw, scores_nc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _decode(xs):
    return [x.decode() if hasattr(x, "decode") else x for x in xs]


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.target_feature_layers) as f:
        layer_list_raw = json.load(f)
    layer_list = [e["name"] if isinstance(e, dict) else e for e in layer_list_raw]

    data_hdf5_path = Path(args.data_hdf5_path)
    assert data_hdf5_path.exists(), f"Neural HDF5 not found: {data_hdf5_path}"
    with h5py.File(data_hdf5_path, "r") as f:
        time_step_ms = float(f.attrs.get("time_step_ms", 20.0))
    fs = 1000.0 / time_step_ms

    subjects, rois, _, _ = load_neural_metadata(data_hdf5_path)
    subjects = _decode(subjects)
    rois = _decode(rois)
    if args.roi_allowlist.strip():
        allow = {r.strip() for r in args.roi_allowlist.split(",")}
        rois = [r for r in rois if r in allow]
    if args.exclude_whole_brain:
        rois = [r for r in rois if r != "whole_brain"]

    lags = lags_in_bins(args.lag_min_ms, args.lag_max_ms, args.lag_step_ms, time_step_ms)
    lag_ms = lags * time_step_ms
    print(f"Mode: {args.mode} | lags (bins): {lags.tolist()} | lags (ms): {lag_ms.tolist()}")
    print(f"highpass_hz: {args.highpass_hz} | subjects: {subjects} | ROIs: {len(rois)} | layers: {layer_list}")

    features_dir = Path(args.features_dir)
    scores_path = output_dir / "mtrf_scores.h5"
    summary_path = output_dir / "mtrf_scores_summary.json"

    if summary_path.exists() and not args.overwrite:
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"model_id": args.model_id, "mode": args.mode,
                   "lags_bins": lags.tolist(), "lags_ms": lag_ms.tolist(),
                   "config": vars(args), "entries": []}

    with h5py.File(scores_path, "a") as out_h5:
        out_h5.attrs["lags_bins"] = lags
        out_h5.attrs["lags_ms"] = lag_ms
        out_h5.attrs["mode"] = args.mode

        for layer_idx, layer_name in enumerate(tqdm(layer_list, desc="Layers")):
            if args.layer_id is not None and layer_idx != args.layer_id:
                continue
            layer_feats, id_map = load_layer_features(layer_name, features_folder=features_dir)
            if layer_feats.ndim != 3:
                print(f"Layer {layer_name}: features {layer_feats.ndim}D, expected 3D — skip.")
                continue
            layer_feats = layer_feats.astype(np.float32)

            for subject in subjects:
                # Skip whole (layer, subject) if every ROI is already done.
                if not args.overwrite and all(
                    f"{layer_name.replace('.', '-')}/{subject}/{roi}" in out_h5 for roi in rois
                ):
                    continue

                # ---- assemble one channel block across ROIs (alignment shared) ----
                tr_fi = te_fi = None
                block_tr, block_te, nc_block = [], [], []
                roi_cols = {}
                col = 0
                for roi in rois:
                    tr_ids, eeg_tr_roi, nc_roi = load_neural_data(data_hdf5_path, subject, roi, "train")
                    te_ids, eeg_te_roi, _ = load_neural_data(data_hdf5_path, subject, roi, "test")
                    if tr_fi is None:
                        tr_ids, te_ids = _decode(tr_ids), _decode(te_ids)
                        tr_fi_raw = [id_map.get(s) for s in tr_ids]
                        te_fi_raw = [id_map.get(s) for s in te_ids]
                        tr_keep = [i for i, v in enumerate(tr_fi_raw) if v is not None]
                        te_keep = [i for i, v in enumerate(te_fi_raw) if v is not None]
                        tr_fi = [v for v in tr_fi_raw if v is not None]
                        te_fi = [v for v in te_fi_raw if v is not None]
                        if not tr_fi or not te_fi:
                            print(f"  Skip {layer_name}/{subject}: no matched IDs.")
                            break
                    kept = mask_channels(nc_roi, args.nc_threshold)
                    if kept.size == 0:
                        continue
                    block_tr.append(eeg_tr_roi[tr_keep][:, :, kept])
                    block_te.append(eeg_te_roi[te_keep][:, :, kept])
                    nc_block.append(nc_roi.mean(0)[kept])
                    roi_cols[roi] = (col, col + kept.size, kept)
                    col += kept.size

                if col == 0 or tr_fi is None:
                    continue

                eeg_tr = np.concatenate(block_tr, axis=2)
                eeg_te = np.concatenate(block_te, axis=2)
                nc_vec = np.concatenate(nc_block)
                feat_train = layer_feats[tr_fi]
                feat_test = layer_feats[te_fi]

                scores_raw, scores_nc = fit_score_block(
                    feat_train, feat_test, eeg_tr, eeg_te, nc_vec, lags, args, rng, fs
                )

                # ---- split block back to ROIs and write ----
                for roi, (s, e, kept) in roi_cols.items():
                    key = f"{layer_name.replace('.', '-')}/{subject}/{roi}"
                    if key in out_h5:
                        del out_h5[key]
                    grp = out_h5.create_group(key)
                    grp.create_dataset("scores_raw", data=scores_raw[:, s:e], compression="gzip")
                    grp.create_dataset("scores_nc", data=scores_nc[:, s:e], compression="gzip")
                    grp.create_dataset("kept_channels", data=kept)
                    use = scores_nc[:, s:e] if args.noise_ceiling_correct else scores_raw[:, s:e]
                    best_i = int(np.nanargmax(np.nanmean(use, axis=1)))
                    summary["entries"].append({
                        "layer": layer_name, "subject": subject, "roi": roi,
                        "n_ch": int(kept.size),
                        "best_lag_ms": float(lag_ms[best_i]) if args.mode == "single_lag" else None,
                        "best_score": float(np.nanmean(use[best_i])),
                        "mean_score": float(np.nanmean(use)),
                    })
                out_h5.flush()
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

    print(f"Scores written to {scores_path}")
    print(f"Summary written to {summary_path}")


def cli():
    main(parse_args())


if __name__ == "__main__":
    cli()
