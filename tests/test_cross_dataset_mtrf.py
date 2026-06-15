"""TDD for cross-dataset (out-of-domain) mTRF transfer scoring (plan §13 follow-up).

cross_score_dataset fits on dataset A, scores dataset B's held-out splits using A's model AND
A's feature standardisation (no peeking at B). The test plants a SHARED feature->EEG mapping in
A and B (transfer should be high) and a DIFFERENT mapping in C (transfer should be low) — i.e. it
checks that transfer r actually measures how shared the mapping is.
"""

from pathlib import Path

import h5py
import numpy as np

from mbs.evaluation.evaluate_features_mtrf import lags_in_bins
from mbs.evaluation.evaluate_features_mtrf_parcels import (
    fit_parcel_mtrf, cross_score_dataset,
)

PARCELS = [("frontal", ["Fz", "F3"], 0.7), ("temporal", ["T7"], 0.7)]
CHANS = ["Fz", "F3", "T7"]


def _make_dataset(tmp_path, tag, W, *, n=24, T=200, d=8, lag=4, seed=0, splits=("train", "test")):
    """Write a neural h5 + matching feature dir whose parcel EEG = lagged(features) @ W[parcel]."""
    rng = np.random.default_rng(seed)
    neural = tmp_path / f"{tag}.h5"
    fdir = tmp_path / f"{tag}_feats"
    fdir.mkdir()
    all_ids, all_feats = [], []
    with h5py.File(neural, "w") as f:
        f.attrs["subjects"] = ["group"]; f.attrs["rois"] = CHANS
        f.attrs["splits"] = list(splits); f.attrs["max_nc"] = 100.0; f.attrs["time_step_ms"] = 20.0
        for split in splits:
            feats = rng.normal(size=(n, T, d)).astype(np.float32)
            ids = [f"{tag}_{split}_{i:04d}" for i in range(n)]
            g = f.create_group(split)
            g.create_dataset("stimulus_ids", data=np.array([s.encode() for s in ids]))
            nd = g.create_group("neural_data").create_group("group")
            # build per-channel EEG from the shared mapping (parcel = mean of its channels)
            chan_eeg = {}
            for pi, (_, members, _) in enumerate(PARCELS):
                sig = np.zeros((n, T), np.float32)
                sig[:, lag:] = feats[:, : T - lag, :] @ W[pi]
                for ch in members:
                    chan_eeg[ch] = sig + 0.02 * rng.normal(size=(n, T)).astype(np.float32)
            for ch in CHANS:
                e = chan_eeg.get(ch, rng.normal(size=(n, T)).astype(np.float32))
                nd.create_dataset(ch, data=e[:, :, None])
            all_ids += ids; all_feats.append(feats)
        ncg = f.create_group("noise_ceilings").create_group("group")
        for ch in CHANS:
            ncg.create_dataset(ch, data=np.full((T, 1), 60.0, np.float32))
    with h5py.File(fdir / "feats.h5", "w") as f:
        f.create_dataset("features/blocks-2", data=np.concatenate(all_feats, 0))
        f.create_dataset("ids", data=np.array([s.encode() for s in all_ids]))
    return neural, fdir


def _fit_on(tmp_path, neural, fdir, lags):
    from mbs.evaluation.evaluate_features_mtrf_parcels import _aligned
    from mbs.evaluation.utils.evaluation_helpers import load_layer_features
    feats, idmap = load_layer_features("blocks.2", features_folder=fdir)
    ftr, etr = _aligned(neural, PARCELS, "train", feats.astype(np.float32), idmap)
    return fit_parcel_mtrf(ftr, etr, lags, highpass_hz=0.0, n_train_time_samples=0,
                           fs=50.0, rng=np.random.default_rng(0))


def test_transfer_high_when_mapping_shared_low_when_not(tmp_path):
    rng = np.random.default_rng(1)
    d = 8
    W = [rng.normal(size=(d,)).astype(np.float32) for _ in PARCELS]      # shared mapping
    W_diff = [rng.normal(size=(d,)).astype(np.float32) for _ in PARCELS] # different mapping
    lags = lags_in_bins(0.0, 160.0, 20.0, 20.0)                          # 0..8 bins, includes lag 4

    A_neural, A_f = _make_dataset(tmp_path, "A", W, seed=0)
    B_neural, B_f = _make_dataset(tmp_path, "B", W, seed=99)             # same mapping, new data
    C_neural, C_f = _make_dataset(tmp_path, "C", W_diff, seed=7)         # different mapping

    fit = _fit_on(tmp_path, A_neural, A_f, lags)

    r_shared = cross_score_dataset(fit, B_neural, B_f, PARCELS, "blocks.2", lags,
                                   highpass_hz=0.0, fs=50.0)["test"]
    r_diff = cross_score_dataset(fit, C_neural, C_f, PARCELS, "blocks.2", lags,
                                 highpass_hz=0.0, fs=50.0)["test"]
    assert r_shared.shape == (len(PARCELS),)
    assert r_shared.min() > 0.8, f"shared-mapping transfer too low: {r_shared}"
    assert r_diff.max() < 0.4, f"different-mapping transfer too high: {r_diff}"
