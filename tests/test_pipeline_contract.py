"""
Contract tests for the full feature → neural data → Ridge score pipeline.

These tests verify the three data contracts:
  1. Feature HDF5 IDs overlap neural HDF5 IDs (no silent empty intersection)
  2. Mean-pool Ridge produces finite Pearson r in [-1, 1]
  3. Temporal Ridge loop produces score of shape [T, n_ch] that is finite

All tests use small synthetic data — no model weights, no real EEG needed.
"""

import numpy as np
import pytest
import h5py


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _write_feature_h5(path, layer_name="blocks-0", n_stimuli=50, d_model=8, T=None):
    """Write a synthetic features HDF5 matching extract_features output format."""
    rng = np.random.default_rng(0)
    ids = np.array([f"audio{i:02d}_{j:06d}" for i, j in enumerate(range(n_stimuli))], dtype="S")
    if T is None:
        data = rng.standard_normal((n_stimuli, d_model)).astype(np.float32)
    else:
        data = rng.standard_normal((n_stimuli, T, d_model)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset(f"features/{layer_name}", data=data)
        f.create_dataset("ids", data=ids)
    return path


def _write_neural_h5(path, stimulus_ids, n_ch=3, T=None):
    """Write a synthetic neural HDF5 using the given stimulus IDs for train/test."""
    rng = np.random.default_rng(1)
    n = len(stimulus_ids)
    n_train = int(0.8 * n)
    subjects, rois = ["sub-01"], ["Fz"]

    with h5py.File(path, "w") as f:
        f.attrs["subjects"] = subjects
        f.attrs["rois"]     = rois
        f.attrs["splits"]   = ["train", "test"]
        f.attrs["max_nc"]   = 100.0
        if T is not None:
            f.attrs["temporal"] = True

        for split, ids_slice in [("train", stimulus_ids[:n_train]),
                                  ("test",  stimulus_ids[n_train:])]:
            n_split = len(ids_slice)
            f.create_dataset(f"{split}/stimulus_ids",
                             data=np.array(ids_slice, dtype="S"))
            for sub in subjects:
                for roi in rois:
                    shape = (n_split, T, n_ch) if T is not None else (n_split, n_ch)
                    f.create_dataset(f"{split}/neural_data/{sub}/{roi}",
                                     data=rng.standard_normal(shape).astype(np.float32))

        for sub in subjects:
            for roi in rois:
                nc_shape = (T, n_ch) if T is not None else (n_ch,)
                f.create_dataset(f"noise_ceilings/{sub}/{roi}",
                                 data=np.full(nc_shape, 50.0, dtype=np.float32))
    return path


# ──────────────────────────────────────────────────────────
# Contract 1: ID alignment
# ──────────────────────────────────────────────────────────

def test_feature_and_neural_ids_overlap(tmp_path):
    """IDs written by extract_features must overlap with IDs in the neural HDF5."""
    n = 50
    ids = [f"audio{i:02d}_{i:06d}" for i in range(n)]

    feat_h5 = _write_feature_h5(tmp_path / "feat.h5", n_stimuli=n)
    _write_neural_h5(tmp_path / "eeg.h5", stimulus_ids=ids)

    from mbs.evaluation.utils.evaluation_helpers import load_layer_features, load_neural_data
    _, feat_id_map = load_layer_features("blocks-0", features_file_path=feat_h5)
    eeg_ids, _, _ = load_neural_data(tmp_path / "eeg.h5", "sub-01", "Fz", "train")

    if hasattr(eeg_ids[0], "decode"):
        eeg_ids = [x.decode() for x in eeg_ids]

    overlap = set(feat_id_map.keys()) & set(eeg_ids)
    assert len(overlap) > 0, (
        "feature IDs and neural IDs must have non-empty overlap — "
        "check that extract_features and format_eeg_hdf5 use the same window ID format"
    )


def test_mismatched_ids_give_empty_overlap(tmp_path):
    """Sanity check: datasets with disjoint IDs must produce zero overlap."""
    feat_ids = [f"feat_{i}" for i in range(10)]
    eeg_ids  = [f"eeg_{i}"  for i in range(10)]

    feat_h5 = _write_feature_h5(tmp_path / "feat.h5", n_stimuli=10)
    # Overwrite IDs in feature file to use feat_ids
    with h5py.File(feat_h5, "a") as f:
        del f["ids"]
        f.create_dataset("ids", data=np.array(feat_ids, dtype="S"))

    _write_neural_h5(tmp_path / "eeg.h5", stimulus_ids=eeg_ids)

    from mbs.evaluation.utils.evaluation_helpers import load_layer_features, load_neural_data
    _, feat_id_map = load_layer_features("blocks-0", features_file_path=feat_h5)
    neural_ids, _, _ = load_neural_data(tmp_path / "eeg.h5", "sub-01", "Fz", "train")

    if hasattr(neural_ids[0], "decode"):
        neural_ids = [x.decode() for x in neural_ids]

    overlap = set(feat_id_map.keys()) & set(neural_ids)
    assert len(overlap) == 0


# ──────────────────────────────────────────────────────────
# Contract 2: mean-pool Ridge score sanity
# ──────────────────────────────────────────────────────────

def test_meanpool_ridge_score_finite_and_bounded(tmp_path):
    """Ridge fit on small synthetic data must produce finite Pearson r in [-1, 1]."""
    import scipy.stats
    from sklearn.linear_model import RidgeCV
    from mbs.evaluation.utils.evaluation_helpers import ALPHA_LIST_SHORT

    rng = np.random.default_rng(42)
    n_train, n_test, d, n_ch = 40, 10, 8, 3
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = rng.standard_normal((n_train, n_ch)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  d)).astype(np.float32)
    y_test  = rng.standard_normal((n_test,  n_ch)).astype(np.float32)

    model = RidgeCV(alphas=ALPHA_LIST_SHORT)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r = scipy.stats.pearsonr(y_test, y_pred, axis=0)[0]
    assert np.all(np.isfinite(r)), f"Pearson r contains NaN/Inf: {r}"
    assert np.all(r >= -1.0) and np.all(r <= 1.0), f"Pearson r out of [-1, 1]: {r}"


def test_meanpool_nc_corrected_score_finite(tmp_path):
    """Noise-ceiling-corrected Pearson r must remain finite (no divide-by-zero)."""
    import scipy.stats
    from sklearn.linear_model import RidgeCV
    from mbs.evaluation.utils.evaluation_helpers import ALPHA_LIST_SHORT

    rng = np.random.default_rng(0)
    n_train, n_test, d, n_ch = 40, 10, 8, 3
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = rng.standard_normal((n_train, n_ch)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  d)).astype(np.float32)
    y_test  = rng.standard_normal((n_test,  n_ch)).astype(np.float32)

    model = RidgeCV(alphas=ALPHA_LIST_SHORT)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r_raw = scipy.stats.pearsonr(y_test, y_pred, axis=0)[0]
    # noise ceiling: sqrt(50/100 + 1e-6) ≈ 0.707, matching what load_neural_data returns
    nc = np.sqrt(np.full(n_ch, 50.0) / 100.0 + 1e-6)
    r_nc = r_raw / nc

    assert np.all(np.isfinite(r_nc)), f"NC-corrected Pearson r is not finite: {r_nc}"


# ──────────────────────────────────────────────────────────
# Contract 3: temporal Ridge loop shape and finiteness
# ──────────────────────────────────────────────────────────

def test_temporal_score_shape_and_finiteness():
    """Per-time-step Ridge loop must produce score[T, n_ch] with finite values."""
    import scipy.stats
    from sklearn.linear_model import RidgeCV
    from mbs.evaluation.utils.evaluation_helpers import ALPHA_LIST_SHORT

    rng = np.random.default_rng(0)
    n_train, n_test, T, d, n_ch = 40, 10, 5, 8, 3
    X_train = rng.standard_normal((n_train, T, d)).astype(np.float32)
    y_train = rng.standard_normal((n_train, T, n_ch)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  T, d)).astype(np.float32)
    y_test  = rng.standard_normal((n_test,  T, n_ch)).astype(np.float32)

    scores = np.empty((T, n_ch))
    for t in range(T):
        model = RidgeCV(alphas=ALPHA_LIST_SHORT)
        model.fit(X_train[:, t, :], y_train[:, t, :])
        y_pred = model.predict(X_test[:, t, :])
        scores[t] = scipy.stats.pearsonr(y_test[:, t, :], y_pred, axis=0)[0]

    assert scores.shape == (T, n_ch)
    assert np.all(np.isfinite(scores)), f"temporal scores contain NaN/Inf: {scores}"


def test_temporal_score_time_axis_varies():
    """Different time steps should generally produce different scores (not all identical)."""
    import scipy.stats
    from sklearn.linear_model import RidgeCV
    from mbs.evaluation.utils.evaluation_helpers import ALPHA_LIST_SHORT

    rng = np.random.default_rng(7)
    n_train, n_test, T, d, n_ch = 60, 15, 10, 16, 3
    X_train = rng.standard_normal((n_train, T, d)).astype(np.float32)
    y_train = rng.standard_normal((n_train, T, n_ch)).astype(np.float32)
    X_test  = rng.standard_normal((n_test,  T, d)).astype(np.float32)
    y_test  = rng.standard_normal((n_test,  T, n_ch)).astype(np.float32)

    scores = np.empty((T, n_ch))
    for t in range(T):
        model = RidgeCV(alphas=ALPHA_LIST_SHORT)
        model.fit(X_train[:, t, :], y_train[:, t, :])
        y_pred = model.predict(X_test[:, t, :])
        scores[t] = scipy.stats.pearsonr(y_test[:, t, :], y_pred, axis=0)[0]

    # With random data, scores across time steps should not be all identical
    assert not np.allclose(scores[0], scores[-1]), (
        "temporal scores at t=0 and t=-1 are identical — "
        "the time loop may not be iterating correctly"
    )
