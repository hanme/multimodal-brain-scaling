"""
Tests for the EEG HDF5 schema produced by format_eeg_hdf5.py (Phase 3).

Two schemas are validated:
  - mean-pool: neural_data [n_stimuli, n_ch],    noise_ceilings [n_ch]
  - temporal:  neural_data [n_stimuli, T, n_ch], noise_ceilings [T, n_ch]

Both must be readable by load_neural_data() and load_neural_metadata() unchanged.
"""

import numpy as np
import pytest
import h5py


# ──────────────────────────────────────────────────────────
# Fixtures: synthetic HDF5 writers
# ──────────────────────────────────────────────────────────

def _write_meanpool_h5(path, n_train=20, n_test=5, n_ch=3,
                       subjects=("sub-01",), rois=("Fz",)):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["subjects"] = list(subjects)
        f.attrs["rois"]     = list(rois)
        f.attrs["splits"]   = ["train", "test"]
        f.attrs["max_nc"]   = 100.0

        offset = 0
        for split, n in [("train", n_train), ("test", n_test)]:
            ids = np.array([f"audio{i:02d}_000000" for i in range(offset, offset + n)], dtype="S")
            offset += n
            f.create_dataset(f"{split}/stimulus_ids", data=ids)
            for sub in subjects:
                for roi in rois:
                    data = rng.standard_normal((n, n_ch)).astype(np.float32)
                    f.create_dataset(f"{split}/neural_data/{sub}/{roi}", data=data)

        for sub in subjects:
            for roi in rois:
                nc = np.full(n_ch, 50.0, dtype=np.float32)
                f.create_dataset(f"noise_ceilings/{sub}/{roi}", data=nc)
    return path


def _write_temporal_h5(path, n_train=20, n_test=5, T=25, n_ch=3,
                       subjects=("sub-01",), rois=("Fz",)):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["subjects"]    = list(subjects)
        f.attrs["rois"]        = list(rois)
        f.attrs["splits"]      = ["train", "test"]
        f.attrs["max_nc"]      = 100.0
        f.attrs["temporal"]    = True
        f.attrs["T_model"]     = T
        f.attrs["time_step_ms"] = 20.0

        offset = 0
        for split, n in [("train", n_train), ("test", n_test)]:
            ids = np.array([f"audio{i:02d}_000000" for i in range(offset, offset + n)], dtype="S")
            f.create_dataset(f"{split}/stimulus_ids", data=ids)
            for sub in subjects:
                for roi in rois:
                    data = rng.standard_normal((n, T, n_ch)).astype(np.float32)
                    f.create_dataset(f"{split}/neural_data/{sub}/{roi}", data=data)
            offset += n

        for sub in subjects:
            for roi in rois:
                nc = np.full((T, n_ch), 50.0, dtype=np.float32)
                f.create_dataset(f"noise_ceilings/{sub}/{roi}", data=nc)
    return path


# ──────────────────────────────────────────────────────────
# Mean-pool schema
# ──────────────────────────────────────────────────────────

def test_meanpool_metadata_readable(tmp_path):
    h5 = _write_meanpool_h5(tmp_path / "eeg.h5")
    from mbs.evaluation.utils.evaluation_helpers import load_neural_metadata
    subjects, rois, splits, nc_max = load_neural_metadata(h5)
    assert "sub-01" in subjects
    assert "Fz" in rois
    assert set(splits) == {"train", "test"}


def test_meanpool_neural_data_shape(tmp_path):
    n_train, n_ch = 20, 3
    h5 = _write_meanpool_h5(tmp_path / "eeg.h5", n_train=n_train, n_ch=n_ch)
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    ids, data, nc = load_neural_data(h5, "sub-01", "Fz", "train")
    assert len(ids) == n_train
    assert data.shape == (n_train, n_ch), f"expected ({n_train}, {n_ch}), got {data.shape}"
    assert nc.shape == (n_ch,)


def test_meanpool_noise_ceiling_is_finite_and_nonneg(tmp_path):
    h5 = _write_meanpool_h5(tmp_path / "eeg.h5")
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    _, _, nc = load_neural_data(h5, "sub-01", "Fz", "train")
    assert np.all(nc >= 0) and np.all(np.isfinite(nc))


def test_meanpool_stimulus_ids_are_strings(tmp_path):
    h5 = _write_meanpool_h5(tmp_path / "eeg.h5")
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    ids, _, _ = load_neural_data(h5, "sub-01", "Fz", "train")
    if hasattr(ids[0], "decode"):
        ids = [x.decode() for x in ids]
    assert all(isinstance(i, str) for i in ids)


def test_meanpool_train_test_ids_disjoint(tmp_path):
    h5 = _write_meanpool_h5(tmp_path / "eeg.h5", n_train=20, n_test=5)
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    train_ids, _, _ = load_neural_data(h5, "sub-01", "Fz", "train")
    test_ids,  _, _ = load_neural_data(h5, "sub-01", "Fz", "test")

    def _decode(ids):
        return {x.decode() if hasattr(x, "decode") else x for x in ids}

    assert _decode(train_ids).isdisjoint(_decode(test_ids))


def test_meanpool_multiple_rois(tmp_path):
    rois = ("Fz", "FCz", "Cz")
    h5 = _write_meanpool_h5(tmp_path / "eeg.h5", rois=rois)
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data, load_neural_metadata
    _, loaded_rois, _, _ = load_neural_metadata(h5)
    assert set(loaded_rois) == set(rois)
    for roi in rois:
        ids, data, nc = load_neural_data(h5, "sub-01", roi, "train")
        assert data.ndim == 2


# ──────────────────────────────────────────────────────────
# Temporal schema
# ──────────────────────────────────────────────────────────

def test_temporal_neural_data_is_3d(tmp_path):
    T, n_ch = 25, 3
    h5 = _write_temporal_h5(tmp_path / "eeg_t.h5", T=T, n_ch=n_ch)
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    ids, data, nc = load_neural_data(h5, "sub-01", "Fz", "train")
    assert data.ndim == 3, f"temporal data must be [n, T, n_ch], got {data.shape}"
    assert data.shape[1] == T
    assert data.shape[2] == n_ch


def test_temporal_noise_ceiling_is_2d(tmp_path):
    T, n_ch = 25, 3
    h5 = _write_temporal_h5(tmp_path / "eeg_t.h5", T=T, n_ch=n_ch)
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    _, _, nc = load_neural_data(h5, "sub-01", "Fz", "train")
    assert nc.ndim == 2, f"temporal noise ceiling must be [T, n_ch], got {nc.shape}"
    assert nc.shape == (T, n_ch)


def test_temporal_metadata_attrs(tmp_path):
    h5 = _write_temporal_h5(tmp_path / "eeg_t.h5", T=25)
    with h5py.File(h5, "r") as f:
        assert f.attrs.get("temporal")  # HDF5 returns numpy bool, not Python True
        assert f.attrs["T_model"] == 25
        assert f.attrs["time_step_ms"] == pytest.approx(20.0)


def test_temporal_train_test_stimulus_counts(tmp_path):
    h5 = _write_temporal_h5(tmp_path / "eeg_t.h5", n_train=20, n_test=5)
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data
    train_ids, train_data, _ = load_neural_data(h5, "sub-01", "Fz", "train")
    test_ids,  test_data,  _ = load_neural_data(h5, "sub-01", "Fz", "test")
    assert len(train_ids) == train_data.shape[0] == 20
    assert len(test_ids)  == test_data.shape[0]  == 5
