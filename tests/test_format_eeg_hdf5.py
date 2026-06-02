"""Tests for format_eeg_hdf5.py (Phase 3).

Fast tests cover the math components and HDF5 schema.
Slow tests run the full formatter against real data (requires mne + dataset).
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — math components (no I/O, no MNE)
# ─────────────────────────────────────────────────────────────────────────────

def test_stim_id_matches_audio_segment_dataset():
    """Stimulus IDs from formatter must match AudioSegmentDataset convention."""
    from mbs.data_prep.format_eeg_hdf5 import _stim_id
    # Window at 30s stride: start_model=0 → audio sample 0 at 16 kHz
    assert _stim_id(1, 0, 50) == "audio01_0000000"
    # Window at 10s stride (1 step): start_model=500 → 500/50 * 16000 = 160000
    assert _stim_id(1, 500, 50) == "audio01_0160000"
    # run 12, stride 2 steps
    assert _stim_id(12, 1000, 50) == "audio12_0320000"


def test_get_segment_starts_count():
    """Number of segments must match (T - W) // stride + 1."""
    from mbs.data_prep.format_eeg_hdf5 import _get_segment_starts
    # audio_dur=180s, window_size=1500 (30s×50Hz), stride=500 (10s×50Hz)
    starts = _get_segment_starts(180.0, 1500, 500, 50)
    expected = (int(180 * 50) - 1500) // 500 + 1  # = (9000-1500)//500+1 = 16
    assert len(starts) == expected
    assert starts[0] == 0
    assert starts[-1] + 1500 <= int(180 * 50)


def test_get_segment_starts_exact_fit():
    """Last window must not extend past end of recording."""
    from mbs.data_prep.format_eeg_hdf5 import _get_segment_starts
    starts = _get_segment_starts(30.0, 1500, 1500, 50)  # non-overlapping, exactly 1 window
    assert starts == [0]


def test_pearsonr_nd_known_values():
    from mbs.data_prep.format_eeg_hdf5 import _pearsonr_nd
    rng = np.random.default_rng(0)
    n, T, n_ch = 50, 10, 4
    x = rng.standard_normal((n, T, n_ch)).astype(np.float32)
    y = x + 0.1 * rng.standard_normal((n, T, n_ch)).astype(np.float32)
    r = _pearsonr_nd(x, y)
    assert r.shape == (T, n_ch)
    assert np.all(r > 0.9), "near-identical arrays should have r > 0.9"


def test_pearsonr_nd_orthogonal():
    from mbs.data_prep.format_eeg_hdf5 import _pearsonr_nd
    rng = np.random.default_rng(1)
    n = 100
    x = rng.standard_normal((n, 3, 2)).astype(np.float32)
    y = rng.standard_normal((n, 3, 2)).astype(np.float32)
    r = _pearsonr_nd(x, y)
    assert r.shape == (3, 2)
    assert np.all(np.abs(r) < 0.4), "random arrays should have low correlation"


def test_noise_ceiling_shape_and_range():
    from mbs.data_prep.format_eeg_hdf5 import _noise_ceiling_from_halves
    rng = np.random.default_rng(2)
    n_stim, T, n_ch = 50, 25, 4
    base = rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    # half1 ≈ half2 → high NC
    h1 = base + 0.05 * rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    h2 = base + 0.05 * rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    nc = _noise_ceiling_from_halves(h1, h2)
    assert nc.shape == (T, n_ch)
    assert np.all(nc >= 0) and np.all(nc <= 100.0 + 1e-4)
    assert np.mean(nc) > 50.0, "near-identical halves should give high NC"


def test_noise_ceiling_independent_halves():
    from mbs.data_prep.format_eeg_hdf5 import _noise_ceiling_from_halves
    rng = np.random.default_rng(3)
    n_stim, T, n_ch = 100, 10, 3
    h1 = rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    h2 = rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    nc = _noise_ceiling_from_halves(h1, h2)
    # Independent noise → NC near 0
    assert np.mean(nc) < 10.0


def test_nc_stored_scale_matches_load_neural_data():
    """NC stored as r_SB^2 * 100 with max_nc=100 must round-trip via load_neural_data."""
    import h5py
    import tempfile, os
    from mbs.data_prep.format_eeg_hdf5 import _noise_ceiling_from_halves
    from mbs.evaluation.utils.evaluation_helpers import load_neural_data

    rng = np.random.default_rng(4)
    n_stim, T, n_ch = 40, 5, 2
    base = rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    h1 = base + 0.1 * rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    h2 = base + 0.1 * rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
    nc_stored = _noise_ceiling_from_halves(h1, h2)  # [T, n_ch], 0-100

    with tempfile.TemporaryDirectory() as td:
        h5_path = os.path.join(td, "eeg.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs["subjects"]    = ["group"]
            f.attrs["rois"]        = ["Fz"]
            f.attrs["splits"]      = ["train", "test"]
            f.attrs["max_nc"]      = 100.0
            f.attrs["temporal"]    = True

            ids = np.array([f"audio01_{i:07d}" for i in range(n_stim)], dtype="S")
            f.create_dataset("train/stimulus_ids", data=ids)
            data = rng.standard_normal((n_stim, T, n_ch)).astype(np.float32)
            f.create_dataset("train/neural_data/group/Fz", data=data)
            f.create_dataset("noise_ceilings/group/Fz", data=nc_stored)

        _, _, nc_loaded = load_neural_data(h5_path, "group", "Fz", "train")
        # load_neural_data does: sqrt(nc_stored / 100) → Pearson r
        # nc_stored = r_SB^2 * 100 → nc_loaded = r_SB
        assert nc_loaded.shape == (T, n_ch)
        assert np.all(nc_loaded >= 0) and np.all(nc_loaded <= 1.01)
        # high-correlation halves → nc_loaded should be substantial
        assert np.mean(nc_loaded) > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — full formatter (requires mne + real dataset)
# ─────────────────────────────────────────────────────────────────────────────

BIDS_ROOT = "/work/upschrimpf1/mehrer/datasets/Broderick_2018_EEG_The_old_man_and_the_sea"


@pytest.mark.slow
def test_format_eeg_hdf5_smoke(tmp_path):
    """Run formatter on 2 subjects × 3 runs with small windows; verify HDF5 schema."""
    pytest.importorskip("mne")
    import h5py
    from pathlib import Path

    bids_root = Path(BIDS_ROOT)
    if not bids_root.exists():
        pytest.skip(f"Dataset not found: {bids_root}")

    output_path = tmp_path / "eeg_test.h5"

    # Monkey-patch subjects + runs to keep the test fast
    from mbs.data_prep import format_eeg_hdf5 as mod

    original_main = mod.main

    import argparse
    args = argparse.Namespace(
        bids_root=str(bids_root),
        output_path=str(output_path),
        window_duration=30.0,
        window_stride=30.0,   # non-overlapping for speed
        target_sr=50,
        n_test_runs=1,
        overwrite=False,
        seed=42,
    )

    # Limit to first 2 subjects and first 3 runs by patching the subject/run list
    _orig_load = mod._load_run_eeg
    _call_count = {"n": 0}

    def _patched_load(subj, run, bids_root_, audio_dur, target_sr):
        if subj not in ("sub-001", "sub-002"):
            return None
        if run > 3:
            return None
        return _orig_load(subj, run, bids_root_, audio_dur, target_sr)

    mod._load_run_eeg = _patched_load
    try:
        mod.main(args)
    finally:
        mod._load_run_eeg = _orig_load

    assert output_path.exists()
    with h5py.File(output_path, "r") as f:
        assert f.attrs.get("temporal")
        assert f.attrs["T_model"] == 1500
        assert f.attrs["time_step_ms"] == pytest.approx(20.0)
        assert "group" in f.attrs["subjects"]
        assert "train" in f and "test" in f
        assert "noise_ceilings" in f

        train_ids = f["train/stimulus_ids"][()]
        assert len(train_ids) > 0

        rois = list(f.attrs["rois"])
        assert "whole_brain" in rois
        subj_roi_path = f"train/neural_data/group/{rois[0]}"
        assert subj_roi_path in f
        shape = f[subj_roi_path].shape
        assert shape[1] == 1500, f"Expected T=1500, got {shape[1]}"
