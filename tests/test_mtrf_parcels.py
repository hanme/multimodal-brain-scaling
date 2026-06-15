"""TDD for the parcel-level mTRF encoder + per-dataset (D2/D3) held-out scoring.

Defines the API of:
  - mbs.evaluation.attn_probe.dataset_temporal.{list_test_splits, recompute_parcel_nc}
  - mbs.evaluation.evaluate_features_mtrf_parcels.{fit_parcel_mtrf, score_parcel_mtrf}

Pins the three D2/D3 invariants (plan §13.2):
  * SAME parcels across datasets, only the NC recomputed per dataset (decision C).
  * per-dataset held-out scoring — score test_d1 / test_d2 SEPARATELY, never pooled, because
    pooling two differently-scaled groups inflates the correlation (the "leak").
All tests run on CPU with tiny synthetic data (no real dataset).
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from mbs.evaluation.evaluate_features_mtrf import (
    lags_in_bins, pearson_along_time,
)
from mbs.evaluation.attn_probe.dataset_temporal import (
    list_test_splits, recompute_parcel_nc, load_parcel_eeg,
)
from mbs.evaluation.evaluate_features_mtrf_parcels import (
    fit_parcel_mtrf, score_parcel_mtrf,
)


# ---------------------------------------------------------------------------
# fixtures: tiny neural HDF5s, single-split and D3-style two-test-group
# ---------------------------------------------------------------------------

def _write_neural(path, splits, chans, T=12, nc_pct=60.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.attrs["subjects"] = ["group"]
        f.attrs["rois"] = chans
        f.attrs["splits"] = [s for s in splits if s in ("train", "test")]
        f.attrs["max_nc"] = 100.0
        f.attrs["time_step_ms"] = 20.0
        for split, n in splits.items():
            g = f.create_group(split)
            g.create_dataset("stimulus_ids",
                             data=np.array([f"{split}_{i:05d}".encode() for i in range(n)]))
            nd = g.create_group("neural_data").create_group("group")
            for c in chans:
                nd.create_dataset(c, data=(scale * rng.normal(size=(n, T, 1))).astype(np.float32))
        nc = f.create_group("noise_ceilings").create_group("group")
        for c in chans:
            nc.create_dataset(c, data=np.full((T, 1), nc_pct, np.float32))


@pytest.fixture
def single_split_h5(tmp_path):
    p = tmp_path / "d2_like.h5"
    _write_neural(p, {"train": 4, "test": 2}, ["Fz", "F3", "T7"])
    return p


@pytest.fixture
def d3_like_h5(tmp_path):
    p = tmp_path / "d3_like.h5"
    _write_neural(p, {"train": 6, "test_d1": 3, "test_d2": 2}, ["Fz", "F3", "T7"])
    return p


# ---------------------------------------------------------------------------
# list_test_splits — D2 -> ["test"], D3 -> ["test_d1","test_d2"]
# ---------------------------------------------------------------------------

def test_list_test_splits_single(single_split_h5):
    assert list_test_splits(single_split_h5) == ["test"]


def test_list_test_splits_d3(d3_like_h5):
    assert list_test_splits(d3_like_h5) == ["test_d1", "test_d2"]


# ---------------------------------------------------------------------------
# recompute_parcel_nc — SAME members, dataset-specific NC (decision C)
# ---------------------------------------------------------------------------

def test_recompute_parcel_nc_keeps_members(single_split_h5):
    parcels = [("frontal", ["Fz", "F3"], 0.99), ("temporal", ["T7"], 0.99)]
    out = recompute_parcel_nc(parcels, single_split_h5)
    assert [(n, list(m)) for n, m, _ in out] == [("frontal", ["Fz", "F3"]), ("temporal", ["T7"])]
    # nc recomputed from this dataset: sqrt(60/100) ~ 0.7746 for every channel here
    np.testing.assert_allclose([r for _, _, r in out], [np.sqrt(0.6)] * 2, atol=1e-3)


# ---------------------------------------------------------------------------
# THE LEAK — pooling two differently-scaled groups inflates r; per-split does not
# ---------------------------------------------------------------------------

def test_pooled_scoring_inflates_but_persplit_does_not():
    """A model that only predicts each group's MEAN (no within-group structure) scores ~0
    within each group but ~1 when the two groups are pooled. This is exactly why D3 must be
    scored per dataset."""
    rng = np.random.default_rng(0)
    n = 600
    y1 = rng.normal(0, 1, (n, 1)).astype(np.float32) + 10.0     # group 1, offset +10
    y2 = rng.normal(0, 1, (n, 1)).astype(np.float32) + 0.0      # group 2, offset 0
    yhat1 = rng.normal(0, 1, (n, 1)).astype(np.float32) + 10.0  # predicts the offset only
    yhat2 = rng.normal(0, 1, (n, 1)).astype(np.float32) + 0.0

    r1 = pearson_along_time(y1, yhat1)[0]
    r2 = pearson_along_time(y2, yhat2)[0]
    assert abs(r1) < 0.15 and abs(r2) < 0.15                    # within-group: no real signal

    yp = np.concatenate([y1, y2], 0)
    yhatp = np.concatenate([yhat1, yhat2], 0)
    rp = pearson_along_time(yp, yhatp)[0]
    assert rp > 0.8                                             # pooled: spuriously high


# ---------------------------------------------------------------------------
# fit + score parcel mTRF — recovers a planted feature->parcel lag relationship
# ---------------------------------------------------------------------------

def test_fit_score_parcel_mtrf_recovers_signal():
    rng = np.random.default_rng(42)
    n_stim, T, d, P = 30, 300, 8, 2
    true_lag = 5
    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    eeg = np.zeros((n_stim, T, P), np.float32)
    for p in range(P):
        w = rng.normal(size=(d,)).astype(np.float32)
        eeg[:, true_lag:, p] = feats[:, : T - true_lag, :] @ w
    eeg += 0.05 * rng.normal(size=eeg.shape).astype(np.float32)

    ftr, fte = feats[:20], feats[20:]
    etr, ete = eeg[:20], eeg[20:]
    lags = lags_in_bins(0.0, 200.0, 20.0, 20.0)  # 0..10 bins, includes lag 5

    fit = fit_parcel_mtrf(ftr, etr, lags, highpass_hz=0.0, n_train_time_samples=0,
                          fs=50.0, rng=np.random.default_rng(0))
    r = score_parcel_mtrf(fit, fte, ete, lags, highpass_hz=0.0, fs=50.0)
    assert r.shape == (P,)
    assert (r > 0.8).all(), f"held-out r too low: {r}"


# ---------------------------------------------------------------------------
# PCA option (pca_var) — variance-preserving feature reduction before lagging
# ---------------------------------------------------------------------------

def test_fit_score_parcel_mtrf_with_pca_reduces_and_recovers():
    """With pca_var=0.95 on (near) rank-3 features, the fit stores a PCA that keeps far fewer
    than d components, yet the planted feature->parcel lag relationship is still recovered. This
    is the model-independent design-shrink that replaces the eigen workaround (project_plan §15)."""
    rng = np.random.default_rng(7)
    n_stim, T, d, rank, P = 30, 300, 12, 3, 2
    true_lag = 5
    latent = rng.normal(size=(n_stim, T, rank)).astype(np.float32)
    A = rng.normal(size=(rank, d)).astype(np.float32)
    feats = (latent @ A).astype(np.float32)
    feats += 0.01 * rng.normal(size=feats.shape).astype(np.float32)   # near rank-3
    eeg = np.zeros((n_stim, T, P), np.float32)
    for p in range(P):
        w = rng.normal(size=(d,)).astype(np.float32)
        eeg[:, true_lag:, p] = feats[:, : T - true_lag, :] @ w
    eeg += 0.05 * rng.normal(size=eeg.shape).astype(np.float32)

    ftr, fte = feats[:20], feats[20:]
    etr, ete = eeg[:20], eeg[20:]
    lags = lags_in_bins(0.0, 200.0, 20.0, 20.0)

    fit = fit_parcel_mtrf(ftr, etr, lags, highpass_hz=0.0, n_train_time_samples=0,
                          fs=50.0, rng=np.random.default_rng(0), pca_var=0.95)
    assert fit["pca"] is not None
    assert fit["pca"].n_components_ < d            # 95% var on rank-3 data -> few PCs, < d
    r = score_parcel_mtrf(fit, fte, ete, lags, highpass_hz=0.0, fs=50.0)
    assert r.shape == (P,)
    assert (r > 0.7).all(), f"held-out r too low with PCA: {r}"


def test_fit_parcel_mtrf_no_pca_by_default():
    """Default (pca_var unset) keeps the raw-feature path: fit['pca'] is None."""
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(5, 50, 6)).astype(np.float32)
    eeg = rng.normal(size=(5, 50, 2)).astype(np.float32)
    lags = lags_in_bins(0.0, 100.0, 20.0, 20.0)
    fit = fit_parcel_mtrf(feats, eeg, lags, highpass_hz=0.0, n_train_time_samples=0,
                          fs=50.0, rng=np.random.default_rng(0))
    assert fit["pca"] is None


def test_score_parcel_mtrf_per_split_uses_only_that_split(d3_like_h5):
    """Loading test_d1 vs test_d2 yields different stimulus sets (no pooling at the data layer)."""
    parcels = [("frontal", ["Fz", "F3"], 0.7), ("temporal", ["T7"], 0.7)]
    ids1, eeg1 = load_parcel_eeg(d3_like_h5, "group", parcels, "test_d1")
    ids2, eeg2 = load_parcel_eeg(d3_like_h5, "group", parcels, "test_d2")
    assert eeg1.shape[0] == 3 and eeg2.shape[0] == 2
    assert set(ids1).isdisjoint(set(ids2))
