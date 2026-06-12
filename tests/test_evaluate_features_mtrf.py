"""Unit + synthetic-recovery tests for the mTRF lagged-Ridge evaluator."""

from types import SimpleNamespace

import numpy as np
import pytest

from mbs.evaluation.evaluate_features_mtrf import (
    lags_in_bins,
    sample_time_indices,
    build_lagged_design,
    pearson_along_time,
    mask_channels,
    highpass_along_time,
    fit_score_block,
)


def test_lags_in_bins_basic():
    # 0..800 ms at 20 ms step on a 20 ms grid -> bins 0..40 inclusive.
    lags = lags_in_bins(0.0, 800.0, 20.0, 20.0)
    assert lags[0] == 0 and lags[-1] == 40
    assert lags.tolist() == list(range(41))


def test_lags_in_bins_snaps_and_dedups():
    # 50..100 ms at 10 ms on a 20 ms grid snaps to bins {3,4,5} (50->2.5->2? rounding)
    lags = lags_in_bins(50.0, 100.0, 10.0, 20.0)
    # rounded: 50->2, 60->3, 70->4 (3.5->4), 80->4, 90->4(4.5->4), 100->5
    assert lags.min() >= 0
    assert np.all(np.diff(lags) > 0)  # sorted, unique


def test_sample_time_indices_in_range_and_unique():
    rng = np.random.default_rng(0)
    idx = sample_time_indices(T=100, max_lag=10, n_samples=20, rng=rng)
    assert idx.min() >= 10 and idx.max() < 100
    assert len(np.unique(idx)) == len(idx)


def test_sample_time_indices_full_when_n_large():
    rng = np.random.default_rng(0)
    idx = sample_time_indices(T=50, max_lag=5, n_samples=10_000, rng=rng)
    assert np.array_equal(idx, np.arange(5, 50))


def test_build_lagged_design_gather_is_correct():
    n_stim, T, d, n_ch = 2, 12, 3, 1
    feats = np.arange(n_stim * T * d, dtype=np.float32).reshape(n_stim, T, d)
    eeg = np.zeros((n_stim, T, n_ch), dtype=np.float32)
    lags = np.array([0, 2])
    time_idx = np.array([5, 8])
    X, Y = build_lagged_design(feats, eeg, lags, time_idx)
    # rows = n_stim * n_t = 2 * 2 = 4 ; cols = n_lags * d = 2 * 3 = 6
    assert X.shape == (4, 6)
    assert Y.shape == (4, 1)
    # first row: stim 0, t=5 -> [feats[0,5], feats[0,3]] concatenated
    expected = np.concatenate([feats[0, 5], feats[0, 3]])
    np.testing.assert_allclose(X[0], expected)


def test_build_lagged_design_rejects_too_early_time():
    feats = np.zeros((1, 10, 2), np.float32)
    eeg = np.zeros((1, 10, 1), np.float32)
    with pytest.raises(AssertionError):
        build_lagged_design(feats, eeg, np.array([5]), np.array([3]))  # t=3 < lag 5


def test_pearson_along_time_perfect_and_zero():
    y = np.random.default_rng(1).normal(size=(200, 2)).astype(np.float32)
    r = pearson_along_time(y, y.copy())
    np.testing.assert_allclose(r, np.ones(2), atol=1e-5)
    # constant prediction -> r defined as 0
    r0 = pearson_along_time(y, np.ones_like(y))
    np.testing.assert_allclose(r0, np.zeros(2), atol=1e-6)


def test_single_lag_recovers_planted_lag():
    """EEG[t] is a linear readout of features at a known lag; the encoding-vs-lag
    curve must peak at that lag."""
    rng = np.random.default_rng(42)
    n_stim, T, d = 30, 300, 8
    true_lag = 6  # bins

    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    w = rng.normal(size=(d, 1)).astype(np.float32)
    eeg = np.zeros((n_stim, T, 1), dtype=np.float32)
    eeg[:, true_lag:, 0] = (feats[:, : T - true_lag, :] @ w)[..., 0]
    eeg += 0.05 * rng.normal(size=eeg.shape).astype(np.float32)  # light noise

    # split stimuli into train/test
    feat_train, feat_test = feats[:20], feats[20:]
    eeg_train, eeg_test = eeg[:20], eeg[20:]
    nc = np.ones((T, 1), dtype=np.float32)

    lags = np.arange(0, 13)  # 0..12 bins, includes the true lag
    args = SimpleNamespace(
        mode="single_lag", n_train_time_samples=0, test_time_stride=1,
        feature_pca=0, standardize_features=True, highpass_hz=0.0,
        noise_ceiling_correct=False, use_wide_range_alphas=False,
    )
    scores_raw, _ = fit_score_block(
        feat_train, feat_test, eeg_train, eeg_test, np.ones(1, np.float32),
        lags, args, np.random.default_rng(0), fs=50.0,
    )
    curve = scores_raw[:, 0]
    assert int(np.argmax(curve)) == true_lag, f"peak at {np.argmax(curve)}, expected {true_lag}"
    assert curve[true_lag] > 0.9
    # a clearly wrong lag should be much weaker
    assert curve[0] < 0.5


def test_mask_channels_threshold():
    T = 40
    nc = np.stack([np.full(T, 0.5), np.zeros(T), np.full(T, 0.01)], axis=1).astype(np.float32)
    kept = mask_channels(nc, threshold=0.0)
    assert kept.tolist() == [0, 2]            # channels with mean NC > 0
    kept2 = mask_channels(nc, threshold=0.05)
    assert kept2.tolist() == [0]              # only the strong channel survives 0.05


def test_alpha_per_target_equals_separate_fit():
    """Joint multi-output fit must give per-channel results identical to separate fits."""
    rng = np.random.default_rng(7)
    n_stim, T, d = 24, 200, 6
    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    w0, w1 = rng.normal(size=(d, 1)), rng.normal(size=(d, 1))
    eeg = np.zeros((n_stim, T, 2), np.float32)
    eeg[:, 3:, 0:1] = (feats[:, :T - 3] @ w0)
    eeg[:, 3:, 1:2] = (feats[:, :T - 3] @ w1) + 3.0 * rng.normal(size=(n_stim, T - 3, 1))  # noisy
    lags = np.array([3])
    args = SimpleNamespace(
        mode="single_lag", n_train_time_samples=0, test_time_stride=1,
        feature_pca=0, standardize_features=True, highpass_hz=0.0,
        noise_ceiling_correct=False, use_wide_range_alphas=False,
    )
    joint, _ = fit_score_block(feats[:16], feats[16:], eeg[:16], eeg[16:],
                               np.ones(2, np.float32), lags, args, np.random.default_rng(0), 50.0)
    sep0, _ = fit_score_block(feats[:16], feats[16:], eeg[:16, :, 0:1], eeg[16:, :, 0:1],
                              np.ones(1, np.float32), lags, args, np.random.default_rng(0), 50.0)
    # high-SNR channel score identical whether fit alone or alongside the noisy channel
    np.testing.assert_allclose(joint[0, 0], sep0[0, 0], atol=1e-5)


def test_highpass_removes_dc():
    x = np.ones((2, 100, 1), np.float32) * 5.0 + np.random.default_rng(0).normal(size=(2, 100, 1))
    xf = highpass_along_time(x, fs=50.0, cutoff_hz=1.0)
    assert abs(xf.mean()) < 0.2          # DC/offset removed
    assert highpass_along_time(x, 50.0, 0.0) is x  # off = passthrough
