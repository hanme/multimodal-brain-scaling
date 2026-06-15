"""Unit + synthetic-recovery tests for the temporal attention probe (Workstream B).

TDD: these tests define the API of the new modules
  - mbs.evaluation.attn_probe.dataset_temporal
  - mbs.evaluation.attn_probe.engine_temporal
before they are implemented. They mirror the style of test_evaluate_features_mtrf.py
and deliberately reuse the mTRF pure functions to keep Method A / Method B consistent.

All tests run on CPU with tiny tensors (no GPU, no MNE, no real dataset). The neural HDF5
is synthesised in a fixture matching the broderick2018_30s.h5 contract, extended with the
per-subject layout that Workstream B's "Kadir individual" variant requires.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from mbs.evaluation.attn_probe.dataset_temporal import (
    build_windowed_design,
    sampled_windowed_design,
    load_parcel_eeg,
    WindowedTemporalDataset,
)
from mbs.evaluation.attn_probe.engine_temporal import (
    corr_loss,
    build_probe_system,
    predict_concat,
    score_heldout,
    train_temporal_probe,
    TemporalTrainConfig,
)
from mbs.evaluation.evaluate_features_mtrf import (
    build_lagged_design,
    pearson_along_time,
)


# ---------------------------------------------------------------------------
# build_windowed_design — the core reframing (lookback window kept as tokens)
# ---------------------------------------------------------------------------

def test_windowed_design_shapes():
    n_stim, T, d, P, L = 3, 40, 5, 2, 8
    feats = np.random.default_rng(0).normal(size=(n_stim, T, d)).astype(np.float32)
    eeg = np.random.default_rng(1).normal(size=(n_stim, T, P)).astype(np.float32)
    time_idx = np.array([10, 20, 30])
    X, Y = build_windowed_design(feats, eeg, lookback=L, time_idx=time_idx)
    # rows = n_stim * n_t ; window kept as a token axis (NOT flattened)
    assert X.shape == (n_stim * len(time_idx), L, d)
    assert Y.shape == (n_stim * len(time_idx), P)


def test_windowed_design_is_causal_and_chronological():
    """Each window is [t-L+1 .. t] (look-back only); last token == current time t."""
    n_stim, T, d, L = 1, 30, 1, 6
    feats = np.tile(np.arange(T, dtype=np.float32)[None, :, None], (n_stim, 1, d))  # feats[0,t,0]=t
    eeg = np.zeros((n_stim, T, 1), np.float32)
    t = 15
    X, _ = build_windowed_design(feats, eeg, lookback=L, time_idx=np.array([t]))
    window = X[0, :, 0]                      # [L]
    # chronological: oldest first, current last; no future leakage
    np.testing.assert_array_equal(window, np.arange(t - L + 1, t + 1))
    assert window[-1] == t                    # last token is the current bin
    assert window.max() <= t                  # never looks into the future


def test_windowed_design_matches_mtrf_lagged_design():
    """Consistency with Method A: the window, read newest->oldest, equals the lag stack
    for lags 0..L-1. Token index (L-1-lag) holds feature at lag `lag`."""
    rng = np.random.default_rng(3)
    n_stim, T, d, L = 2, 25, 4, 5
    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    eeg = rng.normal(size=(n_stim, T, 1)).astype(np.float32)
    time_idx = np.array([12, 18])
    lags = np.arange(L)                       # 0..L-1
    Xw, Yw = build_windowed_design(feats, eeg, lookback=L, time_idx=time_idx)
    Xl, Yl = build_lagged_design(feats, eeg, lags, time_idx)
    np.testing.assert_allclose(Yw, Yl, atol=1e-6)
    for lag in range(L):
        token = L - 1 - lag                   # newest token = lag 0
        np.testing.assert_allclose(Xw[:, token, :], Xl[:, lag * d:(lag + 1) * d], atol=1e-6)


def test_windowed_design_rejects_too_early_time():
    feats = np.zeros((1, 10, 2), np.float32)
    eeg = np.zeros((1, 10, 1), np.float32)
    with pytest.raises(AssertionError):
        build_windowed_design(feats, eeg, lookback=6, time_idx=np.array([4]))  # 4 < L-1=5


def test_sampled_windowed_design_respects_lookback_and_shapes():
    rng = np.random.default_rng(5)
    n_stim, T, d, P, L, n_t = 4, 60, 3, 2, 10, 7
    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    eeg = rng.normal(size=(n_stim, T, P)).astype(np.float32)
    X, Y = sampled_windowed_design(feats, eeg, lookback=L, n_time_samples=n_t, rng=rng)
    assert X.shape == (n_stim * n_t, L, d)
    assert Y.shape == (n_stim * n_t, P)


# ---------------------------------------------------------------------------
# corr_loss — 1 - Pearson over the batch axis (matches the along-time score)
# ---------------------------------------------------------------------------

def test_corr_loss_perfect_and_anti():
    y = torch.randn(128, 3)
    assert corr_loss(y, y.clone()).item() == pytest.approx(0.0, abs=1e-4)   # r=+1 -> 0
    assert corr_loss(-y, y).item() == pytest.approx(2.0, abs=1e-4)          # r=-1 -> 2


def test_corr_loss_matches_numpy_pearson():
    rng = np.random.default_rng(11)
    y = rng.normal(size=(200, 4)).astype(np.float32)
    yhat = (0.7 * y + 0.5 * rng.normal(size=y.shape)).astype(np.float32)
    r_np = pearson_along_time(y, yhat)                    # [P]
    loss = corr_loss(torch.from_numpy(yhat), torch.from_numpy(y)).item()
    assert loss == pytest.approx(float((1.0 - r_np).mean()), abs=1e-5)


def test_corr_loss_is_differentiable():
    y = torch.randn(64, 2)
    yhat = torch.randn(64, 2, requires_grad=True)
    corr_loss(yhat, y).backward()
    assert yhat.grad is not None and torch.isfinite(yhat.grad).all()


# ---------------------------------------------------------------------------
# probe construction + readout-level selectability (group vs individual)
# ---------------------------------------------------------------------------

def test_build_probe_group_has_single_head():
    model = build_probe_system(in_dim=8, n_parcel=4, subjects=["group"])
    assert list(model.heads.heads.keys()) == ["group"]


def test_build_probe_individual_has_one_head_per_subject():
    subs = ["sub-001", "sub-002", "sub-003"]
    model = build_probe_system(in_dim=8, n_parcel=4, subjects=subs)
    assert sorted(model.heads.heads.keys()) == sorted(subs)


def test_probe_forward_window_to_parcels():
    B, L, d, P = 5, 10, 8, 4
    model = build_probe_system(in_dim=d, n_parcel=P, subjects=["group"])
    out = model(torch.randn(B, L, d), subject="group")
    assert out.shape == (B, P)


# ---------------------------------------------------------------------------
# along-time scoring (Kadir-style: concatenate predictions over time, then r)
# ---------------------------------------------------------------------------

def test_predict_concat_shapes():
    n_stim, T, d, P, L = 3, 40, 8, 4, 6
    feats = np.random.default_rng(0).normal(size=(n_stim, T, d)).astype(np.float32)
    model = build_probe_system(in_dim=d, n_parcel=P, subjects=["group"])
    Yhat = predict_concat(model, feats, lookback=L, subject="group", device="cpu")
    # valid output times per stim = T - (L-1); concatenated over stimuli
    assert Yhat.shape == (n_stim * (T - (L - 1)), P)


def test_score_heldout_agrees_with_manual_concat():
    """score_heldout must equal pearson_along_time(Y_concat, predict_concat(...)) —
    same metric Method A reports, computed by concatenating predictions over time."""
    rng = np.random.default_rng(2)
    n_stim, T, d, P, L = 3, 50, 8, 2, 5
    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    eeg = rng.normal(size=(n_stim, T, P)).astype(np.float32)

    model = build_probe_system(in_dim=d, n_parcel=P, subjects=["group"])
    r = score_heldout(model, feats, eeg, lookback=L, subject="group", device="cpu")

    Yhat = predict_concat(model, feats, lookback=L, subject="group", device="cpu")
    Y = eeg[:, L - 1:, :].reshape(n_stim * (T - (L - 1)), P)
    np.testing.assert_allclose(r, pearson_along_time(Y, Yhat), atol=1e-5)


# ---------------------------------------------------------------------------
# end-to-end learning: planted lag is recovered by the trained probe
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_probe_learns_planted_lag():
    """EEG parcels are linear readouts of features at known lags inside the window.
    A trained probe must reach much higher held-out r than at init."""
    rng = np.random.default_rng(42)
    n_stim, T, d, P, L = 24, 120, 6, 2, 10
    lags = [3, 7]                                   # one per parcel, both inside the window
    feats = rng.normal(size=(n_stim, T, d)).astype(np.float32)
    eeg = np.zeros((n_stim, T, P), np.float32)
    for p, k in enumerate(lags):
        w = rng.normal(size=(d,)).astype(np.float32)
        eeg[:, k:, p] = feats[:, : T - k, :] @ w
    eeg += 0.05 * rng.normal(size=eeg.shape).astype(np.float32)

    ftr, fte = feats[:16], feats[16:]
    etr, ete = eeg[:16], eeg[16:]

    model = build_probe_system(in_dim=d, n_parcel=P, subjects=["group"])
    r_init = score_heldout(model, fte, ete, lookback=L, subject="group", device="cpu")

    cfg = TemporalTrainConfig(device="cpu", epochs=120, lr=1e-3,
                              n_train_time_samples=40, batch_size=256, seed=0)
    model, _ = train_temporal_probe(
        feats_train={"group": ftr}, eeg_train={"group": etr},
        feats_val={"group": fte}, eeg_val={"group": ete},
        subjects=["group"], in_dim=d, n_parcel=P, lookback=L, train_cfg=cfg,
    )
    r_trained = score_heldout(model, fte, ete, lookback=L, subject="group", device="cpu")

    assert r_trained.mean() > r_init.mean() + 0.2     # learning happened
    assert r_trained.mean() > 0.5                     # and it is genuinely predictive


# ---------------------------------------------------------------------------
# per-subject neural HDF5 contract (Workstream B "Kadir individual")
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_neural_h5(tmp_path):
    """A tiny neural HDF5 matching the per-subject layout B-data must produce:
       <split>/neural_data/<subject>/<roi>  and  noise_ceilings/<subject>/<roi>,
    with 'group' alongside the individual subjects."""
    path = tmp_path / "neural_tiny.h5"
    subjects = ["group", "sub-001", "sub-002"]
    chans = ["A1", "A2", "A3"]
    n_tr, n_te, T = 4, 2, 12
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["subjects"] = subjects
        f.attrs["rois"] = chans
        f.attrs["splits"] = ["train", "test"]
        f.attrs["max_nc"] = 100.0
        f.attrs["time_step_ms"] = 20.0
        for split, n in [("train", n_tr), ("test", n_te)]:
            g = f.create_group(split)
            ids = np.array([f"audio0{split[0]}_{i:05d}".encode() for i in range(n)])
            g.create_dataset("stimulus_ids", data=ids)
            nd = g.create_group("neural_data")
            for s in subjects:
                gs = nd.create_group(s)
                for c in chans:
                    gs.create_dataset(c, data=rng.normal(size=(n, T, 1)).astype(np.float32))
        nc = f.create_group("noise_ceilings")
        for s in subjects:
            gs = nc.create_group(s)
            for c in chans:
                gs.create_dataset(c, data=np.full((T, 1), 60.0, np.float32))
    return path, subjects, chans


def test_load_parcel_eeg_averages_members_per_subject(synthetic_neural_h5):
    path, subjects, chans = synthetic_neural_h5
    parcels = [("pA", ["A1", "A2"], 0.5), ("pB", ["A3"], 0.3)]
    ids, eeg = load_parcel_eeg(path, subject="sub-001", parcels=parcels, split="train")
    assert eeg.shape == (4, 12, 2)            # [n_stim, T, n_parcel]
    assert len(ids) == 4
    # parcel pA is the raw mean of A1, A2 for that subject
    with h5py.File(path) as f:
        a1 = f["train"]["neural_data"]["sub-001"]["A1"][:, :, 0]
        a2 = f["train"]["neural_data"]["sub-001"]["A2"][:, :, 0]
    np.testing.assert_allclose(eeg[:, :, 0], 0.5 * (a1 + a2), atol=1e-6)


def test_load_parcel_eeg_subjects_differ(synthetic_neural_h5):
    path, subjects, chans = synthetic_neural_h5
    parcels = [("pA", ["A1", "A2"], 0.5)]
    _, eeg1 = load_parcel_eeg(path, subject="sub-001", parcels=parcels, split="train")
    _, eeg2 = load_parcel_eeg(path, subject="sub-002", parcels=parcels, split="train")
    assert not np.allclose(eeg1, eeg2)        # per-subject data really is distinct


def test_probe_driver_scores_each_dataset_split_separately(tmp_path):
    """End-to-end wiring of the probe driver on a D3-shaped h5 (train + test_d1 + test_d2):
    the output must carry per-split held-out r, not a single pooled number."""
    from types import SimpleNamespace
    from mbs.evaluation.evaluate_features_attn_probe_temporal import run_layer

    chans = ["Fz", "F3", "T7"]
    parcels = [("frontal", ["Fz", "F3"], 0.5), ("temporal", ["T7"], 0.5)]
    layer = "blocks.2"
    d, T = 6, 14
    splits = {"train": 5, "test_d1": 3, "test_d2": 2}

    neural = tmp_path / "d3.h5"
    rng = np.random.default_rng(0)
    all_ids = []
    with h5py.File(neural, "w") as f:
        f.attrs["subjects"] = ["group"]; f.attrs["rois"] = chans
        f.attrs["splits"] = ["train"]; f.attrs["max_nc"] = 100.0; f.attrs["time_step_ms"] = 20.0
        for split, n in splits.items():
            g = f.create_group(split)
            ids = np.array([f"{split}_{i:05d}".encode() for i in range(n)])
            all_ids.extend(s.decode() for s in ids)
            g.create_dataset("stimulus_ids", data=ids)
            nd = g.create_group("neural_data").create_group("group")
            for c in chans:
                nd.create_dataset(c, data=rng.normal(size=(n, T, 1)).astype(np.float32))
        nc = f.create_group("noise_ceilings").create_group("group")
        for c in chans:
            nc.create_dataset(c, data=np.full((T, 1), 50.0, np.float32))

    fdir = tmp_path / "feats"; fdir.mkdir()
    with h5py.File(fdir / "feats_0.h5", "w") as f:
        f.create_dataset(f"features/{layer.replace('.', '-')}",
                         data=rng.normal(size=(len(all_ids), T, d)).astype(np.float32))
        f.create_dataset("ids", data=np.array([s.encode() for s in all_ids]))

    args = SimpleNamespace(
        features_dir=str(fdir), data_hdf5_path=str(neural), highpass_hz=0.5,
        d_model=16, nhead=4, num_latents=4, cross_attn_layers=1, dropout=0.0, pos_mode="learned",
        device="cpu", epochs=2, lr=1e-3, weight_decay=1e-4, batch_size=64,
        n_train_time_samples=8, amp=False, seed=0, readout_level="group",
    )
    with h5py.File(tmp_path / "out.h5", "w") as out_h5:
        entry = run_layer(args, layer, parcels, ["group"], lookback=4, out_h5=out_h5)
        g = out_h5[layer.replace(".", "-")]
        assert "heldout_r__test_d1" in g and "heldout_r__test_d2" in g
        assert g["heldout_r__test_d1"].shape == (2,)        # 2 parcels
    assert set(entry["splits"]) == {"test_d1", "test_d2"}


def test_windowed_dataset_yields_window_and_parcels(synthetic_neural_h5):
    """The Dataset ties feature windows to per-subject parcel EEG by stimulus id."""
    path, subjects, chans = synthetic_neural_h5
    parcels = [("pA", ["A1", "A2"], 0.5), ("pB", ["A3"], 0.3)]
    d, L = 5, 4
    # feature table aligned to the train stimulus ids
    with h5py.File(path) as f:
        ids = [s.decode() for s in f["train"]["stimulus_ids"][:]]
    T = 12
    feats = np.random.default_rng(1).normal(size=(len(ids), T, d)).astype(np.float32)
    id_map = {sid: i for i, sid in enumerate(ids)}

    ds = WindowedTemporalDataset(
        neural_h5_path=path, subject="sub-001", parcels=parcels, split="train",
        layer_feats=feats, feature_id_map=id_map, lookback=L,
        n_time_samples=5, seed=0,
    )
    item = ds[0]
    assert item["feats"].shape == (L, d)
    assert item["y"].shape == (len(parcels),)
