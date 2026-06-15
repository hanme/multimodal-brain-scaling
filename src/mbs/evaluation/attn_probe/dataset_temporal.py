"""Time-resolved dataset for the temporal attention probe (Workstream B).

The reframing (see aux/project_plan_20260611.md §3.0): predict ``EEG[t]`` from the
**lookback window of features kept as a token sequence** ``feats[t-L+1 : t+1]`` -> ``[L, d]``.
This is Method A's lagged design with the lag axis *kept* (not flattened), so the existing
``LatentAttentionTrunk`` ingests it directly via ``TokenAdapter``'s 3-D path.

Pure builders (``build_windowed_design`` / ``sampled_windowed_design``) are unit-tested and
shared in spirit with ``evaluate_features_mtrf.build_lagged_design`` (token ``L-1-lag`` holds
the feature at lag ``lag``, so the two stay consistent). ``load_parcel_eeg`` reads the
per-subject (or ``group``) parcel EEG from the neural HDF5; ``WindowedTemporalDataset`` ties
feature windows to that EEG by stimulus id.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mbs.evaluation.evaluate_features_mtrf import (
    sample_time_indices,
    highpass_along_time,
)

# A parcel = (name, member channel names, noise-ceiling r). Matches insilico_mmn.py.
Parcel = Tuple[str, Sequence[str], float]

# Coarse 10-20 parcels — MUST stay in sync with scripts/insilico_mmn.py:CLUSTERS and
# src/mbs/data_prep/format_eeg_hdf5.py cluster definitions.
CLUSTERS = {
    "frontal":   ["Fz", "F3", "F4", "FCz"],
    "central":   ["Cz", "C3", "C4"],
    "temporal":  ["T7", "T8"],
    "parietal":  ["Pz", "P3", "P4", "P7", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}


def channel_r(neural_h5_path: Path, ch: str, nc_subject: str = "group") -> float:
    """Reliability (correlation scale) of one channel: sqrt(mean stored var% / 100)."""
    try:
        with h5py.File(Path(neural_h5_path), "r") as h:
            v = float(np.nanmean(h["noise_ceilings"][nc_subject][ch][:]))
        return float(np.sqrt(max(v, 0.0) / 100.0))
    except Exception:
        return float("nan")


def build_parcels(neural_h5_path: Path, threshold: float,
                  nc_subject: str = "group") -> List[Parcel]:
    """Ordered [(parcel, members_kept, parcel_r)] for parcels with >=1 channel above the NC
    floor. Parcels are defined once from the group NC and applied to every subject, so the
    individual and group probes target the SAME 4 parcels (apples-to-apples)."""
    out: List[Parcel] = []
    for name, members in CLUSTERS.items():
        rs = {c: channel_r(neural_h5_path, c, nc_subject) for c in members}
        kept = [c for c in members if rs[c] > threshold]
        if not kept:
            print(f"  parcel '{name}': no channel passes r>{threshold} -> DROPPED")
            continue
        pr = float(np.mean([rs[c] for c in kept]))
        out.append((name, kept, pr))
        print(f"  parcel '{name}': keep {kept} (r={pr:.2f}); "
              f"drop {[c for c in members if c not in kept]}")
    return out


# ---------------------------------------------------------------------------
# Windowed design (the core reframing)
# ---------------------------------------------------------------------------

def build_windowed_design(feats: np.ndarray, eeg: np.ndarray,
                          lookback: int, time_idx: np.ndarray):
    """Lookback windows kept as a token axis.

    feats [n_stim, T, d], eeg [n_stim, T, P], time_idx [n_t] (each >= lookback-1).
    Returns X [n_stim*n_t, lookback, d], Y [n_stim*n_t, P].

    Window for output time ``t`` is chronological: ``feats[t-lookback+1 .. t]`` (look-back
    only; the last token is the current bin ``t``). Token index ``lookback-1-lag`` holds the
    feature at lag ``lag`` -> consistent with mTRF's ``build_lagged_design``.
    """
    L = int(lookback)
    time_idx = np.asarray(time_idx)
    assert L >= 1, "lookback must be >= 1"
    assert time_idx.min() >= L - 1, "time_idx must be >= lookback-1 (no pre-history leakage)"

    offsets = np.arange(L) - (L - 1)              # [-(L-1) .. 0], oldest -> current
    idx = time_idx[:, None] + offsets[None, :]    # [n_t, L]
    Xg = feats[:, idx, :]                          # [n_stim, n_t, L, d]
    n_stim, n_t, _, d = Xg.shape
    X = Xg.reshape(n_stim * n_t, L, d)
    Y = eeg[:, time_idx, :].reshape(n_stim * n_t, eeg.shape[2])
    return X.astype(np.float32), Y.astype(np.float32)


def sampled_windowed_design(feats: np.ndarray, eeg: np.ndarray, lookback: int,
                            n_time_samples: int, rng: np.random.Generator):
    """``build_windowed_design`` on randomly sampled output times per stimulus.

    Random sampling decorrelates the strong temporal autocorrelation of EEG (Kadir's trick);
    reuses ``evaluate_features_mtrf.sample_time_indices`` with ``max_lag = lookback-1``.
    """
    T = feats.shape[1]
    t_idx = sample_time_indices(T, int(lookback) - 1, n_time_samples, rng)
    return build_windowed_design(feats, eeg, lookback, t_idx)


# ---------------------------------------------------------------------------
# Per-subject parcel EEG loading
# ---------------------------------------------------------------------------

def _decode_ids(ids) -> List[str]:
    return [s.decode("utf-8") if hasattr(s, "decode") else str(s) for s in ids]


def load_parcel_eeg(neural_h5_path: Path, subject: str, parcels: Sequence[Parcel],
                    split: str) -> Tuple[List[str], np.ndarray]:
    """Raw parcel EEG for one (subject, split): mean over each parcel's member channels.

    Returns (stimulus_ids [n], eeg [n, T, n_parcel]). EEG is left raw here (no normalisation);
    high-pass / standardisation happen in the Dataset so train statistics stay leak-free.
    """
    with h5py.File(Path(neural_h5_path), "r") as f:
        ids = _decode_ids(f[split]["stimulus_ids"][:])
        nd = f[split]["neural_data"][subject]
        cols = []
        for _name, members, _nc in parcels:
            stack = [nd[ch][:, :, 0] for ch in members]      # each [n, T]
            cols.append(np.mean(stack, axis=0)[:, :, None])  # [n, T, 1]
        eeg = np.concatenate(cols, axis=2).astype(np.float32)  # [n, T, P]
    return ids, eeg


def parcel_nc_vector(parcels: Sequence[Parcel]) -> np.ndarray:
    """The stored per-parcel noise-ceiling r (third tuple field), as a vector."""
    return np.array([p[2] for p in parcels], dtype=np.float32)


def recompute_parcel_nc(parcels: Sequence[Parcel], neural_h5_path: Path,
                        nc_subject: str = "group") -> List[Parcel]:
    """Same parcels (name + members), with parcel_r recomputed on another dataset.

    Implements decision C of plan §13: D1, D2, D3 use the SAME 4 parcels/members so the
    comparison is apples-to-apples; only the noise ceiling is dataset-specific."""
    out: List[Parcel] = []
    for name, members, _ in parcels:
        rs = [channel_r(neural_h5_path, c, nc_subject) for c in members]
        out.append((name, list(members), float(np.nanmean(rs))))
    return out


def list_test_splits(neural_h5_path: Path) -> List[str]:
    """Held-out splits to score SEPARATELY. D3 carries per-dataset test groups
    (`test_d1`, `test_d2`) which must never be pooled (the dataset-identity leak, plan §13.2);
    a single-dataset file just has `test`."""
    with h5py.File(Path(neural_h5_path), "r") as f:
        keys = set(f.keys())
    if "test_d1" in keys and "test_d2" in keys:
        return ["test_d1", "test_d2"]
    return ["test"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowedTemporalDataset(Dataset):
    """Windowed (feats -> parcel EEG) samples for one (subject, split, layer).

    Holds the full aligned ``feats`` / ``eeg`` and exposes ``resample()`` to redraw the random
    output times each epoch. ``__getitem__`` returns ``{"feats": [L, d], "y": [P]}``.
    """

    def __init__(
        self,
        *,
        neural_h5_path: Path,
        subject: str,
        parcels: Sequence[Parcel],
        split: str,
        layer_feats: np.ndarray,
        feature_id_map: Dict[str, int],
        lookback: int,
        n_time_samples: int,
        seed: int = 42,
        feat_mu: Optional[np.ndarray] = None,
        feat_sd: Optional[np.ndarray] = None,
        highpass_hz: float = 0.0,
        fs: float = 50.0,
    ):
        super().__init__()
        self.subject = subject
        self.split = split
        self.lookback = int(lookback)
        self.n_time_samples = int(n_time_samples)
        self.parcels = list(parcels)
        self._rng = np.random.default_rng(seed)

        ids, eeg = load_parcel_eeg(neural_h5_path, subject, parcels, split)
        raw = [feature_id_map.get(s) for s in ids]
        keep = [i for i, v in enumerate(raw) if v is not None]
        assert keep, f"No stimulus ids matched features for subject={subject}, split={split}."
        fi = [raw[i] for i in keep]

        feats = np.asarray(layer_feats[fi], dtype=np.float32)   # [n, T, d]
        eeg = eeg[keep]                                          # [n, T, P]

        if highpass_hz and highpass_hz > 0:
            feats = highpass_along_time(feats, fs, highpass_hz)
            eeg = highpass_along_time(eeg, fs, highpass_hz)
        if feat_mu is not None and feat_sd is not None:
            feats = (feats - feat_mu) / feat_sd

        self.feats = feats.astype(np.float32)
        self.eeg = eeg.astype(np.float32)
        self.resample()

    def resample(self):
        """Redraw random output times and rebuild (X, Y). Call once per epoch."""
        self.X, self.Y = sampled_windowed_design(
            self.feats, self.eeg, self.lookback, self.n_time_samples, self._rng
        )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return {
            "feats": torch.from_numpy(self.X[i]).float(),   # [L, d]
            "y": torch.from_numpy(self.Y[i]).float(),       # [P]
        }
