"""Shared EEG target builders for the model->EEG mapping (dataset-agnostic; no 'broderick').

A "target" is a column the mTRF predicts. Two granularities, one interface — each target is
(name, member_channels, reliability_r):
  * parcels    : the 5 coarse 10-20 clusters, raw average over NC-surviving members.
  * electrodes : every real electrode passing the NC floor (each its own single-member target).

Reused by eeg_mapping_sweep.py and the in-silico MMN drivers so parcels/electrodes are defined
identically everywhere. The mapping is fit on whatever EEG dataset is passed (default D2 / Cortical
Surprisal); nothing here is specific to a dataset.
"""

from pathlib import Path
import numpy as np
import h5py

from mbs.evaluation.utils.evaluation_helpers import load_neural_data

FS = 50.0          # EEG / feature grid (Hz)
TIME_STEP_MS = 20.0

# Coarse 10-20 parcels (must match src/mbs/data_prep/format_eeg_hdf5.py cluster definitions).
CLUSTERS = {
    "frontal":   ["Fz", "F3", "F4", "FCz"],
    "central":   ["Cz", "C3", "C4"],
    "temporal":  ["T7", "T8"],
    "parietal":  ["Pz", "P3", "P4", "P7", "P8"],
    "occipital": ["O1", "Oz", "O2"],
}
NON_ELECTRODE = ("_cluster", "whole_brain")     # pseudo-channels in the neural file

# Approximate 10-20 scalp positions (nose up, x right) for the electrode topo plots.
PREFIX_Y = {"FP": 0.95, "AF": 0.78, "F": 0.55, "FC": 0.32, "FT": 0.32, "C": 0.0, "T": 0.0,
            "CP": -0.22, "TP": -0.22, "P": -0.5, "PO": -0.72, "O": -0.92, "I": -1.0}


def decode(xs):
    return [x.decode() if hasattr(x, "decode") else x for x in xs]


def channel_r(neural_h5, ch):
    """Reliability (correlation scale) of one channel: sqrt(mean stored var% / 100)."""
    try:
        with h5py.File(neural_h5, "r") as h:
            v = float(np.nanmean(h["noise_ceilings"]["group"][ch][:]))
        return float(np.sqrt(max(v, 0.0) / 100.0))
    except Exception:
        return float("nan")


def montage_pos(ch):
    """Approximate 10-20 (x, y) for a channel name. None if the prefix is unknown."""
    s = ch
    if s[-1] in "zZ":
        col, s = "z", s[:-1]
    else:
        i = len(s)
        while i > 0 and s[i - 1].isdigit():
            i -= 1
        col, s = s[i:], s[:i]
    y = PREFIX_Y.get(s.upper())
    if y is None:
        return None
    if col == "z":
        return 0.0, y
    n = int(col)
    mag = {1: 0.2, 2: 0.2, 3: 0.4, 4: 0.4, 5: 0.6, 6: 0.6, 7: 0.8, 8: 0.8, 9: 1.0, 10: 1.0}.get(n, 0.9)
    return (-mag if n % 2 else mag), y


def build_parcels(neural_h5, threshold):
    """Ordered list of (parcel, members_kept, parcel_r) for parcels with >=1 NC survivor."""
    out = []
    for name, members in CLUSTERS.items():
        rs = {c: channel_r(neural_h5, c) for c in members}
        kept = [c for c in members if rs[c] > threshold]
        if not kept:
            print(f"  parcel '{name}': no channel passes r>{threshold} -> DROPPED")
            continue
        pr = float(np.mean([rs[c] for c in kept]))
        out.append((name, kept, pr))
        print(f"  parcel '{name}': keep {kept} (r={pr:.2f})")
    return out


def build_electrodes(neural_h5, threshold):
    """Ordered list of (channel, [channel], r) for real electrodes passing NC r>threshold."""
    with h5py.File(neural_h5, "r") as h:
        chans = list(h["noise_ceilings"]["group"].keys())
    out = []
    for ch in chans:
        if any(t in ch for t in NON_ELECTRODE) or montage_pos(ch) is None:
            continue
        r = channel_r(neural_h5, ch)
        if r > threshold:
            out.append((ch, [ch], float(r)))
    out.sort(key=lambda e: -e[2])
    print(f"  electrodes passing NC r>{threshold}: {len(out)}")
    return out


def build_targets(neural_h5, level, threshold):
    """Dispatch on target level. Returns the (name, members, r) list."""
    targets = build_parcels(neural_h5, threshold) if level == "parcels" \
        else build_electrodes(neural_h5, threshold)
    assert targets, f"no {level} survived the NC threshold"
    return targets


def load_split_targets(neural_h5, feats_all, id_map, targets, split):
    """(target EEG [n, T, n_target], aligned raw features [n, T, d]) for one dataset split.

    Restricted to stimuli whose ids exist in id_map. Each target column is the raw average over
    its member channels (a single channel for electrodes). Identical id alignment per channel.
    """
    fi = keep = None
    cols = []
    for _, members, _ in targets:
        stack = []
        for ch in members:
            ids, eeg_ch, _ = load_neural_data(Path(neural_h5), "group", ch, split)
            if fi is None:
                ids = decode(ids)
                raw = [id_map.get(s) for s in ids]
                keep = [i for i, v in enumerate(raw) if v is not None]
                fi = [v for v in raw if v is not None]
            stack.append(eeg_ch[keep][:, :, 0])             # [n, T]
        cols.append(np.mean(stack, axis=0)[:, :, None])     # raw avg -> [n, T, 1]
    eeg = np.concatenate(cols, axis=2).astype(np.float32)   # [n, T, n_target]
    return eeg, feats_all[fi]
