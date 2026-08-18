"""Format Weissbart et al. Cortical Surprisal EEG dataset for the mbs temporal encoding pipeline.

Dataset: Zenodo 7775260 — Weissbart et al. (2023) Cortical Surprisal
  13 subjects (P00–P12), 63-channel EEG at 1000 Hz, pre-aligned to audio.
  15 story parts across 3 audiobooks (AUNP01–08, BROP01–03, FLOP01–04).
  Audio: audiobooks/{story_part}.wav, 16kHz mono, 109–197s per segment.

EEG is stored per-subject as HDF5 (data/{story_part}: [n_ch, T_samples]) already
time-aligned with the corresponding wav file — no onset parsing needed.

Stimulus IDs match AudioSegmentDataset convention: {story_part}_{start_16khz:07d}
  e.g. AUNP01_0000000, AUNP01_0160000

Noise ceiling:
  Same split-half Spearman-Brown method as format_eeg_hdf5.py, computed on training stimuli.

Usage:
    python -m mbs.data_prep.format_eeg_hdf5_surprisal \\
        --data_root  cortical_suprisal_dataset \\
        --output_path outputs/neural_data/surprisal_30s.h5 \\
        --window_duration 30.0 \\
        --window_stride   10.0 \\
        --target_sr       50 \\
        --n_test_parts    3 \\
        --seed            42
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from mbs.core import str2bool
from mbs.data_prep.format_eeg_hdf5 import (
    _get_segment_starts,
    _noise_ceiling_from_halves,
    _pearsonr_nd,
    _wav_duration,
)

log = logging.getLogger(__name__)

_AUDIO_SR = 16_000
_STORY_PARTS = [
    "AUNP01", "AUNP02", "AUNP03", "AUNP04",
    "AUNP05", "AUNP06", "AUNP07", "AUNP08",
    "BROP01", "BROP02", "BROP03",
    "FLOP01", "FLOP02", "FLOP03", "FLOP04",
]


# ─────────────────────────────────────────────────────────────────────────────
# Stimulus ID
# ─────────────────────────────────────────────────────────────────────────────

def _stim_id_surprisal(story_part: str, start_model: int, target_sr: int) -> str:
    """Stimulus ID matching AudioSegmentDataset: {story_part}_{start_16khz:07d}."""
    start_16khz = round(start_model / target_sr * _AUDIO_SR)
    return f"{story_part}_{start_16khz:07d}"


# ─────────────────────────────────────────────────────────────────────────────
# ROI discovery via standard_1020 montage
# ─────────────────────────────────────────────────────────────────────────────

def _find_roi_indices_standard1020(channel_names: List[str]) -> Dict[str, List[int]]:
    """Map channel names to ROI indices using the standard_1020 montage.

    Surprisal dataset channels already have standard 10-20 names (AF3, Fz, TP10…),
    so positions are looked up directly in standard_1020 (not BioSemi128).
    Returns {roi_name: [ch_idx, ...]}.
    """
    n_ch = len(channel_names)
    try:
        import mne
        montage_std = mne.channels.make_standard_montage("standard_1020")
        pos_dict = montage_std.get_positions()["ch_pos"]

        ch_xyz = np.array([pos_dict.get(n, [np.nan, np.nan, np.nan])
                           for n in channel_names])

        target_singles = [
            "Fp1", "Fpz", "Fp2",
            "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
            "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
            "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
            "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
            "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
            "PO9", "PO7", "PO3", "PO1", "POz", "PO2", "PO4", "PO8", "PO10",
            "O9", "O1", "Oz", "O2", "O10",
            "AFz", "AF7", "AF3", "AF4", "AF8",
            "Nz", "Iz",
        ]
        MAX_DIST_MM = 25.0

        single_idx: Dict[str, int] = {}
        claimed: Dict[int, str] = {}
        for name in target_singles:
            if name not in pos_dict:
                continue
            target_xyz = pos_dict[name]
            dists = np.linalg.norm(ch_xyz - target_xyz, axis=1)
            valid = ~np.isnan(dists)
            if not valid.any():
                continue
            best_idx = int(np.where(valid, dists, np.inf).argmin())
            best_dist_mm = float(dists[best_idx]) * 1000
            if best_dist_mm > MAX_DIST_MM:
                continue
            if best_idx in claimed:
                prev_name = claimed[best_idx]
                prev_dist = float(np.linalg.norm(ch_xyz[best_idx] - pos_dict[prev_name])) * 1000
                if best_dist_mm < prev_dist:
                    del single_idx[prev_name]
                    claimed[best_idx] = name
                    single_idx[name] = best_idx
            else:
                single_idx[name] = best_idx
                claimed[best_idx] = name

        rois: Dict[str, List[int]] = {n: [i] for n, i in single_idx.items()}
        for cname, members in [
            ("frontal_cluster",   ["Fz", "F3", "F4", "FCz"]),
            ("central_cluster",   ["Cz", "C3", "C4"]),
            ("temporal_cluster",  ["T7", "T8"]),
            ("parietal_cluster",  ["Pz", "P3", "P4", "P7", "P8"]),
            ("occipital_cluster", ["O1", "Oz", "O2"]),
        ]:
            idxs = [single_idx[m] for m in members if m in single_idx]
            if idxs:
                rois[cname] = idxs

        rois["whole_brain"] = list(range(n_ch))
        return rois

    except Exception as exc:
        log.warning("ROI discovery failed (%s); using whole_brain only.", exc)
        return {"whole_brain": list(range(n_ch))}


# ─────────────────────────────────────────────────────────────────────────────
# EEG loading and resampling
# ─────────────────────────────────────────────────────────────────────────────

def _load_segment_eeg(h5_path: Path, story_part: str,
                      window_size: int, target_sr: int,
                      native_sr: int = 1000) -> Optional[np.ndarray]:
    """Load and resample one subject × story_part segment.

    Returns [T_target, n_ch] float32, or None if the part is missing.
    """
    from scipy.signal import resample_poly

    try:
        with h5py.File(h5_path, "r") as f:
            if story_part not in f["data"]:
                log.warning("Missing story part %s in %s", story_part, h5_path.name)
                return None
            eeg = f["data"][story_part][:]  # [n_ch, T_native]
    except Exception as exc:
        log.warning("Failed to read %s / %s: %s", h5_path.name, story_part, exc)
        return None

    # Resample native_sr → target_sr with anti-aliased polyphase FIR
    from math import gcd
    g = gcd(target_sr, native_sr)
    eeg_resampled = resample_poly(eeg, up=target_sr // g, down=native_sr // g, axis=1)
    eeg_resampled = eeg_resampled.astype(np.float32).T  # [T_target, n_ch]

    # Guard: must be long enough for at least one window
    if eeg_resampled.shape[0] < window_size:
        log.warning("Segment %s too short after resample (%d < %d)",
                    story_part, eeg_resampled.shape[0], window_size)
        return None
    return eeg_resampled


def _get_channel_names(h5_paths: List[Path]) -> Optional[List[str]]:
    """Return channel name list from the first readable HDF5 file."""
    for p in h5_paths:
        try:
            with h5py.File(p, "r") as f:
                return [n.decode() if isinstance(n, bytes) else n
                        for n in f["ch_names"][:]]
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(it, **kw):  # type: ignore[misc]
            return it

    data_root  = Path(args.data_root)
    audio_root = Path(args.audio_root) if args.audio_root else data_root / "audiobooks"
    output_path = Path(args.output_path)

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists: {output_path}. Pass --overwrite true to regenerate.")
        return

    window_size  = int(args.window_duration * args.target_sr)
    stride_size  = int(args.window_stride   * args.target_sr)
    time_step_ms = 1000.0 / args.target_sr

    # ── Subjects ────────────────────────────────────────────────────────────
    h5_paths = sorted(data_root.glob("P*.h5"))
    if not h5_paths:
        raise RuntimeError(f"No P*.h5 files found under {data_root}")

    # ── Channel names and ROIs ───────────────────────────────────────────────
    ch_names = _get_channel_names(h5_paths)
    if ch_names is None:
        raise RuntimeError("Could not read channel names from any P*.h5 file.")
    n_ch = len(ch_names)
    roi_indices = _find_roi_indices_standard1020(ch_names)
    roi_names   = sorted(roi_indices.keys())
    print(f"Subjects: {len(h5_paths)} | Channels: {n_ch} | ROIs: {len(roi_names)}")

    # ── Story parts and audio durations ─────────────────────────────────────
    story_parts = _STORY_PARTS
    wav_durations: Dict[str, float] = {}
    for part in story_parts:
        wav = audio_root / f"{part}.wav"
        if not wav.exists():
            log.warning("Missing audio: %s", wav)
            continue
        wav_durations[part] = _wav_duration(wav)

    available_parts = [p for p in story_parts if p in wav_durations]
    if not available_parts:
        raise RuntimeError(f"No audio files found under {audio_root}")

    # ── Train / test split by story part ────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_test = min(args.n_test_parts, len(available_parts) - 1)
    test_part_set = set(rng.choice(available_parts, size=n_test, replace=False).tolist())
    train_parts = [p for p in available_parts if p not in test_part_set]
    test_parts  = [p for p in available_parts if p in test_part_set]

    print(f"Story parts: {len(available_parts)} available | "
          f"train: {len(train_parts)} | test: {len(test_parts)}")
    print(f"Test parts: {sorted(test_part_set)}")
    print(f"Window: {args.window_duration}s / stride: {args.window_stride}s | "
          f"target_sr: {args.target_sr} Hz")

    # ── Segment start positions ──────────────────────────────────────────────
    part_starts: Dict[str, List[int]] = {
        p: _get_segment_starts(wav_durations[p], window_size, stride_size, args.target_sr)
        for p in available_parts
    }
    total_train = sum(len(part_starts[p]) for p in train_parts)
    total_test  = sum(len(part_starts[p]) for p in test_parts)
    print(f"Windows: train={total_train}, test={total_test}")

    # ── Assign subjects to halves for noise ceiling ──────────────────────────
    subj_perm = rng.permutation(len(h5_paths))
    half_n = len(h5_paths) // 2
    half_paths = [
        [h5_paths[i] for i in subj_perm[:half_n]],
        [h5_paths[i] for i in subj_perm[half_n:]],
    ]

    # ── Accumulate per-half sums ─────────────────────────────────────────────
    _zeros = np.zeros((window_size, n_ch), np.float64)
    half_sums   = [defaultdict(lambda: np.zeros((window_size, n_ch), np.float64)),
                   defaultdict(lambda: np.zeros((window_size, n_ch), np.float64))]
    half_counts = [0, 0]

    for h, hpaths in enumerate(half_paths):
        for p in tqdm(hpaths, desc=f"Half {h+1}", unit="subj"):
            any_loaded = False
            for part in tqdm(available_parts, desc=p.name, leave=False, unit="part"):
                eeg = _load_segment_eeg(p, part, window_size, args.target_sr)
                if eeg is None:
                    continue
                for s in part_starts[part]:
                    seg = eeg[s: s + window_size]
                    if seg.shape[0] < window_size:
                        continue
                    half_sums[h][(part, s)] += seg.astype(np.float64)
                any_loaded = True
            if any_loaded:
                half_counts[h] += 1

    all_keys = set(half_sums[0].keys()) | set(half_sums[1].keys())
    group_sums: Dict = {}
    for key in all_keys:
        group_sums[key] = (half_sums[0].get(key, _zeros)
                           + half_sums[1].get(key, _zeros))
    group_count = half_counts[0] + half_counts[1]

    print(f"Subjects loaded: half1={half_counts[0]}, half2={half_counts[1]}, "
          f"total={group_count}")

    # ── Build ordered stimulus lists ─────────────────────────────────────────
    def _ordered_keys(parts):
        return [(p, s) for p in parts for s in part_starts[p]
                if (p, s) in all_keys]

    train_keys = _ordered_keys(train_parts)
    test_keys  = _ordered_keys(test_parts)

    train_ids = [_stim_id_surprisal(p, s, args.target_sr) for p, s in train_keys]
    test_ids  = [_stim_id_surprisal(p, s, args.target_sr) for p, s in test_keys]
    print(f"Stimuli: train={len(train_ids)}, test={len(test_ids)}")

    # ── Group averages and per-ROI arrays ────────────────────────────────────
    def _avg(key):
        return (group_sums[key] / group_count).astype(np.float32)

    def _build_roi_data(keys, desc):
        out: Dict[str, np.ndarray] = {}
        for roi, ch_idxs in tqdm(roi_indices.items(), desc=desc, unit="roi"):
            stack = np.stack([_avg(k)[:, ch_idxs] for k in keys], axis=0)
            out[roi] = stack.astype(np.float32)
        return out

    train_roi = _build_roi_data(train_keys, desc="Assembling train ROIs")
    test_roi  = _build_roi_data(test_keys,  desc="Assembling test ROIs")

    # ── Noise ceiling ────────────────────────────────────────────────────────
    def _half_avg_roi(h, keys, ch_idxs):
        cnt = half_counts[h]
        if cnt == 0:
            return np.zeros((len(keys), window_size, len(ch_idxs)), np.float32)
        return np.stack(
            [(half_sums[h][k] / cnt).astype(np.float32)[:, ch_idxs] for k in keys],
            axis=0,
        )

    nc_dict: Dict[str, np.ndarray] = {}
    for roi, ch_idxs in tqdm(roi_indices.items(), desc="Computing noise ceilings", unit="roi"):
        h1 = _half_avg_roi(0, train_keys, ch_idxs)
        h2 = _half_avg_roi(1, train_keys, ch_idxs)
        nc_dict[roi] = _noise_ceiling_from_halves(h1, h2)

    # ── Write HDF5 ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.attrs["subjects"]          = ["group"]
        f.attrs["rois"]              = roi_names
        f.attrs["splits"]            = ["train", "test"]
        f.attrs["max_nc"]            = 100.0
        f.attrs["temporal"]          = True
        f.attrs["T_model"]           = window_size
        f.attrs["time_step_ms"]      = time_step_ms
        f.attrs["window_duration_s"] = args.window_duration
        f.attrs["window_stride_s"]   = args.window_stride
        f.attrs["target_sr"]         = args.target_sr
        f.attrs["n_subjects"]        = group_count
        f.attrs["dataset"]           = "surprisal"
        f.attrs["test_parts"]        = sorted(test_part_set)
        f.attrs["channel_names"]     = ch_names

        for split_name, ids_list, roi_data in [
            ("train", train_ids, train_roi),
            ("test",  test_ids,  test_roi),
        ]:
            grp = f.create_group(split_name)
            grp.create_dataset("stimulus_ids",
                               data=np.array(ids_list, dtype="S"))
            nd_grp = grp.create_group("neural_data/group")
            for roi in roi_names:
                nd_grp.create_dataset(roi, data=roi_data[roi],
                                      compression="gzip", compression_opts=4)

        nc_grp = f.create_group("noise_ceilings/group")
        for roi in roi_names:
            nc_grp.create_dataset(roi, data=nc_dict[roi],
                                  compression="gzip")

    print(f"Written: {output_path}")
    for roi in roi_names:
        n_roi_ch = len(roi_indices[roi])
        print(f"  ROI '{roi}': {n_roi_ch} channel(s), "
              f"NC mean={float(np.nanmean(nc_dict[roi])):.1f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Format Weissbart Cortical Surprisal EEG into mbs temporal HDF5."
    )
    p.add_argument("--data_root",       required=True,
                   help="Path to cortical_suprisal_dataset/ (contains P00.h5 etc.).")
    p.add_argument("--audio_root",      default=None,
                   help="Dir with AUNP01.wav etc. (default: {data_root}/audiobooks).")
    p.add_argument("--output_path",     required=True,
                   help="Path for output HDF5 file.")
    p.add_argument("--window_duration", type=float, default=30.0,
                   help="Window length in seconds.")
    p.add_argument("--window_stride",   type=float, default=10.0,
                   help="Stride between windows in seconds.")
    p.add_argument("--target_sr",       type=int,   default=50,
                   help="EEG target sampling rate in Hz.")
    p.add_argument("--n_test_parts",    type=int,   default=3,
                   help="Number of story parts held out as test set.")
    p.add_argument("--overwrite",       type=str2bool, default=False)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
