"""Format Broderick 2018 naturalistic EEG dataset for the mbs temporal encoding pipeline.

Dataset: OpenNeuro ds004408 — Broderick (2018) / Di Liberto (2015)
  19 subjects, 20 runs each, 128-channel BioSemi EEG at 512 Hz.
  Audio: stimuli/audioXX.wav, stereo 44100 Hz, ~3 min each.
  EEG starts are aligned with audio onset; recordings extend ~15-25 s past audio end
  (confirmed by dataset README: "starts are aligned, EEG longer to a varying extent").

Stimulus segmentation:
  Each ~3-min audio/EEG pair is sub-segmented into overlapping windows (default 30 s / 10 s stride).
  Stimulus IDs match AudioSegmentDataset convention:  audioXX_SSSSSSS
  where SSSSSSS is the start sample at 16 kHz (the audio model's native sample rate).

Noise ceiling:
  Subjects are randomly split in half; per-half group averages are correlated across stimuli
  per (time-bin, channel), then Spearman-Brown corrected to the full-group estimate.
  Stored as % variance explained (r_SB² × 100) with max_nc=100.0 so that
  load_neural_data() recovers Pearson r via  sqrt(nc_stored / 100).

Usage:
    python -m mbs.data_prep.format_eeg_hdf5 \\
        --bids_root  /path/to/Broderick_2018_EEG_The_old_man_and_the_sea \\
        --output_path outputs/neural_data/broderick2018_30s.h5 \\
        --window_duration 30.0 \\
        --window_stride   10.0 \\
        --target_sr       50 \\
        --n_test_runs     4 \\
        --seed            42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.io import wavfile

from mbs.core import str2bool

log = logging.getLogger(__name__)

# AudioSegmentDataset resamples to this rate; stimulus IDs encode positions at this rate.
_AUDIO_SR = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# WAV utilities
# ─────────────────────────────────────────────────────────────────────────────

def _wav_duration(wav_path: Path) -> float:
    sr, data = wavfile.read(str(wav_path))
    return len(data) / sr


def _get_segment_starts(audio_dur: float, window_size: int, stride_size: int,
                        target_sr: int) -> List[int]:
    """Return list of window start indices (in model-time samples) that fit within audio_dur."""
    T = int(audio_dur * target_sr)
    starts = []
    s = 0
    while s + window_size <= T:
        starts.append(s)
        s += stride_size
    return starts


def _stim_id(run: int, start_model: int, target_sr: int) -> str:
    """Stimulus ID matching AudioSegmentDataset: audioXX_SSSSSSS (at 16 kHz)."""
    start_audio = round(start_model / target_sr * _AUDIO_SR)
    return f"audio{run:02d}_{start_audio:07d}"


# ─────────────────────────────────────────────────────────────────────────────
# EEG loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_run_eeg(subject: str, run: int, bids_root: Path,
                  audio_dur: float, target_sr: int) -> Optional[np.ndarray]:
    """Load, crop, and resample one subject × run.

    Returns [T_model, n_ch] float32, or None if the file is missing.
    """
    import mne  # lazy import so the module loads without mne installed

    vhdr = (bids_root / subject / "eeg"
            / f"{subject}_task-listening_run-{run:02d}_eeg.vhdr")
    if not vhdr.exists():
        log.warning("Missing: %s", vhdr)
        return None

    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
    # EEG is always longer; crop to audio duration (starts are aligned per README)
    tmax = min(audio_dur - 1.0 / raw.info["sfreq"], raw.times[-1])
    raw.crop(tmin=0.0, tmax=tmax, verbose=False)
    raw.resample(target_sr, verbose=False)
    return raw.get_data().astype(np.float32).T  # [T_model, n_ch]


def _get_channel_names(bids_root: Path, subjects: List[str],
                       run_ids: List[int]) -> Optional[List[str]]:
    """Return channel name list from the first readable file."""
    import mne
    for subj in subjects:
        for run in run_ids:
            vhdr = (bids_root / subj / "eeg"
                    / f"{subj}_task-listening_run-{run:02d}_eeg.vhdr")
            if vhdr.exists():
                raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
                return raw.ch_names
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ROI discovery via BioSemi128 montage
# ─────────────────────────────────────────────────────────────────────────────

def _find_roi_indices(channel_names: List[str]) -> Dict[str, List[int]]:
    """Map standard 10-20 labels to the nearest BioSemi128 channel indices.

    Returns {roi_name: [ch_idx, ...]}. Falls back to {"whole_brain": [0..n_ch-1]}
    if MNE montage lookup fails.
    """
    n_ch = len(channel_names)
    try:
        import mne
        montage_bsl = mne.channels.make_standard_montage("biosemi128")
        montage_std = mne.channels.make_standard_montage("standard_1020")

        bsl_pos_dict = montage_bsl.get_positions()["ch_pos"]
        std_pos_dict = montage_std.get_positions()["ch_pos"]

        # Build position array aligned to channel_names
        ch_xyz = np.array([bsl_pos_dict.get(n, [np.nan, np.nan, np.nan])
                           for n in channel_names])

        # Full extended 10-20 set.  Any name not in standard_1020 or whose
        # nearest BioSemi128 channel is > MAX_DIST_MM away is silently skipped.
        target_singles = [
            # Standard 10-20
            "Fp1", "Fpz", "Fp2",
            "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
            "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
            "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
            "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
            "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
            "PO9", "PO7", "PO3", "PO1", "POz", "PO2", "PO4", "PO8", "PO10",
            "O9", "O1", "Oz", "O2", "O10",
            # Additional anterior / frontal
            "AFz", "AF7", "AF3", "AF4", "AF8",
            "Nz", "Iz",
        ]
        MAX_DIST_MM = 25.0

        single_idx: Dict[str, int] = {}
        # Track which BioSemi channel is already claimed to detect collisions.
        claimed: Dict[int, str] = {}
        for name in target_singles:
            if name not in std_pos_dict:
                continue
            target_xyz = std_pos_dict[name]
            dists = np.linalg.norm(ch_xyz - target_xyz, axis=1)
            valid = ~np.isnan(dists)
            if not valid.any():
                continue
            best_idx = int(np.where(valid, dists, np.inf).argmin())
            best_dist_mm = float(dists[best_idx]) * 1000
            if best_dist_mm > MAX_DIST_MM:
                continue
            if best_idx in claimed:
                # Two standards want the same BioSemi channel — keep the closer one.
                prev_name = claimed[best_idx]
                prev_dist = float(np.linalg.norm(ch_xyz[best_idx] - std_pos_dict[prev_name])) * 1000
                if best_dist_mm < prev_dist:
                    del single_idx[prev_name]
                    claimed[best_idx] = name
                    single_idx[name] = best_idx
                # else keep previous, skip current
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
# Noise ceiling
# ─────────────────────────────────────────────────────────────────────────────

def _pearsonr_nd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised Pearson r along axis 0 for arrays [..., *trailing].
    Returns r of same shape as trailing dimensions.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = (x * y).sum(axis=0)
    denom = np.sqrt((x ** 2).sum(axis=0) * (y ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0.0, num / denom, 0.0)
    return r.astype(np.float32)


def _noise_ceiling_from_halves(half1: np.ndarray, half2: np.ndarray) -> np.ndarray:
    """Spearman-Brown corrected split-half NC, stored as % variance explained.

    half1, half2: [n_stim, T, n_ch]  (cross-subject averages for each half)
    Returns [T, n_ch] as  r_SB² × 100  (% variance explained, 0-100).
    This is the scale expected by load_neural_data() when max_nc=100.
    """
    r = _pearsonr_nd(half1, half2)           # [T, n_ch]
    r = np.clip(r, 0.0, 1.0)
    r_sb = (2.0 * r) / (1.0 + r + 1e-9)     # Spearman-Brown correction
    return (r_sb ** 2 * 100.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    import mne
    mne.set_log_level("WARNING")

    bids_root = Path(args.bids_root)
    output_path = Path(args.output_path)

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists: {output_path}. Pass --overwrite true to regenerate.")
        return

    window_size = int(args.window_duration * args.target_sr)
    stride_size = int(args.window_stride * args.target_sr)
    time_step_ms = 1000.0 / args.target_sr  # e.g. 20.0 ms for target_sr=50

    # ── Subjects and runs ───────────────────────────────────────────────────
    subjects = sorted(
        d.name for d in bids_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    stimuli_dir = bids_root / "stimuli"
    wav_durations: Dict[int, float] = {}
    for wav in sorted(stimuli_dir.glob("audio*.wav")):
        idx = int(wav.stem.replace("audio", ""))
        wav_durations[idx] = _wav_duration(wav)
    run_ids = sorted(wav_durations.keys())

    rng = np.random.default_rng(args.seed)
    test_run_set = set(int(r) for r in rng.choice(run_ids, size=args.n_test_runs, replace=False))
    train_runs = [r for r in run_ids if r not in test_run_set]
    test_runs  = [r for r in run_ids if r in test_run_set]

    print(f"Subjects: {len(subjects)} | runs: {len(run_ids)} | "
          f"train: {len(train_runs)} | test: {len(test_runs)}")
    print(f"Window: {args.window_duration}s / stride: {args.window_stride}s | "
          f"target_sr: {args.target_sr} Hz")
    print(f"Test runs: {sorted(test_run_set)}")

    # ── Channel names and ROIs ───────────────────────────────────────────────
    ch_names = _get_channel_names(bids_root, subjects, run_ids)
    if ch_names is None:
        raise RuntimeError("No EEG files found under bids_root.")
    n_ch = len(ch_names)
    roi_indices = _find_roi_indices(ch_names)  # {roi_name: [ch_idx, ...]}
    roi_names = sorted(roi_indices.keys())
    print(f"Channels: {n_ch} | ROIs: {roi_names}")

    # ── Pre-compute segment start positions per run ──────────────────────────
    run_starts: Dict[int, List[int]] = {
        run: _get_segment_starts(wav_durations[run], window_size, stride_size, args.target_sr)
        for run in run_ids
    }

    # ── Assign subjects to split halves for noise ceiling ───────────────────
    subj_perm = rng.permutation(len(subjects))
    half_n = len(subjects) // 2
    half_subjs = [
        [subjects[i] for i in subj_perm[:half_n]],
        [subjects[i] for i in subj_perm[half_n:]],
    ]

    # ── Accumulate per-half sums ─────────────────────────────────────────────
    # half_sums[h][(run, start)] = float64 accumulator [T_win, n_ch]
    # half_counts[h] = int  (number of subjects in this half successfully loaded)
    from collections import defaultdict

    half_sums   = [defaultdict(lambda: np.zeros((window_size, n_ch), np.float64)),
                   defaultdict(lambda: np.zeros((window_size, n_ch), np.float64))]
    half_counts = [0, 0]
    group_sums  = defaultdict(lambda: np.zeros((window_size, n_ch), np.float64))
    group_count = 0

    from tqdm.auto import tqdm

    all_subjects_ordered = half_subjs[0] + half_subjs[1]
    with tqdm(total=len(all_subjects_ordered), desc="Loading EEG subjects", unit="subj") as pbar:
        for h, hsubjs in enumerate(half_subjs):
            for subj in hsubjs:
                any_loaded = False
                for run in tqdm(run_ids, desc=f"{subj}", leave=False, unit="run"):
                    eeg = _load_run_eeg(subj, run, bids_root, wav_durations[run], args.target_sr)
                    if eeg is None:
                        continue
                    starts = run_starts[run]
                    for s in starts:
                        seg = eeg[s : s + window_size]
                        if seg.shape[0] < window_size:
                            continue  # EEG slightly shorter after resample rounding
                        half_sums[h][(run, s)] += seg.astype(np.float64)
                    any_loaded = True
                if any_loaded:
                    half_counts[h] += 1
                pbar.update(1)
                pbar.set_postfix({"half": h + 1, "subj": subj})

    # Rebuild group sums from halves without triggering defaultdict factory for missing keys
    _zeros = np.zeros((window_size, n_ch), np.float64)
    all_keys = set(half_sums[0].keys()) | set(half_sums[1].keys())
    for key in all_keys:
        group_sums[key] = (half_sums[0].get(key, _zeros)
                           + half_sums[1].get(key, _zeros))
    group_count = half_counts[0] + half_counts[1]

    print(f"Subjects loaded: half1={half_counts[0]}, half2={half_counts[1]}, "
          f"total={group_count}")

    # ── Build ordered stimulus lists ─────────────────────────────────────────
    def _ordered_keys(runs):
        return [(run, s) for run in runs for s in run_starts[run]
                if (run, s) in all_keys]

    train_keys = _ordered_keys(train_runs)
    test_keys  = _ordered_keys(test_runs)

    train_ids_list = [_stim_id(run, s, args.target_sr) for run, s in train_keys]
    test_ids_list  = [_stim_id(run, s, args.target_sr) for run, s in test_keys]

    print(f"Stimuli: train={len(train_ids_list)}, test={len(test_ids_list)}")

    # ── Compute group averages and per-ROI arrays ────────────────────────────
    def _avg(sums_dict, key, count):
        return (sums_dict[key] / count).astype(np.float32)

    def _build_roi_data(keys, desc):
        # Returns {roi: [n_stim, T, n_roi_ch]}
        out: Dict[str, np.ndarray] = {}
        for roi, ch_idxs in tqdm(roi_indices.items(), desc=desc, unit="roi"):
            stack = np.stack(
                [_avg(group_sums, k, group_count)[:, ch_idxs] for k in keys],
                axis=0,
            )  # [n_stim, T_win, n_roi_ch]
            out[roi] = stack.astype(np.float32)
        return out

    train_roi = _build_roi_data(train_keys, desc="Assembling train ROIs")
    test_roi  = _build_roi_data(test_keys,  desc="Assembling test ROIs")

    # ── Noise ceiling per ROI (from training stimuli) ────────────────────────
    def _half_avg_roi(h, keys, ch_idxs):
        count = half_counts[h]
        if count == 0:
            return np.zeros((len(keys), window_size, len(ch_idxs)), np.float32)
        return np.stack(
            [(half_sums[h][k] / count).astype(np.float32)[:, ch_idxs] for k in keys],
            axis=0,
        )  # [n_stim, T, n_roi_ch]

    nc_dict: Dict[str, np.ndarray] = {}
    for roi, ch_idxs in tqdm(roi_indices.items(), desc="Computing noise ceilings", unit="roi"):
        h1 = _half_avg_roi(0, train_keys, ch_idxs)
        h2 = _half_avg_roi(1, train_keys, ch_idxs)
        nc_dict[roi] = _noise_ceiling_from_halves(h1, h2)  # [T, n_roi_ch]

    # ── Write HDF5 ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.attrs["subjects"]     = ["group"]
        f.attrs["rois"]         = roi_names
        f.attrs["splits"]       = ["train", "test"]
        f.attrs["max_nc"]       = 100.0
        f.attrs["temporal"]     = True
        f.attrs["T_model"]      = window_size
        f.attrs["time_step_ms"] = time_step_ms
        f.attrs["window_duration_s"] = args.window_duration
        f.attrs["window_stride_s"]   = args.window_stride
        f.attrs["target_sr"]         = args.target_sr
        f.attrs["n_subjects"]        = group_count
        f.attrs["dataset"]           = "broderick2018"
        f.attrs["test_runs"]         = sorted(test_run_set)
        f.attrs["channel_names"]     = ch_names

        for split_name, ids_list, roi_data in [
            ("train", train_ids_list, train_roi),
            ("test",  test_ids_list,  test_roi),
        ]:
            grp = f.create_group(split_name)
            # Use fixed-length byte strings (dtype="S") — consistent with test fixtures
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
        description="Format Broderick 2018 naturalistic EEG into mbs temporal HDF5."
    )
    p.add_argument("--bids_root",       required=True,
                   help="Path to BIDS root (Broderick_2018_EEG_The_old_man_and_the_sea).")
    p.add_argument("--output_path",     required=True,
                   help="Path for output HDF5 file.")
    p.add_argument("--window_duration", type=float, default=30.0,
                   help="Window length in seconds (default 30 for Whisper).")
    p.add_argument("--window_stride",   type=float, default=10.0,
                   help="Stride between windows in seconds.")
    p.add_argument("--target_sr",       type=int,   default=50,
                   help="EEG target sampling rate in Hz (= model time steps/s).")
    p.add_argument("--n_test_runs",     type=int,   default=4,
                   help="Number of runs held out as test set.")
    p.add_argument("--overwrite",       type=str2bool, default=False)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
