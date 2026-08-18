"""Combine two EEG HDF5 files (D1 + D2) into a single D3 training set.

Merges on common ROI names; concatenates train sets; keeps test sets both
separately (test_d1/, test_d2/) and combined (test/) for evaluator compatibility.
Noise ceiling is a stimulus-count-weighted mean of D1 and D2 NCs.

D1 and D2 store neural_data on very different scales/baselines (D1 is
raw-amplitude-like with per-channel offsets in the thousands; D2 is
near-zero/pre-normalized). Each dataset's neural_data is z-scored per
ROI/channel using its own train-split statistics before concatenation, so a
model trained on the combined set can't exploit "which dataset is this
stimulus from" as a shortcut to reproduce each dataset's baseline offset.

ROIs where n_ch differs between D1 and D2 are skipped with a warning; in
practice named ROIs (Fz, T7, …) are all single-channel so skipping is rare.

Usage:
    python -m mbs.data_prep.combine_eeg_hdf5 \\
        --d1_hdf5_path outputs/neural_data/broderick2018_30s.h5 \\
        --d2_hdf5_path outputs/neural_data/surprisal_30s.h5 \\
        --output_path  outputs/neural_data/d3_combined_30s.h5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from mbs.core import str2bool

log = logging.getLogger(__name__)


def _load_split(h5: h5py.File, split: str, roi_names: list[str]):
    """Return (stimulus_ids, {roi: array}) for the given split."""
    ids = h5[f"{split}/stimulus_ids"][:]
    data = {}
    for roi in roi_names:
        key = f"{split}/neural_data/group/{roi}"
        if key in h5:
            data[roi] = h5[key][:]
        else:
            data[roi] = None
    return ids, data


def _standardize(train_arr: np.ndarray, test_arr: np.ndarray):
    """Z-score train/test arrays per channel using train-split statistics.

    D1 and D2 store neural_data on very different scales/baselines (D1 is
    raw-amplitude-like with per-channel offsets in the thousands; D2 is
    near-zero/pre-normalized). Concatenating them as-is lets a model trained
    on D3 trivially predict "which dataset" from the input features and
    reproduce each dataset's baseline offset, inflating the pooled test-set
    Pearson r without any corresponding change in the (within-dataset) noise
    ceiling. This is a per-channel affine transform, so it leaves NC and
    standalone D1/D2 Pearson r unchanged.
    """
    mean = train_arr.mean(axis=(0, 1), keepdims=True)
    std = train_arr.std(axis=(0, 1), keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (train_arr - mean) / std, (test_arr - mean) / std


def main(args: argparse.Namespace) -> None:
    d1_path = Path(args.d1_hdf5_path)
    d2_path = Path(args.d2_hdf5_path)
    out_path = Path(args.output_path)

    if out_path.exists() and not args.overwrite:
        print(f"Output already exists: {out_path}. Pass --overwrite true to regenerate.")
        return

    with h5py.File(d1_path, "r") as f1, h5py.File(d2_path, "r") as f2:
        d1_rois = set(f1.attrs["rois"].tolist())
        d2_rois = set(f2.attrs["rois"].tolist())
        candidate_rois = sorted(d1_rois & d2_rois)

        if not candidate_rois:
            raise RuntimeError(
                f"No common ROIs between D1 ({sorted(d1_rois)}) and D2 ({sorted(d2_rois)})")

        # Filter to ROIs where n_ch matches across both datasets
        common_rois = []
        skipped = []
        for roi in candidate_rois:
            k1 = f"train/neural_data/group/{roi}"
            k2 = f"train/neural_data/group/{roi}"
            if k1 not in f1 or k2 not in f2:
                skipped.append((roi, "missing in one dataset"))
                continue
            nch1 = f1[k1].shape[2] if f1[k1].ndim == 3 else f1[k1].shape[1]
            nch2 = f2[k2].shape[2] if f2[k2].ndim == 3 else f2[k2].shape[1]
            if nch1 != nch2:
                skipped.append((roi, f"n_ch mismatch: D1={nch1} D2={nch2}"))
            else:
                common_rois.append(roi)

        if skipped:
            for roi, reason in skipped:
                log.warning("Skipping ROI '%s': %s", roi, reason)
        print(f"Common ROIs: {len(common_rois)} (skipped {len(skipped)})")

        # Load train splits
        d1_train_ids, d1_train = _load_split(f1, "train", common_rois)
        d2_train_ids, d2_train = _load_split(f2, "train", common_rois)

        n1_train = len(d1_train_ids)
        n2_train = len(d2_train_ids)
        print(f"Train stimuli: D1={n1_train}, D2={n2_train}, combined={n1_train + n2_train}")

        # Load test splits
        d1_test_ids,  d1_test  = _load_split(f1, "test", common_rois)
        d2_test_ids,  d2_test  = _load_split(f2, "test", common_rois)
        print(f"Test stimuli:  D1={len(d1_test_ids)}, D2={len(d2_test_ids)}")

        # Standardize each dataset's neural data per-ROI/channel using its own
        # train-split statistics, so D1's and D2's very different baselines/scales
        # don't create a between-dataset shortcut once concatenated (see
        # _standardize docstring).
        for roi in common_rois:
            d1_train[roi], d1_test[roi] = _standardize(d1_train[roi], d1_test[roi])
            d2_train[roi], d2_test[roi] = _standardize(d2_train[roi], d2_test[roi])

        # Load noise ceilings
        nc_d1, nc_d2 = {}, {}
        for roi in common_rois:
            nc_d1[roi] = f1[f"noise_ceilings/group/{roi}"][:]
            nc_d2[roi] = f2[f"noise_ceilings/group/{roi}"][:]

        # Metadata from D1 (window params must match)
        T_model      = int(f1.attrs["T_model"])
        time_step_ms = float(f1.attrs["time_step_ms"])
        win_dur      = float(f1.attrs.get("window_duration_s", 30.0))
        win_stride   = float(f1.attrs.get("window_stride_s",   10.0))
        target_sr    = int(f1.attrs.get("target_sr", 50))

        # Sanity check window params match
        if f2.attrs.get("T_model") != T_model:
            raise ValueError(
                f"T_model mismatch: D1={T_model}, D2={f2.attrs.get('T_model')}. "
                "Both datasets must use the same window duration and target_sr.")

    # ── Weighted noise ceiling ───────────────────────────────────────────────
    nc_combined = {}
    w1, w2 = n1_train, n2_train
    total_w = w1 + w2
    for roi in common_rois:
        nc_combined[roi] = (
            (nc_d1[roi] * w1 + nc_d2[roi] * w2) / total_w
        ).astype(np.float32)

    # ── Write output HDF5 ────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.attrs["subjects"]          = ["group"]
        f.attrs["rois"]              = common_rois
        f.attrs["splits"]            = ["train", "test"]
        f.attrs["max_nc"]            = 100.0
        f.attrs["temporal"]          = True
        f.attrs["T_model"]           = T_model
        f.attrs["time_step_ms"]      = time_step_ms
        f.attrs["window_duration_s"] = win_dur
        f.attrs["window_stride_s"]   = win_stride
        f.attrs["target_sr"]         = target_sr
        f.attrs["dataset"]           = "d3_combined"
        f.attrs["d1_path"]           = str(d1_path)
        f.attrs["d2_path"]           = str(d2_path)

        # Train: D1 + D2 concatenated
        train_ids = np.concatenate([d1_train_ids, d2_train_ids])
        f.create_dataset("train/stimulus_ids", data=train_ids)
        nd_train = f.create_group("train/neural_data/group")
        for roi in common_rois:
            arr = np.concatenate([d1_train[roi], d2_train[roi]], axis=0)
            nd_train.create_dataset(roi, data=arr,
                                    compression="gzip", compression_opts=4)

        # Test: combined (standard schema for evaluator)
        test_ids_combined = np.concatenate([d1_test_ids, d2_test_ids])
        f.create_dataset("test/stimulus_ids", data=test_ids_combined)
        nd_test = f.create_group("test/neural_data/group")
        for roi in common_rois:
            arr = np.concatenate([d1_test[roi], d2_test[roi]], axis=0)
            nd_test.create_dataset(roi, data=arr,
                                   compression="gzip", compression_opts=4)

        # Per-dataset test groups for post-hoc scoring
        for tag, ids, data in [
            ("test_d1", d1_test_ids, d1_test),
            ("test_d2", d2_test_ids, d2_test),
        ]:
            f.create_dataset(f"{tag}/stimulus_ids", data=ids)
            nd = f.create_group(f"{tag}/neural_data/group")
            for roi in common_rois:
                nd.create_dataset(roi, data=data[roi],
                                  compression="gzip", compression_opts=4)

        # Noise ceilings
        nc_grp = f.create_group("noise_ceilings/group")
        for roi in common_rois:
            nc_grp.create_dataset(roi, data=nc_combined[roi], compression="gzip")

    print(f"Written: {out_path}")
    print(f"  Train: {len(train_ids)} stimuli | Test: {len(test_ids_combined)} stimuli")
    print(f"  ROIs: {len(common_rois)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine D1 + D2 EEG HDF5 files into D3 for combined training."
    )
    p.add_argument("--d1_hdf5_path", required=True, help="Path to D1 HDF5.")
    p.add_argument("--d2_hdf5_path", required=True, help="Path to D2 HDF5.")
    p.add_argument("--output_path",  required=True, help="Path for D3 output HDF5.")
    p.add_argument("--overwrite", type=str2bool, default=False)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
