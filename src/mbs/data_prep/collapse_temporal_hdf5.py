"""Collapse a temporal neural HDF5 (shape [n_stim, T, n_ch]) to mean-pooled ([n_stim, n_ch]).

Reads an HDF5 written by format_eeg_hdf5.py (temporal=True) and writes a new file
compatible with mbs-evaluate-all-layers, which expects 2-D neural data.

  neural_data  [n_stim, T, n_ch]  →  mean over T  →  [n_stim, n_ch]
  noise_ceil   [T, n_ch]          →  mean over T  →  [n_ch]

The mean-over-T noise ceiling is an approximation. The "correct" NC for mean-pooled EEG
would require re-computing split-half correlations on mean-pooled responses, but since we
don't store per-half data in the HDF5, the temporal mean is used instead. This is acceptable
for a sanity-check pilot (Phase 4a); the temporal NC (Phase 4b) is the authoritative metric.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main(args):
    src = Path(args.input_path)
    dst = Path(args.output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        # ── copy and patch attrs ───────────────────────────────────────────
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["temporal"] = False
        for drop in ("T_model", "time_step_ms"):
            if drop in fout.attrs:
                del fout.attrs[drop]

        # ── noise ceilings: [T, n_ch] → [n_ch] ────────────────────────────
        for subj in fin["noise_ceilings"]:
            for roi in fin["noise_ceilings"][subj]:
                data = fin["noise_ceilings"][subj][roi][()]
                if data.ndim == 2:
                    data = data.mean(axis=0)
                fout.create_dataset(
                    f"noise_ceilings/{subj}/{roi}",
                    data=data.astype(np.float32),
                    compression="gzip",
                )

        # ── train / test splits ────────────────────────────────────────────
        for split in ("train", "test"):
            if split not in fin:
                continue
            ids = fin[f"{split}/stimulus_ids"][()]
            fout.create_dataset(f"{split}/stimulus_ids", data=ids)

            for subj in fin[f"{split}/neural_data"]:
                for roi in fin[f"{split}/neural_data"][subj]:
                    data = fin[f"{split}/neural_data"][subj][roi][()]
                    if data.ndim == 3:
                        data = data.mean(axis=1)    # [n_stim, T, n_ch] → [n_stim, n_ch]
                    fout.create_dataset(
                        f"{split}/neural_data/{subj}/{roi}",
                        data=data.astype(np.float32),
                        compression="gzip",
                    )

    print(f"Written: {dst}")
    with h5py.File(dst, "r") as f:
        rois = list(f.attrs.get("rois", []))
        max_nc = float(f.attrs.get("max_nc", 100.0))
        for roi in rois:
            shape = f[f"train/neural_data/group/{roi}"].shape
            nc_raw = f[f"noise_ceilings/group/{roi}"][()].mean()
            nc_pct = nc_raw / max_nc * 100
            print(f"  {roi}: shape={shape}, NC mean={nc_pct:.1f}%")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True, help="Temporal HDF5 written by format_eeg_hdf5.py.")
    parser.add_argument("--output_path", required=True, help="Mean-pooled HDF5 to write.")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
