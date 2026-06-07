"""Diagnose the standard-10-20 → BioSemi128 channel mapping used by format_eeg_hdf5.

Run from the repo root with the venv active:
    python scripts/diagnose_roi_mapping.py

Checks:
    1. Which BioSemi128 channel each standard name maps to and the distance.
    2. Whether any collisions exist (two standards map to the same channel).
    3. Raw split-half correlation (before Spearman-Brown and clipping) for each ROI
       from the existing HDF5, so we can see if FCz's r is slightly negative or zero.
"""

import sys
from pathlib import Path

import h5py
import mne
import numpy as np

REPO = Path(__file__).resolve().parent.parent
HDF5  = REPO / "outputs/neural_data/broderick2018_30s.h5"
TARGETS = ["Fz", "FCz", "Cz", "Pz", "F3", "F4", "C3", "C4", "T7", "T8"]

mne.set_log_level("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Channel mapping
# ─────────────────────────────────────────────────────────────────────────────
montage_bsl = mne.channels.make_standard_montage("biosemi128")
montage_std = mne.channels.make_standard_montage("standard_1020")

bsl_pos_dict = montage_bsl.get_positions()["ch_pos"]
std_pos_dict = montage_std.get_positions()["ch_pos"]

bsl_names = list(bsl_pos_dict.keys())
bsl_xyz   = np.array(list(bsl_pos_dict.values()))

print("=" * 60)
print("BioSemi128 montage channel names (first 10):", bsl_names[:10])
print("=" * 60)

print(f"\n{'Standard':8s}  {'Nearest BioSemi ch':18s}  {'BSL index':>9s}  {'Dist (mm)':>9s}")
print("-" * 55)

mapping: dict[str, int] = {}
for name in TARGETS:
    if name not in std_pos_dict:
        print(f"{name:8s}  NOT IN standard_1020 MONTAGE")
        continue
    target_xyz = std_pos_dict[name]
    dists = np.linalg.norm(bsl_xyz - target_xyz, axis=1)
    idx = int(np.argmin(dists))
    mapping[name] = idx
    print(f"{name:8s}  {bsl_names[idx]:18s}  {idx:9d}  {dists[idx]*1000:9.1f}")

# Collisions
print("\n── Collision check ──")
inv: dict[int, list[str]] = {}
for name, idx in mapping.items():
    inv.setdefault(idx, []).append(name)
collision = False
for idx, names in inv.items():
    if len(names) > 1:
        print(f"  *** COLLISION: BioSemi ch {bsl_names[idx]} (idx {idx}) "
              f"← {names}")
        collision = True
if not collision:
    print("  No collisions — each standard maps to a unique BioSemi channel.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Raw split-half r from the HDF5 noise ceilings (back-compute from stored NC)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── NC stored in {HDF5.name} ──")
print(f"{'ROI':22s}  {'NC mean (%)':>11s}  {'note'}")
print("-" * 55)

if not HDF5.exists():
    print(f"  HDF5 not found: {HDF5}")
    sys.exit(0)

with h5py.File(HDF5) as f:
    rois = list(f.attrs["rois"])
    for roi in sorted(rois):
        if isinstance(roi, bytes):
            roi = roi.decode()
        nc = f["noise_ceilings"]["group"][roi][()]  # [T, n_ch] or [n_ch]
        nc_mean = float(np.mean(nc))
        nc_max  = float(np.max(nc))
        note = ""
        if nc_mean < 1.0:
            note = "⚠ near-zero — raw split-half r likely ≤ 0"
        elif nc_max < 5.0:
            note = "⚠ very low"
        print(f"  {roi:20s}  {nc_mean:11.2f}  {note}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load raw EEG for the FCz-mapped channel and compute split-half manually
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Manual split-half for FCz-mapped channel ──")
try:
    import mne
    bids_root = Path(
        "/work/upschrimpf1/mehrer/datasets/"
        "Broderick_2018_EEG_The_old_man_and_the_sea"
    )
    # Load one subject's first run to inspect the FCz channel
    sub = "sub-001"
    run = "01"
    vhdr = bids_root / sub / "eeg" / f"{sub}_task-listen_run-{run}_eeg.vhdr"
    if not vhdr.exists():
        print("  Could not find raw file to inspect — skipping raw check.")
    else:
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        if "FCz" in mapping:
            fcz_bsl_ch = bsl_names[mapping["FCz"]]
            fz_bsl_ch  = bsl_names[mapping.get("Fz", -1)]
            print(f"  FCz → BioSemi ch: {fcz_bsl_ch}   "
                  f"(Fz → {fz_bsl_ch})")
            if fcz_bsl_ch in raw.ch_names:
                idx = raw.ch_names.index(fcz_bsl_ch)
                data = raw.get_data()[idx]
                print(f"  FCz-ch signal: mean={data.mean():.3e}  "
                      f"std={data.std():.3e}  "
                      f"max_abs={np.abs(data).max():.3e}")
            else:
                print(f"  Channel {fcz_bsl_ch} not found in raw data. "
                      f"Raw channels: {raw.ch_names[:8]} ...")
except Exception as e:
    print(f"  Raw inspection failed: {e}")

print("\nDone.")
