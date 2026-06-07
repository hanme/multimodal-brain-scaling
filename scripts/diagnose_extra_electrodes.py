"""Extract NC for electrodes not in the stored ROI list (e.g. Fp2, C2).

Uses the whole_brain NC array ([T, 128]) already in the HDF5 and does the same
nearest-neighbour montage lookup as format_eeg_hdf5._find_roi_indices().

Run from repo root with venv active:
    python scripts/diagnose_extra_electrodes.py
"""

from pathlib import Path
import h5py
import mne
import numpy as np

REPO   = Path(__file__).resolve().parent.parent
HDF5   = REPO / "outputs/neural_data/broderick2018_30s.h5"
BIDS   = Path("/work/upschrimpf1/mehrer/datasets/"
              "Broderick_2018_EEG_The_old_man_and_the_sea")

# Electrodes to check (standard 10-20 names, including extended montage)
EXTRA = ["Fp1", "Fp2", "C1", "C2", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
         "CP1", "CP2", "CP3", "CP4", "CPz", "AFz", "AF3", "AF4"]

mne.set_log_level("ERROR")

# ── 1. Get channel names from one BrainVision file ───────────────────────────
vhdr = next((BIDS / "sub-001" / "eeg").glob("*.vhdr"), None)
if vhdr is None:
    raise FileNotFoundError("No .vhdr found under sub-001/eeg/")
raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
ch_names = raw.ch_names          # e.g. ['A1', 'A2', ..., 'D32']
n_ch = len(ch_names)
print(f"Dataset channels: {n_ch}  ({ch_names[0]} … {ch_names[-1]})")

# ── 2. Build BioSemi position array aligned to dataset channels ───────────────
montage_bsl = mne.channels.make_standard_montage("biosemi128")
montage_std = mne.channels.make_standard_montage("standard_1020")
bsl_pos = montage_bsl.get_positions()["ch_pos"]
std_pos = montage_std.get_positions()["ch_pos"]

ch_xyz = np.array([bsl_pos.get(n, [np.nan]*3) for n in ch_names])

# ── 3. Nearest-neighbour lookup for extra electrodes ─────────────────────────
mapping: dict[str, int] = {}
bsl_names = list(bsl_pos.keys())
for name in EXTRA:
    if name not in std_pos:
        print(f"  {name}: not in standard_1020 montage — skipping")
        continue
    xyz = std_pos[name]
    dists = np.linalg.norm(ch_xyz - xyz, axis=1)
    valid = ~np.isnan(dists)
    if not valid.any():
        print(f"  {name}: no valid BioSemi channel found")
        continue
    idx = int(np.where(valid, dists, np.inf).argmin())
    mapping[name] = idx

# ── 4. Load whole_brain NC [T, 128] and extract per-electrode mean NC ─────────
with h5py.File(HDF5) as f:
    # whole_brain NC: stored as [T, n_ch] values in % variance explained
    nc_wb = f["noise_ceilings"]["group"]["whole_brain"][()]  # [T, 128]

# Mean over time bins (same as how single-electrode NC is reported)
nc_mean_per_ch = nc_wb.mean(axis=0)    # [128]
nc_max_per_ch  = nc_wb.max(axis=0)

# ── 5. Also grab the already-stored single-electrode NCs for comparison ───────
already_stored = {}
with h5py.File(HDF5) as f:
    nc_grp = f["noise_ceilings"]["group"]
    for roi in nc_grp.keys():
        v = nc_grp[roi][()]
        already_stored[roi] = float(v.mean())

print("\n── Already-stored ROI noise ceilings (mean over T) ──")
print(f"{'ROI':22s}  {'NC mean (%)':>11s}")
print("-" * 38)
for roi, nc in sorted(already_stored.items()):
    flag = " ⚠" if nc < 5 else ""
    print(f"  {roi:20s}  {nc:11.1f}{flag}")

print("\n── Extra electrodes (extracted from whole_brain NC) ──")
print(f"{'Standard':8s}  {'BSL ch':8s}  {'ch_idx':>6s}  "
      f"{'NC mean (%)':>11s}  {'NC max (%)':>10s}  {'dist mm':>7s}")
print("-" * 65)
for name in EXTRA:
    if name not in mapping:
        continue
    idx = mapping[name]
    bsl_ch = ch_names[idx]
    xyz = std_pos[name]
    dists = np.linalg.norm(ch_xyz - xyz, axis=1)
    dist_mm = float(dists[idx]) * 1000
    nc_m = float(nc_mean_per_ch[idx])
    nc_x = float(nc_max_per_ch[idx])
    flag = " ⚠" if nc_m < 5 else ""
    print(f"  {name:6s}  {bsl_ch:8s}  {idx:6d}  "
          f"{nc_m:11.1f}  {nc_x:10.1f}  {dist_mm:7.1f}{flag}")
