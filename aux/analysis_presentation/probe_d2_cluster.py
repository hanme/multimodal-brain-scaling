#!/usr/bin/env python
"""Read the D2 (Weissbart Cortical Surprisal) HDF5 and close the open items in
aux/analysis_presentation/dataset_and_design_counts.md.

Run on the cluster from the repo root after `source env.sh`. Read-only: opens files with
mode 'r' and writes nothing.

    python aux/analysis_presentation/probe_d2_cluster.py
    python aux/analysis_presentation/probe_d2_cluster.py --h5 outputs/neural_data/surprisal_10s.h5

What each block settles (numbering matches the memo's "Still open" list):
  1  the h5's own attributes -- n_subjects, window/stride, target_sr, channel_names
  2  per-part window counts for BOTH splits, including the 3 test parts
  3  whether format_eeg_hdf5_surprisal.py windows like format_eeg_hdf5.py, by predicting each
     part's window count from its wav duration and diffing against the h5
  4  which of the 63 real electrodes survive the NC floor, and which do not (the file's `rois`
     also contain 6 pseudo-channels -- 5 `*_cluster` averages + `whole_brain` -- which the mapping
     drops via eeg_targets.NON_ELECTRODE and which are reported separately here)
  7  n_subjects as recorded by the formatter (group_count)
"""
import argparse
import glob
import os
import wave
from collections import Counter, defaultdict

import h5py
import numpy as np

DEFAULT_AUDIO = ("/work/upschrimpf1/sigfstea/multimodal-brain-scaling-temporal-analysis"
                 "/data/cortical_suprisal_dataset/audiobooks")


def wav_duration_s(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def part_of(sid):
    """'AUNP02_0160000' -> 'AUNP02' (same rule as scripts/eeg_targets.py:part_group)."""
    return sid.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="outputs/neural_data/surprisal_30s.h5")
    ap.add_argument("--audio_dir", default=DEFAULT_AUDIO)
    ap.add_argument("--nc_threshold", type=float, default=0.2)
    a = ap.parse_args()

    f = h5py.File(a.h5, "r")

    # ---- 1. attributes -----------------------------------------------------
    print("=" * 72)
    print("1. FILE ATTRIBUTES  --", a.h5)
    print("=" * 72)
    for k in sorted(f.attrs):
        v = f.attrs[k]
        if isinstance(v, np.ndarray) and v.size > 12:
            print(f"  {k:24s} <{v.dtype} array, n={v.size}>  first 6: {list(v[:6])}")
        else:
            print(f"  {k:24s} {v}")

    win = float(f.attrs.get("window_duration_s", 30.0))
    stride = float(f.attrs.get("window_stride_s", 10.0))
    fs = float(f.attrs.get("target_sr", 50))
    subjects = [s.decode() if hasattr(s, "decode") else s for s in f.attrs.get("subjects", ["group"])]
    print(f"\n  -> window {win} s / stride {stride} s @ {fs} Hz; subjects stored: {subjects}")
    if "n_subjects" not in f.attrs:
        print("  -> WARNING: no n_subjects attr (the D2 formatter may name it differently)")

    # ---- 2. per-part window counts ----------------------------------------
    print("\n" + "=" * 72)
    print("2. WINDOWS PER AUDIOBOOK PART, PER SPLIT")
    print("=" * 72)
    splits = [s.decode() if hasattr(s, "decode") else s
              for s in f.attrs.get("splits", ["train", "test"])]
    per_split, part_split = {}, {}
    for split in splits:
        if split not in f:
            print(f"  (no group '{split}')")
            continue
        ids = [s.decode() if hasattr(s, "decode") else s for s in f[split]["stimulus_ids"][()]]
        counts = Counter(part_of(i) for i in ids)
        per_split[split] = counts
        for p in counts:
            part_split[p] = split
        cov = stride * len(ids) + (win - stride) * len(counts)
        print(f"\n  {split}: {len(ids)} windows over {len(counts)} parts"
              f"   -> covers {stride:.0f}*{len(ids)} + {win - stride:.0f}*{len(counts)} = {cov:.0f} s"
              f" = {cov / 60:.1f} min")
        for p in sorted(counts):
            print(f"      {p:10s} {counts[p]:4d}")

    all_counts = Counter()
    for c in per_split.values():
        all_counts.update(c)
    n_w, n_p = sum(all_counts.values()), len(all_counts)
    total = stride * n_w + (win - stride) * n_p
    print(f"\n  TOTAL: {n_w} windows over {n_p} parts")
    print(f"  covered audio     = {total:.0f} s = {total / 60:.2f} min per subject")
    n_subj = int(f.attrs.get("n_subjects", 13))
    print(f"  subject-hours     = {n_subj} x {total:.0f}/3600 = {n_subj * total / 3600:.2f} h")

    # ---- 3. windowing-rule equivalence + true durations --------------------
    print("\n" + "=" * 72)
    print("3. PREDICTED vs ACTUAL WINDOW COUNTS  (does the D2 formatter tile like the D1 one?)")
    print("=" * 72)
    wavs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(a.audio_dir, "*.wav"))}
    if not wavs:
        print(f"  no wavs under {a.audio_dir} -- pass --audio_dir; skipping")
    else:
        print(f"  {'part':10s} {'dur_s':>8s} {'pred':>5s} {'actual':>7s} {'tail_s':>7s}  split")
        ok, dur_total = True, 0.0
        for p in sorted(set(all_counts) | set(wavs)):
            if p not in wavs:
                print(f"  {p:10s} {'--':>8s} {'--':>5s} {all_counts.get(p, 0):>7d}"
                      f" {'--':>7s}  NO WAV")
                ok = False
                continue
            d = wav_duration_s(wavs[p])
            dur_total += d
            pred = int(np.floor((d - win) / stride)) + 1 if d >= win else 0
            act = all_counts.get(p, 0)
            tail = d - (stride * (pred - 1) + win) if pred > 0 else d
            flag = "" if pred == act else "   <-- MISMATCH"
            ok &= (pred == act)
            print(f"  {p:10s} {d:8.2f} {pred:5d} {act:7d} {tail:7.2f}"
                  f"  {part_split.get(p, '-')}{flag}")
        print(f"\n  total audio       = {dur_total:.1f} s = {dur_total / 60:.2f} min per subject")
        print(f"  covered by windows= {total:.0f} s  ({100 * total / dur_total:.1f} %)"
              if dur_total else "")
        print(f"  subject-hours (full recording) = {n_subj} x {dur_total:.0f}/3600 ="
              f" {n_subj * dur_total / 3600:.2f} h")
        print(f"\n  VERDICT: {'identical tiling rule -- assumption CLOSED' if ok else 'MISMATCH -- the D2 formatter does NOT tile like format_eeg_hdf5.py'}")

    # ---- 4. NC floor: which channels survive -------------------------------
    print("\n" + "=" * 72)
    print(f"4. NOISE-CEILING FLOOR (r > {a.nc_threshold})")
    print("=" * 72)
    rois = [r.decode() if hasattr(r, "decode") else r for r in f.attrs.get("rois", [])]
    chans = [c.decode() if hasattr(c, "decode") else c for c in f.attrs.get("channel_names", [])]
    print(f"  rois in file: {len(rois)}   channel_names: {len(chans)}")
    max_nc = float(f.attrs.get("max_nc", 100.0))
    subj = subjects[0]
    if "noise_ceilings" in f and subj in f["noise_ceilings"]:
        # `rois` mixes real electrodes with pseudo-channels; the mapping keeps only the former
        # (scripts/eeg_targets.py NON_ELECTRODE), so count them apart or you will report 53 not 47.
        pseudo_tags = ("_cluster", "whole_brain")
        keep, drop, pseudo = [], [], []
        for roi in rois:
            if roi not in f["noise_ceilings"][subj]:
                continue
            nc = np.sqrt(np.asarray(f["noise_ceilings"][subj][roi][()]) / max_nc + 1e-6)
            r = float(np.nanmean(nc))            # mean over time (and members, if any)
            if any(t in roi for t in pseudo_tags):
                pseudo.append((roi, r))
            else:
                (keep if r > a.nc_threshold else drop).append((roi, r))
        keep.sort(key=lambda x: -x[1])
        drop.sort(key=lambda x: -x[1])
        print(f"\n  {len(keep) + len(drop)} real electrodes + {len(pseudo)} pseudo-ROIs"
              f" = {len(rois)} rois")
        print(f"\n  SURVIVE ({len(keep)} electrodes): "
              + ", ".join(f"{n}={r:.3f}" for n, r in keep))
        print(f"\n  DROPPED ({len(drop)} electrodes): "
              + ", ".join(f"{n}={r:.3f}" for n, r in drop))
        near0 = [n for n, r in drop if r < 0.01]
        if near0:
            print(f"\n  of those, {len(near0)} sit at r < 0.01 -- functionally exact zero, the same"
                  f" signature that disqualified D1: {', '.join(near0)}")
        print(f"\n  pseudo-ROIs (not mapping targets): "
              + ", ".join(f"{n}={r:.3f}" for n, r in pseudo))
        d = dict(keep + drop + pseudo)
        for e in ("FCz", "Fz", "Cz", "FC1", "FC2", "F1", "F2"):
            print(f"    {e:4s} {d.get(e, float('nan')):.4f}" if e in d else f"    {e:4s} ABSENT")
    else:
        print("  no noise_ceilings group -- inspect manually")

    f.close()
    print("\nDone. Paste this whole output back for the memo/CSV update.")


if __name__ == "__main__":
    main()
