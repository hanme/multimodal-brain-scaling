"""Plot mTRF encoding-vs-lag curves.

Default: the no-highpass vs highpass-1Hz diagnostic for blocks-2 (the question of
whether slow autocorrelation flattens the lag curve). Reads the per-layer HDF5s written
by evaluate_features_mtrf.py.

Usage:
    python scripts/plot_mtrf_scores.py
"""
from pathlib import Path
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_curve(h5_path, layer, roi, which="scores_raw"):
    with h5py.File(h5_path, "r") as f:
        lags = np.array(f.attrs["lags_ms"], dtype=float)
        k = f"{layer}/group/{roi}"
        if k not in f:
            return lags, None
        return lags, f[k][which][:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nohp_dir", default="outputs/results/whisper-base-mtrf-full/layer_2")
    ap.add_argument("--hp_dir", default="outputs/results/whisper-base-mtrf-hp1/layer_2")
    ap.add_argument("--layer", default="blocks-2")
    ap.add_argument("--rois", default="Fz,T7,T8,AF3,FT7,TP7,Pz")
    ap.add_argument("--out", default="outputs/figures/mtrf_highpass_diagnostic.png")
    args = ap.parse_args()

    rois = [r.strip() for r in args.rois.split(",")]
    nohp = f"{args.nohp_dir}/mtrf_scores.h5"
    hp = f"{args.hp_dir}/mtrf_scores.h5"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rois)))

    for ax, h5, title in [
        (axes[0], nohp, "No high-pass (raw r) — slow-autocorrelation confound"),
        (axes[1], hp, "High-pass 1 Hz (raw r) — stimulus-locked component"),
    ]:
        for roi, c in zip(rois, cmap):
            lags, y = get_curve(h5, args.layer, roi)
            if y is None:
                continue
            ax.plot(lags, y, label=roi, color=c, lw=1.8)
            i = int(np.nanargmax(y))
            ax.plot(lags[i], y[i], "o", color=c, ms=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("stimulus→EEG lag (ms)")
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel(f"encoding r ({args.layer})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
