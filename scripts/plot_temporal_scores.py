"""Plot per-time-step prediction score curves (scores[T]) for whisper-base Delta_T evaluation.

For each of 8 selected electrodes, plots all 6 layer curves overlaid, both raw (faded) and
smoothed (bold). Purpose: visual inspection of temporal structure and layer ranking.

Usage:
    python scripts/plot_temporal_scores.py
    python scripts/plot_temporal_scores.py --scores_path outputs/results/whisper-base-delta-t-full/temporal_scores.h5
    python scripts/plot_temporal_scores.py --sigma_bins 15  # wider smoothing
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import gaussian_filter1d


ELECTRODES = [
    # (name, NC%)  — ordered by anatomy / interest
    ("AF3",  94.6),  # left anterior frontal — highest NC
    ("FT7",  91.1),  # left fronto-temporal
    ("P9",   88.0),  # left posterior temporal
    ("TP7",  84.3),  # left temporo-parietal
    ("Fpz",  79.9),  # frontal pole midline
    ("T7",   75.3),  # left temporal — primary auditory
    ("Pz",   74.4),  # parietal midline
    ("Fz",   50.4),  # primary MMN electrode
]

LAYERS = ["blocks-0", "blocks-1", "blocks-2", "blocks-3", "blocks-4", "blocks-5"]
LAYER_LABELS = [f"blocks.{i}" for i in range(6)]

# Perceptually distinct colors for 6 layers — avoids near-white in middle of coolwarm
LAYER_COLORS = [
    "#2166ac",  # blocks.0 — dark blue
    "#74add1",  # blocks.1 — light blue
    "#4dac26",  # blocks.2 — green  (best layer — stands out)
    "#f46d43",  # blocks.3 — orange
    "#d73027",  # blocks.4 — red
    "#7b2d8b",  # blocks.5 — purple
]

TIME_STEP_S = 0.020  # 20 ms per bin


def load_scores(scores_path: Path, layer: str, electrode: str) -> np.ndarray | None:
    """Return scores[T] for a given layer/electrode, or None if key missing."""
    key = f"{layer}/group/{electrode}"
    with h5py.File(scores_path, "r") as f:
        if key not in f:
            return None
        data = f[key][()]  # shape [T, n_ch]
    if data.ndim == 2:
        data = data[:, 0]
    return data.astype(np.float64)


def main(args):
    scores_path = Path(args.scores_path)
    assert scores_path.exists(), f"Scores HDF5 not found: {scores_path}"

    n_elec = len(ELECTRODES)
    n_cols = 2
    n_rows = n_elec // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.0), sharex=True)
    axes = axes.flatten()

    t_axis = np.arange(0, 1500) * TIME_STEP_S  # seconds

    for ax_idx, (electrode, nc) in enumerate(ELECTRODES):
        ax = axes[ax_idx]

        for layer, label, color in zip(LAYERS, LAYER_LABELS, LAYER_COLORS):
            raw = load_scores(scores_path, layer, electrode)
            if raw is None:
                continue

            smoothed = gaussian_filter1d(raw, sigma=args.sigma_bins)

            # Raw: thin, faded
            ax.plot(t_axis, raw, color=color, alpha=0.12, linewidth=0.5)
            # Smoothed: bold, opaque
            ax.plot(t_axis, smoothed, color=color, alpha=0.90, linewidth=1.6, label=label)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(f"{electrode}  (NC={nc:.0f}%)", fontsize=10, fontweight="bold")
        ax.set_ylabel("NC-corr. Pearson r", fontsize=8)
        ax.tick_params(labelsize=8)

        # y-axis: clip display range to avoid NC-blowup artifacts
        ax.set_ylim(-0.3, 1.1)

    for ax in axes:
        ax.set_xlabel("Time (s)", fontsize=8)

    # Single legend (layer colors) on last panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Layer (blue=early, red=late)",
        loc="lower center",
        ncol=len(LAYERS),
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Annotation explaining line styles
    fig.text(
        0.01, 0.01,
        "Thin faded = raw scores (20 ms bins)   |   Bold = Gaussian-smoothed "
        f"(σ={args.sigma_bins} bins = {args.sigma_bins * TIME_STEP_S * 1000:.0f} ms)",
        fontsize=7, color="gray", va="bottom",
    )

    fig.suptitle(
        "whisper-base  ×  Broderick 2018 (ds004408)  —  Delta_T temporal encoding scores",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_path",
        default="outputs/results/whisper-base-delta-t-full/temporal_scores.h5",
    )
    parser.add_argument(
        "--output_path",
        default="outputs/figures/whisper_base_temporal_scores.png",
    )
    parser.add_argument(
        "--sigma_bins",
        type=int,
        default=25,
        help="Gaussian smoothing sigma in time bins (1 bin = 20 ms). Default 25 = 500 ms.",
    )
    args = parser.parse_args()
    main(args)
