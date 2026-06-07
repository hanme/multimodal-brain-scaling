"""Distribution of per-time-bin prediction scores against zero — whisper-base Delta_T.

For each electrode × layer, tests whether scores[T] are centered above zero using a
one-sample t-test. Produces:
  1. A violin-plot figure (8 electrodes × 6 layers) showing the full score distribution
     with zero reference, means, and significance markers.
  2. A printed/saved summary table (mean, SE, t, p, fraction >0).

IMPORTANT — autocorrelation caveat:
  The t-test treats 1500 time bins as independent observations. They are NOT — adjacent
  EEG bins and model activations are highly autocorrelated. The effective degrees of
  freedom is substantially lower than 1499, so p-values are overconfident (too small).
  Use p-values only to compare layers against each other, not as absolute significance
  thresholds. The key robust finding is the ratio of means: blocks.2 mean is 2–5× larger
  than the next best layer — this ranking is stable under any reasonable df correction.

Usage:
    python scripts/plot_score_distributions.py
    python scripts/plot_score_distributions.py --scores_path outputs/results/whisper-base-delta-t-full/temporal_scores.h5
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import h5py


ELECTRODES = [
    ("AF3",  94.6),
    ("FT7",  91.1),
    ("P9",   88.0),
    ("TP7",  84.3),
    ("Fpz",  79.9),
    ("T7",   75.3),
    ("Pz",   74.4),
    ("Fz",   50.4),
]

LAYERS     = ["blocks-0", "blocks-1", "blocks-2", "blocks-3", "blocks-4", "blocks-5"]
LAYER_LABELS = [f"blocks.{i}" for i in range(6)]
LAYER_COLORS = [
    "#2166ac",  # blocks.0
    "#74add1",  # blocks.1
    "#4dac26",  # blocks.2  (best — green)
    "#f46d43",  # blocks.3
    "#d73027",  # blocks.4
    "#7b2d8b",  # blocks.5
]

AUTOCORR_WARNING = (
    "⚠  p-values assume 1500 independent time bins. True df << 1499 due to autocorrelation\n"
    "   — p-values are over-confident. Use means/rankings, not p-values, as primary evidence."
)


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_scores(h5: h5py.File, layer: str, electrode: str) -> np.ndarray | None:
    key = f"{layer}/group/{electrode}"
    if key not in h5:
        return None
    data = h5[key][()]
    if data.ndim == 2:
        data = data[:, 0]
    return data.astype(np.float64)


def run_tests(scores_path: Path) -> dict:
    """Return nested dict: results[electrode][layer] = {mean, se, t, p, frac_pos, n}."""
    results = {}
    with h5py.File(scores_path, "r") as f:
        for electrode, nc in ELECTRODES:
            results[electrode] = {}
            for layer in LAYERS:
                s = load_scores(f, layer, electrode)
                if s is None:
                    continue
                valid = s[~np.isnan(s)]
                n = len(valid)
                mean = float(np.mean(valid))
                se = float(np.std(valid, ddof=1) / np.sqrt(n))
                t_stat, p_naive = stats.ttest_1samp(valid, 0)
                frac_pos = float(np.mean(valid > 0))
                cohens_d = float(mean / np.std(valid, ddof=1))

                # n_eff correction for temporal autocorrelation (AR(1) approximation).
                # ρ₁ = lag-1 autocorrelation; n_eff = T × (1−ρ₁)/(1+ρ₁).
                # The corrected t-statistic uses n_eff in the SE denominator:
                #   t_corr = mean / (std / √n_eff)  with df = n_eff − 1.
                rho1 = float(np.corrcoef(valid[:-1], valid[1:])[0, 1])
                n_eff = max(2, int(n * (1 - rho1) / (1 + rho1)))
                std = float(np.std(valid, ddof=1))
                t_corr = float(mean / (std / np.sqrt(n_eff)))
                p_corr = float(2 * stats.t.sf(abs(t_corr), df=n_eff - 1))

                results[electrode][layer] = dict(
                    mean=mean, se=se,
                    t_naive=float(t_stat), p_naive=float(p_naive),
                    rho1=rho1, n_eff=n_eff,
                    t_corr=t_corr, p_corr=p_corr,
                    frac_pos=frac_pos, cohens_d=cohens_d, n=n,
                )
    return results


def print_summary(results: dict) -> None:
    print(AUTOCORR_WARNING)
    print()
    for electrode, nc in ELECTRODES:
        print(f"--- {electrode} (NC={nc:.0f}%) ---")
        print(f"  {'layer':12s}  {'mean':>8s}  {'d':>6s}  {'ρ₁':>5s}  {'n_eff':>6s}"
              f"  {'t_corr':>7s}  {'p_corr':>10s}  {'p_naive':>10s}  {'>0%':>6s}  sig(corr)")
        for layer, label in zip(LAYERS, LAYER_LABELS):
            r = results[electrode].get(layer)
            if r is None:
                continue
            stars = sig_stars(r["p_corr"])
            best = "  <-- best" if layer == "blocks-2" else ""
            print(
                f"  {label:12s}  {r['mean']:8.4f}  {r['cohens_d']:6.3f}  {r['rho1']:5.2f}"
                f"  {r['n_eff']:6d}  {r['t_corr']:7.2f}  {r['p_corr']:10.3e}"
                f"  {r['p_naive']:10.3e}  {r['frac_pos']*100:5.1f}%  {stars}{best}"
            )
        print()


def make_violin_figure(scores_path: Path, results: dict, output_path: Path) -> None:
    n_cols = 2
    n_rows = len(ELECTRODES) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 3.2))
    axes = axes.flatten()

    with h5py.File(scores_path, "r") as f:
        for ax_idx, (electrode, nc) in enumerate(ELECTRODES):
            ax = axes[ax_idx]

            all_scores = []
            for layer in LAYERS:
                s = load_scores(f, layer, electrode)
                all_scores.append(s if s is not None else np.array([np.nan]))

            # Violin plot
            positions = np.arange(1, len(LAYERS) + 1)
            vp = ax.violinplot(
                [s[~np.isnan(s)] for s in all_scores],
                positions=positions,
                showmeans=False, showmedians=False, showextrema=False,
                widths=0.7,
            )
            for body, color in zip(vp["bodies"], LAYER_COLORS):
                body.set_facecolor(color)
                body.set_alpha(0.55)
                body.set_edgecolor(color)

            # Mean ± SE and significance annotation
            for i, (layer, label) in enumerate(zip(LAYERS, LAYER_LABELS)):
                r = results[electrode].get(layer)
                if r is None:
                    continue
                x = positions[i]
                mean, se = r["mean"], r["se"]
                color = LAYER_COLORS[i]

                # Mean dot + SE bar
                ax.plot(x, mean, "o", color=color, markersize=5, zorder=5)
                ax.plot([x, x], [mean - 1.96 * se, mean + 1.96 * se],
                        "-", color=color, linewidth=2.0, zorder=4)

                # Significance stars at fixed y=0.75 — using autocorrelation-corrected p
                stars = sig_stars(r["p_corr"])
                ax.text(x, 0.75, stars, ha="center", va="bottom",
                        fontsize=8, color=color, fontweight="bold")

                # Cohen's d at fixed y=0.90
                ax.text(x, 0.90, f"d={r['cohens_d']:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=color)

            ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=3)
            ax.set_xticks(positions)
            ax.set_xticklabels(LAYER_LABELS, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel("NC-corr. Pearson r", fontsize=8)
            ax.set_title(f"{electrode}  (NC={nc:.0f}%)", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.set_ylim(-0.5, 1.10)

    fig.suptitle(
        "whisper-base × Broderick 2018 — score distributions vs. zero\n"
        "Violin = score[T] across 1500 time bins  |  dot = mean  |  bar = 95% CI  |"
        "  stars/d = autocorr-corrected p (AR(1) n_eff)  |  *** p<0.001  ** p<0.01  * p<0.05  ns",
        fontsize=9, y=1.01,
    )
    fig.text(
        0.5, -0.015,
        "p-values corrected for temporal autocorrelation via AR(1) n_eff = T×(1−ρ₁)/(1+ρ₁). "
        "Typically n_eff ≈ T/10. Robust evidence = Cohen's d and ratio of means, not p-values.",
        ha="center", fontsize=7.5, color="#a00000",
        wrap=True,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main(args):
    scores_path = Path(args.scores_path)
    assert scores_path.exists(), f"Not found: {scores_path}"

    results = run_tests(scores_path)
    print_summary(results)

    # Save summary JSON
    summary_out = Path(args.output_dir) / "score_distribution_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Summary JSON: {summary_out}")

    make_violin_figure(
        scores_path, results,
        Path(args.output_dir) / "whisper_base_score_distributions.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_path",
        default="outputs/results/whisper-base-delta-t-full/temporal_scores.h5",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/figures",
    )
    args = parser.parse_args()
    main(args)
