#!/usr/bin/env python
"""Figures for the novel tone-pair search (Phases 1 and 2).

Reads outputs/results_novel_search/{phase1_ranked_directions,phase2_final_ranking}.csv, the
ranking emitted by rank_novel_phase1.py / rank_novel_phase2.py. mTRF only, FCz, X = 0.75 uV,
6 models -- the same slice the ranking is defined on.

  1. novel_n_agree_heatmap.png     n_agree over the full 33x33 frequency space. Every off-diagonal
                                   cell is one direction-instance: row = standard, column =
                                   deviant, so (i,j) and (j,i) are the two directions of one pair
                                   and asymmetry about the diagonal IS the frequency-preference
                                   effect. The diagonal is empty by design -- a same-frequency
                                   "deviant" synthesizes to the standard's waveform, so the
                                   difference is exactly zero by construction.
  2. novel_deviance_scaling.png    S7 rate vs deviance, per model + pooled. A monotone rise then
                                   plateau is the expected signature; flat means the metric is
                                   not tracking deviance.
  3. novel_rank_stability.png      Phase-1 vs Phase-2 rank on the shared direction-instances.
                                   The methodological result: whether a 4-clip screen is a valid
                                   proxy for a 32-clip evaluation.

Figures 1-2 need only Phase 1; figure 3 needs both. Whatever is missing is skipped with a note,
so this is runnable as soon as Phase 1 lands.

  python aux/analysis_novel_search/plots/novel_search_plots.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "aux/analysis_novel_search/plots"
RESULTS = REPO / "outputs/results_novel_search"
sys.path.insert(0, str(REPO / "scripts"))
from novel_search_common import SEARCH_MODELS, DIP_UV_THRESHOLD       # noqa: E402

# Okabe-Ito CVD-safe categorical, identical to scripts/analyze_mmn_screen_24freq.py MODEL_STYLE
# minus whisper-large (excluded from this search). Every series also carries a distinct marker,
# so model identity is never colour-alone.
MODEL_STYLE = {
    "whisper-tiny":    dict(color="#56B4E9", marker="o"),   # sky blue        o
    "whisper-base":    dict(color="#0072B2", marker="s"),   # blue            []
    "whisper-small":   dict(color="#009E73", marker="^"),   # bluish green    ^
    "whisper-medium":  dict(color="#E69F00", marker="D"),   # orange          <>
    "wav2vec2-medium": dict(color="#CC79A7", marker="P"),   # reddish purple  +
    "wav2vec2-large":  dict(color="#000000", marker="X"),   # black           x
}
MLABEL = {"whisper-tiny": "whisper tiny", "whisper-base": "whisper base",
          "whisper-small": "whisper small", "whisper-medium": "whisper medium",
          "wav2vec2-medium": "wav2vec2 medium", "wav2vec2-large": "wav2vec2 large"}

POOL_COLOR = "#0072B2"        # slot-1 hue for a single pooled series
INK, MUTED = "#1a1a1a", "#6b6b6b"
REF_STYLE = dict(color="#8c8c8c", lw=0.9, ls=(0, (4, 3)), zorder=1)
# n_agree is ordinal magnitude -> ONE hue, light->dark, discretised to its 7 levels with a scale
# legend. Not a rainbow: a multi-hue ramp would invent category boundaries the data doesn't have.
SEQ_CMAP = "Blues"
# The excluded diagonal. Must be clearly distinct from the ramp's n_agree-0 step, which is
# near-white -- otherwise "no pair here" and "no model agreed" read as the same cell.
EMPTY_CELL = "#d0d0d0"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#cccccc",
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def _load(name):
    path = RESULTS / name
    if not path.exists():
        print(f"  skipped: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"  loaded {path.name}: {len(df)} direction-instances")
    return df


def _models_in(df):
    return [m for m in SEARCH_MODELS if f"s7__{m}" in df.columns]


# ──────────────────────────────────────────────────────────
# 1. n_agree over the 33x33 frequency space
# ──────────────────────────────────────────────────────────

def plot_heatmap(ph1, models):
    freqs = sorted(set(ph1["f_low"]) | set(ph1["f_high"]))
    idx = {f: i for i, f in enumerate(freqs)}
    n = len(freqs)

    grid = np.full((n, n), np.nan)
    for r in ph1.itertuples(index=False):
        std, dev = (r.f_low, r.f_high) if r.direction == "regular" else (r.f_high, r.f_low)
        grid[idx[std], idx[dev]] = r.n_agree

    n_models = len(models)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.grid(False)
    cmap = plt.get_cmap(SEQ_CMAP, n_models + 1).with_extremes(bad=EMPTY_CELL)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, origin="lower",
                   vmin=-0.5, vmax=n_models + 0.5, interpolation="nearest")

    ticks = range(0, n, 4)
    ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
    ax.set_xticklabels([f"{freqs[i]:g}" for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([f"{freqs[i]:g}" for i in ticks])
    ax.set_xlabel("deviant frequency (Hz)")
    ax.set_ylabel("standard frequency (Hz)")
    ax.set_title(f"Cross-model agreement over the novel frequency grid\n"
                 f"n_agree = models showing S7 at FCz (X = {DIP_UV_THRESHOLD} µV), "
                 f"of {n_models}", pad=12)

    cbar = fig.colorbar(im, ax=ax, ticks=range(n_models + 1), fraction=0.046, pad=0.03)
    cbar.set_label(f"n_agree (0–{n_models})")
    cbar.outline.set_visible(False)

    # Caption below the axes, not over the cells: annotation text on a dark ramp step is
    # unreadable in either ink or white.
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(0.5, 0.045,
             "Each off-diagonal cell is one direction-instance; (i,j) and (j,i) are the two "
             "directions of one pair, so\nasymmetry about the diagonal is the frequency-preference "
             "effect. Grey diagonal = excluded by design (a same-\nfrequency deviant synthesizes "
             "to the standard's waveform, so the difference is exactly zero).",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(OUT / "novel_n_agree_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    filled = np.isfinite(grid).sum()
    print(f"  wrote novel_n_agree_heatmap.png  ({filled} cells filled of {n * n - n} off-diagonal)")


# ──────────────────────────────────────────────────────────
# 2. Deviance scaling
# ──────────────────────────────────────────────────────────

def plot_deviance_scaling(ph1, models, n_bins=8):
    d = ph1.dropna(subset=["semitones"]).copy()
    if d.empty:
        print("  skipped deviance scaling: no semitone column")
        return
    d["bin"] = pd.qcut(d["semitones"], q=min(n_bins, d["semitones"].nunique()),
                       duplicates="drop")
    centres = d.groupby("bin", observed=True)["semitones"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

    ax = axes[0]
    for m in models:
        rate = d.groupby("bin", observed=True)[f"s7__{m}"].mean() * 100
        st = MODEL_STYLE[m]
        ax.plot(centres.values, rate.values, marker=st["marker"], color=st["color"],
                lw=1.6, ms=6, label=MLABEL[m],
                ls="--" if m.startswith("wav2vec2") else "-")
    ax.set_xlabel("deviance (semitones)")
    ax.set_ylabel(f"direction-instances with S7 at FCz (%)")
    ax.set_title(f"Per-model S7 rate vs deviance (X = {DIP_UV_THRESHOLD} µV)")
    ax.set_ylim(-2, 102)
    # Curves rise from bottom-left and plateau top-right, so the free corner is bottom-right.
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")

    ax = axes[1]
    mean_n = d.groupby("bin", observed=True)["n_agree"].mean()
    sem_n = d.groupby("bin", observed=True)["n_agree"].sem()
    ax.errorbar(centres.values, mean_n.values, yerr=sem_n.values, marker="o",
                color=POOL_COLOR, lw=1.8, ms=6, capsize=3)
    ax.axhline(len(models) / 2, **REF_STYLE)
    ax.text(0.985, len(models) / 2 + 0.12, f"{len(models) / 2:g} = half the models",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=8, color=MUTED)
    ax.set_xlabel("deviance (semitones)")
    ax.set_ylabel("mean n_agree")
    ax.set_ylim(-0.2, len(models) + 0.2)
    rho, p = stats.spearmanr(d["semitones"], d["n_agree"])
    ax.set_title(f"Cross-model agreement vs deviance\nSpearman ρ = {rho:+.3f} (p = {p:.2g}, "
                 f"n = {len(d)})")

    fig.tight_layout()
    fig.savefig(OUT / "novel_deviance_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote novel_deviance_scaling.png  (ρ = {rho:+.3f})")


# ──────────────────────────────────────────────────────────
# 3. Phase-1 -> Phase-2 rank stability
# ──────────────────────────────────────────────────────────

def plot_rank_stability(ph1, ph2, models):
    merged = ph2.merge(ph1[["pair_id", "direction", "rank", "n_agree"]],
                       on=["pair_id", "direction"], suffixes=("_p2", "_p1"))
    if len(merged) < 3:
        print(f"  skipped rank stability: only {len(merged)} shared direction-instances")
        return

    # Phase-1 ranks are positions in the full 1056-instance list; re-rank within the shared
    # subset so both axes count the same population.
    x = merged["rank_p1"].rank().values
    y = merged["rank_p2"].values
    rho, p = stats.spearmanr(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    ax = axes[0]
    lim = [0, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, **REF_STYLE)
    ax.scatter(x, y, s=26, color=POOL_COLOR, alpha=0.65, linewidths=0.8, edgecolors="white")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Phase-1 rank (1 deviant, re-ranked within this subset)")
    ax.set_ylabel("Phase-2 rank (15 deviants)")
    ax.set_title(f"Rank stability across phases\nSpearman ρ = {rho:+.3f} "
                 f"(p = {p:.2g}, n = {len(merged)})")
    # Backdrop, not a border: the scatter fills the panel and this note would otherwise sit on
    # top of the points.
    ax.text(0.97, 0.04, "on the dashed line = rank unchanged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))

    ax = axes[1]
    tiers = np.arange(len(models) + 1)
    cross = pd.crosstab(merged["n_agree_p1"], merged["n_agree_p2"]).reindex(
        index=tiers, columns=tiers, fill_value=0)
    ax.set_facecolor("white"); ax.grid(False)
    im = ax.imshow(cross.values, cmap=SEQ_CMAP, origin="lower", interpolation="nearest")
    ax.set_xticks(tiers); ax.set_yticks(tiers)
    ax.set_xlabel("Phase-2 n_agree"); ax.set_ylabel("Phase-1 n_agree")
    ax.plot([-0.5, len(models) + 0.5], [-0.5, len(models) + 0.5],
            color="#bbbbbb", lw=0.8, zorder=3)
    unchanged = np.trace(cross.values) / cross.values.sum() * 100
    ax.set_title(f"n_agree tier movement\n{unchanged:.0f}% of instances keep their tier")
    for i in tiers:
        for j in tiers:
            v = cross.values[i, j]
            if v:
                # label on the cell, in ink not the ramp colour; light text only where the cell
                # is dark enough to need it
                dark = v > 0.6 * cross.values.max()
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if dark else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("direction-instances")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "novel_rank_stability.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote novel_rank_stability.png  (ρ = {rho:+.3f}, {unchanged:.0f}% tier-stable)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"reading {RESULTS}")
    ph1 = _load("phase1_ranked_directions.csv")
    ph2 = _load("phase2_final_ranking.csv")
    if ph1 is None:
        raise SystemExit("phase1_ranked_directions.csv is required -- run rank_novel_phase1.py")

    models = _models_in(ph1)
    print(f"models: {len(models)} -> {models}")
    if "f_low" not in ph1.columns:
        raise SystemExit("ranking CSV lacks frequency columns -- rerun rank_novel_phase1.py "
                         "with --grid_index")

    plot_heatmap(ph1, models)
    plot_deviance_scaling(ph1, models)
    if ph2 is None:
        print("  skipped rank stability: Phase 2 not scored yet")
    else:
        plot_rank_stability(ph1, ph2, models)


if __name__ == "__main__":
    main()
