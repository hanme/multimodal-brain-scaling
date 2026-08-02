#!/usr/bin/env python
"""Figures for the novel tone-pair search (Phases 1 and 2).

Reads outputs/results_novel_search/{phase1_ranked_directions,phase2_final_ranking}.csv, the
ranking emitted by rank_novel_phase1.py / rank_novel_phase2.py. mTRF only, FCz, X = 0.75 uV,
6 models -- the same slice the ranking is defined on.

  1. novel_n_agree_heatmap.png     n_agree over the full 43x43 frequency space. Every off-diagonal
                                   cell is one direction-instance: row = standard, column =
                                   deviant, so (i,j) and (j,i) are the two directions of one pair
                                   and asymmetry about the diagonal IS the frequency-preference
                                   effect. The diagonal is empty by design -- a same-frequency
                                   "deviant" synthesizes to the standard's waveform, so the
                                   difference is exactly zero by construction.
  2. novel_deviance_scaling.png    S7 rate vs deviance, per model + pooled. A monotone rise then
                                   plateau is the expected signature; flat means the metric is
                                   not tracking deviance.
  3. novel_rank_stability.png      Phase-1 vs Phase-2 rank on the shared direction-instances,
    + phase2_tier_migration.csv    plus the 7x7 n_agree migration matrix behind the second panel.
                                   The methodological result: whether a 4-clip screen is a valid
                                   proxy for a 32-clip evaluation.
  4. phase2_rank_shift.png         Where the movement went: the rank-shift distribution, and mean
    + phase2_rank_shift.csv        Phase-2 n_agree against Phase-1 tier. The second panel is the
                                   regression-to-the-mean check -- the 254 were SELECTED on their
                                   Phase-1 score, so a net decline is what re-measuring them has
                                   to produce whether or not anything about the stimuli changed.
  5. phase2_near_misses.csv        Every pair that reached n_agree >= 5 in BOTH directions under
                                   one deviant, tracked to what it does under 15. The row-level
                                   version of the consensus-set result.

Figures 1-2 need only Phase 1; 3-5 need both. Whatever is missing is skipped with a note, so this
is runnable as soon as Phase 1 lands.

Every figure is written twice: the PNG into --out_dir, and a vector SVG of the same figure, same
stem, into --svg_dir (default: svgs/ beside --out_dir). --no_svg writes only the PNGs.

  python aux/analysis_novel_search/plots/novel_search_plots.py
  python aux/analysis_novel_search/plots/novel_search_plots.py \
      --results_dir outputs/results_novel_search_rerun --out_dir /tmp/figs
"""
import argparse
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
from novel_search_common import (                                      # noqa: E402
    SEARCH_MODELS, DIP_UV_THRESHOLD, MIN_AGREE_PHASE2, direction_gap,
)

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
# A cell a phase never measured -- distinct from BOTH of the above, so "not measured", "excluded
# by design" and "measured, no model agreed" stay three different things on a partial grid.
UNEVALUATED_CELL = "#f4f2ee"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#cccccc",
    "xtick.color": MUTED, "ytick.color": MUTED,
    # Text goes into the SVG as glyph outlines, matplotlib's default. The alternative,
    # svg.fonttype = "none", gives smaller files and selectable text but only NAMES the font, so
    # every viewer has to have DejaVu Sans or the renderer substitutes. These figures carry µ, →,
    # ≥, −, ✓ and ρ, and a substituted font drops or reflows exactly those glyphs -- a caption
    # reading "0.75 V" or a title with a tofu box where ρ was. Outlines render identically
    # everywhere at the cost of size, which is the right trade for a figure in a memo.
    "svg.fonttype": "path",
})


# ──────────────────────────────────────────────────────────
# PNG + SVG output
# ──────────────────────────────────────────────────────────
# Every figure is written twice: the PNG the memo embeds and the PDF build reads, and a vector
# SVG twin under the same stem. The SVG destination is resolved once per run into this module
# global, so the 19 call sites across both scripts stay one line each.
_SVG_DIR = None


def default_svg_dir(out_dir):
    """`svgs/` beside the plots directory, derived rather than hardcoded.

    The committed run writes aux/analysis_novel_search/plots -> aux/analysis_novel_search/svgs;
    a test's --out_dir /tmp/figs writes /tmp/svgs. Never a subdirectory of the plots dir, which
    would put SVGs inside the tree the memo globs for PNGs.
    """
    return Path(out_dir).parent / "svgs"


def configure_svg_output(out_dir, svg_dir=None, no_svg=False):
    """Resolve where savefig_both() puts its SVGs. Call once, from main(), before any figure."""
    global _SVG_DIR
    if no_svg:
        _SVG_DIR = None
        print("  SVG output disabled (--no_svg)")
    else:
        _SVG_DIR = Path(svg_dir) if svg_dir is not None else default_svg_dir(out_dir)
        _SVG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  SVG twins -> {_SVG_DIR}")
    return _SVG_DIR


def savefig_both(fig, out_dir, name, svg_dir=None):
    """Write `name` (a .png) into out_dir, and the same figure as .svg into the SVG dir.

    bbox_inches="tight" on both, or the two crop differently and stop being interchangeable.
    """
    fig.savefig(Path(out_dir) / name, bbox_inches="tight")
    target = Path(svg_dir) if svg_dir is not None else _SVG_DIR
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{Path(name).stem}.svg", bbox_inches="tight")


def _load(results_dir, name):
    path = Path(results_dir) / name
    if not path.exists():
        print(f"  skipped: {path} not found")
        return None
    df = pd.read_csv(path)
    print(f"  loaded {path.name}: {len(df)} direction-instances")
    return df


def _models_in(df):
    return [m for m in SEARCH_MODELS if f"s7__{m}" in df.columns]


# ──────────────────────────────────────────────────────────
# 1. n_agree over the 43x43 frequency space
# ──────────────────────────────────────────────────────────

def plot_heatmap(ranked, models, out_dir=OUT, suffix="", all_freqs=None, n_deviants=1):
    """n_agree over the frequency space, at whatever coverage `ranked` carries.

    `all_freqs` pins the axes to the whole ladder so a subset heatmap is comparable to the
    full-grid one cell for cell; unmeasured cells get their own colour, distinct from the
    excluded diagonal.
    """
    freqs = sorted(all_freqs) if all_freqs is not None else sorted(
        set(ranked["f_low"]) | set(ranked["f_high"]))
    idx = {f: i for i, f in enumerate(freqs)}
    n = len(freqs)

    grid = np.full((n, n), np.nan)
    for r in ranked.itertuples(index=False):
        std, dev = (r.f_low, r.f_high) if r.direction == "regular" else (r.f_high, r.f_low)
        grid[idx[std], idx[dev]] = r.n_agree

    off_diagonal = n * n - n
    filled = int(np.isfinite(grid).sum())
    partial = filled < off_diagonal

    n_models = len(models)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.grid(False)
    cmap = plt.get_cmap(SEQ_CMAP, n_models + 1).with_extremes(
        bad=UNEVALUATED_CELL if partial else EMPTY_CELL)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, origin="lower",
                   vmin=-0.5, vmax=n_models + 0.5, interpolation="nearest")
    if partial:
        for i in range(n):
            ax.add_patch(mpl.patches.Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor=EMPTY_CELL,
                                               edgecolor="none", zorder=2))

    ticks = range(0, n, 4)
    ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
    ax.set_xticklabels([f"{freqs[i]:g}" for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([f"{freqs[i]:g}" for i in ticks])
    ax.set_xlabel("deviant frequency (Hz)")
    ax.set_ylabel("standard frequency (Hz)")
    coverage = (f"\n{filled} of {off_diagonal} off-diagonal cells evaluated "
                f"({100.0 * filled / off_diagonal:.0f}% coverage)" if partial else "")
    ax.set_title(f"Cross-model agreement over the novel frequency grid\n"
                 f"n_agree = models showing S7 at FCz (X = {DIP_UV_THRESHOLD} µV), "
                 f"of {n_models}; {n_deviants} deviant(s) per condition{coverage}", pad=12)

    cbar = fig.colorbar(im, ax=ax, ticks=range(n_models + 1), fraction=0.046, pad=0.03)
    cbar.set_label(f"n_agree (0–{n_models})")
    cbar.outline.set_visible(False)

    absence = ("Two kinds of empty cell: the mid-grey diagonal is excluded by design, the pale "
               "background was never measured\nin this phase. "
               if partial else
               "Grey diagonal = excluded by design (a same-\nfrequency deviant synthesizes "
               "to the standard's waveform, so the difference is exactly zero).")
    # Caption below the axes, not over the cells: annotation text on a dark ramp step is
    # unreadable in either ink or white.
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.text(0.5, 0.045,
             "Each off-diagonal cell is one direction-instance; (i,j) and (j,i) are the two "
             "directions of one pair, so\nasymmetry about the diagonal is the frequency-preference "
             "effect. " + absence,
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    name = f"novel_n_agree_heatmap{suffix}.png"
    savefig_both(fig, out_dir, name)
    plt.close(fig)
    print(f"  wrote {name}  ({filled} cells filled of {off_diagonal} off-diagonal)")


# ──────────────────────────────────────────────────────────
# 2. Deviance scaling
# ──────────────────────────────────────────────────────────

def plot_deviance_scaling(ranked, models, out_dir=OUT, n_bins=8, suffix="", n_deviants=1):
    d = ranked.dropna(subset=["semitones"]).copy()
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
    ax.set_title(f"Per-model S7 rate vs deviance (X = {DIP_UV_THRESHOLD} µV)\n"
                 f"{n_deviants} deviant(s) per condition, n = {len(d)} direction-instances")
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
    name = f"novel_deviance_scaling{suffix}.png"
    savefig_both(fig, out_dir, name)
    plt.close(fig)
    print(f"  wrote {name}  (ρ = {rho:+.3f})")


# ──────────────────────────────────────────────────────────
# 3. Phase-1 -> Phase-2 rank stability
# ──────────────────────────────────────────────────────────

def _shared(ph1, ph2):
    """Phase-2 rows with Phase 1's rank and tier for the SAME direction-instances."""
    return ph2.merge(ph1[["pair_id", "direction", "rank", "n_agree", "mean_uv"]],
                     on=["pair_id", "direction"], suffixes=("_p2", "_p1"))


def plot_rank_stability(ph1, ph2, models, out_dir=OUT):
    """Phase-1 vs Phase-2 rank, as its own figure.

    Split from the tier-migration matrix it used to share a canvas with: the two answer different
    questions (does the ORDER hold, does the TIER hold) and at half-width neither axis was
    legible. Titles describe what is plotted; the findings live in the memo's bullets.
    """
    merged = _shared(ph1, ph2)
    if len(merged) < 3:
        print(f"  skipped rank stability: only {len(merged)} shared direction-instances")
        return

    # Phase-1 ranks are positions in the full 1806-instance list; re-rank within the shared
    # subset so both axes count the same population.
    x = merged["rank_p1"].rank().values
    y = merged["rank_p2"].values
    rho, p = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lim = [0, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, **REF_STYLE)
    ax.scatter(x, y, s=28, color=POOL_COLOR, alpha=0.65, linewidths=0.8, edgecolors="white")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Phase-1 rank (1 deviant, re-ranked within this subset)")
    ax.set_ylabel("Phase-2 rank (15 deviants)")
    ax.set_title(f"Phase-1 rank against Phase-2 rank\n{len(merged)} shared direction-instances; "
                 f"Spearman \u03c1 = {rho:+.3f} (p = {p:.2g})", pad=10)
    ax.text(0.97, 0.04, "on the dashed line = rank unchanged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))
    fig.tight_layout()
    savefig_both(fig, out_dir, "novel_rank_stability.png")
    plt.close(fig)
    print(f"  wrote novel_rank_stability.png  (\u03c1 = {rho:+.3f})")


def plot_tier_migration(ph1, ph2, models, out_dir=OUT):
    """The 7x7 n_agree migration matrix, as its own figure."""
    merged = _shared(ph1, ph2)
    if len(merged) < 3:
        print(f"  skipped tier migration figure: only {len(merged)} shared instances")
        return
    tiers = np.arange(len(models) + 1)
    cross = pd.crosstab(merged["n_agree_p1"], merged["n_agree_p2"]).reindex(
        index=tiers, columns=tiers, fill_value=0)

    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    ax.set_facecolor("white"); ax.grid(False)
    im = ax.imshow(cross.values, cmap=SEQ_CMAP, origin="lower", interpolation="nearest")
    ax.set_xticks(tiers); ax.set_yticks(tiers)
    ax.set_xlabel("Phase-2 n_agree (15 deviants)")
    ax.set_ylabel("Phase-1 n_agree (1 deviant)")
    ax.plot([-0.5, len(models) + 0.5], [-0.5, len(models) + 0.5],
            color="#bbbbbb", lw=0.8, zorder=3)
    unchanged = np.trace(cross.values) / cross.values.sum() * 100
    ax.set_title(f"n_agree tier, Phase 1 against Phase 2\n{len(merged)} shared "
                 f"direction-instances; {unchanged:.0f}% on the diagonal", pad=10)
    for i in tiers:
        for j in tiers:
            v = cross.values[i, j]
            if v:
                dark = v > 0.6 * cross.values.max()
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if dark else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("direction-instances"); cbar.outline.set_visible(False)
    fig.tight_layout()
    savefig_both(fig, out_dir, "novel_tier_migration.png")
    plt.close(fig)
    print(f"  wrote novel_tier_migration.png  ({unchanged:.0f}% tier-stable)")


# ──────────────────────────────────────────────────────────
# 3b. The migration matrix behind that second panel, as a table
# ──────────────────────────────────────────────────────────

def table_tier_migration(ph1, ph2, models, out_dir=OUT):
    """7x7 Phase-1 x Phase-2 n_agree counts, with marginals and each row's mean outcome.

    The row means are the point. A tier whose mean Phase-2 n_agree sits BELOW its own Phase-1
    value is regressing toward the middle, which is what re-measuring a set selected on a noisy
    first score must produce. Reading the aggregate decline without these rows would attribute a
    selection effect to the stimuli.
    """
    merged = _shared(ph1, ph2)
    if len(merged) < 3:
        print(f"  skipped tier migration: only {len(merged)} shared direction-instances")
        return None
    tiers = list(range(len(models), -1, -1))
    cross = pd.crosstab(merged["n_agree_p1"], merged["n_agree_p2"]).reindex(
        index=tiers, columns=tiers, fill_value=0)
    out = cross.copy()
    out.columns = [f"phase2_{c}" for c in out.columns]
    out.insert(0, "phase1_n_agree", cross.index)
    out["n"] = cross.sum(axis=1).values
    out["mean_phase2_n_agree"] = [
        float(merged.loc[merged["n_agree_p1"] == t, "n_agree_p2"].mean()) if n else np.nan
        for t, n in zip(cross.index, out["n"])]
    out["held"] = [int(cross.loc[t, t]) for t in cross.index]
    out["fell"] = [int(cross.loc[t, [c for c in cross.columns if c < t]].sum())
                   for t in cross.index]
    out["rose"] = [int(cross.loc[t, [c for c in cross.columns if c > t]].sum())
                   for t in cross.index]
    out.round(4).to_csv(Path(out_dir) / "phase2_tier_migration.csv", index=False)
    print(f"  wrote phase2_tier_migration.csv  (held {int(out['held'].sum())}, "
          f"fell {int(out['fell'].sum())}, rose {int(out['rose'].sum())} of {len(merged)}; "
          f"mean n_agree {merged['n_agree_p1'].mean():.3f} → {merged['n_agree_p2'].mean():.3f})")
    return out


def table_rank_shift(ph1, ph2, out_dir=OUT):
    """How far instances moved between the two rankings, in ranks."""
    merged = _shared(ph1, ph2)
    if len(merged) < 3:
        print(f"  skipped rank shift: only {len(merged)} shared direction-instances")
        return None
    shift = merged["rank_p1"].rank() - merged["rank_p2"]
    rho, pval = stats.spearmanr(merged["rank_p1"].rank(), merged["rank_p2"])
    rho_t, p_t = stats.spearmanr(merged["n_agree_p1"], merged["n_agree_p2"])
    out = pd.DataFrame([{
        "n": len(merged),
        "spearman_rho_rank": rho, "spearman_p_rank": pval,
        "spearman_rho_n_agree": rho_t, "spearman_p_n_agree": p_t,
        "mean_abs_shift": shift.abs().mean(), "median_abs_shift": shift.abs().median(),
        "p90_abs_shift": shift.abs().quantile(0.90), "max_abs_shift": shift.abs().max(),
        "within_10": int((shift.abs() <= 10).sum()), "within_25": int((shift.abs() <= 25).sum()),
        "within_50": int((shift.abs() <= 50).sum()),
        "tier_held": int((merged["n_agree_p1"] == merged["n_agree_p2"]).sum()),
        "tier_rose": int((merged["n_agree_p2"] > merged["n_agree_p1"]).sum()),
        "tier_fell": int((merged["n_agree_p2"] < merged["n_agree_p1"]).sum()),
        "mean_n_agree_phase1": merged["n_agree_p1"].mean(),
        "mean_n_agree_phase2": merged["n_agree_p2"].mean(),
    }])
    out.round(4).to_csv(Path(out_dir) / "phase2_rank_shift.csv", index=False)
    print(f"  wrote phase2_rank_shift.csv  (ρ = {rho:+.3f}, mean |shift| = "
          f"{shift.abs().mean():.1f} of {len(merged)})")
    return out


def plot_rank_shift(ph1, ph2, models, out_dir=OUT):
    """Distribution of how far instances moved between the two rankings."""
    merged = _shared(ph1, ph2)
    if len(merged) < 3:
        print(f"  skipped rank-shift figure: only {len(merged)} shared instances")
        return
    shift = (merged["rank_p1"].rank() - merged["rank_p2"]).values

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.grid(True, axis="y")
    lim = int(np.ceil(np.abs(shift).max() / 10.0) * 10)
    ax.hist(shift, bins=np.arange(-lim, lim + 10, 10), color=POOL_COLOR, alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax.axvline(0, **REF_STYLE)
    ax.set_xlabel("rank shift (Phase-1 rank within the subset \u2212 Phase-2 rank; "
                  "> 0 = moved up)")
    ax.set_ylabel("direction-instances")
    ax.set_title(f"How far instances moved between the rankings\nn = {len(merged)}; "
                 f"mean |shift| = {np.abs(shift).mean():.1f} places", pad=10)
    fig.tight_layout()
    savefig_both(fig, out_dir, "phase2_rank_shift.png")
    plt.close(fig)
    print("  wrote phase2_rank_shift.png")


def plot_tier_regression(ph1, ph2, models, out_dir=OUT):
    """Mean Phase-2 n_agree against Phase-1 tier, with the identity line."""
    merged = _shared(ph1, ph2)
    if len(merged) < 3:
        print(f"  skipped tier-regression figure: only {len(merged)} shared instances")
        return
    tiers = np.arange(len(models) + 1)
    g = merged.groupby("n_agree_p1")["n_agree_p2"]
    mean, sem, n = g.mean().reindex(tiers), g.sem().reindex(tiers), g.size().reindex(tiers)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(tiers, tiers, **REF_STYLE)
    ax.errorbar(mean.index, mean.values, yerr=sem.values, marker="o", ms=7, lw=1.8,
                color=POOL_COLOR, capsize=3)
    for tt in tiers:
        if n.get(tt, 0):
            ax.annotate(f"n = {int(n[tt])}", (tt, mean[tt]), textcoords="offset points",
                        xytext=(7, -12), fontsize=8, color=MUTED)
    ax.text(0.03, 0.95, "dashed line = no change between phases", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Phase-1 n_agree (1 deviant) \u2014 the score the subset was selected on")
    ax.set_ylabel("mean Phase-2 n_agree (15 deviants)")
    ax.set_xticks(tiers); ax.set_yticks(tiers)
    ax.set_xlim(-0.4, len(models) + 0.4); ax.set_ylim(-0.4, len(models) + 0.4)
    ax.set_title(f"Mean Phase-2 agreement by Phase-1 tier\nn = {len(merged)}; overall mean "
                 f"{merged['n_agree_p1'].mean():.2f} \u2192 "
                 f"{merged['n_agree_p2'].mean():.2f}", pad=10)
    fig.tight_layout()
    savefig_both(fig, out_dir, "phase2_tier_regression.png")
    plt.close(fig)
    print("  wrote phase2_tier_regression.png")


def _bootstrap_rho_ci(a, b, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI for Spearman rho, so each correlation carries an interval.

    Seeded, so a rerun reproduces the memo's numbers rather than jittering them.
    """
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 5:
        return np.nan, np.nan, np.nan, 0
    rho = float(stats.spearmanr(d["a"], d["b"])[0])
    rng = np.random.default_rng(seed)
    av, bv = d["a"].values, d["b"].values
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        boots[i] = stats.spearmanr(av[idx], bv[idx])[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi), len(d)


def uv_stability_frame(ph1, ph2, models):
    """The two phases' scores for the shared instances, including a COMPARABLE µV quantity.

    `mean_uv` averages the agreeing models only, so once `n_agree` changes between phases the two
    phases' values are means over different model sets and a correlation between them is partly
    measuring that change of composition. `mean_uv_all6` averages all six models regardless of S7,
    is defined for every instance, and is over the same six models in both phases -- it is the
    quantity that answers "did the underlying µV response move".
    """
    uv = [f"trough_uv__{m}" for m in models]
    left = ph1[["pair_id", "direction", "rank", "n_agree", "mean_uv"] + uv].copy()
    left["mean_uv_all6"] = ph1[uv].mean(axis=1)
    merged = ph2.merge(left, on=["pair_id", "direction"], suffixes=("_p2", "_p1"))
    merged["mean_uv_all6_p2"] = ph2[uv].mean(axis=1).values
    return merged


def table_uv_rank_correlations(ph1, ph2, models, out_dir=OUT):
    """Every Phase-1 vs Phase-2 correlation the memo quotes, one row each, with a bootstrap CI."""
    m = uv_stability_frame(ph1, ph2, models)
    rows = [
        ("rank (within the shared subset)", "rank", m["rank_p1"].rank(), m["rank_p2"], "ranking"),
        ("n_agree", "n_agree", m["n_agree_p1"], m["n_agree_p2"], "ranking"),
        ("mean_uv (agreeing models only)", "mean_uv (agreeing only)",
         m["mean_uv_p1"], m["mean_uv_p2"], "µV"),
        ("mean_uv_all6 (all 6 models)", "mean_uv_all6 (all 6)",
         m["mean_uv_all6"], m["mean_uv_all6_p2"], "µV"),
    ]
    rows += [(f"trough_uv, {MLABEL[mm]}", MLABEL[mm],
              m[f"trough_uv__{mm}_p1"], m[f"trough_uv__{mm}_p2"], "per model") for mm in models]

    out = []
    for label, short, a, b, group in rows:
        rho, lo, hi, n = _bootstrap_rho_ci(a, b)
        p = float(stats.spearmanr(pd.DataFrame({"a": a, "b": b}).dropna().iloc[:, 0],
                                  pd.DataFrame({"a": a, "b": b}).dropna().iloc[:, 1])[1])
        out.append({"quantity": label, "short": short, "group": group, "spearman_rho": rho,
                    "ci_lo": lo, "ci_hi": hi, "p": p, "n": n})
    tbl = pd.DataFrame(out)
    tbl.round(6).to_csv(Path(out_dir) / "phase2_correlation_summary.csv", index=False)
    print(f"  wrote phase2_correlation_summary.csv  ({len(tbl)} quantities; "
          f"rank {tbl.iloc[0].spearman_rho:+.3f}, mean_uv_all6 {tbl.iloc[3].spearman_rho:+.3f})")
    return tbl


def plot_uv_scatter(ph1, ph2, models, out_dir=OUT):
    """Phase-1 against Phase-2 mean_uv_all6 -- the comparable µV quantity, as its own figure."""
    m = uv_stability_frame(ph1, ph2, models)
    d = m.dropna(subset=["mean_uv_all6", "mean_uv_all6_p2"])
    rho = float(stats.spearmanr(d["mean_uv_all6"], d["mean_uv_all6_p2"])[0])
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    lim = [min(d["mean_uv_all6"].min(), d["mean_uv_all6_p2"].min()) - 0.15,
           max(d["mean_uv_all6"].max(), d["mean_uv_all6_p2"].max()) + 0.15]
    ax.plot(lim, lim, **REF_STYLE)
    sc = ax.scatter(d["mean_uv_all6"], d["mean_uv_all6_p2"], c=d["n_agree_p2"], cmap=SEQ_CMAP,
                    vmin=-0.5, vmax=n_models + 0.5, s=30, linewidths=0.4, edgecolors="#5a5a5a")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Phase-1 mean_uv_all6 (1 deviant, \u00b5V)")
    ax.set_ylabel("Phase-2 mean_uv_all6 (15 deviants, \u00b5V)")
    ax.set_title(f"Predicted trough depth, Phase 1 against Phase 2\nmean over all {n_models} "
                 f"models; {len(d)} instances; Spearman \u03c1 = {rho:+.3f}", pad=10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, ticks=range(n_models + 1))
    cb.set_label("Phase-2 n_agree"); cb.outline.set_visible(False)
    fig.tight_layout()
    savefig_both(fig, out_dir, "phase2_uv_scatter.png")
    plt.close(fig)
    print(f"  wrote phase2_uv_scatter.png  (\u03c1 = {rho:+.3f})")


def plot_correlation_summary(ph1, ph2, models, out_dir=OUT):
    """Every Phase-1 vs Phase-2 correlation on one axis, with bootstrap CIs."""
    tbl = table_uv_rank_correlations(ph1, ph2, models, out_dir)
    colors = {"ranking": "#0072B2", "\u00b5V": "#D55E00", "per model": "#9a9a9a"}
    t = tbl.sort_values("spearman_rho").reset_index(drop=True)
    y = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.grid(True, axis="x")
    ax.hlines(y, t["ci_lo"], t["ci_hi"], color=[colors[g] for g in t["group"]], lw=2.2,
              alpha=0.45)
    ax.scatter(t["spearman_rho"], y, s=46, color=[colors[g] for g in t["group"]], zorder=3,
               linewidths=0.5, edgecolors="white")
    for yi, r in zip(y, t["spearman_rho"]):
        ax.annotate(f"{r:+.3f}", (r, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(t["short"], fontsize=8.8)
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.set_xlim(0.72, 1.03)
    ax.set_xlabel("Spearman \u03c1 between the phases (bars = bootstrap 95% CI, 2000 samples)")
    ax.set_title(f"Phase-1 vs Phase-2 agreement, by quantity\n"
                 f"n = {int(t['n'].max())} shared direction-instances", pad=10)
    handles = [Line2D([], [], color=c, lw=2.6, label=g) for g, c in colors.items()]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    savefig_both(fig, out_dir, "phase2_correlation_summary.png")
    plt.close(fig)
    print("  wrote phase2_correlation_summary.png")
    return tbl


def plot_uv_bland_altman(ph1, ph2, models, out_dir=OUT):
    """Agreement of the two phases' µV in absolute terms, not just in rank order."""
    m = uv_stability_frame(ph1, ph2, models)
    d = m.dropna(subset=["mean_uv_all6", "mean_uv_all6_p2"])
    mean_uv = (d["mean_uv_all6"] + d["mean_uv_all6_p2"]) / 2.0
    diff = d["mean_uv_all6_p2"] - d["mean_uv_all6"]
    bias, sd = float(diff.mean()), float(diff.std())

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhline(0, color="#bbbbbb", lw=0.9, zorder=1)
    ax.scatter(mean_uv, diff, s=24, color=POOL_COLOR, alpha=0.6, linewidths=0)
    ax.axhline(bias, color="#c1272d", lw=1.3, zorder=3)
    for k in (1.96, -1.96):
        ax.axhline(bias + k * sd, color="#c1272d", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.01, bias, f"mean {bias:+.3f} \u00b5V", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8.5, color="#c1272d",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    ax.text(0.01, bias + 1.96 * sd, "\u00b11.96 SD", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8, color="#c1272d",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    ax.set_xlabel("mean of the two phases (\u00b5V)")
    ax.set_ylabel("Phase 2 \u2212 Phase 1 (\u00b5V)")
    ax.set_title(f"Bland-Altman: how the two phases' \u00b5V differ\n"
                 f"n = {len(d)}; positive = shallower on 15 deviants", pad=10)
    fig.tight_layout()
    savefig_both(fig, out_dir, "phase2_uv_bland_altman.png")
    plt.close(fig)
    print(f"  wrote phase2_uv_bland_altman.png  (bias {bias:+.3f} \u00b5V, SD {sd:.3f})")


def table_near_misses(ph1, ph2, out_dir=OUT, tier=MIN_AGREE_PHASE2):
    """Pairs that cleared `tier` in BOTH directions under one deviant, tracked under 15.

    Phase 1's consensus set was empty, so these were the closest thing it produced to a symmetric
    deviance response. Whether they survive a 15-deviant mean is the row-level version of the
    consensus-set question, and the one result Phase 2 can settle that Phase 1 could not.
    """
    g1 = direction_gap(ph1)
    near = g1[(g1["n_agree_regular"] >= tier) & (g1["n_agree_counter"] >= tier)]
    if near.empty:
        print(f"  no Phase-1 pair reached n_agree >= {tier} in both directions; nothing to track")
        return None
    g2 = direction_gap(ph2[ph2["pair_id"].isin(near["pair_id"])]).set_index("pair_id")
    meta = ph1.drop_duplicates("pair_id").set_index("pair_id")

    rows = []
    for _, r in near.iterrows():
        pid = int(r["pair_id"])
        got = g2.loc[pid] if pid in g2.index else None
        rows.append({
            "pair_id": pid,
            "f_low": meta.loc[pid, "f_low"] if "f_low" in meta.columns else np.nan,
            "f_high": meta.loc[pid, "f_high"] if "f_high" in meta.columns else np.nan,
            "semitones": meta.loc[pid, "semitones"] if "semitones" in meta.columns else np.nan,
            "phase1_n_agree_regular": int(r["n_agree_regular"]),
            "phase1_n_agree_counter": int(r["n_agree_counter"]),
            "phase1_mean_uv_regular": r["mean_uv_regular"],
            "phase1_mean_uv_counter": r["mean_uv_counter"],
            "phase2_n_agree_regular": int(got["n_agree_regular"]) if got is not None else np.nan,
            "phase2_n_agree_counter": int(got["n_agree_counter"]) if got is not None else np.nan,
            "phase2_mean_uv_regular": got["mean_uv_regular"] if got is not None else np.nan,
            "phase2_mean_uv_counter": got["mean_uv_counter"] if got is not None else np.nan,
            "held_both_ways": (bool(got["n_agree_regular"] >= tier
                                    and got["n_agree_counter"] >= tier)
                               if got is not None else False),
        })
    out = pd.DataFrame(rows).sort_values("pair_id")
    out.round(4).to_csv(Path(out_dir) / "phase2_near_misses.csv", index=False)
    print(f"  wrote phase2_near_misses.csv  ({len(out)} pair(s) at >= {tier} both ways in "
          f"Phase 1; {int(out['held_both_ways'].sum())} still there under 15 deviants)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are the committed locations; overridable so a re-run can be rendered without
    # clobbering the figures the memo links to (and so this is testable).
    p.add_argument("--results_dir", default=str(RESULTS),
                   help="directory holding the phase1/phase2 ranking CSVs")
    p.add_argument("--out_dir", default=str(OUT), help="where to write the PNGs")
    p.add_argument("--svg_dir", default=None,
                   help="where to write the SVG twin of every figure (default: svgs/ beside "
                        "--out_dir). Same stem, .svg instead of .png")
    p.add_argument("--no_svg", action="store_true",
                   help="write only the PNGs")
    p.add_argument("--skip_phase1_figures", action="store_true",
                   help="emit only the cross-phase outputs. Figures 1-2 are committed Phase-1 "
                        "artifacts that a Phase-2 run has no reason to rewrite")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_svg_output(out_dir, args.svg_dir, args.no_svg)
    print(f"reading {args.results_dir}")
    ph1 = _load(args.results_dir, "phase1_ranked_directions.csv")
    ph2 = _load(args.results_dir, "phase2_final_ranking.csv")
    if ph1 is None:
        raise SystemExit("phase1_ranked_directions.csv is required -- run rank_novel_phase1.py")

    models = _models_in(ph1)
    print(f"models: {len(models)} -> {models}")
    if "f_low" not in ph1.columns:
        raise SystemExit("ranking CSV lacks frequency columns -- rerun rank_novel_phase1.py "
                         "with --grid_index")

    if not args.skip_phase1_figures:
        plot_heatmap(ph1, models, out_dir, n_deviants=1)
        plot_deviance_scaling(ph1, models, out_dir, n_deviants=1)
    if ph2 is None:
        print("  skipped the cross-phase outputs: Phase 2 not scored yet")
    else:
        # The n_agree heatmap for Phase 2, on the full-grid axes so it reads against its
        # Phase-1 counterpart. No Phase-2 deviance figure: the selection removed the
        # small-deviance instances the correlation lived in, so the curve measures the
        # selection rather than the stimuli.
        all_freqs = sorted(set(ph1["f_low"]) | set(ph1["f_high"]))
        plot_heatmap(ph2, models, out_dir, suffix="_phase2", all_freqs=all_freqs, n_deviants=15)
        plot_rank_stability(ph1, ph2, models, out_dir)
        plot_tier_migration(ph1, ph2, models, out_dir)
        table_tier_migration(ph1, ph2, models, out_dir)
        table_rank_shift(ph1, ph2, out_dir)
        plot_uv_scatter(ph1, ph2, models, out_dir)
        plot_correlation_summary(ph1, ph2, models, out_dir)
        plot_uv_bland_altman(ph1, ph2, models, out_dir)
        plot_rank_shift(ph1, ph2, models, out_dir)
        plot_tier_regression(ph1, ph2, models, out_dir)


if __name__ == "__main__":
    main()
