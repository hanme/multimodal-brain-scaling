#!/usr/bin/env python
"""Results for the novel tone-pair search: every table and figure the memo cites, either phase.

Reads only files already on disk -- the scored slice, the grid index, the prediction HDF5s and
the literature screen. Re-runs no mapping, layer selection, extraction or in-silico step.

The file keeps its Phase-1 name because the memo's Sections 2-5 cite it by that path, but
**--phase 2** renders the same tables and figures from `phase2_mmn_s7_roi.csv` under a `phase2_`
prefix, so neither phase can clobber the other's committed output. Both phases rank through the
same `novel_search_common` criteria; the only difference is that a Phase-2 `deviant_mean` is an
average over 15 deviant realizations where a Phase-1 one is a single draw.

Emits, into --out_dir under `--prefix` (each figure beside the CSV of the table it draws):

  <p>_agreement_tiers.csv           items 1+2  n_agree tiers at X = 0.75 and X = 0.50, side by
                                               side, in direction-instances AND in pairs
  <p>_strong_waveforms_<k>.png      item 3     uV difference wave at FCz for every pair with a
                                               direction at n_agree >= 5; both directions
                                               overlaid, per-model traces
  <p>_direction_waveforms_<k>.png              the same panels collapsed to one line per
                                               direction, in z (averageable across models)
  <p>_strong_waveforms_1__<model>.png          the combined panel de-overlaid, one figure per
                                               model, for chunk --wave_chunk only. Shares a
                                               y-axis, which the combined figure cannot
  <p>_model_uv_box.png/.csv         item 4     per-model trough_uv over the n_agree >= 5 set
  <p>_top30.csv                     item 5     the top 30 direction-instances
  <p>_mean_uv_heatmap.png/.csv      item 6     43x43 grid coloured by mean trough_uv over ALL six
    + <p>_frequency_stripes.csv                models (mean_uv_all6, NOT the ranking's mean_uv).
                                               Carries its own coverage: at Phase-2 coverage only
                                               the evaluated cells are filled and the figure says
                                               so on its face, and the stripe marginals carry the
                                               n they are averaged over
  <p>_deviance_octiles.csv          item 6     mean n_agree by octile of semitone distance, with
                                               Spearman rho(semitones, n_agree)
  <p>_frequency_involvement.csv     item 6     per frequency: mean n_agree and mean_uv_all6 when
    + <p>_deviance_and_frequency.png           it is the standard and when it is the deviant, each
                                               with its n. The presentation that survives a
                                               subset where a 43x43 heatmap does not
  <p>_model_dissent.csv             item 4     per model: marginal S7 rate, and how often it is
                                               the SOLE dissenter on an otherwise-unanimous row
  <p>_ranking_structure.png         item 7     four panels on how the ranking is structured:
    + <p>_cutoff_curve.csv                     yield at every cutoff, the rank decay (elbow or
    + <p>_direction_asymmetry.csv              not), where the strong pairs sit in the grid, and
                                               direction asymmetry. --cost_overlay adds a CHF
                                               axis; off by default, since the results memo
                                               carries no costing
  <p>_novel_vs_literature.png       item 7     novel vs literature tier distributions against the
    + <p>_novel_vs_literature.csv              independent-models chance baseline
    + <p>_chance_baseline.csv

Usage:
    python aux/analysis_novel_search/plots/phase1_results.py
    python aux/analysis_novel_search/plots/phase1_results.py --phase 2 --skip_literature
    python aux/analysis_novel_search/plots/phase1_results.py \
        --results_dir outputs/results_novel_search_rerun --out_dir /tmp/figs
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy import stats
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE
RESULTS = REPO / "outputs/results_novel_search"
PREDICTIONS = REPO / "outputs/insilico_mmn_predictions_novel"
LITERATURE = REPO / "outputs/results_24freq_7models/mmn_s7_roi.csv"
LIT_META = REPO / "data/metadata/literature_frequency_intensity_duration_metadata.csv"

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from novel_search_common import (                                      # noqa: E402
    SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD, CHF_PER_PAIR_PHASE2,
    load_scored, rank_directions, agreement_tiers, expected_tier_counts,
    marginal_s7_rates, pair_cutoff_curve, direction_gap,
)
from analyze_mmn_criteria_s5_s6 import uv_diff_wave                    # noqa: E402
from analyze_mmn_criteria import compute_z_diff                        # noqa: E402
# Palette, labels and rcParams come from the sibling figure script rather than a third copy --
# the two sets of figures sit in the same memo and a drift between them would read as meaning.
from novel_search_plots import (                                       # noqa: E402
    MODEL_STYLE, MLABEL, POOL_COLOR, INK, MUTED, REF_STYLE, EMPTY_CELL, SEQ_CMAP,
)

STRONG_TIER = 5               # "strong" = a direction-instance with n_agree >= this
MMN_WINDOW = (100.0, 240.0)   # the scoring window, shaded in the waveform panels
N_TOP = 50
PANELS_PER_FIG = (5, 6)       # rows x cols of the waveform small-multiples
N_WAVE = 30                   # waveform figures cover the top N pairs of the ranking
# Top-X cut points for the consensus heatmap, per phase -- the ladders differ because the two
# rankings differ in size by 7x. Over the whole 1806-instance grid the models only begin to
# overlap at the far end, so Phase 1 starts at 10 and runs to 300. Over the 254 selected
# instances a top-10 cut is already 4% of the set, so Phase 2 resolves 10-100 in tens -- the
# range where models actually begin to overlap -- and then coarsens to 250. Thresholds above the number of instances in a ranking are dropped rather than
# rendered as saturated columns.
CONSENSUS_TOP_X = (10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300)
CONSENSUS_TOP_X_PHASE2 = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                          100, 125, 150, 175, 200, 250)
# Multi-hue sequential, distinct from the Blues used for n_agree elsewhere: this figure counts
# STIMULI, not agreeing models, and reusing the n_agree ramp would imply they are the same scale.
CONSENSUS_CMAP = "YlGnBu"

# Per-phase defaults. The scored CSV, the output prefix, and how many deviant realizations each
# condition's mean is built from -- the last is asserted against the prediction HDF5s before any
# waveform is drawn, because a Phase-1 trace rendered under a phase2_ name would be a silent lie.
PHASE = {
    1: dict(s7="phase1_mmn_s7_roi.csv", prefix="phase1", n_deviants=1,
            top_x=CONSENSUS_TOP_X),
    2: dict(s7="phase2_mmn_s7_roi.csv", prefix="phase2", n_deviants=15,
            top_x=CONSENSUS_TOP_X_PHASE2),
}

# Committed electrode layer per model (§1.5) -- the <layer> in the prediction HDF5 filename.
LAYER = {
    "whisper-tiny": "blocks.0", "whisper-base": "blocks.0", "whisper-small": "blocks.1",
    "whisper-medium": "blocks.12", "wav2vec2-medium": "encoder.layers.2",
    "wav2vec2-large": "encoder.layers.12",
}

# mean_uv_all6 straddles zero over the full grid, so the heatmap needs a diverging ramp with a
# neutral midpoint at 0 -- a sequential ramp would put "no response" and "weak MMN" on the same
# visual step. "RdBu_r" puts BLUE at the negative end, so a strong MMN (a deep negative trough)
# reads blue and a positive excursion (no trough at all) reads red.
DIV_CMAP = "RdBu_r"
# Fallback when every cell is negative: dark blue = most negative = strongest, same polarity.
SEQ_NEG_CMAP = "Blues_r"

# A cell this phase never measured. Must be distinguishable from BOTH the excluded diagonal
# (EMPTY_CELL, mid grey) and the ramp's near-neutral step, so "not measured", "excluded by design"
# and "measured, no response" stay three different things on a partially covered grid.
UNEVALUATED_CELL = "#f4f2ee"

# Trace palette for the waveform panels ONLY (item 3). Twelve traces share a panel there, and
# Okabe-Ito's slot for wav2vec2-large is pure black, which at that density reads as the subject
# of the figure rather than one series of six. Lightened to a mid grey: still separated from
# every hue by lightness alone, so the CVD-safety of the set is preserved, and still clearly
# darker than the 0.75-lightness gridline greys it sits on. Every other figure -- the boxplot
# here, and novel_search_plots.py's -- keeps the canonical MODEL_STYLE.
# wav2vec2-large's canonical slot is pure black, which is now the CROSS-MODEL MEAN's colour.
# Re-hued to a violet that is distinct from all five other model hues and from the ink mean, so
# "one model" and "the mean of six" can never be confused.
WAVE_MODEL_STYLE = {**MODEL_STYLE, "wav2vec2-large": dict(color="#8A5AC8", marker="X")}
WAVE_LW, WAVE_ALPHA = 1.05, 1.0        # darker/heavier than the 0.85 lw, 0.9 alpha default
MEAN_LW = 2.0                          # the cross-model mean, drawn over the per-model traces


# ──────────────────────────────────────────────────────────
# Loading and invariants
# ──────────────────────────────────────────────────────────

def load_phase(results_dir, s7_name=PHASE[1]["s7"], x=DIP_UV_THRESHOLD, models=None):
    """The scored slice and its ranking at floor X, with every documented invariant asserted.

    The invariants are written in terms of the pairs the slice actually carries, not a hardcoded
    903, so a Phase-2 slice over a selected subset is checked exactly as strictly as a Phase-1
    slice over the whole grid.
    """
    models = list(models or SEARCH_MODELS)
    s7_csv = Path(results_dir) / s7_name
    grid = Path(results_dir) / "grid_index.csv"
    scored = load_scored(s7_csv, models=models, roi=ROI, mapping=MAPPING, x=x)

    # Asserted, not assumed: a silently short slice would change every count below.
    n_pairs = scored["pair_id"].nunique()
    assert len(scored) == 2 * n_pairs * len(models), (
        f"X={x}: {len(scored)} rows in the {ROI}/{MAPPING} slice, expected "
        f"{2 * n_pairs * len(models)} ({n_pairs} pairs x 2 directions x {len(models)} models)")
    assert set(scored["model"]) == set(models), f"X={x}: models {sorted(set(scored['model']))}"
    assert not (scored["s7"] & ~scored["s2"]).any(), "S7 must be a subset of S2"
    per_pair = scored.groupby(["model", "pair_id"])["direction"].nunique()
    assert (per_pair == 2).all(), (
        f"X={x}: pair(s) {sorted(per_pair[per_pair != 2].index.get_level_values(1).unique())[:5]} "
        f"do not have both directions")

    ranked = rank_directions(scored, grid_index=str(grid), models=models)
    assert len(ranked) == 2 * n_pairs and ranked["mean_uv"].isna().sum() == (
        ranked["n_agree"] == 0).sum(), "mean_uv must be defined exactly where n_agree > 0"

    # mean over ALL six models -- a different quantity from the ranking's mean_uv (agreeing
    # models only), so it carries a different name everywhere it is used.
    ranked["mean_uv_all6"] = ranked[[f"trough_uv__{m}" for m in models]].mean(axis=1)

    # S2 per model, carried alongside S7. A model that satisfies S2 but not S7 produced an
    # MMN-SHAPED trough that simply did not reach the µV floor -- that is the dissent the
    # per-model figure exists to show, and it is invisible in a frame that keeps only S7.
    s2 = scored.pivot_table(index=["pair_id", "direction"], columns="model", values="s2",
                            aggfunc="first").reindex(columns=models).astype("boolean")
    s2 = s2.fillna(False).astype(bool).reset_index()
    ranked = ranked.merge(s2.rename(columns={m: f"s2__{m}" for m in models}),
                          on=["pair_id", "direction"], how="left")
    for m in models:
        assert not (ranked[f"s7__{m}"] & ~ranked[f"s2__{m}"]).any(), f"S7 without S2 for {m}"
    print(f"  X = {x:.2f}: {len(scored)} scored rows -> {len(ranked)} direction-instances "
          f"over {n_pairs} pairs, {len(models)} models")
    return scored, ranked


def _uv_cols(models):
    return [f"trough_uv__{m}" for m in models]


# ──────────────────────────────────────────────────────────
# Items 1 + 2. Agreement tiers at both µV floors
# ──────────────────────────────────────────────────────────

def table_agreement_tiers(ranked, models, out_dir, prefix="phase1"):
    """n_agree tiers at the single reporting floor X = 0.75.

    The search reports one µV floor. An earlier revision carried X = 0.50 alongside as a
    sensitivity check; it moved every count and no conclusion, so it is no longer computed --
    two floors in one table invited the counts to be read against each other when only one of
    them defines the search.
    """
    tbl = agreement_tiers(ranked, models)
    tbl.to_csv(Path(out_dir) / f"{prefix}_agreement_tiers.csv", index=False)

    n_dir, n_pair = len(ranked), ranked["pair_id"].nunique()
    print(f"\n  agreement tiers at X = {DIP_UV_THRESHOLD} µV "
          f"(of {n_dir} direction-instances / {n_pair} pairs)")
    print(f"  {'tier':>4} | {'dirs':>7} {'%':>7} {'both':>6} {'either':>7}")
    for _, r in tbl.iterrows():
        print(f"  {r['n_agree']:>4.0f} | {r['directions']:>7.0f} "
              f"{r['pct_directions']:>6.1f}% {r['pairs_both']:>6.0f} {r['pairs_either']:>7.0f}")
    print(f"  wrote {prefix}_agreement_tiers.csv")
    return tbl


# ──────────────────────────────────────────────────────────
# Item 3. Waveforms for the strong pairs
# ──────────────────────────────────────────────────────────

def _n_deviants_in(h5):
    """The `n_deviants` attribute of the first method group, or None if the file carries none."""
    for key in h5:
        if key.startswith("method_") and "n_deviants" in h5[key].attrs:
            return int(h5[key].attrs["n_deviants"])
    return None


def load_fcz_waves(pair_ids, models, predictions_root=PREDICTIONS, roi=ROI,
                   expect_n_deviants=None):
    """{(pair_id, direction): {model: (time_ms, uv, z)}} at the reporting electrode.

    Read from the prediction HDF5s, not the scored CSV: the CSV keeps one number per trace
    (the trough) and the point of the figure is the shape around it.

    Both traces are returned because they answer different questions. `uv` is the
    mean-only baseline-corrected difference wave in microvolts -- interpretable per model, but
    NOT comparable across models (Caveat 2). `z` is the same difference baseline-z-scored, which
    is the trace the S2 verdict is actually computed on and the only one of the two that can be
    averaged over models without mixing incomparable scales.

    `expect_n_deviants` gates on the HDF5's own `n_deviants` attribute. The two phases write the
    same group names into the same tree, so a Phase-1 prediction file left in place would happily
    render a 1-deviant trace under a phase2_ filename -- a figure that is wrong in a way no reader
    could detect. Mismatching models are skipped loudly instead.
    """
    waves = {}
    for m in models:
        path = Path(predictions_root) / m / f"electrode_predictions__{LAYER[m]}.h5"
        if not path.exists():
            print(f"  skipped waveforms for {m}: {path} not found")
            continue
        with h5py.File(path, "r") as h5:
            names = [t.decode() if hasattr(t, "decode") else t for t in h5["electrodes"][:]]
            if roi not in names:
                print(f"  skipped waveforms for {m}: no {roi} electrode")
                continue
            if expect_n_deviants is not None:
                got = _n_deviants_in(h5)
                if got is not None and got != expect_n_deviants:
                    print(f"  SKIPPED waveforms for {m}: {path.name} holds n_deviants = {got}, "
                          f"this phase needs {expect_n_deviants}. These predictions are from the "
                          f"other phase; drawing them here would mislabel the figure.")
                    continue
            col = names.index(roi)
            for pid in pair_ids:
                for direction, key in (("regular", f"method_{pid}"),
                                       ("counter", f"method_{pid}_counter")):
                    g = h5.get(key)
                    if g is None:
                        continue
                    t = g["time_ms"][:].astype(float)
                    std = g["standard"][:].astype(float)
                    dev = g["deviant_mean"][:].astype(float)
                    soa = float(g.attrs["soa_ms"])
                    uv = uv_diff_wave(t, std, dev, soa, names, MAPPING)
                    z = compute_z_diff(t, std, dev, soa)
                    waves.setdefault((pid, direction), {})[m] = (t, uv[:, col], z[:, col])
    return waves


def top_pairs(ranked, n=N_WAVE):
    """The first `n` distinct pair ids in rank order.

    The waveform figures cover the top of the ranking rather than the whole n_agree >= 5 tier:
    127 pairs over four chunked figures was a contact sheet nobody read, and the pairs that carry
    a claim are the ones at the top. Shared by all three waveform figures so chunk k means the
    same pairs in every one of them.
    """
    return ranked["pair_id"].drop_duplicates().head(n).tolist()


def _cross_model_mean(per_model, idx, lo=-120, hi=460, window=MMN_WINDOW):
    """(time, mean) across models on the samples they SHARE, or None.

    The models do not always carry identical in-window grids. On the literature stimuli with a
    short SOA the wav2vec2 pair runs one 20 ms sample shorter at the tail than the whisper models
    -- same step, same start, fewer samples. Averaging those by position would silently pair
    different latencies, so the mean is taken over the intersection of the time grids instead.
    That is exact, keeps all six models, and only ever shortens the trace at its edges.

    Raises if the shared grid no longer covers the MMN scoring window, which would mean the mean
    is being drawn over samples the S7 verdict never saw.
    """
    if not per_model:
        return None
    sliced = {}
    for m, tup in per_model.items():
        tt = tup[0]
        sel = (tt >= lo) & (tt <= hi)
        # Round to the nearest microsecond for the key: float time stamps that differ in the
        # last bit are the same sample, and should not fall out of the intersection.
        sliced[m] = (np.round(tt[sel], 3), tup[idx][sel])

    shared = None
    for grid, _ in sliced.values():
        shared = grid if shared is None else np.intersect1d(shared, grid)
    if shared is None or len(shared) == 0:
        raise SystemExit("models share no in-window time samples; a cross-model mean is undefined")
    if not (shared.min() <= window[0] and shared.max() >= window[1]):
        raise SystemExit(f"the models' shared time grid ({shared.min():.1f}–{shared.max():.1f} ms) "
                         f"does not cover the {window[0]:g}–{window[1]:g} ms scoring window")

    stack = [vals[np.isin(grid, shared)] for grid, vals in sliced.values()]
    return shared, np.vstack(stack).mean(axis=0)


def plot_strong_waveforms(ranked, models, out_dir, n=N_WAVE,
                          predictions_root=PREDICTIONS, prefix="phase1", expect_n_deviants=None,
                          lit=None):
    """One panel per top-ranked pair: the six per-model traces, regular direction only.

    **Regular direction only.** The counter traces doubled the ink for a contrast this figure is
    not the right place to make; the direction contrast lives in the direction-collapsed figure.

    **Top `n` pairs, not the whole n_agree >= 5 tier.** 127 pairs over four figures was a contact
    sheet; the pairs that carry a claim are the ones at the top of the ranking.

    **No cross-model mean.** A mean over six models whose µV scales differ ~5x (Caveat 2) is
    dominated by the largest of them, so the ink line was drawing attention to whisper-medium and
    wav2vec2-large rather than to the panel. The averageable version of that idea is the z-scored
    direction figure.
    """
    pairs = top_pairs(ranked, n)
    if not pairs:
        print("  skipped waveforms: ranking is empty")
        return []
    geo = ranked.drop_duplicates("pair_id").set_index("pair_id")[["f_low", "f_high"]]
    waves = load_fcz_waves(pairs, models, predictions_root,
                           expect_n_deviants=expect_n_deviants)
    if not waves:
        print("  skipped waveforms: no prediction HDF5s readable")
        return []

    nrow, ncol = PANELS_PER_FIG
    per_fig = nrow * ncol
    n_figs = math.ceil(len(pairs) / per_fig)
    lo, hi = MMN_WINDOW
    written = []

    for k in range(n_figs):
        chunk = pairs[k * per_fig:(k + 1) * per_fig]
        rows = nrow
        # sharey=False deliberately. The six models' predicted µV scales differ by ~5x
        # (Caveat 2), so one shared axis is set by wav2vec2-large and flattens the four whisper
        # traces into a line at zero -- the shape the panel exists to show.
        fig, axes = plt.subplots(rows, ncol, figsize=(2.45 * ncol, 2.05 * rows),
                                 sharex=True, sharey=False)
        axes = np.atleast_1d(axes).ravel()

        panels = [(pid, waves, f"method_{pid}: {geo.loc[pid, 'f_low']:g} → "
                                f"{geo.loc[pid, 'f_high']:g} Hz") for pid in chunk]

        for ax, (pid, src, title) in zip(axes, panels):
            ax.grid(False)
            ax.axvspan(lo, hi, color="#eef2f7", zorder=0)
            ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
            for m, (tt, uv, _z) in src.get((pid, "regular"), {}).items():
                sel = (tt >= -120) & (tt <= 460)
                ax.plot(tt[sel], uv[sel], color=WAVE_MODEL_STYLE[m]["color"], lw=WAVE_LW,
                        alpha=WAVE_ALPHA, zorder=2)
            ax.set_title(title, fontsize=7.5, pad=3)
            ax.tick_params(labelsize=6.5)
            ax.set_xlim(-120, 460)
        for ax in axes[len(panels):]:
            ax.set_visible(False)

        # Row labels: the leftmost panel of the literature row is named, so the tint is explained
        # inside the figure rather than only in the caption.
        for ax in axes[:len(panels)]:
            if ax.get_subplotspec().rowspan.start == (len(panels) - 1) // ncol:
                ax.set_xlabel("time (ms)", fontsize=8)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("µV", fontsize=8)

        handles = [Line2D([], [], color=WAVE_MODEL_STYLE[m]["color"], lw=1.6, label=MLABEL[m])
                   for m in models]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.012))
        fig.suptitle(f"FCz µV difference wave, top {len(pairs)} pairs of the ranking — "
                     f"regular direction (f_low → f_high) only"
                     + (f"   —   figure {k + 1} of {n_figs}" if n_figs > 1 else "") + "\n"
                     f"Each panel autoscales: the models' µV scales are not comparable, so read "
                     f"shape, never one trace's depth against another's",
                     fontsize=9.5, y=1.008)
        fig.tight_layout(rect=(0, 0.030, 1, 0.986))
        name = f"{prefix}_strong_waveforms_{k + 1}.png"
        fig.savefig(Path(out_dir) / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)
        print(f"  wrote {name}  ({len(chunk)} pairs)")

    print(f"  shaded band = the {lo:g}–{hi:g} ms MMN scoring window; "
          f"{len(pairs)} pairs over {n_figs} figure(s)")
    return written


# Direction-only variant: one regular line and one counter line per panel.
DIR_STYLE = {
    "regular": dict(color="#0072B2", ls="-", label="regular (f_low → f_high)"),
    "counter": dict(color="#D55E00", ls=(0, (3, 2)), label="counter (f_high → f_low)"),
}
# Fixed y-window for the direction panels. A common window across 127 panels is what makes them
# comparable to each other; autoscaling let a handful of wide ±1 SD bands set a range that
# flattened the mean lines everywhere else. Traces are CLIPPED, not rescaled -- the excursions
# that leave the window are in the per-model figures and in the CSVs.
DIRECTION_YLIM = (-2.0, 2.0)


def plot_direction_waveforms(ranked, models, out_dir, n=N_WAVE,
                             predictions_root=PREDICTIONS, units="z", ylim=DIRECTION_YLIM,
                             prefix="phase1", expect_n_deviants=None):
    """The same panels with the six model traces collapsed to one line per direction.

    units="z" averages the BASELINE-Z-SCORED difference wave -- the trace the S2 verdict is
    computed on, and the only one of the two that may be averaged over models without mixing
    incomparable scales. units="uv" averages raw microvolts instead, which contradicts Caveat 2.

    No ±1 SD band: it was the dominant ink on most panels, and the spread is legible directly in
    the per-model figures.
    """
    if units not in ("z", "uv"):
        raise SystemExit(f"--wave_units must be 'z' or 'uv', got {units!r}")
    pairs = top_pairs(ranked, n)
    if not pairs:
        print("  skipped direction waveforms: ranking is empty")
        return []
    geo = ranked.drop_duplicates("pair_id").set_index("pair_id")[["f_low", "f_high"]]
    waves = load_fcz_waves(pairs, models, predictions_root,
                           expect_n_deviants=expect_n_deviants)
    if not waves:
        print("  skipped direction waveforms: no prediction HDF5s readable")
        return []

    idx = 1 if units == "uv" else 2                    # (t, uv, z)
    ylabel = "µV (mean of 6 models)" if units == "uv" else "z (mean of 6 models)"

    nrow, ncol = PANELS_PER_FIG
    per_fig = nrow * ncol
    n_figs = math.ceil(len(pairs) / per_fig)
    lo, hi = MMN_WINDOW
    written = []

    for k in range(n_figs):
        chunk = pairs[k * per_fig:(k + 1) * per_fig]
        rows = nrow
        # A shared, fixed y-window is safe in z and worth having: with one normalised line per
        # direction there is no cross-model scale to flatten, and a common axis lets the panels
        # be compared to each other. In raw µV the scales differ ~5x, so that phase autoscales.
        share = units == "z"
        fig, axes = plt.subplots(rows, ncol, figsize=(2.45 * ncol, 2.05 * rows),
                                 sharex=True, sharey=share)
        axes = np.atleast_1d(axes).ravel()

        panels = [(pid, waves, f"method_{pid}: {geo.loc[pid, 'f_low']:g} → "
                                f"{geo.loc[pid, 'f_high']:g} Hz") for pid in chunk]

        for ax, (pid, src, title) in zip(axes, panels):
            ax.grid(False)
            ax.axvspan(lo, hi, color="#eef2f7", zorder=0)
            ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
            for direction, st in DIR_STYLE.items():
                got = _cross_model_mean(src.get((pid, direction), {}), idx)
                if got is None:
                    continue
                ax.plot(got[0], got[1], color=st["color"], ls=st["ls"], lw=1.7, zorder=3)
            ax.set_title(title, fontsize=7.5, pad=3)
            ax.tick_params(labelsize=6.5)
            ax.set_xlim(-120, 460)
            if share and ylim is not None:
                ax.set_ylim(*ylim)
        for ax in axes[len(panels):]:
            ax.set_visible(False)

        for ax in axes[:len(panels)]:
            if ax.get_subplotspec().rowspan.start == (len(panels) - 1) // ncol:
                ax.set_xlabel("time (ms)", fontsize=8)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(ylabel, fontsize=7.5)

        handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
                   for st in DIR_STYLE.values()]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.012))
        unit_note = ("baseline-z-scored per model before averaging, so no model's amplitude "
                     "shrinkage dominates the mean"
                     if units == "z" else
                     "RAW µV averaged across models — dominated by wav2vec2-large and "
                     "whisper-medium; see Caveat 2. Each panel autoscales")
        if share and ylim is not None:
            unit_note += (f". Shared y-window {ylim[0]:g} to {ylim[1]:g}; traces outside it are "
                          f"clipped, not rescaled")
        fig.suptitle(f"FCz difference wave by direction, top {len(pairs)} pairs of the ranking"
                     + (f"   —   figure {k + 1} of {n_figs}" if n_figs > 1 else "")
                     + f"\n{unit_note}", fontsize=9.5, y=1.008)
        fig.tight_layout(rect=(0, 0.030, 1, 0.986))
        # The µV variant carries its unit in the name so a --wave_units uv run cannot silently
        # overwrite the z figures the memo links to.
        name = (f"{prefix}_direction_waveforms_{k + 1}.png" if units == "z"
                else f"{prefix}_direction_waveforms_uv_{k + 1}.png")
        fig.savefig(Path(out_dir) / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)
        print(f"  wrote {name}  ({len(chunk)} pairs)")

    print(f"  {len(pairs)} pairs over {n_figs} figure(s), units = {units}")
    return written


def plot_per_model_waveforms(ranked, models, out_dir, n=N_WAVE,
                             predictions_root=PREDICTIONS, chunk=0, ylim_pct=99.0,
                             prefix="phase1", expect_n_deviants=None):
    """One figure PER MODEL for a single chunk of pairs -- the combined panel, de-overlaid.

    Only chunk `chunk` (default the first 35 pairs, i.e. what phase1_strong_waveforms_1.png
    covers) is rendered, so this is 6 figures rather than 24.

    Two things become possible once a figure holds one model. The panels can **share a y-axis**,
    because within a model µV is on one scale -- Caveat 2 forbids comparing µV across models, not
    across pairs of the same model -- so pairs become comparable to each other, which is the point
    of a small multiple. And direction can own the colour channel (blue/orange, as in the
    direction-collapsed figure) instead of spending it on model identity, which the title carries.

    The shared limit is a symmetric `ylim_pct` percentile of |µV| over the figure rather than the
    max: a handful of deep troughs would otherwise flatten the other 30-odd panels. Traces beyond
    it are clipped, and the subtitle says so.
    """
    pairs = top_pairs(ranked, n)
    if not pairs:
        print("  skipped per-model waveforms: ranking is empty")
        return []
    nrow, ncol = PANELS_PER_FIG
    per_fig = nrow * ncol
    n_chunks = math.ceil(len(pairs) / per_fig)
    if not 0 <= chunk < n_chunks:
        raise SystemExit(f"--wave_chunk {chunk} out of range; {n_chunks} chunk(s) available")
    chunk_pairs = pairs[chunk * per_fig:(chunk + 1) * per_fig]

    geo = ranked.drop_duplicates("pair_id").set_index("pair_id")[["f_low", "f_high"]]
    tiers = ranked.set_index(["pair_id", "direction"])["n_agree"]
    waves = load_fcz_waves(chunk_pairs, models, predictions_root,
                           expect_n_deviants=expect_n_deviants)
    if not waves:
        print("  skipped per-model waveforms: no prediction HDF5s readable")
        return []

    lo, hi = MMN_WINDOW
    written = []
    for m in models:
        present = [(pid, d) for pid in chunk_pairs for d in ("regular", "counter")
                   if m in waves.get((pid, d), {})]
        if not present:
            print(f"  skipped per-model waveforms for {m}: no traces")
            continue

        stacked = np.concatenate([waves[(pid, d)][m][1][
            (waves[(pid, d)][m][0] >= -120) & (waves[(pid, d)][m][0] <= 460)]
            for pid, d in present])
        lim = float(np.nanpercentile(np.abs(stacked), ylim_pct))
        clipped = int((np.abs(stacked) > lim).sum())

        fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 1.95 * nrow),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, pid in zip(axes, chunk_pairs):
            ax.grid(False)
            ax.axvspan(lo, hi, color="#eef2f7", zorder=0)
            ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
            for direction, st in DIR_STYLE.items():
                got = waves.get((pid, direction), {}).get(m)
                if got is None:
                    continue
                t, uv, _z = got
                sel = (t >= -120) & (t <= 460)
                ax.plot(t[sel], uv[sel], color=st["color"], ls=st["ls"], lw=1.5, zorder=3)
            f_lo, f_hi = geo.loc[pid, "f_low"], geo.loc[pid, "f_high"]
            # This model's own verdict per direction -- the combined figure cannot show it.
            nr = tiers.get((pid, "regular"), np.nan)
            nc = tiers.get((pid, "counter"), np.nan)
            ax.set_title(f"method_{pid}: {f_lo:g} → {f_hi:g} Hz\nn_agree {nr:.0f} / {nc:.0f}",
                         fontsize=7.0, pad=3)
            ax.tick_params(labelsize=6.5)
            ax.set_xlim(-120, 460)
            ax.set_ylim(-lim, lim)
        for ax in axes[len(chunk_pairs):]:
            ax.set_visible(False)
        for ax in axes[:len(chunk_pairs)]:
            if ax.get_subplotspec().rowspan.start == (len(chunk_pairs) - 1) // ncol:
                ax.set_xlabel("time (ms)", fontsize=8)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("µV", fontsize=8)

        handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
                   for st in DIR_STYLE.values()]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.012))
        fig.suptitle(
            f"{MLABEL[m]} — FCz µV difference wave, top {len(pairs)} pairs of the ranking"
            f"\nfigure {chunk + 1} of {n_chunks}, pairs "
            f"{chunk * per_fig + 1}–{chunk * per_fig + len(chunk_pairs)} of {len(pairs)}. "
            f"Shared y-axis ±{lim:.2f} µV ({ylim_pct:g}th pct of |µV|; {clipped} sample(s) "
            f"clipped) — comparable across pairs, NOT across models",
            fontsize=9.5, y=1.012)
        fig.tight_layout(rect=(0, 0.035, 1, 0.985))
        name = f"{prefix}_strong_waveforms_{chunk + 1}__{m}.png"
        fig.savefig(Path(out_dir) / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)
        print(f"  wrote {name}  (±{lim:.2f} µV, {clipped} sample(s) clipped)")
    return written


# ──────────────────────────────────────────────────────────
# Item 4. Per-model µV over the strong set
# ──────────────────────────────────────────────────────────

def plot_model_uv_box(ranked, models, out_dir, tier=STRONG_TIER, prefix="phase1"):
    """Trough depth per model over ALL its responses on the strong set, one distribution each.

    Every direction-instance in the set contributes its `trough_uv` for this model, whether or
    not the model agreed. Splitting the row by S7 verdict cut each distribution at the µV floor
    by construction, which manufactured two tight clusters either side of a line the reader can
    already see -- the floor is drawn, so where a model's mass sits relative to it is readable
    from one continuous distribution. The agreed/dissented counts stay in the row labels and in
    the CSV, where they are counts rather than an axis split.
    """
    strong = ranked[ranked["n_agree"] >= tier]
    if strong.empty:
        print(f"  skipped µV boxplot: no direction-instance at n_agree >= {tier}")
        return None

    data, ns = {}, {}
    for m in models:
        agree = strong[f"s7__{m}"]
        data[m] = strong[f"trough_uv__{m}"].dropna().values
        ns[(m, "s7")] = int(agree.sum())
        ns[(m, "s2_only")] = int((strong[f"s2__{m}"] & ~agree).sum())
        ns[(m, "no_s2")] = int((~strong[f"s2__{m}"]).sum())

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.grid(True, axis="x")
    pos = np.arange(len(models))[::-1]                 # first model at the top
    vals = [data[m] for m in models]
    bp = ax.boxplot(vals, positions=pos, vert=False, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"))
    for patch, m in zip(bp["boxes"], models):
        patch.set(facecolor=MODEL_STYLE[m]["color"], alpha=0.28,
                  edgecolor=MODEL_STYLE[m]["color"], lw=1.1)
    rng = np.random.default_rng(0)                     # jitter only; seeded so reruns match
    for pp, v, m in zip(pos, vals, models):
        if len(v):
            ax.scatter(v, pp + rng.uniform(-0.16, 0.16, len(v)), s=10,
                       color=MODEL_STYLE[m]["color"], alpha=0.5, linewidths=0)

    ax.axvline(-DIP_UV_THRESHOLD, **REF_STYLE)
    ax.text(-DIP_UV_THRESHOLD, len(models) - 0.35, f" X = −{DIP_UV_THRESHOLD} µV floor",
            fontsize=8, color=MUTED, ha="left", va="top")
    ax.set_yticks(pos)
    ax.set_yticklabels(
        [f"{MLABEL[m]}\nS7 {ns[(m, 's7')]} / {len(strong)}" for m in models], fontsize=8.5)
    ax.set_xlabel("trough_uv at FCz (µV)")
    ax.set_title(f"Predicted trough depth per model over the n_agree ≥ {tier} set\n"
                 f"every direction-instance contributes, agreeing or not; the row label gives "
                 f"how many cleared S7", pad=10)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.5, 0.04,
             "Each model's µV scale is its own: mTRF predictions are ~4× amplitude-shrunk and the "
             "shrinkage differs by model and\nlayer. Read each row against its own X = −0.75 µV "
             "floor — a deeper box is NOT a stronger MMN than a shallower one.",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(Path(out_dir) / f"{prefix}_model_uv_box.png", bbox_inches="tight")
    plt.close(fig)

    def _stat(v, fn, *a):
        return float(fn(v, *a)) if len(v) else np.nan

    summary = pd.DataFrame({
        "model": models,
        "n_instances": [len(strong)] * len(models),
        "n_agreed_s7": [ns[(m, "s7")] for m in models],
        "n_dissented_s2_only": [ns[(m, "s2_only")] for m in models],
        "n_no_s2": [ns[(m, "no_s2")] for m in models],
        "median_uv_all": [_stat(data[m], np.median) for m in models],
        "q25_uv_all": [_stat(data[m], np.percentile, 25) for m in models],
        "q75_uv_all": [_stat(data[m], np.percentile, 75) for m in models],
        "min_uv_all": [_stat(data[m], np.min) for m in models],
        "max_uv_all": [_stat(data[m], np.max) for m in models],
        "median_uv_agreed": [_stat(strong.loc[strong[f"s7__{m}"], f"trough_uv__{m}"]
                                   .dropna().values, np.median) for m in models],
        "s7_rate_all": [float(ranked[f"s7__{m}"].mean()) for m in models],
        "s2_rate_all": [float(ranked[f"s2__{m}"].mean()) for m in models],
    })
    summary.round(4).to_csv(Path(out_dir) / f"{prefix}_model_uv_box.csv", index=False)
    print(f"  wrote {prefix}_model_uv_box.png/.csv  (S7 count per model: "
          f"{ {m: ns[(m, 's7')] for m in models} } of {len(strong)})")
    return summary


def table_model_dissent(ranked, models, out_dir, prefix="phase1"):
    """Which model breaks a would-be unanimous row, and how often.

    A model's marginal S7 rate says how strict it is overall; the sole-dissent count says whether
    it disagrees on the direction-instances the OTHER five rank highest, which is a different
    claim and the one the memo makes about whisper-tiny. Reported as a share of the
    (n_models - 1)/n_models rows, since that denominator moves between phases.
    """
    n_models = len(models)
    one_short = ranked[ranked["n_agree"] == n_models - 1]
    rows = []
    for m in models:
        sole = int((~one_short[f"s7__{m}"]).sum())
        rows.append({
            "model": m,
            "s7_rate": float(ranked[f"s7__{m}"].mean()),
            "n_s7": int(ranked[f"s7__{m}"].sum()),
            "sole_dissenter": sole,
            "pct_of_one_short": (100.0 * sole / len(one_short)) if len(one_short) else np.nan,
        })
    tbl = pd.DataFrame(rows).sort_values("sole_dissenter", ascending=False)
    tbl.round(4).to_csv(Path(out_dir) / f"{prefix}_model_dissent.csv", index=False)
    top = tbl.iloc[0]
    print(f"  wrote {prefix}_model_dissent.csv  ({len(one_short)} rows at "
          f"{n_models - 1}/{n_models}; most frequent sole dissenter = {top['model']}, "
          f"{int(top['sole_dissenter'])} of them)")
    return tbl


# ──────────────────────────────────────────────────────────
# Item 5. Top-30 direction-instances
# ──────────────────────────────────────────────────────────

def table_topn(ranked, models, out_dir, n=N_TOP, prefix="phase1", other_rank=None,
               other_label="phase2_rank"):
    """The top `n` direction-instances, with the other phase's rank for each where available.

    `other_rank` is a {(pair_id, direction): rank} mapping. Carrying it makes the table
    self-contained: a reader can see that a Phase-1 rank-1 instance is Phase-2 rank 94 without
    cross-referencing another section. Instances the other phase never evaluated get NaN, which
    is the honest value -- they were not ranked lower, they were not measured.
    """
    cols = _uv_cols(models)
    top = ranked.head(n).copy()
    if other_rank is not None:
        top[other_label] = [other_rank.get((int(p), d), np.nan)
                            for p, d in zip(top["pair_id"], top["direction"])]
    # mean/median/max/min are over ALL SIX models' trough_uv, agreeing or not -- a spread that
    # only covered the agreeing models would narrow itself by construction as n_agree falls.
    top["mean_uv_all6"] = top[cols].mean(axis=1)
    top["median_uv_all6"] = top[cols].median(axis=1)
    top["max_uv_all6"] = top[cols].max(axis=1)
    top["min_uv_all6"] = top[cols].min(axis=1)
    top["models_not_agreeing"] = [
        ", ".join(m for m in models if not r[f"s7__{m}"]) for _, r in top.iterrows()]

    # The stimulus as the listener meets it: standard -> deviant, in Hz. Semitones and % deviance
    # are dropped -- Section 5 shows they do not order the ranking, and two derived numbers were
    # crowding out the one description a reader actually needs to identify the stimulus.
    reg = top["direction"].eq("regular")
    top["stimulus"] = [f"{a:g} → {b:g} Hz" for a, b in
                       zip(np.where(reg, top["f_low"], top["f_high"]),
                           np.where(reg, top["f_high"], top["f_low"]))]
    keep = ["rank", "method", "stimulus", "n_agree", "mean_uv", "mean_uv_all6",
            "median_uv_all6", "max_uv_all6", "min_uv_all6", "models_not_agreeing"]
    if other_rank is not None:
        keep.insert(2, other_label)
    out = top[keep].round(
        {"mean_uv": 3, "mean_uv_all6": 3,
         "median_uv_all6": 3, "max_uv_all6": 3, "min_uv_all6": 3})
    out.to_csv(Path(out_dir) / f"{prefix}_top{n}.csv", index=False)
    print(f"  wrote {prefix}_top{n}.csv  (ranks 1–{n}; "
          f"{(top['n_agree'] == len(models)).sum()} at {len(models)}/{len(models)})")
    return out


def table_yield(ranked, models, out_dir, prefix="phase1"):
    """Pairs and direction-instances surviving each agreement threshold.

    Shares are of the direction-instances THIS ranking covers -- 1806 for the whole grid, 254 for
    a selected subset. Against the pair count they would compare an instance count to a pair
    count and read as roughly twice the yield they are; against the whole grid they would make a
    subset's rows look like a coverage statistic rather than a yield.
    """
    n_models = len(models)
    total = len(ranked)
    rows = []
    for t in range(n_models, 0, -1):
        at = ranked[ranked["n_agree"] >= t]
        rows.append({
            "tier_min": t,
            "direction_instances": len(at),
            "pct_of_ranked_directions": 100.0 * len(at) / total,
            "distinct_pairs": int(at["pair_id"].nunique()),
        })
    tbl = pd.DataFrame(rows)
    tbl.round(4).to_csv(Path(out_dir) / f"{prefix}_yield.csv", index=False)
    print(f"  wrote {prefix}_yield.csv  (shares against this ranking's {total} "
          f"direction-instances)")
    return tbl


# ──────────────────────────────────────────────────────────
# Item 6. Mean-µV heatmap over the 43x43 grid
# ──────────────────────────────────────────────────────────

def plot_mean_uv_heatmap(ranked, models, out_dir, prefix="phase1", all_freqs=None):
    """43x43 grid coloured by mean_uv_all6, with its own coverage drawn on its face.

    `all_freqs` fixes the axes to the whole frequency ladder even when `ranked` covers only a
    subset of the grid, so a Phase-2 heatmap is directly comparable to the Phase-1 one cell for
    cell. Cells the phase did not evaluate get their own colour, distinct from the excluded
    diagonal -- "not measured" and "measured, no response" must never read as the same cell.

    At partial coverage the row/column marginals are means over unequal and SELECTION-BIASED
    subsets of each frequency's partners, so the stripes CSV carries the n behind every value and
    the caption says the marginals are not comparable to the full-grid ones.
    """
    freqs = sorted(all_freqs) if all_freqs is not None else sorted(
        set(ranked["f_low"]) | set(ranked["f_high"]))
    idx = {f: i for i, f in enumerate(freqs)}
    n = len(freqs)
    grid = np.full((n, n), np.nan)
    for r in ranked.itertuples(index=False):
        std, dev = ((r.f_low, r.f_high) if r.direction == "regular" else (r.f_high, r.f_low))
        grid[idx[std], idx[dev]] = r.mean_uv_all6

    off_diagonal = n * n - n
    filled = int(np.isfinite(grid).sum())
    partial = filled < off_diagonal
    finite = grid[np.isfinite(grid)]
    straddles = finite.min() < 0 < finite.max()
    # Blue is the MMN end throughout: "RdBu_r" when values straddle zero, "Blues_r" when they are
    # all negative so the most negative cell still gets the darkest blue.
    # The masked colour is every empty cell, so at full coverage it can only be the excluded
    # diagonal; at partial coverage it is mostly "never measured", and the diagonal is repainted
    # over it below in the darker EMPTY_CELL so the two absences stay distinguishable.
    cmap = plt.get_cmap(DIV_CMAP if straddles else SEQ_NEG_CMAP).with_extremes(
        bad=UNEVALUATED_CELL if partial else EMPTY_CELL)
    # TwoSlopeNorm, not symmetric limits: the negative tail runs 3x deeper than the positive one,
    # so symmetric limits would spend half the ramp on a range that holds almost no cells. This
    # pins the ramp's neutral colour to exactly 0 while using its full extent on both sides.
    norm = (mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=float(finite.min()),
                                    vmax=float(finite.max())) if straddles
            else mpl.colors.Normalize(vmin=float(finite.min()), vmax=0.0))

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.grid(False)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, origin="lower", norm=norm,
                   interpolation="nearest")
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
                f"({100.0 * filled / off_diagonal:.0f}% coverage) — the rest were never measured "
                f"in this phase" if partial else "")
    ax.set_title("Predicted trough depth over the novel frequency grid\n"
                 f"mean trough_uv at FCz across all {len(models)} models "
                 f"(mean_uv_all6); more negative = stronger{coverage}", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("mean trough_uv across all 6 models (µV)")
    cbar.outline.set_visible(False)

    note = ("Values straddle zero, so this is a diverging ramp with its neutral midpoint at "
            "exactly 0 µV: blue = negative (an MMN-\nlike trough), red = positive (no trough). "
            if straddles else
            "All values are negative, so this is a single-hue ramp: dark blue = most negative. ")
    absence = ("Two kinds of empty cell: the mid-grey diagonal is excluded by design (a "
               "same-frequency deviant synthesizes to\nthe standard's waveform, so the difference "
               "is exactly zero); the pale background is a cell this phase never\nmeasured. "
               if partial else
               "Grey diagonal = excluded by design (a same-frequency deviant synthesizes to the "
               "standard's waveform, so\nthe difference is exactly zero). ")
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.text(0.5, 0.05,
             note + "Each off-diagonal cell is one direction-instance;\n(i,j) and (j,i) are the "
             "two directions of one pair, so asymmetry about the diagonal is the "
             "frequency-preference effect.\n" + absence + "Averaged across models whose µV scales "
             "are not comparable — read the pattern, not the value.",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(Path(out_dir) / f"{prefix}_mean_uv_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(grid, index=pd.Index(freqs, name="standard_hz"),
                 columns=pd.Index(freqs, name="deviant_hz")).to_csv(
        Path(out_dir) / f"{prefix}_mean_uv_grid.csv")

    # Row and column marginals. A frequency whose ROW is uniformly dark produces a trough
    # whatever it is paired against -- that is a frequency preference, not a deviance response,
    # and it is the thing a heatmap makes visible that a ranking cannot.
    #
    # n_as_standard / n_as_deviant are not decoration: at full coverage they are 42 everywhere and
    # the means are comparable across frequencies; at partial coverage they differ by frequency
    # AND are biased by whatever selected the subset, so a marginal read without its n is not a
    # frequency preference, it is a selection artifact.
    with np.errstate(invalid="ignore"):                # all-NaN rows are legitimate at coverage
        stripes = pd.DataFrame({
            "hz": freqs,
            "n_as_standard": np.isfinite(grid).sum(axis=1),
            "mean_uv_as_standard": np.nanmean(np.where(np.isfinite(grid), grid, np.nan), axis=1),
            "n_as_deviant": np.isfinite(grid).sum(axis=0),
            "mean_uv_as_deviant": np.nanmean(np.where(np.isfinite(grid), grid, np.nan), axis=0),
        })
    stripes.round(4).to_csv(Path(out_dir) / f"{prefix}_frequency_stripes.csv", index=False)
    s_row = stripes.loc[stripes["mean_uv_as_standard"].idxmin()]
    s_col = stripes.loc[stripes["mean_uv_as_deviant"].idxmin()]
    print(f"  wrote {prefix}_mean_uv_heatmap.png, {prefix}_mean_uv_grid.csv, "
          f"{prefix}_frequency_stripes.csv")
    print(f"  coverage {filled}/{off_diagonal} off-diagonal cells "
          f"({100.0 * filled / off_diagonal:.1f}%)")
    print(f"  range {finite.min():+.2f} to {finite.max():+.2f} µV, "
          f"{'diverging (midpoint 0)' if straddles else 'sequential'} ramp; grand mean "
          f"{finite.mean():+.3f} µV")
    print(f"  darkest stripe: standard = {s_row.hz:g} Hz "
          f"({s_row.mean_uv_as_standard:+.3f} µV over its {int(s_row.n_as_standard)} deviants), "
          f"deviant = {s_col.hz:g} Hz ({s_col.mean_uv_as_deviant:+.3f} µV over its "
          f"{int(s_col.n_as_deviant)} standards)")
    # The norm rides out so the literature companion figure can be drawn on the SAME scale.
    return grid, norm


# ──────────────────────────────────────────────────────────
# Consensus yield: how many stimuli clear a top-X cut in at least Y models
# ──────────────────────────────────────────────────────────

def consensus_grid(ranked, models, thresholds=CONSENSUS_TOP_X):
    """counts[Y, X] = direction-instances in the top `X` of AT LEAST `Y` models.

    Each model gets its OWN ranking of the direction-instances, ordered by that model's
    `trough_uv` ascending (most negative first) among the instances where that model satisfies
    **S2**. The S2 gate matters: without it a model's "top" would be led by whatever produced the
    deepest excursion anywhere in the epoch, including troughs at latencies that are not MMN
    latencies -- the same reason `mean_uv` averages only over agreeing models.

    Cumulative in Y by construction, so each column is non-increasing downward: an instance in the
    top X of 4 models is also in the top X of at least 3.
    """
    per_model_rank = {}
    for m in models:
        d = ranked[ranked[f"s2__{m}"]].dropna(subset=[f"trough_uv__{m}"])
        order = d.sort_values(f"trough_uv__{m}", ascending=True, kind="mergesort")
        per_model_rank[m] = order["method"].tolist()

    counts = np.zeros((len(models), len(thresholds)), dtype=int)
    for xi, x in enumerate(thresholds):
        tally = {}
        for m in models:
            for meth in per_model_rank[m][:x]:
                tally[meth] = tally.get(meth, 0) + 1
        for yi, y in enumerate(range(1, len(models) + 1)):
            counts[yi, xi] = sum(1 for v in tally.values() if v >= y)
    return counts, {m: len(v) for m, v in per_model_rank.items()}


def plot_consensus_heatmap(ranked, models, out_dir, prefix="phase1",
                           thresholds=CONSENSUS_TOP_X):
    """The consensus-yield grid: how far a top-X cut has to be relaxed to get Y models to agree."""
    thresholds = tuple(x for x in thresholds if x <= len(ranked)) or (len(ranked),)
    counts, n_s2 = consensus_grid(ranked, models, thresholds)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(10.6, 0.72 * len(thresholds) + 2.0), 5.2))
    ax.grid(False)
    im = ax.imshow(counts, cmap=CONSENSUS_CMAP, origin="lower", aspect="auto",
                   interpolation="nearest")
    hi = counts.max() if counts.max() else 1
    for yi in range(counts.shape[0]):
        for xi in range(counts.shape[1]):
            v = counts[yi, xi]
            ax.text(xi, yi, str(v), ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 * hi else INK)

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(x) for x in thresholds], fontsize=8.5, rotation=45)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([str(y) for y in range(1, n_models + 1)], fontsize=9)
    ax.set_xlabel("top X rank for the model (a stimulus must be in a model's top X)")
    ax.set_ylabel("number of models agreeing")
    ax.set_title(f"Consensus yield: direction-instances in the top X of at least Y models\n"
                 f"per-model ranking by that model's own trough_uv among its S2 responses; "
                 f"{len(ranked)} instances, {n_models} models", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("number of stimuli"); cbar.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.035,
             "Cumulative in Y: a stimulus in the top X of 4 models is also in the top X of at "
             "least 3, so every column is non-increasing\nupward. The top-left corner is the "
             "strict end — few stimuli, many models agreeing — and is where a stimulus set would "
             "be drawn from.",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(Path(out_dir) / f"{prefix}_consensus_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    tbl = pd.DataFrame(counts, index=pd.Index(range(1, n_models + 1), name="n_models_min"),
                       columns=pd.Index(thresholds, name="top_x"))
    tbl.to_csv(Path(out_dir) / f"{prefix}_consensus_heatmap.csv")
    print(f"  wrote {prefix}_consensus_heatmap.png/.csv  (S2 responses per model: {n_s2})")
    return tbl


# ──────────────────────────────────────────────────────────
# Item 6b. Deviance scaling and frequency structure, on whatever set is passed
# ──────────────────────────────────────────────────────────

def table_deviance_octiles(ranked, out_dir, prefix="phase1", n_bins=8, compare_col=None):
    """Mean n_agree by octile of semitone distance, plus Spearman rho(semitones, n_agree).

    `compare_col` (e.g. "phase1_n_agree") adds the other phase's score for the SAME instances, so
    the two can be read against each other without a second population creeping in.
    """
    d = ranked.dropna(subset=["semitones"]).copy()
    if d.empty:
        print("  skipped deviance octiles: no semitone column")
        return None, (np.nan, np.nan)
    d["bin"] = pd.qcut(d["semitones"], q=min(n_bins, d["semitones"].nunique()), duplicates="drop")

    g = d.groupby("bin", observed=True)
    tbl = pd.DataFrame({
        "semitones_lo": [iv.left for iv in g.groups],
        "semitones_hi": [iv.right for iv in g.groups],
        "mean_semitones": g["semitones"].mean().values,
        "mean_pct_deviance": g["pct_deviance"].mean().values,
        "n": g.size().values,
        "mean_n_agree": g["n_agree"].mean().values,
        "sem_n_agree": g["n_agree"].sem().values,
        "pct_at_ge5": (g.apply(lambda s: 100.0 * (s["n_agree"] >= STRONG_TIER).mean(),
                               include_groups=False).values),
    })
    rho, pval = stats.spearmanr(d["semitones"], d["n_agree"])
    tbl["spearman_rho_semitones_n_agree"] = rho
    tbl["spearman_p"] = pval
    if compare_col and compare_col in d.columns:
        tbl[f"mean_{compare_col}"] = g[compare_col].mean().values
        rho1, p1 = stats.spearmanr(d["semitones"], d[compare_col])
        tbl[f"spearman_rho_semitones_{compare_col}"] = rho1
        tbl[f"spearman_p_{compare_col}"] = p1
        print(f"  rho(semitones, {compare_col}) on the same {len(d)} instances = {rho1:+.3f} "
              f"(p = {p1:.3g})")
    tbl.round(4).to_csv(Path(out_dir) / f"{prefix}_deviance_octiles.csv", index=False)
    print(f"  wrote {prefix}_deviance_octiles.csv  "
          f"(rho(semitones, n_agree) = {rho:+.3f}, p = {pval:.3g}, n = {len(d)})")
    return tbl, (rho, pval)


def table_frequency_involvement(ranked, out_dir, prefix="phase1"):
    """Per frequency: how it scores as the standard and as the deviant, each with its n.

    The presentation that survives partial grid coverage where a 43x43 heatmap does not. A
    heatmap asks the reader to see a stripe against 42 same-row cells; when only a handful of a
    row's cells exist, that reading is unavailable and the honest object is the marginal WITH
    its denominator.
    """
    rows = []
    for role, hz_col in (("standard", "f_std"), ("deviant", "f_dev")):
        d = ranked.copy()
        d["f_std"] = np.where(d["direction"] == "regular", d["f_low"], d["f_high"])
        d["f_dev"] = np.where(d["direction"] == "regular", d["f_high"], d["f_low"])
        g = d.groupby(hz_col)
        rows.append(pd.DataFrame({
            f"n_as_{role}": g.size(),
            f"mean_n_agree_as_{role}": g["n_agree"].mean(),
            f"mean_uv_all6_as_{role}": g["mean_uv_all6"].mean(),
        }))
    out = pd.concat(rows, axis=1).rename_axis("hz").reset_index()
    out["n_instances"] = out["n_as_standard"].fillna(0) + out["n_as_deviant"].fillna(0)
    out.round(4).to_csv(Path(out_dir) / f"{prefix}_frequency_involvement.csv", index=False)
    print(f"  wrote {prefix}_frequency_involvement.csv  ({len(out)} frequencies, "
          f"n per frequency {int(out['n_instances'].min())}–{int(out['n_instances'].max())})")
    return out


# ──────────────────────────────────────────────────────────
# Item 7. Chance baseline, literature comparison, ranking structure
# ──────────────────────────────────────────────────────────

def load_literature(models, lit_csv=LITERATURE, meta_csv=LIT_META):
    """The 24-method literature screen, ranked on this search's criteria and carrying frequencies.

    Scored on exactly the slice the novel grid uses -- same S7@0.75, FCz, mTRF, same 6 models --
    so every comparison against it is like-for-like. The metadata's `standard_freq`/`deviant_freq`
    describe the REGULAR direction, so a `_counter` instance has them swapped; getting that
    backwards would mirror the literature points about the diagonal and invert the very asymmetry
    the heatmaps are for.
    """
    lit_csv, meta_csv = Path(lit_csv), Path(meta_csv)
    if not lit_csv.exists():
        print(f"  skipped literature: {lit_csv} not found")
        return None
    lit = rank_directions(load_scored(lit_csv, models=models, roi=ROI, mapping=MAPPING,
                                      x=DIP_UV_THRESHOLD), models=models)
    lit["mean_uv_all6"] = lit[[f"trough_uv__{m}" for m in models]].mean(axis=1)
    if not meta_csv.exists():
        print(f"  literature frequencies unavailable: {meta_csv} not found")
        return lit
    md = pd.read_csv(meta_csv)[["method_id", "standard_freq", "deviant_freq"]]
    lit = lit.merge(md.rename(columns={"method_id": "pair_id"}), on="pair_id", how="left")
    missing = lit["standard_freq"].isna()
    if missing.all():
        # None of the ids are in the sheet, so this is not the literature screen -- a stand-in
        # CSV. Frequencies are simply unavailable; the tier and top-N comparisons still work.
        print(f"  no method_id in {meta_csv.name} matches this screen; "
              f"skipping the frequency-space figures")
        return lit.drop(columns=["standard_freq", "deviant_freq"])
    if missing.any():
        # A PARTIAL match is a real inconsistency: half the set would be silently dropped from
        # every frequency figure, and the reader would have no way to tell.
        raise SystemExit(f"literature metadata has no entry for method_id "
                         f"{sorted(lit.loc[missing, 'pair_id'].unique())[:10]}")
    counter = lit["direction"].eq("counter")
    lit["f_std"] = np.where(counter, lit["deviant_freq"], lit["standard_freq"])
    lit["f_dev"] = np.where(counter, lit["standard_freq"], lit["deviant_freq"])
    lit["f_low"] = lit[["standard_freq", "deviant_freq"]].min(axis=1)
    lit["f_high"] = lit[["standard_freq", "deviant_freq"]].max(axis=1)
    return lit


def table_novel_vs_literature_top(ranked, lit, out_dir, n=10, prefix="phase1"):
    """The top `n` of each set on the two quantities the search ranks by, with a mean row.

    Deliberately only n_agree and mean_uv. Semitones and % deviance describe the stimulus, not
    the response, and Section 5 already shows they do not order the ranking; carrying them here
    invited the two sets to be compared on a dimension neither was selected for.
    """
    if lit is None:
        print("  skipped top-N literature comparison: no literature ranking")
        return None
    rows = []
    for label, df in (("novel", ranked), ("literature", lit)):
        top = df.head(n).copy()
        if "f_std" in top.columns:                      # literature carries these directly
            std, dev = top["f_std"], top["f_dev"]
        elif "f_low" in top.columns:
            reg = top["direction"].eq("regular")
            std = np.where(reg, top["f_low"], top["f_high"])
            dev = np.where(reg, top["f_high"], top["f_low"])
        else:
            # A literature CSV with no metadata match carries no frequencies at all; the row
            # still has a rank and a score, so the comparison stands without the description.
            std = dev = None
        top["stimulus"] = ([f"{a:g} → {b:g} Hz" for a, b in zip(std, dev)]
                           if std is not None else "")
        for _, r in top.iterrows():
            rows.append({"set": label, "rank": int(r["rank"]), "method": r["method"],
                         "stimulus": r["stimulus"],
                         "n_agree": int(r["n_agree"]), "mean_uv": r["mean_uv"]})
        rows.append({"set": label, "rank": np.nan, "method": f"mean of the top {n}",
                    "stimulus": "", "n_agree": float(top["n_agree"].mean()),
                    "mean_uv": float(top["mean_uv"].mean())})
    tbl = pd.DataFrame(rows)
    tbl.round(4).to_csv(Path(out_dir) / f"{prefix}_top{n}_vs_literature.csv", index=False)
    for label in ("novel", "literature"):
        d = tbl[(tbl["set"] == label) & tbl["rank"].notna()]
        print(f"  top-{n} {label:<10}: mean n_agree {d['n_agree'].mean():.1f}, "
              f"mean mean_uv {d['mean_uv'].mean():+.3f} µV")
    print(f"  wrote {prefix}_top{n}_vs_literature.csv")
    return tbl


def plot_literature_heatmaps(lit, models, out_dir, uv_norm=None, prefix="phase1",
                             ranked=None, n_novel=24):
    """The literature stimuli in (standard, deviant) frequency space, on the novel colour scales.

    The literature pairs do not sit on the novel grid's 1.5-semitone ladder, so they cannot be
    cells of the same 43x43 matrix. They are drawn as a scatter on log-frequency axes with the
    SAME colormaps and the SAME normalisation as the novel heatmaps, which is what makes the two
    comparable: a cell and a marker of the same colour mean the same number.

    `uv_norm` is the novel figure's TwoSlopeNorm. Passing it is the whole point -- renormalising
    to the literature set's own range would make its weakest response look like the novel grid's
    strongest.

    `ranked` adds the top `n_novel` novel direction-instances as triangles on both panels, on the
    same scales, so the comparison is inside one pair of axes rather than across two figures. The
    axis limits then have to span both sets, which is itself the point: the novel winners sit
    where no literature pair goes.
    """
    if lit is None or "f_std" not in lit.columns:
        print("  skipped literature heatmaps: no literature frequencies")
        return
    n_models = len(models)
    # The top novel instances, in the same (standard, deviant) coordinates.
    nov = None
    if ranked is not None:
        nov = ranked.head(n_novel).copy()
        reg = nov["direction"].eq("regular")
        nov["f_std"] = np.where(reg, nov["f_low"], nov["f_high"])
        nov["f_dev"] = np.where(reg, nov["f_high"], nov["f_low"])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))

    # (a) n_agree, on the same discrete Blues ramp as the novel n_agree heatmap
    ax = axes[0]
    cmap = plt.get_cmap(SEQ_CMAP, n_models + 1)
    sc = ax.scatter(lit["f_std"], lit["f_dev"], c=lit["n_agree"], cmap=cmap,
                    vmin=-0.5, vmax=n_models + 0.5, s=95, linewidths=0.8, edgecolors="#4a4a4a",
                    label=f"literature ({len(lit)} instances)")
    if nov is not None:
        ax.scatter(nov["f_std"], nov["f_dev"], c=nov["n_agree"], cmap=cmap, marker="^",
                   vmin=-0.5, vmax=n_models + 0.5, s=120, linewidths=0.8, edgecolors="#7a3d00",
                   zorder=4, label=f"novel, top {len(nov)}")
    cb = fig.colorbar(sc, ax=ax, ticks=range(n_models + 1), fraction=0.046, pad=0.03)
    cb.set_label(f"n_agree (0–{n_models})"); cb.outline.set_visible(False)
    ax.set_title("Cross-model agreement: literature vs the novel top instances\n"
                 f"circles = {len(lit)} literature instances, triangles = novel; one shared ramp", pad=10)

    # (b) mean_uv_all6, on the novel figure's diverging ramp AND its normalisation
    ax = axes[1]
    cmap = plt.get_cmap(DIV_CMAP)
    sc = ax.scatter(lit["f_std"], lit["f_dev"], c=lit["mean_uv_all6"], cmap=cmap, norm=uv_norm,
                    s=95, linewidths=0.8, edgecolors="#4a4a4a",
                    label=f"literature ({len(lit)} instances)")
    if nov is not None:
        ax.scatter(nov["f_std"], nov["f_dev"], c=nov["mean_uv_all6"], cmap=cmap, norm=uv_norm,
                   marker="^", s=120, linewidths=0.8, edgecolors="#7a3d00", zorder=4,
                   label=f"novel, top {len(nov)}")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean trough_uv across all 6 models (µV)"); cb.outline.set_visible(False)
    ax.set_title("Predicted trough depth: literature vs the novel top instances\n"
                 "one shared ramp and one shared 0-centred µV scale", pad=10)

    ticks = [200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 8000]
    lim = (170, 9500) if nov is not None else (450, 3200)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8, rotation=45)
        ax.set_yticklabels([f"{v:g}" for v in ticks], fontsize=8)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.plot(lim, lim, color="#bbbbbb", lw=0.9, zorder=1)
        ax.set_xlabel("standard frequency (Hz, log scale)")
        ax.set_ylabel("deviant frequency (Hz, log scale)")
        ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.95,
                  edgecolor="#4a4a4a", facecolor="white", borderpad=0.6)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.5, 0.045,
             "Every literature pair lives in 600–2000 Hz, a corner the novel grid's 200–7611 Hz "
             "ladder covers in 14 of its 43 rungs; the novel\nwinners sit outside it, mostly "
             "against a 7611 Hz deviant. Points above the grey line are ascending (deviant higher "
             "than\nstandard); below it, descending. Colour scales are identical to the "
             "novel-grid figures, so a marker and a cell of the same\ncolour mean the same "
             "number — circles and triangles on this figure are directly comparable.",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    fig.savefig(Path(out_dir) / f"{prefix}_literature_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {prefix}_literature_heatmap.png  ({len(lit)} literature instances"
          f"{f', {len(nov)} novel' if nov is not None else ''})")


def compare_to_literature(ranked, models, out_dir, lit, prefix="phase1"):
    """Tier distribution of the two sets, scored identically."""
    if lit is None:
        return None
    rows = []
    for label, df in (("novel", ranked), ("literature", lit)):
        for t in range(len(models), -1, -1):
            k = int((df["n_agree"] == t).sum())
            rows.append({"set": label, "n_agree": t, "directions": k,
                         "pct": 100.0 * k / len(df)})
    dist = pd.DataFrame(rows)
    dist.to_csv(Path(out_dir) / f"{prefix}_novel_vs_literature.csv", index=False)
    for label, df in (("novel", ranked), ("literature", lit)):
        b = df.iloc[0]
        print(f"  {label:>10}: {len(df):>4} direction-instances, best = {b['method']} "
              f"at n_agree {int(b['n_agree'])}/{len(models)}, mean_uv {b['mean_uv']:.3f} µV")
    print(f"  wrote {prefix}_novel_vs_literature.csv")
    return dist


def plot_ranking_structure(ranked, models, out_dir, chf_per_pair=CHF_PER_PAIR_PHASE2,
                           prefix="phase1", grid_pairs=None, lit_pairs=None):
    """The structure of the ranking, as THREE standalone figures rather than one 2x2 panel.

    Split because the three ask unrelated questions and were being read as one exhibit; at 2x2
    each was also too small to read its own axis labels. The former top-right panel (|mean_uv|
    against rank over the top 300) is dropped: it only ever showed that the decay is smooth and
    that its two visible breaks are the n_agree tier boundaries, i.e. artifacts of the sort key.

      <prefix>_yield_curve.png       how many pairs survive each (n_agree, mean_uv) cutoff
      <prefix>_grid_position.png     where the strong pairs sit in (f_low, f_high)
      <prefix>_direction_matrix.png  each pair's two directions cross-tabulated

    `grid_pairs` is the whole frequency grid, used as the backdrop of the position figure.
    Without it a Phase-2 frame would plot its 127 selected pairs against a backdrop of the same
    127, which shows "the strong pairs cover the grid" no matter where they actually sit.
    """
    n_pairs_total = ranked["pair_id"].nunique()
    n_models = len(models)

    # From 0 so the axis starts where 'no µV requirement at all' sits, out to 3.5 where
    # every curve has already reached zero.
    uv_cutoffs = np.round(np.arange(0.0, 3.51, 0.25) * -1, 2)[::-1]
    curve = pair_cutoff_curve(ranked, thresholds=range(1, n_models + 1),
                              uv_cutoffs=uv_cutoffs, chf_per_pair=chf_per_pair)
    curve.round(4).to_csv(Path(out_dir) / f"{prefix}_cutoff_curve.csv", index=False)
    gaps = direction_gap(ranked)
    gaps.round(4).to_csv(Path(out_dir) / f"{prefix}_direction_asymmetry.csv", index=False)

    # 1. Yield at every cutoff -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for t in range(n_models, 0, -1):
        d = curve[curve["n_agree_min"] == t].sort_values("mean_uv_max")
        if d["pairs"].max() == 0:
            continue
        ax.plot(-d["mean_uv_max"], d["pairs"], marker="o", ms=3.8, lw=1.6,
                color=plt.get_cmap("Blues")(0.30 + 0.11 * t), label=f"n_agree \u2265 {t}")
    ax.set_xlabel("mean_uv cutoff (|\u00b5V|; a direction must be at least this deep)")
    ax.set_ylabel("pairs qualifying")
    ax.set_xlim(0, 3.5)
    ax.set_xticks(np.arange(0, 3.51, 0.5))
    # symlog keeps the low tiers legible beside a 903-pair curve, but its default decade ticks
    # render as 10^x and its linear region draws a -10^1 tick below zero, which is a count that
    # cannot exist. Explicit plain-integer ticks from 0 up, and a floor at 0.
    ax.set_yscale("symlog", linthresh=10)
    # Drop any candidate tick that would collide with the total's own label.
    yt = sorted({v for v in (0, 5, 10, 25, 50, 100, 200, 400, 900)
                 if v <= n_pairs_total and abs(v - n_pairs_total) > 0.06 * n_pairs_total}
                | {n_pairs_total})
    ax.set_yticks(yt)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(0, n_pairs_total * 1.35)
    ax.set_title(f"Pairs qualifying at each agreement and \u00b5V cutoff\n"
                 f"of the {n_pairs_total} pairs in this ranking", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=2)
    for t in (n_models, STRONG_TIER):
        k = int(ranked[ranked["n_agree"] >= t]["pair_id"].nunique())
        if k:
            ax.axhline(k, **REF_STYLE)
            ax.text(0.99, k, f"n_agree \u2265 {t}: {k} pairs ", fontsize=8, color=MUTED,
                    transform=ax.get_yaxis_transform(), va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{prefix}_yield_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Where the strong pairs sit --------------------------------------------------------------
    pairs = ranked.drop_duplicates("pair_id").set_index("pair_id")
    backdrop = grid_pairs if grid_pairs is not None else pairs
    best = ranked.groupby("pair_id")["n_agree"].max()
    strong = best[best >= STRONG_TIER].index
    sp = pairs.loc[strong]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.scatter(backdrop["f_low"], backdrop["f_high"], s=11, color="#d9d9d9", linewidths=0,
               label=f"all {len(backdrop)} pairs in the grid")
    ax.scatter(sp["f_low"], sp["f_high"], s=30, color=POOL_COLOR, alpha=0.85, linewidths=0,
               label=f"a direction at n_agree \u2265 {STRONG_TIER} ({len(sp)} pairs)")
    if lit_pairs is not None and len(lit_pairs):
        ax.scatter(lit_pairs["f_low"], lit_pairs["f_high"], s=64, marker="D",
                   facecolor="#E69F00", edgecolor="#7a5300", linewidths=0.8, zorder=4,
                   label=f"the {len(lit_pairs)} literature pairs")
    ax.set_xscale("log"); ax.set_yscale("log")
    # Explicit decade-and-a-half ticks: the default log locator labels only 10^3 here, which
    # leaves a reader unable to place any point in Hz.
    ticks = [200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    for setter, labeller, lim in ((ax.set_xticks, ax.set_xticklabels, ax.set_xlim),
                                  (ax.set_yticks, ax.set_yticklabels, ax.set_ylim)):
        setter(ticks)
        labeller([f"{v:g}" for v in ticks], fontsize=8)
    ax.set_xlim(180, 8600); ax.set_ylim(180, 8600)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("f_low (Hz, log scale)"); ax.set_ylabel("f_high (Hz, log scale)")
    ax.set_title(f"Where the strong pairs sit in the frequency grid\n"
                 f"median {sp['semitones'].median():.1f} st / f_high "
                 f"{sp['f_high'].median():.0f} Hz, vs {backdrop['semitones'].median():.1f} st / "
                 f"{backdrop['f_high'].median():.0f} Hz grid-wide", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{prefix}_grid_position.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Direction asymmetry as a matrix ---------------------------------------------------------
    tiers = np.arange(n_models + 1)
    cross = pd.crosstab(gaps["n_agree_regular"], gaps["n_agree_counter"]).reindex(
        index=tiers, columns=tiers, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.grid(False)
    im = ax.imshow(cross.values, cmap="Blues", origin="lower", interpolation="nearest")
    ax.plot([-0.5, n_models + 0.5], [-0.5, n_models + 0.5], color="#bbbbbb", lw=0.9, zorder=3)
    for i in tiers:
        for j in tiers:
            v = int(cross.values[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 * cross.values.max() else INK)
    ax.set_xticks(tiers); ax.set_yticks(tiers)
    ax.set_xlabel("n_agree, counter direction"); ax.set_ylabel("n_agree, regular direction")
    same = float((gaps["n_agree_regular"] == gaps["n_agree_counter"]).mean() * 100)
    both_strong = int(((gaps["n_agree_regular"] >= STRONG_TIER)
                       & (gaps["n_agree_counter"] >= STRONG_TIER)).sum())
    ax.set_title(f"Each pair's two directions, cross-tabulated\n"
                 f"all {n_pairs_total} pairs in this ranking", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("pairs"); cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{prefix}_direction_matrix.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {prefix}_yield_curve.png, {prefix}_grid_position.png, "
          f"{prefix}_direction_matrix.png")
    print(f"  wrote {prefix}_cutoff_curve.csv, {prefix}_direction_asymmetry.csv")
    print(f"  asymmetry: {same:.0f}% of pairs same tier both ways; {both_strong} pairs reach "
          f">= {STRONG_TIER} in both directions; mean |gap| = {gaps['n_agree_gap'].mean():.2f} tiers")
    for t in range(n_models, 0, -1):
        print(f"  pairs with a direction at n_agree >= {t}: {int((best >= t).sum()):>4}")
    return curve, gaps


# ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, choices=(1, 2), default=1,
                   help="which phase to render: 1 reads phase1_mmn_s7_roi.csv and writes "
                        "phase1_* outputs, 2 reads phase2_mmn_s7_roi.csv and writes phase2_*. "
                        "Same criteria either way; only the number of deviants behind each "
                        "score differs (1 vs 15)")
    p.add_argument("--results_dir", default=str(RESULTS),
                   help="directory holding the phase's mmn_s7_roi.csv and grid_index.csv")
    p.add_argument("--out_dir", default=str(OUT),
                   help="where to write the figures and their table CSVs")
    p.add_argument("--predictions_root", default=str(PREDICTIONS),
                   help="prediction HDF5 root, for the waveform panels")
    p.add_argument("--literature_csv", default=str(LITERATURE),
                   help="the 24-method screen, scored on the same slice")
    p.add_argument("--skip_literature", action="store_true",
                   help="skip the novel-vs-literature comparison. Use it for a SELECTED subset: "
                        "the literature set is an unselected 48, so comparing tier shares against "
                        "a set chosen for scoring highly compares selection, not stimuli")
    p.add_argument("--chf_per_pair", type=float, default=CHF_PER_PAIR_PHASE2,
                   help="marginal cost per pair; used only to fill a column of the cutoff CSV. "
                        "Nothing in the memo or the figures carries a price")
    p.add_argument("--skip_waveforms", action="store_true",
                   help="skip the waveform panels (they read every prediction HDF5)")
    p.add_argument("--wave_chunk", type=int, default=0,
                   help="which chunk of 35 pairs gets per-model waveform figures, 0-indexed "
                        "(default 0 = the pairs in <prefix>_strong_waveforms_1.png). One figure "
                        "per model, so only one chunk is rendered")
    p.add_argument("--only_waveforms", action="store_true",
                   help="emit ONLY the waveform figures and exit. Use it for the --wave_units uv "
                        "pass: without it that run also regenerates every other figure, and any "
                        "flag it carries for the waveforms' sake (--skip_literature, say) "
                        "silently rewrites unrelated figures from a different configuration")
    p.add_argument("--skip_deviance", action="store_true",
                   help="skip the deviance-octile table and the deviance/frequency figure. Used "
                        "for a SELECTED subset, where the selection has already removed the "
                        "small-deviance instances the correlation lives in, so the number "
                        "measures the selection rather than the stimuli")
    p.add_argument("--wave_ylim", type=float, nargs=2, metavar=("LO", "HI"),
                   default=list(DIRECTION_YLIM),
                   help="shared y-window for the direction-only waveform panels "
                        f"(default {DIRECTION_YLIM[0]:g} {DIRECTION_YLIM[1]:g}); traces outside "
                        "it are clipped, not rescaled")
    p.add_argument("--wave_units", choices=("z", "uv"), default="z",
                   help="units for the direction-only waveform panels. 'z' (default) averages "
                        "the baseline-z-scored trace the S2 verdict uses and is comparable "
                        "across models; 'uv' averages raw microvolts, which Caveat 2 says are "
                        "not cross-model comparable")
    args = p.parse_args()

    cfg = PHASE[args.phase]
    prefix = cfg["prefix"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading {args.results_dir}/{cfg['s7']}  (phase {args.phase}, "
          f"{cfg['n_deviants']} deviant(s) per condition, prefix {prefix}_)")
    _, ranked = load_phase(args.results_dir, cfg["s7"], x=DIP_UV_THRESHOLD)
    models = [m for m in SEARCH_MODELS if f"s7__{m}" in ranked.columns]
    print(f"  marginal S7 rates: "
          f"{ {m: round(v, 3) for m, v in marginal_s7_rates(ranked, models).items()} }")

    # The whole frequency ladder and the whole pair list, so a phase covering a subset still
    # draws its heatmap and its grid scatter on the full-grid axes rather than on its own extent.
    grid_csv = Path(args.results_dir) / "grid_index.csv"
    grid_pairs = pd.read_csv(grid_csv) if grid_csv.exists() else None
    all_freqs = (sorted(set(grid_pairs["f_low"]) | set(grid_pairs["f_high"]))
                 if grid_pairs is not None else None)
    selected = grid_pairs is not None and ranked["pair_id"].nunique() < len(grid_pairs)
    if selected:
        print(f"  SELECTED SUBSET: {ranked['pair_id'].nunique()} of {len(grid_pairs)} pairs. "
              f"Chance baselines, deviance correlations and\n  literature comparisons over a "
              f"selected set measure the selection, not the stimuli.")

    # The other phase's rank for the same instances, so the top-N table can carry it.
    other = {}
    other_label = "phase2_rank" if args.phase == 1 else "phase1_rank"
    other_csv = Path(args.results_dir) / ("phase2_final_ranking.csv" if args.phase == 1
                                          else "phase1_ranked_directions.csv")
    if other_csv.exists():
        o = pd.read_csv(other_csv)
        other = {(int(a), b): int(r) for a, b, r in
                 zip(o["pair_id"], o["direction"], o["rank"])}
        print(f"  carrying {other_label} from {other_csv.name} ({len(other)} instances)")

    print("\n[1] agreement tiers")
    table_agreement_tiers(ranked, models, out_dir, prefix=prefix)

    print("\n[2] waveforms for the strong pairs")
    if args.skip_waveforms:
        print("  skipped (--skip_waveforms)")
    else:
        wave_kw = dict(predictions_root=args.predictions_root, prefix=prefix,
                       expect_n_deviants=cfg["n_deviants"])
        plot_strong_waveforms(ranked, models, out_dir, **wave_kw)
        plot_direction_waveforms(ranked, models, out_dir, units=args.wave_units,
                                 ylim=tuple(args.wave_ylim), **wave_kw)
        plot_per_model_waveforms(ranked, models, out_dir, chunk=args.wave_chunk, **wave_kw)
    if args.only_waveforms:
        print("\nstopped after the waveforms (--only_waveforms); no other output was rewritten")
        return

    print("\n[3] per-model µV over the strong set")
    plot_model_uv_box(ranked, models, out_dir, prefix=prefix)
    table_model_dissent(ranked, models, out_dir, prefix=prefix)

    print(f"\n[4] top-{N_TOP} direction-instances")
    table_topn(ranked, models, out_dir, n=N_TOP, prefix=prefix,
               other_rank=other or None, other_label=other_label)

    print("\n[5] mean-µV heatmap and frequency structure")
    _, uv_norm = plot_mean_uv_heatmap(ranked, models, out_dir, prefix=prefix,
                                      all_freqs=all_freqs)
    table_frequency_involvement(ranked, out_dir, prefix=prefix)
    if args.skip_deviance:
        print("  skipped the deviance octiles and the deviance/frequency figure "
              "(--skip_deviance)")
    else:
        table_deviance_octiles(ranked, out_dir, prefix=prefix)

    print("\n[6] literature comparison")
    lit = None
    if args.skip_literature:
        print("  skipped literature comparison (--skip_literature)")
    else:
        lit = load_literature(models, args.literature_csv)
        compare_to_literature(ranked, models, out_dir, lit, prefix=prefix)
        table_novel_vs_literature_top(ranked, lit, out_dir, n=10, prefix=prefix)
        plot_literature_heatmaps(lit, models, out_dir, uv_norm=uv_norm, prefix=prefix,
                                 ranked=ranked)

    print("\n[7] the structure of the ranking")
    table_yield(ranked, models, out_dir, prefix=prefix)
    plot_consensus_heatmap(ranked, models, out_dir, prefix=prefix,
                           thresholds=cfg["top_x"])
    # Phase 1 only: the literature overlay answers "where does the published set sit in this
    # grid", which is a property of the grid and is settled once. Repeating it on the Phase-2
    # scatter would add the same 24 diamonds to a figure about a selected subset of that grid.
    lit_pairs = (lit.drop_duplicates("pair_id")[["f_low", "f_high"]]
                 if args.phase == 1 and lit is not None and "f_low" in lit.columns else None)
    plot_ranking_structure(ranked, models, out_dir, chf_per_pair=args.chf_per_pair,
                           prefix=prefix, grid_pairs=grid_pairs, lit_pairs=lit_pairs)


if __name__ == "__main__":
    main()
