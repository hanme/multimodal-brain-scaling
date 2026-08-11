#!/usr/bin/env python
"""Plot 2 -- does the in-silico MMN trough deepen with deviance size (semitones)? FCz only.

One of the two "realness" dose-response checks. The human/clinical MMN deepens as the
standard->deviant frequency separation grows; this asks whether the model's does, over two
independent condition sources reported side by side and NEVER pooled into one regression.

Site: the FCz electrode, mTRF, S7@0.75-gated troughs, six models (whisper-large excluded).
Sources, sets, gate and style all come from mmn_dose_response_common.py -- see its docstring.

WHY THE COMBINED SET NEEDS THE OVERLAP DIAGNOSTIC
-------------------------------------------------
LIT spans 0.84-12 semitones and NOVEL-P2 spans 4.50-63, so LIT supplies almost all the low end
and NOVEL-P2 almost all the high end. A slope fitted across the union is therefore substantially
a BETWEEN-SOURCE contrast wearing a deviance label -- the two sources also differ in SOA,
tone duration, deviant probability and selection. Figure 4 is the test that separates them:
inside the 4.50-12 st overlap, where both sources have conditions, do they agree? If they do, the
combined slope is credible; if they diverge, the combined slope is a source effect and is
reported as one. The `lit_p2` rho is never reported without the overlap rhos beside it.

FIGURES (into this directory)
  1. deviance_pooled_s7gated.png        3 panels (lit | lit_p2 | p2), pooled over the 6 models,
                                        raw uV, n annotated; ONE pooled line per panel.
  2. deviance_per_model_s7gated__{set}.png   3 files x 6 panels: raw points + OLS + rho.
  3. deviance_s7_rate.png               3 panels: S7@0.75 count / TOTAL conditions per bin.
  4. deviance_overlap_lit_vs_p2.png     the 4.50-12 st diagnostic, two sources, two fits.
  + deviance_scaling_s7gated_stats.csv  Spearman rho per (set x model), (set x pooled), the
                                        per-source overlap rhos, and a <=24 st subset.

BINNING differs by set and is stated in every caption:
  lit     10 discrete semitone values -> mean +- SEM at each.
  p2      122 distinct values -> ~9 equal-count bins -> bin median with IQR.
  lit_p2  the p2 bin edges extended down to 0.84, so the whole LIT low end forms one leading bin.
The amplitude figure and the rate figure share the same bin edges within a set, so the censored
(amplitude) and uncensored (rate) views line up column for column.

The x-axis is LINEAR in semitones in every set, and each set keeps its own x-range. A log axis was
tried for the wider sets and rejected -- semitones are already log-frequency, and a log axis
compressed the >24 st region where the trend reverses. See the "X-AXIS IS LINEAR" comment in the
binning section for all three reasons.

NOT a replacement for deviance_scaling_plots_24freq_7models.py, which is the ungated, symlog,
median-based, 7-model, two-site cousin referenced by name from the parent memo and is untouched.

READ-ONLY over the scored CSVs. Usage:  python deviance_scaling_s7gated.py
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mmn_dose_response_common as C

N_P2_BINS = 9                      # equal-count bins over the NOVEL-P2 semitone range
LIT_EXT_EDGE = 0.84                # lit_p2 = the p2 edges extended down to LIT's minimum
POOLED_INK = "#333333"             # pooled series: colour is reserved for model identity
SUBSET_MAX_ST = 24.0               # 2 octaves -- above this is arguably a different sound
# Deep-tail clip. Medians sit in a -1.1..-2.4 uV band but individual troughs reach -9.6 uV, so
# letting the tail set a SHARED axis flattens every panel. The axis is clipped to cover all but
# the deepest CLIP_FRAC of troughs and everything past it is ANNOTATED, never silently dropped.
CLIP_FRAC = 0.02          # per-POINT clip, for the scatter figures
MIN_N_FOR_SCALE = 10      # a bin below this size cannot set the pooled figure's shared y-axis


# ------------------------------------------------------------------------------------------
# Binning
# ------------------------------------------------------------------------------------------
def bin_edges(tidy, set_key):
    """Bin edges for one set, computed over ALL its conditions (not just the gated ones).

    Shared by the amplitude and rate figures so their x-bins agree. `lit` gets no edges: it has
    only 10 distinct values and is plotted at each of them.
    """
    if set_key == "lit":
        return None
    p2_st = tidy[tidy["dataset"] == "p2"]["semitones"].to_numpy(float)
    q = np.linspace(0, 1, N_P2_BINS + 1)
    edges = np.unique(np.quantile(p2_st, q))
    edges[0] = np.floor(p2_st.min() * 100) / 100        # keep the minimum inside the first bin
    edges[-1] = np.ceil(p2_st.max() * 100) / 100
    if set_key == "lit_p2":
        edges = np.concatenate([[LIT_EXT_EDGE], edges])
    return edges


def assign_bins(frame, edges):
    """Add bin index + bin centre. With edges=None every distinct semitone value is its own bin."""
    out = frame.copy()
    if edges is None:
        out["bin_x"] = out["semitones"].round(4)
        out["bin_lo"] = out["bin_x"]
        out["bin_hi"] = out["bin_x"]
        return out
    idx = np.clip(np.digitize(out["semitones"].to_numpy(float), edges[1:-1], right=True),
                  0, len(edges) - 2)
    out["bin_lo"] = edges[idx]
    out["bin_hi"] = edges[idx + 1]
    out["bin_x"] = 0.5 * (out["bin_lo"] + out["bin_hi"])        # midpoint (linear x-axis)
    return out


def bin_caption(set_key, edges, short=False, kind="median"):
    """How this set is binned -- restated in every caption, because it differs by set."""
    if set_key == "lit":
        head = "10 discrete semitone values"
    elif set_key == "lit_p2":
        head = f"{len(edges) - 1} bins: the p2 equal-count edges + a {LIT_EXT_EDGE:g} st extension"
    else:
        head = f"{len(edges) - 1} equal-count bins"
    return head + ("" if short else f"; {C.CENTRAL_LABEL[kind]}")


# X-AXIS IS LINEAR IN SEMITONES, IN EVERY SET.
# A log axis was tried for the wider sets and rejected on three counts:
#   1. Semitones are ALREADY logarithmic in frequency (12*log2(f_dev/f_std)), so a log axis on
#      them is log-of-log in Hz. The whole reason semitones are the right unit is that they
#      linearize pitch distance; taking the log again undoes that for no perceptual gain.
#   2. It hides the result. NOVEL-P2 spans 4.5-63 st and the trend REVERSES above ~24 st; a log
#      axis compresses 24-63 into a narrow right-hand slice and flattens exactly the feature the
#      analysis turns on, while wasting the left half of the panel on 1-4 st where p2 has no data.
#   3. The stated justification (more even bins) does not hold: the lit_p2 bin widths span 7.5x
#      linearly against 11.7x in log2, so linear is the more even of the two.


def summarise(sub, set_key, kind="median"):
    """Per-bin summary of the gated troughs -- C.central for EVERY set (see its comment).

    `set_key` no longer selects the statistic; it is kept because callers pass it and because the
    BINNING still differs by set. `kind` picks median+bootstrap CI or the mean+-SEM companion.
    """
    rows = []
    for bx, g in sub.groupby("bin_x", observed=True):
        v = g["trough_uv"].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        centre, lo, hi = C.central(v, kind)
        rows.append(dict(bin_x=bx, centre=centre, lo=lo, hi=hi, n=v.size,
                         bin_lo=g["bin_lo"].iloc[0], bin_hi=g["bin_hi"].iloc[0]))
    return pd.DataFrame(rows).sort_values("bin_x")


def _decorate(ax, set_key, xrange=None, ylabel=True):
    """Shared axis furniture: the zero reference, the x-range, and labels. Linear x throughout."""
    ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
    if xrange is not None:
        span = xrange[1] - xrange[0]
        ax.set_xlim(xrange[0] - 0.06 * span, xrange[1] + 0.06 * span)
    ax.set_xlabel("Deviance size (semitones)")
    if ylabel:
        ax.set_ylabel("MMN trough (µV)\n↓ deeper")


def deep_clip(values, frac=CLIP_FRAC):
    """The deep-side axis limit that keeps all but the deepest `frac` of troughs on-axis."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, frac)) if v.size else float("nan")


def panel_title(ax, main, sub):
    """Bold panel title with a grey explanatory line under it (never overlapping it)."""
    ax.set_title(main, fontweight="bold", loc="left", pad=20)
    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.6, color="#6b6b6b", va="bottom")


# ------------------------------------------------------------------------------------------
# Figure 1 -- pooled over the six models
# ------------------------------------------------------------------------------------------
def pooled_ylim(tidy, gate):
    """One y-range covering BOTH summary statistics, so the two variants overlay exactly.

    Bins under MIN_N_FOR_SCALE are excluded: a bootstrap CI widens as n shrinks, and LIT's
    2-condition bins would otherwise drag every panel's axis down by ~2 uV.
    """
    los, his = [], []
    for kind in C.CENTRAL_KINDS:
        for set_key in ("lit", "lit_p2", "p2"):
            edges = bin_edges(tidy, set_key)
            s = summarise(assign_bins(C.gated(C.set_frame(tidy, set_key), gate), edges),
                          set_key, kind)
            big = s[s["n"] >= MIN_N_FOR_SCALE]
            if len(big):
                los.append(float(big["lo"].min()))
            his.append(float(s["hi"].max()))
    y_deep, y_shallow = min(los), max(his)
    pad = 0.06 * (y_shallow - y_deep)
    return y_deep - pad, y_shallow + pad



def fig_pooled(tidy, out_png, gate="s7", kind="median", ylim=None):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4))

    # One SHARED y-axis across the three sets so they are visually comparable.
    #
    # The limits come from what this figure actually DRAWS -- the per-bin summaries -- not from the
    # individual troughs behind them. Those are different distributions by an order of magnitude:
    # single troughs reach -9.6 uV while every bin summary sits within about -2.5 uV, so scaling to
    # the trough tail left ~60% of each panel empty and squashed the entire result into a strip.
    panels = {}
    summaries = {}
    for set_key in ("lit", "lit_p2", "p2"):
        edges = bin_edges(tidy, set_key)
        gat = assign_bins(C.gated(C.set_frame(tidy, set_key), gate), edges)
        panels[set_key] = (edges, gat)
        summaries[set_key] = summarise(gat, set_key, kind)

    # Only WELL-POPULATED bins set the shared scale. A bootstrap CI widens as n shrinks, so LIT's
    # 2-condition bins carry CIs an order of magnitude longer than everything else and would drag
    # the axis (and every other panel with it) down to -4.5 uV for a figure whose centres all sit
    # above -1.8. Small bins are still drawn -- their CI simply runs off the bottom, and any bin
    # whose CENTRE is off-axis is marked on the boundary with its true value.
    if ylim is not None:
        y_deep, y_shallow = ylim          # shared with the companion figure, so the two overlay
    else:
        big = [s[s["n"] >= MIN_N_FOR_SCALE] for s in summaries.values()]
        los = np.concatenate([b["lo"].to_numpy(float) for b in big if len(b)])
        his = np.concatenate([s["hi"].to_numpy(float) for s in summaries.values()])
        y_deep, y_shallow = float(los.min()), float(his.max())
        pad = 0.06 * (y_shallow - y_deep)
        y_deep, y_shallow = y_deep - pad, y_shallow + pad

    for ax, set_key in zip(axes, ("lit", "lit_p2", "p2")):
        edges, gat = panels[set_key]
        clipped = []

        # ONE pooled series per panel -- this is the POOLED figure, so each panel shows the
        # pooled result for its set, and lit_p2's line runs over the UNION of both sources.
        # It used to split that panel into a LIT series and a NOVEL-P2 series, which turned the
        # pooled figure into a source comparison; that is a different question and already has its
        # own figure (deviance_overlap_lit_vs_p2). The lit_p2 source composition is stated in the
        # caption instead of drawn.
        s = summaries[set_key]
        xs = s["bin_x"].to_numpy(float)
        ax.errorbar(xs, s["centre"], yerr=[s["centre"] - s["lo"], s["hi"] - s["centre"]],
                    color=POOLED_INK, marker="o", ms=7, lw=1.8, elinewidth=1.1, capsize=3,
                    mfc=POOLED_INK, mew=1.4, zorder=3)
        for x, y, n in zip(xs, s["centre"], s["n"]):
            ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7, color="#6b6b6b")
        # Any bin centre past the clip is drawn AT the boundary with its true value written
        # beside it -- clipped points stay visible and readable, never silently dropped.
        for x, y, n in zip(xs, s["centre"], s["n"]):
            if y < y_deep:
                ax.plot([x], [y_deep], marker="v", ms=8, color=POOLED_INK,
                        mfc=POOLED_INK, clip_on=False, zorder=6)
                ax.annotate(f"{y:.1f} µV\n(n={n})", (x, y_deep), textcoords="offset points",
                            xytext=(7, 2), ha="left", va="bottom", fontsize=7,
                            color=POOLED_INK, zorder=6)
                clipped.append((x, y, n))

        n_s7, n_tot = int(gat.shape[0]), len(C.set_frame(tidy, set_key))
        note = bin_caption(set_key, edges, kind=kind)
        if clipped:
            note += f"; {len(clipped)} bin(s) off-axis, labelled"
        panel_title(ax, f"{C.SET_LABEL[set_key]}  ({set_key})",
                    f"{C.gate_label(gate, long=False)} only: {n_s7}/{n_tot} rows\n{note}")
        xr = (gat["bin_x"].min(), gat["bin_x"].max())
        _decorate(ax, set_key, xrange=xr, ylabel=(set_key == "lit"))
        ax.set_ylim(y_deep, y_shallow)               # conventional: more negative = DOWN

    fig.suptitle(f"MMN trough vs deviance size at FCz — pooled over 6 models "
                 f"(mTRF, {C.gate_label(gate, long=False)}-gated)",
                 fontweight="bold", x=0.006, ha="left", y=1.02)
    fig.text(0.006, -0.02, C.wrap(
             f"Shared y-axis, scaled to the plotted bin summaries; bins under n={MIN_N_FOR_SCALE} "
             f"do not set the scale (a bootstrap CI widens as n shrinks), so their CIs may run off "
             f"the bottom. Any bin whose CENTRE is off-axis is drawn on the boundary with its true "
             f"value. Gate = {C.gate_label(gate)}: "
             + ("every plotted trough passes ≤ −0.75 µV, so each bin's shallow tail is absent and "
                "the slope is a LOWER BOUND on any amplitude effect."
                if gate == "s7" else
                "no µV floor, so the shallow tail that S7 removes is present here. This is the "
                "UNCENSORED companion to the S7 figure, and the gap between the two is what the "
                "floor was hiding.")
             + " lit_p2 is dominated by NOVEL-P2 rows and its two sources barely overlap in x — "
               "read it with the overlap diagnostic, not alone."),
             fontsize=7.8, color="#555555", ha="left", va="top")
    return C.finish(fig, out_png)


# ------------------------------------------------------------------------------------------
# Figure 2 -- small multiples, one panel per model
# ------------------------------------------------------------------------------------------
def fig_per_model(tidy, set_key, out_png, gate="s7"):
    gat = C.gated(C.set_frame(tidy, set_key), gate)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    y_deep = deep_clip(gat["trough_uv"])
    xr = (gat["semitones"].min(), gat["semitones"].max())

    for ax, model in zip(axes.ravel(), C.MODEL_ORDER):
        st = C.style(model)
        sub = gat[gat["model"] == model]
        for ds, g in sub.groupby("dataset", observed=True):
            x, y = g["semitones"].to_numpy(float), g["trough_uv"].to_numpy(float)
            xj = x + rng.uniform(-0.25, 0.25, x.size)
            mk = C.DATASET_MARKER[ds] if set_key == "lit_p2" else st["marker"]
            on = y >= y_deep
            ax.scatter(xj[on], y[on], s=17, color=st["color"], alpha=0.42, marker=mk,
                       edgecolors="white", linewidths=0.3, zorder=3)
            # Off-scale points get a hollow up-caret ON the boundary: visible and located, but
            # plainly not a data value at that depth. Their true depths still drive rho and OLS.
            if (~on).any():
                ax.scatter(xj[~on], np.full((~on).sum(), y_deep), s=22, marker="v",
                           facecolors="none", edgecolors=st["color"], linewidths=0.9,
                           alpha=0.75, clip_on=False, zorder=4)
        x, y = sub["semitones"].to_numpy(float), sub["trough_uv"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            sl, ic = np.polyfit(x[ok], y[ok], 1)     # OLS in LINEAR semitones, drawn on the axis
            xs = np.linspace(x[ok].min(), x[ok].max(), 200)
            ax.plot(xs, ic + sl * xs, color=st["color"], ls=st["ls"], lw=1.8, zorder=4)
        rho, p, n = C.spearman(x, y)
        n_clip = int((y < y_deep).sum())
        ax.set_title(model, fontweight="bold", loc="left", fontsize=10)
        # Top-left is the deep/low-deviance corner and is empty in every panel, so the stat block
        # sits there rather than over the dense shallow band along the bottom.
        ax.text(0.025, 0.955, f"ρ={rho:+.2f}   p={p:.3g}   n={n}"
                              + (f"\n▽ {n_clip} off-scale" if n_clip else ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=8.4, color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        _decorate(ax, set_key, xrange=xr, ylabel=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("MMN trough (µV)\n↓ deeper")
    axes[0, 0].set_ylim(y_deep, -0.6)                # conventional; shared across 6 panels
    for ax in axes[0, :]:
        ax.set_xlabel("")

    extra = ("  Marker shape = source (● LIT, ▲ NOVEL-P2)." if set_key == "lit_p2" else "")
    rho_note = "rank-based, computed on UNCLIPPED values"
    fig.suptitle(f"MMN trough vs deviance at FCz — per model — {C.SET_LABEL[set_key]} ({set_key})",
                 fontweight="bold", x=0.006, ha="left", y=1.015)
    fig.text(0.006, -0.015, C.wrap(
             f"{C.gate_label(gate, long=False)}-gated troughs, mTRF. Shared y-axis across all six panels, clipped at the "
             f"deepest {CLIP_FRAC:.0%} of the set's troughs (△ = points past it, drawn on the "
             f"boundary and counted per panel). Line = OLS in linear semitones; ρ = Spearman "
             f"({rho_note}).{extra}"),
             fontsize=7.8, color="#555555", ha="left", va="top")
    return C.finish(fig, out_png)


# ------------------------------------------------------------------------------------------
# Figure 3 -- the UNCENSORED view: S7@0.75 rate per bin
# ------------------------------------------------------------------------------------------
def fig_rate(tidy, out_png, gate="s7"):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4), sharey=True)
    handles = []
    for ax, set_key in zip(axes, ("lit", "lit_p2", "p2")):
        edges = bin_edges(tidy, set_key)
        allrows = assign_bins(C.set_frame(tidy, set_key), edges)
        for model in C.MODEL_ORDER:
            st = C.style(model)
            r = C.pass_rate(allrows[allrows["model"] == model], ["bin_x"], gate).sort_values("bin_x")
            ax.plot(r["bin_x"], r["rate"], color=st["color"], marker=st["marker"], ls=st["ls"],
                    lw=1.1, ms=5.0, alpha=0.55, mec="white", mew=0.5, zorder=3)
        pooled = C.pass_rate(allrows, ["bin_x"], gate).sort_values("bin_x")
        ax.plot(pooled["bin_x"], pooled["rate"], color=POOLED_INK, marker="o", ls="-", lw=2.8,
                ms=7.5, mec="white", mew=1.0, zorder=5)

        # LIT's 10 discrete values carry wildly different condition counts, so each is labelled.
        # The equal-count sets have a near-constant denominator -- stated once, in the subtitle.
        nt = pooled["n_total"]
        if set_key == "lit":
            for x, y, n in zip(pooled["bin_x"], pooled["rate"], nt):
                ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=7, color="#6b6b6b")
            denom = "denominator labelled per point"
        else:
            denom = (f"{int(nt.min())}–{int(nt.max())} conditions per bin"
                     if nt.min() != nt.max() else f"{int(nt.min())} conditions per bin")
        ax.set_ylim(0, 1.03)
        panel_title(ax, f"{C.SET_LABEL[set_key]}  ({set_key})",
                    f"{bin_caption(set_key, edges, short=True)}\n{denom}")
        _decorate(ax, set_key, xrange=(pooled["bin_x"].min(), pooled["bin_x"].max()), ylabel=False)
        ax.set_ylabel(f"{C.gate_label(gate, long=False)} rate  (pass / all conditions)"
                      if set_key == "lit" else "")

    handles = [Line2D([0], [0], color=C.style(m)["color"], marker=C.style(m)["marker"],
                      ls=C.style(m)["ls"], lw=1.1, ms=5.5, mec="white", label=m)
               for m in C.MODEL_ORDER]
    handles.append(Line2D([0], [0], color=POOLED_INK, marker="o", ls="-", lw=2.8, ms=7.5,
                          mec="white", label="pooled (6 models)"))
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=8.5,
               ncol=len(handles), bbox_to_anchor=(0.5, 1.055))
    fig.suptitle(f"{C.gate_label(gate, long=False)} rate vs deviance size at FCz — denominator is ALL conditions",
                 fontweight="bold", x=0.006, ha="left", y=1.10)
    fig.text(0.006, -0.02, C.wrap(
             "The only UNCENSORED view in this deliverable: a count outcome is not floored by the "
             "−0.75 µV gate, so a dose-response can appear here that the gated amplitude axis "
             "cannot show. Bin edges match the amplitude figure within each set. LIT's per-model "
             "lines swing between 0 and 1 because most of its semitone values carry only 2 "
             "conditions — read the pooled line there."),
             fontsize=7.8, color="#555555", ha="left", va="top")
    return C.finish(fig, out_png)


# ------------------------------------------------------------------------------------------
# Figure 4 -- the overlap diagnostic: is the lit_p2 slope deviance, or source?
# ------------------------------------------------------------------------------------------
def fig_overlap(tidy, out_png, gate="s7"):
    lo, hi = C.OVERLAP_ST
    frame = tidy[(tidy["semitones"] >= lo) & (tidy["semitones"] <= hi)]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    rng = np.random.default_rng(0)
    lines = []
    y_deep = deep_clip(C.gated(frame, gate)["trough_uv"], CLIP_FRAC)
    n_off = 0
    for ds in ("lit", "p2"):
        g = C.gated(frame[frame["dataset"] == ds], gate)
        x, y = g["semitones"].to_numpy(float), g["trough_uv"].to_numpy(float)
        mk = C.DATASET_MARKER[ds]
        xj = x + rng.uniform(-0.08, 0.08, x.size)
        on = y >= y_deep
        ax.scatter(xj[on], y[on], s=26, marker=mk,
                   facecolors="none" if ds == "p2" else POOLED_INK,
                   edgecolors=POOLED_INK, linewidths=1.0, alpha=0.55, zorder=3)
        # Off-scale points sit on the bottom edge as hollow down-carets: located and counted,
        # but plainly not a value at that depth. Their true depths still drive rho and the fit.
        if (~on).any():
            ax.scatter(xj[~on], np.full((~on).sum(), y_deep), s=26, marker="v",
                       facecolors="none", edgecolors=POOLED_INK, linewidths=1.0,
                       alpha=0.75, clip_on=False, zorder=4)
            n_off += int((~on).sum())
        ok = np.isfinite(x) & np.isfinite(y)
        sl, ic = np.polyfit(x[ok], y[ok], 1)
        # Source is carried by MARKER SHAPE here, as in every other panel. Every line in this
        # deliverable is solid, so both fits are solid and the marker riding each one names it.
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, ic + sl * xs, color=POOLED_INK, lw=2.0, ls="-", marker=mk, ms=7,
                markevery=18, mfc="none" if ds == "p2" else POOLED_INK, mew=1.4, zorder=4)
        rho, p, n = C.spearman(x, y)
        n_all = int((frame["dataset"] == ds).sum())
        lines.append(Line2D([0], [0], color=POOLED_INK, marker=mk, lw=2.0, ls="-",
                            mfc="none" if ds == "p2" else POOLED_INK,
                            label=(f"{C.DATASET_LABEL[ds]}: ρ={rho:+.2f}, p={p:.3g}, "
                                   f"n={n} of {n_all} rows")))
    ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
    ax.set_xlabel("Deviance size (semitones)")
    ax.set_ylabel("MMN trough (µV)\n↓ deeper")
    ax.set_ylim(y_deep, -0.5)
    ax.legend(handles=lines, fontsize=9, loc="lower left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.85)
    ax.set_title(f"Overlap diagnostic — LIT vs NOVEL-P2 within {lo:g}–{hi:g} semitones at FCz",
                 fontweight="bold", loc="left")
    fig.text(0.0, -0.115, C.wrap(
             f"The one region where both sources have conditions, so the two can be compared "
             f"WITHOUT the range confound. Pooled over the 6 models, "
             f"{C.gate_label(gate, long=False)}-gated, mTRF; lines are per-source OLS, labelled "
             f"by the marker riding them (● LIT, △ NOVEL-P2). DESCENDING left-to-right = deeper "
             f"with more deviance."
             + (f" ▽ {n_off} point(s) lie deeper than the axis and are drawn on the bottom edge; "
                f"they still count in ρ and the fits." if n_off else "") + "\n"
             "If the two agree here, the lit_p2 combined slope is credible as a deviance effect; "
             "if they diverge, that slope is a source effect — the sources also differ in SOA, "
             "tone duration, deviant probability, and selection.", 96),
             transform=ax.transAxes, fontsize=7.8, color="#555555", va="top")
    return C.finish(fig, out_png)


# ------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------
def stats_table(tidy, gate="s7"):
    """Spearman rho of trough_uv vs semitones for every cell reported in the memo."""
    rows = []

    def add(set_key, model, subset, frame):
        g = C.gated(frame, gate)
        rho, p, n = C.spearman(g["semitones"], g["trough_uv"])
        rows.append(dict(set=set_key, model=model, subset=subset, rho=rho, p=p, n_s7=n,
                         n_total=len(frame),
                         median_trough_uv=float(np.median(g["trough_uv"])) if n else float("nan")))

    for set_key in ("lit", "lit_p2", "p2"):
        f = C.set_frame(tidy, set_key)
        add(set_key, "POOLED", "all", f)
        for m in C.MODEL_ORDER:
            add(set_key, m, "all", f[f["model"] == m])
        # <=24 st (2 octaves): above this a "deviant" is arguably a different sound, not a deviant.
        sub = f[f["semitones"] <= SUBSET_MAX_ST]
        if len(sub) and sub["semitones"].nunique() > 1:
            add(set_key, "POOLED", f"<= {SUBSET_MAX_ST:g} st", sub)
            for m in C.MODEL_ORDER:
                add(set_key, m, f"<= {SUBSET_MAX_ST:g} st", sub[sub["model"] == m])

    # The overlap diagnostic, per source -- the rows that qualify the lit_p2 slope.
    lo, hi = C.OVERLAP_ST
    ov = tidy[(tidy["semitones"] >= lo) & (tidy["semitones"] <= hi)]
    label = f"overlap {lo:g}-{hi:g} st"
    for ds in ("lit", "p2"):
        add(ds, "POOLED", label, ov[ov["dataset"] == ds])
        for m in C.MODEL_ORDER:
            add(ds, m, label, ov[(ov["dataset"] == ds) & (ov["model"] == m)])
    add("lit_p2", "POOLED", label, ov)
    return pd.DataFrame(rows)


def print_summary(tidy, st, gate="s7"):
    lo, hi = C.OVERLAP_ST
    print("\n" + "=" * 86)
    print(f"Spearman rho: trough_uv vs semitones, FCz, mTRF, {C.gate_label(gate, long=False)}-gated (negative rho = "
          "DEEPER with more deviance = the MMN-like direction)")
    print("=" * 86)
    for subset in ("all", f"<= {SUBSET_MAX_ST:g} st"):
        print(f"\n-- subset: {subset} --")
        print(f"  {'model':<17}" + "".join(f"{k:>22}" for k in ("lit", "lit_p2", "p2")))
        for m in ["POOLED"] + C.MODEL_ORDER:
            cells = []
            for k in ("lit", "lit_p2", "p2"):
                r = st[(st["set"] == k) & (st["model"] == m) & (st["subset"] == subset)]
                cells.append(f"{r.iloc[0].rho:+.2f} (p={r.iloc[0].p:.3g}, n={int(r.iloc[0].n_s7)})"
                             if len(r) else "--")
            print(f"  {m:<17}" + "".join(f"{c:>22}" for c in cells))

    print(f"\n-- OVERLAP {lo:g}-{hi:g} st (the qualifier on any lit_p2 slope) --")
    lab = f"overlap {lo:g}-{hi:g} st"
    for ds in ("lit", "p2", "lit_p2"):
        r = st[(st["set"] == ds) & (st["model"] == "POOLED") & (st["subset"] == lab)].iloc[0]
        n_pass = int(r.n_s7)
        print(f"  {C.DATASET_LABEL.get(ds, 'COMBINED'):<26} rho={r.rho:+.3f}  p={r.p:.3g}  "
              f"S7 {n_pass}/{int(r.n_total)} rows  median {r.median_trough_uv:+.2f} uV")
    rl = st[(st["set"] == "lit") & (st.model == "POOLED") & (st.subset == lab)].iloc[0]
    rp = st[(st["set"] == "p2") & (st.model == "POOLED") & (st.subset == lab)].iloc[0]
    agree = (np.sign(rl.rho) == np.sign(rp.rho))
    both_null = (rl.p >= 0.05) and (rp.p >= 0.05)
    print(f"  -> sources {'AGREE' if agree else 'DISAGREE'} on sign inside the overlap "
          f"(rho {rl.rho:+.3f} vs {rp.rho:+.3f}).")
    if agree and both_null:
        # Agreeing on the sign of two effects that are each indistinguishable from zero is weak
        # evidence, and saying only "they agree" would overstate it.
        print("     BUT both are non-significant, so this is agreement on a NULL: it rules out a "
              "source ARTEFACT driving the combined slope, and does not establish a deviance "
              "effect inside the overlap.")
    elif agree:
        print("     Both are significant and same-signed: the lit_p2 slope is credible as a "
              "deviance effect.")
    else:
        print("     The sources disagree, so the lit_p2 slope is NOT interpretable as a deviance "
              "effect and must be reported as a source effect.")

    print("\n" + "=" * 86)
    print(f"{C.gate_label(gate, long=False)} rate by set (count / ALL conditions)")
    print("=" * 86)
    for k in ("lit", "lit_p2", "p2"):
        f = C.set_frame(tidy, k)
        print(f"  {k:<8} {int(f['s7'].sum()):>5}/{len(f):<6} = {f['s7'].mean():.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(C.PLOTS_DIR),
                   help="where the PNGs go; SVGs go to the sibling svgs/")
    p.add_argument("--csv_dir", default=str(C.ANALYSIS_DIR),
                   help="where the stats CSV goes (data, not a figure)")
    p.add_argument("--gate", default="s7", choices=("s7", "s2"),
                   help="amplitude gate. s7 = shape + 0.75 uV floor (the headline). s2 = shape "
                        "only, no floor -- the UNCENSORED companion, whose outputs all take an "
                        "__s2 suffix so the two sets never overwrite each other.")
    args = p.parse_args()
    gate, sfx = args.gate, ("" if args.gate == "s7" else f"__{args.gate}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tidy = C.load_tidy()
    print(f"Loaded {len(tidy)} FCz mTRF rows @ X=0.75 over {len(C.MODEL_ORDER)} models "
          f"({tidy[tidy.dataset == 'lit'].method.nunique()} LIT + "
          f"{tidy[tidy.dataset == 'p2'].method.nunique()} NOVEL-P2 conditions)")

    # Overlap-region counts are asserted, not just plotted: they are what makes figure 4 mean
    # anything, and a silent change in either source would gut the diagnostic.
    lo, hi = C.OVERLAP_ST
    ov = tidy[(tidy["semitones"] >= lo) & (tidy["semitones"] <= hi)]
    for ds, n_rows, n_s7 in ((("lit", 120, 65), ("p2", 336, 189)) if gate == "s7" else ()):
        d = ov[ov["dataset"] == ds]
        assert (len(d), int(d["s7"].sum())) == (n_rows, n_s7), \
            f"overlap {ds}: {len(d)} rows / {int(d['s7'].sum())} S7, expected {n_rows}/{n_s7}"
    print(f"Overlap {lo:g}-{hi:g} st verified: LIT 120 rows/65 S7, NOVEL-P2 336 rows/189 S7")

    print("\nFigures:")
    root = out if args.out_dir != str(C.PLOTS_DIR) else None
    # Both summary statistics, on ONE shared axis so they can be compared directly.
    ylim = pooled_ylim(tidy, gate)
    for kind in C.CENTRAL_KINDS:
        fig_pooled(tidy, C.fig_path(gate, None, f"deviance_pooled{C.CENTRAL_SUFFIX[kind]}", root),
                   gate, kind, ylim)
    for set_key in ("lit", "lit_p2", "p2"):
        fig_per_model(tidy, set_key,
                      C.fig_path(gate, set_key, f"deviance_per_model__{set_key}", root), gate)
    fig_rate(tidy, C.fig_path(gate, None, "deviance_pass_rate", root), gate)
    fig_overlap(tidy, C.fig_path(gate, "lit_p2", "deviance_overlap_lit_vs_p2", root), gate)

    st = stats_table(tidy, gate)
    st_path = Path(args.csv_dir) / f"deviance_scaling_s7gated_stats{sfx}.csv"
    st.to_csv(st_path, index=False, float_format="%.6g")
    print(f"  wrote {st_path}  ({len(st)} rows)")
    print_summary(tidy, st, gate)


if __name__ == "__main__":
    main()
