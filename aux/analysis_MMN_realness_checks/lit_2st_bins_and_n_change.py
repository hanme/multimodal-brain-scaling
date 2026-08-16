#!/usr/bin/env python
"""LIT-only supplementary views: 2-semitone deviance bins, and an N-change table.

Three deliverables, all restricted to the LITERATURE condition set (24 Frequency methods x
{regular, counter} = 48 conditions), at FCz, mTRF, over the six models:

  1. lit_deviance_2st_trough.png     mean MMN trough +- 95% CI per 2-semitone bin.
  2. lit_deviance_2st_pass_rate.png  mean per-stimulus pass rate (x/15) +- 95% CI per bin.
  3. lit_n_change_table.csv          mean change in trough from N=3->5, 3->7 and 5->7.

WHICH FILE FEEDS WHICH. (1) comes from the condition-level scored CSV
(`results_soafix_full/mmn_s7_roi.csv`), whose rows are already the 15-trial average. (2) and (3)
come from the PER-TRIAL CSV (`results_soafix_full/mmn_per_trial_n_fcz.csv`): a "x out of 15" pass
rate across the 3 N-levels x 5 variations does not exist at the condition level, and neither does
a per-N trough.

THE BINS. LIT carries 10 distinct semitone values over 0.84-12.0, so 2-semitone bins give six,
of which **[4,6) is empty** -- there is no literature method between 3.16 and 7.02 st. The top bin
is closed, [10,12], so the single 12.0 st method is included rather than dropped. Bin occupancy is
very uneven (4, 10, 0, 3, 1, 6 methods), which is a property of the literature set, not a choice:
10 of the 24 methods sit at 3.16 st alone. Every bin is annotated with its n for that reason.

THE STATISTIC HERE IS THE MEAN +- 95% CI, not the median + bootstrap CI used by the main figures
in this directory. That is deliberate -- it is what was asked for, and these are supplementary
views. The trough distributions are strongly skewed (see the memo), so these means sit deeper than
the corresponding medians; do not mix numbers between the two figure sets.

Usage:  python lit_2st_bins_and_n_change.py [--gate s7|s2]
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import mmn_dose_response_common as C

# Six 2-semitone bins over LIT's 0.84-12.0 st. The top edge is inclusive so the 12.0 st method
# lands in [10,12] instead of falling outside every bin.
BIN_EDGES = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
BIN_LABELS = ["[0,2)", "[2,4)", "[4,6)", "[6,8)", "[8,10)", "[10,12]"]
INK = "#333333"


def assign_2st_bins(frame):
    """Add `bin` (ordered categorical) and `bin_mid` to a LIT frame."""
    out = frame.copy()
    # right=False makes every bin half-open; the final bin is then closed by hand.
    idx = np.digitize(out["semitones"].to_numpy(float), BIN_EDGES[1:-1], right=False)
    idx = np.clip(idx, 0, len(BIN_LABELS) - 1)
    out["bin"] = pd.Categorical([BIN_LABELS[i] for i in idx], categories=BIN_LABELS, ordered=True)
    out["bin_mid"] = (BIN_EDGES[:-1] + 1.0)[idx]
    return out


def mean_ci(values, alpha=0.05):
    """(mean, lo, hi) with a t-based 95% CI of the mean. n<2 -> a point with no interval."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if v.size < 2:
        return m, m, m, int(v.size)
    half = stats.t.ppf(1 - alpha / 2, v.size - 1) * (v.std(ddof=1) / np.sqrt(v.size))
    return m, m - half, m + half, int(v.size)


def _bin_axis(ax):
    ax.set_xticks([BIN_EDGES[i] + 1.0 for i in range(len(BIN_LABELS))])
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlim(BIN_EDGES[0] - 0.6, BIN_EDGES[-1] + 0.6)
    ax.set_xlabel("Deviance size (semitones, 2-st bins)")


# ------------------------------------------------------------------------------------------
# 1. Mean trough per 2-semitone bin
# ------------------------------------------------------------------------------------------
def fig_trough(tidy, out_png, gate="s7"):
    lit_all = assign_2st_bins(C.set_frame(tidy, "lit"))     # denominator: every LIT condition
    lit = lit_all[lit_all[C.GATES[gate][0]]]                # numerator: the gate-passing rows
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    rows = []
    for lab in BIN_LABELS:
        g = lit[lit["bin"] == lab]
        m, lo, hi, n = mean_ci(g["trough_uv"])
        # n_conditions is EVERY LIT condition in the bin, so it matches the pass-rate table;
        # n_conditions_passing is how many of them contributed at least one gated row.
        rows.append(dict(bin=lab, mean_uv=m, ci_lo=lo, ci_hi=hi, n_rows=n,
                         n_conditions=lit_all.loc[lit_all["bin"] == lab, "method"].nunique(),
                         n_conditions_passing=g["method"].nunique()))
        if n == 0:
            continue
        x = BIN_EDGES[BIN_LABELS.index(lab)] + 1.0
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], color=INK, marker="o", ms=8,
                    lw=0, elinewidth=1.4, capsize=5, mfc=INK, zorder=3)
        ax.annotate(f"n={n}", (x, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color="#6b6b6b")

    summ = pd.DataFrame(rows)
    xs = [BIN_EDGES[BIN_LABELS.index(l)] + 1.0 for l in summ.dropna(subset=["mean_uv"])["bin"]]
    ax.plot(xs, summ.dropna(subset=["mean_uv"])["mean_uv"], color=INK, lw=1.6, zorder=2)

    empty = [l for l in BIN_LABELS if summ.loc[summ["bin"] == l, "n_rows"].iloc[0] == 0]
    for lab in empty:
        ax.annotate("no LIT\nconditions", (BIN_EDGES[BIN_LABELS.index(lab)] + 1.0, 0.5),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="center", fontsize=8, color="#999999", style="italic")

    ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
    _bin_axis(ax)
    ax.set_ylabel("MMN trough (µV)\n↓ deeper")
    ax.set_title(f"LIT — MMN trough by deviance, 2-semitone bins "
                 f"({C.gate_label(gate, long=False)}-gated)", fontweight="bold", loc="left")
    fig.text(0.0, -0.19, C.wrap(
        f"Mean ± 95% CI of the trough within each 2-semitone bin, over the "
        f"{C.gate_label(gate, long=False)}-passing rows of all 6 models at FCz (mTRF); n = rows "
        f"per bin. Source: results_soafix_full/mmn_s7_roi.csv. Bin occupancy is very uneven — 10 "
        f"of LIT's 24 methods sit at 3.16 st and none fall between 3.16 and 7.02 st, so [4,6) is "
        f"empty. NOTE this is a MEAN ± t-based CI, not the median + bootstrap CI used by the main "
        f"figures in this directory; the trough distribution is strongly skewed, so these means "
        f"sit deeper than the corresponding medians and the two sets should not be mixed."),
        transform=ax.transAxes, fontsize=7.8, color="#555555", va="top")
    C.finish(fig, out_png)
    return summ


# ------------------------------------------------------------------------------------------
# 2. Mean per-stimulus pass rate per 2-semitone bin
# ------------------------------------------------------------------------------------------
def fig_pass_rate(per_trial, tidy, out_png, gate="s7"):
    """Per (model, condition): passes / 15 over the 3 N-levels x 5 variations. Then per bin,
    the mean of those rates with a 95% CI."""
    lit = C.set_frame(per_trial, "lit")
    st = (C.set_frame(tidy, "lit")[["method", "semitones"]].drop_duplicates())
    col = C.GATES[gate][0]

    rate = (lit.groupby(["model", "method"], observed=True)[col]
            .agg(n_pass="sum", n_trials="size").reset_index())
    assert (rate["n_trials"] == 15).all(), "expected 15 trials per (model, condition)"
    rate["pass_rate"] = rate["n_pass"] / rate["n_trials"]
    rate = assign_2st_bins(rate.merge(st, on="method", how="left"))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    rows = []
    for lab in BIN_LABELS:
        g = rate[rate["bin"] == lab]
        m, lo, hi, n = mean_ci(g["pass_rate"])
        rows.append(dict(bin=lab, mean_pass_rate=m, ci_lo=lo, ci_hi=hi,
                         n_model_conditions=n, n_conditions=g["method"].nunique()))
        if n == 0:
            continue
        x = BIN_EDGES[BIN_LABELS.index(lab)] + 1.0
        ax.errorbar([x], [100 * m], yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=INK,
                    marker="o", ms=8, lw=0, elinewidth=1.4, capsize=5, mfc=INK, zorder=3)
        ax.annotate(f"n={n}", (x, 100 * m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color="#6b6b6b")

    summ = pd.DataFrame(rows)
    ok = summ.dropna(subset=["mean_pass_rate"])
    ax.plot([BIN_EDGES[BIN_LABELS.index(l)] + 1.0 for l in ok["bin"]],
            100 * ok["mean_pass_rate"], color=INK, lw=1.6, zorder=2)
    for lab in summ.loc[summ["n_model_conditions"] == 0, "bin"]:
        ax.annotate("no LIT\nconditions", (BIN_EDGES[BIN_LABELS.index(lab)] + 1.0, 50),
                    ha="center", va="center", fontsize=8, color="#999999", style="italic")

    ax.set_ylim(0, 100)
    _bin_axis(ax)
    ax.set_ylabel(f"{C.gate_label(gate, long=False)} pass rate (% of the 15 trials)")
    ax.set_title(f"LIT — MMN pass rate by deviance, 2-semitone bins "
                 f"({C.gate_label(gate, long=False)})", fontweight="bold", loc="left")
    fig.text(0.0, -0.19, C.wrap(
        f"Each (model, condition) contributes ONE pass rate = passing trials / 15, over the 3 "
        f"N-levels × 5 variations. The plotted point is the mean of those rates within the bin, "
        f"± a 95% CI; n = (model × condition) pairs. Source: "
        f"results_soafix_full/mmn_per_trial_n_fcz.csv — a x/15 rate does not exist in the "
        f"condition-level CSV, whose rows are already the 15-trial average. Unlike the gated "
        f"trough, this is a count outcome and so is not floored by the µV threshold."),
        transform=ax.transAxes, fontsize=7.8, color="#555555", va="top")
    C.finish(fig, out_png)
    return summ


# ------------------------------------------------------------------------------------------
# 3. Mean change in trough across N
# ------------------------------------------------------------------------------------------
def n_change_table(per_trial, gate="s7"):
    """Mean trough change N3->N5, N3->N7, N5->N7 for stimuli passing at ALL THREE N levels.

    QUALIFICATION: a (model, condition) cell is included only if it has at least one gate-passing
    trial at N=3 AND N=5 AND N=7.

    VALUE: once a cell qualifies, the trough at each N is the mean over ALL FIVE variations, not
    only the passing ones. So N=5 troughs of -1.3, -1.2, -0.8, -0.7, -1.0 give -1.0 even though
    only four of the five clear a 0.75 uV floor.

    This differs deliberately from n_effect_change_table.csv, which takes the MEDIAN of only the
    PASSING trials -- so the two tables will not agree, and the difference is the shallow trials
    this one admits.
    """
    lit = C.set_frame(per_trial, "lit")
    col = C.GATES[gate][0]

    passed = lit[lit[col]]
    n_levels = passed.groupby(["model", "method"], observed=True)["N"].nunique()
    qualified = set(n_levels[n_levels == len(C.N_LEVELS)].index)

    keep = lit[[(m, c) in qualified for m, c in zip(lit["model"], lit["method"])]]
    cell = (keep.groupby(["model", "method", "N"], observed=True)["trough_uv"]
            .agg(mean_uv="mean", n_trials="size").reset_index())
    assert (cell["n_trials"] == 5).all(), "expected 5 variations per (model, condition, N)"

    wide = cell.pivot_table(index=["model", "method"], columns="N", values="mean_uv")
    wide = wide.dropna(subset=list(C.N_LEVELS))

    rows = []
    for model in ["POOLED"] + C.MODEL_ORDER:
        w = wide if model == "POOLED" else wide[wide.index.get_level_values("model") == model]
        if w.empty:
            continue
        rec = dict(model=model, n_stimuli=len(w),
                   mean_N3=w[3].mean(), mean_N5=w[5].mean(), mean_N7=w[7].mean())
        for a, b in ((3, 5), (3, 7), (5, 7)):
            d = (w[b] - w[a])
            m, lo, hi, n = mean_ci(d)
            rec[f"d_N{a}_to_N{b}"] = m
            rec[f"d_N{a}_to_N{b}_ci_lo"] = lo
            rec[f"d_N{a}_to_N{b}_ci_hi"] = hi
            rec[f"d_N{a}_to_N{b}_pct_deeper"] = 100.0 * float((d < 0).mean())
            rec[f"d_N{a}_to_N{b}_p"] = (float(stats.wilcoxon(w[b], w[a]).pvalue)
                                        if len(w) >= 10 else float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows)


CONTRASTS = [(3, 5), (3, 7), (5, 7)]


def fig_change_trajectory(ct, out_png, gate="s7"):
    """The LEVELS in the table: mean trough at each N, one line per model plus pooled."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    xs = list(range(len(C.N_LEVELS)))
    for _, r in ct.iterrows():
        pooled = r.model == "POOLED"
        st = dict(color=INK, marker="o", ls="-") if pooled else C.style(r.model)
        ax.plot(xs, [r.mean_N3, r.mean_N5, r.mean_N7], color=st["color"], marker=st["marker"],
                ls=st["ls"], lw=3.0 if pooled else 1.5, ms=9 if pooled else 6,
                alpha=1.0 if pooled else 0.75, mec="white", mew=0.8,
                zorder=5 if pooled else 3,
                label=f"{r.model} (n={int(r.n_stimuli)})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n({C.N_TO_P_DEVIANT[n]:.1%})" for n in C.N_LEVELS])
    ax.set_xlim(-0.35, len(C.N_LEVELS) - 0.65)
    ax.set_xlabel("N standards between deviants\n(oddball probability)")
    ax.set_ylabel("Mean MMN trough (µV)\n↓ deeper")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"LIT — mean trough across N, per model "
                 f"({C.gate_label(gate, long=False)})", fontweight="bold", loc="left")
    fig.text(0.0, -0.19, C.wrap(
        "The LEVELS behind lit_n_change_table.csv. Only stimuli passing at N=3 AND 5 AND 7 are "
        "included; each stimulus's trough at a given N is the mean over ALL five variations, "
        "including any that fail the µV floor. Lines are means over stimuli, so the vertical "
        "offsets between models are amplitude-scale differences, not effects — read the SLOPES. "
        "The companion forest figure carries the confidence intervals.", 104),
        transform=ax.transAxes, fontsize=7.8, color="#555555", va="top")
    C.finish(fig, out_png)


def fig_change_forest(ct, out_png, gate="s7"):
    """The CONTRASTS in the table: mean Delta with its 95% CI, three panels."""
    order = [m for m in C.MODEL_ORDER if m in set(ct.model)] + ["POOLED"]
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order)}
    # sharex too: the three contrasts must be comparable in magnitude, which is
    # most of the point of putting them side by side.
    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(13.0, 4.6),
                             sharey=True, sharex=True)

    lo_all, hi_all = [], []
    for ax, (a, b) in zip(axes, CONTRASTS):
        for _, r in ct.iterrows():
            if r.model not in ypos:
                continue
            pooled = r.model == "POOLED"
            col = INK if pooled else C.style(r.model)["color"]
            mk = "D" if pooled else C.style(r.model)["marker"]
            m = r[f"d_N{a}_to_N{b}"]; lo = r[f"d_N{a}_to_N{b}_ci_lo"]; hi = r[f"d_N{a}_to_N{b}_ci_hi"]
            y = ypos[r.model]
            ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], color=col, marker=mk,
                        ms=10 if pooled else 7, lw=0, elinewidth=2.0 if pooled else 1.3,
                        capsize=4, mec="white", mew=0.8, zorder=4 if pooled else 3)
            lo_all.append(lo); hi_all.append(hi)
        ax.axvline(0, color="#9a9a9a", lw=1.2, ls=":", zorder=1)
        ax.axhline(0.5, color="#d0d0d0", lw=1.0, zorder=1)      # separate POOLED
        ax.set_title(f"N={a} → N={b}", fontweight="bold", loc="left", fontsize=10.5)
        ax.set_xlabel("Δ mean trough (µV)")
    n_by = dict(zip(ct.model, ct.n_stimuli))
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([f"{m}  ({int(n_by[m])})" for m in ypos], fontsize=9)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    pad = 0.1 * (max(hi_all) - min(lo_all))
    axes[0].set_xlim(min(lo_all) - pad, max(hi_all) + pad)

    fig.suptitle(f"LIT — mean within-stimulus change in trough across N, with 95% CIs "
                 f"({C.gate_label(gate, long=False)})",
                 fontweight="bold", x=0.006, ha="left", y=1.03)
    fig.text(0.006, -0.03, C.wrap(
        "Every value in lit_n_change_table.csv, drawn. Points left of the dotted zero line mean a "
        "DEEPER trough at the higher N — the MMN-like direction. A CI crossing zero is a null for "
        "that model. All three panels share one x-axis, so the contrasts are directly comparable. "
        "Numbers beside each model are its qualifying stimulus count. POOLED (diamond, below the "
        "rule) is over all model×stimulus pairs, so it is not an average of the six rows above it. "
        "The share of stimuli that deepened is in lit_n_change_table.csv and drawn per stimulus in "
        "n_effect_change_dist.png."),
        fontsize=7.8, color="#555555", ha="left", va="top")
    C.finish(fig, out_png)



CONTRASTS = [(3, 5), (3, 7), (5, 7)]


# ------------------------------------------------------------------------------------------
# 3b. Visualising the N-change table
# ------------------------------------------------------------------------------------------
def fig_change_forest(ct, out_png, gate="s7"):
    """Forest plot of the three N contrasts -- the canonical view of a table of effects + CIs.

    One row per model with POOLED separated at the top, one panel per contrast, a dotted zero
    line, and the CI drawn as the bar. Filled markers are p < 0.05. Reading left of zero as
    "deeper at the higher N" makes the MMN-like direction the left side throughout.
    """
    order = ["POOLED"] + [m for m in C.MODEL_ORDER if m in set(ct["model"])]
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order)}     # POOLED at the top

    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(13.2, 4.6), sharey=True)
    for ax, (a, b) in zip(axes, CONTRASTS):
        for m in order:
            r = ct[ct["model"] == m].iloc[0]
            d = r[f"d_N{a}_to_N{b}"]
            lo, hi = r[f"d_N{a}_to_N{b}_ci_lo"], r[f"d_N{a}_to_N{b}_ci_hi"]
            pv = r[f"d_N{a}_to_N{b}_p"]
            sig = (pv == pv) and pv < 0.05
            col = INK if m == "POOLED" else C.style(m)["color"]
            mk = "D" if m == "POOLED" else C.style(m)["marker"]
            ax.errorbar([d], [ypos[m]], xerr=[[d - lo], [hi - d]], color=col, marker=mk,
                        ms=9 if m == "POOLED" else 7.5, lw=0,
                        elinewidth=2.0 if m == "POOLED" else 1.4, capsize=4,
                        mfc=col if sig else "white", mew=1.6, zorder=3)
        ax.axvline(0, color="#9a9a9a", lw=1.2, ls=":", zorder=1)
        ax.axhline(ypos["POOLED"] - 0.5, color="#d8d8d8", lw=1.0, zorder=1)
        ax.set_yticks([ypos[m] for m in order])
        ax.set_yticklabels(order, fontsize=9)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlabel("Δ trough (µV)")
        ax.set_title(f"N={a} → N={b}", fontweight="bold", loc="left")
    axes[0].set_ylabel("")
    xmin = min(ax.get_xlim()[0] for ax in axes); xmax = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(xmin, xmax)                 # one scale, so the three contrasts are comparable
        ax.annotate("← deeper at higher N", (0, 1.005), xycoords=("data", "axes fraction"),
                    ha="right", va="bottom", fontsize=7.5, color="#6b6b6b")

    fig.suptitle(f"LIT — mean within-stimulus change in MMN trough across N "
                 f"({C.gate_label(gate, long=False)})",
                 fontweight="bold", x=0.006, ha="left", y=1.04)
    fig.text(0.006, -0.03, C.wrap(
        f"Point = mean change over the {int(ct[ct.model == 'POOLED'].n_stimuli.iloc[0])} "
        f"(model × stimulus) cells that pass at N=3 AND 5 AND 7; bar = 95% CI of that mean; "
        f"FILLED marker = p < 0.05 (Wilcoxon signed-rank). Once a cell qualifies, its trough at "
        f"each N is the mean over ALL FIVE variations, including ones below the µV floor. Shared "
        f"x-axis across the three panels. A CI crossing zero means the direction is not resolved "
        f"for that model."),
        fontsize=7.8, color="#555555", ha="left", va="top")
    C.finish(fig, out_png)


def fig_change_trajectory(ct, out_png, gate="s7"):
    """Trough across N per model, re-baselined to N=3 so the SHAPES are comparable.

    The raw levels span ~1.1 to ~2.4 uV between models, which on one axis compresses every
    trajectory into a flat line. Subtracting each model's own N=3 value puts them all at zero
    there and shows what the table's contrast columns actually say.
    """
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    xs = list(range(len(C.N_LEVELS)))
    for m in ["POOLED"] + [x for x in C.MODEL_ORDER if x in set(ct["model"])]:
        r = ct[ct["model"] == m].iloc[0]
        ys = [r["mean_N3"], r["mean_N5"], r["mean_N7"]]
        ys = [y - ys[0] for y in ys]                       # re-baseline to N=3
        pooled = m == "POOLED"
        st = dict(color=INK, marker="D") if pooled else C.style(m)
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls="-",
                lw=3.0 if pooled else 1.7, ms=9 if pooled else 7,
                mec="white", mew=0.9, zorder=5 if pooled else 3,
                label="POOLED" if pooled else m)
        ax.annotate(f" {m}", (xs[-1], ys[-1]), fontsize=8,
                    color=st["color"], va="center", ha="left",
                    fontweight="bold" if pooled else "normal")

    ax.axhline(0, color="#9a9a9a", lw=1.2, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n({C.N_TO_P_DEVIANT[n]:.1%})" for n in C.N_LEVELS])
    ax.set_xlim(-0.15, len(C.N_LEVELS) - 1 + 0.95)
    ax.set_xlabel("N standards between deviants\n(oddball probability)")
    ax.set_ylabel("Δ trough from N=3 (µV)\n↓ deeper")
    ax.set_title(f"LIT — trough trajectory across N, re-baselined to N=3 "
                 f"({C.gate_label(gate, long=False)})", fontweight="bold", loc="left")
    fig.text(0.0, -0.16, C.wrap(
        "Each model's own N=3 mean is subtracted, so every line starts at zero and the panel shows "
        "SHAPE rather than level — the raw means span ~1.1 to ~2.4 µV between models, which on a "
        "shared axis flattens every trajectory. Downward = deeper at higher N. This is the same "
        "data as the forest plot's contrast columns, drawn as a path.", 96),
        transform=ax.transAxes, fontsize=7.8, color="#555555", va="top")
    C.finish(fig, out_png)


def print_change_table(ct, gate):
    print("\n" + "=" * 104)
    print(f"LIT — mean change in MMN trough across N ({C.gate_label(gate, long=False)}; µV, "
          f"negative = deeper at the higher N)")
    print("Stimuli passing at N=3 AND 5 AND 7; trough at each N = mean over ALL 5 variations.")
    print("=" * 104)
    hdr = (f"  {'model':<17}{'stim':>5}{'N=3':>8}{'N=5':>8}{'N=7':>8}"
           f"{'3→5':>9}{'3→7':>9}{'5→7':>9}{'p(3→7)':>10}{'%deeper 3→7':>13}")
    print(hdr)
    for _, r in ct.iterrows():
        p = r["d_N3_to_N7_p"]
        print(f"  {r.model:<17}{int(r.n_stimuli):>5}{r.mean_N3:>8.3f}{r.mean_N5:>8.3f}"
              f"{r.mean_N7:>8.3f}{r.d_N3_to_N5:>+9.3f}{r.d_N3_to_N7:>+9.3f}{r.d_N5_to_N7:>+9.3f}"
              + (f"{p:>10.3g}" if p == p else f"{'--':>10}")
              + f"{r.d_N3_to_N7_pct_deeper:>12.0f}%")
    r = ct[ct.model == "POOLED"].iloc[0]
    print(f"\n  POOLED 95% CIs:  3→5 [{r.d_N3_to_N5_ci_lo:+.3f}, {r.d_N3_to_N5_ci_hi:+.3f}]"
          f"   3→7 [{r.d_N3_to_N7_ci_lo:+.3f}, {r.d_N3_to_N7_ci_hi:+.3f}]"
          f"   5→7 [{r.d_N5_to_N7_ci_lo:+.3f}, {r.d_N5_to_N7_ci_hi:+.3f}]")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="s7", choices=("s7", "s2"))
    p.add_argument("--out_dir", default=str(C.PLOTS_DIR))
    p.add_argument("--csv_dir", default=str(C.ANALYSIS_DIR))
    args = p.parse_args()
    gate, sfx = args.gate, ("" if args.gate == "s7" else f"__{args.gate}")

    tidy = C.load_tidy()
    per_trial = C.load_per_trial()
    root = Path(args.out_dir) if args.out_dir != str(C.PLOTS_DIR) else None
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("Figures:")
    s1 = fig_trough(tidy, C.fig_path(gate, "lit", f"lit_deviance_2st_trough{sfx}", root), gate)
    s2 = fig_pass_rate(per_trial, tidy,
                       C.fig_path(gate, "lit", f"lit_deviance_2st_pass_rate{sfx}", root), gate)
    ct = n_change_table(per_trial, gate)
    fig_change_forest(ct, C.fig_path(gate, "lit", f"lit_n_change_forest{sfx}", root), gate)
    fig_change_trajectory(ct, C.fig_path(gate, "lit", f"lit_n_change_trajectory{sfx}", root), gate)
    fig_change_trajectory(ct, C.fig_path(gate, "lit", f"lit_n_change_trajectory{sfx}", root), gate)
    fig_change_forest(ct, C.fig_path(gate, "lit", f"lit_n_change_forest{sfx}", root), gate)

    for name, d in ((f"lit_deviance_2st_trough{sfx}", s1),
                    (f"lit_deviance_2st_pass_rate{sfx}", s2),
                    (f"lit_n_change_table{sfx}", ct)):
        d.to_csv(csv_dir / f"{name}.csv", index=False, float_format="%.6g")
        print(f"  wrote {csv_dir / f'{name}.csv'}")

    print("\n" + "=" * 104)
    print(f"LIT — mean trough per 2-semitone bin ({C.gate_label(gate, long=False)})")
    print("=" * 104)
    print(f"  {'bin':<9}{'conds':>7}{'w/pass':>8}{'rows':>7}{'mean µV':>10}{'95% CI':>22}")
    for _, r in s1.iterrows():
        ci = (f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]" if r.n_rows >= 2 else "--")
        mu = f"{r.mean_uv:+.3f}" if r.n_rows else "--"
        print(f"  {r['bin']:<9}{int(r.n_conditions):>7}{int(r.n_conditions_passing):>8}"
              f"{int(r.n_rows):>7}{mu:>10}{ci:>22}")

    print("\n" + "=" * 104)
    print(f"LIT — mean per-stimulus pass rate per 2-semitone bin "
          f"({C.gate_label(gate, long=False)}, x/15)")
    print("=" * 104)
    print(f"  {'bin':<9}{'conds':>7}{'model×cond':>12}{'mean rate':>12}{'95% CI':>22}")
    for _, r in s2.iterrows():
        ci = (f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if r.n_model_conditions >= 2 else "--")
        mu = f"{r.mean_pass_rate:.3f}" if r.n_model_conditions else "--"
        print(f"  {r['bin']:<9}{int(r.n_conditions):>7}{int(r.n_model_conditions):>12}"
              f"{mu:>12}{ci:>22}")

    print_change_table(ct, gate)


if __name__ == "__main__":
    main()
