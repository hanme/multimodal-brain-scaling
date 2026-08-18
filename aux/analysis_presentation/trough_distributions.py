#!/usr/bin/env python
"""CP3 -- per-model trough distributions: literature vs model-selected, and the tail count.

THE CLAIM THIS FIGURE MAKES, stated so it cannot drift: it is NOT that the two distributions
differ. They overlap heavily and are expected to. The claim is a COUNT IN THE TAIL -- that for a
given model there exist some model-selected frequency pairs whose predicted MMN trough is deeper
than ANY literature stimulus that model was given. "Some" is the word; it does not have to be many.
The group comparison (Welch t, Cohen's d, Mann-Whitney) is reported because it was asked for, as a
secondary, and is expected to be unimpressive. A small d does not weaken the tail count -- the two
are answering different questions, and only the tail one is the motivation for the project.

Two populations, one row per (model, direction-instance):
  LIT       24 literature Frequency methods x {regular, counter} = 48 conditions per model
  NOVEL-P2  127 selected tone pairs x {regular, counter} = 254 conditions per model

ALL 254 phase-2 instances are included, deliberately. The sub-threshold reversals carried along
only for counterbalancing are NOT dropped: excluding them would narrow the distribution on the
selected-direction side and make the tail look like a property of the filter rather than of the
stimuli. Decision recorded because it is not the default a reader would assume.

Site FCz, mapping mTRF, six models (whisper-large excluded -- its predicted uV run ~20-35x the
others and it was never run in the novel search, so LIT and NOVEL-P2 would not be comparable).

THE GATE QUESTION, ANSWERED FROM THE DATA (see verify(), which asserts it every run):
`trough_uv` is finite for EVERY row in both CSVs -- 0 NaN in 288 LIT and 1524 NOVEL-P2 rows. It is
the uV difference-wave value sampled at the argmin latency of the z-scored trace inside 100-240 ms
(scripts/analyze_mmn_s7_roi.py), and that argmin exists whether or not the trace passes any
criterion. So a trough is DEFINED for all traces, and the widest population is every row.

Three populations are therefore reported, filed by gate the way the realness-checks deliverable is:

  all  PRIMARY. Every trace. No shape criterion, no uV floor. This is the population the tail
       comparison needs: the 0.75 uV floor censors exactly the deep tail being counted on one side
       and the shallow end on the other, so gating first and counting the tail after would be
       measuring the gate.
  s7   COMPANION, clearly labelled. S2 shape AND trough <= -0.75 uV. Both sets lose their shallow
       mass, so the medians move deeper in both -- the tail count is barely affected because the
       floor sits nowhere near the tail, which is the useful thing to be able to say.
  s2   Shape only, no uV floor. The middle rung, and what the sibling deliverable calls "ungated".
       Reported because `all` includes traces whose argmin is not an MMN latency at all (14% of LIT
       and 21% of NOVEL-P2 rows have a POSITIVE trough, which is a peak, not a trough); s2 is the
       population where every trough_uv is at least measured at a defensible latency.

READ-ONLY over the two committed scored CSVs plus the two metadata tables.

  python aux/analysis_presentation/trough_distributions.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis_MMN_realness_checks"))

# The load path, the model list, the site/mapping/floor constants and the palette all come from the
# sibling deliverable rather than being retyped -- that is what keeps this figure describing the
# same experiment as the dose-response memo.
from mmn_dose_response_common import (  # noqa: E402
    LIT_CSV, P2_CSV, P2_GRID, LIT_META, MODEL_ORDER, N_CONDITIONS,
    load_fcz, style, wrap,
)
from novel_search_common import DIP_UV_THRESHOLD  # noqa: E402
from analyze_mmn_screen_24freq import load_semitones  # noqa: E402

PLOTS_DIR = HERE / "plots"
SVG_DIR = HERE / "svgs"
P2_RANKING = LIT_CSV.parents[1] / "results_novel_search/phase2_final_ranking.csv"

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
    "svg.hashsalt": "cp3-trough-distributions",
})

# Colour grammar, inherited and extended by exactly one rule. Model identity is colour, as
# everywhere else in this project -- so NOVEL-P2 is drawn in the panel's own model colour. LIT is
# the fixed neutral it is compared against. Within a panel the model is constant, so colour is free
# to carry the source; across panels the coloured series is still the model's own colour.
LIT_COLOR = "#7a7a7a"
LIT_FILL = "#b8b8b8"
SET_LABEL = {"lit": "LIT (literature)", "p2": "NOVEL-P2 (model-selected)"}
MUTED = "#666666"
INK = "#222222"

# The tail-count annotation sits top-left inside the axes. Without headroom it lands ON the
# densities, which are tallest near their mode -- close to the left edge for exactly the models
# whose tail count matters most. Reserving a fixed fraction above the peak is what keeps the
# number legible without moving the box around per panel and making the six read inconsistently.
HEADROOM = 1.32

GATES = {
    "all": ("PRIMARY", "all traces — no shape criterion, no µV floor"),
    "s7":  ("COMPANION", f"S7 — shape + trough ≤ −{DIP_UV_THRESHOLD} µV floor"),
    "s2":  ("COMPANION", "S2 — shape only, no µV floor"),
}


# ------------------------------------------------------------------------------------------
# Load and verify
# ------------------------------------------------------------------------------------------
def load():
    """One tidy frame: dataset, model, method, method_id, direction, s2, s7, trough_uv + stimulus."""
    lit = load_fcz(LIT_CSV, "lit")
    p2 = load_fcz(P2_CSV, "p2")

    lit_st = load_semitones(LIT_META)
    lit["semitones"] = lit["method_id"].map(lit_st)
    lit["f_low"] = np.nan
    lit["f_high"] = np.nan
    lit["pct_deviance"] = np.nan

    grid = pd.read_csv(P2_GRID)[["method_id", "semitones", "f_low", "f_high", "pct_deviance"]]
    p2 = p2.merge(grid, on="method_id", how="left")
    if p2["semitones"].isna().any():
        raise SystemExit(f"{P2_GRID} is missing method_ids present in {P2_CSV}")

    # n_agree comes from the search's own committed ranking, per direction-instance -- it is a
    # property of the stimulus across all six models, not of the (model, stimulus) row.
    rank = pd.read_csv(P2_RANKING)[["method", "direction", "n_agree"]]
    p2 = p2.merge(rank, on=["method", "direction"], how="left")
    if p2["n_agree"].isna().any():
        raise SystemExit(f"{P2_RANKING} does not cover every phase-2 direction-instance")
    lit["n_agree"] = np.nan

    return pd.concat([lit, p2], ignore_index=True)


def verify(tidy):
    """Every invariant the brief asks to be confirmed, asserted rather than eyeballed."""
    print("VALIDATION")
    assert list(MODEL_ORDER) and len(MODEL_ORDER) == 6, MODEL_ORDER
    assert "whisper-large" not in set(tidy["model"]), "whisper-large leaked in"
    print(f"  six models present, whisper-large absent: {MODEL_ORDER}")

    for ds, n_cond in N_CONDITIONS.items():
        d = tidy[tidy["dataset"] == ds]
        per_model = d.groupby("model").size()
        assert set(per_model.index) == set(MODEL_ORDER), sorted(per_model.index)
        assert (per_model == n_cond).all(), per_model.to_dict()
        assert not d.duplicated(["model", "method_id", "direction"]).any(), \
            f"{ds}: duplicate (model, method_id, direction) rows"
        n_dir = d.groupby(["model", "method_id"])["direction"].nunique()
        assert (n_dir == 2).all(), f"{ds}: pairs without both directions"
        print(f"  {ds}: {n_cond} cells x 6 models = {len(d)} rows; no duplicates; "
              f"{d['method_id'].nunique()} pairs x 2 directions")

    n_nan = int(tidy["trough_uv"].isna().sum())
    assert n_nan == 0, f"{n_nan} NaN trough_uv -- the 'all' population is not what it claims"
    print(f"  trough_uv finite for all {len(tidy)} rows -> a trough IS defined for every trace")

    assert not (tidy["s7"] & ~tidy["s2"]).any(), "S7 rows that are not S2"
    for m in MODEL_ORDER:
        for ds in N_CONDITIONS:
            v = tidy[(tidy["model"] == m) & (tidy["dataset"] == ds)]["trough_uv"]
            assert v.min() < 0, f"{m}/{ds}: deepest trough {v.min()} is not negative"
    print("  deepest trough is negative for every (model, set) -- sign convention holds")

    pos = tidy.groupby("dataset")["trough_uv"].apply(lambda s: 100.0 * (s >= 0).mean())
    print(f"  positive trough_uv (a peak, not a trough): "
          f"lit {pos['lit']:.1f}%, p2 {pos['p2']:.1f}% of rows in the 'all' population")
    print("  units: µV, predicted difference wave (deviant − standard) at FCz — the same signed "
          "µV column as the head-to-head and dose-response figures")
    print()


def gated(frame, gate):
    if gate == "all":
        return frame
    return frame[frame[gate]]


# ------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------
def descriptives(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, iqr_low=np.nan,
                    iqr_high=np.nan, deepest_uv=np.nan, shallowest_uv=np.nan)
    return dict(n=int(v.size), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if v.size > 1 else np.nan,
                median=float(np.median(v)),
                iqr_low=float(np.percentile(v, 25)), iqr_high=float(np.percentile(v, 75)),
                deepest_uv=float(v.min()), shallowest_uv=float(v.max()))


def tail_counts(lit_v, p2_v):
    """How much of NOVEL-P2 lies beyond the literature's deep end, at three reference lines.

    STRICTLY deeper (`<`), never `<=`. The three lines answer the same question at three levels of
    robustness: the literature MINIMUM is the headline but rests on one condition, so a single deep
    literature outlier could drive the strict count to zero while the bulk of the literature sits
    far shallower. The 90th and 95th percentiles OF DEPTH -- i.e. the 10th and 5th percentiles of
    the signed uV, since deeper is more negative -- say the same thing without that dependence.
    """
    lit_v = np.asarray(lit_v, float)
    p2_v = np.asarray(p2_v, float)
    out = {}
    refs = {"lit_min": float(lit_v.min()) if lit_v.size else np.nan,
            "lit_p90": float(np.percentile(lit_v, 10)) if lit_v.size else np.nan,
            "lit_p95": float(np.percentile(lit_v, 5)) if lit_v.size else np.nan}
    for key, ref in refs.items():
        n = int((p2_v < ref).sum()) if np.isfinite(ref) else 0
        out[f"ref_{key}_uv"] = ref
        out[f"n_deeper_than_{key}"] = n
        out[f"pct_deeper_than_{key}"] = 100.0 * n / p2_v.size if p2_v.size else np.nan
    return out


def group_tests(lit_v, p2_v):
    """Welch t + Cohen's d + 95% CI on the mean difference, and Mann-Whitney U + rank-biserial r.

    Sign convention throughout: the contrast is NOVEL-P2 minus LIT, so a NEGATIVE difference and a
    NEGATIVE d mean NOVEL-P2 troughs are deeper. The two sets are INDEPENDENT samples -- different
    stimuli entirely, no pairing -- which is why Welch and Mann-Whitney are the right forms here
    (unlike the paired N-effect contrast in the sibling deliverable).

    Cohen's d uses the CLASSIC POOLED SD, sqrt(((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2)), stated because
    with n=48 against n=254 and unequal variances the pooled-SD choice is not neutral: Glass's
    delta (LIT's SD alone) and Hedges' g would each give a different number on the same data.

    A rank-based companion is reported alongside because these distributions are skewed -- the
    project memo documents 0 of 31 per-bin trough distributions consistent with normality -- so the
    t-test's own assumption is not met and its p is the optimistic one of the two.
    """
    lit_v = np.asarray(lit_v, float)
    p2_v = np.asarray(p2_v, float)
    n1, n2 = lit_v.size, p2_v.size
    rows = []
    if n1 < 3 or n2 < 3:
        return rows

    m1, m2 = lit_v.mean(), p2_v.mean()
    s1, s2_ = lit_v.var(ddof=1), p2_v.var(ddof=1)
    diff = m2 - m1                                   # p2 - lit; negative = p2 deeper
    se = np.sqrt(s1 / n1 + s2_ / n2)
    df = (s1 / n1 + s2_ / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2_ / n2) ** 2 / (n2 - 1))
    t, p = stats.ttest_ind(p2_v, lit_v, equal_var=False)
    crit = stats.t.ppf(0.975, df)
    rows.append(dict(test="welch_t", statistic=float(t), df=float(df), p=float(p),
                     effect_size_name="mean_diff_uv_p2_minus_lit", effect_size=float(diff),
                     ci_low=float(diff - crit * se), ci_high=float(diff + crit * se),
                     n_lit=n1, n_p2=n2))

    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2_) / (n1 + n2 - 2))
    d = diff / sp if sp > 0 else np.nan
    # CI on d from the SE of a two-sample d (Hedges & Olkin large-sample form).
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))) if np.isfinite(d) else np.nan
    rows.append(dict(test="cohens_d_pooled_sd", statistic=float(d), df=float(n1 + n2 - 2), p=np.nan,
                     effect_size_name="cohens_d_p2_minus_lit_pooled_sd", effect_size=float(d),
                     ci_low=float(d - 1.96 * se_d), ci_high=float(d + 1.96 * se_d),
                     n_lit=n1, n_p2=n2))

    u, pu = stats.mannwhitneyu(p2_v, lit_v, alternative="two-sided")
    r_rb = 2.0 * u / (n1 * n2) - 1.0                 # negative = p2 stochastically deeper
    rows.append(dict(test="mannwhitney_u", statistic=float(u), df=np.nan, p=float(pu),
                     effect_size_name="rank_biserial_r_p2_minus_lit", effect_size=float(r_rb),
                     ci_low=np.nan, ci_high=np.nan, n_lit=n1, n_p2=n2))
    return rows


# ------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------
def save_both(fig, gate, stem):
    """plots/<gate>/<stem>.png and the mirrored svgs/<gate>/<stem>.svg from ONE render.

    Deliberately does NOT call tight_layout: every caller has already run it with a `rect` that
    reserves the caption strip, and a second bare tight_layout() here discards that rect and drops
    the caption back on top of the tick labels.
    """
    png = PLOTS_DIR / gate / f"{stem}.png"
    svg = SVG_DIR / gate / f"{stem}.svg"
    for p in (png, svg):
        p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"  wrote plots/{gate}/{png.name} + svgs/{gate}/{svg.name}")


def _kde(v, xs):
    v = np.asarray(v, float)
    if v.size < 3 or np.unique(v).size < 2:
        return None
    return stats.gaussian_kde(v)(xs)


def draw_panel(ax, model, lit_v, p2_v, form, show_legend=False, compact=False):
    """One model's two overlaid distributions, as density. Returns the literature minimum drawn.

    DENSITY, never counts: with 48 against 254 a count axis would make NOVEL-P2 look five times
    more prevalent at every depth, which is an artefact of how many stimuli each phase happened to
    contain and says nothing about the stimuli.
    """
    col = style(model)["color"]
    lo = min(lit_v.min(), p2_v.min())
    hi = max(lit_v.max(), p2_v.max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    lo, hi = lo - pad, hi + pad
    lit_min = float(lit_v.min())

    # The tail band goes down FIRST, at zorder 0, so it reads as ground rather than as a mark
    # sitting on top of the densities. Drawn from the explicit axis floor, not from get_xlim(),
    # which at this point still holds matplotlib's autoscale guess rather than the range below.
    ax.axvspan(lo, lit_min, color=col, alpha=0.09, lw=0, zorder=0)

    if form == "hist":
        bins = np.linspace(lo, hi, 31)               # COMMON bins -- both sets, same edges
        ax.hist(lit_v, bins=bins, density=True, color=LIT_FILL, alpha=0.85,
                edgecolor=LIT_COLOR, lw=0.6, label=SET_LABEL["lit"], zorder=2)
        ax.hist(p2_v, bins=bins, density=True, histtype="step", color=col, lw=1.9,
                label=SET_LABEL["p2"], zorder=3)
        ax.set_xlim(lo, hi)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, HEADROOM * ymax)
    else:
        xs = np.linspace(lo, hi, 512)
        for v, c, lab, z in ((lit_v, LIT_COLOR, SET_LABEL["lit"], 2),
                             (p2_v, col, SET_LABEL["p2"], 3)):
            d = _kde(v, xs)
            if d is None:
                continue
            ax.fill_between(xs, d, color=c, alpha=0.20, lw=0, zorder=z)
            ax.plot(xs, d, color=c, lw=1.9, zorder=z + 0.1, label=lab)
        ax.set_xlim(lo, hi)
        ax.autoscale(axis="y")
        ymax = ax.get_ylim()[1]
        # Rug of the ACTUAL observations. A KDE smooths mass past the extremes, and the extremes
        # are the entire claim -- the rug is what keeps the deepest real point visible and honest.
        ax.plot(lit_v, np.full(lit_v.size, -0.045 * ymax), "|", color=LIT_COLOR,
                ms=6, mew=1.0, alpha=0.9, zorder=4, clip_on=False)
        ax.plot(p2_v, np.full(p2_v.size, -0.095 * ymax), "|", color=col,
                ms=6, mew=0.8, alpha=0.55, zorder=4, clip_on=False)
        ax.set_ylim(-0.125 * ymax, HEADROOM * ymax)

    ax.axvline(lit_min, color=INK, ls="--", lw=1.4, zorder=5)
    ax.axvline(float(np.median(lit_v)), color=LIT_COLOR, ls=":", lw=1.6, zorder=5)
    ax.axvline(float(np.median(p2_v)), color=col, ls=":", lw=1.6, zorder=5)

    n_beat = int((p2_v < lit_min).sum())
    pct = 100.0 * n_beat / p2_v.size
    lit_p90 = float(np.percentile(lit_v, 10))
    n_p90 = int((p2_v < lit_p90).sum())
    ax.set_title(f"{model}", fontsize=10.5 if compact else 11.5, color=INK, pad=6)
    ax.text(0.025, 0.965, f"n: LIT {lit_v.size}  ·  NOVEL-P2 {p2_v.size}\n"
                          f"{n_beat} deeper than LIT's deepest "
                          f"({lit_min:.2f} µV)  ·  {pct:.1f}%\n"
                          f"{n_p90} deeper than LIT's 90th pct ({lit_p90:.2f} µV)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.4 if compact else 8.4,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dddddd", lw=0.7, alpha=0.94))
    ax.set_ylabel("density", fontsize=8.5)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8.0, frameon=True, framealpha=0.94)
    return lit_min


CAPTION_HEAD = (
    "MMN trough depth at FCz (mTRF, µV, deviant − standard sampled at the trough latency). "
    "More negative = deeper = further LEFT = a stronger model MMN. "
    "Dashed black rule = that model's DEEPEST literature stimulus; the shaded band left of it is "
    "the region no literature stimulus reaches. Dotted rules = each set's median. "
    "Areas are DENSITIES, not counts, because n = 48 against n = 254.")

CAPTION_FOOT = (
    "Each model's µV scale is its own — mTRF predictions are amplitude-shrunk and the shrinkage "
    "differs by model and layer — so compare each panel against its OWN literature rule, never "
    "one model's depth against another's. The claim here is the count in the left tail, not a "
    "difference between the two distributions: they overlap heavily by design.")


def figure_per_model(tidy, gate, form):
    """One figure per model, in both distributional forms."""
    for model in MODEL_ORDER:
        d = gated(tidy[tidy["model"] == model], gate)
        lit_v = d[d["dataset"] == "lit"]["trough_uv"].to_numpy(float)
        p2_v = d[d["dataset"] == "p2"]["trough_uv"].to_numpy(float)
        if lit_v.size < 2 or p2_v.size < 2:
            print(f"  skipped {model} at gate={gate}: n_lit={lit_v.size} n_p2={p2_v.size}")
            continue
        role, desc = GATES[gate]
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        draw_panel(ax, model, lit_v, p2_v, form, show_legend=True)
        ax.set_xlabel("trough_uv at FCz (µV)  —  deeper (more negative) to the left", labelpad=8)
        ax.set_title(f"{model} — literature vs model-selected trough depth\n"
                     f"{role} population: {desc}   (n_LIT={lit_v.size}, n_P2={p2_v.size})",
                     fontsize=11, pad=10)
        # The caption is placed in FIGURE coordinates under a reserved strip, not floated over the
        # axes: with bbox_inches="tight" an overlapping caption silently sits on the tick labels.
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.text(0.5, 0.115, wrap(CAPTION_HEAD, 108), ha="center", va="top", fontsize=7.6,
                 color=MUTED, linespacing=1.55)
        save_both(fig, gate, f"trough_dist__{model}__{form}")


def figure_panel(tidy, gate, form):
    """The 2x3 slide panel -- all six models on one canvas."""
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))
    for ax, model in zip(axes.ravel(), MODEL_ORDER):
        d = gated(tidy[tidy["model"] == model], gate)
        lit_v = d[d["dataset"] == "lit"]["trough_uv"].to_numpy(float)
        p2_v = d[d["dataset"] == "p2"]["trough_uv"].to_numpy(float)
        if lit_v.size < 2 or p2_v.size < 2:
            ax.set_axis_off()
            continue
        draw_panel(ax, model, lit_v, p2_v, form, show_legend=False, compact=True)
    for ax in axes[1]:
        ax.set_xlabel("trough_uv at FCz (µV) — deeper to the left", fontsize=9, labelpad=6)

    handles = [mpl.patches.Patch(fc=LIT_FILL, ec=LIT_COLOR, label=SET_LABEL["lit"]),
               mpl.lines.Line2D([], [], color=INK, lw=1.9,
                                label=SET_LABEL["p2"] + " — drawn in each model's own colour"),
               mpl.lines.Line2D([], [], color=INK, ls="--", lw=1.4,
                                label="deepest literature stimulus for that model")]
    role, desc = GATES[gate]
    fig.suptitle("Results: comparing selected vs. literature responses — MMN trough depth at FCz\n"
                 f"{role} population: {desc}", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.20, 1, 0.965))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.185),
               ncol=3, fontsize=9.0, frameon=False)
    fig.text(0.5, 0.135, wrap(CAPTION_HEAD + " " + CAPTION_FOOT, 172), ha="center", va="top",
             fontsize=8.0, color=MUTED, linespacing=1.6)
    save_both(fig, gate, f"trough_dist__panel6__{form}")


# ------------------------------------------------------------------------------------------
# Tables
# ------------------------------------------------------------------------------------------
def build_tables(tidy):
    desc_rows, stat_rows, winner_rows, summary_rows = [], [], [], []

    for gate in GATES:
        for model in MODEL_ORDER:
            d = gated(tidy[tidy["model"] == model], gate)
            lit = d[d["dataset"] == "lit"]
            p2 = d[d["dataset"] == "p2"]
            lit_v = lit["trough_uv"].to_numpy(float)
            p2_v = p2["trough_uv"].to_numpy(float)
            if lit_v.size == 0 or p2_v.size == 0:
                continue
            tails = tail_counts(lit_v, p2_v)

            for set_key, v in (("LIT", lit_v), ("NOVEL-P2", p2_v)):
                row = dict(gate=gate, model=model, set=set_key, **descriptives(v))
                if set_key == "NOVEL-P2":
                    row.update(
                        n_deeper_than_lit_min=tails["n_deeper_than_lit_min"],
                        pct_deeper_than_lit_min=tails["pct_deeper_than_lit_min"],
                        n_deeper_than_lit_p90=tails["n_deeper_than_lit_p90"],
                        pct_deeper_than_lit_p90=tails["pct_deeper_than_lit_p90"],
                        n_deeper_than_lit_p95=tails["n_deeper_than_lit_p95"],
                        pct_deeper_than_lit_p95=tails["pct_deeper_than_lit_p95"])
                desc_rows.append(row)

            for r in group_tests(lit_v, p2_v):
                stat_rows.append(dict(model=model, gate=gate, **r))

            ref = tails["ref_lit_min_uv"]
            win = p2[p2["trough_uv"] < ref].sort_values("trough_uv")
            for _, w in win.iterrows():
                winner_rows.append(dict(
                    gate=gate, model=model, method=w["method"], method_id=int(w["method_id"]),
                    direction=w["direction"], f_low=w["f_low"], f_high=w["f_high"],
                    semitones=w["semitones"], pct_deviance=w["pct_deviance"],
                    trough_uv=w["trough_uv"], n_agree=int(w["n_agree"]),
                    lit_deepest_uv=ref, margin_uv=w["trough_uv"] - ref))

            summary_rows.append(dict(
                gate=gate, model=model, n_lit=lit_v.size, n_p2=p2_v.size,
                lit_median_uv=float(np.median(lit_v)), p2_median_uv=float(np.median(p2_v)),
                lit_deepest_uv=tails["ref_lit_min_uv"], p2_deepest_uv=float(p2_v.min()),
                n_deeper_than_lit_min=tails["n_deeper_than_lit_min"],
                pct_deeper_than_lit_min=tails["pct_deeper_than_lit_min"],
                n_deeper_than_lit_p90=tails["n_deeper_than_lit_p90"],
                n_deeper_than_lit_p95=tails["n_deeper_than_lit_p95"]))

    cols = ["gate", "model", "set", "n", "mean", "sd", "median", "iqr_low", "iqr_high",
            "deepest_uv", "shallowest_uv", "n_deeper_than_lit_min", "pct_deeper_than_lit_min",
            "n_deeper_than_lit_p90", "pct_deeper_than_lit_p90",
            "n_deeper_than_lit_p95", "pct_deeper_than_lit_p95"]
    desc = pd.DataFrame(desc_rows).reindex(columns=cols)
    stat = pd.DataFrame(stat_rows).reindex(columns=[
        "model", "test", "statistic", "df", "p", "effect_size_name", "effect_size",
        "ci_low", "ci_high", "gate", "n_lit", "n_p2"])
    win = pd.DataFrame(winner_rows)
    summ = pd.DataFrame(summary_rows)
    return desc, stat, win, summ


def one_liners(summ, gate="all"):
    """One plain-language sentence per model, phrased as the TAIL claim, not a group difference."""
    lines = []
    for _, r in summ[summ["gate"] == gate].iterrows():
        n, pct = int(r["n_deeper_than_lit_min"]), r["pct_deeper_than_lit_min"]
        if n == 0:
            s = (f"{r['model']}: no model-selected stimulus goes deeper than this model's deepest "
                 f"literature stimulus ({r['lit_deepest_uv']:.2f} µV), though "
                 f"{int(r['n_deeper_than_lit_p90'])} beat its literature 90th-percentile depth.")
        else:
            s = (f"{r['model']}: {n} of {int(r['n_p2'])} model-selected stimuli ({pct:.1f}%) drive "
                 f"a deeper trough than ANY of the 48 literature conditions — the deepest reaches "
                 f"{r['p2_deepest_uv']:.2f} µV against the literature's {r['lit_deepest_uv']:.2f} µV.")
        lines.append(s)
    return lines


def main():
    tidy = load()
    verify(tidy)

    desc, stat, win, summ = build_tables(tidy)
    desc.round(4).to_csv(HERE / "trough_distributions_by_model.csv", index=False)
    stat.round(6).to_csv(HERE / "trough_distribution_stats.csv", index=False)
    win.round(4).to_csv(HERE / "tail_winners.csv", index=False)
    summ.round(4).to_csv(HERE / "trough_distribution_summary.csv", index=False)
    print(f"  wrote trough_distributions_by_model.csv ({len(desc)} rows)")
    print(f"  wrote trough_distribution_stats.csv ({len(stat)} rows)")
    print(f"  wrote tail_winners.csv ({len(win)} rows)")
    print(f"  wrote trough_distribution_summary.csv ({len(summ)} rows)\n")

    for gate in GATES:
        print(f"FIGURES [{gate}] — {GATES[gate][0]}: {GATES[gate][1]}")
        for form in ("kde", "hist"):
            figure_per_model(tidy, gate, form)
            figure_panel(tidy, gate, form)
        print()

    print("CROSS-MODEL SUMMARY (primary population: all traces, ungated)")
    prim = summ[summ["gate"] == "all"]
    print(prim.drop(columns=["gate"]).to_string(index=False))
    print()
    print("ONE-SENTENCE SUMMARY PER MODEL (primary population)")
    for s in one_liners(summ, "all"):
        print("  " + s)


if __name__ == "__main__":
    main()
