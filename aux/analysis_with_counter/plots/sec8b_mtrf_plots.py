#!/usr/bin/env python
"""Sections 8b/8c — mTRF amplitude-floor figures for the 24-method screen (5 whisper models).

Reads outputs/results_24freq_7models/mmn_s7_roi.csv (mTRF only). S7@X from the X-independent
`trough_uv` column:  S7@X present  <=>  s2 AND (trough_uv <= -X).  24 frequency methods ×
{regular, counter} = 48 conditions per model per site; /48 per model, /240 pooled over 5 models.

Each figure is a 2×2 grid of fronto-central sites (flat order = top-left, top-right, bottom-left,
bottom-right):
    frontal parcel  |  FCz electrode      (top)
    central parcel  |  Fz  electrode      (bottom)

Section 8b (5 figures):
  1. sec8b_x_vs_mmn_per_model.png            — count /48 vs floor X ∈ {S2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5}
  2. sec8b_x_vs_mmn_pooled.png               — count /240 vs the same floors, pooled over 5 models
  3. sec8b_x_vs_mmn_per_model_continuous.png — count /48 vs a CONTINUOUS floor X, per model
  4. sec8b_x_vs_mmn_pooled_continuous.png    — count /240 vs a CONTINUOUS floor X, pooled
  5. sec8b_trough_uv_distribution.png        — trough_uv (µV) per model over S2-passing conditions,
                                               dotted floors at −0.25/−0.5/−0.75/−1.0/−1.5/−2.5 µV
                                               (symlog x — whisper-large's predicted µV are ~40× the
                                               others, a scale artifact)
Section 8c (1 figure, single panel):
  6. sec8c_fz_vs_fcz_trough.png              — paired Fz vs FCz predicted trough (µV), matched conditions
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/Users/sophiesigfstead/Documents/multimodal-brain-scaling-2"
OUT = f"{REPO}/aux/analysis_with_counter/plots"
CSV = f"{REPO}/outputs/results_24freq_7models/mmn_s7_roi.csv"

# 5-model Okabe-Ito CVD-safe style (matches scripts/analyze_mmn_screen_24freq.py MODEL_STYLE);
# every series also carries a distinct marker shape, so identity is never colour-alone.
MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large"]
MODEL_STYLE = {
    "whisper-tiny":   dict(color="#56B4E9", marker="o"),   # sky blue    ○
    "whisper-base":   dict(color="#0072B2", marker="s"),   # blue        □
    "whisper-small":  dict(color="#009E73", marker="^"),   # bluish green △
    "whisper-medium": dict(color="#E69F00", marker="D"),   # orange      ◇
    "whisper-large":  dict(color="#D55E00", marker="v"),   # vermillion  ▽
}
MLABEL = {m: m.split("-")[1] for m in MODEL_ORDER}

# 2×2 reporting sites; flat order = top-left, top-right, bottom-left, bottom-right
SITES = [("parcel", "frontal", "frontal parcel"),
         ("electrode", "FCz", "FCz electrode"),
         ("parcel", "central", "central parcel"),
         ("electrode", "Fz", "Fz electrode")]
BOTTOM = {2, 3}      # flat panel indices in the bottom row  -> get the x-label
LEFT = {0, 2}        # flat panel indices in the left column -> get the y-label
LEGEND_PANEL = 1     # FCz (top-right) carries the per-model legend

# amplitude floors on the x-axis; S2 is the X->0 reference (first, evenly-spaced slot)
X_FLOORS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.5]
XLAB = ["S2\n(X→0)", "0.25", "0.5", "0.75", "1.0", "1.5", "2.5"]
XPOS = list(range(len(X_FLOORS) + 1))
I05 = X_FLOORS.index(0.5) + 1        # present_counts / x-slot index of the 0.5 headline
POOL_COLOR = "#0072B2"

FIGSIZE = (11.0, 8.8)                # 2×2 line grids
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

df = pd.read_csv(CSV)
d = df[(df.mapping == "mtrf") & (df.dip_uv_threshold == 0.25)].copy()   # one row per trace


def site_df(kind, roi):
    return d[(d.roi_kind == kind) & (d.roi == roi)]


def present_counts(sub):
    """[S2, S7@0.25, S7@0.5, S7@0.75, S7@1.0, S7@1.5, S7@2.5] present-count for a subframe."""
    out = [int(sub.s2.sum())]
    for X in X_FLOORS:
        out.append(int(((sub.s2) & (sub.trough_uv <= -X)).sum()))
    return out


CAP = ("S7@X = S2 ∧ (trough_uv ≤ −X µV); trough_uv = deviant−standard µV at the S2 trough latency "
       "(X-independent). mTRF only; 48 conditions/model = 24 frequency methods × {regular, counter}.")
handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                  ls="-", lw=1.9, ms=7.5, mec="white", label=MLABEL[m]) for m in MODEL_ORDER]

# ============ FIG 1: per-model, count /48 vs floor, 2×2 sites ============
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    ax.axvline(0, color="#b5b5b5", lw=1.1, ls=":", zorder=1)     # S2 (X→0) reference
    for m in MODEL_ORDER:
        st = MODEL_STYLE[m]
        ax.plot(XPOS, present_counts(sub[sub.model == m]), color=st["color"], marker=st["marker"],
                ms=7, lw=1.8, mec="white", mew=0.9, zorder=3)
    ax.set_xticks(XPOS); ax.set_xticklabels(XLAB)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylim(0, 49)
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 48 per model)")
axes.flat[LEGEND_PANEL].legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
                               title="whisper model", title_fontsize=9)
fig.suptitle("mTRF MMN count vs amplitude floor X, per whisper model — by fronto-central site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Count/48 per model; y at the leftmost slot is the S2 count.",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec8b_x_vs_mmn_per_model.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: pooled /240, single series, 2×2 sites ============
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    y = present_counts(sub)              # pooled over 5 models -> /240
    s2 = y[0]
    ax.axhline(s2, color="#9a9a9a", lw=1.4, ls="--", zorder=1)
    ax.annotate(f"S2 = {s2} / 240", xy=(XPOS[-1], s2), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.3, color="#666")
    ax.plot(XPOS, y, color=POOL_COLOR, marker="o", ms=8, lw=2.3, mec="white", mew=1.0, zorder=3)
    for xp, val in zip(XPOS, y):
        ax.annotate(str(val), (xp, val), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#222", fontweight="bold")
    ret = 100 * y[I05] / s2 if s2 else 0
    ax.annotate(f"S7@0.5 / S2 = {y[I05]}/{s2} = {ret:.0f}%  (headline)", xy=(0.03, 0.06),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8.4, color=POOL_COLOR)
    ax.set_xticks(XPOS); ax.set_xticklabels(XLAB)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylim(0, 252)
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 240, pooled over 5 models)")
fig.suptitle("mTRF MMN count vs amplitude floor X, pooled over the 5 whisper models — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Pooled count/240; dashed line = the S2 total (the X→0 reference).",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec8b_x_vs_mmn_pooled.png", bbox_inches="tight")
plt.close(fig)

# ============ continuous-X survival curves (companions to FIG 1 & 2) ============
XGRID = np.linspace(0.0, 2.6, 521)


def s7_curve(sub):
    """S7 present-count over the continuous floor XGRID (X-independent trough_uv)."""
    tt = sub.loc[sub.s2, "trough_uv"].values
    return np.array([int((tt <= -x).sum()) for x in XGRID])


# ---- FIG 3: per-model, continuous X, 2×2 sites ----
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    for xf in X_FLOORS:
        ax.axvline(xf, color="#ededed", lw=1.0, zorder=0)
    ax.axvline(0.5, color="#c9c9c9", lw=1.3, ls="--", zorder=0)   # 0.5 headline floor
    for m in MODEL_ORDER:
        st = MODEL_STYLE[m]
        ms = sub[sub.model == m]
        ax.plot(XGRID, s7_curve(ms), color=st["color"], lw=1.8, zorder=3)
        ax.plot(X_FLOORS, present_counts(ms)[1:], color=st["color"], marker=st["marker"],
                ms=6, ls="none", mec="white", mew=0.7, zorder=4)
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 49)
    ax.set_title(title, fontweight="bold", loc="left")
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV, continuous)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 48 per model)")
axes.flat[LEGEND_PANEL].legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
                               title="whisper model", title_fontsize=9)
fig.suptitle("mTRF MMN count vs continuous amplitude floor X, per whisper model — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Continuous survival S7(X); markers at the reporting floors "
         "{0.25, 0.5, 0.75, 1.0, 1.5, 2.5}; dashed = 0.5 headline. whisper-large stays flat/high "
         "(µV ~40× the others — scale artifact).", ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec8b_x_vs_mmn_per_model_continuous.png", bbox_inches="tight")
plt.close(fig)

# ---- FIG 4: pooled /240, continuous X, 2×2 sites ----
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    s2 = int(sub.s2.sum())
    for xf in X_FLOORS:
        ax.axvline(xf, color="#ededed", lw=1.0, zorder=0)
    ax.axvline(0.5, color="#c9c9c9", lw=1.3, ls="--", zorder=0)
    ax.axhline(s2, color="#9a9a9a", lw=1.4, ls="--", zorder=1)
    ax.annotate(f"S2 = {s2} / 240", xy=(2.6, s2), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.3, color="#666")
    ax.plot(XGRID, s7_curve(sub), color=POOL_COLOR, lw=2.3, zorder=3)
    yv = present_counts(sub)[1:]
    ax.plot(X_FLOORS, yv, color=POOL_COLOR, marker="o", ms=7, ls="none", mec="white",
            mew=0.9, zorder=4)
    for xf, val in zip(X_FLOORS, yv):
        ax.annotate(str(val), (xf, val), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#222", fontweight="bold")
    ret = 100 * yv[X_FLOORS.index(0.5)] / s2 if s2 else 0
    ax.annotate(f"S7@0.5 / S2 = {yv[X_FLOORS.index(0.5)]}/{s2} = {ret:.0f}%  (headline)",
                xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.4, color=POOL_COLOR)
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 252)
    ax.set_title(title, fontweight="bold", loc="left")
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV, continuous)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 240, pooled over 5 models)")
fig.suptitle("mTRF MMN count vs continuous amplitude floor X, pooled over the 5 models — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Continuous survival S7(X) pooled /240; markers at the reporting floors "
         "{0.25, 0.5, 0.75, 1.0, 1.5, 2.5}; dashed horizontal = the S2 total (the X→0 ceiling).",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec8b_x_vs_mmn_pooled_continuous.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 5: trough_uv distribution per model, symlog x, 2×2 sites ============
fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2), sharex=True, sharey=True)
FLOORS = [-0.25, -0.5, -0.75, -1.0, -1.5, -2.5]
XLIM = (-700, 20)
rng = np.random.default_rng(0)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    for j, m in enumerate(MODEL_ORDER):
        pos = len(MODEL_ORDER) - j          # tiny at top (5) ... large at bottom (1)
        st = MODEL_STYLE[m]
        t = sub[(sub.model == m) & (sub.s2)].trough_uv.values
        if len(t) == 0:
            continue
        ax.boxplot([t], positions=[pos], vert=False, widths=0.58, patch_artist=True,
                   showfliers=False, medianprops=dict(color="#222", lw=1.5),
                   whiskerprops=dict(color=st["color"], lw=1.1),
                   capprops=dict(color=st["color"], lw=1.1),
                   boxprops=dict(facecolor=st["color"], alpha=0.28, edgecolor=st["color"], lw=1.2),
                   zorder=2)
        yj = pos + rng.uniform(-0.17, 0.17, len(t))
        ax.scatter(t, yj, s=18, color=st["color"], alpha=0.75, edgecolors="white",
                   linewidths=0.3, marker=st["marker"], zorder=4)
        ax.annotate(f"med {np.median(t):+.2f}", xy=(0.995, pos + 0.32),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7.0, color="#444")
    for xf in FLOORS:
        ax.axvline(xf, color="#888", ls=":", lw=1.0, zorder=1)
    ax.axvline(0, color="#9a9a9a", lw=1.2, ls="-", zorder=1)
    ax.set_xscale("symlog", linthresh=2, linscale=0.9)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-100, -10, -1, 0, 1])
    ax.set_xticklabels(["−100", "−10", "−1", "0", "1"])
    ax.set_yticks([5, 4, 3, 2, 1]); ax.set_yticklabels([MLABEL[m] for m in MODEL_ORDER])
    ax.set_ylim(0.4, 5.95)
    ax.set_title(title, fontweight="bold", loc="left")
    if i in BOTTOM:
        ax.set_xlabel("S2/S7 trough  (µV, symlog;  negative = deeper MMN)")
axes.flat[0].annotate("dotted = X floors {0.25, 0.5, 0.75, 1.0, 1.5, 2.5} µV · x-axis is symlog",
                      xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
                      fontsize=7.2, color="#888")
fig.suptitle("mTRF S2-passing trough distribution per model, and how each X floor cuts it — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, "Distribution of trough_uv over each model's S2-passing conditions "
         "(box = IQR + median; points = individual conditions). Dotted verticals = the amplitude "
         "floors X: a condition passes S7@X iff its trough sits left of that line. whisper-large's "
         "predicted µV are ~40× the other models (median ≈ −40 µV) — treat its S7 as scale-confounded.",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec8b_trough_uv_distribution.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 6 (Section 8c): paired Fz vs FCz predicted MMN trough (µV) ============
fz = d[d.roi == "Fz"].set_index(["model", "method", "is_counter"])
fcz = d[d.roi == "FCz"].set_index(["model", "method", "is_counter"])
j = fz[["trough_uv", "s2"]].join(fcz[["trough_uv", "s2"]], lsuffix="_fz", rsuffix="_fcz")
both = j[(j.s2_fz) & (j.s2_fcz)].reset_index()
diff = both.trough_uv_fz - both.trough_uv_fcz
n_pair = len(both)
fcz_deeper = int((both.trough_uv_fcz < both.trough_uv_fz).sum())
p_w = stats.wilcoxon(both.trough_uv_fz, both.trough_uv_fcz).pvalue

CLIP = (-4.0, 1.0)
fig, ax = plt.subplots(figsize=(6.4, 6.2))
ax.plot(CLIP, CLIP, color="#9a9a9a", ls="--", lw=1.2, zorder=1)      # y = x (equal depth)
ax.axhline(0, color="#e2e2e2", lw=0.8, zorder=0)
ax.axvline(0, color="#e2e2e2", lw=0.8, zorder=0)
n_off = 0
for m in MODEL_ORDER:
    st = MODEL_STYLE[m]
    sub = both[both.model == m]
    x = np.clip(sub.trough_uv_fcz, *CLIP)
    y = np.clip(sub.trough_uv_fz, *CLIP)
    n_off += int(((sub.trough_uv_fcz < CLIP[0]) | (sub.trough_uv_fz < CLIP[0])
                  | (sub.trough_uv_fcz > CLIP[1]) | (sub.trough_uv_fz > CLIP[1])).sum())
    ax.scatter(x, y, s=34, color=st["color"], marker=st["marker"], alpha=0.8,
               edgecolors="white", linewidths=0.5, label=MLABEL[m], zorder=3)
ax.set_xlim(*CLIP); ax.set_ylim(*CLIP); ax.set_aspect("equal")
ax.set_xlabel("FCz trough  (µV;  negative = deeper MMN)")
ax.set_ylabel("Fz trough  (µV;  negative = deeper MMN)")
ax.annotate(f"FCz deeper in {fcz_deeper}/{n_pair} matched S2 pairs\n"
            f"median (Fz − FCz) = {diff.median():+.2f} µV  (Wilcoxon p = {p_w:.3f})\n"
            f"above the dashed y = x line ⇒ FCz deeper",
            xy=(0.035, 0.975), xycoords="axes fraction", va="top", ha="left", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
if n_off:
    ax.annotate(f"{n_off} pair(s) off-scale (incl. whisper-large, ~40× deeper)", xy=(0.5, 0.015),
                xycoords="axes fraction", ha="center", va="bottom", fontsize=7.4, color="#777")
ax.legend(loc="lower right", frameon=False, fontsize=9, title="whisper model", title_fontsize=9)
ax.set_title("Fz vs FCz predicted MMN trough (µV) — matched mTRF conditions (S2 at both sites)",
             fontweight="bold", loc="left", fontsize=10)
fig.text(0.5, -0.05, "One point per (model × method × direction) condition with an S2 dip at both "
         "electrodes; trough_uv = deviant−standard µV at the S2 latency (negative = deeper). Axes "
         "clipped to [−4, 1] µV; whisper-large's inflated µV are off-scale but do not affect the "
         "sign test.", ha="center", fontsize=7.5, color="#666", wrap=True)
fig.tight_layout()
fig.savefig(f"{OUT}/sec8c_fz_vs_fcz_trough.png", bbox_inches="tight")
plt.close(fig)

print("saved 6 figures to", OUT)
print("rows = [S2, S7@0.25, S7@0.5, S7@0.75, S7@1.0, S7@1.5, S7@2.5]")
for kind, roi, title in SITES:
    sub = site_df(kind, roi)
    print(f"\n{title} (mtrf):")
    for m in MODEL_ORDER:
        print(f"  {MLABEL[m]:7}", present_counts(sub[sub.model == m]))
    print(f"  {'Total':7}", present_counts(sub), " (/240)")
print(f"\nFz vs FCz (S2-both n={n_pair}): FCz deeper {fcz_deeper}, "
      f"median(Fz-FCz)={diff.median():+.2f}, wilcoxon p={p_w:.4f}")
