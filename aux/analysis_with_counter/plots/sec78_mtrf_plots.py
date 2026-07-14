#!/usr/bin/env python
"""Sections 7 & 8 — mTRF amplitude-floor figures over the fronto-central reporting sites (2×2).

Reads outputs/results_with_counter/mmn_s7_roi.csv (mtrf only). S7@X is computed directly from the
X-independent `trough_uv` column:

    S7@X present  <=>  s2 AND (trough_uv <= -X)

so X = 0.75 (absent from the committed grid) is a one-line recompute like every other floor.
Each figure is a **2×2 grid** of reporting sites:

    frontal parcel  |  FCz electrode      (top row)
    central parcel  |  Fz  electrode      (bottom row)

Five figures:
  1. sec78_x_vs_mmn_per_model.png            — count /20 vs floor X ∈ {S2, 0.25, 0.5, 0.75, 1.0, 1.5}
  2. sec78_x_vs_mmn_pooled.png               — count /80 vs the same discrete floors, pooled
  3. sec78_x_vs_mmn_per_model_continuous.png — count /20 vs a CONTINUOUS floor X, per model
  4. sec78_x_vs_mmn_pooled_continuous.png    — count /80 vs a CONTINUOUS floor X, pooled
  5. sec78_trough_uv_distribution.png        — trough_uv (µV) per model over its S2-passing traces,
                                               with dotted floors at −0.25/−0.5/−0.75/−1.0/−1.5 µV
  6. sec78b_fz_vs_fcz_trough.png             — Section 8b: paired Fz vs FCz trough (µV), matched
                                               mTRF conditions (S2 at both sites), y = x reference
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/Users/sophiesigfstead/Documents/multimodal-brain-scaling-2"
# Superseded 10-method / 4-model figures — kept for reference under old_sec78_plots/.
# (The current 24-method / 5-model figures are produced by sec8b_mtrf_plots.py.)
OUT = f"{REPO}/aux/analysis_with_counter/plots/old_sec78_plots"

# per-model Okabe-Ito CVD-safe style (matches scripts/analyze_mmn_screen_24freq.py MODEL_STYLE);
# every series also carries a distinct marker shape, so identity is never colour-alone.
MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"]
MODEL_STYLE = {
    "whisper-tiny":   dict(color="#56B4E9", marker="o"),   # sky blue    ○
    "whisper-base":   dict(color="#0072B2", marker="s"),   # blue        □
    "whisper-small":  dict(color="#009E73", marker="^"),   # bluish green △
    "whisper-medium": dict(color="#E69F00", marker="D"),   # orange      ◇
}
MLABEL = {m: m.split("-")[1] for m in MODEL_ORDER}

# 2×2 reporting sites; flat order = top-left, top-right, bottom-left, bottom-right
SITES = [("parcel", "frontal", "frontal parcel"),
         ("electrode", "FCz", "FCz electrode"),
         ("parcel", "central", "central parcel"),
         ("electrode", "Fz", "Fz electrode")]
BOTTOM = {2, 3}   # flat panel indices in the bottom row  -> get the x-label
LEFT = {0, 2}     # flat panel indices in the left column -> get the y-label
LEGEND_PANEL = 1  # FCz (top-right) carries the per-model legend

# amplitude floors on the x-axis; S2 is the X->0 reference (first, evenly-spaced slot)
X_FLOORS = [0.25, 0.5, 0.75, 1.0, 1.5]
XLAB = ["S2\n(X→0)", "0.25", "0.5", "0.75", "1.0", "1.5"]
XPOS = [0, 1, 2, 3, 4, 5]
I05 = X_FLOORS.index(0.5) + 1   # x-slot / present_counts index of the 0.5 headline
POOL_COLOR = "#0072B2"          # single pooled series -> one Okabe-Ito blue

FIGSIZE = (10.8, 8.8)           # 2×2 line grids
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

# ---- data: mtrf only, one row per trace (trough_uv X-independent) ----
df = pd.read_csv(f"{REPO}/outputs/results_with_counter/mmn_s7_roi.csv")
d = df[(df.mapping == "mtrf") & (df.dip_uv_threshold == 0.25)].copy()


def site_df(kind, roi):
    return d[(d.roi_kind == kind) & (d.roi == roi)]


def present_counts(sub):
    """[S2, S7@0.25, S7@0.5, S7@0.75, S7@1.0, S7@1.5] present-count for a subframe."""
    out = [int(sub.s2.sum())]
    for X in X_FLOORS:
        out.append(int(((sub.s2) & (sub.trough_uv <= -X)).sum()))
    return out


CAP = ("S7@X = S2 ∧ (trough_uv ≤ −X µV); trough_uv = deviant−standard µV at the S2 trough latency "
       "(X-independent). mTRF only; 20 conditions/model = 10 literature methods × {regular, counter}.")

# per-model legend handles (reused by FIG 1 & FIG 3)
handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                  ls="-", lw=1.9, ms=7.5, mec="white", label=MLABEL[m]) for m in MODEL_ORDER]


# ============ FIG 1: per-model, 2×2 sites, shared axes ============
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    ax.axvline(0, color="#b5b5b5", lw=1.1, ls=":", zorder=1)   # marks the S2 (X→0) reference
    for m in MODEL_ORDER:
        st = MODEL_STYLE[m]
        ax.plot(XPOS, present_counts(sub[sub.model == m]), color=st["color"], marker=st["marker"],
                ms=7.5, lw=1.9, mec="white", mew=0.9, zorder=3)
    ax.set_xticks(XPOS); ax.set_xticklabels(XLAB)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylim(0, 20.6)
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 20 per model)")
axes.flat[LEGEND_PANEL].legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
                               title="whisper model", title_fontsize=9)
fig.suptitle("mTRF MMN count vs amplitude floor X, per whisper model — by fronto-central site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Count/20 per model; y at the leftmost slot is the S2 count.",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec78_x_vs_mmn_per_model.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: pooled /80, single series, 2×2 sites, shared axes ============
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    y = present_counts(sub)              # pooled over 4 models -> /80
    s2 = y[0]
    ax.axhline(s2, color="#9a9a9a", lw=1.4, ls="--", zorder=1)
    ax.annotate(f"S2 = {s2} / 80", xy=(5, s2), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color="#666")
    ax.plot(XPOS, y, color=POOL_COLOR, marker="o", ms=9, lw=2.4, mec="white", mew=1.0, zorder=3)
    for xp, val in zip(XPOS, y):
        ax.annotate(str(val), (xp, val), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color="#222", fontweight="bold")
    ret = 100 * y[I05] / s2 if s2 else 0
    ax.annotate(f"S7@0.5 / S2 = {y[I05]}/{s2} = {ret:.0f}%  (headline floor)", xy=(0.03, 0.06),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8.6, color=POOL_COLOR)
    ax.set_xticks(XPOS); ax.set_xticklabels(XLAB)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylim(0, 84)
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 80, pooled over 4 models)")
fig.suptitle("mTRF MMN count vs amplitude floor X, pooled over the 4 whisper models — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Pooled count/80; dashed line = the S2 total (the X→0 reference).",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec78_x_vs_mmn_pooled.png", bbox_inches="tight")
plt.close(fig)

# ============ continuous-X survival curves (companions to FIG 1 & 2) ============
# S7(X) is a step function of the continuous floor X: present iff trough_uv <= -X.
XGRID = np.linspace(0.0, 2.5, 501)


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
        ax.plot(XGRID, s7_curve(ms), color=st["color"], lw=1.9, zorder=3)
        ax.plot(X_FLOORS, present_counts(ms)[1:], color=st["color"], marker=st["marker"],
                ms=6.5, ls="none", mec="white", mew=0.8, zorder=4)
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 20.6)
    ax.set_title(title, fontweight="bold", loc="left")
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV, continuous)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 20 per model)")
axes.flat[LEGEND_PANEL].legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
                               title="whisper model", title_fontsize=9)
fig.suptitle("mTRF MMN count vs continuous amplitude floor X, per whisper model — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Continuous survival S7(X); markers at the reporting floors "
         "{0.25, 0.5, 0.75, 1.0, 1.5}; dashed = 0.5 headline. y at X→0 is S7@0 ≈ S2.",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec78_x_vs_mmn_per_model_continuous.png", bbox_inches="tight")
plt.close(fig)

# ---- FIG 4: pooled /80, continuous X, 2×2 sites ----
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    s2 = int(sub.s2.sum())
    for xf in X_FLOORS:
        ax.axvline(xf, color="#ededed", lw=1.0, zorder=0)
    ax.axvline(0.5, color="#c9c9c9", lw=1.3, ls="--", zorder=0)
    ax.axhline(s2, color="#9a9a9a", lw=1.4, ls="--", zorder=1)
    ax.annotate(f"S2 = {s2} / 80", xy=(2.5, s2), xytext=(0, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color="#666")
    ax.plot(XGRID, s7_curve(sub), color=POOL_COLOR, lw=2.4, zorder=3)
    yv = present_counts(sub)[1:]
    ax.plot(X_FLOORS, yv, color=POOL_COLOR, marker="o", ms=8, ls="none", mec="white",
            mew=1.0, zorder=4)
    for xf, val in zip(X_FLOORS, yv):
        ax.annotate(str(val), (xf, val), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color="#222", fontweight="bold")
    ret = 100 * yv[X_FLOORS.index(0.5)] / s2 if s2 else 0
    ax.annotate(f"S7@0.5 / S2 = {yv[X_FLOORS.index(0.5)]}/{s2} = {ret:.0f}%  (headline)",
                xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.6, color=POOL_COLOR)
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 84)
    ax.set_title(title, fontweight="bold", loc="left")
    if i in BOTTOM:
        ax.set_xlabel("amplitude floor  X  (µV, continuous)")
    if i in LEFT:
        ax.set_ylabel("MMN present  (count / 80, pooled over 4 models)")
fig.suptitle("mTRF MMN count vs continuous amplitude floor X, pooled over the 4 models — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, CAP + "  Continuous survival S7(X) pooled /80; markers at the reporting floors "
         "{0.25, 0.5, 0.75, 1.0, 1.5}; dashed horizontal = the S2 total (the X→0 ceiling).",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec78_x_vs_mmn_pooled_continuous.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 5: trough_uv distribution per model, 2×2 sites ============
fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), sharex=True, sharey=True)
XCLIP = (-5.0, 1.3)
FLOORS = [-0.25, -0.5, -0.75, -1.0, -1.5]
rng = np.random.default_rng(0)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    n_off = 0
    for j, m in enumerate(MODEL_ORDER):
        pos = len(MODEL_ORDER) - j      # tiny at top (4) ... medium at bottom (1)
        st = MODEL_STYLE[m]
        t = sub[(sub.model == m) & (sub.s2)].trough_uv.values
        if len(t) == 0:
            continue
        ax.boxplot([t], positions=[pos], vert=False, widths=0.56, patch_artist=True,
                   showfliers=False, medianprops=dict(color="#222", lw=1.7),
                   whiskerprops=dict(color=st["color"], lw=1.3),
                   capprops=dict(color=st["color"], lw=1.3),
                   boxprops=dict(facecolor=st["color"], alpha=0.28, edgecolor=st["color"], lw=1.4),
                   zorder=2)
        tc = np.clip(t, XCLIP[0], None)                       # pile deep outliers at the left edge
        yj = pos + rng.uniform(-0.17, 0.17, len(t))
        ax.scatter(tc, yj, s=22, color=st["color"], alpha=0.8, edgecolors="white",
                   linewidths=0.4, marker=st["marker"], zorder=4)
        n_off += int((t < XCLIP[0]).sum())
        ax.annotate(f"med {np.median(t):+.2f}  (n={len(t)})", xy=(XCLIP[1], pos + 0.33),
                    xytext=(-4, 0), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.6, color="#444")
    for xf in FLOORS:
        ax.axvline(xf, color="#888", ls=":", lw=1.0, zorder=1)
    ax.axvline(0, color="#9a9a9a", lw=1.2, ls="-", zorder=1)
    if n_off:
        ax.annotate(f"{n_off} deeper than {XCLIP[0]:.0f} µV → shown at left edge",
                    xy=(0.02, 0.02), xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=7.4, color="#777")
    ax.set_yticks([4, 3, 2, 1]); ax.set_yticklabels([MLABEL[m] for m in MODEL_ORDER])
    ax.set_ylim(0.4, 4.9)
    ax.set_xlim(*XCLIP)
    ax.set_title(title, fontweight="bold", loc="left")
    if i in BOTTOM:
        ax.set_xlabel("S2/S7 trough  (µV;  negative = deeper MMN)")
axes.flat[0].annotate("dotted verticals = X floors {0.25, 0.5, 0.75, 1.0, 1.5} µV", xy=(0.02, 0.98),
                      xycoords="axes fraction", ha="left", va="top", fontsize=7.6, color="#888")
fig.suptitle("mTRF S2-passing trough distribution per model, and how each X floor cuts it — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, 0.005, "Distribution of trough_uv over each model's S2-passing conditions "
         "(box = IQR + median; points = individual conditions). Dotted verticals = the amplitude "
         "floors X ∈ {0.25, 0.5, 0.75, 1.0, 1.5} µV: a condition passes S7@X iff its trough sits left "
         "of that line.", ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{OUT}/sec78_trough_uv_distribution.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 6 (Section 8b): paired Fz vs FCz predicted MMN trough (µV), mtrf ============
# Match the two midline electrodes on (model, method, direction) and compare trough depth.
fz = d[d.roi == "Fz"].set_index(["model", "method", "is_counter"])
fcz = d[d.roi == "FCz"].set_index(["model", "method", "is_counter"])
j = fz[["trough_uv", "s2"]].join(fcz[["trough_uv", "s2"]], lsuffix="_fz", rsuffix="_fcz")
both = j[(j.s2_fz) & (j.s2_fcz)].reset_index()          # conditions with an S2 dip at BOTH sites
diff = both.trough_uv_fz - both.trough_uv_fcz            # >0 => Fz shallower (FCz deeper)
n_pair = len(both)
fcz_deeper = int((both.trough_uv_fcz < both.trough_uv_fz).sum())
r_ff = np.corrcoef(both.trough_uv_fz, both.trough_uv_fcz)[0, 1]
p_w = stats.wilcoxon(both.trough_uv_fz, both.trough_uv_fcz).pvalue

CLIP = (-3.5, 0.9)
fig, ax = plt.subplots(figsize=(6.4, 6.2))
ax.plot(CLIP, CLIP, color="#9a9a9a", ls="--", lw=1.2, zorder=1)   # y = x (equal depth)
ax.axhline(0, color="#e2e2e2", lw=0.8, zorder=0)
ax.axvline(0, color="#e2e2e2", lw=0.8, zorder=0)
n_off = 0
for m in MODEL_ORDER:
    st = MODEL_STYLE[m]
    sub = both[both.model == m]
    x = np.clip(sub.trough_uv_fcz, *CLIP)
    y = np.clip(sub.trough_uv_fz, *CLIP)
    n_off += int(((sub.trough_uv_fcz < CLIP[0]) | (sub.trough_uv_fz < CLIP[0])).sum())
    ax.scatter(x, y, s=36, color=st["color"], marker=st["marker"], alpha=0.8,
               edgecolors="white", linewidths=0.5, label=MLABEL[m], zorder=3)
ax.set_xlim(*CLIP); ax.set_ylim(*CLIP); ax.set_aspect("equal")
ax.set_xlabel("FCz trough  (µV;  negative = deeper MMN)")
ax.set_ylabel("Fz trough  (µV;  negative = deeper MMN)")
ax.annotate(f"FCz deeper in {fcz_deeper}/{n_pair} matched S2 pairs\n"
            f"median (Fz − FCz) = {diff.median():+.2f} µV  (Wilcoxon p = {p_w:.3f})\n"
            f"Fz ↔ FCz trough r = {r_ff:+.2f}\n"
            f"above the dashed y = x line ⇒ FCz deeper",
            xy=(0.035, 0.975), xycoords="axes fraction", va="top", ha="left", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
if n_off:
    ax.annotate(f"{n_off} pair(s) clipped to the left/bottom edge", xy=(0.5, 0.015),
                xycoords="axes fraction", ha="center", va="bottom", fontsize=7.4, color="#777")
ax.legend(loc="lower right", frameon=False, fontsize=9, title="whisper model", title_fontsize=9)
ax.set_title("Fz vs FCz predicted MMN trough (µV) — matched mTRF conditions (S2 at both sites)",
             fontweight="bold", loc="left", fontsize=10)
fig.text(0.5, -0.05, "One point per (model × method × direction) condition where both electrodes show "
         "an S2 dip; trough_uv = deviant−standard µV at the S2 latency (negative = deeper). Points on "
         "the dashed line have equal depth; the two sites are strongly correlated and FCz is the "
         "modestly deeper of the two.", ha="center", fontsize=7.5, color="#666", wrap=True)
fig.tight_layout()
fig.savefig(f"{OUT}/sec78b_fz_vs_fcz_trough.png", bbox_inches="tight")
plt.close(fig)

print("saved 6 figures to", OUT)
print(f"Fz vs FCz (mtrf, S2-both n={n_pair}): FCz deeper in {fcz_deeper}, "
      f"median(Fz-FCz)={diff.median():+.2f}uV, r={r_ff:+.2f}, wilcoxon p={p_w:.3f}")
print("rows = [S2, S7@0.25, S7@0.5, S7@0.75, S7@1.0, S7@1.5]")
for kind, roi, title in SITES:
    sub = site_df(kind, roi)
    print(f"\n{title} (mtrf):")
    for m in MODEL_ORDER:
        print(f"  {MLABEL[m]:7}", present_counts(sub[sub.model == m]))
    print(f"  {'Total':7}", present_counts(sub), " (/80)")
