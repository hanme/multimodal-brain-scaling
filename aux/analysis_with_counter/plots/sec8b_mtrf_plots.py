#!/usr/bin/env python
"""Sections 8b/8c — mTRF amplitude-floor figures for the 24-method screen (all 7 models).

Reads outputs/results_24freq_7models/mmn_s7_roi.csv (mTRF only). S7@X from the X-independent
`trough_uv` column:  S7@X present  <=>  s2 AND (trough_uv <= -X).  24 frequency methods ×
{regular, counter} = 48 conditions per model per site; /48 per model, /336 pooled over 7 models.

Models: whisper {tiny, base, small, medium, large} + wav2vec2 {medium, large}.

Each Section-8b figure is a 2-panel row of the two canonical reporting sites:
    frontal parcel  |  FCz electrode

Section 8b (3 figures):
  1. sec8b_x_vs_mmn_per_model.png      — count /48 vs floor X ∈ {S2(X→0), 0.5, 0.75, 1.0, 1.5}
  2. sec8b_x_vs_mmn_pooled.png         — count /336 vs the same floors, pooled over the 7 models
  3. sec8b_trough_uv_distribution.png  — trough_uv (µV) per model over its S2-passing conditions,
                                         dotted floors at −0.5/−0.75/−1.0/−1.5 µV (symlog x —
                                         whisper-large's predicted µV are ~40× the others, a
                                         scale artifact)
Section 8c (1 figure):
  4. sec8c_fz_vs_fcz_trough.png        — paired Fz vs FCz predicted trough (µV), matched conditions
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/Users/sophiesigfstead/Documents/multimodal-brain-scaling-2"
OUT = f"{REPO}/aux/analysis_with_counter/plots"
CSV = f"{REPO}/outputs/results_24freq_7models/mmn_s7_roi.csv"

# 7-model Okabe-Ito CVD-safe style (extends scripts/analyze_mmn_screen_24freq.py MODEL_STYLE with
# the two wav2vec2 slots). Worst pair separation under protan/deutan simulation is ΔE = 17.9,
# above the ΔE ≥ 12 target; every series also carries a distinct marker, so identity is never
# colour-alone.
MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large",
               "wav2vec2-medium", "wav2vec2-large"]
MODEL_STYLE = {
    "whisper-tiny":    dict(color="#56B4E9", marker="o"),   # sky blue        ○
    "whisper-base":    dict(color="#0072B2", marker="s"),   # blue            □
    "whisper-small":   dict(color="#009E73", marker="^"),   # bluish green    △
    "whisper-medium":  dict(color="#E69F00", marker="D"),   # orange          ◇
    "whisper-large":   dict(color="#D55E00", marker="v"),   # vermillion      ▽
    "wav2vec2-medium": dict(color="#CC79A7", marker="P"),   # reddish purple  ✚
    "wav2vec2-large":  dict(color="#000000", marker="X"),   # black           ✖
}
# explicit labels: "medium"/"large" alone would collide across the two families
MLABEL = {"whisper-tiny": "whisper tiny", "whisper-base": "whisper base",
          "whisper-small": "whisper small", "whisper-medium": "whisper medium",
          "whisper-large": "whisper large", "wav2vec2-medium": "wav2vec2 medium",
          "wav2vec2-large": "wav2vec2 large"}

N_MODELS = len(MODEL_ORDER)
N_COND = 48                       # per model per site
N_POOL = N_COND * N_MODELS        # 336

# the two canonical reporting sites; panel order = left, right
SITES = [("parcel", "frontal", "frontal parcel"),
         ("electrode", "FCz", "FCz electrode")]
LEGEND_PANEL = 1                  # FCz (right) carries the per-model legend

# amplitude floors on the x-axis; S2 is the X->0 reference (first, evenly-spaced slot)
X_FLOORS = [0.5, 0.75, 1.0, 1.5]
XLAB = ["S2\n(X→0)", "0.5", "0.75", "1.0", "1.5"]
XPOS = list(range(len(X_FLOORS) + 1))
I05 = X_FLOORS.index(0.5) + 1     # present_counts / x-slot index of the 0.5 headline
POOL_COLOR = "#0072B2"

FIGSIZE = (11.6, 5.4)             # 1×2 line grids
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

df = pd.read_csv(CSV)
d = df[(df.mapping == "mtrf") & (df.dip_uv_threshold == 0.25)].copy()   # one row per trace

missing = set(MODEL_ORDER) - set(d.model.unique())
assert not missing, f"missing models in {CSV}: {sorted(missing)}"


def site_df(kind, roi):
    return d[(d.roi_kind == kind) & (d.roi == roi)]


def present_counts(sub):
    """[S2, S7@0.5, S7@0.75, S7@1.0, S7@1.5] present-count for a subframe."""
    out = [int(sub.s2.sum())]
    for X in X_FLOORS:
        out.append(int(((sub.s2) & (sub.trough_uv <= -X)).sum()))
    return out


# verify the screen is complete before plotting anything off it
for kind, roi, title in SITES:
    g = site_df(kind, roi).groupby("model").size()
    assert (g == N_COND).all() and len(g) == N_MODELS, f"{title}: expected {N_COND}×{N_MODELS}\n{g}"

CAP = ("S7@X = S2 ∧ (trough_uv ≤ −X µV); trough_uv = deviant−standard µV at the S2 trough latency "
       "(X-independent). mTRF only; 48 conditions/model = 24 frequency methods × {regular, counter}.")
handles = [Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                  ls="-", lw=1.9, ms=7.5, mec="white", label=MLABEL[m]) for m in MODEL_ORDER]

# ============ FIG 1: per-model, count /48 vs floor, frontal | FCz ============
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
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
    ax.set_xlabel("amplitude floor  X  (µV)")
    if i == 0:
        ax.set_ylabel(f"MMN present  (count / {N_COND} per model)")
# figure-level legend below the panels: the per-model lines descend across the whole plot area,
# so an in-axes legend collides with them at every corner.
fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=8.6, ncol=7,
           columnspacing=1.1, handletextpad=0.5, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("mTRF MMN count vs amplitude floor X, per model — by fronto-central site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, -0.10, CAP + f"  Count/{N_COND} per model; y at the leftmost slot is the S2 count.",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{OUT}/sec8b_x_vs_mmn_per_model.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: pooled /336, single series, frontal | FCz ============
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True, sharey=True)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    y = present_counts(sub)              # pooled over 7 models -> /336
    s2 = y[0]
    ax.axhline(s2, color="#9a9a9a", lw=1.4, ls="--", zorder=1)
    ax.annotate(f"S2 = {s2} / {N_POOL}", xy=(XPOS[-1], s2), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.3, color="#666")
    ax.plot(XPOS, y, color=POOL_COLOR, marker="o", ms=8, lw=2.3, mec="white", mew=1.0, zorder=3)
    for xp, val in zip(XPOS, y):
        ax.annotate(str(val), (xp, val), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#222", fontweight="bold")
    ret = 100 * y[I05] / s2 if s2 else 0
    ax.annotate(f"S7@0.5 / S2 = {y[I05]}/{s2} = {ret:.0f}%  (headline)", xy=(0.03, 0.06),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8.4, color=POOL_COLOR)
    ax.set_xticks(XPOS); ax.set_xticklabels(XLAB)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_ylim(0, 348)
    ax.set_xlabel("amplitude floor  X  (µV)")
    if i == 0:
        ax.set_ylabel(f"MMN present  (count / {N_POOL}, pooled over {N_MODELS} models)")
fig.suptitle(f"mTRF MMN count vs amplitude floor X, pooled over the {N_MODELS} models — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, -0.03, CAP + f"  Pooled count/{N_POOL}; dashed line = the S2 total (the X→0 "
         "reference). The pool mixes whisper and wav2vec2, which are not a strictly controlled "
         "comparison (see Section 8c) — read it as a convenience summary.",
         ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{OUT}/sec8b_x_vs_mmn_pooled.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 3: trough_uv distribution per model, symlog x, frontal | FCz ============
fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0), sharex=True, sharey=True)
FLOORS = [-0.5, -0.75, -1.0, -1.5]
XLIM = (-700, 20)
rng = np.random.default_rng(0)
for i, (ax, (kind, roi, title)) in enumerate(zip(axes.flat, SITES)):
    sub = site_df(kind, roi)
    for j, m in enumerate(MODEL_ORDER):
        pos = N_MODELS - j          # whisper-tiny at top (7) ... wav2vec2-large at bottom (1)
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
        ax.annotate(f"med {np.median(t):+.2f}  (n={len(t)})", xy=(0.995, pos + 0.30),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=7.0, color="#444")
    for xf in FLOORS:
        ax.axvline(xf, color="#888", ls=":", lw=1.0, zorder=1)
    ax.axvline(0, color="#9a9a9a", lw=1.2, ls="-", zorder=1)
    ax.set_xscale("symlog", linthresh=2, linscale=0.9)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-100, -10, -1, 0, 1])
    ax.set_xticklabels(["−100", "−10", "−1", "0", "1"])
    ax.set_yticks(list(range(N_MODELS, 0, -1)))
    ax.set_yticklabels([MLABEL[m] for m in MODEL_ORDER])
    ax.set_ylim(0.4, N_MODELS + 0.95)
    ax.set_title(title, fontweight="bold", loc="left")
    ax.set_xlabel("S2/S7 trough  (µV, symlog;  negative = deeper MMN)")
axes.flat[0].annotate("dotted = X floors {0.5, 0.75, 1.0, 1.5} µV · x-axis is symlog",
                      xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
                      fontsize=7.2, color="#888")
fig.suptitle("mTRF S2-passing trough distribution per model, and how each X floor cuts it — by site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, -0.04, "Distribution of trough_uv over each model's S2-passing conditions "
         "(box = IQR + median; points = individual conditions). Dotted verticals = the amplitude "
         "floors X: a condition passes S7@X iff its trough sits left of that line. whisper-large's "
         "predicted µV are ~40× the other models (median ≈ −46 µV frontal / −34 µV FCz) — treat its "
         "S7 as scale-confounded.", ha="center", fontsize=7.6, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{OUT}/sec8b_trough_uv_distribution.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 4 (Section 8c): paired Fz vs FCz predicted MMN trough (µV) ============
fz = d[d.roi == "Fz"].set_index(["model", "method", "is_counter"])
fcz = d[d.roi == "FCz"].set_index(["model", "method", "is_counter"])
j = fz[["trough_uv", "s2"]].join(fcz[["trough_uv", "s2"]], lsuffix="_fz", rsuffix="_fcz")
both = j[(j.s2_fz) & (j.s2_fcz)].dropna(subset=["trough_uv_fz", "trough_uv_fcz"]).reset_index()
diff = both.trough_uv_fz - both.trough_uv_fcz
n_pair = len(both)
fcz_deeper = int((both.trough_uv_fcz < both.trough_uv_fz).sum())
p_w = stats.wilcoxon(both.trough_uv_fz, both.trough_uv_fcz).pvalue

CLIP = (-4.0, 1.0)
fig, ax = plt.subplots(figsize=(6.6, 6.4))
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
ax.legend(loc="lower right", frameon=False, fontsize=8.2, title="model", title_fontsize=8.4)
ax.set_title("Fz vs FCz predicted MMN trough (µV) — matched mTRF conditions (S2 at both sites)",
             fontweight="bold", loc="left", fontsize=10)
# the off-scale count lives in the caption, not in-axes: every in-axes corner is taken by the
# stat box, the legend, or the point cloud.
fig.text(0.5, -0.05, "One point per (model × method × direction) condition with an S2 dip at both "
         "electrodes; trough_uv = deviant−standard µV at the S2 latency (negative = deeper). Axes "
         f"clipped to [−4, 1] µV, which puts {n_off} of {n_pair} pairs off-scale (mostly "
         "whisper-large, ~40× deeper); clipping is display-only — all pairs are in the sign test "
         "and the Wilcoxon.", ha="center", fontsize=7.5, color="#666", wrap=True)
fig.tight_layout()
fig.savefig(f"{OUT}/sec8c_fz_vs_fcz_trough.png", bbox_inches="tight")
plt.close(fig)

print("saved 4 figures to", OUT)
print("rows = [S2, S7@0.5, S7@0.75, S7@1.0, S7@1.5]")
for kind, roi, title in SITES:
    sub = site_df(kind, roi)
    print(f"\n{title} (mtrf):")
    for m in MODEL_ORDER:
        print(f"  {MLABEL[m]:16}", present_counts(sub[sub.model == m]))
    print(f"  {'Total':16}", present_counts(sub), f" (/{N_POOL})")
    med = sub[sub.s2].trough_uv.median()
    print(f"  median S2-passing trough = {med:+.2f} µV")
print(f"\nFz vs FCz (S2-both n={n_pair}): FCz deeper {fcz_deeper}, "
      f"median(Fz-FCz)={diff.median():+.2f}, wilcoxon p={p_w:.5f}")
