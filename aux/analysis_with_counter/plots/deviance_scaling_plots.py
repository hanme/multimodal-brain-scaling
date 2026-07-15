#!/usr/bin/env python
"""Section 9 — deviance-scaling: MMN trough amplitude vs physical deviance size.

MMN amplitude = the S2/S7 trough: the deviant-standard difference wave in µV at the
S2 trough latency (`trough_uv` in mmn_s7_roi.csv), the exact quantity S7 gates on.
Restricted to the two canonical fronto-central reporting sites and split by level:
  parcel   = frontal
  electrode = FCz
Figures (2 panels = 2 levels) + binned CSV under aux/analysis_with_counter/plots/.
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/Users/sophiesigfstead/Documents/multimodal-brain-scaling-2"
OUT = f"{REPO}/aux/analysis_with_counter/plots"

# Okabe-Ito CVD-safe categorical pair (fixed order: mTRF, encoder)
C_MTRF, C_ENC = "#0072B2", "#D55E00"
STYLE = {"mtrf": dict(color=C_MTRF, marker="o", ls="-",  label="mTRF"),
         "encoder": dict(color=C_ENC, marker="s", ls="--", label="encoder")}
LEVELS = [("parcel", "frontal", "parcel (frontal)"),
          ("electrode", "FCz", "electrode (FCz)")]

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

# ---- data: S2/S7 trough (trough_uv), X-independent ----
s = pd.read_csv(f"{REPO}/outputs/results_with_counter/mmn_s7_roi.csv")
d = s[s.dip_uv_threshold == 0.25].copy()
d = d[((d.roi == "frontal") & (d.roi_kind == "parcel")) |
      ((d.roi == "FCz") & (d.roi_kind == "electrode"))].copy()
FREQ = {27:(1000,1064),37:(1000,1050),43:(633,700),44:(633,1000),53:(1000,1200),
        55:(1000,2000),60:(1000,1500),72:(1000,1200),74:(1000,1500),75:(1000,1200)}
d["mnum"] = d.method.str.extract(r"(\d+)").astype(int)
d["semitones"] = d["mnum"].map(lambda n: 12*abs(np.log2(FREQ[n][1]/FREQ[n][0])))
d["amp"] = d["trough_uv"]            # S2/S7 trough, signed µV (negative = deeper MMN)

def blk(sub):
    x, y = sub.semitones.values, sub.amp.values
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    rho, pr = stats.spearmanr(x, y)
    sl, ic, r, pv, se = stats.linregress(x, y)
    return dict(n=len(x), rho=rho, prho=pr, slope=sl, intercept=ic, pslope=pv)

xs = np.linspace(0, 12.5, 50)

def sub_of(kind, roi, mp):
    return d[(d.roi_kind == kind) & (d.roi == roi) & (d.mapping == mp)]

# ============ FIG 1: dose-response, 2 panels (levels), shared y ============
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
YCLIP = (-5.5, 1.5)
for ax, (kind, roi, title) in zip(axes, LEVELS):
    for mp in ["mtrf", "encoder"]:
        sub = sub_of(kind, roi, mp); st = STYLE[mp]; s_ = blk(sub)
        g = sub.groupby(sub.semitones.round(2)).amp.agg(["mean", "sem"]).reset_index()
        ax.errorbar(g.semitones, g["mean"], yerr=g["sem"], color=st["color"],
                    marker=st["marker"], ls="none", ms=7, capsize=3, lw=1.6, zorder=3,
                    mec="white", mew=0.8)
        ax.plot(xs, s_["intercept"] + s_["slope"]*xs, color=st["color"], ls=st["ls"],
                lw=2, alpha=0.9, zorder=2)
    ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
    ax.set_xlabel("Deviance size  (semitones)")
    ax.set_title(title, fontweight="bold", loc="left")
    sm, se = blk(sub_of(kind, roi, "mtrf")), blk(sub_of(kind, roi, "encoder"))
    leg = [Line2D([0],[0], color=C_MTRF, marker="o", ls="-", lw=2, ms=7, mec="white",
                  label=f"mTRF  (ρ={sm['rho']:+.2f}, p={sm['prho']:.2f})"),
           Line2D([0],[0], color=C_ENC, marker="s", ls="--", lw=2, ms=7, mec="white",
                  label=f"encoder (ρ={se['rho']:+.2f}, n.s.)")]
    ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=8.5)
axes[0].set_ylabel("S2/S7 trough  (µV;  negative = deeper MMN)")
axes[0].set_ylim(*YCLIP)
fig.suptitle("Deviance-scaling of the S2/S7 MMN trough (µV), by reporting site",
             fontweight="bold", x=0.01, ha="left")
fig.text(0.5, -0.02, "S2/S7 trough (deviant−standard µV at the S2 trough latency; negative = deeper); "
         "mean ± SEM per stimulus pair, pooled over 4 models × both directions (n=80 per site×mapping). "
         f"y clipped to [{YCLIP[0]}, {YCLIP[1]}] µV.", ha="center", fontsize=7.5, color="#666")
fig.tight_layout()
fig.savefig(f"{OUT}/deviance_scaling_dose_response.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: scatter, 2 panels (levels), shared y, clipped ============
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
SCLIP = (-8, 4)
rng = np.random.default_rng(0)
for ax, (kind, roi, title) in zip(axes, LEVELS):
    n_off = 0
    for mp in ["mtrf", "encoder"]:
        sub = sub_of(kind, roi, mp); st = STYLE[mp]; s_ = blk(sub)
        xj = sub.semitones + rng.uniform(-0.14, 0.14, len(sub))
        ax.scatter(xj, sub.amp, s=20, color=st["color"], alpha=0.5,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot(xs, s_["intercept"] + s_["slope"]*xs, color=st["color"], ls=st["ls"],
                lw=2.2, zorder=4)
        n_off += int((sub.amp < SCLIP[0]).sum() + (sub.amp > SCLIP[1]).sum())
    ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
    ax.set_xlabel("Deviance size (semitones)")
    ax.set_title(title, fontweight="bold", loc="left")
    sm, se = blk(sub_of(kind, roi, "mtrf")), blk(sub_of(kind, roi, "encoder"))
    ax.annotate(f"mTRF  ρ={sm['rho']:+.2f} (p={sm['prho']:.2f})\n"
                f"encoder ρ={se['rho']:+.2f} (n.s.)\n{n_off} pts off-scale",
                xy=(0.96, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc"))
axes[0].set_ylabel("S2/S7 trough (µV;  negative = deeper MMN)")
axes[0].set_ylim(*SCLIP)
fig.legend(handles=[Line2D([0],[0], color=C_MTRF, marker="o", ls="", ms=7, label="mTRF"),
                    Line2D([0],[0], color=C_ENC, marker="s", ls="", ms=7, label="encoder")],
           loc="upper right", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.99, 1.02))
fig.suptitle("S2/S7 MMN trough vs deviance size — raw points + OLS fit, by site",
             fontweight="bold", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"{OUT}/deviance_scaling_scatter.png", bbox_inches="tight")
plt.close(fig)

# ============ numbers for the doc ============
rows = []
print("=== per level × mapping (n=80) ===")
for kind, roi, title in LEVELS:
    for mp in ["mtrf", "encoder"]:
        sub = sub_of(kind, roi, mp); s_ = blk(sub)
        s2 = sub[sub.s2.astype(bool)]
        rho2 = stats.spearmanr(s2.semitones, s2.amp)[0] if len(s2) > 3 else float("nan")
        print(f"  {title:18} {mp:8} rho={s_['rho']:+.3f} p={s_['prho']:.1e} "
              f"slope={s_['slope']:+.4f} p={s_['pslope']:.1e}  S2-only rho={rho2:+.3f} (n={len(s2)})")
print("=== per-model mTRF Spearman ===")
for kind, roi, title in LEVELS:
    for m in ["whisper-tiny","whisper-base","whisper-small","whisper-medium"]:
        sub = sub_of(kind, roi, "mtrf"); sub = sub[sub.model == m]
        print(f"  {title:18} {m.split('-')[1]:7} rho={stats.spearmanr(sub.semitones, sub.amp)[0]:+.3f}")
print("=== dose-response binned means ===")
tab = d.groupby([d.roi, "mapping", d.semitones.round(2)]).amp.agg(["mean","count"]).round(3)
print(tab)
tab.reset_index().to_csv(f"{OUT}/deviance_scaling_binned.csv", index=False)
print("\nsaved figures + binned CSV to", OUT)
