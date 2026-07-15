#!/usr/bin/env python
"""Section 9 (24-method / 7-model copy) — deviance-scaling of the S2/S7 MMN trough, mTRF only.

The 7-model companion to deviance_scaling_plots.py (which stays on the 20-method / 4-model
mTRF+encoder vintage and is NOT touched by this script). Same quantity, same two reporting sites:

  MMN amplitude = the S2/S7 trough: deviant-standard difference wave in µV at the S2 trough
  latency (`trough_uv`, X-independent), at parcel=frontal and electrode=FCz.

Scope: 24 frequency methods × {regular, counter} = 48 conditions per model per site, mTRF only,
7 models (whisper tiny..large + wav2vec2 medium/large) = 336 rows per site.
Deviance size = 12·|log2(f_dev/f_std)| semitones, read from the SAME canonical metadata CSV the
stimulus generator uses (data/metadata/..., change_type == "Frequency"), not a hardcoded dict.

TWO DELIBERATE DEPARTURES from the 4-model original, both forced by the data:

  1. PER MODEL, NEVER POOLED IN RAW µV. whisper-large's predicted µV run ~40× every other model
     (Section 7, scale caveat 2), so a pooled mean per deviance bin is dominated by it and is an
     artifact of feature-norm scale, not of deviance (frontal, 7.02 st: pooled mean -13.2 µV with
     whisper-large vs -1.4 µV without). Each model therefore gets its own series/panel, and the
     y-axis is symlog so all 7 scales coexist honestly.
  2. MEDIAN, not mean±SEM, per deviance bin. Within a single model the trough tails are heavy
     (whisper-large reaches -383 µV, whisper-medium -36 µV) and some bins hold as few as 2
     conditions, so a bin mean is unstable. The primary statistic is Spearman rho (rank-based,
     scale-free) exactly as in the original.

Figures + stats CSV under aux/analysis_with_counter/plots/:
  1. deviance_scaling_dose_response_24freq_7models.png — 2 panels (frontal | FCz), 7 model lines,
     median trough per deviance size, symlog y.
  2. deviance_scaling_scatter_24freq_7models.png — small multiples, 2 rows (sites) × 7 cols
     (models), raw points + per-model OLS fit + rho; each panel keeps its OWN y-scale (the only
     scale-correct way to show a ~40× spread of models side by side).
"""
import csv
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = "/Users/sophiesigfstead/Documents/multimodal-brain-scaling-2"
OUT = f"{REPO}/aux/analysis_with_counter/plots"
CSV_IN = f"{REPO}/outputs/results_24freq_7models/mmn_s7_roi.csv"
META = f"{REPO}/data/metadata/literature_frequency_intensity_duration_metadata.csv"

# 7-model Okabe-Ito style, identical to sec8b_mtrf_plots.py / analyze_mmn_screen_24freq.py
MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large",
               "wav2vec2-medium", "wav2vec2-large"]
MODEL_STYLE = {
    "whisper-tiny":    dict(color="#56B4E9", marker="o"),
    "whisper-base":    dict(color="#0072B2", marker="s"),
    "whisper-small":   dict(color="#009E73", marker="^"),
    "whisper-medium":  dict(color="#E69F00", marker="D"),
    "whisper-large":   dict(color="#D55E00", marker="v"),
    "wav2vec2-medium": dict(color="#CC79A7", marker="P"),
    "wav2vec2-large":  dict(color="#000000", marker="X"),
}
MLABEL = {"whisper-tiny": "whisper tiny", "whisper-base": "whisper base",
          "whisper-small": "whisper small", "whisper-medium": "whisper medium",
          "whisper-large": "whisper large", "wav2vec2-medium": "wav2vec2 medium",
          "wav2vec2-large": "wav2vec2 large"}
SITES = [("parcel", "frontal", "parcel (frontal)"), ("electrode", "FCz", "electrode (FCz)")]

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

# ---- deviance size from the canonical stimulus metadata (same source as insilico_mmn.py) ----
FREQ = {int(r["method_id"]): (float(r["standard_freq"]), float(r["deviant_freq"]))
        for r in csv.DictReader(open(META))
        if r.get("change_type", "").strip().lower() == "frequency"}
assert len(FREQ) == 24, f"expected 24 frequency methods in metadata, got {len(FREQ)}"

d = pd.read_csv(CSV_IN)
d = d[(d.mapping == "mtrf") & np.isclose(d.dip_uv_threshold, 0.25)].copy()   # one row per trace
d = d[((d.roi == "frontal") & (d.roi_kind == "parcel")) |
      ((d.roi == "FCz") & (d.roi_kind == "electrode"))].copy()
d["mnum"] = d.method.str.extract(r"(\d+)").astype(int)
assert set(d.mnum) <= set(FREQ), f"methods missing from metadata: {sorted(set(d.mnum) - set(FREQ))}"
d["semitones"] = d["mnum"].map(lambda n: 12 * abs(np.log2(FREQ[n][1] / FREQ[n][0]))).round(2)
d["amp"] = d["trough_uv"]              # S2/S7 trough, signed µV (negative = deeper MMN)

# scope invariant: 48 conditions × 7 models at each site
for kind, roi, title in SITES:
    g = d[d.roi == roi].groupby("model").size()
    assert len(g) == 7 and (g == 48).all(), f"{title}: expected 48×7\n{g}"


def blk(sub):
    x, y = sub.semitones.values, sub.amp.values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, pr = stats.spearmanr(x, y)
    sl, ic, r, pv, se = stats.linregress(x, y)
    return dict(n=len(x), rho=rho, prho=pr, slope=sl, intercept=ic, pslope=pv)


def sub_of(roi, m=None):
    s = d[d.roi == roi]
    return s if m is None else s[s.model == m]


# ============ FIG 1: dose-response, 2 panels (sites), 7 model lines, symlog y ============
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6), sharey=True)
handles = []
for ax, (kind, roi, title) in zip(axes, SITES):
    for m in MODEL_ORDER:
        sub = sub_of(roi, m)
        st = MODEL_STYLE[m]
        g = sub.groupby("semitones").amp.median().reset_index()
        s_ = blk(sub)
        ln, = ax.plot(g.semitones, g.amp, color=st["color"], marker=st["marker"], ms=6, lw=1.7,
                      mec="white", mew=0.7, zorder=3)
        if roi == "frontal":       # per-panel rho differs, so the shared legend carries the name only
            handles.append(Line2D([0], [0], color=st["color"], marker=st["marker"], lw=1.7, ms=6,
                                  mec="white", label=MLABEL[m]))
    ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
    ax.set_yscale("symlog", linthresh=1, linscale=0.8)
    ax.set_xlabel("Deviance size  (semitones)")
    ax.set_title(title, fontweight="bold", loc="left")
axes[0].set_ylabel("S2/S7 trough  (µV, symlog;  negative = deeper MMN)")
# figure-level legend: the model lines wander across every corner of both panels, so an in-axes
# legend collides with whisper-large. Per-model rho is on the small-multiples figure instead.
fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=8.4, ncol=7,
           columnspacing=1.0, handletextpad=0.5, bbox_to_anchor=(0.5, -0.045))
fig.suptitle("Deviance-scaling of the S2/S7 MMN trough (µV) per model — 24 methods × {regular, "
             "counter}, mTRF", fontweight="bold", x=0.01, ha="left")
fig.text(0.5, -0.13, "Median S2/S7 trough per deviance size, one line per model (48 conditions per "
         "model per site); per-model Spearman ρ is on the small-multiples figure. y is symlog: "
         "whisper-large's predicted µV run ~40× the others (Section 7 scale caveat), so the models "
         "are NOT pooled — a pooled raw-µV mean would track feature-norm scale, not deviance. "
         "Median (not mean) per bin: the within-model tails are heavy and some bins hold 1–2 methods.",
         ha="center", fontsize=7.5, color="#666", wrap=True)
fig.tight_layout()
fig.savefig(f"{OUT}/deviance_scaling_dose_response_24freq_7models.png", bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: small multiples — 2 rows (sites) × 7 cols (models), own y per panel ========
fig, axes = plt.subplots(2, 7, figsize=(19.0, 6.2), sharex=True)
xs = np.linspace(0, 12.5, 50)
rng = np.random.default_rng(0)
for r, (kind, roi, title) in enumerate(SITES):
    for c, m in enumerate(MODEL_ORDER):
        ax = axes[r, c]
        sub = sub_of(roi, m)
        st = MODEL_STYLE[m]
        s_ = blk(sub)
        xj = sub.semitones + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(xj, sub.amp, s=17, color=st["color"], alpha=0.62, edgecolors="white",
                   linewidths=0.35, marker=st["marker"], zorder=3)
        ax.plot(xs, s_["intercept"] + s_["slope"] * xs, color=st["color"], lw=1.9, zorder=4)
        ax.axhline(0, color="#9a9a9a", lw=0.9, ls=":", zorder=1)
        sig = "*" if s_["prho"] < 0.05 else ""
        ax.annotate(f"ρ={s_['rho']:+.2f}{sig}\np={s_['prho']:.2f}", xy=(0.95, 0.05),
                    xycoords="axes fraction", ha="right", va="bottom", fontsize=7.6,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#ddd", alpha=0.9))
        if r == 0:
            ax.set_title(MLABEL[m], fontweight="bold", fontsize=9)
        if r == 1:
            ax.set_xlabel("deviance (st)", fontsize=8.5)
        if c == 0:
            ax.set_ylabel(f"{title}\nS2/S7 trough (µV)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
fig.suptitle("S2/S7 MMN trough vs deviance size — raw points + OLS fit, per model × site "
             "(24 methods × {regular, counter}, mTRF)", fontweight="bold", x=0.01, ha="left")
fig.text(0.5, -0.02, "One point per condition (48 per panel); line = OLS fit; ρ = Spearman "
         "(primary — rank-based, robust to the deep µV outliers that make OLS unreliable); "
         "* = p < 0.05. Each panel keeps its OWN y-scale: whisper-large's µV are ~40× the others, so "
         "a shared axis would flatten every other model to a line. Negative = deeper MMN; a negative "
         "ρ is the human-like (deepening) direction.", ha="center", fontsize=7.6, color="#666",
         wrap=True)
fig.tight_layout()
fig.savefig(f"{OUT}/deviance_scaling_scatter_24freq_7models.png", bbox_inches="tight")
plt.close(fig)

# ============ numbers for the doc ============
print("=== pooled per site (mTRF, n=336) — RANK stat only; raw-µV pooling is scale-confounded ===")
rows = []
for kind, roi, title in SITES:
    sub = sub_of(roi)
    s_ = blk(sub)
    s2 = sub[sub.s2.astype(bool)]
    rho2, p2 = stats.spearmanr(s2.semitones, s2.amp)
    print(f"  {title:18} rho={s_['rho']:+.3f} p={s_['prho']:.2e}  slope={s_['slope']:+.4f} "
          f"p={s_['pslope']:.2e}  S2-only rho={rho2:+.3f} (n={len(s2)}, p={p2:.3f})")
    rows.append(dict(site=title, model="POOLED (7 models)", n=s_["n"], rho=round(s_["rho"], 3),
                     p_rho=round(s_["prho"], 4), slope=round(s_["slope"], 4),
                     p_slope=round(s_["pslope"], 4), s2_only_rho=round(rho2, 3), n_s2=len(s2)))

print("\n=== per-model Spearman (mTRF, n=48 each) ===")
for kind, roi, title in SITES:
    print(f"-- {title}")
    for m in MODEL_ORDER:
        sub = sub_of(roi, m)
        s_ = blk(sub)
        s2 = sub[sub.s2.astype(bool)]
        rho2 = stats.spearmanr(s2.semitones, s2.amp)[0] if len(s2) > 3 else float("nan")
        flag = "  <-- REVERSED (anti-scaling)" if s_["rho"] > 0 and s_["prho"] < 0.05 else ""
        print(f"   {MLABEL[m]:17} rho={s_['rho']:+.3f} p={s_['prho']:.3f}  "
              f"S2-only rho={rho2:+.3f} (n={len(s2)}){flag}")
        rows.append(dict(site=title, model=m, n=s_["n"], rho=round(s_["rho"], 3),
                         p_rho=round(s_["prho"], 4), slope=round(s_["slope"], 4),
                         p_slope=round(s_["pslope"], 4),
                         s2_only_rho=round(rho2, 3) if np.isfinite(rho2) else "", n_s2=len(s2)))
pd.DataFrame(rows).to_csv(f"{OUT}/deviance_scaling_stats_24freq_7models.csv", index=False)

print("\n=== dose-response binned MEDIAN trough (µV) per model × site × deviance size ===")
tab = d.groupby(["roi", "model", "semitones"]).amp.agg(["median", "mean", "count"]).round(3)
print(tab.to_string())
tab.reset_index().to_csv(f"{OUT}/deviance_scaling_binned_24freq_7models.csv", index=False)
print("\nsaved 2 figures + 2 CSVs to", OUT)
