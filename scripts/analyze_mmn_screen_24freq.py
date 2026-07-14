#!/usr/bin/env python
"""24 frequency methods x 7 models MMN screen -- tidy CSV, figures, and the two decisions.

Consumes the long-format `mmn_s7_roi.csv` written by analyze_mmn_s7_roi.py (one row per
model x mapping x method x roi x X) and produces the deliverables for the metric-selection and
model-qualification decisions. mTRF only (the encoder is deferred). Two reporting sites, each a
single target (no ROI averaging):
    frontal parcel   = roi_kind == "parcel"    & roi == "frontal"
    FCz electrode    = roi_kind == "electrode" & roi == "FCz"

For each (model, site) there are 48 conditions = 24 Frequency methods x {regular, counter}.
Present-count denominator is #/48. S2 is the X-independent shape verdict; S7@X = S2 AND
(trough_uv <= -X uV); by construction S7 <= S2 in every cell. `trough_uv` (signed uV, negative =
deeper MMN) is the amplitude S7 gates on and the deviance-scaling response variable.

Outputs (into --out_dir, default outputs/results_24freq_7models/):
  * mmn_screen_24freq.csv        -- tidy: model, site, method_id, direction, S2, S7@{0.5,0.75,1.0,1.5}, trough_uv
  * summary_counts_by_site.csv   -- #/48 per model at S2 and each S7/X, per site
  * per_model_spearman.csv       -- deviance-scaling Spearman rho per model x site
  * plots/x_vs_count_by_model__{site}.png   (Fig 1: 7 lines, S2 at X=0)
  * plots/x_vs_count_pooled__{site}.png     (Fig 2: pooled mean+-spread vs S2)
  * plots/deviance_scaling__mtrf_7models.png (Fig 3: trough_uv vs semitones, both sites)
plus console summary tables (ranking stability across X; frontal<->FCz concordance).

READ-ONLY over the input CSV + metadata; writes only under --out_dir. Style (Okabe-Ito CVD-safe
palette + marker/linestyle secondary encoding) mirrors
aux/analysis_with_counter/plots/deviance_scaling_plots.py.

Usage:
    python scripts/analyze_mmn_screen_24freq.py \
        --s7_csv outputs/results_24freq_7models/mmn_s7_roi.csv \
        --out_dir outputs/results_24freq_7models
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- the amplitude floors this screen decides among (subset of UV_SWEEP) ----
X_CHOICES = (0.5, 0.75, 1.0, 1.5)

# ---- canonical model order (small -> large within family) + Okabe-Ito style ----
MODEL_ORDER = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
               "whisper-large", "wav2vec2-medium", "wav2vec2-large"]
MODEL_STYLE = {
    "whisper-tiny":    dict(color="#56B4E9", marker="o", ls="-"),   # sky blue
    "whisper-base":    dict(color="#0072B2", marker="s", ls="-"),   # blue
    "whisper-small":   dict(color="#009E73", marker="^", ls="-"),   # bluish green
    "whisper-medium":  dict(color="#E69F00", marker="D", ls="-"),   # orange
    "whisper-large":   dict(color="#D55E00", marker="v", ls="-"),   # vermillion
    "wav2vec2-medium": dict(color="#CC79A7", marker="P", ls="--"),  # reddish purple
    "wav2vec2-large":  dict(color="#000000", marker="X", ls="--"),  # black
}

# ---- the two reporting sites: (roi_kind, roi, site key, pretty label) ----
SITES = [("parcel", "frontal", "frontal_parcel", "frontal parcel"),
         ("electrode", "FCz", "FCz_electrode", "FCz electrode")]

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


def _fmt_x(x):
    """0.5 -> '0.5', 1.0 -> '1.0' for stable column names S7@0.5 ... S7@1.5."""
    return f"{x:g}"


def load_semitones(metadata_csv):
    """method_id -> deviance size in semitones = 12*|log2(f_dev/f_std)| (symmetric under counter)."""
    md = pd.read_csv(metadata_csv)
    md = md[md["change_type"].astype(str).str.strip().str.lower() == "frequency"].copy()
    md["semitones"] = 12.0 * np.abs(np.log2(md["deviant_freq"] / md["standard_freq"]))
    return dict(zip(md["method_id"].astype(int), md["semitones"]))


def load_screen(s7_csv, semitones):
    """Long mTRF rows for the two sites, annotated with site/method_id/direction/semitones."""
    df = pd.read_csv(s7_csv)
    df = df[df["mapping"] == "mtrf"].copy()
    df["dip_uv_threshold"] = df["dip_uv_threshold"].round(3)

    frames = []
    for kind, roi, site_key, _ in SITES:
        sub = df[(df["roi_kind"] == kind) & (df["roi"] == roi)].copy()
        sub["site"] = site_key
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)

    long["method_id"] = long["method"].str.extract(r"method_(\d+)").astype(int)
    long["direction"] = np.where(long["method"].str.endswith("_counter"), "counter", "regular")
    long["semitones"] = long["method_id"].map(semitones)
    return long


def build_tidy(long, expected_conditions):
    """One row per (model, site, method_id, direction): S2, S7@X..., trough_uv. Verified invariants."""
    keys = ["model", "site", "method_id", "direction"]
    tidy = long.drop_duplicates(keys)[keys + ["s2", "trough_uv", "semitones"]].copy()
    tidy = tidy.rename(columns={"s2": "S2"})
    for x in X_CHOICES:
        sx = long[np.isclose(long["dip_uv_threshold"], x)][keys + ["s7"]]
        sx = sx.rename(columns={"s7": f"S7@{_fmt_x(x)}"})
        tidy = tidy.merge(sx, on=keys, how="left")
    tidy = tidy.sort_values(["site", "model", "method_id", "direction"]).reset_index(drop=True)

    # invariant 1: S7 <= S2 in every cell
    for x in X_CHOICES:
        col = f"S7@{_fmt_x(x)}"
        viol = int((tidy[col].astype(bool) & ~tidy["S2"].astype(bool)).sum())
        assert viol == 0, f"{viol} rows violate S7@{_fmt_x(x)} <= S2"

    # invariant 2: exactly `expected_conditions` per (model, site)
    counts = tidy.groupby(["model", "site"]).size()
    bad = counts[counts != expected_conditions]
    if len(bad):
        print(f"  WARN: not all (model, site) cells have {expected_conditions} conditions:")
        print(bad.to_string())
    else:
        print(f"  OK: every (model, site) cell has exactly {expected_conditions} conditions.")
    return tidy


def counts_by_site(tidy):
    """#/48 present-count per (model, site) at S2 and each S7/X. Long -> wide summary DataFrame."""
    cols = ["S2"] + [f"S7@{_fmt_x(x)}" for x in X_CHOICES]
    g = tidy.groupby(["site", "model"])[cols].sum().astype(int).reset_index()
    return g


def order_models(present):
    """Present models in canonical order, unknown appended alphabetically."""
    known = [m for m in MODEL_ORDER if m in present]
    return known + sorted(m for m in present if m not in known)


# ------------------------------- figures -------------------------------------
def _count_series(tidy, site, model):
    """[S2, S7@0.5, S7@0.75, S7@1.0, S7@1.5] present-counts for one (model, site)."""
    sub = tidy[(tidy["site"] == site) & (tidy["model"] == model)]
    return [int(sub["S2"].sum())] + [int(sub[f"S7@{_fmt_x(x)}"].sum()) for x in X_CHOICES]


def fig_x_vs_count_by_model(tidy, site_key, pretty, denom, out_png):
    """Fig 1: X vs #/denom, one line per model, with S2 as the X=0 (shape-only) reference."""
    models = order_models(tidy[tidy.site == site_key]["model"].unique().tolist())
    xpos = [0.0] + list(X_CHOICES)                     # x=0 carries the S2 count
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for m in models:
        st = MODEL_STYLE.get(m, dict(color="#666", marker="o", ls="-"))
        y = _count_series(tidy, site_key, m)
        ax.plot(xpos, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8,
                ms=7, mec="white", mew=0.7, label=m, zorder=3)
    ax.axvline(0.0, color="#bbb", lw=1, ls=":", zorder=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["S2\n(X→0)"] + [f"{x:g}" for x in X_CHOICES])
    ax.set_xlabel("S7 amplitude floor  X  (µV)")
    ax.set_ylabel(f"MMN-present count  (#/{denom})")
    ax.set_ylim(0, denom * 1.02)
    ax.set_title(f"S7 present-count vs amplitude floor — {pretty}", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def fig_x_vs_count_pooled(tidy, site_key, pretty, denom, out_png):
    """Fig 2: pooled-across-models mean #/denom vs X (min-max band), with the S2 mean reference."""
    models = order_models(tidy[tidy.site == site_key]["model"].unique().tolist())
    xpos = [0.0] + list(X_CHOICES)
    mat = np.array([_count_series(tidy, site_key, m) for m in models], dtype=float)  # [n_model, 5]
    mean, lo, hi = mat.mean(0), mat.min(0), mat.max(0)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.fill_between(xpos, lo, hi, color="#0072B2", alpha=0.15, zorder=1,
                    label="model min–max")
    ax.plot(xpos, mean, color="#0072B2", marker="o", lw=2.2, ms=8, mec="white", mew=0.8,
            zorder=3, label="mean over models")
    ax.axhline(mean[0], color="#9a9a9a", lw=1.2, ls="--", zorder=2,
               label=f"S2 mean = {mean[0]:.1f}")
    ax.axvline(0.0, color="#bbb", lw=1, ls=":", zorder=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["S2\n(X→0)"] + [f"{x:g}" for x in X_CHOICES])
    ax.set_xlabel("S7 amplitude floor  X  (µV)")
    ax.set_ylabel(f"MMN-present count  (#/{denom})")
    ax.set_ylim(0, denom * 1.02)
    ax.set_title(f"S7 present-count vs X, pooled over {len(models)} models — {pretty}",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan"), float("nan")
    return stats.spearmanr(x[ok], y[ok])


def fig_deviance_scaling(tidy, out_png):
    """Fig 3: trough_uv (signed µV) vs deviance (semitones), per model, both sites. Spearman rho."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharey=True)
    YCLIP = (-5.5, 1.5)
    xs = np.linspace(0, 12.5, 50)
    for ax, (_, _, site_key, pretty) in zip(axes, SITES):
        site = tidy[tidy.site == site_key]
        models = order_models(site["model"].unique().tolist())
        for m in models:
            st = MODEL_STYLE.get(m, dict(color="#666", marker="o", ls="-"))
            sub = site[site.model == m]
            x, y = sub["semitones"].to_numpy(float), sub["trough_uv"].to_numpy(float)
            ax.scatter(x + np.random.default_rng(0).uniform(-0.1, 0.1, len(x)), y, s=16,
                       color=st["color"], alpha=0.45, edgecolors="white", linewidths=0.3, zorder=3)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3:
                sl, ic = np.polyfit(x[ok], y[ok], 1)
                ax.plot(xs, ic + sl * xs, color=st["color"], ls=st["ls"], lw=1.3, alpha=0.9, zorder=2)
        rho, p = _spearman(site["semitones"].to_numpy(float), site["trough_uv"].to_numpy(float))
        ax.axhline(0, color="#9a9a9a", lw=1, ls=":", zorder=1)
        ax.set_xlabel("Deviance size  (semitones)")
        ax.set_title(pretty, fontweight="bold", loc="left")
        ax.annotate(f"pooled ρ = {rho:+.2f}\n(p = {p:.3f})", xy=(0.96, 0.05),
                    xycoords="axes fraction", ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc"))
    axes[0].set_ylabel("S2/S7 trough  (µV;  negative = deeper MMN)")
    axes[0].set_ylim(*YCLIP)
    handles = [Line2D([0], [0], color=MODEL_STYLE.get(m, {}).get("color", "#666"),
                      marker=MODEL_STYLE.get(m, {}).get("marker", "o"), ls="",
                      ms=7, mec="white", label=m)
               for m in order_models(tidy["model"].unique().tolist())]
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=8.5,
               ncol=len(handles), bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Deviance-scaling of the in-silico MMN trough (µV), mTRF — by site",
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.text(0.5, -0.03, "trough_uv = deviant−standard µV at the S2 trough latency (negative = deeper); "
             "points jittered on x; thin line = per-model OLS. Deviance = 12·|log₂(f_dev/f_std)| "
             f"(symmetric under counterbalancing). y clipped to [{YCLIP[0]}, {YCLIP[1]}] µV.",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def per_model_spearman(tidy):
    """Deviance-scaling Spearman rho per (model, site), on all 48 and on the S2-passing subset."""
    rows = []
    for _, _, site_key, pretty in SITES:
        site = tidy[tidy.site == site_key]
        for m in order_models(site["model"].unique().tolist()):
            sub = site[site.model == m]
            rho, p = _spearman(sub["semitones"].to_numpy(float), sub["trough_uv"].to_numpy(float))
            s2 = sub[sub["S2"].astype(bool)]
            rho2, _ = _spearman(s2["semitones"].to_numpy(float), s2["trough_uv"].to_numpy(float))
            rows.append(dict(site=site_key, model=m, n=len(sub), rho=round(rho, 3),
                             p=round(p, 4), n_s2=len(s2), rho_s2_only=round(rho2, 3)))
    return pd.DataFrame(rows)


# ------------------------------- console tables -------------------------------
def print_ranking_stability(counts, denom):
    """Does the model ordering by #/denom stay stable across S2 and each S7/X? (rank-corr vs S2)."""
    print("\n" + "=" * 78)
    print(f"Model ranking stability across X (Spearman rank-corr of #/{denom} vs the S2 ranking)")
    print("=" * 78)
    cols = ["S2"] + [f"S7@{_fmt_x(x)}" for x in X_CHOICES]
    for site_key in counts["site"].unique():
        c = counts[counts.site == site_key].set_index("model")[cols]
        c = c.loc[order_models(c.index.tolist())]
        print(f"\n-- {site_key} --")
        print("  ranking by S2 (high→low): " + ", ".join(c["S2"].sort_values(ascending=False).index))
        base = c["S2"].rank(ascending=False)
        row = "  rank-corr vs S2:  "
        for col in cols[1:]:
            tau = stats.spearmanr(base, c[col].rank(ascending=False))[0]
            row += f"{col}={tau:+.2f}  "
        print(row)


def print_concordance(tidy, ref_x):
    """frontal-parcel <-> FCz-electrode agreement on S2 and S7@ref_x (per model x method x direction)."""
    ref_col = f"S7@{_fmt_x(ref_x)}"
    print("\n" + "=" * 78)
    print(f"Frontal-parcel <-> FCz-electrode concordance (% agree; per condition; ref X = {ref_x:g} µV)")
    print("=" * 78)
    keys = ["model", "method_id", "direction"]
    wide = tidy.pivot_table(index=keys, columns="site", values=["S2", ref_col], aggfunc="first")
    for crit in ["S2", ref_col]:
        if ("frontal_parcel" in wide[crit].columns) and ("FCz_electrode" in wide[crit].columns):
            a = wide[crit]["frontal_parcel"].astype(bool)
            b = wide[crit]["FCz_electrode"].astype(bool)
            agree = float((a == b).mean()) * 100
            both = int((a & b).sum())
            print(f"  {crit:6}  agree={agree:5.1f}%   both-present={both:3d}   "
                  f"frontal-only={int((a & ~b).sum()):3d}   FCz-only={int((~a & b).sum()):3d}")


def print_summary_counts(counts, denom):
    print("\n" + "=" * 78)
    print(f"Present-count #/{denom} per model, per site (S2 and each S7/X)")
    print("=" * 78)
    cols = ["S2"] + [f"S7@{_fmt_x(x)}" for x in X_CHOICES]
    for site_key in counts["site"].unique():
        c = counts[counts.site == site_key].set_index("model")[cols]
        c = c.loc[order_models(c.index.tolist())]
        print(f"\n-- {site_key} (#/{denom}) --")
        print(c.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s7_csv", default="outputs/results_24freq_7models/mmn_s7_roi.csv")
    p.add_argument("--metadata_csv",
                   default="data/metadata/literature_frequency_intensity_duration_metadata.csv")
    p.add_argument("--out_dir", default="outputs/results_24freq_7models")
    p.add_argument("--expected_conditions", type=int, default=48,
                   help="conditions per (model, site); 48 for the full 24-method x 2-direction screen "
                        "(use 20 for the whisper x 10-method smoke test).")
    p.add_argument("--ref_x", type=float, default=0.5, choices=list(X_CHOICES),
                   help="headline S7 amplitude floor X (uV) used for the concordance table. "
                        "Default 0.5 uV is PROVISIONAL; all of {0.5, 0.75, 1.0, 1.5} are reported.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    denom = args.expected_conditions

    semitones = load_semitones(args.metadata_csv)
    long = load_screen(args.s7_csv, semitones)
    if long.empty:
        print(f"No mTRF rows for the two sites in {args.s7_csv} -- nothing to do.")
        return

    tidy = build_tidy(long, args.expected_conditions)
    tidy_out = out_dir / "mmn_screen_24freq.csv"
    tidy.drop(columns=["semitones"]).to_csv(tidy_out, index=False)
    print(f"Wrote {len(tidy)} rows -> {tidy_out}")

    counts = counts_by_site(tidy)
    counts.to_csv(out_dir / "summary_counts_by_site.csv", index=False)
    sp = per_model_spearman(tidy)
    sp.to_csv(out_dir / "per_model_spearman.csv", index=False)

    for _, _, site_key, pretty in SITES:
        if (tidy.site == site_key).any():
            fig_x_vs_count_by_model(tidy, site_key, pretty, denom,
                                    plots_dir / f"x_vs_count_by_model__{site_key}.png")
            fig_x_vs_count_pooled(tidy, site_key, pretty, denom,
                                  plots_dir / f"x_vs_count_pooled__{site_key}.png")
    fig_deviance_scaling(tidy, plots_dir / "deviance_scaling__mtrf_7models.png")

    print_summary_counts(counts, denom)
    print_ranking_stability(counts, denom)
    print_concordance(tidy, args.ref_x)
    print("\n=== per-model deviance-scaling Spearman rho ===")
    print(sp.to_string(index=False))
    print(f"\nFigures + CSVs saved under {out_dir}")


if __name__ == "__main__":
    main()
