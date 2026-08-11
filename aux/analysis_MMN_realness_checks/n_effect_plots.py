#!/usr/bin/env python
"""Plot 1 -- does the in-silico MMN trough deepen with N (tones between deviants)? FCz only.

The other "realness" dose-response check. In humans, more standards between deviants gives a
deeper MMN; this asks whether the model's trough tracks that, over the same two condition sources
as the deviance analysis, reported side by side.

Site: the FCz electrode, mTRF, S7@0.75-gated troughs, six models (whisper-large excluded).
Sources, sets, gate and style all come from mmn_dose_response_common.py -- see its docstring.

WHY THE COMBINED SET IS WELL JUSTIFIED HERE (unlike the deviance one)
--------------------------------------------------------------------
N is produced by the SAME code path in both sources -- generate_deviant_sequence, the same
1/(N+1) prefix rule, the same 3 x 5 grid -- and both span the identical N in {3,5,7}. So `lit_p2`
is a straightforward n-boost on an identical manipulation, not a between-source contrast, and it
is the headline N result. (The deviance analysis has no such luck: its two sources barely overlap
in x, which is why deviance_scaling_s7gated.py carries an overlap diagnostic and this does not.)
Whether the two sources agree per model is still reported, since that is what licenses reading
lit_p2 as one experiment rather than as a mixture.

THE STATISTIC, AND WHY IT IS NOT COMPUTED ON RAW TRIALS
------------------------------------------------------
Each (model, condition, N) cell holds 5 variations -- the same paradigm re-rolled -- so its trials
are not independent. The PRIMARY rho is therefore computed on one value per
(model, condition, N) cell (`per_method_cells`, the median of that cell's S7-passing troughs).
The raw-trial rho is emitted alongside, labelled `trial`, as the optimistic bound; both go in the
stats CSV with a `unit` column, and the memo quotes the cell-level one.

TWO GATES, RUN SEPARATELY -- and the choice changes the answer
--------------------------------------------------------------
    --gate s7   S2 AND trough <= -0.75 uV. The headline. Outputs keep the bare names.
    --gate s2   shape only, no uV floor. The UNCENSORED companion; every output takes an __s2
                suffix, so the two sets coexist and neither overwrites the other.
The S7 floor removes the shallow tail BY CONSTRUCTION, so it censors exactly the trials a weak
dose-response would move. That is not hypothetical here: the paired N7-N3 shift is -0.085 uV at S2
(p=9e-5) against -0.047 uV at S7 (p=0.004), and the pooled cell-level rho only reaches significance
at S2. Read the two together -- the S7 numbers are a LOWER BOUND, and the gap is the floor's doing.

FIGURES (into plots/, with SVG twins in svgs/; add __s2 for the ungated set)
  1. n_effect_pooled.png            3 panels (lit | lit_p2 | p2), pooled over the 6 models,
                                    mean +- SEM per N, raw uV, shared y-axis.
  2. n_effect_per_model__{set}.png  3 files x 6 panels, shared y within each file.
  3. n_effect_s7_rate.png           3 panels: gate-passing count / TOTAL trials per N.
  + n_effect_stats.csv              Spearman rho per (set x model x unit).
  + n_effect_source_agreement.csv   per-model LIT-vs-NOVEL-P2 sign agreement, which is what
                                    licenses (or refuses) reading lit_p2 as one experiment.
  + n_effect_paired_n3_vs_n7.csv    the paired within-condition test -- the one that separates a
                                    real deepening from a shift in which conditions pass the gate.

Needs the per-trial CSVs from analyze_mmn_per_trial_n.py; load_per_trial() explains how to get
them if they are absent.

Usage:  python n_effect_plots.py [--gate s7|s2]
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy import stats

import mmn_dose_response_common as C

POOLED_INK = "#333333"
XPOS = {n: i for i, n in enumerate(C.N_LEVELS)}   # 3 DISCRETE positions, not a continuous axis


def sfx_for(gate):
    """Output-name suffix for a gate: S7 keeps the bare names, S2 takes __s2."""
    return "" if gate == "s7" else f"__{gate}"


def _n_axis(ax):
    """x = N at three evenly spaced positions, labelled with the rarity each N implies."""
    ax.set_xticks(list(XPOS.values()))
    ax.set_xticklabels([f"{n}\n({C.N_TO_P_DEVIANT[n]:.1%})" for n in C.N_LEVELS])
    ax.set_xlim(-0.45, len(C.N_LEVELS) - 0.55)
    ax.set_xlabel("N standards between deviants\n(oddball probability)")


# ------------------------------------------------------------------------------------------
# Figure 1 -- pooled over the six models
# ------------------------------------------------------------------------------------------
def fig_pooled(tidy, out_png, gate="s7"):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.9))
    lo_all, hi_all = [], []
    for ax, set_key in zip(axes, ("lit", "lit_p2", "p2")):
        frame = C.set_frame(tidy, set_key)
        gat = C.gated(frame, gate)

        groups = ([("lit", gat[gat.dataset == "lit"]), ("p2", gat[gat.dataset == "p2"])]
                  if set_key == "lit_p2" else [(None, gat)])
        for ds, g in groups:
            xs, ys, es, ns = [], [], [], []
            for n in C.N_LEVELS:
                v = g.loc[g["N"] == n, "trough_uv"].to_numpy(float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                xs.append(XPOS[n] + (0.0 if ds is None else (-0.06 if ds == "lit" else 0.06)))
                ys.append(v.mean()); es.append(C.sem(v)); ns.append(v.size)
            if not xs:
                continue
            mk = C.DATASET_MARKER[ds] if ds else "o"
            ax.errorbar(xs, ys, yerr=es, color=POOLED_INK, marker=mk, ms=8, lw=2.0,
                        elinewidth=1.2, capsize=4, mew=1.4,
                        mfc="white" if ds == "p2" else POOLED_INK,
                        label=C.DATASET_LABEL[ds] if ds else None, zorder=3)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 11),
                            ha="center", fontsize=7.5, color="#6b6b6b")
            lo_all += [a - b for a, b in zip(ys, es)]
            hi_all += [a + b for a, b in zip(ys, es)]

        rho, p, n = C.spearman(*_cell_xy(frame, gate))
        ax.set_title(f"{C.SET_LABEL[set_key]}  ({set_key})", fontweight="bold", loc="left", pad=26)
        ax.text(0.0, 1.012, f"{C.gate_label(gate, long=False)} only: {len(gat)}/{len(frame)} trials"
                            f"\nρ={rho:+.2f} (p={p:.2g}, n={n} cells)",
                transform=ax.transAxes, fontsize=7.6, color="#6b6b6b", va="bottom")
        _n_axis(ax)
        if set_key == "lit":
            ax.set_ylabel("MMN trough (µV)\n↑ deeper")
        if set_key == "lit_p2":
            ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
                      fontsize=8.5, loc="upper right")

    lo, hi = min(lo_all), max(hi_all)
    pad = 0.16 * (hi - lo)
    for ax in axes:
        ax.set_ylim(hi + pad, lo - pad)              # INVERTED + SHARED across the three sets
    fig.suptitle(f"MMN trough vs N (tones between deviants) at FCz — pooled over 6 models "
                 f"(mTRF, {C.gate_label(gate, long=False)}-gated)", fontweight="bold", x=0.006, ha="left", y=1.02)
    fig.text(0.006, -0.02, C.wrap(
        f"Mean ± SEM over the {C.gate_label(gate, long=False)}-passing trials of all 6 models; grey numbers = trials per "
        "point. Shared y-axis across the three sets. ρ is Spearman on one value per "
        "(model, condition, N) cell, NOT on raw trials — the 5 variations of a cell are the same "
        "paradigm re-rolled and are not independent. N is confounded with oddball probability by "
        "construction: the generator sets rare-tone probability to 1/(N+1), so N = 3/5/7 means "
        "25%/16.7%/12.5% (shown on the x-axis). A trough that deepens across N is MMN-like but "
        "cannot be attributed to local spacing rather than global rarity.\n"
        f"READ WITH CARE: these means are UNPAIRED, and the gate admits a different mix of "
        f"conditions at each N, so a rising line here can be composition rather than deepening. "
        f"The paired within-condition test (n_effect_paired_n3_vs_n7{sfx_for(gate)}.csv) is what "
        f"settles it; on LIT it is null at both gates despite the rise drawn here."),
        fontsize=7.8, color="#555555", ha="left", va="top")
    return C.finish(fig, out_png)


def _cell_xy(frame, gate="s7"):
    """(N, trough) at the CELL level -- the primary unit for every rho reported here."""
    cells = C.per_method_cells(frame, gate=gate)
    return cells["N"].to_numpy(float), cells["trough_uv"].to_numpy(float)


# ------------------------------------------------------------------------------------------
# Figure 2 -- small multiples, one panel per model
# ------------------------------------------------------------------------------------------
def fig_per_model(tidy, set_key, out_png, gate="s7"):
    frame = C.set_frame(tidy, set_key)
    gat = C.gated(frame, gate)
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0), sharex=True, sharey=True)
    lo_all, hi_all = [], []

    for ax, model in zip(axes.ravel(), C.MODEL_ORDER):
        st = C.style(model)
        sub = gat[gat["model"] == model]
        xs, ys, es, ns = [], [], [], []
        for n in C.N_LEVELS:
            v = sub.loc[sub["N"] == n, "trough_uv"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            xs.append(XPOS[n]); ys.append(v.mean()); es.append(C.sem(v)); ns.append(v.size)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=st["color"], marker=st["marker"], ls=st["ls"],
                        ms=8, lw=2.0, elinewidth=1.2, capsize=4, mec="white", mew=0.8, zorder=3)
            lo_all += [a - b for a, b in zip(ys, es)]
            hi_all += [a + b for a, b in zip(ys, es)]
        rho, p, n = C.spearman(*_cell_xy(frame[frame["model"] == model], gate))
        ax.set_title(model, fontweight="bold", loc="left", fontsize=10)
        ax.text(0.03, 0.955, f"ρ={rho:+.2f}  p={p:.3g}\nn={n} cells", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.2, color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        _n_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("MMN trough (µV)\n↑ deeper")
    for ax in axes[0, :]:
        ax.set_xlabel("")
    lo, hi = min(lo_all), max(hi_all)
    pad = 0.18 * (hi - lo)
    axes[0, 0].set_ylim(hi + pad, lo - pad)          # INVERTED, shared across all six panels

    fig.suptitle(f"MMN trough vs N at FCz — per model — {C.SET_LABEL[set_key]} ({set_key})",
                 fontweight="bold", x=0.006, ha="left", y=1.015)
    fig.text(0.006, -0.02, C.wrap(
        f"Mean ± SEM of the {C.gate_label(gate, long=False)}-passing trials. Shared y-axis across all six panels: with "
        "whisper-large excluded the models span a ~2× µV range rather than ~35×, so one scale is "
        "readable for all of them. ρ is Spearman at the (model, condition, N) cell level."),
        fontsize=7.8, color="#555555", ha="left", va="top")
    return C.finish(fig, out_png)


# ------------------------------------------------------------------------------------------
# Figure 3 -- the UNCENSORED view: S7@0.75 rate per N
# ------------------------------------------------------------------------------------------
def fig_rate(tidy, out_png, gate="s7"):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.9), sharey=True)
    for ax, set_key in zip(axes, ("lit", "lit_p2", "p2")):
        frame = C.set_frame(tidy, set_key)
        for model in C.MODEL_ORDER:
            st = C.style(model)
            r = C.pass_rate(frame[frame["model"] == model], ["N"], gate).sort_values("N")
            ax.plot([XPOS[n] for n in r["N"]], r["rate"], color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=1.3, ms=6, alpha=0.65, mec="white", mew=0.6, zorder=3)
        pooled = C.pass_rate(frame, ["N"], gate).sort_values("N")
        ax.plot([XPOS[n] for n in pooled["N"]], pooled["rate"], color=POOLED_INK, marker="o",
                ls="-", lw=2.8, ms=8, mec="white", mew=1.0, zorder=5)
        tot = pooled["n_total"]
        denom = (f"{int(tot.min())} trials per point" if tot.min() == tot.max()
                 else f"{int(tot.min())}–{int(tot.max())} trials per point")
        rho, p, _ = C.spearman(frame["N"].to_numpy(float),
                               frame[C.GATES[gate][0]].astype(float).to_numpy())
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{C.SET_LABEL[set_key]}  ({set_key})", fontweight="bold", loc="left", pad=26)
        ax.text(0.0, 1.012, f"pooled rate {frame[C.GATES[gate][0]].mean():.3f} · ρ={rho:+.3f} (p={p:.2g})"
                            f"\n{denom}", transform=ax.transAxes, fontsize=7.6,
                color="#6b6b6b", va="bottom")
        _n_axis(ax)
        ax.set_ylabel(f"{C.gate_label(gate, long=False)} rate  (pass / all trials)" if set_key == "lit" else "")

    handles = [Line2D([0], [0], color=C.style(m)["color"], marker=C.style(m)["marker"],
                      ls=C.style(m)["ls"], lw=1.3, ms=6, mec="white", label=m)
               for m in C.MODEL_ORDER]
    handles.append(Line2D([0], [0], color=POOLED_INK, marker="o", ls="-", lw=2.8, ms=8,
                          mec="white", label="pooled (6 models)"))
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=8.5,
               ncol=4, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(f"{C.gate_label(gate, long=False)} rate vs N at FCz — denominator is ALL trials",
                 fontweight="bold", x=0.006, ha="left", y=1.20)
    fig.text(0.006, -0.02, C.wrap(
        "The only UNCENSORED view of the N-effect: a count outcome is not floored by the "
        "−0.75 µV gate, so a dose-response can appear here that the gated amplitude axis cannot "
        "show. A monotone rate beside a flat gated trough is a real positive result, not a null."),
        fontsize=7.8, color="#555555", ha="left", va="top")
    return C.finish(fig, out_png)


# ------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------
def stats_table(tidy, gate="s7"):
    rows = []

    def add(set_key, model, unit, frame):
        if unit == "cell":
            x, y = _cell_xy(frame, gate)
        else:
            g = C.gated(frame, gate)
            x, y = g["N"].to_numpy(float), g["trough_uv"].to_numpy(float)
        rho, p, n = C.spearman(x, y)
        rows.append(dict(set=set_key, model=model, unit=unit, rho=rho, p=p, n=n,
                         n_pass=int(frame[C.GATES[gate][0]].sum()), n_trials=len(frame),
                         pass_rate=float(frame[C.GATES[gate][0]].mean()) if len(frame)
                         else float("nan"),
                         median_trough_uv=float(np.median(y)) if len(y) else float("nan")))

    for set_key in ("lit", "lit_p2", "p2"):
        f = C.set_frame(tidy, set_key)
        for unit in ("cell", "trial"):
            add(set_key, "POOLED", unit, f)
            for m in C.MODEL_ORDER:
                add(set_key, m, unit, f[f["model"] == m])
    return pd.DataFrame(rows)


def paired_n3_vs_n7(tidy, gate="s7"):
    """Within-(model, condition) paired comparison of the N=3 and N=7 gated troughs.

    THE test for an amplitude N-effect, and it disagrees with the unpaired means the pooled figure
    draws. Pairing makes each condition its own control, which removes the between-condition
    variance that dominates the trough distribution AND -- the reason it matters here -- removes
    the composition confound: the S7 gate admits a DIFFERENT MIX of conditions at each N, so an
    unpaired mean can shift across N without any single condition deepening. On LIT it does
    exactly that (unpaired p=0.003, paired p=0.23 with 50% of pairs deeper, i.e. a coin flip).
    """
    rows = []
    for set_key in ("lit", "lit_p2", "p2"):
        cells = C.per_method_cells(C.set_frame(tidy, set_key), gate=gate)
        wide = cells.pivot_table(index=["model", "method"], columns="N", values="trough_uv")
        for model in ["POOLED"] + C.MODEL_ORDER:
            w = wide if model == "POOLED" else wide[
                wide.index.get_level_values("model") == model]
            w = w.dropna(subset=[3, 7])
            if len(w) < 10:
                continue
            delta = (w[7] - w[3])                      # negative = deeper at N=7
            p = stats.wilcoxon(w[7], w[3]).pvalue
            rows.append(dict(set=set_key, model=model, n_pairs=len(w),
                             mean_n3=w[3].mean(), mean_n7=w[7].mean(),
                             delta_uv=delta.mean(), wilcoxon_p=p,
                             frac_pairs_deeper_at_n7=float((delta < 0).mean())))
    return pd.DataFrame(rows)


def source_agreement(st):
    """Per model: do LIT and NOVEL-P2 give the same rho sign? This is what licenses lit_p2.

    N is the identical manipulation in both sources, so a shared sign means the combined set is
    one experiment with more n. Opposite signs would make lit_p2 a mixture.
    """
    rows = []
    for m in ["POOLED"] + C.MODEL_ORDER:
        rl = st[(st["set"] == "lit") & (st.model == m) & (st.unit == "cell")].iloc[0]
        rp = st[(st["set"] == "p2") & (st.model == m) & (st.unit == "cell")].iloc[0]
        rc = st[(st["set"] == "lit_p2") & (st.model == m) & (st.unit == "cell")].iloc[0]
        rows.append(dict(model=m, rho_lit=rl.rho, p_lit=rl.p, rho_p2=rp.rho, p_p2=rp.p,
                         rho_lit_p2=rc.rho, p_lit_p2=rc.p,
                         same_sign=bool(np.sign(rl.rho) == np.sign(rp.rho))))
    return pd.DataFrame(rows)


def print_summary(tidy, st, agree, paired, gate="s7"):
    print("\n" + "=" * 92)
    print(f"Spearman rho: trough_uv vs N, FCz, mTRF, {C.gate_label(gate)} "
          "(NEGATIVE rho = deeper with more standards = the MMN-like direction)")
    print("=" * 92)
    for unit in ("cell", "trial"):
        lab = ("one value per (model, condition, N) cell -- PRIMARY" if unit == "cell"
               else "raw trials, not independent -- optimistic bound")
        print(f"\n-- unit: {unit} ({lab}) --")
        print(f"  {'model':<17}" + "".join(f"{k:>24}" for k in ("lit", "lit_p2", "p2")))
        for m in ["POOLED"] + C.MODEL_ORDER:
            cells = []
            for k in ("lit", "lit_p2", "p2"):
                r = st[(st["set"] == k) & (st.model == m) & (st.unit == unit)].iloc[0]
                cells.append(f"{r.rho:+.2f} (p={r.p:.3g}, n={int(r.n)})")
            print(f"  {m:<17}" + "".join(f"{c:>24}" for c in cells))

    print("\n" + "=" * 92)
    print("Does LIT agree with NOVEL-P2 on the N-effect? (both span the identical N in {3,5,7})")
    print("=" * 92)
    print(f"  {'model':<17}{'rho lit':>10}{'rho p2':>10}{'same sign':>12}{'rho lit_p2':>13}")
    for _, r in agree.iterrows():
        print(f"  {r.model:<17}{r.rho_lit:>+10.2f}{r.rho_p2:>+10.2f}"
              f"{('yes' if r.same_sign else 'NO'):>12}{r.rho_lit_p2:>+13.2f}")
    n_same = int(agree[agree.model != "POOLED"]["same_sign"].sum())
    print(f"  -> {n_same}/{len(C.MODEL_ORDER)} models agree on sign. lit_p2 reads as "
          f"{'one experiment with more n' if n_same >= 5 else 'a MIXTURE -- report per source'}.")

    print("\n" + "=" * 92)
    print("PAIRED N=3 vs N=7 within (model, condition) -- the amplitude test that controls for")
    print("which conditions the S7 gate admits at each N (the pooled figure's means do not)")
    print("=" * 92)
    print(f"  {'set':<9}{'model':<17}{'pairs':>7}{'delta uV':>10}{'p':>11}{'% deeper':>10}")
    for _, r in paired[paired.model == "POOLED"].iterrows():
        print(f"  {r['set']:<9}{r.model:<17}{int(r.n_pairs):>7}{r.delta_uv:>+10.3f}"
              f"{r.wilcoxon_p:>11.3g}{100 * r.frac_pairs_deeper_at_n7:>9.0f}%")

    print("\n" + "=" * 92)
    print(f"{C.gate_label(gate, long=False)} rate per N (count / ALL trials)")
    print("=" * 92)
    for k in ("lit", "lit_p2", "p2"):
        f = C.set_frame(tidy, k)
        r = C.pass_rate(f, ["N"], gate).sort_values("N")
        cells = "  ".join(f"N={int(n)}: {rate:.3f} ({s7}/{tot})"
                          for n, rate, s7, tot in zip(r.N, r.rate, r.n_s7, r.n_total))
        print(f"  {k:<8} {cells}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lit_csv", default=str(C.LIT_PER_TRIAL))
    p.add_argument("--p2_csv", default=str(C.P2_PER_TRIAL))
    p.add_argument("--out_dir", default=str(C.PLOTS_DIR),
                   help="where the PNGs go; SVGs go to the sibling svgs/")
    p.add_argument("--csv_dir", default=str(C.ANALYSIS_DIR),
                   help="where the stats CSVs go (data, not figures)")
    p.add_argument("--gate", default="s7", choices=("s7", "s2"),
                   help="amplitude gate. s7 = shape + 0.75 uV floor (the headline). s2 = shape "
                        "only, no floor -- the UNCENSORED companion, whose outputs all take an "
                        "__s2 suffix so the two sets never overwrite each other.")
    p.add_argument("--no_verify", action="store_true",
                   help="skip the committed row-count assertions (validation runs only)")
    args = p.parse_args()
    gate, sfx = args.gate, sfx_for(args.gate)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tidy = C.load_per_trial(Path(args.lit_csv), Path(args.p2_csv), verify=not args.no_verify)
    print(f"Loaded {len(tidy)} per-trial rows over {len(C.MODEL_ORDER)} models "
          f"({tidy[tidy.dataset == 'lit'].method.nunique()} LIT + "
          f"{tidy[tidy.dataset == 'p2'].method.nunique()} NOVEL-P2 conditions x 15 trials)")

    print("\nFigures:")
    fig_pooled(tidy, out / f"n_effect_pooled{sfx}.png", gate)
    for set_key in ("lit", "lit_p2", "p2"):
        fig_per_model(tidy, set_key, out / f"n_effect_per_model__{set_key}{sfx}.png", gate)
    fig_rate(tidy, out / f"n_effect_s7_rate{sfx}.png", gate)

    st = stats_table(tidy, gate)
    agree = source_agreement(st)
    paired = paired_n3_vs_n7(tidy, gate)
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    st_path = csv_dir / f"n_effect_stats{sfx}.csv"
    st.to_csv(st_path, index=False, float_format="%.6g")
    agree.to_csv(csv_dir / f"n_effect_source_agreement{sfx}.csv", index=False, float_format="%.6g")
    paired.to_csv(csv_dir / f"n_effect_paired_n3_vs_n7{sfx}.csv", index=False, float_format="%.6g")
    print(f"  wrote {st_path}  ({len(st)} rows)")
    print(f"  wrote {csv_dir / f'n_effect_source_agreement{sfx}.csv'}")
    print(f"  wrote {csv_dir / f'n_effect_paired_n3_vs_n7{sfx}.csv'}")
    print_summary(tidy, st, agree, paired, gate)


if __name__ == "__main__":
    main()
