"""Section 0 of the novel-search memo: the 24-method literature screen on its own terms.

Sections 1-7 use the literature set only as a COMPARISON arm -- a tier distribution and a top 10
held up against the novel grid. This script reports the same set the way Sections 2-5 report the
novel grid: its own agreement tiers, its own consensus-yield heatmap, its full ranking, and the
predicted waveforms behind its top 10.

Nothing here re-scores anything. Every number comes from the same (roi=FCz, mapping=mtrf, X=0.75)
slice `novel_search_common.load_scored` defines, over the same six SEARCH_MODELS -- whisper-large
is dropped on load, so the counts here are commensurable with every Phase-1 and Phase-2 number in
the memo. 24 methods x {regular, counter} = 48 direction-instances.

Outputs (all into --out_dir, with an SVG twin of every figure in ../svgs/):

  literature_agreement_tiers.csv     Table A   n_agree x stimulus counts at X = 0.75
  literature_consensus_heatmap.png   Figure A  top-X rank vs n_agree, as for Phases 1 and 2
    + literature_consensus_heatmap.csv
  literature_top48.csv               Table B   all 48 instances ranked, Table 10's columns
  literature_waveforms.png           Figure B  FCz difference waves, top 10 instances

Usage:
  python aux/analysis_novel_search/plots/literature_results.py
  python aux/analysis_novel_search/plots/literature_results.py --n_wave 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

from novel_search_common import (                                      # noqa: E402
    SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD, load_scored,
)
# The literature loader, the two tables and the heatmap are the SAME functions Sections 2-5 use.
# Importing them rather than restating them is what makes "scored identically" true of the
# reporting as well as of the scoring: if the novel tables change shape, these follow.
from phase1_results import (                                           # noqa: E402
    MMN_WINDOW, WAVE_MODEL_STYLE, WAVE_LW, WAVE_ALPHA,
    load_literature, table_agreement_tiers, table_topn, plot_consensus_heatmap,
    load_fcz_waves,
)
from novel_search_plots import MLABEL, MUTED, configure_svg_output, savefig_both  # noqa: E402

OUT = HERE
# Not phase1_results.LITERATURE: that module's default is the superseded screen, kept so an old
# run reproduces. Section 3 and this section are both defined on the current one.
LITERATURE = REPO / "outputs/results_soafix/mmn_s7_roi.csv"
# The corrected literature predictions, whose epochs carry the full post-tone readout.
PREDICTIONS = REPO / "outputs/insilico_mmn_predictions_soafix"
# The literature screen averages 15 deviant realizations per condition. Asserted against each
# HDF5 before a trace is drawn, so a 1-deviant file left in this root cannot be rendered here.
N_DEVIANTS = 15

# Top-X ladder for the consensus heatmap. Phases 1 and 2 run to 300 and 250 over 1806 and 254
# instances; 48 instances need their own ladder, and it ends AT 48 because a top-48 cut of a
# 48-instance set is every instance -- the right-hand column is the degenerate case by
# construction, and is kept so the ramp has its ceiling visible.
LIT_TOP_X = (2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 48)

# The MMN criteria read out to 360 ms past the final tone's onset (S2's recovery search can sit as
# late as 240 ms and looks 6 samples beyond). Drawing to exactly that bound shows the whole span
# the verdict was computed over and no more.
WAVE_WINDOW = (-120.0, 360.0)
N_WAVE = 10
PANELS = (2, 5)


def load_literature_ranked(lit_csv=LITERATURE, models=None):
    """The 48 literature direction-instances, ranked, carrying BOTH S7 and S2 per model.

    `load_literature` gives the ranking and the frequencies but only S7; `consensus_grid` needs
    S2, because a model's own top-X ordering is taken over its S2 responses (a top-X led by the
    deepest excursion anywhere in the epoch would not be an MMN ordering). Re-reading the scored
    slice to attach S2 mirrors what `load_phase` does for the novel grid.
    """
    models = list(models or SEARCH_MODELS)
    lit = load_literature(models, lit_csv)
    if lit is None:
        raise SystemExit(f"no literature screen at {lit_csv}")

    scored = load_scored(lit_csv, models=models, roi=ROI, mapping=MAPPING, x=DIP_UV_THRESHOLD)
    s2 = (scored.pivot_table(index=["pair_id", "direction"], columns="model", values="s2",
                             aggfunc="first")
          .reindex(columns=models).astype("boolean").fillna(False).astype(bool).reset_index())
    lit = lit.merge(s2.rename(columns={m: f"s2__{m}" for m in models}),
                    on=["pair_id", "direction"], how="left")
    for m in models:
        assert not (lit[f"s7__{m}"] & ~lit[f"s2__{m}"]).any(), f"S7 without S2 for {m}"

    n_pairs = lit["pair_id"].nunique()
    assert len(lit) == 2 * n_pairs, f"{len(lit)} instances over {n_pairs} pairs"
    print(f"  X = {DIP_UV_THRESHOLD:.2f}: {len(lit)} direction-instances over {n_pairs} "
          f"literature methods, {len(models)} models")
    return lit


def plot_literature_waveforms(ranked, models, out_dir, n=N_WAVE,
                              predictions_root=PREDICTIONS, prefix="literature",
                              expect_n_deviants=N_DEVIANTS):
    """FCz µV difference wave for the top `n` direction-INSTANCES of the literature ranking.

    Instances, not pairs. The novel figure draws the regular direction of its top pairs because
    the novel ranking is led by regular instances; the literature ranking is not -- five of its
    top ten are `_counter`, and drawing method_19's regular trace under a panel titled by
    method_19_counter's rank would show a trace the ranking never scored.

    Each panel autoscales: the six models' predicted µV scales differ by ~5x (Caveat 2), so a
    shared axis is set by wav2vec2-large and flattens the whisper traces into a line at zero.
    Read shape, never one trace's depth against another's.
    """
    top = ranked.head(n)
    if top.empty:
        print("  skipped waveforms: ranking is empty")
        return None
    waves = load_fcz_waves(top["pair_id"].unique().tolist(), models, predictions_root,
                           expect_n_deviants=expect_n_deviants)
    if not waves:
        print("  skipped waveforms: no prediction HDF5s readable")
        return None

    lo_w, hi_w = MMN_WINDOW
    lo_x, hi_x = WAVE_WINDOW
    nrow, ncol = PANELS
    # +1.0in of height is reserved below the axes for the legend and the two caption lines;
    # tight_layout's rect keeps the panels out of it so the three never overlap.
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.25 * nrow + 1.0),
                             sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    drawn = 0
    for ax, (_, r) in zip(axes, top.iterrows()):
        ax.grid(False)
        ax.axvspan(lo_w, hi_w, color="#eef2f7", zorder=0)
        ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
        per_model = waves.get((int(r["pair_id"]), r["direction"]), {})
        for m, (tt, uv, _z) in per_model.items():
            sel = (tt >= lo_x) & (tt <= hi_x)
            ax.plot(tt[sel], uv[sel], color=WAVE_MODEL_STYLE[m]["color"], lw=WAVE_LW,
                    alpha=WAVE_ALPHA, zorder=2)
        drawn += bool(per_model)
        ax.set_title(f"{r['rank']:.0f}. {r['method']}\n{r['f_std']:g} → {r['f_dev']:g} Hz   "
                     f"n_agree {int(r['n_agree'])}/{len(models)}   {r['mean_uv']:+.2f} µV",
                     fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(lo_x, hi_x)
    for ax in axes[len(top):]:
        ax.set_visible(False)

    for ax in axes[:len(top)]:
        if ax.get_subplotspec().rowspan.start == (len(top) - 1) // ncol:
            ax.set_xlabel("time from final tone onset (ms)", fontsize=8)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("µV", fontsize=8)

    handles = [Line2D([], [], color=WAVE_MODEL_STYLE[m]["color"], lw=1.6, label=MLABEL[m])
               for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.093))
    fig.suptitle(f"FCz µV difference wave, top {len(top)} literature direction-instances\n"
                 f"ranked by n_agree then mean_uv; each panel autoscales, so read shape rather "
                 f"than one trace's depth against another's", fontsize=9.5, y=0.985)
    fig.tight_layout(rect=(0, 0.155, 1, 0.945))
    fig.text(0.5, 0.070,
             f"Shaded band = the {lo_w:g}–{hi_w:g} ms MMN scoring window. The axis runs to "
             f"{hi_x:g} ms, the full span the criteria read out over: S2's recovery search "
             f"follows a trough that\nmay sit as late as 240 ms. `_counter` panels are the "
             f"reversed direction, so their standard is the pair's higher tone.",
             ha="center", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    name = f"{prefix}_waveforms.png"
    savefig_both(fig, out_dir, name)
    plt.close(fig)
    print(f"  wrote {name}  ({drawn} of {len(top)} panels carry traces)")
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--literature_csv", default=str(LITERATURE),
                   help="the scored literature screen (default: this script's committed screen)")
    p.add_argument("--out_dir", default=str(OUT))
    p.add_argument("--svg_dir", default=None)
    p.add_argument("--no_svg", action="store_true")
    p.add_argument("--predictions_root", default=str(PREDICTIONS))
    p.add_argument("--n_wave", type=int, default=N_WAVE,
                   help=f"how many top instances get a waveform panel (default {N_WAVE})")
    p.add_argument("--skip_waveforms", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_svg_output(out_dir, args.svg_dir, args.no_svg)
    print(f"reading {args.literature_csv}")
    ranked = load_literature_ranked(args.literature_csv)
    models = [m for m in SEARCH_MODELS if f"s7__{m}" in ranked.columns]

    print("\n[1] agreement tiers")
    table_agreement_tiers(ranked, models, out_dir, prefix="literature")

    print("\n[2] the full ranking")
    table_topn(ranked, models, out_dir, n=len(ranked), prefix="literature")

    print("\n[3] consensus yield")
    plot_consensus_heatmap(ranked, models, out_dir, prefix="literature", thresholds=LIT_TOP_X)

    print(f"\n[4] waveforms for the top {args.n_wave}")
    if args.skip_waveforms:
        print("  skipped (--skip_waveforms)")
    else:
        plot_literature_waveforms(ranked, models, out_dir, n=args.n_wave,
                                  predictions_root=args.predictions_root)


if __name__ == "__main__":
    main()
