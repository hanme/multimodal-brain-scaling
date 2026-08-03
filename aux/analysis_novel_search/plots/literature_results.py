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

import matplotlib as mpl

mpl.use("Agg")

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
    load_literature, table_agreement_tiers, table_topn, plot_consensus_heatmap,
    plot_instance_waveforms, plot_instance_panels_individual, INSTANCE_MODES,
)
from novel_search_plots import configure_svg_output                    # noqa: E402

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

# The x-window itself lives in phase1_results.WAVE_XLIM, shared with every other waveform figure.
N_WAVE = 10
# Where the one-figure-per-stimulus panels go, under plots/ and svgs/ alike.
PANEL_SUBDIR = "literature_panels"


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
    p.add_argument("--skip_panels", action="store_true",
                   help="skip the one-figure-per-stimulus panels (48 figures x PNG+SVG)")
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

    wave_kw = dict(predictions_root=args.predictions_root, prefix="literature",
                   expect_n_deviants=N_DEVIANTS)

    print(f"\n[4] waveforms for the top {args.n_wave}")
    if args.skip_waveforms:
        print("  skipped (--skip_waveforms)")
    else:
        # Three views of the same ten instances: every model overlaid, and the cross-model mean
        # in each of the two units. The µV mean is the one to read with care (Caveat 2).
        for mode in INSTANCE_MODES:
            plot_instance_waveforms(ranked, models, out_dir, n=args.n_wave, mode=mode, **wave_kw)

    print(f"\n[5] one panel per stimulus -> {PANEL_SUBDIR}/")
    if args.skip_panels:
        print("  skipped (--skip_panels)")
    else:
        # All 48, not just the top 10: the point of this directory is that any literature
        # stimulus can be looked at on its own, including the ones the ranking puts last.
        plot_instance_panels_individual(ranked, models, out_dir, PANEL_SUBDIR, **wave_kw)


if __name__ == "__main__":
    main()
