#!/usr/bin/env python
"""Phase 1 of the novel tone-pair search: rank all 992 direction-instances, select for Phase 2.

Input is the FCz / mTRF / X=0.75 slice of analyze_mmn_s7_roi.py's output over the novel grid,
where each condition was scored from ONE deviant realization (N7/var1). Output is the ranking and
the pair set that Phase 2 re-evaluates with the full 15-deviant grid.

Phase 2 evaluates the top --n_pairs pairs (default 145). That is a flat constant: it does not
depend on what Phase 1 cost, and nothing here reads sacct. The projected extraction cost is
printed for a sanity check only.

  python scripts/rank_novel_phase1.py

Reported but NOT gated on (see the console summary): whether n_agree tracks deviance, how often
the two directions of a pair disagree, and how the novel n_agree distribution compares to the
literature methods scored identically.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from novel_search_common import (
    SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD, N_PAIRS_PHASE2, load_scored,
    rank_directions, find_exact_ties, selection_walk, method_dirs_for_pairs, tier_summary,
    extraction_cost_chf,
)

# Phase 2 adds the 14 deviants each method dir is missing (3 trial levels x 5 variations = 15
# deviants, of which N7/var1 already exists from Phase 1) -- the standard is already there too.
PHASE2_NEW_CLIPS_PER_DIR = 14


def report_deviance_scaling(ranked, n_bins=8):
    """Does n_agree rise with deviance? A monotone rise then plateau is the expected
    deviance-scaling signature; a flat relationship says the metric is not tracking deviance."""
    d = ranked.dropna(subset=["pct_deviance"]).copy()
    if d.empty:
        print("  (no deviance info -- pass --grid_index)")
        return
    d["bin"] = pd.qcut(d["pct_deviance"], q=min(n_bins, d["pct_deviance"].nunique()),
                       duplicates="drop")
    print(f"  {'deviance bin (%)':>26}  {'n':>5}  {'mean n_agree':>12}")
    for b, g in d.groupby("bin", observed=True):
        print(f"  {f'{b.left:8.1f} - {b.right:8.1f}':>26}  {len(g):5d}  {g['n_agree'].mean():12.2f}")
    rho = d["pct_deviance"].corr(d["n_agree"], method="spearman")
    print(f"  Spearman rho(pct_deviance, n_agree) = {rho:+.3f} over {len(d)} direction-instances")


def report_direction_asymmetry(ranked):
    """How often do the two directions of a pair land in different n_agree tiers? Large asymmetry
    is the frequency-preference artifact the counterbalanced design exists to detect."""
    piv = ranked.pivot_table(index="pair_id", columns="direction", values="n_agree")
    if not {"regular", "counter"}.issubset(piv.columns):
        print("  (need both directions present)")
        return
    diff = (piv["regular"] - piv["counter"]).abs()
    print(f"  pairs with both directions: {len(diff)}")
    print(f"  identical n_agree:          {(diff == 0).sum()} ({100 * (diff == 0).mean():.1f}%)")
    print(f"  differ by >=2 tiers:        {(diff >= 2).sum()} ({100 * (diff >= 2).mean():.1f}%)")
    print(f"  mean |regular - counter|:   {diff.mean():.2f} tiers  (max {int(diff.max())})")


def report_vs_literature(ranked, literature_csv, models):
    """The whole point of the exercise: is the novel grid finding better stimuli than the
    published ones, scored the same way?"""
    if not Path(literature_csv).exists():
        print(f"  (literature CSV not found: {literature_csv} -- skipped)")
        return
    lit_ranked = rank_directions(load_scored(literature_csv, models=models), models=models)
    n_models = len(models)
    print(f"  {'n_agree':>8}  {'novel':>16}  {'literature':>16}")
    for tier in range(n_models, -1, -1):
        nv = (ranked["n_agree"] == tier).sum()
        lt = (lit_ranked["n_agree"] == tier).sum()
        print(f"  {tier:>8}  {nv:6d} ({100 * nv / len(ranked):5.1f}%)  "
              f"{lt:6d} ({100 * lt / len(lit_ranked):5.1f}%)")
    print(f"  mean n_agree: novel {ranked['n_agree'].mean():.2f}  "
          f"literature {lit_ranked['n_agree'].mean():.2f} (n={len(lit_ranked)})")
    best_n, best_l = ranked.iloc[0], lit_ranked.iloc[0]
    print(f"  best novel:      {best_n['method']:<22} n_agree={best_n['n_agree']} "
          f"mean_uv={best_n['mean_uv']:.3f}")
    print(f"  best literature: {best_l['method']:<22} n_agree={best_l['n_agree']} "
          f"mean_uv={best_l['mean_uv']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s7_csv", default="outputs/results_novel_search/phase1_mmn_s7_roi.csv",
                   help="analyze_mmn_s7_roi.py output over the novel-grid predictions")
    p.add_argument("--grid_index", default="outputs/results_novel_search/grid_index.csv",
                   help="build_novel_grid_csv.py index (frequencies, semitones, deviance)")
    p.add_argument("--out_dir", default="outputs/results_novel_search")
    p.add_argument("--method_list_out", default="outputs/novel_methods_phase2.txt",
                   help="METHOD_LIST for the Phase-2 extraction arrays (both directions)")
    p.add_argument("--n_pairs", type=int, default=N_PAIRS_PHASE2,
                   help=f"how many top pairs go to Phase 2 (default {N_PAIRS_PHASE2})")
    p.add_argument("--literature_s7_csv",
                   default="outputs/results_24freq_7models/mmn_s7_roi.csv",
                   help="literature screen scored the same way, for the comparison summary")
    p.add_argument("--roi", default=ROI)
    p.add_argument("--mapping", default=MAPPING)
    p.add_argument("--dip_uv_threshold", type=float, default=DIP_UV_THRESHOLD)
    p.add_argument("--models", default=",".join(SEARCH_MODELS))
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = load_scored(args.s7_csv, models=models, roi=args.roi, mapping=args.mapping,
                         x=args.dip_uv_threshold)
    grid_index = args.grid_index if Path(args.grid_index).exists() else None
    if grid_index is None:
        print(f"WARNING: {args.grid_index} missing -- frequency/deviance columns and the "
              f"deviance-scaling check will be omitted.")
    ranked = rank_directions(scored, grid_index=grid_index, models=models)

    n_inst = len(ranked)
    print(f"=== Phase 1 ranking ({args.roi}, {args.mapping}, X={args.dip_uv_threshold} uV, "
          f"{len(models)} models) ===")
    print(f"scored rows: {len(scored)} = {n_inst} direction-instances x {len(models)} models")
    if len(scored) != n_inst * len(models):
        print(f"  WARNING: not a complete grid -- expected {n_inst * len(models)} rows. "
              f"A model missing a condition is counted as non-agreeing.")

    print("\nn_agree tiers (direction-instances):")
    for tier, n in tier_summary(ranked, models).items():
        print(f"  {tier}/{len(models)}: {n:5d} ({100 * n / n_inst:5.1f}%)")

    ties = find_exact_ties(ranked)
    print(f"\nexact ties on (n_agree, mean_uv), broken by (pair_id, direction): {len(ties)}")
    for row in ties.head(10).itertuples(index=False):
        print(f"  rank {row.rank:4d}  {row.method:<22} n_agree={row.n_agree} "
              f"mean_uv={row.mean_uv:.6f}")
    if len(ties) > 10:
        print(f"  ... and {len(ties) - 10} more (all listed in the ranked CSV)")

    n_pairs = min(args.n_pairs, ranked["pair_id"].nunique())
    selected, walk_depth = selection_walk(ranked, n_pairs)
    print(f"\nselection walk: {len(selected)} unique pairs consumed the top {walk_depth} of "
          f"{n_inst} ranked direction-instances")
    sel_rank = ranked[ranked["pair_id"].isin(selected)]
    print(f"  their {len(sel_rank)} direction-instances span n_agree "
          f"{sel_rank['n_agree'].min()}-{sel_rank['n_agree'].max()}; "
          f"{(sel_rank['rank'] > walk_depth).sum()} were pulled in below the walk depth as the "
          f"other direction of a selected pair")
    # Informational only -- nothing branches on this.
    projected = extraction_cost_chf(2 * len(selected), PHASE2_NEW_CLIPS_PER_DIR, models)
    print(f"  projected Phase-2 extraction: {2 * len(selected)} dirs x "
          f"{PHASE2_NEW_CLIPS_PER_DIR} new clips = "
          f"{2 * len(selected) * PHASE2_NEW_CLIPS_PER_DIR} clips, ~{projected:.0f} CHF")

    print("\n--- sanity check: n_agree vs deviance (reported, not gating) ---")
    report_deviance_scaling(ranked)
    print("\n--- sanity check: direction asymmetry (reported, not gating) ---")
    report_direction_asymmetry(ranked)
    print("\n--- sanity check: novel vs literature, scored identically ---")
    report_vs_literature(ranked, args.literature_s7_csv, models)

    ranked_path = out_dir / "phase1_ranked_directions.csv"
    pairs_path = out_dir / "phase2_selected_pairs.csv"
    ranked.to_csv(ranked_path, index=False)
    pd.DataFrame({"method_id": selected,
                  "selection_order": np.arange(1, len(selected) + 1)}).to_csv(pairs_path,
                                                                             index=False)
    Path(args.method_list_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.method_list_out).write_text("\n".join(method_dirs_for_pairs(selected)) + "\n")

    print(f"\nwrote {ranked_path}  ({len(ranked)} rows)")
    print(f"wrote {pairs_path}  ({len(selected)} pairs)")
    print(f"wrote {args.method_list_out}  ({2 * len(selected)} method dirs)")


if __name__ == "__main__":
    main()
