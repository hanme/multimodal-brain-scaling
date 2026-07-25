#!/usr/bin/env python
"""Phase 2 of the novel tone-pair search: final ranking, consensus set, rank stability.

Same criteria as Phase 1 (imported from novel_search_common, not restated), applied to scores
computed from all 15 deviants instead of one. These supersede the Phase-1 values.

Three outputs:
  phase2_final_ranking.csv  all selected direction-instances, Phase-1 columns plus the Phase-1
                            rank and the shift.
  consensus_set.csv         the headline result: pairs where BOTH directions reach n_agree = 6/6,
                            sorted by the mean of the two directions' mean_uv. "All models agree,
                            in both directions" is the strongest claim this design supports --
                            requiring both directions is what rules out a pure frequency
                            preference masquerading as a deviance response.
  console                   Spearman rho between the Phase-1 and Phase-2 rankings on the shared
                            direction-instances. This is the methodological result: it says
                            whether a 4-clip screen is a valid proxy for a 32-clip evaluation,
                            and therefore whether this two-phase design is reusable.

A moderate rho is not a failure. Phase 1 scores one draw from the stochastic-prefix distribution;
Phase 2 scores its 15-draw mean. The models are deterministic, so the difference is sampling of
the stimulus ensemble, not measurement noise.

  python scripts/rank_novel_phase2.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from novel_search_common import (
    SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD, load_scored, rank_directions,
    find_exact_ties, tier_summary,
)


def rank_stability(phase1_csv, ranked2):
    """Spearman rho between the two phases' ranks over the direction-instances they share."""
    if not Path(phase1_csv).exists():
        print(f"  (Phase-1 ranking not found: {phase1_csv} -- skipped)")
        return ranked2.assign(phase1_rank=np.nan, rank_shift=np.nan)

    r1 = pd.read_csv(phase1_csv)[["pair_id", "direction", "rank", "n_agree", "mean_uv"]]
    r1 = r1.rename(columns={"rank": "phase1_rank", "n_agree": "phase1_n_agree",
                            "mean_uv": "phase1_mean_uv"})
    merged = ranked2.merge(r1, on=["pair_id", "direction"], how="left")

    both = merged.dropna(subset=["phase1_rank"])
    if len(both) < 3:
        print(f"  only {len(both)} shared direction-instances -- rho not computed")
        return merged

    # Phase-1 ranks are positions in the 992-instance list; re-rank within the shared subset so
    # the comparison is like-for-like.
    rho, pval = stats.spearmanr(both["phase1_rank"].rank(), both["rank"])
    print(f"  Spearman rho(Phase-1 rank, Phase-2 rank) = {rho:+.3f}  (p={pval:.2g}, "
          f"n={len(both)} shared direction-instances)")

    tier_same = (both["n_agree"] == both["phase1_n_agree"]).mean()
    print(f"  n_agree tier unchanged between phases: {100 * tier_same:.1f}%")
    print(f"  n_agree moved up: {(both['n_agree'] > both['phase1_n_agree']).sum()}, "
          f"down: {(both['n_agree'] < both['phase1_n_agree']).sum()}")

    merged["rank_shift"] = merged["phase1_rank"].rank() - merged["rank"]
    moved = merged["rank_shift"].abs().dropna()
    if len(moved):
        print(f"  |rank shift| within the shared subset: median {moved.median():.0f}, "
              f"90th pct {moved.quantile(0.9):.0f}, max {moved.max():.0f}")
    return merged


def build_consensus(ranked, models):
    """Pairs whose BOTH directions reach the top tier, sorted by the mean of their mean_uv."""
    top = len(models)
    piv = ranked.pivot_table(index="pair_id", columns="direction",
                             values=["n_agree", "mean_uv"], aggfunc="first")
    if not {"regular", "counter"}.issubset(piv["n_agree"].columns):
        return pd.DataFrame()

    both_top = (piv["n_agree"]["regular"] == top) & (piv["n_agree"]["counter"] == top)
    cons = pd.DataFrame({
        "pair_id": piv.index[both_top],
        "mean_uv_regular": piv["mean_uv"]["regular"][both_top].values,
        "mean_uv_counter": piv["mean_uv"]["counter"][both_top].values,
    })
    if cons.empty:
        return cons
    cons["mean_uv_both"] = cons[["mean_uv_regular", "mean_uv_counter"]].mean(axis=1)
    cons["direction_gap_uv"] = (cons["mean_uv_regular"] - cons["mean_uv_counter"]).abs()

    meta = ranked.drop_duplicates("pair_id").set_index("pair_id")
    for col in ("f_low", "f_high", "semitones", "pct_deviance"):
        if col in meta.columns:
            cons[col] = cons["pair_id"].map(meta[col])
    return cons.sort_values("mean_uv_both").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s7_csv", default="outputs/results_novel_search/phase2_mmn_s7_roi.csv")
    p.add_argument("--grid_index", default="outputs/results_novel_search/grid_index.csv")
    p.add_argument("--phase1_ranked",
                   default="outputs/results_novel_search/phase1_ranked_directions.csv")
    p.add_argument("--out_dir", default="outputs/results_novel_search")
    p.add_argument("--literature_s7_csv",
                   default="outputs/results_24freq_7models/mmn_s7_roi.csv")
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
    ranked = rank_directions(scored, grid_index=grid_index, models=models)

    print(f"=== Phase 2 final ranking ({args.roi}, {args.mapping}, X={args.dip_uv_threshold} uV, "
          f"{len(models)} models) ===")
    print(f"{len(ranked)} direction-instances from {ranked['pair_id'].nunique()} pairs "
          f"(scores now averaged over all 15 deviants)")

    print("\nn_agree tiers:")
    for tier, n in tier_summary(ranked, models).items():
        print(f"  {tier}/{len(models)}: {n:5d} ({100 * n / len(ranked):5.1f}%)")

    ties = find_exact_ties(ranked)
    print(f"exact ties on (n_agree, mean_uv), broken by (pair_id, direction): {len(ties)}")

    print("\n--- Phase 1 -> Phase 2 rank stability ---")
    ranked = rank_stability(args.phase1_ranked, ranked)

    print("\n--- consensus set: both directions at "
          f"{len(models)}/{len(models)} ---")
    cons = build_consensus(ranked, models)
    if cons.empty:
        print("  EMPTY: no pair reaches the top tier in both directions. Report this as the "
              "result -- it is the same cross-model disagreement the literature screen found, "
              "not a pipeline fault.")
    else:
        print(f"  {len(cons)} pairs ({100 * len(cons) / ranked['pair_id'].nunique():.1f}% of "
              f"those evaluated)")
        cols = [c for c in ("pair_id", "f_low", "f_high", "semitones", "pct_deviance",
                            "mean_uv_both", "direction_gap_uv") if c in cons.columns]
        print(cons[cols].head(15).to_string(index=False))

    print("\n--- best pair vs the best literature pair ---")
    if Path(args.literature_s7_csv).exists():
        lit = rank_directions(load_scored(args.literature_s7_csv, models=models), models=models)
        b_n, b_l = ranked.iloc[0], lit.iloc[0]
        print(f"  novel:      {b_n['method']:<22} n_agree={b_n['n_agree']}/{len(models)}  "
              f"mean_uv={b_n['mean_uv']:.3f}")
        print(f"  literature: {b_l['method']:<22} n_agree={b_l['n_agree']}/{len(models)}  "
              f"mean_uv={b_l['mean_uv']:.3f}")
        beats = ((b_n["n_agree"], -b_n["mean_uv"]) > (b_l["n_agree"], -b_l["mean_uv"]))
        print(f"  novel beats literature on (n_agree, then mean_uv): {beats}")
        n_beat = ((ranked["n_agree"] > b_l["n_agree"])
                  | ((ranked["n_agree"] == b_l["n_agree"])
                     & (ranked["mean_uv"] < b_l["mean_uv"]))).sum()
        print(f"  novel direction-instances beating the best literature one: {n_beat}"
              f"/{len(ranked)}")
    else:
        print(f"  (literature CSV not found: {args.literature_s7_csv})")

    ranked_path = out_dir / "phase2_final_ranking.csv"
    cons_path = out_dir / "consensus_set.csv"
    ranked.to_csv(ranked_path, index=False)
    cons.to_csv(cons_path, index=False)
    print(f"\nwrote {ranked_path}  ({len(ranked)} rows)")
    print(f"wrote {cons_path}  ({len(cons)} pairs)")


if __name__ == "__main__":
    main()
