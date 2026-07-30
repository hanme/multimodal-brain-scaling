"""Shared vocabulary for the novel tone-pair search: the ranking criteria and the cost model.

Both phases rank the SAME way -- Phase 2 only differs in that its scores come from 15 deviants
instead of 1 -- so the criteria live here once and are imported by rank_novel_phase1.py and
rank_novel_phase2.py rather than restated. If the two ever disagree, the Phase-1 -> Phase-2 rank
stability (the headline methodological result) would be measuring the code, not the stimuli.

Vocabulary
----------
pair              one unordered {f_low, f_high} pair == one method_id == one row of the grid CSV.
direction         'regular' (f_low -> f_high) or 'counter' (f_high -> f_low).
direction-instance  one (pair, direction). 903 pairs -> 1806 of these. This is the unit of
                  ranking, matching the existing convention that regular and counter each count
                  as one MMN observation.
n_agree           how many of the 6 models show S7 at FCz for this direction-instance (0-6).
mean_uv           mean trough_uv across ONLY the agreeing models; undefined when n_agree == 0.
                  Averaging over non-agreeing models would mix in traces that failed S2, whose
                  trough latency is not an MMN latency.
mean_uv_all6      mean trough_uv across ALL six models regardless of S7. A DIFFERENT quantity
                  from mean_uv, deliberately named apart so the two are never conflated: it is
                  defined everywhere (including n_agree == 0) and is the right cell value for a
                  heatmap over the whole grid, but it mixes in non-MMN trough latencies.
"""

import numpy as np
import pandas as pd

# whisper-large is excluded from the search: it alone was 51% of the literature screen's cost.
SEARCH_MODELS = ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
                 "wav2vec2-medium", "wav2vec2-large"]

# Fixed reporting choices (see the brief). X is calibrated to the model's own trough distribution,
# NOT to literature EEG scale -- mTRF predictions are ~4x amplitude-shrunk.
ROI = "FCz"
MAPPING = "mtrf"
DIP_UV_THRESHOLD = 0.75

# Measured per-clip extraction cost from the completed literature mmn_extract jobs (core-h/clip).
COST_CORE_H_PER_CLIP = {
    "whisper-medium": 5.87, "whisper-small": 1.81, "wav2vec2-large": 0.74,
    "whisper-base": 0.62, "wav2vec2-medium": 0.39, "whisper-tiny": 0.38,
}
CHF_PER_CORE_H = 0.0055
# Fixed cost of an array task before any clip is processed: python/torch import + model load,
# ~2 min on 8 cores. Measured per-CLIP costs came from 16-clip batches, where this is amortised
# 16 ways; with one method (2 clips) per task it is amortised 2 ways and stops being negligible.
TASK_OVERHEAD_CORE_H = 0.27

# Phase 2 evaluates every pair with AT LEAST ONE direction at n_agree >= this. A criterion on the
# evidence, not a pair count: "5 of the 6 models saw an MMN" is a claim about the stimulus, whereas
# a top-N cut carries whatever happens to sit at rank N. On the 2026-07-27 ranking the previous
# top-145 rule admitted 18 pairs whose best direction only reached 4/6 -- selected to fill a quota,
# not because the models agreed. How many pairs qualify is therefore an OUTPUT of the screen
# (127 on that ranking), not an input, and it moves if the Phase-1 scores move.
MIN_AGREE_PHASE2 = 5

# Marginal CHF to carry ONE pair (both directions, 14 new clips each, all 6 models) into Phase 2.
# Measured from the completed Phase-1 run, not projected from COST_CORE_H_PER_CLIP -- the
# literature per-clip table ran pessimistic for every model. Used only to price a pair count in
# the memo; nothing in the search branches on it.
CHF_PER_PAIR_PHASE2 = 1.325


def extraction_cost_chf(n_method_dirs, clips_per_dir, models=None,
                        overhead_core_h=TASK_OVERHEAD_CORE_H):
    """Projected CHF to extract `clips_per_dir` clips in each of `n_method_dirs` dirs, all models.

    REPORTING ONLY -- printed so you can sanity-check a submission before it runs. Nothing in the
    search branches on this, so a wrong cost constant misleads a log line and nothing else.

    One array task per (model, method dir), so the per-task overhead is paid n_method_dirs x
    n_models times regardless of how many clips each task does.
    """
    models = list(models or SEARCH_MODELS)
    per_dir = (clips_per_dir * sum(COST_CORE_H_PER_CLIP[m] for m in models)
               + overhead_core_h * len(models))
    return n_method_dirs * per_dir * CHF_PER_CORE_H


def load_scored(s7_csv, models=None, roi=ROI, mapping=MAPPING, x=DIP_UV_THRESHOLD):
    """The one row per (model, direction-instance) that the ranking uses.

    analyze_mmn_s7_roi.py emits every ROI x every floor in UV_SWEEP; this is the single
    (roi, mapping, X) slice the search is defined on.
    """
    models = list(models or SEARCH_MODELS)
    df = pd.read_csv(s7_csv)
    sel = df[(df["roi"] == roi) & (df["mapping"] == mapping)
             & np.isclose(df["dip_uv_threshold"], x) & df["model"].isin(models)].copy()
    if sel.empty:
        raise SystemExit(
            f"no rows in {s7_csv} for roi={roi} mapping={mapping} X={x} models={models}. "
            f"Present: roi={sorted(df['roi'].unique())} mapping={sorted(df['mapping'].unique())} "
            f"X={sorted(df['dip_uv_threshold'].unique())} models={sorted(df['model'].unique())}")

    missing = sorted(set(models) - set(sel["model"]))
    if missing:
        raise SystemExit(f"{s7_csv} is missing model(s) {missing}; the search scores all "
                         f"{len(models)} or none (n_agree would not be comparable across "
                         f"direction-instances).")

    sel["pair_id"] = sel["method"].str.split("_").str[1].astype(int)
    sel["direction"] = np.where(sel["is_counter"], "counter", "regular")
    return sel


def rank_directions(scored, grid_index=None, models=None):
    """One row per direction-instance, ranked best-first.

    Sort: n_agree descending, then mean_uv ascending (most negative = strongest MMN wins).
    n_agree == 0 has undefined mean_uv and lands last. Any remaining exact tie is broken
    deterministically by (pair_id, direction) so reruns give byte-identical rankings.
    """
    models = list(models or SEARCH_MODELS)

    wide = scored.pivot_table(index=["pair_id", "direction"], columns="model",
                              values=["s7", "trough_uv"], aggfunc="first")
    s7 = wide["s7"].reindex(columns=models).astype("boolean")
    uv = wide["trough_uv"].reindex(columns=models)

    out = pd.DataFrame(index=wide.index)
    out["n_agree"] = s7.fillna(False).sum(axis=1).astype(int)
    # mean over agreeing models only; all-NaN rows (n_agree == 0) give NaN, which sorts last.
    out["mean_uv"] = uv.where(s7.fillna(False)).mean(axis=1, skipna=True)
    for m in models:
        out[f"s7__{m}"] = s7[m].fillna(False).astype(bool)
        out[f"trough_uv__{m}"] = uv[m]

    out = out.reset_index()
    out = out.sort_values(["n_agree", "mean_uv", "pair_id", "direction"],
                          ascending=[False, True, True, True],
                          na_position="last", kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.insert(2, "method", out["pair_id"].map(lambda p: f"method_{p}")
               + np.where(out["direction"].eq("counter"), "_counter", ""))

    if grid_index is not None:
        gi = pd.read_csv(grid_index) if isinstance(grid_index, str) else grid_index
        out = out.merge(gi.rename(columns={"method_id": "pair_id"}), on="pair_id", how="left")
        if out["f_low"].isna().any():
            missing = sorted(out.loc[out["f_low"].isna(), "pair_id"].unique())
            raise SystemExit(f"grid index has no entry for pair_id {missing[:10]}...")
    return out


def find_exact_ties(ranked):
    """Direction-instances that tie on BOTH sort keys, so only the deterministic tiebreak
    separates them. Logged rather than silently resolved."""
    key = ranked[["n_agree", "mean_uv"]].copy()
    key["mean_uv"] = key["mean_uv"].round(9)
    dup = key.duplicated(keep=False) & ranked["n_agree"].gt(0)
    return ranked[dup]


def select_pairs_by_agreement(ranked, min_agree=MIN_AGREE_PHASE2):
    """Pairs with AT LEAST ONE direction at n_agree >= min_agree, best-ranked pair first.

    One qualifying direction selects the pair, and then BOTH of its directions go to Phase 2
    (method_dirs_for_pairs emits both) -- including one that scored far below the cut. That is
    deliberate: a strong asymmetry between the two directions of a pair is the frequency-preference
    artifact the counterbalanced design exists to detect, and it can only be measured if the
    reversal is tested too. Requiring both directions to clear the bar instead would presuppose
    that answer, and on the Phase-1 ranking would leave 4 pairs out of 127.

    Returns (pair_ids, n_qualifying_directions). The second is how many direction-instances cleared
    the bar: between len(pair_ids) and 2 * len(pair_ids), and the gap is the asymmetry.
    """
    qual = ranked[ranked["n_agree"] >= min_agree].sort_values("rank", kind="mergesort")
    return [int(p) for p in dict.fromkeys(qual["pair_id"])], len(qual)


def method_dirs_for_pairs(pair_ids):
    """Both direction dirs for each pair, in stable order -- the METHOD_LIST for extraction."""
    return [name for p in pair_ids for name in (f"method_{p}", f"method_{p}_counter")]


def tier_summary(ranked, models=None):
    """Count of direction-instances at each n_agree tier, best first."""
    n_models = len(list(models or SEARCH_MODELS))
    counts = ranked["n_agree"].value_counts().reindex(range(n_models, -1, -1), fill_value=0)
    return counts


def agreement_tiers(ranked, models=None):
    """Direction-instance and pair counts at each n_agree tier, best first.

    Three counts per tier, because a pair contributes two direction-instances and they need not
    land together:
      directions    direction-instances at the tier (of 2 x n_pairs)
      pairs_both    pairs whose BOTH directions are at the tier
      pairs_either  pairs with AT LEAST ONE direction at the tier

    pairs_either - pairs_both is the direction-asymmetry signal: pairs the tier only half claims.
    A tier where the two columns are nearly equal is measuring a deviance response; one where
    pairs_both collapses to ~0 is measuring a frequency preference (see
    select_pairs_by_agreement). pairs_either at tier >= MIN_AGREE_PHASE2 is the Phase-2 count.
    """
    n_models = len(list(models or SEARCH_MODELS))
    tiers = list(range(n_models, -1, -1))
    n_dir, n_pair = len(ranked), ranked["pair_id"].nunique()

    rows = []
    for t in tiers:
        at = ranked[ranked["n_agree"] == t]
        per_pair = at.groupby("pair_id").size()
        both, either = int((per_pair == 2).sum()), int(len(per_pair))
        rows.append({
            "n_agree": t,
            "directions": len(at),
            "pct_directions": 100.0 * len(at) / n_dir if n_dir else float("nan"),
            "pairs_both": both,
            "pct_pairs_both": 100.0 * both / n_pair if n_pair else float("nan"),
            "pairs_either": either,
            "pct_pairs_either": 100.0 * either / n_pair if n_pair else float("nan"),
        })
    return pd.DataFrame(rows)


def marginal_s7_rates(ranked, models=None):
    """Each model's own S7 rate over the direction-instances -- the chance-baseline inputs."""
    models = list(models or SEARCH_MODELS)
    return pd.Series({m: float(ranked[f"s7__{m}"].mean()) for m in models})


def expected_tier_counts(ranked, models=None):
    """Direction-instances expected at each n_agree tier if the models were INDEPENDENT.

    The exact Poisson-binomial over each model's own marginal S7 rate, convolved one model at a
    time -- not a plain binomial, because the six rates differ (0.33 to 0.60) and pooling them
    would understate the spread and so overstate how surprising a 6/6 is.

    This is the baseline the observed tier counts have to beat. Models that share an architecture
    family, a training corpus and a frozen mTRF mapping are not independent, so agreement above
    this line is a weaker claim than it looks; agreement AT this line means the search found no
    cross-model structure at all, which is the result that decides what Phase 2 is for.
    """
    models = list(models or SEARCH_MODELS)
    p = marginal_s7_rates(ranked, models)

    dist = np.zeros(len(models) + 1)
    dist[0] = 1.0
    for m in models:                                   # convolve in one Bernoulli(p_m) at a time
        dist = np.r_[0.0, dist[:-1]] * p[m] + dist * (1.0 - p[m])

    tiers = list(range(len(models), -1, -1))
    return pd.DataFrame({
        "n_agree": tiers,
        "p_chance": [dist[t] for t in tiers],
        "expected": [dist[t] * len(ranked) for t in tiers],
        "observed": [int((ranked["n_agree"] == t).sum()) for t in tiers],
    })


def pair_cutoff_curve(ranked, thresholds=None, uv_cutoffs=None,
                      chf_per_pair=CHF_PER_PAIR_PHASE2):
    """Unique pairs (and their Phase-2 cost) qualifying at each (n_agree, mean_uv) cutoff.

    Counted in PAIRS, not direction-instances, because Phase 2 is priced per pair:
    select_pairs_by_agreement carries both directions of any pair it selects, so a pair whose two
    directions both qualify costs exactly as much as one where only a single direction does. The
    row at (n_agree_min=MIN_AGREE_PHASE2, mean_uv_max=0) is the Phase-2 selection itself.

    A NaN mean_uv (n_agree == 0) never qualifies at any finite uv cutoff.
    """
    n_models = len(_models_in_ranked(ranked))
    thresholds = list(range(1, n_models + 1)) if thresholds is None else list(thresholds)
    uv_cutoffs = [0.0] if uv_cutoffs is None else list(uv_cutoffs)

    rows = []
    for t in thresholds:
        for u in uv_cutoffs:
            ok = ranked[(ranked["n_agree"] >= t) & (ranked["mean_uv"] <= u)]
            n_pairs = int(ok["pair_id"].nunique())
            rows.append({"n_agree_min": t, "mean_uv_max": u, "directions": len(ok),
                         "pairs": n_pairs, "chf": n_pairs * chf_per_pair})
    return pd.DataFrame(rows)


def direction_gap(ranked):
    """One row per pair: each direction's n_agree/mean_uv and the gap between them.

    The gap is the frequency-preference measure. A pair that only works f_low -> f_high is
    evidence the models respond to one of its two TONES, not to the change between them; a
    deviance response should survive reversal.
    """
    wide = ranked.pivot_table(index="pair_id", columns="direction",
                              values=["n_agree", "mean_uv"], aggfunc="first")
    out = pd.DataFrame(index=wide.index)
    for d in ("regular", "counter"):
        out[f"n_agree_{d}"] = (wide["n_agree"][d].astype(int) if d in wide["n_agree"] else 0)
        out[f"mean_uv_{d}"] = (wide["mean_uv"][d] if d in wide["mean_uv"] else np.nan)
    out["n_agree_max"] = out[["n_agree_regular", "n_agree_counter"]].max(axis=1)
    out["n_agree_min"] = out[["n_agree_regular", "n_agree_counter"]].min(axis=1)
    out["n_agree_gap"] = out["n_agree_regular"].sub(out["n_agree_counter"]).abs()
    return out.reset_index()


def _models_in_ranked(ranked):
    """The models a ranked frame actually carries, in SEARCH_MODELS order."""
    return [m for m in SEARCH_MODELS if f"s7__{m}" in ranked.columns]
