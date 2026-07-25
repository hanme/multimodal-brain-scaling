"""Shared vocabulary for the novel tone-pair search: the ranking criteria and the cost model.

Both phases rank the SAME way -- Phase 2 only differs in that its scores come from 15 deviants
instead of 1 -- so the criteria live here once and are imported by rank_novel_phase1.py and
rank_novel_phase2.py rather than restated. If the two ever disagree, the Phase-1 -> Phase-2 rank
stability (the headline methodological result) would be measuring the code, not the stimuli.

Vocabulary
----------
pair              one unordered {f_low, f_high} pair == one method_id == one row of the grid CSV.
direction         'regular' (f_low -> f_high) or 'counter' (f_high -> f_low).
direction-instance  one (pair, direction). 496 pairs -> 992 of these. This is the unit of
                  ranking, matching the existing convention that regular and counter each count
                  as one MMN observation.
n_agree           how many of the 6 models show S7 at FCz for this direction-instance (0-6).
mean_uv           mean trough_uv across ONLY the agreeing models; undefined when n_agree == 0.
                  Averaging over non-agreeing models would mix in traces that failed S2, whose
                  trough latency is not an MMN latency.
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

# Phase 2 evaluates a FIXED top-145 pairs. Deliberately a constant, not derived from a budget or
# from Phase 1's measured spend: the pair count is a design decision, and making it depend on how
# an earlier job happened to run would make the search non-reproducible. The cost model below is
# REPORTING ONLY -- nothing branches on it.
N_PAIRS_PHASE2 = 145


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


def selection_walk(ranked, n_pairs):
    """Accumulate UNIQUE pairs down the ranked list until n_pairs is reached.

    Deduping on pair rather than list position is what makes both directions of a selected pair go
    to Phase 2 -- including a direction that did not itself rank highly. That is deliberate: a
    strong asymmetry between the two directions of a pair is the frequency-preference artifact the
    counterbalanced design exists to detect, and it can only be measured if both are carried
    forward.
    """
    selected, order = [], {}
    for row in ranked.itertuples(index=False):
        if row.pair_id not in order:
            order[row.pair_id] = row.rank
            selected.append(row.pair_id)
            if len(selected) >= n_pairs:
                return selected, row.rank
    return selected, (ranked["rank"].max() if len(ranked) else 0)


def method_dirs_for_pairs(pair_ids):
    """Both direction dirs for each pair, in stable order -- the METHOD_LIST for extraction."""
    return [name for p in pair_ids for name in (f"method_{p}", f"method_{p}_counter")]


def tier_summary(ranked, models=None):
    """Count of direction-instances at each n_agree tier, best first."""
    n_models = len(list(models or SEARCH_MODELS))
    counts = ranked["n_agree"].value_counts().reindex(range(n_models, -1, -1), fill_value=0)
    return counts
