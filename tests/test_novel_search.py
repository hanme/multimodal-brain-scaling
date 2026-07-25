"""Tests for the novel tone-pair search: grid construction, scoring, ranking, selection, cost.

The expensive parts of this search run on the cluster, so the parts that decide WHAT gets run
there -- the grid, the ranking criteria and the selection walk -- are exercised here
against a synthetic in-silico prediction set. The fixture writes prediction HDF5s in the exact
layout insilico_mmn_electrodes.py produces, so analyze_mmn_s7_roi.py scores them for real rather
than through a stub.
"""

import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_novel_grid_csv as bg  # noqa: E402
import novel_search_common as nsc  # noqa: E402

SOA_MS = 580.0
ELECTRODES = ["Fz", "F3", "F4", "FCz", "Cz", "C3", "C4"]
MODELS = nsc.SEARCH_MODELS


# ──────────────────────────────────────────────────────────
# Grid construction
# ──────────────────────────────────────────────────────────

APPROVED_GRID = [200, 224, 252, 283, 317, 356, 400, 449, 504, 566, 635, 713, 800, 898,
                 1008, 1131, 1270, 1425, 1600, 1796, 2016, 2263, 2540, 2851, 3200, 3592,
                 4032, 4525, 5080, 5702, 6400, 7184, 7500]


def test_grid_matches_the_approved_frequency_list_exactly():
    assert bg.build_grid() == APPROVED_GRID


def test_ladder_is_exact_whole_tones_with_one_deliberate_extra():
    freqs = bg.build_grid()
    ladder, extra = freqs[:-1], freqs[-1]
    steps = [12 * math.log2(b / a) for a, b in zip(ladder, ladder[1:])]
    # 2.000 st exactly, up to integer-Hz rounding of each rung
    assert max(abs(s - 2.0) for s in steps) < 0.05, steps
    assert extra == 7500
    # the one irregular rung, and the grid's only sub-2-semitone pair
    assert 12 * math.log2(extra / ladder[-1]) == pytest.approx(0.745, abs=0.01)


def test_octaves_of_the_ladder_start_land_exactly_on_the_grid():
    """A whole-tone step means 6 steps = 12 semitones, so true 2:1 pairs exist. At the 2.024 st
    an evenly-spanned 200-7500 grid would imply, no pair anywhere would be an exact octave."""
    freqs = set(bg.build_grid())
    assert {200, 400, 800, 1600, 3200, 6400} <= freqs


def test_top_of_grid_stays_below_nyquist():
    """16 kHz sample rate for every model; a tone at or above 8 kHz would alias."""
    assert max(bg.build_grid()) < 8000


def test_grid_rows_are_unordered_pairs_with_no_diagonal():
    freqs = bg.build_grid()
    rows, index = bg.build_rows(freqs)
    bg.verify(freqs, rows)  # raises on duplicates / diagonal / off-grid / id collision / Nyquist
    n = len(freqs)
    assert len(rows) == 528 == len(index) == n * (n - 1) // 2
    assert all(r["standard_freq"] < r["deviant_freq"] for r in rows)
    assert [r["method_id"] for r in rows] == list(range(1001, 1529))
    assert min(i["pct_deviance"] for i in index) == pytest.approx(4.4, abs=0.1)


def test_grid_schema_matches_the_literature_sheet_exactly():
    """00aa_generate_audio_stimuli.py and insilico_mmn read this schema by name."""
    lit = (REPO / "data/metadata/literature_frequency_intensity_duration_metadata.csv")
    assert bg.COLUMNS == lit.read_text().splitlines()[0].split(",")


def test_verify_rejects_a_degenerate_pair():
    rows, _ = bg.build_rows([100, 200])
    rows[0]["deviant_freq"] = rows[0]["standard_freq"]
    with pytest.raises(AssertionError, match="degenerate"):
        bg.verify([100, 200], rows)


# ──────────────────────────────────────────────────────────
# Synthetic in-silico predictions
# ──────────────────────────────────────────────────────────

def _mmn_trace(t, trough_uv):
    """A dip at 170 ms that recovers by ~290 ms -- passes S2 with the requested trough depth.

    Returned in native units (Volts): uv_diff_wave multiplies by 1e6.
    """
    return -abs(trough_uv) * np.exp(-(((t - 170.0) / 45.0) ** 2)) * 1e-6


def _write_predictions(root, model, methods, trough_by_method, layer="blocks.0", seed=0):
    """One electrode_predictions__<layer>.h5, laid out like insilico_mmn_electrodes.py's."""
    rng = np.random.default_rng(seed)
    t = np.arange(-3 * SOA_MS, 500.0, 20.0)          # 20 ms grid, full pre-onset baseline
    d = root / model
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"electrode_predictions__{layer}.h5", "w") as h5:
        h5.create_dataset("electrodes", data=np.array(ELECTRODES, dtype="S8"))
        h5.create_dataset("electrode_nc_r", data=np.full(len(ELECTRODES), 0.5, np.float32))
        for method in methods:
            trough = trough_by_method(model, method)
            # Baseline noise gives the z-score a finite sd; small vs the MMN so S2 is decided by
            # the dip, not by chance.
            std = rng.normal(0, 2e-8, size=(len(t), len(ELECTRODES)))
            dev = std.copy()
            if trough:
                dev += _mmn_trace(t, trough)[:, None]
            g = h5.create_group(method)
            g.attrs.update(soa_ms=SOA_MS, context_final="", source="novel_grid",
                           final_tone_onset_s=0.0, n_deviants=1)
            g.create_dataset("time_ms", data=t.astype(np.float32))
            g.create_dataset("standard", data=std.astype(np.float32))
            g.create_dataset("deviant_mean", data=dev.astype(np.float32))
    return t


@pytest.fixture
def scored_grid(tmp_path):
    """A 5-frequency grid (10 pairs, 20 direction-instances) scored by the real scorer.

    Trough depth is engineered so n_agree is known in advance: pair 1001 agrees in all 6 models
    in both directions, 1002 in 4, 1003 in 2, and the rest in none.
    """
    grid_csv = tmp_path / "grid.csv"
    index_csv = tmp_path / "grid_index.csv"
    subprocess.run([sys.executable, str(SCRIPTS / "build_novel_grid_csv.py"),
                    "--n_ladder", "5", "--extra_freqs", "",
                    "--out", str(grid_csv), "--index_out", str(index_csv)],
                   check=True, capture_output=True, cwd=REPO)

    pairs = pd.read_csv(grid_csv)["method_id"].tolist()
    methods = [f"method_{p}" for p in pairs] + [f"method_{p}_counter" for p in pairs]

    # n_agree by pair: how many models get a supra-threshold trough (X = 0.75 uV).
    n_agree_by_pair = {1001: 6, 1002: 4, 1003: 2}

    def trough_by_method(model, method):
        pair = int(method.split("_")[1])
        k = n_agree_by_pair.get(pair, 0)
        if MODELS.index(model) >= k:
            return 0.0                                    # sub-threshold: no dip at all
        # deeper for the lower pair_id so mean_uv orders 1001 < 1002 < 1003
        return 3.0 - 0.5 * (pair - 1001) - 0.05 * MODELS.index(model)

    pred_root = tmp_path / "predictions"
    for i, model in enumerate(MODELS):
        _write_predictions(pred_root, model, methods, trough_by_method, seed=i)

    s7_csv = tmp_path / "s7.csv"
    subprocess.run([sys.executable, str(SCRIPTS / "analyze_mmn_s7_roi.py"),
                    "--predictions_root", str(pred_root), "--out", str(s7_csv)],
                   check=True, capture_output=True, cwd=REPO)
    return dict(s7_csv=s7_csv, index_csv=index_csv, grid_csv=grid_csv, tmp=tmp_path,
                n_agree_by_pair=n_agree_by_pair, n_pairs=len(pairs))


# ──────────────────────────────────────────────────────────
# Ranking criteria
# ──────────────────────────────────────────────────────────

def test_scorer_produces_the_fcz_mtrf_slice_the_search_needs(scored_grid):
    scored = nsc.load_scored(scored_grid["s7_csv"])
    n_inst = 2 * scored_grid["n_pairs"]
    assert len(scored) == n_inst * len(MODELS)
    assert set(scored["model"]) == set(MODELS)
    assert set(scored["direction"]) == {"regular", "counter"}
    # S7 <= S2 is asserted inside analyze_mmn_s7_roi.py; re-check on our slice.
    assert not (scored["s7"] & ~scored["s2"]).any()


def test_n_agree_and_mean_uv_match_the_engineered_fixture(scored_grid):
    ranked = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]),
                                 grid_index=str(scored_grid["index_csv"]))
    assert len(ranked) == 2 * scored_grid["n_pairs"]
    by_pair = ranked.groupby("pair_id")["n_agree"].agg(set)
    for pair, expected in scored_grid["n_agree_by_pair"].items():
        assert by_pair[pair] == {expected}, f"pair {pair}"
    others = set(by_pair.index) - set(scored_grid["n_agree_by_pair"])
    assert all(by_pair[p] == {0} for p in others)

    # mean_uv averages ONLY the agreeing models, and is undefined when none agree.
    top = ranked[ranked.pair_id == 1001].iloc[0]
    agreeing = [top[f"trough_uv__{m}"] for m in MODELS if top[f"s7__{m}"]]
    assert len(agreeing) == 6
    assert top["mean_uv"] == pytest.approx(np.mean(agreeing))
    assert ranked[ranked.n_agree == 0]["mean_uv"].isna().all()


def test_ranking_sorts_by_n_agree_then_most_negative_mean_uv(scored_grid):
    ranked = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]),
                                 grid_index=str(scored_grid["index_csv"]))
    assert list(ranked["rank"]) == list(range(1, len(ranked) + 1))
    assert ranked["n_agree"].is_monotonic_decreasing
    for _, g in ranked.groupby("n_agree"):
        uv = g["mean_uv"].dropna()
        assert uv.is_monotonic_increasing, "within a tier, most negative must rank first"
    assert set(ranked.head(2)["pair_id"]) == {1001}
    assert ranked["mean_uv"].isna().iloc[-1], "n_agree == 0 must sort last"


def test_ranking_is_deterministic_across_runs(scored_grid):
    a = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]))
    b = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]))
    pd.testing.assert_frame_equal(a, b)


def test_load_scored_rejects_an_incomplete_model_set(scored_grid):
    with pytest.raises(SystemExit, match="missing model"):
        nsc.load_scored(scored_grid["s7_csv"], models=MODELS + ["whisper-large"])


# ──────────────────────────────────────────────────────────
# Selection walk
# ──────────────────────────────────────────────────────────

def test_selection_walk_counts_unique_pairs_not_list_positions():
    """The brief's worked example: ranks 1,2 are two pairs' regular directions, 3,4 their
    counters, 5 a third pair -- that is 3 unique pairs across 5 list positions."""
    ranked = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "pair_id": [1001, 1002, 1001, 1002, 1003],
        "direction": ["regular", "regular", "counter", "counter", "regular"],
    })
    assert nsc.selection_walk(ranked, 3)[0] == [1001, 1002, 1003]
    assert nsc.selection_walk(ranked, 2) == ([1001, 1002], 2)


def test_selection_walk_carries_both_directions_including_a_low_ranked_one(scored_grid):
    ranked = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]),
                                 grid_index=str(scored_grid["index_csv"]))
    selected, _ = nsc.selection_walk(ranked, 2)
    dirs = nsc.method_dirs_for_pairs(selected)
    assert dirs == ["method_1001", "method_1001_counter", "method_1002", "method_1002_counter"]
    # every selected pair contributes exactly 2 direction-instances downstream
    assert len(dirs) == 2 * len(selected)


def test_selection_walk_stops_at_the_requested_pair_count(scored_grid):
    ranked = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]))
    for n in (1, 3, 5):
        selected, _ = nsc.selection_walk(ranked, n)
        assert len(selected) == n == len(set(selected))


# ──────────────────────────────────────────────────────────
# Cost model
# ──────────────────────────────────────────────────────────

def test_phase1_cost_is_the_predicted_123_chf():
    """1056 method dirs x 2 clips, one array task each. The per-clip figures were measured on
    16-clip batches, so the model-load overhead they omit is amortised 2 ways here, not 16."""
    base = nsc.extraction_cost_chf(1056, 2, overhead_core_h=0.0)
    with_overhead = nsc.extraction_cost_chf(1056, 2)
    assert base == pytest.approx(113.9, abs=0.5)
    assert with_overhead == pytest.approx(123.4, abs=1.0)


def test_phase2_takes_a_flat_145_pairs():
    """The pair count is a design constant, not a derived quantity -- nothing in the search reads
    a budget or an sacct total to decide it."""
    assert nsc.N_PAIRS_PHASE2 == 145
    assert not hasattr(nsc, "max_pairs_within_budget"), \
        "budget-derived pair selection was removed; the count is a flat constant"
    projected = nsc.extraction_cost_chf(2 * nsc.N_PAIRS_PHASE2, 14)
    assert projected == pytest.approx(221.6, abs=1.0)   # slightly over 220, accepted


def test_cost_scales_with_the_model_subset():
    assert (nsc.extraction_cost_chf(10, 4, models=["whisper-tiny"])
            < nsc.extraction_cost_chf(10, 4, models=nsc.SEARCH_MODELS))


# ──────────────────────────────────────────────────────────
# End-to-end: the two CLI steps that decide what Phase 2 runs
# ──────────────────────────────────────────────────────────

def test_rank_phase1_cli_emits_all_three_artifacts(scored_grid):
    tmp = scored_grid["tmp"]
    out_dir, method_list = tmp / "results", tmp / "methods_phase2.txt"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "rank_novel_phase1.py"),
         "--s7_csv", str(scored_grid["s7_csv"]),
         "--grid_index", str(scored_grid["index_csv"]),
         "--out_dir", str(out_dir), "--method_list_out", str(method_list),
         "--n_pairs", "3", "--literature_s7_csv", str(tmp / "nope.csv")],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr

    ranked = pd.read_csv(out_dir / "phase1_ranked_directions.csv")
    assert len(ranked) == 2 * scored_grid["n_pairs"]
    assert {"rank", "method", "pair_id", "direction", "n_agree", "mean_uv",
            "pct_deviance"} <= set(ranked.columns)
    for m in MODELS:
        assert f"s7__{m}" in ranked.columns and f"trough_uv__{m}" in ranked.columns

    pairs = pd.read_csv(out_dir / "phase2_selected_pairs.csv")
    assert list(pairs["method_id"]) == [1001, 1002, 1003]

    dirs = method_list.read_text().split()
    assert dirs == ["method_1001", "method_1001_counter", "method_1002", "method_1002_counter",
                    "method_1003", "method_1003_counter"]
    assert "n_agree tiers" in r.stdout and "selection walk" in r.stdout


def test_phase2_subset_csv_is_a_verbatim_row_copy(scored_grid, tmp_path):
    """The standard and N7/var1 must regenerate byte-identically from these rows, so any
    per-row edit here would invalidate the Phase-1 features."""
    out_dir = scored_grid["tmp"] / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"method_id": [1003, 1001], "selection_order": [1, 2]}).to_csv(
        out_dir / "phase2_selected_pairs.csv", index=False)
    subset = tmp_path / "subset.csv"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_novel_phase2_csv.py"),
         "--grid_csv", str(scored_grid["grid_csv"]),
         "--selected_pairs", str(out_dir / "phase2_selected_pairs.csv"),
         "--out", str(subset)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr

    full = pd.read_csv(scored_grid["grid_csv"]).set_index("method_id")
    sub = pd.read_csv(subset)
    assert list(sub["method_id"]) == [1003, 1001], "selection order preserved"
    assert list(sub.columns) == bg.COLUMNS
    for mid in (1001, 1003):
        got = sub[sub.method_id == mid].iloc[0].drop("method_id")
        pd.testing.assert_series_equal(got, full.loc[mid], check_names=False)


# ──────────────────────────────────────────────────────────
# Phase-2 ranking: consensus set and rank stability
# ──────────────────────────────────────────────────────────

def _ranked_frame(rows):
    """A minimal ranking frame in the shape rank_directions() emits."""
    df = pd.DataFrame(rows)
    df = df.sort_values(["n_agree", "mean_uv"], ascending=[False, True],
                        na_position="last", kind="mergesort").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def test_consensus_requires_BOTH_directions_at_the_top_tier():
    """One-directional agreement is a frequency preference, not a deviance response -- the
    whole reason the counterbalanced design exists. It must not reach the consensus set."""
    import rank_novel_phase2 as p2
    n = len(MODELS)
    ranked = _ranked_frame([
        # both directions 6/6 -> consensus
        dict(pair_id=1001, direction="regular", n_agree=n, mean_uv=-2.0),
        dict(pair_id=1001, direction="counter", n_agree=n, mean_uv=-1.8),
        # only one direction 6/6 -> excluded
        dict(pair_id=1002, direction="regular", n_agree=n, mean_uv=-3.0),
        dict(pair_id=1002, direction="counter", n_agree=n - 1, mean_uv=-2.9),
        # deeper but neither at top tier -> excluded
        dict(pair_id=1003, direction="regular", n_agree=n - 1, mean_uv=-5.0),
        dict(pair_id=1003, direction="counter", n_agree=n - 1, mean_uv=-5.0),
    ])
    cons = p2.build_consensus(ranked, MODELS)
    assert list(cons["pair_id"]) == [1001]
    assert cons.iloc[0]["mean_uv_both"] == pytest.approx(-1.9)
    assert cons.iloc[0]["direction_gap_uv"] == pytest.approx(0.2)


def test_consensus_is_sorted_strongest_first():
    import rank_novel_phase2 as p2
    n = len(MODELS)
    ranked = _ranked_frame([
        dict(pair_id=p, direction=d, n_agree=n, mean_uv=uv)
        for p, uv in [(1001, -1.0), (1002, -3.0), (1003, -2.0)] for d in ("regular", "counter")
    ])
    cons = p2.build_consensus(ranked, MODELS)
    assert list(cons["pair_id"]) == [1002, 1003, 1001]
    assert cons["mean_uv_both"].is_monotonic_increasing


def test_consensus_is_empty_rather_than_erroring_when_nothing_qualifies():
    import rank_novel_phase2 as p2
    ranked = _ranked_frame([dict(pair_id=1001, direction=d, n_agree=2, mean_uv=-1.0)
                            for d in ("regular", "counter")])
    assert p2.build_consensus(ranked, MODELS).empty


def test_rank_stability_reranks_phase1_within_the_shared_subset(tmp_path, capsys):
    """Phase-1 ranks index the full 1056-instance list; correlating them raw against Phase-2's
    1..N would compare against a population Phase 2 never scored."""
    import rank_novel_phase2 as p2
    pairs = [1001, 1002, 1003, 1004]
    ph1 = pd.DataFrame([
        dict(pair_id=p, direction=d, rank=r, n_agree=6, mean_uv=-1.0)
        for r, (p, d) in enumerate(((p, d) for p in pairs for d in ("regular", "counter")),
                                   start=500)])   # ranks far from 1..N
    ph1_csv = tmp_path / "phase1_ranked_directions.csv"
    ph1.to_csv(ph1_csv, index=False)

    ph2 = _ranked_frame([dict(pair_id=p, direction=d, n_agree=6, mean_uv=-1.0 - 0.1 * i)
                         for i, (p, d) in enumerate((p, d) for p in pairs
                                                    for d in ("regular", "counter"))])
    merged = p2.rank_stability(str(ph1_csv), ph2)
    out = capsys.readouterr().out
    assert "Spearman rho" in out and "n=8 shared" in out
    assert {"phase1_rank", "rank_shift"} <= set(merged.columns)
    assert merged["phase1_rank"].notna().all()


def test_rank_stability_survives_a_missing_phase1_file(tmp_path, capsys):
    import rank_novel_phase2 as p2
    ph2 = _ranked_frame([dict(pair_id=1001, direction="regular", n_agree=6, mean_uv=-1.0)])
    merged = p2.rank_stability(str(tmp_path / "absent.csv"), ph2)
    assert "not found" in capsys.readouterr().out
    assert merged["phase1_rank"].isna().all()


# ──────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────

def _fake_ranking(pairs, seed):
    rng = np.random.default_rng(seed)
    freqs = APPROVED_GRID
    rows = []
    for i, p in enumerate(pairs):
        lo, hi = freqs[i % (len(freqs) - 1)], freqs[(i % (len(freqs) - 1)) + 1]
        for d in ("regular", "counter"):
            n_ag = int(rng.integers(0, len(MODELS) + 1))
            row = dict(pair_id=p, direction=d, n_agree=n_ag,
                       mean_uv=-rng.random() - 0.75 if n_ag else np.nan,
                       f_low=lo, f_high=hi,
                       semitones=12 * np.log2(hi / lo),
                       pct_deviance=100 * (hi / lo - 1))
            for m in MODELS:
                row[f"s7__{m}"] = bool(rng.random() < n_ag / len(MODELS))
                row[f"trough_uv__{m}"] = -rng.random()
            rows.append(row)
    return _ranked_frame(rows)


def test_plot_script_writes_all_three_figures(tmp_path):
    results, figs = tmp_path / "results", tmp_path / "figs"
    results.mkdir()
    pairs = list(range(1001, 1041))
    _fake_ranking(pairs, 1).to_csv(results / "phase1_ranked_directions.csv", index=False)
    _fake_ranking(pairs[:12], 2).to_csv(results / "phase2_final_ranking.csv", index=False)

    script = REPO / "aux/analysis_novel_search/plots/novel_search_plots.py"
    r = subprocess.run([sys.executable, str(script), "--results_dir", str(results),
                        "--out_dir", str(figs)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    for name in ("novel_n_agree_heatmap.png", "novel_deviance_scaling.png",
                 "novel_rank_stability.png"):
        assert (figs / name).stat().st_size > 5000, name


def test_plot_script_runs_on_phase1_alone(tmp_path):
    """Phase 1 lands weeks before Phase 2; the figures must not wait on it."""
    results, figs = tmp_path / "results", tmp_path / "figs"
    results.mkdir()
    _fake_ranking(list(range(1001, 1041)), 1).to_csv(
        results / "phase1_ranked_directions.csv", index=False)
    script = REPO / "aux/analysis_novel_search/plots/novel_search_plots.py"
    r = subprocess.run([sys.executable, str(script), "--results_dir", str(results),
                        "--out_dir", str(figs)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (figs / "novel_n_agree_heatmap.png").exists()
    assert not (figs / "novel_rank_stability.png").exists()
    assert "Phase 2 not scored yet" in r.stdout
