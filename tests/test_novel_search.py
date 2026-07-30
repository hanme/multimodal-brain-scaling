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

APPROVED_GRID = [200, 218, 238, 259, 283, 308, 336, 367, 400, 436, 476, 519, 566, 617,
                 673, 734, 800, 872, 951, 1037, 1131, 1234, 1345, 1467, 1600, 1745, 1903,
                 2075, 2263, 2468, 2691, 2934, 3200, 3490, 3805, 4150, 4525, 4935, 5382,
                 5869, 6400, 6979, 7611]


def test_grid_matches_the_approved_frequency_list_exactly():
    assert bg.build_grid() == APPROVED_GRID


def test_ladder_is_a_uniform_1_5_semitone_step_with_no_extras():
    freqs = bg.build_grid()
    steps = [12 * math.log2(b / a) for a, b in zip(freqs, freqs[1:])]
    # 1.500 st exactly, up to integer-Hz rounding of each rung
    assert max(abs(s - 1.5) for s in steps) < 0.05, steps
    assert len(freqs) == 43
    assert bg.EXTRA_FREQS == (), "the grid is a pure ladder; no irregular rung to caveat"


def test_octaves_of_the_ladder_start_land_exactly_on_the_grid():
    """12 / 1.5 = 8 exactly, so eight steps is an octave and true 2:1 pairs exist. At the 1.494 st
    needed to land 43 points exactly on 7500 Hz, 12/1.494 = 8.03 and no pair would be an octave."""
    freqs = set(bg.build_grid())
    assert {200, 400, 800, 1600, 3200, 6400} <= freqs


def test_top_of_grid_stays_below_nyquist():
    """16 kHz sample rate for every model; a tone at or above 8 kHz would alias. The ladder
    overshoots the nominal 7500 top by one step to keep every step identical."""
    assert 7500 < max(bg.build_grid()) < 8000


def test_grid_rows_are_unordered_pairs_with_no_diagonal():
    freqs = bg.build_grid()
    rows, index = bg.build_rows(freqs)
    bg.verify(freqs, rows)  # raises on duplicates / diagonal / off-grid / id collision / Nyquist
    n = len(freqs)
    assert len(rows) == 903 == len(index) == n * (n - 1) // 2
    assert all(r["standard_freq"] < r["deviant_freq"] for r in rows)
    assert [r["method_id"] for r in rows] == list(range(1001, 1904))
    assert min(i["pct_deviance"] for i in index) == pytest.approx(8.8, abs=0.1)


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
                    "--n_ladder", "5",
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
# Selection: pairs where 5 or 6 of the 6 models agreed
# ──────────────────────────────────────────────────────────

def test_selection_takes_pairs_on_agreement_not_list_position():
    """A pair is in because a direction of it cleared n_agree, not because it landed above some
    rank. The 4/6 pair here sits at rank 2 -- above a qualifying 5/6 pair -- and is still out."""
    ranked = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5, 6],
        "pair_id": [1001, 1002, 1001, 1003, 1002, 1003],
        "direction": ["regular", "regular", "counter", "regular", "counter", "counter"],
        "n_agree": [6, 4, 5, 5, 4, 1],
    })
    selected, n_qual = nsc.select_pairs_by_agreement(ranked, min_agree=5)
    assert selected == [1001, 1003], "1002 tops out at 4/6 and is excluded despite ranking 2nd"
    assert n_qual == 3, "1001 qualifies in both directions, 1003 in one"


def test_selection_carries_both_directions_including_one_below_the_bar():
    """The asymmetric case the counterbalanced design exists to measure: 1003 got in on its
    regular direction alone, and its 1/6 counter is tested anyway."""
    ranked = pd.DataFrame({
        "rank": [1, 2, 3, 4],
        "pair_id": [1001, 1001, 1003, 1003],
        "direction": ["regular", "counter", "regular", "counter"],
        "n_agree": [6, 5, 5, 1],
    })
    selected, _ = nsc.select_pairs_by_agreement(ranked, min_agree=5)
    dirs = nsc.method_dirs_for_pairs(selected)
    assert dirs == ["method_1001", "method_1001_counter", "method_1003", "method_1003_counter"]
    assert len(dirs) == 2 * len(selected)


def test_selection_threshold_is_what_moves_the_pair_count(scored_grid):
    """The fixture's pairs agree in 6, 4 and 2 models. Lowering the bar admits more pairs; the
    count is an output of the criterion, never a target."""
    ranked = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]))
    assert nsc.select_pairs_by_agreement(ranked, 5)[0] == [1001]
    assert nsc.select_pairs_by_agreement(ranked, 4)[0] == [1001, 1002]
    assert nsc.select_pairs_by_agreement(ranked, 2)[0] == [1001, 1002, 1003]
    # nothing reaches 7/6, and an impossible bar selects nothing rather than falling back
    assert nsc.select_pairs_by_agreement(ranked, 7) == ([], 0)


# ──────────────────────────────────────────────────────────
# Cost model
# ──────────────────────────────────────────────────────────

def test_phase1_cost_from_the_literature_per_clip_figures():
    """1806 method dirs x 2 clips, one array task each. These are the LITERATURE-derived
    predictions; jed measured whisper-tiny at 0.38x its prediction, so treat them as an upper
    bound (see scripts/report_extraction_cost.sh)."""
    base = nsc.extraction_cost_chf(1806, 2, overhead_core_h=0.0)
    with_overhead = nsc.extraction_cost_chf(1806, 2)
    assert base == pytest.approx(194.9, abs=1.0)
    assert with_overhead == pytest.approx(211.0, abs=1.5)


def test_phase2_selects_on_agreement_not_a_pair_count():
    """Phase 2 takes the pairs 5 or 6 of the 6 models agreed on. The count that yields is an
    output; nothing in the search reads a budget, an sacct total, or a target pair count."""
    assert nsc.MIN_AGREE_PHASE2 == 5
    assert not hasattr(nsc, "max_pairs_within_budget"), \
        "budget-derived pair selection was removed; selection is on model agreement"
    assert not hasattr(nsc, "N_PAIRS_PHASE2"), \
        "the top-N pair count was removed; it admitted 4/6 pairs to fill a quota"
    # 127 pairs qualified on the 2026-07-27 ranking -- priced here, never used as a threshold.
    projected = nsc.extraction_cost_chf(2 * 127, 14)
    assert projected == pytest.approx(194.1, abs=1.0)


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
         "--min_agree", "2", "--literature_s7_csv", str(tmp / "nope.csv")],
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
    assert "n_agree tiers" in r.stdout and "selection (n_agree >= 2" in r.stdout


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
    """Phase-1 ranks index the full 1806-instance list; correlating them raw against Phase-2's
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


# ──────────────────────────────────────────────────────────
# Phase-1 reporting: agreement tiers, chance baseline, sizing curve
# ──────────────────────────────────────────────────────────

def _tier_frame(per_pair_tiers):
    """A ranking frame with each pair's two directions set to the requested n_agree.

    s7__<model> is filled left-to-right so n_agree is exactly the number requested, which is
    what agreement_tiers / expected_tier_counts read.
    """
    rows = []
    for pair, (n_reg, n_ctr) in per_pair_tiers.items():
        for direction, n in (("regular", n_reg), ("counter", n_ctr)):
            row = dict(pair_id=pair, direction=direction, n_agree=n,
                       mean_uv=(-1.0 - 0.1 * n) if n else np.nan)
            for i, m in enumerate(MODELS):
                row[f"s7__{m}"] = i < n
                row[f"trough_uv__{m}"] = -1.5 if i < n else -0.1
            rows.append(row)
    return _ranked_frame(rows)


def test_agreement_tiers_counts_direction_instances_and_pairs_separately(scored_grid):
    """The fixture's pairs agree symmetrically, so at every tier both-directions == either."""
    ranked = nsc.rank_directions(nsc.load_scored(scored_grid["s7_csv"]),
                                 grid_index=str(scored_grid["index_csv"]))
    tiers = nsc.agreement_tiers(ranked).set_index("n_agree")

    assert list(tiers.index) == list(range(len(MODELS), -1, -1)), "tiers run best-first"
    assert tiers["directions"].sum() == len(ranked)
    assert tiers["pairs_either"].sum() >= ranked["pair_id"].nunique()
    for tier, n_pairs in ((6, 1), (4, 1), (2, 1)):
        assert tiers.loc[tier, "directions"] == 2 * n_pairs
        assert tiers.loc[tier, "pairs_both"] == n_pairs == tiers.loc[tier, "pairs_either"]
    assert tiers.loc[0, "pairs_both"] == scored_grid["n_pairs"] - 3
    assert tiers["pct_directions"].sum() == pytest.approx(100.0)


def test_agreement_tiers_separates_both_from_either_when_the_directions_split():
    """pairs_either - pairs_both IS the direction-asymmetry signal; a tier that only ever
    claims one direction of a pair is measuring a frequency preference."""
    tiers = nsc.agreement_tiers(_tier_frame({
        1001: (6, 6),      # symmetric at the top tier
        1002: (6, 2),      # top tier claims one direction only
        1003: (6, 3),      # ditto
    })).set_index("n_agree")

    assert tiers.loc[6, "directions"] == 4, "1001 both ways + 1002 and 1003 one way each"
    assert tiers.loc[6, "pairs_both"] == 1, "only 1001 reaches the top tier in both directions"
    assert tiers.loc[6, "pairs_either"] == 3
    assert tiers.loc[2, "pairs_both"] == 0 and tiers.loc[2, "pairs_either"] == 1


def test_expected_tier_counts_uses_the_exact_poisson_binomial():
    """The six marginal S7 rates differ by ~2x, so pooling them into one binomial would
    understate the spread and make a 6/6 look rarer -- i.e. more impressive -- than it is."""
    rates = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]        # one model always fires, one never
    n_inst = 10
    rows = []
    for j in range(n_inst):
        row = dict(pair_id=1001 + j // 2, direction="regular" if j % 2 else "counter",
                   n_agree=0, mean_uv=np.nan)
        for m, r in zip(MODELS, rates):
            row[f"s7__{m}"] = j < round(r * n_inst)
            row[f"trough_uv__{m}"] = -1.0
        row["n_agree"] = sum(row[f"s7__{m}"] for m in MODELS)
        rows.append(row)
    ranked = _ranked_frame(rows)

    assert nsc.marginal_s7_rates(ranked).tolist() == pytest.approx(rates)
    exp = nsc.expected_tier_counts(ranked).set_index("n_agree")
    # Both extremes are exact products regardless of how the rates are distributed.
    assert exp.loc[6, "p_chance"] == pytest.approx(np.prod(rates))          # a 0.0 rate -> 0
    assert exp.loc[0, "p_chance"] == pytest.approx(np.prod([1 - r for r in rates]))
    assert exp["p_chance"].sum() == pytest.approx(1.0)
    assert exp["expected"].sum() == pytest.approx(len(ranked))
    assert exp["observed"].sum() == len(ranked)

    # A pooled binomial at the mean rate would put mass at 6/6; the true answer is zero.
    pooled = math.comb(6, 6) * np.mean(rates) ** 6
    assert pooled > 0.01 and exp.loc[6, "p_chance"] == pytest.approx(0.0)


def test_expected_tier_counts_matches_a_plain_binomial_when_the_rates_are_equal():
    """All 2**6 agreement patterns exactly once: every marginal rate is 0.5 and the models are
    independent by construction, so expected must land on Binomial(6, 0.5) AND on the observed
    counts. This is the calibration check -- if it drifts, every 'above chance' claim drifts."""
    rows = []
    for j in range(64):
        row = dict(pair_id=1001 + j // 2, direction="regular" if j % 2 else "counter",
                   mean_uv=np.nan)
        for i, m in enumerate(MODELS):
            row[f"s7__{m}"] = bool(j >> i & 1)
            row[f"trough_uv__{m}"] = -1.0
        row["n_agree"] = sum(row[f"s7__{m}"] for m in MODELS)
        rows.append(row)
    ranked = _ranked_frame(rows)

    assert nsc.marginal_s7_rates(ranked).tolist() == pytest.approx([0.5] * 6)
    exp = nsc.expected_tier_counts(ranked).set_index("n_agree")
    for k in range(7):
        assert exp.loc[k, "p_chance"] == pytest.approx(math.comb(6, k) / 64)
        assert exp.loc[k, "expected"] == pytest.approx(math.comb(6, k))
        assert exp.loc[k, "observed"] == math.comb(6, k), "observed must sit exactly on chance"


def test_pair_cutoff_curve_counts_pairs_not_direction_instances():
    """Phase 2 is priced per PAIR: selection_walk carries both directions of anything it picks,
    so a pair qualifying twice costs the same as one qualifying once."""
    ranked = _tier_frame({1001: (6, 6), 1002: (6, 0), 1003: (5, 5)})
    curve = nsc.pair_cutoff_curve(ranked, thresholds=[6], uv_cutoffs=[0.0],
                                  chf_per_pair=2.0).iloc[0]
    assert curve.directions == 3, "1001 twice + 1002 once"
    assert curve.pairs == 2, "but only two distinct pairs to extract"
    assert curve.chf == pytest.approx(4.0)


def test_pair_cutoff_curve_is_monotone_in_both_cutoffs():
    ranked = _tier_frame({1001 + i: (i % 7, (i + 3) % 7) for i in range(40)})
    curve = nsc.pair_cutoff_curve(ranked, uv_cutoffs=[-1.6, -1.3, -1.0, 0.0])
    for _, g in curve.groupby("n_agree_min"):
        assert g.sort_values("mean_uv_max")["pairs"].is_monotonic_increasing, \
            "relaxing the uV cutoff can only add pairs"
    for _, g in curve.groupby("mean_uv_max"):
        assert g.sort_values("n_agree_min")["pairs"].is_monotonic_decreasing, \
            "raising the agreement floor can only remove pairs"
    assert (curve["chf"] == curve["pairs"] * nsc.CHF_PER_PAIR_PHASE2).all()


def test_pair_cutoff_curve_never_admits_an_undefined_mean_uv():
    """n_agree == 0 leaves mean_uv NaN; a NaN must not slip past a comparison as 'qualifying'."""
    ranked = _tier_frame({1001: (0, 0), 1002: (4, 4)})
    curve = nsc.pair_cutoff_curve(ranked, thresholds=[1], uv_cutoffs=[0.0]).iloc[0]
    assert curve.pairs == 1 and curve.directions == 2


def test_direction_gap_reports_each_pair_once_with_both_directions():
    gaps = nsc.direction_gap(_tier_frame({
        1001: (6, 6), 1002: (6, 1), 1003: (0, 0),
    })).set_index("pair_id")
    assert len(gaps) == 3
    assert gaps.loc[1001, "n_agree_gap"] == 0
    assert gaps.loc[1002, "n_agree_gap"] == 5
    assert gaps.loc[1002, "n_agree_max"] == 6 and gaps.loc[1002, "n_agree_min"] == 1
    assert np.isnan(gaps.loc[1003, "mean_uv_regular"])


def test_phase2_pair_cost_is_the_measured_rate_not_the_literature_projection():
    """The literature per-clip table ran pessimistic for every model, so the Phase-2 pair price
    is taken from Phase 1's actual spend. The projection is kept for comparison only."""
    projected_per_pair = nsc.extraction_cost_chf(2, 14) / 1
    assert nsc.CHF_PER_PAIR_PHASE2 == pytest.approx(1.325)
    assert nsc.CHF_PER_PAIR_PHASE2 < projected_per_pair, \
        "measured must undercut the literature projection; if not, re-read report_extraction_cost"


def test_phase1_results_script_emits_every_table_and_figure(scored_grid, tmp_path):
    """End to end on the synthetic grid: the memo cites these filenames, so their absence is a
    broken memo, not just a missing plot."""
    results, figs = tmp_path / "results", tmp_path / "figs"
    results.mkdir()
    (results / "phase1_mmn_s7_roi.csv").write_text(
        Path(scored_grid["s7_csv"]).read_text())
    (results / "grid_index.csv").write_text(Path(scored_grid["index_csv"]).read_text())

    script = REPO / "aux/analysis_novel_search/plots/phase1_results.py"
    r = subprocess.run(
        [sys.executable, str(script), "--results_dir", str(results), "--out_dir", str(figs),
         "--predictions_root", str(scored_grid["tmp"] / "predictions"),
         # the fixture's own scored CSV stands in for the literature screen: same schema, same
         # 6 models, so the comparison path is exercised rather than skipped
         "--literature_csv", str(scored_grid["s7_csv"])],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr

    for name in ("phase1_agreement_tiers.csv", "phase1_top30.csv", "phase1_cutoff_curve.csv",
                 "phase1_chance_baseline.csv", "phase1_novel_vs_literature.csv",
                 "phase1_direction_asymmetry.csv", "phase1_model_uv_box.csv",
                 "phase1_mean_uv_grid.csv", "phase1_frequency_stripes.csv"):
        assert (figs / name).stat().st_size > 0, name
    # One per-model figure per model whose committed layer the fixture actually wrote.
    for m in [m for m in MODELS if m in ("whisper-tiny", "whisper-base")]:
        assert (figs / f"phase1_strong_waveforms_1__{m}.png").stat().st_size > 5000, m
    for name in ("phase1_strong_waveforms_1.png", "phase1_direction_waveforms_1.png",
                 "phase1_model_uv_box.png", "phase1_mean_uv_heatmap.png",
                 "phase1_ranking_structure.png", "phase1_novel_vs_literature.png"):
        assert (figs / name).stat().st_size > 5000, name

    # The fixture engineers pair 1001 to 6/6 in both directions and nothing else above 4.
    tiers = pd.read_csv(figs / "phase1_agreement_tiers.csv").set_index("n_agree")
    assert tiers.loc[6, "directions__x0.75"] == 2
    assert tiers.loc[6, "pairs_both__x0.75"] == 1
    top = pd.read_csv(figs / "phase1_top30.csv")
    assert top.iloc[0]["n_agree"] == len(MODELS) and top.iloc[0]["models_not_agreeing"] != \
        top.iloc[0]["models_not_agreeing"], "6/6 must leave the not-agreeing column empty (NaN)"


def test_phase1_results_script_refuses_a_short_slice(scored_grid, tmp_path):
    """A silently truncated s7 CSV would move every count in the memo, so it must abort."""
    results, figs = tmp_path / "results", tmp_path / "figs"
    results.mkdir()
    df = pd.read_csv(scored_grid["s7_csv"])
    df[df["method"] != "method_1001_counter"].to_csv(results / "phase1_mmn_s7_roi.csv",
                                                     index=False)
    (results / "grid_index.csv").write_text(Path(scored_grid["index_csv"]).read_text())

    script = REPO / "aux/analysis_novel_search/plots/phase1_results.py"
    r = subprocess.run([sys.executable, str(script), "--results_dir", str(results),
                        "--out_dir", str(figs), "--skip_waveforms"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0
    assert "do not have both directions" in r.stderr or "expected" in r.stderr


def test_direction_waveform_uv_variant_does_not_overwrite_the_z_figures(scored_grid, tmp_path):
    """The memo links the z figures. A --wave_units uv re-run must land beside them, not on
    top of them -- otherwise the memo silently starts showing raw cross-model µV means, which
    Caveat 2 says are not comparable."""
    results, figs = tmp_path / "results", tmp_path / "figs"
    results.mkdir()
    (results / "phase1_mmn_s7_roi.csv").write_text(Path(scored_grid["s7_csv"]).read_text())
    (results / "grid_index.csv").write_text(Path(scored_grid["index_csv"]).read_text())
    script = REPO / "aux/analysis_novel_search/plots/phase1_results.py"
    base = [sys.executable, str(script), "--results_dir", str(results), "--out_dir", str(figs),
            "--predictions_root", str(scored_grid["tmp"] / "predictions"),
            "--literature_csv", str(tmp_path / "nope.csv")]

    assert subprocess.run(base, capture_output=True, text=True, cwd=REPO).returncode == 0
    z_bytes = (figs / "phase1_direction_waveforms_1.png").read_bytes()

    r = subprocess.run(base + ["--wave_units", "uv"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (figs / "phase1_direction_waveforms_uv_1.png").stat().st_size > 5000
    assert (figs / "phase1_direction_waveforms_1.png").read_bytes() == z_bytes


def test_fcz_waves_carry_both_the_uv_and_the_z_trace(scored_grid):
    """The per-model panels need µV; the direction-collapsed panels need z, because only z is
    normalised per model and so safe to average across them."""
    sys.path.insert(0, str(REPO / "aux/analysis_novel_search/plots"))
    import phase1_results as pr

    # The fixture writes every model's HDF5 at the default layer, so only the models whose
    # COMMITTED layer is blocks.0 are findable here; the loader skips the rest by design.
    present = [m for m in MODELS if pr.LAYER[m] == "blocks.0"]
    assert present, "fixture layout no longer matches any committed layer"

    waves = pr.load_fcz_waves([1001], present,
                              predictions_root=scored_grid["tmp"] / "predictions")
    assert set(waves) == {(1001, "regular"), (1001, "counter")}
    for per_model in waves.values():
        assert set(per_model) == set(present)
        for t, uv, z in per_model.values():
            assert t.shape == uv.shape == z.shape
            # The fixture's dip is ~3 uV deep; z is baseline-normalised, so the two traces are
            # on genuinely different scales rather than one being a copy of the other.
            assert not np.allclose(uv, z)
            assert np.isfinite(uv).all() and np.isfinite(z).all()
