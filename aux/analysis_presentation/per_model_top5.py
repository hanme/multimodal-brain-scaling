#!/usr/bin/env python
"""Per-model head-to-head: each model's OWN top-5 literature vs its OWN top-5 Phase-2 stimuli.

WHY THIS EXISTS, AND HOW IT DIFFERS FROM THE COMMITTED FIGURES
--------------------------------------------------------------
`aux/analysis_novel_search/plots/{literature,phase1}_results.py` already emit per-model waveform
figures (`literature_waveforms__<model>.png`, `phase2_waveforms__<model>.png`). Those de-overlay
the CONSENSUS ranking: every model's figure shows the SAME ten stimuli, the ones the six models
jointly ranked highest (`rank_directions`: n_agree desc, then mean_uv over the agreeing models).
A model that dislikes the consensus top 10 is shown its own weakest responses.

This module ranks each model by ITS OWN trough depth at FCz, so a model's row is the five stimuli
that model actually responds to most strongly. Two different questions; both are kept.

Nothing is re-scored. The traces come from the committed prediction HDF5s through
`phase1_results.load_fcz_waves`, and the shape statistics through `analyze_mmn_criteria`'s
`trace_stats`/`decide` at exactly the knobs `analyze_mmn_s7_roi.py` commits to. Every derived
`trough_uv` is asserted against the committed scored CSV before anything is drawn.

Fixed reporting choices, inherited (never restated) from `novel_search_common`: FCz electrode,
mTRF read-out, S7 = S2 AND trough <= -0.75 uV, six models with whisper-large dropped at load.

Outputs (figures follow this directory's existing tree: plots/<subdir>/ mirrored by svgs/<subdir>/,
the convention trough_distributions.py already uses here):
  plots/per_model_top5/<model>.png   2 rows x 5 cols: row A = that model's top-5 literature,
  svgs/per_model_top5/<model>.svg    row B = its top-5 Phase-2 difference waves
  per_model_top5_comparison.csv      every panel's row: model, set, rank, method_id, direction,
                                     f_standard, f_deviant, semitones, trough_uv, latency_ms,
                                     recovery_frac, s7_pass (+ the shape columns)
  per_model_full_ranking.csv         the un-truncated per-model ranking behind those top 5s
  model_choice_evidence.csv          one row per model: the numbers behind the model choice

Usage:
    PYTHONPATH="$PWD/src:$PWD/scripts" conda run -n mbs-env \
        python aux/analysis_presentation/per_model_top5.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLOTS = REPO / "aux/analysis_novel_search/plots"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(PLOTS))

from novel_search_common import (                                      # noqa: E402
    SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD, load_scored, rank_directions,
)
# The trace construction, the layer map, the x-window and the scoring window are IMPORTED, never
# restated: a divergence between this figure and the committed ones would read as a result.
from phase1_results import (                                           # noqa: E402
    LAYER, WAVE_XLIM, MMN_WINDOW, WAVE_MODEL_STYLE, load_fcz_waves, load_literature,
    _with_std_dev, _padded,
)
from novel_search_plots import MLABEL, INK, MUTED                      # noqa: E402
from analyze_mmn_criteria import trace_stats, decide                   # noqa: E402
# The exact committed criteria knobs. Imported so a change there propagates here.
from analyze_mmn_s7_roi import (                                       # noqa: E402
    WINDOW, RECOVERY_MS, RECOVERY_FRAC, CTRL_WINDOW, EDGE_GUARD_BINS,
)
from analyze_mmn_screen_24freq import load_semitones                   # noqa: E402

# ── Sources ────────────────────────────────────────────────────────────────────────────────
# LIT: the 24 literature methods x {regular, counter}. results_soafix_full is the committed
# screen the realness checks read; its (FCz, mtrf, X=0.75) slice is byte-identical to
# results_soafix/mmn_s7_roi.csv, which literature_results.py uses -- verified, not assumed.
LIT_CSV = REPO / "outputs/results_soafix_full/mmn_s7_roi.csv"
LIT_PREDICTIONS = REPO / "outputs/insilico_mmn_predictions_soafix"
LIT_META = REPO / "data/metadata/literature_frequency_intensity_duration_metadata.csv"
# P2: the 127 pairs that reached n_agree >= 5 in Phase 1, both directions, 15 deviants each.
P2_CSV = REPO / "outputs/results_novel_search/phase2_mmn_s7_roi.csv"
P2_PREDICTIONS = REPO / "outputs/insilico_mmn_predictions_novel_phase2"
P2_GRID = REPO / "outputs/results_novel_search/grid_index.csv"

SETS = ("literature", "phase2")
SET_LABEL = {"literature": "literature", "phase2": "model-selected (Phase 2)"}
# Both screens average 15 deviant realizations per condition. Asserted per method group, not just
# on the first one -- a condition short of 15 is excluded and named rather than quietly plotted.
N_DEVIANTS = 15
N_CONDITIONS = {"literature": 48, "phase2": 254}
N_TOP = 5                              # panel size, per row (fixed during the call)
# Figures are filed the way this directory already files them: plots/<subdir>/ and a byte-mirrored
# svgs/<subdir>/. The deck needs the SVG -- the current head-to-head raster is soft when projected.
FIG_SUBDIR = "per_model_top5"

# Held-out mean test r at the ELECTRODE level for the committed mapping layer, read from the
# sweep's own JSON. The two wav2vec2 sweeps' JSONs are not in this checkout (see MISSING_R_NOTE),
# so those two cells are reported as unavailable rather than guessed.
MAPPING_JSON = REPO / "outputs/results/eeg_mapping/{model}__electrodes__D2.json"
MISSING_R_NOTE = ("not in this checkout: outputs/results/eeg_mapping/{model}__electrodes__D2.json "
                  "(regenerate from the cluster, see aux/handoff_enable_large_wav2vec2_models.md)")

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.family": "DejaVu Sans",
    "svg.hashsalt": "per-model-top5",       # deterministic SVG ids; see realness-checks §18.6
})


def save_both(fig, out_dir, stem, svg_dir=None):
    """PNG under plots/<subdir>/ and the same render as SVG under svgs/<subdir>/.

    `metadata={"Date": None}` and the fixed hashsalt above make the SVG byte-reproducible, so a
    re-render of unchanged data does not rewrite a tracked file with thousands of no-op lines.
    """
    png = Path(out_dir) / "plots" / FIG_SUBDIR / f"{stem}.png"
    svg = (Path(svg_dir) if svg_dir else Path(out_dir) / "svgs") / FIG_SUBDIR / f"{stem}.svg"
    for path in (png, svg):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    return png


# ──────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────
def literature_conditions(models):
    """The 48 literature direction-instances with f_std/f_dev/semitones and per-model S7+trough.

    `load_literature` owns the direction convention (the metadata's standard/deviant describe the
    REGULAR direction, so a `_counter` instance has them swapped). Its consensus RANK is carried
    along for reference but is not what this module sorts by.
    """
    lit = load_literature(models, LIT_CSV, LIT_META)
    if lit is None:
        raise SystemExit(f"no literature screen at {LIT_CSV}")
    st = load_semitones(LIT_META)
    lit["semitones"] = lit["pair_id"].map(st)
    if lit["semitones"].isna().any():
        bad = sorted(lit.loc[lit["semitones"].isna(), "pair_id"].unique())
        raise SystemExit(f"no semitone entry in {LIT_META.name} for method_id {bad}")
    return lit.rename(columns={"rank": "consensus_rank"})


def phase2_conditions(models):
    """The 254 Phase-2 direction-instances, same columns, from the search's own ranking code."""
    scored = load_scored(P2_CSV, models=models, roi=ROI, mapping=MAPPING, x=DIP_UV_THRESHOLD)
    p2 = _with_std_dev(rank_directions(scored, grid_index=str(P2_GRID), models=models))
    return p2.rename(columns={"rank": "consensus_rank"})


def n_deviants_per_condition(models, predictions_root):
    """{(pair_id, direction): {model: n_deviants}} straight off the HDF5 group attrs.

    `load_fcz_waves` gates on the FIRST method group's `n_deviants` only; the validation this
    module owes ("every panel is a mean of 15") is per condition, so the attribute is read for
    all of them.
    """
    out = {}
    for m in models:
        path = Path(predictions_root) / m / f"electrode_predictions__{LAYER[m]}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as h5:
            for key, g in h5.items():
                if not isinstance(g, h5py.Group) or "n_deviants" not in g.attrs:
                    continue
                direction = "counter" if key.endswith("_counter") else "regular"
                pid = int(key.replace("_counter", "").split("_")[1])
                out.setdefault((pid, direction), {})[m] = int(g.attrs["n_deviants"])
    return out


def score_conditions(conds, models, predictions_root, set_name):
    """One row per (model, condition): trough, latency, recovery and the shape proxies.

    Shape statistics are computed on the Z trace, because that is the trace the S2 verdict is
    defined on; `trough_uv` is then read off the uV trace at that argmin, exactly as
    `analyze_mmn_s7_roi.collect_rows` does. Both traces come from `load_fcz_waves`, so the numbers
    here are the committed pipeline's, re-read rather than recomputed differently.
    """
    pair_ids = conds["pair_id"].unique().tolist()
    waves = load_fcz_waves(pair_ids, models, predictions_root, roi=ROI,
                           expect_n_deviants=N_DEVIANTS)
    if not waves:
        raise SystemExit(f"no prediction HDF5s readable under {predictions_root}")
    ndev = n_deviants_per_condition(models, predictions_root)

    lo, hi = WINDOW
    rows = []
    for _, c in conds.iterrows():
        pid, direction = int(c["pair_id"]), c["direction"]
        for m in models:
            got = waves.get((pid, direction), {}).get(m)
            if got is None:
                continue
            t, uv, z = got
            s = trace_stats(t, z, lo, hi, RECOVERY_MS, RECOVERY_FRAC,
                            CTRL_WINDOW[0], CTRL_WINDOW[1], EDGE_GUARD_BINS)
            argmin_ms = s.get("argmin_ms", float("nan"))
            trough_uv = (float("nan") if np.isnan(argmin_ms)
                         else float(uv[int(np.argmin(np.abs(t - argmin_ms)))]))
            d = decide(s)
            rows.append(dict(
                model=m, set=set_name, method=c["method"], method_id=pid, direction=direction,
                f_standard=float(c["f_std"]), f_deviant=float(c["f_dev"]),
                semitones=float(c["semitones"]), consensus_rank=int(c["consensus_rank"]),
                n_agree=int(c["n_agree"]), n_deviants=ndev.get((pid, direction), {}).get(m, -1),
                trough_uv=trough_uv, latency_ms=float(argmin_ms),
                recovery_frac=float(s.get("recovery_frac", 0.0)),
                s2_pass=bool(d["S2_recovery"]),
                s7_pass=bool(c.get(f"s7__{m}", False)),
                trough_uv_committed=float(c.get(f"trough_uv__{m}", float("nan"))),
                # --- shape quality: is the post-trough recovery real, or a ramp still falling? ---
                # S4 in `decide`: the in-window trough is more negative than the deepest point of
                # the 300-440 ms control window. FALSE = the trace's true minimum is later, i.e.
                # what got scored at 100-240 ms is the shoulder of a continuing downslope. This is
                # the committed criterion, not a new one invented for this figure.
                s4_specificity=bool(d["S4_specificity"]),
                ctrl_min_uv=float(uv[(t >= CTRL_WINDOW[0]) & (t <= CTRL_WINDOW[1])].min()),
                post_slope_uv_per_100ms=_post_trough_slope(t, uv, argmin_ms),
            ))
    return pd.DataFrame(rows)


def _post_trough_slope(t, uv, argmin_ms, end_ms=WAVE_XLIM[1]):
    """OLS slope of the uV trace from the trough to the end of the drawn window, per 100 ms.

    POSITIVE = the trace climbs back toward baseline over the whole readout (a clean dip-and-
    recover). NEGATIVE = it is still, on balance, heading down after the scored trough -- the
    "down-sloping effect afterwards" that drew the objection to one literature panel. This is a
    reading aid over the full window; `recovery_frac` (120 ms) and `s4_specificity` are the
    criteria. Fit over the drawn window and no further, so the number matches what the eye sees.
    """
    if np.isnan(argmin_ms):
        return float("nan")
    sel = (t >= argmin_ms) & (t <= end_ms)
    if sel.sum() < 3:
        return float("nan")
    slope = np.polyfit(t[sel], uv[sel], 1)[0]
    return float(slope * 100.0)


def build_table(models):
    """The tidy (model x condition) frame for both sets, with every documented check run."""
    frames = []
    for set_name, conds, root in (("literature", literature_conditions(models), LIT_PREDICTIONS),
                                  ("phase2", phase2_conditions(models), P2_PREDICTIONS)):
        n = conds["pair_id"].nunique()
        assert len(conds) == N_CONDITIONS[set_name], \
            f"{set_name}: {len(conds)} conditions, expected {N_CONDITIONS[set_name]}"
        # Every regular condition has its counter counterpart, and direction is a column rather
        # than something collapsed into the trough.
        by_dir = conds.groupby("direction")["pair_id"].apply(set)
        assert by_dir["regular"] == by_dir["counter"], \
            f"{set_name}: regular and counter cover different pairs"
        assert len(conds) == 2 * n, f"{set_name}: {len(conds)} instances over {n} pairs"
        print(f"  {set_name}: {len(conds)} direction-instances over {n} pairs")
        frames.append(score_conditions(conds, models, root, set_name))
    tidy = pd.concat(frames, ignore_index=True)
    # A condition whose mean is over fewer than 15 deviants is not comparable to the rest and is
    # DROPPED, not annotated -- it would otherwise be eligible for a top-5 slot on a shallower
    # average. Named on the way out so the exclusion is never silent.
    short = tidy[tidy["n_deviants"] != N_DEVIANTS]
    if len(short):
        print(f"  EXCLUDED {len(short)} cells with n_deviants != {N_DEVIANTS}: "
              f"{sorted(set(zip(short['set'], short['method'], short['model'])))}")
        tidy = tidy[tidy["n_deviants"] == N_DEVIANTS].reset_index(drop=True)
    _validate(tidy, models)
    return tidy


def _validate(tidy, models):
    """Assertions the figure must not be drawn without. Each failure is a real data problem."""
    assert sorted(tidy["model"].unique()) == sorted(models), \
        f"models present: {sorted(tidy['model'].unique())}"
    assert len(models) == 6, f"{len(models)} models, expected exactly 6"
    assert "whisper-large" not in set(tidy["model"]), "whisper-large leaked in"

    for set_name in SETS:
        d = tidy[tidy["set"] == set_name]
        assert len(d) == N_CONDITIONS[set_name] * len(models), \
            f"{set_name}: {len(d)} rows, expected {N_CONDITIONS[set_name] * len(models)}"

    # The derived trough must reproduce the committed scored CSV. This is the check that says the
    # traces being drawn are the traces that were ranked.
    delta = (tidy["trough_uv"] - tidy["trough_uv_committed"]).abs()
    worst = float(delta.max())
    # The CSV rounds to 5 decimals, so 1e-5 is the floor of agreement it can express.
    assert worst < 2e-5, (f"derived trough_uv disagrees with the committed CSV by up to {worst:.2e} "
                          f"uV -- the prediction HDF5s and the scored table are different vintages")
    print(f"  trough_uv reproduces the committed CSVs to {worst:.2e} uV over {len(tidy)} rows")

    assert (tidy["n_deviants"] == N_DEVIANTS).all(), "short conditions survived the exclusion"
    print(f"  every cell is a mean of {N_DEVIANTS} deviants")

    # Sign convention. A "top" stimulus with a positive trough would mean the ranking is picking
    # the least-positive excursion of a set with no MMN in it -- reported, never silently ranked.
    for m in models:
        for set_name in SETS:
            d = tidy[(tidy["model"] == m) & (tidy["set"] == set_name)]
            top = d.nsmallest(N_TOP, "trough_uv")
            bad = top[top["trough_uv"] >= 0]
            if len(bad):
                raise SystemExit(
                    f"STOP: {m}/{set_name} has {len(bad)} of its top {N_TOP} at a POSITIVE "
                    f"trough ({bad['trough_uv'].tolist()}). A positive trough is not an MMN; the "
                    f"ranking would be reporting the shallowest non-response as the best.")
    print("  sign convention OK: every model's top 5 is negative in both sets")

    # Continuity with the existing across-model figure: its headline model-selected pairs must
    # still surface here. Reported, not asserted -- a per-model ranking is entitled to disagree,
    # but a set that shared NOTHING with the consensus view would mean the wrong root was read.
    for f_std, f_dev in ((2263, 7611), (1600, 3200)):
        hit = [m for m in models
               if ((rank_for_model(tidy, m, "phase2")["f_standard"] == f_std)
                   & (rank_for_model(tidy, m, "phase2")["f_deviant"] == f_dev)).any()]
        print(f"  {f_std:g}→{f_dev:g} Hz in the Phase-2 top {N_TOP} of: "
              f"{', '.join(hit) if hit else 'NO MODEL'}")


# ──────────────────────────────────────────────────────────
# Per-model ranking and the de-duplication rule
# ──────────────────────────────────────────────────────────
def rank_for_model(tidy, model, set_name, dedupe=False, n=N_TOP):
    """That model's own ranking of one set: trough_uv ascending (most negative wins).

    DE-DUPLICATION (literature only, `dedupe=True`). Ten of the 24 literature methods are
    1000 -> 1200 Hz and five are 1000 -> 1850 Hz: distinct papers that share a frequency pair and
    differ only in SOA, tone duration or intensity. Undeduplicated, a top-5 row spends three of
    its five slots on near-identical traces of the same tone pair -- which is what the current
    across-model figure does, and it is the first thing a reader asks about.

    The rule here: **five distinct ORDERED frequency pairs**, keeping the deepest-troughing
    instance of each. Ordered, so 1000 -> 1850 and 1850 -> 1000 stay two different stimuli --
    collapsing them would erase the direction asymmetry the counterbalancing exists to measure.
    Applied to the literature row ONLY; the Phase-2 grid has no duplicate pairs by construction
    (`build_novel_grid_csv.py` asserts it), so the rule is a no-op there and is left off to keep
    that visible. The undeduplicated ranking is written out in full to per_model_full_ranking.csv,
    so what the rule removed is auditable rather than invisible.
    """
    d = tidy[(tidy["model"] == model) & (tidy["set"] == set_name)].copy()
    d = d.sort_values(["trough_uv", "method_id", "direction"], kind="mergesort")
    if dedupe:
        d = d.drop_duplicates(subset=["f_standard", "f_deviant"], keep="first")
    d = d.head(n).copy()
    d["rank"] = np.arange(1, len(d) + 1)
    return d


def cross_check_committed(tidy, models, n_check=2):
    """Reproduce the committed ranking tables' µV columns from these traces, for `n_check` models.

    `literature_top48.csv` / `phase2_top50.csv` carry no per-model trough column, so the check is
    against what they DO carry: mean/median/min/max of `trough_uv` over all six models, for the
    methods in a model's own top 5. Agreeing there means the traces behind this figure and the
    traces behind the committed ranking are the same numbers. Discrepancies are REPORTED, never
    reconciled.
    """
    tables = {"literature": PLOTS / "literature_top48.csv", "phase2": PLOTS / "phase2_top50.csv"}
    checked = failures = 0
    for set_name, path in tables.items():
        if not path.exists():
            print(f"  cross-check skipped: {path} not found")
            continue
        ref = pd.read_csv(path).set_index("method")
        for m in models[:n_check]:
            top = rank_for_model(tidy, m, set_name, dedupe=(set_name == "literature"))
            for _, r in top.iterrows():
                if r["method"] not in ref.index:
                    continue                       # phase2_top50 only covers ranks 1-50
                row = ref.loc[r["method"]]
                mine = tidy[(tidy["set"] == set_name) & (tidy["method"] == r["method"])]
                mine = mine.set_index("model").reindex(models)["trough_uv"]
                for col, val in (("mean_uv_all6", mine.mean()), ("median_uv_all6", mine.median()),
                                 ("min_uv_all6", mine.min()), ("max_uv_all6", mine.max())):
                    checked += 1
                    # The committed tables are rounded to 3 decimals.
                    if abs(float(row[col]) - float(val)) > 5.1e-4:
                        failures += 1
                        print(f"  DISCREPANCY {set_name}/{m}/{r['method']}/{col}: "
                              f"committed {float(row[col]):+.3f} vs derived {float(val):+.3f}")
    verdict = "MATCH" if not failures else f"{failures} DISCREPANCIES"
    print(f"  committed-table cross-check over {models[:n_check]}: {checked} values, {verdict}")
    return failures


# ──────────────────────────────────────────────────────────
# The figure
# ──────────────────────────────────────────────────────────
def _panel_title(r):
    """Frequency pair, direction, trough and this model's own S7 verdict -- four facts, no more."""
    arrow = f"{r['f_standard']:g} → {r['f_deviant']:g} Hz"
    verdict = "✓" if r["s7_pass"] else "✗"
    return (f"{r['rank']:.0f}. {arrow}  ({r['direction']})\n"
            f"{r['trough_uv']:+.2f} µV @ {r['latency_ms']:.0f} ms   "
            f"S7@{DIP_UV_THRESHOLD:g} {verdict}")


def figure_for_model(tidy, model, models, out_dir, waves, share_y=True, n=N_TOP, svg_dir=None):
    """One 2 x `n` figure: row A = this model's top-n literature, row B = its top-n Phase-2."""
    rows = {s: rank_for_model(tidy, model, s, dedupe=(s == "literature"), n=n) for s in SETS}
    colour = WAVE_MODEL_STYLE[model]["color"]
    lo_w, hi_w = MMN_WINDOW

    fig, axes = plt.subplots(2, n, figsize=(2.75 * n, 5.3), sharex=True, sharey=share_y)
    ylim = None
    if share_y:
        # One y-window for the whole figure, from the traces it actually draws over the drawn
        # x-window -- not from the full epoch, which would leave the panels floating in whitespace.
        vals = []
        for s in SETS:
            for _, r in rows[s].iterrows():
                got = waves[s].get((int(r["method_id"]), r["direction"]), {}).get(model)
                if got is not None:
                    t, uv, _z = got
                    vals.append(uv[(t >= WAVE_XLIM[0]) & (t <= WAVE_XLIM[1])])
        if vals:
            allv = np.concatenate(vals)
            pad = 0.08 * (allv.max() - allv.min())
            ylim = (allv.min() - pad, allv.max() + pad)

    for i, s in enumerate(SETS):
        for j, (_, r) in enumerate(rows[s].iterrows()):
            ax = axes[i, j]
            ax.axvspan(lo_w, hi_w, color="#eef2f7", zorder=0)
            ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
            got = waves[s].get((int(r["method_id"]), r["direction"]), {}).get(model)
            if got is not None:
                t, uv, _z = got
                sel = (t >= WAVE_XLIM[0]) & (t <= WAVE_XLIM[1])
                ax.plot(t[sel], uv[sel], color=colour, lw=1.6, zorder=3)
                # The scored trough, marked so the annotated number is locatable on the trace.
                ax.plot([r["latency_ms"]], [r["trough_uv"]], marker="v", ms=5, color=INK,
                        zorder=4, clip_on=False)
            ax.set_title(_panel_title(r), fontsize=7.6, pad=4,
                         color=INK if r["s7_pass"] else "#b03030")
            ax.set_xlim(*WAVE_XLIM)
            if ylim:
                ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7)
            if i == 1:
                ax.set_xlabel("time from final tone onset (ms)", fontsize=8)
        axes[i, 0].set_ylabel(f"{'A' if i == 0 else 'B'}. {SET_LABEL[s]}\nµV", fontsize=9)
        for j in range(len(rows[s]), n):
            axes[i, j].set_visible(False)

    fig.suptitle(
        f"{MLABEL[model]} — FCz difference wave (mean of 15 deviants − standard), mTRF\n"
        f"Row A: this model's own top {n} literature conditions.  "
        f"Row B: its own top {n} model-selected (Phase-2) conditions.\n"
        f"Both rows ranked by THIS model's trough depth in the 100–240 ms window — not by "
        f"cross-model agreement.", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.055, 1, 0.885), h_pad=1.6)
    # tight_layout leaves a band of dead space under a multi-line suptitle on a 2-row grid; the
    # explicit top reclaims it so the panels, not the whitespace, carry the figure.
    fig.subplots_adjust(top=0.845)
    fig.text(0.5, 0.036,
             f"Shaded band = the {lo_w:g}–{hi_w:g} ms scoring window; ▾ marks the scored trough. "
             f"S7 = recovers ≥{RECOVERY_FRAC:.0%} of its depth within {RECOVERY_MS:g} ms AND "
             f"trough ≤ −{DIP_UV_THRESHOLD:g} µV; a red title is a fail.\n"
             f"Row A shows 5 distinct frequency pairs (the deepest instance of each); ten "
             f"literature methods share the 1000→1200 Hz pair and five share 1000→1850 Hz. "
             f"{'Shared' if share_y else 'Free'} y-axis within this figure; "
             f"NEVER comparable across models.",
             ha="center", va="top", fontsize=7.6, color=MUTED, linespacing=1.5)
    png = save_both(fig, out_dir, model, svg_dir)
    plt.close(fig)
    print(f"  wrote plots/{FIG_SUBDIR}/{png.name} (+ svg)")
    return png


# ──────────────────────────────────────────────────────────
# Model-choice evidence
# ──────────────────────────────────────────────────────────
def heldout_test_r(model):
    """Mean held-out TEST r over electrodes at the committed layer, or NaN if the JSON is absent.

    Also VERIFIES that the sweep's own `chosen_layer` is the layer the prediction HDF5s were
    written at (`LAYER`). Those two travelled separately -- the layer reached the in-silico step as
    a hardcoded constant in `slurm_insilico_mmn_electrodes.sh`, the r only ever existed in this
    JSON -- so a mismatch would mean the reported fit quality belongs to a different layer than the
    waveforms. Cheap to check, and only checkable now that the JSONs are all present.
    """
    import json
    path = Path(str(MAPPING_JSON).format(model=model))
    if not path.exists():
        return float("nan"), MISSING_R_NOTE.format(model=model)
    d = json.loads(path.read_text())
    if d["chosen_layer"] != LAYER[model]:
        raise SystemExit(
            f"STOP: {model}'s sweep chose layer {d['chosen_layer']!r} but the predictions were "
            f"written at {LAYER[model]!r}. The reported test r would describe a different mapping "
            f"than the waveforms in the figure.")
    tr = [x for x in d.get("test_r_chosen", []) if x == x]
    return (float(np.mean(tr)) if tr else float("nan")), f"{d['chosen_layer']}, n={len(tr)}"


def evidence_table(tidy, models, out_dir, n=N_TOP):
    """One row per model: depth, the LIT→P2 gap, S7 rates, shape quality, and mapping r."""
    rows = []
    for m in models:
        top = {s: rank_for_model(tidy, m, s, dedupe=(s == "literature"), n=n) for s in SETS}
        full = {s: tidy[(tidy["model"] == m) & (tidy["set"] == s)] for s in SETS}
        r, r_note = heldout_test_r(m)
        rec = dict(model=m)
        for key, s in (("lit", "literature"), ("p2", "phase2")):
            t, f = top[s], full[s]
            rec[f"{key}_top5_best_uv"] = round(float(t["trough_uv"].min()), 3)
            rec[f"{key}_top5_median_uv"] = round(float(t["trough_uv"].median()), 3)
            rec[f"{key}_top5_median_recovery_frac"] = round(float(t["recovery_frac"].median()), 3)
            rec[f"{key}_top5_n_s7_pass"] = int(t["s7_pass"].sum())
            rec[f"{key}_top5_n_s4_specific"] = int(t["s4_specificity"].sum())
            rec[f"{key}_top5_n_downsloping"] = int((t["post_slope_uv_per_100ms"] < 0).sum())
            rec[f"{key}_top5_median_post_slope"] = round(
                float(t["post_slope_uv_per_100ms"].median()), 3)
            rec[f"{key}_n_conditions"] = int(len(f))
            rec[f"{key}_s7_rate"] = round(float(f["s7_pass"].mean()), 3)
        # The visual story on the slide: how much deeper the model's own best novel stimuli go
        # than its own best literature stimuli. Negative = Phase 2 deeper.
        rec["gap_median_uv"] = round(rec["p2_top5_median_uv"] - rec["lit_top5_median_uv"], 3)
        rec["gap_best_uv"] = round(rec["p2_top5_best_uv"] - rec["lit_top5_best_uv"], 3)
        rec["gap_s7_rate"] = round(rec["p2_s7_rate"] - rec["lit_s7_rate"], 3)
        # Clean panels = pass S7, pass the control-window specificity check, and are not still
        # falling at the end of the readout. This is the number that answers the objection.
        rec["top10_n_clean"] = int(sum(
            int(((t["s7_pass"]) & (t["s4_specificity"]) &
                 (t["post_slope_uv_per_100ms"] > 0)).sum()) for t in top.values()))
        rec["heldout_test_r_electrodes"] = None if np.isnan(r) else round(r, 4)
        rec["heldout_r_source"] = r_note
        rows.append(rec)
    ev = pd.DataFrame(rows)
    ev.to_csv(Path(out_dir) / "model_choice_evidence.csv", index=False)
    print(f"  wrote model_choice_evidence.csv ({len(ev)} models)")
    return ev


def report_shared_y_across_models(tidy, models, n=N_TOP):
    """Is one y-axis across all six figures legible? Answered with the numbers, not an opinion."""
    span = {}
    for m in models:
        vals = pd.concat([rank_for_model(tidy, m, s, dedupe=(s == "literature"), n=n)
                          for s in SETS])["trough_uv"]
        span[m] = float(vals.min())
    deepest, shallowest = min(span.values()), max(span.values())
    ratio = deepest / shallowest
    print(f"\n  y-axis across models: deepest top-{n} trough {deepest:.2f} µV "
          f"({min(span, key=span.get)}), shallowest {shallowest:.2f} µV "
          f"({max(span, key=span.get)}) — a {ratio:.1f}× range.")
    print(f"  => a single across-model y-axis would compress the shallowest model's traces into "
          f"{100 / ratio:.0f}% of the axis. Per-model scaling is required; the figures share a "
          f"y-axis WITHIN each model only.")
    return span


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(HERE))
    # svgs/ INSIDE this deliverable, not the shared aux/svgs the default derivation would pick
    # (out_dir.parent/"svgs"): this directory is the analysis, not one figure set inside a larger one.
    p.add_argument("--svg_dir", default=str(HERE / "svgs"))
    p.add_argument("--n_top", type=int, default=N_TOP,
                   help=f"panels per row (default {N_TOP}, the size agreed on the call)")
    p.add_argument("--free_y", action="store_true",
                   help="autoscale every panel instead of sharing a y-axis within the figure. "
                        "Inspection aid only; the deck figures use the shared axis.")
    p.add_argument("--skip_figures", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list(SEARCH_MODELS)
    print(f"\n[1] loading and scoring — {len(models)} models, {ROI}, {MAPPING}, "
          f"X = {DIP_UV_THRESHOLD}")
    tidy = build_table(models)

    print("\n[2] cross-check against the committed ranking tables")
    cross_check_committed(tidy, models)

    print("\n[3] per-model rankings")
    panels = pd.concat([rank_for_model(tidy, m, s, dedupe=(s == "literature"), n=args.n_top)
                        for m in models for s in SETS], ignore_index=True)
    cols = ["model", "set", "rank", "method_id", "method", "direction", "f_standard", "f_deviant",
            "semitones", "trough_uv", "latency_ms", "recovery_frac", "s7_pass", "s2_pass",
            "s4_specificity", "post_slope_uv_per_100ms", "ctrl_min_uv", "n_deviants",
            "consensus_rank", "n_agree"]
    panels[cols].round(4).to_csv(out_dir / "per_model_top5_comparison.csv", index=False)
    print(f"  wrote per_model_top5_comparison.csv ({len(panels)} panel rows)")
    full = tidy.sort_values(["model", "set", "trough_uv"], kind="mergesort")
    full[cols[:1] + cols[1:2] + cols[3:]].round(4).to_csv(
        out_dir / "per_model_full_ranking.csv", index=False)
    print(f"  wrote per_model_full_ranking.csv ({len(full)} rows)")

    print("\n[4] figures")
    if args.skip_figures:
        print("  skipped (--skip_figures)")
    else:
        waves = {"literature": load_fcz_waves(
                     sorted(tidy[tidy["set"] == "literature"]["method_id"].unique()), models,
                     LIT_PREDICTIONS, roi=ROI, expect_n_deviants=N_DEVIANTS),
                 "phase2": load_fcz_waves(
                     sorted(tidy[tidy["set"] == "phase2"]["method_id"].unique()), models,
                     P2_PREDICTIONS, roi=ROI, expect_n_deviants=N_DEVIANTS)}
        for m in models:
            figure_for_model(tidy, m, models, out_dir, waves, share_y=not args.free_y,
                             n=args.n_top, svg_dir=args.svg_dir)

    print("\n[5] model-choice evidence")
    ev = evidence_table(tidy, models, out_dir, n=args.n_top)
    pd.set_option("display.width", 200)
    print(ev[["model", "lit_top5_median_uv", "p2_top5_median_uv", "gap_median_uv",
              "lit_s7_rate", "p2_s7_rate", "top10_n_clean",
              "heldout_test_r_electrodes"]].to_string(index=False))
    report_shared_y_across_models(tidy, models, n=args.n_top)


if __name__ == "__main__":
    main()
