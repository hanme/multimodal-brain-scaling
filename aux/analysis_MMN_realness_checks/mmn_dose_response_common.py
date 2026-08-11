#!/usr/bin/env python
"""Shared vocabulary for the two FCz dose-response checks (N-effect and deviance scaling).

Both checks ask the same question -- does the in-silico MMN trough deepen as the paradigm makes
the oddball more surprising? -- over the same two condition sources, the same six models, the same
site, and the same gate. Everything they share lives here once so the two scripts cannot drift:

  n_effect_plots.py            Part B: trough vs N (tones between deviants)
  deviance_scaling_s7gated.py  Part C: trough vs deviance size (semitones)

The two sources
---------------
  lit  LIT      24 literature Frequency methods x {regular, counter} = 48 conditions, from the
                SOAFIX prediction vintage. Unselected. SOA 200-1000 ms, duration 50-200 ms.
  p2   NOVEL-P2 127 selected tone pairs x {regular, counter} = 254 conditions. Fixed
                580 ms SOA / 80 ms duration / 10% deviants; SELECTED on model MMN agreement.

Phase 1 is excluded entirely (1806 conditions at n_deviants=1: no N-effect to measure, and a
single-trial trough is not the 15-trial average behind LIT and NOVEL-P2).

The three analysis sets
-----------------------
  lit = LIT only (48) | lit_p2 = both (302) | p2 = NOVEL-P2 only (254)
Derived by FILTERING the one tidy frame, never by a second loading path -- see set_frame().

Fixed reporting choices, none of them swept here (see the brief):
  site = FCz electrode, mapping = mTRF, floor X = 0.75 uV, 6 models (whisper-large excluded).

whisper-large is dropped at load time, in load_fcz(), so it cannot leak into a per-model panel or
an appendix table downstream. Two reasons: its predicted uV run ~20-35x every other model (median
S7-passing LIT trough at FCz -37.3 uV vs a -1.08..-1.75 uV band for the other six), which would
dominate any raw-uV pooled mean and force a symlog axis; and it was never run in the novel search,
so keeping it would make `lit` and `p2` incomparable.

READ-ONLY over the scored CSVs and the metadata.
"""
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def _repo_root(start):
    """Walk up to the checkout root (the dir holding pyproject.toml).

    Found by marker rather than by a fixed parents[k] index: this directory has already been
    relocated once, and a hardcoded depth breaks silently when it moves again.
    """
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists():
            return d
    raise RuntimeError(f"no pyproject.toml above {start} -- not inside the checkout?")


REPO = _repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO / "scripts"))

# The canonical 6-model list and the fixed reporting choices come from the search's own module --
# retyping them here is how `lit` and `p2` would silently stop being the same experiment.
from novel_search_common import SEARCH_MODELS, ROI, MAPPING, DIP_UV_THRESHOLD  # noqa: E402
# Style is the committed screen's, reused verbatim: same palette, same per-model colour+marker.
from analyze_mmn_screen_24freq import (  # noqa: E402
    MODEL_STYLE, MODEL_ORDER as _SCREEN_MODEL_ORDER, load_semitones,
)

# Model order = the screen's, with the whisper-large entry FILTERED OUT (never edited out of the
# shared MODEL_STYLE dict, which other scripts still read at 7 models).
MODEL_ORDER = [m for m in _SCREEN_MODEL_ORDER if m in SEARCH_MODELS]
assert MODEL_ORDER == SEARCH_MODELS or sorted(MODEL_ORDER) == sorted(SEARCH_MODELS), \
    f"model order {MODEL_ORDER} is not the search's {SEARCH_MODELS}"
assert "whisper-large" not in MODEL_ORDER

LIT_CSV = REPO / "outputs/results_soafix_full/mmn_s7_roi.csv"
P2_CSV = REPO / "outputs/results_novel_search/phase2_mmn_s7_roi.csv"
LIT_META = REPO / "data/metadata/literature_frequency_intensity_duration_metadata.csv"
P2_GRID = REPO / "outputs/results_novel_search/grid_index.csv"
# Every figure is emitted twice: a PNG under plots/ for reading and review, and a vector SVG under
# the sibling svgs/ for the write-up (these panels carry dense scatter and small type, which a
# raster copy loses at print size). finish() writes both from the one render.
ANALYSIS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = ANALYSIS_DIR / "plots"
SVG_DIRNAME = "svgs"

# THE VISUAL GRAMMAR -- one channel, one meaning, everywhere in this deliverable:
#   colour     = model identity          (MODEL_STYLE, the committed Okabe-Ito palette)
#   linestyle  = NOTHING                 every model line is solid (see style() below)
#   marker     = model identity, and DATA SOURCE wherever the two sources share an axis
#                                        (the lit_p2 panels and the overlap diagnostic)
#   dotted grey = the y=0 reference line, which is NOT data
# Source is never encoded by linestyle either: with every line solid, a dashed line in these
# figures would be read as a distinct series that does not exist.
DATASET_MARKER = {"lit": "o", "p2": "^"}
DATASET_LABEL = {"lit": "LIT (literature)", "p2": "NOVEL-P2 (search)"}

# Which sources each analysis set draws from. The ONLY place the three sets are defined.
SETS = {"lit": ("lit",), "lit_p2": ("lit", "p2"), "p2": ("p2",)}
SET_LABEL = {"lit": "LIT only", "lit_p2": "LIT + NOVEL-P2", "p2": "NOVEL-P2 only"}

# Expected condition counts per source -- asserted at load, so a wrong vintage or a bad filter
# fails loudly instead of quietly producing a plausible-looking figure.
N_CONDITIONS = {"lit": 48, "p2": 254}
OVERLAP_ST = (4.50, 12.0)          # semitone range where BOTH sources have conditions

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


# Every model line is drawn SOLID here. The shared MODEL_STYLE dashes the wav2vec2 entries to
# encode architecture family a second time, but that reads as a semantic difference between the
# series when the analysis treats all six models alike, so this deliverable overrides it.
#
# What that costs, and why it is affordable: the dashes were the redundant channel keeping the
# families apart in greyscale and print. Identity now rests on colour + marker alone. The palette
# validator's one WARN pair (whisper-small vs wav2vec2-medium, CVD dE 7.6, inside the 6-8 band
# that is legal ONLY with secondary encoding) is still covered, because those two carry different
# MARKERS (^ vs P) -- marker, not linestyle, is what satisfies that requirement.
#
# The override is LOCAL. MODEL_STYLE is imported by analyze_mmn_screen_24freq.py and others, so
# editing the shared dict would silently restyle their committed figures.
LINESTYLE = "-"


def style(model):
    """Per-model colour/marker, identical in every set and every panel; linestyle always solid."""
    st = dict(MODEL_STYLE.get(model, dict(color="#666666", marker="o")))
    st["ls"] = LINESTYLE
    return st


# ------------------------------------------------------------------------------------------
# Loading -- ONE tidy frame; the three sets are filters over it.
# ------------------------------------------------------------------------------------------
def load_fcz(csv_path, dataset, x=DIP_UV_THRESHOLD):
    """The (FCz, mTRF, X) slice of one scored CSV, 6 models, tagged with `dataset`.

    One row per (model, condition). whisper-large is dropped HERE, which is what guarantees it
    cannot reappear downstream.
    """
    df = pd.read_csv(csv_path)
    sel = df[(df["roi_kind"] == "electrode") & (df["roi"] == ROI)
             & (df["mapping"] == MAPPING) & np.isclose(df["dip_uv_threshold"], x)
             & (df["model"].isin(SEARCH_MODELS))].copy()
    if sel.empty:
        raise SystemExit(f"{csv_path}: no rows at roi={ROI} mapping={MAPPING} X={x}")

    missing = sorted(set(SEARCH_MODELS) - set(sel["model"]))
    if missing:
        raise SystemExit(f"{csv_path} is missing model(s) {missing}; all six or none.")

    sel["dataset"] = dataset
    sel["method_id"] = sel["method"].str.extract(r"method_(\d+)").astype(int)
    sel["direction"] = np.where(sel["is_counter"], "counter", "regular")
    return sel[["dataset", "model", "method", "method_id", "direction", "is_counter",
                "s2", "trough_uv", "s7"]]


def load_tidy(verify=True):
    """The single tidy frame behind every figure in Parts B(rate) and C: LIT + NOVEL-P2 at FCz.

    Columns: dataset, model, method, method_id, direction, is_counter, semitones, s2, trough_uv, s7.
    `verify` asserts the committed counts -- the cheapest way to catch a wrong-vintage CSV. The S2
    totals are load-time integrity checks ONLY; S2 is not a reporting set anywhere in this
    deliverable (the one gate is S7@0.75).
    """
    lit = load_fcz(LIT_CSV, "lit")
    p2 = load_fcz(P2_CSV, "p2")

    # LIT semitones: computed from the canonical literature metadata, never a hardcoded dict.
    lit_st = load_semitones(LIT_META)
    lit["semitones"] = lit["method_id"].map(lit_st)
    if lit["semitones"].isna().any():
        bad = sorted(lit.loc[lit["semitones"].isna(), "method_id"].unique())
        raise SystemExit(f"no semitone entry in {LIT_META} for method_id {bad}")

    # NOVEL-P2 semitones: the precomputed canonical column, joined on method_id.
    grid = pd.read_csv(P2_GRID)[["method_id", "semitones", "f_low", "f_high"]]
    p2 = p2.merge(grid, on="method_id", how="left")
    if p2["semitones"].isna().any():
        bad = sorted(p2.loc[p2["semitones"].isna(), "method_id"].unique())
        raise SystemExit(f"{P2_GRID} has no entry for method_id {bad}")

    tidy = pd.concat([lit, p2], ignore_index=True)
    if verify:
        _verify(tidy)
    return tidy


def _verify(tidy):
    """Assert the committed FCz numbers. Any deviation = wrong filter or wrong prediction vintage."""
    for ds, n_cond in N_CONDITIONS.items():
        d = tidy[tidy["dataset"] == ds]
        assert d["method"].nunique() == n_cond, \
            f"{ds}: {d['method'].nunique()} conditions, expected {n_cond}"
        assert len(d) == n_cond * len(MODEL_ORDER), \
            f"{ds}: {len(d)} rows, expected {n_cond * len(MODEL_ORDER)}"
    assert "whisper-large" not in set(tidy["model"]), "whisper-large leaked into the tidy frame"

    expect_s7 = {"lit": 125, "p2": 856}
    expect_s2 = {"lit": 262, "p2": 1185}         # integrity check only -- never reported
    for ds in N_CONDITIONS:
        d = tidy[tidy["dataset"] == ds]
        assert int(d["s7"].sum()) == expect_s7[ds], \
            f"{ds}: S7@0.75 = {int(d['s7'].sum())}, expected {expect_s7[ds]}"
        assert int(d["s2"].sum()) == expect_s2[ds], \
            f"{ds}: S2 = {int(d['s2'].sum())}, expected {expect_s2[ds]}"
    # S7 is a subset of S2 by construction, everywhere.
    assert not (tidy["s7"] & ~tidy["s2"]).any(), "S7 rows that are not S2"


# Per-trial CSVs written by scripts/analyze_mmn_per_trial_n.py (Part B). These do NOT exist until
# both prediction roots are re-run with the deviants_fc patch -- see that script's docstring.
LIT_PER_TRIAL = REPO / "outputs/results_soafix_full/mmn_per_trial_n_fcz.csv"
P2_PER_TRIAL = REPO / "outputs/results_novel_search/phase2_mmn_per_trial_n_fcz.csv"
N_LEVELS = (3, 5, 7)
# Oddball probability is 1/(N+1) BY CONSTRUCTION of the generator
# (scripts/00aa_generate_audio_stimuli.py:339), so N and rarity cannot be separated here.
N_TO_P_DEVIANT = {n: 1.0 / (n + 1) for n in N_LEVELS}
EXPECTED_TRIAL_ROWS = {"lit": 6 * 48 * 15, "p2": 6 * 254 * 15}


def load_per_trial(lit_csv=LIT_PER_TRIAL, p2_csv=P2_PER_TRIAL, verify=True):
    """The tidy per-trial frame behind Part B: one row per (dataset, model, condition, N, var).

    Same three-sets-by-filtering contract as load_tidy(); `set_frame` and `gated` work on either.
    """
    missing = [p for p in (lit_csv, p2_csv) if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "Part B needs the per-trial CSVs, and these are absent:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n\nThey are produced by scripts/analyze_mmn_per_trial_n.py, which needs the "
              "per-trial deviant stack (`deviants_fc`) in the ELECTRODE prediction h5s. The "
              "committed h5s do not carry it -- the driver averaged the trials away. Both "
              "prediction roots must be re-run once with the patched driver (Part A2) first.")

    frames = []
    for path, ds in ((lit_csv, "lit"), (p2_csv, "p2")):
        d = pd.read_csv(path)
        d["dataset"] = ds
        frames.append(d)
    tidy = pd.concat(frames, ignore_index=True)

    tidy = tidy[tidy["model"].isin(SEARCH_MODELS)].copy()
    if verify:
        assert "whisper-large" not in set(tidy["model"]), "whisper-large leaked in"
        for ds, n_rows in EXPECTED_TRIAL_ROWS.items():
            d = tidy[tidy["dataset"] == ds]
            assert len(d) == n_rows, f"{ds}: {len(d)} per-trial rows, expected {n_rows}"
            assert sorted(d["N"].unique()) == list(N_LEVELS), f"{ds}: N levels {d['N'].unique()}"
        assert not (tidy["s7"] & ~tidy["s2"]).any(), "S7 rows that are not S2"
    return tidy


def per_method_cells(frame, extra_keys=()):
    """Collapse the 5 variations to ONE value per (dataset, model, condition, N).

    The 5 variations of a given (model, condition, N) are the SAME paradigm re-rolled, so their
    trials are not independent observations; correlating N against all of them would inflate n
    ~5x and understate p. The primary N statistic is therefore computed on these cells (median of
    the cell's S7-passing troughs), with the raw-trial correlation reported alongside as the
    optimistic bound. Cells with no S7-passing trial are absent, not zero.
    """
    keys = ["dataset", "model", "method", "N", *extra_keys]
    g = gated(frame)
    if g.empty:
        return g.assign(trough_uv=[]).loc[:, keys + ["trough_uv"]]
    return (g.groupby(keys, observed=True)["trough_uv"].median().reset_index())


def set_frame(tidy, set_key):
    """One analysis set, by filtering the tidy frame. The three sets share this one path."""
    return tidy[tidy["dataset"].isin(SETS[set_key])].copy()


def gated(frame):
    """S7@0.75-passing rows -- the ONLY amplitude reporting set in this deliverable."""
    return frame[frame["s7"]].copy()


# ------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------
def spearman(x, y):
    """(rho, p, n) over the finite pairs; (nan, nan, n) when n < 3 or x is constant."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.unique(x[ok]).size < 2:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p), n


def s7_rate(frame, by):
    """S7@0.75 count / TOTAL conditions in each bin -- denominator is all conditions, not S2.

    This is the one uncensored view in the deliverable: unlike the gated trough, a count outcome
    is not floored by the gate, so a dose-response can show here that the amplitude axis cannot.
    """
    g = frame.groupby(by, observed=False)["s7"]
    out = g.agg(n_s7="sum", n_total="size").reset_index()
    out["rate"] = out["n_s7"] / out["n_total"].where(out["n_total"] > 0)
    return out


def sem(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")


def wrap(text, width=118):
    """Hard-wrap a caption.

    Captions are drawn with fig.text and saved with bbox_inches="tight", so ONE long line silently
    widens the whole canvas and squeezes the panels into a strip. Wrapping is what keeps the plot
    area the thing that sets the figure width.
    """
    return "\n".join(textwrap.fill(par, width=width) for par in text.split("\n"))


def finish(fig, out_png, tight_rect=None):
    """Save one figure as PNG under plots/ AND as SVG under the sibling svgs/.

    Both come from the SAME render, so the vector copy can never drift from the raster one. The
    svgs/ dir is derived from the png's parent rather than hardcoded, so a custom --out_dir keeps
    the pair together.
    """
    out_png = Path(out_png)
    out_svg = out_png.parent.parent / SVG_DIRNAME / (out_png.stem + ".svg")
    for p in (out_png, out_svg):
        p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=tight_rect) if tight_rect else fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}  (+ {SVG_DIRNAME}/{out_svg.name})")
    return out_png
