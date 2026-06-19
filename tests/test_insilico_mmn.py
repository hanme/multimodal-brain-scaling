"""Tests for the in-silico MMN drivers under scripts/: the literature METHODS registry +
SOA lookup, the finalize_method() mean-vs-z-score split, the previously-broken
load_split_parcels imports (regression), and the combined results table builder.

No real EEG/model data needed -- finalize_method is pure array math, and the import tests
just need the modules to be importable.
"""

import csv
import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import insilico_mmn as im  # noqa: E402

# Real SOA values from data/metadata/literature_frequency_intensity_duration_metadata.csv,
# keyed by method_id, for the 10 methods in METHODS.
EXPECTED_SOA_MS = {75: 500.0, 74: 1000.0, 72: 500.0, 60: 300.0, 53: 333.0,
                   55: 500.0, 37: 310.0, 43: 510.0, 44: 510.0, 27: 900.0}


# ──────────────────────────────────────────────────────────
# Regression: the load_split_parcels -> load_split_targets import bug (was ImportError)
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", [
    "insilico_mmn", "insilico_mmn_attn", "insilico_mmn_electrodes",
    "score_mtrf_fitquality", "plot_fit_quality", "build_mmn_results_table",
])
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)


def test_no_module_imports_the_removed_load_split_parcels_name():
    for module_name in ["insilico_mmn", "insilico_mmn_attn", "insilico_mmn_electrodes",
                         "score_mtrf_fitquality", "plot_fit_quality"]:
        mod = importlib.import_module(module_name)
        assert not hasattr(mod, "load_split_parcels")


# ──────────────────────────────────────────────────────────
# METHODS registry + SOA table
# ──────────────────────────────────────────────────────────

def test_methods_registry_is_the_10_literature_methods():
    ids = [int(m[0].split("_")[1]) for m in im.METHODS]
    assert len(im.METHODS) == 10
    assert set(ids) == set(EXPECTED_SOA_MS)
    assert len(set(ids)) == 10  # no duplicate method ids
    for method, label, source in im.METHODS:
        assert method.startswith("method_")
        assert "→" in label  # "standard->deviant" label, not the old DOWN/UP direction string
        assert source and source[0].isupper()  # citation, e.g. "Javitt_2000a"


def test_soa_for_method_matches_metadata_csv():
    soa_table = im.load_soa_table()
    for method, _, _ in im.METHODS:
        n = int(method.split("_")[1])
        assert im.soa_for_method(method, soa_table) == pytest.approx(EXPECTED_SOA_MS[n])


# ──────────────────────────────────────────────────────────
# finalize_method: plotted (mean-only) vs verdict (z-scored) split
# ──────────────────────────────────────────────────────────

@pytest.fixture
def stim_dir(tmp_path):
    (tmp_path / "method_test_standard.wav").write_bytes(b"")  # finalize_method only globs the name
    return tmp_path


def _synthetic_traces(final_s, soa_ms, n_t=300, n_target=2, baseline_sd=5.0, seed=0,
                       amp_other=50.0, amp_n7v1=200.0):
    """std_raw (pure baseline-sd noise, no bump) + 3 deviant traces, one of them ('N7var1')
    with a bigger negative bump in the [100,240] ms MMN window than the other two."""
    rng = np.random.default_rng(seed)
    t_idx = np.arange(n_t)
    rel_ms = t_idx * im.TIME_STEP_MS - final_s * 1000.0
    mmn_win = (rel_ms >= 100.0) & (rel_ms <= 240.0)
    assert mmn_win.any(), "fixture must cover the MMN window"

    std_raw = rng.normal(0.0, baseline_sd, size=(n_t, n_target)).astype(np.float32)

    def bumped(amp):
        sig = rng.normal(0.0, baseline_sd, size=(n_t, n_target)).astype(np.float32)
        sig[mmn_win] -= amp
        return sig

    dev_preds = [bumped(amp_other), bumped(amp_n7v1), bumped(amp_other)]
    dev_ids = ["method_test_N3_var2_deviant", "method_test_N7_var1_deviant",
               "method_test_N5_var4_deviant"]
    return t_idx, std_raw, dev_preds, dev_ids


def test_finalize_method_plotted_traces_are_mean_only_not_zscored(monkeypatch, stim_dir):
    final_s, soa_ms, baseline_sd = 5.0, 300.0, 5.0
    monkeypatch.setattr(im, "detect_final_tone_onset_s", lambda wav: final_s)
    t_idx, std_raw, dev_preds, dev_ids = _synthetic_traces(final_s, soa_ms, baseline_sd=baseline_sd)

    res = im.finalize_method("method_test", t_idx, std_raw, dev_preds, dev_ids, stim_dir, soa_ms)

    rel_ms = res["rel_ms"]
    base = (rel_ms >= -3.0 * soa_ms) & (rel_ms < 0)
    # Mean-corrected: baseline mean ~0 ...
    np.testing.assert_allclose(res["dev_b"][base].mean(0), 0.0, atol=1.0)
    np.testing.assert_allclose(res["std_b"][base].mean(0), 0.0, atol=1.0)
    # ... but NOT divided by std -- baseline std should stay well above 1 (dev_b is the mean of
    # 3 noisy deviants, so its baseline std is reduced by ~sqrt(3) from baseline_sd=5 to ~2.9;
    # std_b is a single trace so it stays close to 5). Either way, both must be far from the ~1
    # a full z-score would force -- that's the property under test, not the exact value.
    assert (res["dev_b"][base].std(0) > 1.5).all()
    assert (res["std_b"][base].std(0) > 1.5).all()
    assert np.allclose(res["diff_b"], res["dev_b"] - res["std_b"])


def test_finalize_method_peak_is_zscored_and_negative_in_mmn_window(monkeypatch, stim_dir):
    final_s, soa_ms = 5.0, 300.0
    monkeypatch.setattr(im, "detect_final_tone_onset_s", lambda wav: final_s)
    # Only the N7/var1 deviant carries a bump; the other two are pure noise. So dev_raw's bump
    # (the 3-way average) is the N7/var1 bump diluted by 3, while n7v1_peak sees the bump
    # undiluted -- n7v1_peak must come out more negative regardless of noise realization.
    t_idx, std_raw, dev_preds, dev_ids = _synthetic_traces(final_s, soa_ms, amp_other=0.0, amp_n7v1=150.0)

    res = im.finalize_method("method_test", t_idx, std_raw, dev_preds, dev_ids, stim_dir, soa_ms)

    assert np.isfinite(res["peak"]).all()
    assert (res["peak"] < 0).all()          # diluted bump still drags the 3-deviant average down
    assert np.isfinite(res["n7v1_peak"]).all()
    assert (res["n7v1_peak"] < 0).all()
    assert (res["n7v1_peak"] < res["peak"]).all()


def test_finalize_method_n7v1_peak_is_nan_when_absent(monkeypatch, stim_dir):
    final_s, soa_ms = 5.0, 300.0
    monkeypatch.setattr(im, "detect_final_tone_onset_s", lambda wav: final_s)
    t_idx, std_raw, dev_preds, dev_ids = _synthetic_traces(final_s, soa_ms)
    dev_ids = [d.replace("N7_var1", "N7_var2") for d in dev_ids]  # remove the N7/var1 id

    res = im.finalize_method("method_test", t_idx, std_raw, dev_preds, dev_ids, stim_dir, soa_ms)
    assert np.isnan(res["n7v1_peak"]).all()


# ──────────────────────────────────────────────────────────
# build_mmn_results_table.py
# ──────────────────────────────────────────────────────────

import build_mmn_results_table as bmrt  # noqa: E402


def _write_predictions_h5(path, layer, methods):
    with h5py.File(path, "w") as h5:
        h5.attrs["layer"] = layer
        h5.create_dataset("parcels", data=np.array(["frontal", "central", "temporal"], dtype="S12"))
        for method, source, peak, n7v1_peak in methods:
            g = h5.create_group(method)
            g.attrs["source"] = source
            g.attrs["context_final"] = "1000→1200 Hz"
            g.create_dataset("peak", data=np.array(peak, np.float32))
            g.create_dataset("n7v1_peak", data=np.array(n7v1_peak, np.float32))


def test_rows_from_h5_reads_peak_and_n7v1_peak_per_parcel(tmp_path):
    h5_path = tmp_path / "predictions__blocks.0.h5"
    _write_predictions_h5(h5_path, "blocks.0",
                           [("method_75", "Karger_2014", [-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5])])

    rows = bmrt.rows_from_h5(h5_path, "whisper-tiny", "parcels", "mtrf")
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "whisper-tiny" and row["level"] == "parcels" and row["mapping"] == "mtrf"
    assert row["method"] == "method_75" and row["source"] == "Karger_2014"
    assert row["layer"] == "blocks.0"
    assert row["frontal_peak"] == pytest.approx(-1.0)
    assert row["temporal_peak"] == pytest.approx(-3.0)
    assert row["central_n7v1_peak"] == pytest.approx(-2.5)


def test_main_globs_model_dirs_and_writes_combined_csv(tmp_path, monkeypatch):
    # mirrors the real cluster layout: mTRF parcels/electrodes share one bare "<model>" dir
    # (told apart by filename prefix); encoder dirs are namespaced "<model>-<level>".
    root = tmp_path / "predictions"
    tiny_dir = root / "whisper-tiny"
    tiny_dir.mkdir(parents=True)
    _write_predictions_h5(tiny_dir / "predictions__blocks.0.h5", "blocks.0",
                           [("method_75", "Karger_2014", [-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5])])
    _write_predictions_h5(tiny_dir / "electrode_predictions__blocks.0.h5", "blocks.0",
                           [("method_75", "Karger_2014", [-1.1, -2.1, -3.1], [-1.6, -2.6, -3.6])])

    enc_parcels_dir = root / "whisper-tiny-parcels" / "method_75"
    enc_parcels_dir.mkdir(parents=True)
    _write_predictions_h5(enc_parcels_dir / "predictions__blocks.3__attn.h5", "blocks.3",
                           [("method_75", "Karger_2014", [-0.5, -0.6, -0.7], [-0.8, -0.9, -1.0])])

    enc_elec_dir = root / "whisper-tiny-electrodes" / "method_75"
    enc_elec_dir.mkdir(parents=True)
    _write_predictions_h5(enc_elec_dir / "predictions__blocks.3__attn.h5", "blocks.3",
                           [("method_75", "Karger_2014", [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9])])

    out_csv = tmp_path / "out" / "table.csv"
    monkeypatch.setattr(sys, "argv", ["build_mmn_results_table.py",
                                       "--predictions_root", str(root), "--out", str(out_csv)])
    bmrt.main()

    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    seen = {(r["model"], r["level"], r["mapping"]) for r in rows}
    assert seen == {
        ("whisper-tiny", "parcels", "mtrf"),
        ("whisper-tiny", "electrodes", "mtrf"),
        ("whisper-tiny", "parcels", "encoder"),
        ("whisper-tiny", "electrodes", "encoder"),
    }


def test_main_writes_nothing_when_predictions_root_missing(tmp_path, monkeypatch, capsys):
    out_csv = tmp_path / "table.csv"
    monkeypatch.setattr(sys, "argv", ["build_mmn_results_table.py",
                                       "--predictions_root", str(tmp_path / "does_not_exist"),
                                       "--out", str(out_csv)])
    bmrt.main()
    assert not out_csv.exists()
