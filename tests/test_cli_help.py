import subprocess
import sys

import pytest

def run_help(module: str):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_training_help():
    pytest.importorskip("lightning")
    run_help("mbs.training.train")


def test_extraction_help():
    run_help("mbs.extraction.extract_features")


def test_evaluation_help():
    pytest.importorskip("sklearnex")
    run_help("mbs.evaluation.evaluate_features_all_layers")
    run_help("mbs.evaluation.evaluate_features_committed_layers")
    run_help("mbs.evaluation.evaluate_features_attn_probe")


def test_analysis_help():
    pytest.importorskip("numba")
    run_help("mbs.analysis.curve_fitting.start_fitting")


def test_temporal_evaluation_help():
    pytest.importorskip("sklearnex")
    run_help("mbs.evaluation.evaluate_features_temporal")
