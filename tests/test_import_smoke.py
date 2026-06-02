import importlib

import pytest


def test_package_import_smoke():
    pytest.importorskip("torch")
    modules = [
        "mbs.core",
        "mbs.metrics",
        "mbs.modeling",
        "mbs.training",
        "mbs.extraction",
        "mbs.evaluation",
        "mbs.analysis",
        "mbs.visualization",
    ]
    for module in modules:
        importlib.import_module(module)


def test_audio_module_imports():
    """Audio extension modules must be importable once the audio extra is installed."""
    pytest.importorskip("whisper")
    modules = [
        "mbs.extraction.data.datasets_audio",
        "mbs.extraction.modeling.backbones.audio_models",
        "mbs.data_prep.format_eeg_hdf5",
        "mbs.evaluation.evaluate_features_temporal",
    ]
    for module in modules:
        importlib.import_module(module)
