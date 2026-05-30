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
