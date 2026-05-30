import numpy as np
import pytest


def test_metrics_import():
    torch = pytest.importorskip("torch")
    from mbs.metrics import CenteredKernelAlignmentTorch, RepresentationalSimilarityAnalysisTorch

    x = torch.eye(4)
    rsa = RepresentationalSimilarityAnalysisTorch()
    cka = CenteredKernelAlignmentTorch()
    assert torch.is_tensor(rsa(x, x))
    assert torch.is_tensor(cka(x, x))


def test_projection_seed_and_weight_initialization():
    torch = pytest.importorskip("torch")
    from mbs.modeling.projection import create_projector

    p1 = create_projector(input_dim=4, output_dim=2, random_seed=7)
    p2 = create_projector(input_dim=4, output_dim=2, random_seed=7)
    assert torch.allclose(
        p1.projection_layer.linear.weight,
        p2.projection_layer.linear.weight,
    )

    weights = torch.ones(2, 4)
    p3 = create_projector(input_dim=4, output_dim=2, projector_weights=weights)
    assert torch.allclose(p3.projection_layer.linear.weight, weights)


def test_metric_primitives_on_small_arrays():
    pytest.importorskip("torch")
    from mbs.metrics import RepresentationalSimilarityAnalysis

    x = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [1.0, 3.0, 0.0],
            [4.0, 1.0, 3.0],
        ]
    )
    rsa = RepresentationalSimilarityAnalysis()
    assert np.isfinite(rsa(x, x))
