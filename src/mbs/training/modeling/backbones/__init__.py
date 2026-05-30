from pathlib import Path

import torch
import torch.nn as nn

MODEL_LOADERS = [
    "timm",
    "spvvs",
    "hf",
]

class BackboneWrapper(nn.Module):
    """Wrapper class for the backbone model that preserves all its layers, 
        enabling the extraction of a complete computational graph with torch.fx."""
    def __init__(self, backbone):
        super(BackboneWrapper, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def create_backbone(**model_config) -> nn.Module:
    """
    Create a model from a configuration dictionary.
    
    Args:
        **model_config: Arbitrary keyword arguments containing the model configuration.
            Expected keys include:
            - `backbone_source`: The source of the model (e.g., 'torchvision', 'timm', 'spvvs' for scaling_primate_vvs, 'hf' for Hugging Face).

    Returns:
        nn.Module: The created model instance.
    """
    
    backbone_source = model_config.get('backbone_source', None)
    assert backbone_source in MODEL_LOADERS, \
        f"backbone_source {model_config['backbone_source']} not recognized. Available sources: {list(MODEL_LOADERS.keys())}"

    if backbone_source == "timm":
        from .timm_models import load_model_timm
        model = load_model_timm(**model_config)
    elif backbone_source == "spvvs":
        from .scaling_models import load_model_spvvs
        model = load_model_spvvs(**model_config)
    elif backbone_source == "hf":
        from .hf_models import load_model_hf
        model = load_model_hf(**model_config)

    # Wrap the model in the Backbone wrapper
    model = BackboneWrapper(model)
    
    return model