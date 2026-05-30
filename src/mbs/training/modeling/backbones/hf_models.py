import torch.nn as nn


def load_model_hf(**model_config) -> nn.Module:
    model_config = dict(model_config)
    model_id = (
        model_config.pop("model_id", None)
        or model_config.get("pretrained_model_id")
        or model_config.get("arch")
    )
    if model_id is None:
        raise ValueError(
            "HF backbone loading requires one of model_id, pretrained_model_id, or arch."
        )

    from mbs.extraction.modeling.backbones.hf_models import load_model_hf as _load_model_hf

    model, _ = _load_model_hf(model_id=model_id, **model_config)
    return model
