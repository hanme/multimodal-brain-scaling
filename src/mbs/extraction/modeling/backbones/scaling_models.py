import os
from pathlib import Path

import torch.nn as nn

from scaling_primate_vvs.training import create_model as create_model_spvvs
from scaling_primate_vvs import get_model_checkpoints_dataframe
from scaling_primate_vvs.training.src.dataloaders.transforms import create_transforms

from mbs.core import find_repo_root, load_yaml

config_dir = find_repo_root() / "configs" / "models" / "spvvs"

def resolve_config(model_id: str, config_dir: Path):
    available_configs = config_dir.glob("*.yaml")
    matched_configs = []
    for config_path in available_configs:
        config_name = config_path.stem
        if config_name in model_id:
            matched_configs.append(config_path)
    return matched_configs

def load_config(model_id: str):
    config = {}
    
    if "simclr" in model_id:
        if "resnet" in model_id:
            matched_configs = resolve_config(model_id, config_dir / "simclr" / "resnet")
            config.update(load_yaml(matched_configs[0]))
            return config
        
        elif "vit" in model_id:
            matched_configs = resolve_config(model_id, config_dir / "simclr" / "vit")
            config.update(load_yaml(matched_configs[0]))
            return config
                
    elif "dino" in model_id:
        matched_configs = resolve_config(model_id, config_dir / "dino")
        config.update(load_yaml(matched_configs[0]))
        return config

    if 'resnet' in model_id:
        if "resnetflexfilters" in model_id:
            config_file = config_dir / "resnet" / "resnetflexfilters.yaml"
            config.update(load_yaml(config_file))
        else:
            matched_configs = resolve_config(model_id, config_dir / "resnet")
            config.update(load_yaml(matched_configs[0]))
    
    if 'efficientnet' in model_id:
        matched_configs = resolve_config(model_id, config_dir / "efficientnet")
        config.update(load_yaml(matched_configs[0]))
    
    elif 'convnext' in model_id:
        if "convnextflex-l-" in model_id:
            config_file = config_dir / "convnext" / "convnextflex.yaml"
            config.update(load_yaml(config_file))
        else:
            matched_configs = resolve_config(model_id, config_dir / "convnext")
            config.update(load_yaml(matched_configs[0]))
    
    elif 'deit' in model_id:
        if "deitflex-l-" in model_id:
            config_file = config_dir / "deit" / "deitflex.yaml"
            config.update(load_yaml(config_file))
        else:
            matched_configs = resolve_config(model_id, config_dir / "deit")
            config.update(load_yaml(matched_configs[0]))
    
    elif "alexnet" in model_id:
        matched_configs = resolve_config(model_id, config_dir / "others")
        config.update(load_yaml(matched_configs[0]))
        return config
    
    elif "cornet_s" in model_id:
        matched_configs = resolve_config(model_id, config_dir / "others")
        config.update(load_yaml(matched_configs[0]))
        return config
    
    if "adv" in model_id:
        matched_configs = resolve_config(model_id, config_dir / "adversarial")
        config.update(load_yaml(matched_configs[0]))
        
    return config



def load_model_spvvs(model_id:str, **model_config) -> nn.Module:
    """
    Create a model from a configuration dictionary
    using `create_model` from the `scaling_primate_vvs` package.
    
    Args:
        **model_config: Arbitrary keyword arguments containing the model configuration.
            Expected keys include:
            - 'arch': The architecture of the model (e.g., 'resnet18').
            - 'checkpoint': The checkpoint URL or path for the pretrained model.
            - 'pretrained_model_id': The ID of the pretrained model to load (optional).
            - 'checkpoints_dir': The directory containing model checkpoints (optional).
    
    Returns:
        nn.Module: The created model instance.
    """
    
    
    # Load the model configuration
    config = load_config(model_id)
    config.update(model_config)
    model_config = config

    checkpoints_dir = os.getenv("SCALING_MODELS_CKPTS_DIR")
    df_model_checkpoints = get_model_checkpoints_dataframe()
    model_config['run_name'] = model_id

    # Determine the checkpoint path
    checkpoint_info = df_model_checkpoints[df_model_checkpoints['model_id'] == model_id]
    checkpoint_info = checkpoint_info.sort_values('epoch', ascending=False).iloc[0]
    if checkpoints_dir is not None:
        checkpoint = Path(checkpoints_dir) / checkpoint_info['checkpoint_path']
        if not checkpoint.exists():
            print(f"Checkpoint path {checkpoint} does not exist. Using URL instead.")
            checkpoint = checkpoint_info['checkpoint_url']
        checkpoint = str(checkpoint)
    else:
        checkpoint = checkpoint_info['checkpoint_url']
    del df_model_checkpoints
    
    # Add the checkpoint path to the configuration
    model_config['checkpoint'] = checkpoint
    
    # Monkey patch: Adjust num_classes for ecoset models
    if "ecoset" in model_id:
        model_config["num_classes"] = 565

    # Create and return the ComposerModel instance
    model = create_model_spvvs(**model_config)
    
    if hasattr(model, 'module'):
        # Unwrap the model from the Composer wrapper
        model = model.module
    elif hasattr(model, 'backbone'):
        model = model.backbone
    
    # Create the data transform
    model_config["transform_lib"] = "pytorch"
    model_config["pytorch_aug_set"] = "default"
    _, transform = create_transforms(**model_config)

    return model, transform
