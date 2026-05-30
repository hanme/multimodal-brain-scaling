from pathlib import Path

import torch.nn as nn

from scaling_primate_vvs.training import create_model as create_model_spvvs
from scaling_primate_vvs import get_model_checkpoints_dataframe


def load_model_spvvs(**model_config) -> nn.Module:
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
    training_mode = model_config.pop('training_mode', 'finetune')
    pretrained_model_id = model_config.pop('pretrained_model_id', None)
    if pretrained_model_id is not None and training_mode == 'finetune':
        checkpoints_dir = model_config.get('checkpoints_dir', None)
        df_model_checkpoints = get_model_checkpoints_dataframe()
        model_id = pretrained_model_id
        model_config['run_name'] = model_id

        # Determine the checkpoint path
        checkpoint_info = df_model_checkpoints[df_model_checkpoints['model_id'] == model_id]
        checkpoint_info = checkpoint_info.sort_values('epoch', ascending=False).iloc[0]
        if checkpoints_dir is not None:
            checkpoint = Path(checkpoints_dir) / checkpoint_info['checkpoint_path']
            checkpoint = str(checkpoint)
        else:
            checkpoint = checkpoint_info['checkpoint_url']
        del df_model_checkpoints
        # Add the checkpoint path to the configuration
        model_config['checkpoint'] = checkpoint

    # Create and return the ComposerModel instance
    model = create_model_spvvs(**model_config)
    
    # Unwrap the model from the Composer wrapper
    model = model.module
    
    return model