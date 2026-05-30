import os

import torch.nn as nn 
import torchvision.transforms as T

from scaling_primate_vvs import load_model as load_model_

RESIZE_SIZE = {
    'efficientnet_b0': 256,
    'efficientnet_b1': 255,
    'efficientnet_b2': 288,
}

IMAGE_SIZE = {
    'efficientnet_b0': 256,
    'efficientnet_b1': 240,
    'efficientnet_b2': 288,
}

def create_transform(model_id: str, **kwargs) -> nn.Module:
    """
    Create a transform for the input images based on the model architecture.

    Args:
        model_id (str): The ID of the model for which to create the transform.

    Returns:
        nn.Module: The created transform.
    """
    # Determine default sizes based on model architecture
    archs = [arch for arch in RESIZE_SIZE.keys() if arch in model_id]
    if len(archs) == 1:
        arch = archs[0]
    else:
        arch = 'default'
    resize_size_default = RESIZE_SIZE.get(arch, 256)
    image_size_default = IMAGE_SIZE.get(arch, 224)

    # Override with kwargs if provided
    resize_size = kwargs.get("resize_size", resize_size_default)
    image_size = kwargs.get("image_size", image_size_default)

    # Create the transform
    transform = T.Compose([
        T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform


def load_model_spvvs(model_id: str, checkpoints_dir: str = os.getenv("SCALING_MODELS_CKPTS_DIR"), **kwargs) -> nn.Module:
    """
    Load a pretrained model from the Scaling Primate VVS library.

    Args:
        model_id (str): The ID of the pretrained model to load.
        checkpoints_dir (str, optional): The directory containing model checkpoints.

    Returns:
        nn.Module: The created model instance.
    """
    
    
    # Load the model
    try:
        model = load_model_(model_id=model_id, checkpoints_dir=checkpoints_dir)
    except FileNotFoundError as e:
        print(
            f"Could not find the checkpoint for model_id '{model_id}' in directory '{checkpoints_dir}'. "
            "Downloading the pretrained weights from S3."
        )
        model = load_model_(model_id=model_id, checkpoints_dir=None)
        

    # Unwrap the model from the Composer wrapper
    model = model.module
    
    transform = create_transform(model_id, **kwargs)
    
    return model, transform