import torch

from .backbones import create_backbone
from .encoder_feature_extractor import create_feature_encoder
from .encoder_hooks import create_hook_feature_encoder

def create_feature_extractor(**kwargs):
    
    backbone, transform = create_backbone(**kwargs)
    
    if kwargs.get("backbone_source") in ['hf']:
        print("Using hook-based feature extractor...")
        feature_extractor_model = create_hook_feature_encoder(backbone=backbone, transform=transform, **kwargs)
    else:
        print("Using standard feature extractor...")
        feature_extractor_model = create_feature_encoder(backbone=backbone, transform=transform, **kwargs)

    if kwargs.get('lora_config', "None") != "None":
        from .lora import create_lora_model

        lora_model = create_lora_model(feature_extractor_model, **kwargs['lora_config'])
        feature_extractor_model = lora_model.get_base_model() # Access the original backbone model from the LoRA model
        feature_extractor_model = feature_extractor_model.eval() # Set the model to evaluation mode
        
    if kwargs.get("backbone_checkpoint", None) is not None:
        print("Loading backbone weights from custom checkpoint...")
        checkpoint_path = kwargs["backbone_checkpoint"]
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # state_dict = checkpoint["state_dict"]
        
        if 'http' in checkpoint_path:
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint_path,
                check_hash=True,
                file_name=f"{kwargs['model_id']}.pt",
                map_location="cpu",
            )
            state_dict = checkpoint["state"]["model"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove 'module.' prefix if present
            
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
        
        if kwargs.get("backbone_source") in ['hf']:
            feature_extractor_model.encoder.backbone.backbone.load_state_dict(state_dict, strict=True)
        else:
            feature_extractor_model.encoder.backbone.load_state_dict(state_dict, strict=True)
        print(f"Loaded backbone weights from checkpoint: {checkpoint_path}")

    return feature_extractor_model, transform
