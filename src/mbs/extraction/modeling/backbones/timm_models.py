import timm

def load_model_timm(model_id: str, pretrained=True, **kwargs):
        
    print(f"Loading model {model_id} from timm")
    model = timm.create_model(model_id, pretrained=pretrained)
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    return model, transform



