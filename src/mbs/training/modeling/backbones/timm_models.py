import timm

def load_model_timm(**model_config):
    
    arch = model_config.get('arch', None)
    pretrained_model_id = model_config.get('pretrained_model_id', None)
    training_mode = model_config.get('training_mode', 'finetune')
    pretrained = True if training_mode == 'finetune' else False
    
    if pretrained:
        assert pretrained_model_id, \
            "Pretrained model ID must be provided for finetuning"
        print(f"Loading model {pretrained_model_id} from timm")
        model = timm.create_model(pretrained_model_id, pretrained=True)
    else:
        print(f"Loading model {arch} from timm")
        model = timm.create_model(arch, pretrained=False)
    
    return model


