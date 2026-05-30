import json

from .backbones import create_backbone
from .encoder_decoder import create_encoder_decoder
from .wrapper import LightningWrapper
from .lora import create_lora_model
# from ._wrapper import LightningWrapper


def create_lightning_model(**kwargs):
    
    # target_brain_regions = kwargs.get('target_brain_regions', [])
    # target_brain_regions = kwargs.get('data_neural_regions', "None")
    # target_brain_regions = [] if target_brain_regions in ["", "None", None] else target_brain_regions.split(",")
    target_brain_regions = kwargs['rois']
    
    ## Need to keep it or multiple definitions will occur
    feat_layers_label = kwargs.get('feat_layers_label', 'spvvs')
    feat_layers = kwargs.pop('feat_layers', {}).get(feat_layers_label, {})
    with open(kwargs['layer_commitments'], 'r') as f:
        layer_commitments = json.load(f)
    model_layer_commitments = layer_commitments[kwargs['pretrained_model_id']]
    dataset_name = kwargs['layer_commitment_dataset']
    assert dataset_name in model_layer_commitments, f"Dataset {dataset_name} not found in layer commitments"
    feat_layers = {region:model_layer_commitments[dataset_name][region]['layer_name'] for region in target_brain_regions}
    print("Using feature layers:", feat_layers)


    backbone = create_backbone(**kwargs)
    encoder_decoder_model = create_encoder_decoder(backbone, feat_layers=feat_layers, **kwargs)
    
    if kwargs.get('config_lora', False):
        lora_model = create_lora_model(encoder_decoder_model, **kwargs['config_lora'])
        lora_model.print_trainable_parameters()
        encoder_decoder_model = lora_model.get_base_model() # Access the original backbone model from the LoRA model
    
    #### Flatten dictionary of variables to kwargs
    # Unpack the loss functions
    loss_func_image = kwargs.get('loss_fn', 'cross_entropy')
    # loss_func_neural_behavior = kwargs.get('decoders', {}).get('neural_behavior', {}).get('loss_fn', 'cross_entropy')
    neural_loss_fn = kwargs.get('neural_loss_fn', 'mse')
    loss_func_neural_response = {}
    subjects = kwargs['subjects']
    for subj in subjects:
        for region in target_brain_regions:
            key = f"subj_{subj}-roi_{region}"
            if neural_loss_fn is not None:
                loss_func_neural_response[key] = neural_loss_fn
    kwargs.update({
        'loss_func_image': loss_func_image,
        'loss_func_neural_response': loss_func_neural_response,
        # 'loss_func_neural_behavior': loss_func_neural_behavior,
    })
    
    # Unpack the loss weights
    lambda_image = kwargs.get('loss_weights', {}).get('image', 1.0)
    lambda_neural_response = kwargs.get('loss_weights', {}).get('neural_response', 1.0)
    # lambda_neural_behavior = kwargs.get('loss_weights', {}).get('neural_behavior', 1.0)
    kwargs.update({
        'lambda_image': float(lambda_image),
        'lambda_neural_response': float(lambda_neural_response),
        # 'lambda_neural_behavior': float(lambda_neural_behavior)
    })
    ####

    model = LightningWrapper(encoder_decoder_model, **kwargs)


    return model