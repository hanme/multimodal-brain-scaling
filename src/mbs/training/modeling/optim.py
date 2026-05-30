import copy

import torch.nn as nn
from torch.optim import Optimizer

from timm.optim import create_optimizer_v2

def get_params_groups(model: nn.Module, **kwargs) -> list:
    encoder = []
    decoders = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        # if name.endswith(".bias") or len(param.shape) == 1:
        if 'bias' in name \
            or 'norm' in name \
            or 'bn' in name \
            or 'ln' in name:
            # print(f"Not regularizing {name}")
            not_regularized.append(param)
        elif "decoder" in name:
            decoders.append(param)
        else:
            encoder.append(param)
    return [
        # {'params': not_regularized, 'lr':kwargs.get('lr_encoder'), 'weight_decay': 0.},
        {'params': not_regularized, 'lr':kwargs.get('lr_encoder'), 'weight_decay': kwargs.get('wd_encoder')},
        {'params': encoder, 'lr':kwargs.get('lr_encoder'), 'weight_decay': kwargs.get('wd_encoder')},
        {'params': decoders, 'lr':kwargs.get('lr_decoder'), 'weight_decay': kwargs.get('wd_decoder')}, 
    ]
        

def create_optimizer(model: nn.Module, **kwargs) -> Optimizer:
        
        # We use different weight decay for encoder and decoder parameters
        # see get_params_groups() method
        opt_args = {
            'opt': kwargs.get('opt', 'adamw'),
            'momentum': kwargs.get('momentum', 0.9),
        }
        opt_args_all = copy.deepcopy(opt_args)
        opt_args_all['lr_encoder'] = kwargs.get('lr_encoder', 1e-4)
        opt_args_all['lr_decoder'] = kwargs.get('lr_decoder', 1e-4)
        opt_args_all['wd_encoder'] = kwargs.get('wd_encoder', 1e-4)
        opt_args_all['wd_decoder'] = kwargs.get('wd_decoder', 1e-4)

        if kwargs.get('opt_eps', None):
            opt_args['eps'] = kwargs['opt_eps']
        if kwargs.get('opt_betas', None) and opt_args['opt'] != 'sgd':
            opt_args['betas'] = kwargs['opt_betas']
        if 'lamb' in opt_args['opt']:
            clip_grad = kwargs['clip_grad'] if kwargs['clip_grad'] > 0 else 1.0
            opt_args['max_grad_norm'] = clip_grad
        print(f"Creating optimizer with parameters: {opt_args}")
            
        optimizer = create_optimizer_v2(
            get_params_groups(model, **opt_args_all), 
            **opt_args
        )
            
        print(f"Using optimizer: {optimizer.__class__.__name__}")
        print(f"Optimizer arguments: {opt_args_all}")
        
        return optimizer