from collections import OrderedDict

import torch.nn as nn


def create_decoder(input_dim:int, output_dim:int, hidden_dim:int=512, num_hidden_layers:int=0, activation:nn.Module=nn.Sigmoid(), is_frozen:bool=False) -> nn.Sequential:
    """Creates a decoder with configurable architecture.
    This function builds a decoder with a flexible number of hidden layers
    and customizable dimensions. The decoder consists of fully connected layers with
    specified activation functions between them.
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features 
        hidden_dim (int, optional): Number of neurons in hidden layers. Defaults to 512.
        num_hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.Sigmoid().
        is_frozen (bool, optional): If True, freeze the decoder parameters. Defaults to False.
    Returns:
        nn.Sequential: A PyTorch Sequential model representing the decoder network
    Examples:
        >>> decoder = create_decoder(784, 10, hidden_dim=256, num_hidden_layers=2)
        >>> decoder = create_decoder(100, 20, activation=nn.ReLU())
        >>> decoder = create_decoder(50, 10, num_hidden_layers=0) # Linear decoder
    """

    if num_hidden_layers == -1:
        decoder = nn.Sequential(
            OrderedDict([
                ('decoder_input', nn.Identity()),
            ])
        )
    
    elif num_hidden_layers == 0:
        decoder = nn.Sequential(
            OrderedDict([
                ('decoder_input', nn.Identity()),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(input_dim, output_dim))
            ])
        )
    elif num_hidden_layers == 1:
        decoder = nn.Sequential(
            OrderedDict([
                ('decoder_input', nn.Identity()),
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(input_dim, hidden_dim)),
                ('activation1', activation),
                ('fc2', nn.Linear(hidden_dim, output_dim))
            ])
        )
    else:
        layers = OrderedDict([
            ('decoder_input', nn.Identity()),
            ('flatten', nn.Flatten()),
            ("fc1", nn.Linear(input_dim, hidden_dim)), 
            ("activation1", activation)
        ])
        for i in range(1, num_hidden_layers):
            layers[f"fc{i+1}"] = nn.Linear(hidden_dim, hidden_dim)
            layers[f"activation{i+1}"] = activation
        layers[f"fc{num_hidden_layers+1}"] = nn.Linear(hidden_dim, output_dim)
        decoder = nn.Sequential(layers)
        
    if is_frozen:
        for param in decoder.parameters():
            param.requires_grad = False
        
    return decoder