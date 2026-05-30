from typing import Literal

import numpy as np

import torch
import torch.nn as nn

from .projection_layer import ProjectionLayer


class ProjectionModule(nn.Module):

    def __init__(
            self, 
            input_feats_dim:int,
            token_reduce_method:Literal[None, "cls", "avg", "max"] = None,
            proj_feats_dim:int = 1024,
            proj_type:Literal["gaussian", "sparse"] = "gaussian",
            proj_bias:bool = False,
            freeze_projection:bool = True,
            random_seed:int = 42,
            device:torch.device = None,
            dtype:torch.dtype = torch.float32,
            projector_weights: torch.Tensor = None
            ):
        super(ProjectionModule, self).__init__()
        self.input_feats_dim = input_feats_dim
        self.token_reduce_method = token_reduce_method
        self.proj_feats_dim = proj_feats_dim
        self.proj_type = proj_type
        self.proj_bias = proj_bias
        self.freeze_projection = freeze_projection
        self.random_seed = random_seed
        self.device = device
        self.dtype = dtype

        if self.token_reduce_method is None:
            # Flatten the output feature dimensions
            self.token_reduce_method = None
            self.input_feats_dim = self.input_feats_dim if isinstance(self.input_feats_dim, int) else np.prod(self.input_feats_dim)
        elif self.token_reduce_method in ["cls", "avg", "max"]:
            # For cls, avg, and max, we will reduce the feature dimensions to a single token
            assert len(self.input_feats_dim) == 3, \
                "All encoder feature dimensions must be 3D tensors (batch_size, tokens, features) to apply token reduction methods." \
                    f"Got: {self.input_feats_dim}"
            # Set the token reduction method
            self._set_token_reduce_method(self.token_reduce_method)
            self.input_feats_dim = self.input_feats_dim[2]  # Keep only the feature dimension
        else:
            raise ValueError(f"Invalid token_reduce_method: {self.token_reduce_method}. "
                             "Must be one of [None, 'cls', 'avg', 'max']")
        

        self.projection_layer = self._create_projection_layer(
                input_dim=self.input_feats_dim,
                output_dim=self.proj_feats_dim,
                projector_weights=projector_weights
            )
        
        
    def _set_token_reduce_method(self, method: Literal["cls", "avg", "max"]):
        match method:
            case "cls":
                # For cls, we will use the first token as the representative feature
                self.token_reduce_method = lambda x: x[:, 0, :]  # Assuming the first token is the cls token
            case "avg":
                # For avg, we will average the tokens across the sequence dimension
                self.token_reduce_method = lambda x: x.mean(dim=1)  # Average across tokens
            case "max":
                # For max, we will take the maximum across the sequence dimension
                self.token_reduce_method = lambda x: x.max(dim=1).values  # Max across tokens
            case _:
                raise ValueError(f"Invalid token_reduce_method: {method}. "
                                 "Must be one of ['cls', 'avg', 'max']")
                
                
    def _create_projection_layer(self, input_dim: int, output_dim: int, projector_weights: torch.Tensor = None):
        if output_dim > 0:
            return ProjectionLayer(
                in_features=input_dim,
                out_features=output_dim,
                init=self.proj_type,
                bias=self.proj_bias,
                freeze=self.freeze_projection,
                seed=self.random_seed,
                device=self.device,
                dtype=self.dtype,
                W=projector_weights
            )
        elif output_dim == 0:
            return nn.Identity()
        else:
            raise ValueError(f"Invalid output_dim: {output_dim}. Must be >= 0.")
        

    def forward(self, feats):
        
        if self.token_reduce_method is not None:
            # Apply the token reduction method to each feature map
            feats =  self.token_reduce_method(feats) if isinstance(feats, torch.Tensor) else feats

        projected_features =  self.projection_layer(feats.reshape(feats.shape[0], -1))

        return projected_features



def create_projector(
    input_dim: int, 
    output_dim: int, 
    proj_type: Literal["gaussian", "sparse"] = "gaussian", 
    token_reduce_method: Literal[None, "cls", "avg", "max"] = None, 
    freeze_projection: bool = True, 
    random_seed: int = 42,
    projector_weights: torch.Tensor = None
) -> ProjectionModule:
    return ProjectionModule(
        input_feats_dim=input_dim,
        proj_feats_dim=output_dim,
        proj_type=proj_type,
        token_reduce_method=token_reduce_method,
        freeze_projection=freeze_projection,
        random_seed=random_seed,
        projector_weights=projector_weights
    )