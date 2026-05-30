import time
from collections import OrderedDict
from typing import Union, Dict, Literal, List
from pathlib import Path
import pickle

import numpy as np
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms as T
from torch.nn import functional as F

from tqdm.auto import tqdm

from .projection import create_projector

class FeatureExtractor(nn.Module):
    """
    A neural network model that builds an encoder to extract features from multiple layers.

    This model is designed to extract layer activations and save them for further processing.

    Attributes:

    Args:

    Methods:
    
    """
    

    def __init__(self, 
                 encoder: Union[GraphModule, torch.nn.Module],
                 projectors: Dict[str, nn.Module] = {},
                 projector_backend: Literal["sklearn", "pytorch"] = "pytorch",
                 flatten_features: bool = True
                 ):
        super(FeatureExtractor, self).__init__()
        
        
        self.encoder = encoder
        self.projectors = {}
        self.projector_backend = projector_backend
        self.flatten_features = nn.Flatten(start_dim=1) if flatten_features else nn.Identity()
        for proj_id, proj in projectors.items():
            self.projectors[proj_id] = proj

        if self.projector_backend == "pytorch":
            self.projectors = nn.ModuleDict(self.projectors)
        elif self.projector_backend == "sklearn":
            pass
        else:
            raise ValueError(f"Invalid projector implementation: {self.projector_backend}. Must be one of ['sklearn', 'pytorch']")

    def forward(self, input:torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Standard forward pass for a model that takes a single tensor input. It expects a single tensor input and returns a dictionary of outputs from different feature layers.
        """
        
        batch_size = input.shape[0]
        features = self.encoder(input)

        for proj_name, proj in tqdm(self.projectors.items(), desc="Projecting features", leave=False, total=len(self.projectors)):
            # print(f"Projecting features from layer {proj_name} using {self.projector_backend} projector {features[proj_name].shape}")
            if hasattr(proj, "forward"):
                features[proj_name] = proj(self.flatten_features(features[proj_name])).detach().cpu().numpy()
                # features[proj_name] = proj(features[proj_name]).reshape(batch_size, -1).detach().cpu().numpy()
                # features[proj_name] = proj(features[proj_name]).view(batch_size, -1)
            else:
                features[proj_name] = proj.transform(self.flatten_features(features[proj_name]).detach().cpu().numpy())
                # features[proj_name] = proj.transform(features[proj_name].reshape(batch_size, -1).detach().cpu().numpy())
                # features[proj_name] = proj.transform(features[proj_name])

        return features
        
    


def create_feature_encoder(
        backbone: nn.Module,
        transform: T.Compose,
        feat_layers:List[str]=[], 
        max_feature_dim: int = 0, 
        token_reduce_method: Literal[None, "cls", "avg", "max"] = None, 
        device: Union[str, torch.device]="cpu",
        flatten_features: bool = True,
        **model_config
    ) -> nn.Module:
    
    assert len(feat_layers) > 0, "Feature layers must be specified to create the feature encoder."

    # return_nodes = {f"backbone.{feat_layer}":f"feats_{feat_id}" for feat_id, feat_layer in enumerate(feat_layers)}
    return_nodes = {f"backbone.{feat_layer}":f"{feat_layer.replace('.', '-')}" for feat_id, feat_layer in enumerate(feat_layers)}
    neural_encoding_layers = list(return_nodes.values())
    print(f"Using feature layers: {return_nodes}")
    return_nodes["backbone"] = "output"  # Always include the final output as well
    
    # print(backbone)
    # for name, module in backbone.named_modules():
    #     print(name)
    # print(backbone)

    # Create the feature extractor
    if 'cornet' in model_config.get('arch', '') or 'cornet' in model_config.get('model_id', ''):
        # Loops inside CORnet blocks are not supported by the default feature extractor
        from scaling_primate_vvs.training.src.models.cornet import CORblock_S
        tracer_kwargs = {'leaf_modules': [CORblock_S]}
    else:
        tracer_kwargs = None

    backbone = backbone.to(device=device) # Some backbones may require being on the target device during tracing
    encoder = create_feature_extractor(backbone, return_nodes=return_nodes, tracer_kwargs=tracer_kwargs)


    # Get the dimensions of inputs to decoders
    n_channels = 3
    # image_resize_size = model_config.get('val_resize_size', 224)
    # image_size = model_config.get('image_size', image_resize_size)
    # h, w = image_size, image_size
    
    x = transform(T.ToPILImage()(torch.randn(3,224,224)))
    h, w = x.shape[1], x.shape[2]
    print(f"Using image size: {h}x{w}")

    model_input_dims = (n_channels, h, w)
    inp = torch.randn(1, *model_input_dims, device=device)
    out = encoder(inp)
    # latent_feats_dims = {k: v.numel() for k,v in out.items()}
    latent_feats_dims = {k: v.shape for k,v in out.items()}
    
    
    start = time.time()
    projectors = {}
    projector_backend = model_config.get('projector_backend', 'sklearn')
    projector_cache = Path(model_config.get('projector_cache', 'cache/projectors/'))
    projector_type = model_config.get('projector_type', 'sparse')
    random_seed = model_config.get('random_seed', 42)
    projector_weights = None
    projector = None
    if max_feature_dim > 0:
        for feat_name in tqdm(neural_encoding_layers):
            input_dim = latent_feats_dims[feat_name]
            proj_output_dim = max_feature_dim if input_dim.numel() > max_feature_dim else 0
            if proj_output_dim > 0:
                print(f"Creating projector for layer {feat_name} with input dim {input_dim} ({input_dim.numel()}) and output dim {proj_output_dim} ({input_dim.numel()/proj_output_dim:.2f}x reduction).")
            
                if projector_backend == "sklearn":

                    projector_name = f"{projector_type}-in_{input_dim.numel()}-out_{proj_output_dim}-seed_{random_seed}"
                    projector_path = projector_cache / f"{projector_name}.pkl"
                    if projector_path.exists() and model_config.get('use_cached_projectors', False):
                        print(f"Loading cached projector from {projector_cache} for {projector_name}.")
                        
                        with open(projector_path, 'rb') as f:
                            projector = pickle.load(f)
                    else:
                        print(f"Creating new random projector for {projector_name} and saving to {projector_path}.")

                        if projector_type == 'sparse':
                            projector = SparseRandomProjection(n_components=proj_output_dim, random_state=random_seed)
                        elif projector_type == 'gaussian':
                            projector = GaussianRandomProjection(n_components=proj_output_dim, random_state=random_seed)
                        else:
                            raise ValueError(f"Invalid random projector type: {projector_type}. Must be one of ['sparse', 'gaussian']")
                    
                        # Initialize the sklearn projector
                        rng = np.random.default_rng(random_seed)
                        x = rng.normal(size=(1, input_dim.numel()))
                        projector.fit(x)
                        
                        # Save the projector for future use
                        if not projector_path.parent.exists():
                            projector_path.parent.mkdir(parents=True, exist_ok=False)

                        with open(projector_path, 'wb') as f:
                            pickle.dump(projector, f)

                elif projector_backend == "pytorch":
                    projector_name = f"{projector_type}-in_{input_dim.numel()}-out_{proj_output_dim}-seed_{random_seed}"
                    projector_path = projector_cache / f"{projector_name}.pt"
                    if projector_path.exists() and model_config.get('use_cached_projectors', False):
                        print(f"Loading cached projector from {projector_cache} for {projector_name}.")

                        if projector_weights is None \
                            or projector_weights.shape[0]!=proj_output_dim \
                            or projector_weights.shape[1]!=input_dim.numel():
                            print(f"Loading projector weights from {projector_path}.")   
                            projector_weights = torch.load(projector_path)

                    else:
                        print(f"Creating new random projector for {projector_name} and saving to {projector_path}.")
                        projector_weights = None
                    
                    if projector is None \
                        or not hasattr(projector, 'projection_layer') \
                        or projector.projection_layer.linear.weight.data.shape[0]!=proj_output_dim \
                        or projector.projection_layer.linear.weight.data.shape[1]!=input_dim.numel():
                        projector = create_projector(
                            input_dim=input_dim, 
                            output_dim=proj_output_dim, 
                            token_reduce_method=token_reduce_method,
                            freeze_projection=True,
                            proj_type=projector_type,
                            random_seed=random_seed,
                            projector_weights=projector_weights
                        )
                    
                    if projector_weights is None:
                        # Save the projector for future use
                        if not projector_path.parent.exists():
                            projector_path.parent.mkdir(parents=True, exist_ok=False)

                        torch.save(projector.projection_layer.linear.weight.data, projector_path)
                else:
                    raise ValueError(f"Invalid projector implementation: {projector_backend}. Must be one of ['sklearn', 'pytorch']")
                
            else:
                print(f"No projector created for layer {feat_name} with input dim {input_dim} ({input_dim.numel()}).")
                projector = nn.Identity()
                
            projectors[feat_name] = projector

        print(f"Created projectors in {time.time()-start:.2f} seconds.")

    else:
        print("No projectors created since max_feature_dim <= 0.")
        projectors = {feat_name: nn.Identity() for feat_name in neural_encoding_layers}
    
    projectors['output'] = nn.Identity()  # No projection for the final output
        
    

    feature_extractor_model = FeatureExtractor(
        encoder=encoder, 
        projectors=projectors,
        projector_backend=projector_backend,
        flatten_features=flatten_features,
    )
    # print(feature_extractor_model)

    return feature_extractor_model

