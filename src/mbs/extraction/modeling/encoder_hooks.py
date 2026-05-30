import time
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Callable, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm.auto import tqdm

from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

from .projection import create_projector


# -----------------------------
# Utilities
# -----------------------------

def _default_get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Resolve a dotted module path like 'vision_model.encoder.layers.3'."""
    cur = model
    for part in name.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"Module path '{name}' not found at '{part}'. "
                           f"Available children: {list(dict(cur.named_children()).keys())}")
        cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Resolved object for '{name}' is not an nn.Module: {type(cur)}")
    return cur

def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()



# -----------------------------
# Hook-based encoder wrapper
# -----------------------------

class HookedEncoder(nn.Module):
    """
    Runs a backbone and captures activations from selected modules via forward hooks.
    """
    def __init__(
        self,
        backbone: nn.Module,
        feat_layers: Dict[str, str],
        include_output: bool = True,
        output_key: str = "output",
        module_resolver: Callable[[nn.Module, str], nn.Module] = _default_get_module_by_name,
        hook_on: Literal["output", "input"] = "output",
        detach: bool = True,
    ):
        super().__init__()
        if len(feat_layers) == 0:
            raise ValueError("feat_layers must be non-empty.")
        self.backbone = backbone
        self.feat_layers = feat_layers
        self.include_output = include_output
        self.output_key = output_key
        self.module_resolver = module_resolver
        self.hook_on = hook_on
        self.detach = detach

        self._handles: List[Any] = []
        self._buffer: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self) -> None:
        self.remove_hooks()
        for name, alias in self.feat_layers.items():
            mod = self.module_resolver(self.backbone, name)

            def _make_hook(layer_name: str, alias: str):
                def hook_fn(module, inputs, output):
                    val = inputs[0] if (self.hook_on == "input") else output
                    # Some HF modules return tuples (hidden_states, ...)
                    if isinstance(val, (tuple, list)):
                        # Take first tensor-like element by default
                        for v in val:
                            if torch.is_tensor(v):
                                val = v
                                break
                    if torch.is_tensor(val):
                        self._buffer[alias] = val.detach() if self.detach else val
                    else:
                        raise TypeError(f"Hooked value for layer '{layer_name}' is not a tensor: {type(val)}")
                return hook_fn

            handle = mod.register_forward_hook(_make_hook(name, alias))
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        self._buffer = {}
        out = self.backbone(*args, **kwargs)  # HF models may accept extra kwargs

        feats: Dict[str, torch.Tensor] = dict(self._buffer)

        if self.include_output:
            # HF can return dict-like objects, tuples, ModelOutput, tensors...
            if torch.is_tensor(out):
                feats[self.output_key] = out.detach() if self.detach else out
            elif isinstance(out, dict):
                # pick a sensible default
                if "logits" in out and torch.is_tensor(out["logits"]):
                    feats[self.output_key] = out["logits"].detach() if self.detach else out["logits"]
                elif "last_hidden_state" in out and torch.is_tensor(out["last_hidden_state"]):
                    feats[self.output_key] = out["last_hidden_state"].detach() if self.detach else out["last_hidden_state"]
                else:
                    # last resort: first tensor value
                    for v in out.values():
                        if torch.is_tensor(v):
                            feats[self.output_key] = v.detach() if self.detach else v
                            break
            elif isinstance(out, (tuple, list)):
                for v in out:
                    if torch.is_tensor(v):
                        feats[self.output_key] = v.detach() if self.detach else v
                        break
                    
        image_grid_thw = kwargs.get('image_grid_thw')
        if image_grid_thw is not None:
            g = image_grid_thw.to(dtype=torch.long)
            assert g.unique(dim=0).shape[0] == 1, f"image_grid_thw differs across batch: {g.tolist()}"

            image_grid_thw = image_grid_thw[0]
            # reshape visual features from (B*T, N) to (B, T, N)
            for k, v in feats.items():
                if v.ndim == 2:
                    # print(f"Reshaping features '{k}' from {v.shape} to (B, T, N) using image_grid_thw={tuple(image_grid_thw.tolist())}")
                    T = image_grid_thw[1] * image_grid_thw[2]
                    v = v.view(-1, T, v.shape[-1])
                    feats[k] = v

        return feats


# -----------------------------
# Projected feature extractor
# -----------------------------

class HookFeatureExtractor(nn.Module):
    def __init__(
        self,
        encoder: HookedEncoder,
        projectors: Dict[str, nn.Module],
        projector_backend: Literal["sklearn", "pytorch"] = "pytorch",
        flatten_features: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector_backend = projector_backend
        self.flatten = nn.Flatten(start_dim=1) if flatten_features else nn.Identity()

        if projector_backend == "pytorch":
            self.projectors = nn.ModuleDict(projectors)
        elif projector_backend == "sklearn":
            self.projectors = projectors
        else:
            raise ValueError("projector_backend must be 'sklearn' or 'pytorch'")

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        feats = self.encoder(*args, **kwargs)

        out: Dict[str, np.ndarray] = {}
        for feat_name, features in tqdm(feats.items(), desc="Projecting features", leave=False, total=len(feats)):
            flat = self.flatten(features)

            proj = self.projectors[feat_name]

            if self.projector_backend == "pytorch":
                out[feat_name] = _as_numpy(proj(flat))
            else:
                # print(feat_name, features.shape, flat.shape)
                out[feat_name] = proj.transform(_as_numpy(flat))

        return out


# -----------------------------
# Factory: create hook-based feature encoder + projectors
# -----------------------------

def create_hook_feature_encoder(
    backbone: nn.Module,
    transform: T.Compose,
    feat_layers: List[str],
    max_feature_dim: int = 0,
    token_reduce_method: Optional[Literal["cls", "avg", "max"]] = None,
    device: Union[str, torch.device] = "cpu",
    flatten_features: bool = True,
    include_output: bool = True,
    output_key: str = "output",
    hook_on: Literal["output", "input"] = "output",
    module_resolver: Callable[[nn.Module, str], nn.Module] = _default_get_module_by_name,
    forward_kwargs_for_shape_inference: Optional[Dict[str, Any]] = None,
    **model_config,
) -> nn.Module:
    """
    Hook-based alternative to create_feature_extractor pipeline.

    - feat_layers are module paths relative to 'backbone' (no "backbone." prefix)
    - forward_kwargs_for_shape_inference: pass HF-specific kwargs if needed (e.g., output_hidden_states=True)
    """

    assert len(feat_layers) > 0, "Feature layers must be specified."
    return_nodes = {f'backbone.{layer}': layer.replace(".", "-") for layer in feat_layers}

    backbone = backbone.to(device=device)
    backbone.eval()

    encoder = HookedEncoder(
        backbone=backbone,
        feat_layers=return_nodes,
        include_output=include_output,
        output_key=output_key,
        module_resolver=module_resolver,
        hook_on=hook_on,
        detach=True,
    )

    # ---- Infer feature shapes with a dummy input ----
    inp = transform(torch.randn(3, 256, 256)).to(device=device)
    # h, w = dummy_img.shape[-2], dummy_img.shape[-1]
    # print(f"Using image size: {h}x{w}")

    # inp = torch.randn(1, 3, h, w, device=device)

    fw_kwargs = forward_kwargs_for_shape_inference or {}
    with torch.no_grad():
        feats = encoder(**inp, **fw_kwargs)

    latent_shapes = {k: v.shape for k, v in feats.items() if torch.is_tensor(v)}
    print(f"Captured feature keys: {list(latent_shapes.keys())}")

    # ---- Create / load projectors ----
    start = time.time()
    projector_backend = model_config.get("projector_backend", "sklearn")
    projector_cache = Path(model_config.get("projector_cache", "cache/projectors/"))
    projector_type = model_config.get("projector_type", "sparse")
    random_seed = model_config.get("random_seed", 42)
    use_cached = bool(model_config.get("use_cached_projectors", False))
    projector_weights = None
    projector = None


    # Build list of layers we will project:
    proj_layers = [k for k in latent_shapes.keys() if k != output_key] + ([output_key] if include_output else [])

    projectors: Dict[str, Any] = {}
    if max_feature_dim > 0:
        print(f"Projecting features to max {max_feature_dim} dimensions using {projector_backend} {projector_type} random projection...")
        for layer_name in tqdm(proj_layers, desc="Building projectors", leave=False):
            input_dim = latent_shapes.get(layer_name, None)
            if input_dim is None:
                # if output wasn't captured etc.
                projectors[layer_name] = nn.Identity()
                continue
            
            numel = input_dim.numel()
            proj_out_dim = max_feature_dim if (max_feature_dim > 0 and numel > max_feature_dim) else 0

            if proj_out_dim <= 0 or layer_name == output_key:
                print(f"No projector created for layer {layer_name} with input dim {input_dim} ({numel}).")
                projectors[layer_name] = nn.Identity()
                continue
            print(f"Creating projector for layer {layer_name} with input dim {input_dim} ({numel}) and output dim {proj_out_dim} ({numel/proj_out_dim:.2f}x reduction).")

            projector_name = f"{projector_type}-in_{numel}-out_{proj_out_dim}-seed_{random_seed}"

            if projector_backend == "sklearn":
                projector_path = projector_cache / f"{projector_name}.pkl"
                if projector_path.exists() and use_cached:
                    print(f"Loading cached sklearn projector: {projector_path}")
                    with open(projector_path, "rb") as f:
                        projector = pickle.load(f)
                else:
                    print(f"Creating sklearn projector: {projector_name}")
                    if projector_type == "sparse":
                        projector = SparseRandomProjection(n_components=proj_out_dim, random_state=random_seed)
                    elif projector_type == "gaussian":
                        projector = GaussianRandomProjection(n_components=proj_out_dim, random_state=random_seed)
                    else:
                        raise ValueError("projector_type must be 'sparse' or 'gaussian'")

                    rng = np.random.default_rng(random_seed)
                    projector.fit(rng.normal(size=(1, numel)))

                    projector_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(projector_path, "wb") as f:
                        pickle.dump(projector, f)

                projectors[layer_name] = projector

            elif projector_backend == "pytorch":
                projector_path = projector_cache / f"{projector_name}.pt"


                if projector_path.exists() and use_cached:                    
                    print(f"Loading cached projector from {projector_cache} for {projector_name}.")

                    if projector_weights is None \
                        or projector_weights.shape[0]!=proj_out_dim \
                        or projector_weights.shape[1]!=numel:
                        print(f"Loading projector weights from {projector_path}.")   
                        projector_weights = torch.load(projector_path)

                else:
                    print(f"Creating new random projector for {projector_name} and saving to {projector_path}.")
                    projector_weights = None
                    
                if projector is None \
                    or not hasattr(projector, 'projection_layer') \
                    or projector.projection_layer.linear.weight.data.shape[0]!=proj_out_dim \
                    or projector.projection_layer.linear.weight.data.shape[1]!=numel:
                    projector = create_projector(
                        input_dim=numel,
                        output_dim=proj_out_dim,
                        token_reduce_method=token_reduce_method,
                        freeze_projection=True,
                        proj_type=projector_type,
                        random_seed=random_seed,
                        projector_weights=projector_weights,
                    )

                if projector_weights is None:
                    if not projector_path.parent.exists():
                        projector_path.parent.mkdir(parents=True, exist_ok=False)
                    torch.save(projector.projection_layer.linear.weight.data.cpu(), projector_path)

                projectors[layer_name] = projector
            else:
                raise ValueError("projector_backend must be 'sklearn' or 'pytorch'")
    else:
        print("No projection will be applied to features (max_feature_dim <= 0).")
        for layer_name in proj_layers:
            projectors[layer_name] = nn.Identity()

    print(f"Created projectors in {time.time() - start:.2f} seconds.")

    # # Ensure all keys exist in projectors
    # for k in latent_shapes.keys():
    #     projectors.setdefault(k, nn.Identity())

    feature_extractor_model = HookFeatureExtractor(
        encoder=encoder,
        projectors=projectors,
        projector_backend=projector_backend,
        flatten_features=flatten_features,
    )
    return feature_extractor_model
