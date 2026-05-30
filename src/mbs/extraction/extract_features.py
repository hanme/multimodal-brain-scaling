from pathlib import Path
import argparse
import json
from typing import Tuple, List


import numpy as np
import torch
import h5py

from tqdm.auto import tqdm

from .data import create_dataloader
from .modeling import create_feature_extractor
from .utils import str2bool, load_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from a dataset using a feature extractor model.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier.")
    parser.add_argument("--backbone_source", choices=["timm", "spvvs", "hf"], default="spvvs", help="Source of the backbone model.")
    parser.add_argument("--layer_config", default="None", help="Layer configuration string for custom models.")
    parser.add_argument("--lora_config", default="None", help="LoRA configuration YAML file path.")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Path to the backbone model checkpoint. Will use default weights if not provided.")
    parser.add_argument("--target_feature_layers", type=str, required=False, help="Path to a JSON file specifying target feature layers.")
    parser.add_argument("--committed_extraction_layers", type=str, required=False, help="Path to a JSON file specifying committed extraction layers.")
    parser.add_argument("--stimulus_set_id", type=str, required=False, help="Stimulus-set key to use with --committed_extraction_layers. Defaults to output_dir.name.")
    parser.add_argument("--layer_commitments_model_id", type=str, required=False, help="Model ID for committed extraction layers, if different from model_id.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset directory or stimulus set id.")
    parser.add_argument("--dataset_type", type=str, choices=["things", "h5", "brain_score"], help="Type of dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted features.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--flatten_features", type=str2bool, default=True, help="Whether to flatten features before projection.")
    parser.add_argument("--max_feature_dim", type=int, default=30_000, help="Maximum feature dimension. Use 0 for no projection.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Whether to overwrite existing output directory.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the model inputs.")
    parser.add_argument("--prompt", type=str, default=None, help="Optional fixed prompt for HF multimodal backbones. Applied to every sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--projector_backend", type=str, choices=["pytorch", "sklearn"], default="sklearn", help="Backend for random projection.")
    parser.add_argument("--projector_type", type=str, choices=["sparse", "gaussian"], default="sparse", help="Type of random projection.")
    parser.add_argument("--projector_cache", type=str, default="cache/projectors/", help="Path to cache directory for projectors.")
    parser.add_argument("--use_cached_projectors", type=str2bool, default=False, help="Whether to use cached projectors if available.")
    parser.add_argument("--append_features", type=str2bool, default=False, help="Whether to append features to existing ones. Opens in 'a' mode.")
    args = parser.parse_args()
    return args

def main(args):
    
    config = vars(args)
    config["random_seed"] = args.seed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)
    
    if args.lora_config != "None":
        config["lora_config"] = load_yaml(args.lora_config)
    
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        
    assert args.target_feature_layers is not None or args.committed_extraction_layers is not None, \
        "Either --target_feature_layers or --committed_extraction_layers must be provided."
    assert not (args.target_feature_layers is not None and args.committed_extraction_layers is not None), \
        "Only one of --target_feature_layers or --committed_extraction_layers should be provided."
        
    assert args.flatten_features or args.max_feature_dim <= 0, \
        "Original feature shape can only be preserved when no projection is applied (max_feature_dim <= 0)."
    

    if args.committed_extraction_layers is not None:
        comitted_extraction_layers_path = Path(args.committed_extraction_layers)
        with open(comitted_extraction_layers_path, 'r') as f:
            comitted_extraction_layers = json.load(f)
        layer_commitments_model_id = args.model_id
        if args.layer_commitments_model_id is not None:
            layer_commitments_model_id = args.layer_commitments_model_id
        assert layer_commitments_model_id in comitted_extraction_layers, \
            f"Model ID {layer_commitments_model_id} not found in layer commitments."
        target_feature_layers = comitted_extraction_layers[layer_commitments_model_id]
        stimulus_set_id = args.stimulus_set_id or output_dir.name
        target_layers = target_feature_layers[stimulus_set_id]
    else:
        with open(args.target_feature_layers, 'r') as f:
            target_feature_layers = json.load(f)
        target_layers = [layer['name'] for layer in target_feature_layers]
    
    # Create feature extractor model
    feature_extractor, transform = create_feature_extractor(
        feat_layers=target_layers,
        **config
    )
    feature_extractor.eval()
    dtype = getattr(torch, args.dtype)
    feature_extractor = feature_extractor.to(device=device, dtype=dtype)
    
    # Create dataloader
    dataloader = create_dataloader(
        data_root=args.data_root,
        dataset_type=args.dataset_type,
        transform=transform,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    
    # all_features = []
    

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            features_path = output_dir / f"feats_{args.max_feature_dim}-bs_{args.batch_size}-batch_{batch_idx}-seed_{args.seed}.h5"  
            if features_path.exists() and not args.overwrite and not args.append_features:
                print(f"Features file {features_path} already exists. Skipping...")
                continue
            
            if (isinstance(batch, List) and torch.is_tensor(batch[0])) or (isinstance(batch, Tuple) and torch.is_tensor(batch[0])):
                inputs = batch[0].to(device=device, dtype=dtype)  # Assuming batch[0] contains the input images
                features = feature_extractor(inputs)
            elif isinstance(batch, Tuple) and isinstance(batch[0], dict):
                inputs = {}
                for k, v in batch[0].items():
                    if not torch.is_tensor(v):
                        inputs[k] = v
                    else:
                        v = v.to(device=device)
                        # keep index-like tensors integer
                        if v.is_floating_point():
                            v = v.to(dtype=dtype)
                        else:
                            # optional: ensure ids/masks are long
                            if k in ("input_ids", "image_grid_thw", "position_ids", "attention_mask"):
                                v = v.long()
                        inputs[k] = v
                # for k, v in inputs.items():
                    # print(f"Input '{k}': shape={tuple(v.shape) if torch.is_tensor(v) else 'N/A'}, dtype={v.dtype if torch.is_tensor(v) else 'N/A'}")
                features = feature_extractor(**inputs)
            else:
                raise ValueError("Unsupported batch format. Expected a tuple of (inputs, ids) where inputs is either a tensor or a dict of tensors.",
                                 f"Got batch of type {type(batch)} with contents: {batch}")
            
            if args.append_features:
                open_mode = "a"
            else:
                open_mode = "w"

            with h5py.File(features_path, open_mode) as hf:
                for feat_name, feat_array in features.items():
                    key = f"features/{feat_name}"
                    if key in hf:
                        print(f"Overwriting existing features for layer {feat_name} in {features_path}.")
                        del hf[key]
                    hf.create_dataset(key, data=feat_array, dtype=np.float16)
                
                if "ids" not in hf:
                    hf.create_dataset("ids", data=batch[1])
                else:
                    ids_ = [id_.decode("utf-8") if isinstance(id_, bytes) else id_ for id_ in hf["ids"][:]]
                    assert np.array_equal(ids_, batch[1]), "IDs in the existing file do not match the current batch IDs."
                    
                if "model_id" not in hf.attrs:
                    hf.attrs["model_id"] = args.model_id
                else:
                    assert hf.attrs["model_id"] == args.model_id, "Model ID in the existing file does not match the current model ID."

                if "backbone_source" not in hf.attrs:
                    hf.attrs["backbone_source"] = args.backbone_source
                else:
                    assert hf.attrs["backbone_source"] == args.backbone_source, "Backbone source in the existing file does not match the current backbone source."

                if "target_feature_layers" in hf.attrs:
                    target_layers_prev = sorted(json.loads(hf.attrs["target_feature_layers"]))
                    target_layers_all = sorted(target_layers + target_layers_prev)
                    print("Previous target layers:", target_layers_prev)
                    print("Current target layers:", target_layers_all)
                    hf.attrs["target_feature_layers"] = json.dumps(target_layers_all)
                else:
                    hf.attrs["target_feature_layers"] = json.dumps(target_layers)

                if "config_json" in hf.attrs:
                    del hf.attrs["config_json"]
                hf.attrs["config_json"] = json.dumps(config)

    print("Feature extraction completed.")


def cli():
    args = parse_args()
    print("Arguments:", args)
    main(args)


if __name__ == "__main__":
    cli()
