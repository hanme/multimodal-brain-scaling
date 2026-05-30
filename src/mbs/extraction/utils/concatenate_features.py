import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import h5py


def main(args):
    files = sorted(Path(args.input_dir).glob("*.h5"))
    assert len(files) > 0, f"No h5 files found in input directory: {args.input_dir}"
    print(f"Found {len(files)} files to concatenate.")
    
    output_file = Path(args.output_file)
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=False)
    
    
    ##### 1 First, validate that all files have the same model_id, backbone_source, target_feature_layers, and config_json
    # and collect ids and get feature shapes for layers
    
    model_id = None
    backbone_source = None
    target_feature_layers = None
    config_json = None
    all_ids = []
    layer_dimensions = {}
    
    for file in tqdm(files, desc="Validating files and collecting stimulus IDs"):
        with h5py.File(file, "r") as hf:
            #### Validate attributes
            model_id_ = hf.attrs["model_id"]
            if model_id is None:
                model_id = model_id_
            else:
                assert model_id == model_id_, f"Model ID mismatch: {model_id} vs {model_id_}"
                
            backbone_source_ = hf.attrs["backbone_source"]
            if backbone_source is None:
                backbone_source = backbone_source_
            else:
                assert backbone_source == backbone_source_, f"Backbone source mismatch: {backbone_source} vs {backbone_source_}"
                
            target_feature_layers_ = json.loads(hf.attrs["target_feature_layers"])
            if target_feature_layers is None:
                target_feature_layers = target_feature_layers_
            else:
                assert set(target_feature_layers) == set(target_feature_layers_), f"Target feature layers mismatch: {target_feature_layers} vs {target_feature_layers_}"
                
            config_json_ = hf.attrs["config_json"]
            if config_json is None:
                config_json = config_json_
            else:
                assert config_json == config_json_, "Config JSON mismatch."
                
            ids = hf["ids"][:]
            all_ids.extend(ids)
            
            for layer in hf["features/"].keys():
                if args.drop_output_layer and layer.lower() in ["output", "logits"]:
                    print(f"Dropping output layer {layer} from file {file}")
                    continue
                layer_name = layer.split("features/")[-1]
                layer_shape = hf[f"features/{layer_name}"].shape[1:]  # Exclude the first dimension (number of samples)
                if layer_name not in layer_dimensions:
                    layer_dimensions[layer_name] = layer_shape
                else:
                    assert layer_dimensions[layer_name] == layer_shape, f"Feature shape mismatch for layer {layer_name}: {layer_dimensions[layer_name]} vs {layer_shape}"
            
    print(f"Validated {len(files)} files. Model ID: {model_id}, Backbone Source: {backbone_source}, Number of Target Feature Layers: {len(target_feature_layers)}")
    assert len(all_ids) == len(set(all_ids)), "Duplicate stimulus IDs found across files."
    print("Layer dimensions:", layer_dimensions)
    
    
    ### 2  Now allocate feature arrays
    with h5py.File(output_file, "w") as hf_out:
        for layer_name, layer_shape in layer_dimensions.items():
            total_samples = len(all_ids)
            dataset_shape = (total_samples, *layer_shape)
            hf_out.create_dataset(f"features/{layer_name}", shape=dataset_shape, dtype=np.float16)
            
            
    ##### 3 Now, fill in the feature arrays
    with h5py.File(output_file, "a") as hf_out:
        current_index = 0
        for file in tqdm(files, desc="Concatenating features"):
            with h5py.File(file, "r") as hf_in:
                ids = hf_in["ids"][:]
                num_samples = len(ids)
                
                for layer_name in layer_dimensions.keys():
                    features = hf_in[f"features/{layer_name}"][:]
                    hf_out[f"features/{layer_name}"][current_index:current_index+num_samples] = features
                    
                current_index += num_samples
                
        # Save combined IDs
        hf_out.create_dataset("ids", data=all_ids)
        
        # Save attributes
        hf_out.attrs["model_id"] = model_id
        hf_out.attrs["backbone_source"] = backbone_source
        hf_out.attrs["target_feature_layers"] = json.dumps(target_feature_layers)
        hf_out.attrs["config_json"] = config_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate features from batched files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing batched feature files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the concatenated feature file.")
    parser.add_argument("--drop_output_layer", action="store_true", help="Whether to drop the output layer from the concatenated features.")
    
    
    args = parser.parse_args()
    print(args)
    print(f"Concatenating features from {args.input_dir} to {args.output_file}")
    
    main(args)