import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mbs.core import find_repo_root


KNOW_BENCHMARKS = [
    "bs_fz",
    "bs_mh",
    "tvsd",
    "things_fmri",
    "things_eeg1",
    "things_eeg2",
    "things_meg",
    "nsd_func1pt8mm_individualROIs",
    "nsd_fsaverage_individualROIs",
    "nsd_nativesurface_individualROIs",
]

BENCHMARKS_WITH_WHOLE_BRAIN_ROI = [
    "things_fmri",
    "things_eeg1",
    "things_eeg2",
    "things_meg",
    "nsd_func1pt8mm_individualROIs",
    "nsd_fsaverage_individualROIs",
    "nsd_nativesurface_individualROIs",
]

KNOWN_METRICS = [
    "cv_score",
    "pearsonr",
    "r2",
    "evs",
    "mae",
    "mse",
    "approx_exp_var",
    "noise_ceiling",
    "pearsonr_nc",
    "approx_exp_var_nc",
    "rsa_c_train",
    "rsa_ve_train",
    "cka_c_train",
    "cka_ve_train",
    "rsa_c_test",
    "rsa_ve_test",
    "cka_c_test",
    "cka_ve_test",   
]


STIMULUS_SET_TO_BENCHMARK = {
    "FreemanZiemba2013.aperture-public": ["bs_fz"],
    "hvm-public": ["bs_mh"],
    "object_images": [
        "tvsd",
        "things_fmri",
        "things_eeg1",
        "things_eeg2",
        "things_meg",
    ],
    "nsd_stimuli": [
        "nsd_func1pt8mm_individualROIs", 
        "nsd_fsaverage_individualROIs", 
        "nsd_nativesurface_individualROIs"
    ],
}


if __name__ == "__main__":
    repo_root = find_repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer_eval_results_dir",
        type=str,
        default=str(repo_root / "results" / "proj_feats"),
        help="Path to the layer evaluation results directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(repo_root / "configs" / "evaluation" / "layer_commitment"),
        help="Directory to save the layer commitment file.",
    )
    parser.add_argument(
        "--valid_filenames",
        default=["bs_fz", "bs_mh", "tvsd", "things_fmri", "things_eeg1", "things_eeg2", "things_meg", "nsd_func1pt8mm_individualROIs"],
        nargs="*",
        help="List of valid filenames for layer evaluation results.",
    )
    parser.add_argument(
        "--eval_metric",
        default="cv_score",
        help="Metric used for selecting the best layer.",
    )
    parser.add_argument(
        "--eval_metric_lower_is_better",
        default=False,
        action="store_true",
        help="Whether a lower value of the evaluation metric indicates better performance.",
    )
    parser.add_argument(
        "--target_model_id",
        default=None,
        type=str,
        help="Target model identifier, if specified, only process this model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    args = parser.parse_args()
    
    selected_benchmarks = set(args.valid_filenames)
    assert args.eval_metric in KNOWN_METRICS, f"Evaluation metric {args.eval_metric} is not recognized. Known metrics are: {KNOWN_METRICS}"
    assert selected_benchmarks.issubset(set(KNOW_BENCHMARKS)), f"Some selected benchmarks are not recognized. Known benchmarks are: {KNOW_BENCHMARKS}"
    print(f"Selected benchmarks: {selected_benchmarks}")

    layer_eval_results_dir = Path(args.layer_eval_results_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)


    if args.target_model_id:
        model_folder = layer_eval_results_dir / args.target_model_id
        assert model_folder.exists(), f"Model folder {model_folder} does not exist."
        model_folders = [model_folder]
    else:
        model_folders = sorted(list(layer_eval_results_dir.iterdir()))
        print(f"Found {len(model_folders)} model folders.")
        
    
    layer_commitments = {}
    config = vars(args)
    config.update({
        "time_created": pd.Timestamp.now().isoformat(),
    })
    layer_commitments["config"] = config
    
    uniqe_layers_per_model = []
    for model_folder in tqdm(model_folders, desc="Processing models"):
        model_id = model_folder.name
        model_commitments = {}
        
        for benchmark_file in tqdm(list(model_folder.glob("*.json")), desc=f"Processing benchmarks for model {model_id}", leave=False):
            
            benchmark_roi_commitments = {}
            
            benchmark_name = benchmark_file.stem
            if benchmark_name not in selected_benchmarks:
                if args.verbose:
                    print(f"Skipping benchmark {benchmark_name} for model {model_id} as it is not in the selected benchmarks.")
                continue
            
            with open(benchmark_file, "r") as f:
                benchmark_results = json.load(f)
            results = benchmark_results["results"]
            df_results = pd.DataFrame(results)
            
            # Process each benchmark and ROI individually
            for (model_id, benchmark_name, roi), df_roi_group in tqdm(df_results.groupby(["model_id", "benchmark_name", "roi"]), desc=f"Processing ROIs for benchmark {benchmark_name}", leave=False):
                # Aggregate results by layer name and position across subjects
                df_group = df_roi_group.groupby(["layer_name", "layer_position"]).agg({args.eval_metric: "mean"})
                df_group = df_group.reset_index()
                df_group = df_group.sort_values(by=args.eval_metric, ascending=args.eval_metric_lower_is_better)
                max_layer_pos = df_group["layer_position"].max()
                
                # Select the best layer
                best_row = df_group.iloc[0]
                benchmark_roi_commitments[roi] = {
                    "layer_name": str(best_row["layer_name"]),
                    "layer_position": int(best_row["layer_position"]),
                    "layer_position_normalized": float(best_row["layer_position"] / max_layer_pos),
                    "score": float(best_row[args.eval_metric]),
                }
                
            if  benchmark_name in BENCHMARKS_WITH_WHOLE_BRAIN_ROI and 'whole_brain' not in benchmark_roi_commitments:
                if args.verbose:
                    print(f"Whole brain ROI results not found for benchmark {benchmark_name} in model {model_id}. Selecting best layer across all ROIs.")
                
                # Select the layer with the best average score across all ROIs
                df_group = df_results.groupby(["layer_name", "layer_position"]).agg({args.eval_metric: "mean"})
                df_group = df_group.reset_index()
                df_group = df_group.sort_values(by=args.eval_metric, ascending=args.eval_metric_lower_is_better)
                max_layer_pos = df_group["layer_position"].max()
                
                best_row = df_group.iloc[0]
                benchmark_roi_commitments['whole_brain'] = {
                    "layer_name": str(best_row["layer_name"]),
                    "layer_position": int(best_row["layer_position"]),
                    "layer_position_normalized": float(best_row["layer_position"] / max_layer_pos),
                    "score": float(best_row[args.eval_metric]),
                }
            
            model_commitments[benchmark_name] = benchmark_roi_commitments
            
        layer_commitments[model_id] = model_commitments
        uniqe_layers_per_model.append(len(set(
            (commitment["layer_name"], commitment["layer_position"])
            for benchmark_commitments in model_commitments.values()
            for commitment in benchmark_commitments.values()
        )))
        
    output_file = output_dir / f"layer_commitments.json"
    with open(output_file, "w") as f:
        json.dump(layer_commitments, f, indent=4)
        
    
    print("Constructing committed extraction layers per stimulus set...")
    commited_extraction_layers = {}
    commited_extraction_layers["config"] = config
    for model_id, model_commitments in tqdm(layer_commitments.items()):
        if model_id == "config":
            continue
        
        model_extraction_layers = {}
        
        for stimulus_set, benchmarks in STIMULUS_SET_TO_BENCHMARK.items():
            committed_layers = set()
            for benchmark in benchmarks:
                if benchmark in model_commitments:
                    for roi_commitment in model_commitments[benchmark].values():
                        committed_layers.add(roi_commitment["layer_name"])
            
            model_extraction_layers[stimulus_set] = sorted(list(committed_layers))
        commited_extraction_layers[model_id] = model_extraction_layers
    output_file = output_dir / f"committed_extraction_layers.json"
    with open(output_file, "w") as f:
        json.dump(commited_extraction_layers, f, indent=4)
    
    print(f"Saved layer commitments to {output_file}")
    print(f"Average number of unique layers committed per model: {np.mean(uniqe_layers_per_model):.2f}, std: {np.std(uniqe_layers_per_model):.2f}, min: {np.min(uniqe_layers_per_model)}, max: {np.max(uniqe_layers_per_model)}")
