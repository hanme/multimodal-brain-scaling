from pathlib import Path
import argparse
import json
import pickle
import copy

import numpy as np
import torch

from tqdm.auto import tqdm


from tqdm.auto import tqdm
# from tqdm import tqdm

from .attn_probe.model import ProbeConfig
from .attn_probe.engine import TrainConfig, train_single_roi_single_layer, evaluate

from .utils import str2bool
from .utils.evaluation_helpers import (
    compute_metrics,
    compute_rsa_cka,
    pearsonr_score,
    load_layer_features,
    load_neural_metadata,
    load_neural_data,
    get_pipeline,
    save_results,
    load_yaml,
)



def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from a dataset using a feature extractor model.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier.")
    parser.add_argument("--finetuned_model_id", type=str, default=None, required=False, help="Finetuned model identifier.")
    parser.add_argument("--layer_commitments", type=str, required=True, help="Path to a JSON file specifying layer commitments.")
    parser.add_argument("--layer_commitments_model_id", type=str, required=False, help="Model ID for layer commitments, if different from model_id.")
    parser.add_argument("--data_hdf5_path", type=str, required=True, help="Path to the dataset HDF5 file.")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to the features directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted features.")
    parser.add_argument("--debug_mode", type=str2bool, default=False, help="Whether to run in debug mode.")

    parser.add_argument("--probes_input_dir", type=str, default=None, help="Provide a directory to load probe models, otherwise probes are fitted anew.")
    parser.add_argument("--probes_output_dir", type=str, default=None, help="Provide a directory to save probe models, otherwise probes are not saved.")
    parser.add_argument("--exclude_whole_brain", type=str2bool, default=False, help="Whether to exclude whole brain ROI.")
    parser.add_argument("--use_gpu", type=str2bool, default=False, help="Whether to use GPU for regression.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Whether to overwrite existing output directory.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the model inputs.")
    
    # Subjects, comma-separated list
    parser.add_argument("--probe_config_subject", type=str, choices=["individual", "shared"], default="individual", help="Whether to use individual or shared probes per subject.")
    parser.add_argument("--data_pct", type=int, default=100, help="Percentage of data to fit on.")
    # parser.add_argument("--subjects", type=str, default=None)
    
    # Probe config
    parser.add_argument("--in_dim", type=int, required=False)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--token_encoder_layers", type=int, default=0)
    parser.add_argument("--num_latents", type=int, default=64)
    parser.add_argument("--cross_attn_layers", type=int, default=2)
    parser.add_argument("--query_self_attn", action="store_true")
    parser.add_argument("--pos_mode", type=str, choices=["none", "mlp_coords", "sin", "learned"], default="none")
    parser.add_argument("--head_type", type=str, choices=["linear", "lowrank", "shallow_mlp"], default="linear")
    parser.add_argument("--head_rank", type=int, default=256)
    parser.add_argument("--head_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--head_mlp_dropout", type=float, default=0.0)

    # Train config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--seed", type=int, default=0)
    
    
    config_parser = argparse.ArgumentParser(description='Attention probe configuration', add_help=False)
    config_parser.add_argument("--probe_config_file", type=str, default=None, help="Path to a YAML file specifying probe configuration.")
    args_config, remaining = config_parser.parse_known_args()
    
    # Load config file
    if args_config.probe_config_file:
        print(f"Loading probe config from {args_config.probe_config_file}")
        cfg = load_yaml(args_config.probe_config_file)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    return args


def fit_n_eval_attn_probe(
    args: argparse.Namespace,
    data_hdf5_path: Path,
    features_dir: Path,
    subjects: list,
    probe_fitting_subjects: list,
    roi: str,
    model_benchmark_commitments: dict,
    null_results: dict,
):
    layer = model_benchmark_commitments[roi]
    layer_name = layer['layer_name']
    layer_pos = layer['layer_position']
    layer_pos_norm = layer['layer_position_normalized']
    
    roi_results_template = copy.deepcopy(null_results)
    roi_results_template['layer_name'] = layer_name
    roi_results_template['layer_position'] = layer_pos
    roi_results_template['layer_position_normalized'] = layer_pos_norm
    roi_results_template['roi'] = roi
    
    probe_cfg = ProbeConfig(
        in_dim=args.in_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        token_encoder_layers=args.token_encoder_layers,
        num_latents=args.num_latents,
        cross_attn_layers=args.cross_attn_layers,
        query_self_attn=args.query_self_attn,
        pos_mode=args.pos_mode,
        head_type=args.head_type,
        head_rank=args.head_rank,
        head_mlp_hidden_dim=args.head_mlp_hidden_dim,
        head_mlp_dropout=args.head_mlp_dropout,
    )

    train_cfg = TrainConfig(
        device=torch.device('cuda' if args.use_gpu else 'cpu'),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=(not args.no_amp),
        grad_accum_steps=args.grad_accum_steps,
        eval_every=args.eval_every,
        lr_schedule=args.lr_schedule,
    )

    model, loaders, noise_ceilings = train_single_roi_single_layer(
        neural_h5_path=data_hdf5_path,
        roi=roi,
        layer_name=layer_name,
        features_folder=features_dir,
        features_file_path=None,
        train_split="train",
        val_split="test",
        probe_cfg=probe_cfg,
        train_cfg=train_cfg,
        subjects_allowlist=probe_fitting_subjects,
        seed=args.seed,
        data_pct=args.data_pct,
    )
    
    eval_results = evaluate(
        model,
        loaders=loaders,
        noise_ceilings=noise_ceilings,
        subjects=subjects,
        split="test",
        device=torch.device('cuda' if args.use_gpu else 'cpu'),
        # amp=train_cfg.amp,
        amp=False,  # Disable AMP during eval for stability
    )
    
    eval_results = eval_results['test_all_subject_results']
    
    results = []
    for subj_result in eval_results:
        subj_roi_results = copy.deepcopy(roi_results_template)
        subj_roi_results.update(subj_result)
        results.append(subj_roi_results)
    
    return results

def main(args):
    
    config = vars(args)
    
    if args.finetuned_model_id is not None:
        model_name = args.finetuned_model_id
    else:
        model_name = args.model_id
    
    features_dir = Path(args.features_dir)
    feature_files = [file.stem for file in features_dir.glob("*.h5")]
    input_dims = set([int(file.split('-')[0].split('_')[-1]) for file in feature_files])
    assert len(input_dims) == 1, \
        f"Expected exactly one input dimension, but found multiple: {input_dims}"
    input_dim = input_dims.pop()
    print(f"Detected input dimension: {input_dim}")
    config['input_dim'] = input_dim
    
    layer_commitments_file = Path(args.layer_commitments)
    assert layer_commitments_file.exists(), f"Layer commitments file does not exist: {layer_commitments_file}"

    with open(layer_commitments_file, 'r') as f:
        layer_commitments = json.load(f)
        
    layer_commitments_model_id = args.model_id
    if args.layer_commitments_model_id is not None:
        layer_commitments_model_id = args.layer_commitments_model_id
    assert layer_commitments_model_id in layer_commitments, f"Model ID {layer_commitments_model_id} not found in layer commitments."
    model_layer_commitments = layer_commitments[layer_commitments_model_id]
        
    data_hdf5_path = Path(args.data_hdf5_path)
    assert data_hdf5_path.exists(), f"Data HDF5 file does not exist: {data_hdf5_path}"
    benchmark_name = data_hdf5_path.stem
    
    model_benchmark_commitments = model_layer_commitments[benchmark_name]

    output_dir = Path(args.output_dir)
    output_file_path = None

    # Final file containing all results
    output_file_path = output_dir / f'{benchmark_name}-{args.probe_config_subject}-pct{args.data_pct:03d}-s{args.seed:02d}.json'
    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=False)
    if output_file_path.exists() and not args.overwrite:
        print(f"Output file already exists and overwrite is False: {output_file_path}")
        return
    
    subjects, rois, splits, nc_max = load_neural_metadata(data_hdf5_path)
    
    print(f"Evaluating model {model_name} on benchmark {benchmark_name} for {len(subjects)} subjects and {len(rois)} ROIs.")
    print(f"ROIs: {rois}")
    print(f"Committed layers: {model_benchmark_commitments}")
    
    null_results = {
        'benchmark_name': benchmark_name,
        'model_id': model_name,
        'layer_name': None,
        'layer_position': None,
        'layer_position_normalized': None,
        'subject': None,
        'roi': None,
        'cv_score': None,
        'r2': None,
        'evs': None,
        'mae': None,
        'mse': None,
        'pearsonr': None,
        'approx_exp_var': None,
        'noise_ceiling': None,
        'pearsonr_nc': None,
        'approx_exp_var_nc': None,
        'rsa_c_train': None,
        'rsa_ve_train': None,
        'cka_c_train': None,
        'cka_ve_train': None,
        'rsa_c_test': None,
        'rsa_ve_test': None,
        'cka_c_test': None,
        'cka_ve_test': None,
    }
    
    all_results = []
    if args.probe_config_subject == "shared":
        ## Fit a single probe across all subjects and evaluate per subject ##
        
        for roi in tqdm(rois, desc='ROIs', leave=False):
            
            if args.exclude_whole_brain and roi == "whole_brain":
                continue
            
            eval_results = fit_n_eval_attn_probe(
                args=args,
                data_hdf5_path=data_hdf5_path,
                features_dir=features_dir,
                subjects=subjects,
                probe_fitting_subjects=subjects,
                roi=roi,
                model_benchmark_commitments=model_benchmark_commitments,
                null_results=null_results,
            )
            
            all_results.extend(eval_results)
                
    else:
        ## Fit and evaluate per subject ##
        
        
        for subj in tqdm(subjects, desc='Subjects', leave=False):
            for roi in tqdm(rois, desc='ROIs', leave=False):
                
                if args.exclude_whole_brain and roi == "whole_brain":
                    continue
                
                eval_results = fit_n_eval_attn_probe(
                    args=args,
                    data_hdf5_path=data_hdf5_path,
                    features_dir=features_dir,
                    subjects=[subj],
                    probe_fitting_subjects=[subj],
                    roi=roi,
                    model_benchmark_commitments=model_benchmark_commitments,
                    null_results=null_results,
                )

                assert len(eval_results) == 1, "Expected a single subject result."

                all_results.append(eval_results[0])                
            
            
    
    
            
    save_results(output_file_path, config, all_results)
            
    print("Evaluation completed.")

def cli():
    args = parse_args()
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
