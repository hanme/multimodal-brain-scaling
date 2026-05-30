from pathlib import Path
import argparse
import json

import numpy as np
import torch

from tqdm.auto import tqdm


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from tqdm.auto import tqdm
# from tqdm import tqdm



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
)



def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from a dataset using a feature extractor model.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier.")
    parser.add_argument("--target_feature_layers", type=str, required=True, help="Path to a JSON file specifying target feature layers.")
    parser.add_argument("--data_hdf5_path", type=str, required=True, help="Path to the dataset HDF5 file.")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to the features directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted features.")
    parser.add_argument("--save_results_layerwise", type=str2bool, default=True, help="Whether to save results layer-wise.")
    parser.add_argument("--layer_id", type=int, default=None, help="If specified, only evaluate this layer index.")
    parser.add_argument("--debug_mode", type=str2bool, default=False, help="Whether to run in debug mode.")
    parser.add_argument("--backbone_checkpoint", type=str, default=None, help="Path to the backbone model checkpoint. Will use default weights if not provided.")

    parser.add_argument("--calculate_rsa_cka", type=str2bool, default=False, help="Whether to calculate RSA and CKA metrics.")
    parser.add_argument("--use_wide_range_alphas", type=str2bool, default=False, help="Whether to use a wide range of alphas for Ridge regression.")
    parser.add_argument("--exclude_whole_brain", type=str2bool, default=False, help="Whether to exclude whole brain ROI.")
    parser.add_argument("--use_gpu", type=str2bool, default=False, help="Whether to use GPU for regression.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Whether to overwrite existing output directory.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the model inputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args

def main(args):
    
    config = vars(args)
    
    features_dir = Path(args.features_dir)
    assert features_dir.exists(), f"Features folder does not exist: {features_dir}"

    layer_list_file = Path(args.target_feature_layers)
    assert layer_list_file.exists(), f"Layer list file does not exist: {layer_list_file}"

    with open(layer_list_file, 'r') as f:
        layer_list = json.load(f)

    data_hdf5_path = Path(args.data_hdf5_path)
    assert data_hdf5_path.exists(), f"Data HDF5 file does not exist: {data_hdf5_path}"
    benchmark_name = data_hdf5_path.stem

    output_dir = Path(args.output_dir)
    output_files_dir = None
    output_file_path = None
    if args.save_results_layerwise:
        # Save the results of each layer in a separate file within a directory
        output_files_dir = output_dir / f'{benchmark_name}'
        if not output_files_dir.exists():
            output_files_dir.mkdir(parents=True, exist_ok=False)

    # Final file containing all results
    output_file_path = output_dir / f'{benchmark_name}.json'
    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=False)
    if output_file_path.exists() and not args.overwrite:
        print(f"Output file already exists and overwrite is False: {output_file_path}")
        return
    
    subjects, rois, splits, nc_max = load_neural_metadata(data_hdf5_path)
    
    all_results = []
    
    for layer_idx, layer in tqdm(enumerate(layer_list), desc='Layers', leave=True, total=len(layer_list)):
        
        if args.layer_id is not None and layer_idx != args.layer_id:
            print(f"Skipping layer index {layer_idx}.")
            continue
        if args.layer_id is not None and layer_idx == args.layer_id:
            print(f"Evaluating only layer index {layer_idx}")
        
        layer_name = layer['name']
        layer_pos = layer['position']
        layer_results = []
        
        
        if args.save_results_layerwise:
            output_file_path = output_files_dir / f'{layer_name.replace(".", "-")}.json'
            if output_file_path.exists() and not args.overwrite:
                print(f"Output file for layer {layer_name} already exists and overwrite is False: {output_file_path}")
                continue
    
        layer_feats, stimulus_ids_mapping = load_layer_features(features_folder=features_dir, layer_name=layer_name)
        
        for subj in tqdm(subjects, desc='Subjects', leave=False):

            for roi in tqdm(rois, desc='ROIs', leave=False):

                if args.exclude_whole_brain and roi == "whole_brain":
                    continue

                stimulus_ids_train, neural_data_train, noise_ceiling = load_neural_data(data_hdf5_path, subject=subj, roi=roi, split='train')
                stimulus_ids_test, neural_data_test, noise_ceiling = load_neural_data(data_hdf5_path, subject=subj, roi=roi, split='test')

                y_train = neural_data_train
                y_test = neural_data_test

                stimulus_ids_train = [stimulus_ids_mapping[stim_id] for stim_id in stimulus_ids_train]
                X_train = layer_feats[stimulus_ids_train]

                stimulus_ids_test = [stimulus_ids_mapping[stim_id] for stim_id in stimulus_ids_test]
                X_test = layer_feats[stimulus_ids_test]
                
                if y_test.shape[1] == 0 or y_train.shape[1] == 0 or args.debug_mode:
                    # print(f'Skipping Layer: {layer_name}, Subject: {subj}, ROI: {roi} -- No valid neural data.')
                    layer_results.append({
                        'benchmark_name': benchmark_name,
                        'model_id': args.model_id,
                        'layer_name': layer_name,
                        'layer_position': layer_pos,
                        'subject': subj,
                        'roi': roi,
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
                    })
                    continue
                
                # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
                
                # print(y_test.shape)
                if len(y_test.shape) == 3:
                    y_train = y_train.reshape(y_train.shape[0], -1)
                    y_test = y_test.reshape(y_test.shape[0], -1)
                    noise_ceiling = noise_ceiling.reshape(1, -1)
                    
                    noise_ceiling_mask = noise_ceiling > 0.1
                    y_train = y_train[:, noise_ceiling_mask.flatten()]
                    y_test = y_test[:, noise_ceiling_mask.flatten()]
                    noise_ceiling = noise_ceiling[:, noise_ceiling_mask.flatten()]
                    
                    
                    # print(f'Reshaped y_train to {y_train.shape}, y_test to {y_test.shape}, noise_ceiling to {noise_ceiling.shape}')
                
                # print(y_train.shape, y_test.shape, X_train.shape, X_test.shape)
                # print(f'Layer: {layer_name}, Subject: {subj}, ROI: {roi}, X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
                
                # scoring = {'r2': 'r2', 'pearsonr': pearsonr_scorer}
                
                pearsonr_nc = lambda y_true, y_pred: pearsonr_score(y_true, y_pred, noise_ceiling=noise_ceiling)
                scorer = make_scorer(pearsonr_nc, greater_is_better=True)

                cross_val_scores = cross_validate(get_pipeline(use_gpu=args.use_gpu, use_wide_range_alphas=args.use_wide_range_alphas), X_train, y_train, cv=5, scoring=scorer, n_jobs=1)
                cross_val_score = cross_val_scores['test_score'].mean()
                # cross_val_score = np.nan  # Placeholder since cross-validation is skipped

                pipeline = get_pipeline(use_gpu=args.use_gpu, use_wide_range_alphas=args.use_wide_range_alphas)
                # torch.empty_cache()
                # torch.cuda.empty_cache()
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                r2, evs, mae, mse, pearsonr, app_evs, pearsonr_nc, app_evs_nc = compute_metrics(y_test, y_pred, verbose=False, noise_ceiling=noise_ceiling)
                
                if args.calculate_rsa_cka:
                    y_train_pred = pipeline.predict(X_train)
                    ( rsa_c_train, rsa_ve_train, rsa_c_test, rsa_ve_test,
                        cka_c_train, cka_ve_train, cka_c_test, cka_ve_test 
                    ) = compute_rsa_cka(
                        X_train=X_train,
                        y_train=y_train,
                        y_train_pred=y_train_pred,
                        X_test=X_test,
                        y_test=y_test,
                        y_test_pred=y_pred,
                        verbose=False,
                        use_gpu=args.use_gpu,
                    )
                else:
                    rsa_c_train, rsa_ve_train, cka_c_train, cka_ve_train = (None, None, None, None)
                    rsa_c_test, rsa_ve_test, cka_c_test, cka_ve_test = (None, None, None, None)
                    
                
                layer_results.append({
                    'benchmark_name': benchmark_name,
                    'model_id': args.model_id,
                    'layer_name': layer_name,
                    'layer_position': layer_pos,
                    'subject': subj,
                    'roi': roi,
                    'cv_score': float(cross_val_score),
                    'r2': float(r2),
                    'evs': float(evs),
                    'mae': float(mae),
                    'mse': float(mse),
                    'pearsonr': float(pearsonr),
                    'approx_exp_var': float(app_evs),
                    'noise_ceiling': float(noise_ceiling.mean()),
                    'pearsonr_nc': float(pearsonr_nc),
                    'approx_exp_var_nc': float(app_evs_nc),
                    'rsa_c_train': float(rsa_c_train) if rsa_c_train is not None else None,
                    'rsa_ve_train': float(rsa_ve_train) if rsa_ve_train is not None else None,
                    'cka_c_train': float(cka_c_train) if cka_c_train is not None else None,
                    'cka_ve_train': float(cka_ve_train) if cka_ve_train is not None else None,
                    'rsa_c_test': float(rsa_c_test) if rsa_c_test is not None else None,
                    'rsa_ve_test': float(rsa_ve_test) if rsa_ve_test is not None else None,
                    'cka_c_test': float(cka_c_test) if cka_c_test is not None else None,
                    'cka_ve_test': float(cka_ve_test) if cka_ve_test is not None else None,
                })


        # print(f'Layer: {layer_name}, Subject: {subj}, ROI: {roi} -- CV PearsonR (NC): {cross_val_score:.4f}, Test R2: {r2:.4f}, Test EVS: {evs:.4f}, Test MAE: {mae:.4f}, Test MSE: {mse:.4f}, Test PearsonR: {pearsonr:.4f}, Test PearsonR (NC): {pearsonr_nc:.4f}')
        if args.save_results_layerwise:
            save_results(output_file_path, config, layer_results)
        
        else:
            all_results.append(*layer_results)
            
    if args.layer_id is not None:
        print(f"Evaluated only layer index {args.layer_id}, skipping aggregation of all layers.")
    elif args.save_results_layerwise:
        # Collect all layer results into a single file as well
        output_files = sorted(output_files_dir.glob('*.json'))
        assert len(output_files) == len(layer_list), "Number of layer result files does not match number of layers."
        for layer_results_file in tqdm(output_files, desc='Collecting layer results', leave=True):
            config = None
            with open(layer_results_file, 'r') as f:
                data = json.load(f)
                all_results.extend(data['results'])
                if config is None:
                    config = data['config']
                else:
                    assert config == data['config'], "Config mismatch between layer results files."
        output_file_path = output_files_dir.parent / f'{benchmark_name}.json'
        save_results(output_file_path, config, all_results)
        
    else:
        save_results(output_file_path, config, all_results)
            
    print("Evaluation completed.")

def cli():
    args = parse_args()
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
