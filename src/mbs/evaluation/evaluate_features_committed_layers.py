from pathlib import Path
import argparse
import json
import pickle

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
    parser.add_argument("--finetuned_model_id", type=str, default=None, required=False, help="Finetuned model identifier.")
    parser.add_argument("--layer_commitments", type=str, required=True, help="Path to a JSON file specifying layer commitments.")
    parser.add_argument("--layer_commitments_model_id", type=str, required=False, help="Model ID for layer commitments, if different from model_id.")
    parser.add_argument("--data_hdf5_path", type=str, required=True, help="Path to the dataset HDF5 file.")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to the features directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted features.")
    parser.add_argument("--debug_mode", type=str2bool, default=False, help="Whether to run in debug mode.")

    parser.add_argument("--data_pct", type=int, default=None, help="Percentage of data to fit on.")
    parser.add_argument("--skip_cv", type=str2bool, default=True, help="Whether to skip cross-validation on training data.")
    parser.add_argument("--calculate_rsa_cka", type=str2bool, default=False, help="Whether to calculate RSA and CKA metrics.")
    parser.add_argument("--use_wide_range_alphas", type=str2bool, default=False, help="Whether to use a wide range of alphas for Ridge regression.")
    parser.add_argument("--probes_input_dir", type=str, default=None, help="Provide a directory to load probe models, otherwise probes are fitted anew.")
    parser.add_argument("--probes_output_dir", type=str, default=None, help="Provide a directory to save probe models, otherwise probes are not saved.")
    parser.add_argument("--exclude_whole_brain", type=str2bool, default=False, help="Whether to exclude whole brain ROI.")
    parser.add_argument("--use_gpu", type=str2bool, default=False, help="Whether to use GPU for regression.")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Whether to overwrite existing output directory.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the model inputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args

def main(args):
    
    config = vars(args)
    
    if args.finetuned_model_id is not None:
        model_name = args.finetuned_model_id
    else:
        model_name = args.model_id
    
    features_dir = Path(args.features_dir)
    feature_files = [
        f.stem
        for pattern in ("*.h5", "*.hdf5")
        for f in features_dir.glob(pattern)
    ]
    assert len(feature_files) > 0, f"No feature files found in {features_dir}"
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
    output_file_path = output_dir / f'{benchmark_name}.json'
    if args.data_pct is not None:
        output_file_path = output_file_path.with_name(f'{output_file_path.stem}-pct{args.data_pct:03d}-s{args.seed:02d}.json')
    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=False)
    if output_file_path.exists() and not args.overwrite:
        print(f"Output file already exists and overwrite is False: {output_file_path}")
        return
    
    subjects, rois, splits, nc_max = load_neural_metadata(data_hdf5_path)
    
    print(f"Evaluating model {model_name} on benchmark {benchmark_name} for {len(subjects)} subjects and {len(rois)} ROIs.")
    print(f"ROIs: {rois}")
    print(f"Committed layers: {model_benchmark_commitments}")
    
    all_results = []
    for subj in tqdm(subjects, desc='Subjects', leave=False):
        for roi in tqdm(rois, desc='ROIs', leave=False):
            
            if args.exclude_whole_brain and roi == "whole_brain":
                continue
        
            layer = model_benchmark_commitments.get(roi, None)
            if layer is None:
                print(f"No committed layer for Subject: {subj}, ROI: {roi}. Skipping.")
                continue
            layer_name = layer.get('layer_name', None)
            layer_pos = layer.get('layer_position', None)
            layer_pos_norm = layer.get('layer_position_normalized', None)
            
            null_results = {
                'benchmark_name': benchmark_name,
                'model_id': model_name,
                'layer_name': layer_name,
                'layer_position': layer_pos,
                'layer_position_normalized': layer_pos_norm,
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
            }

            # Load layer features
            layer_feats, stimulus_ids_mapping = load_layer_features(features_folder=features_dir, layer_name=layer_name)
            layer_feats = np.reshape(layer_feats, (layer_feats.shape[0], -1)) # Flatten features
        
            # Load neural data
            stimulus_ids_train, neural_data_train, noise_ceiling = load_neural_data(data_hdf5_path, subject=subj, roi=roi, split='train')
            stimulus_ids_test, neural_data_test, noise_ceiling = load_neural_data(data_hdf5_path, subject=subj, roi=roi, split='test')

            y_train = neural_data_train
            y_test = neural_data_test

            stimulus_ids_train = [stimulus_ids_mapping[stim_id] for stim_id in stimulus_ids_train]
            X_train = layer_feats[stimulus_ids_train]

            stimulus_ids_test = [stimulus_ids_mapping[stim_id] for stim_id in stimulus_ids_test]
            X_test = layer_feats[stimulus_ids_test]
            
            if args.data_pct is not None and args.data_pct < 100:
                assert 0 < args.data_pct <= 100, "data_pct must be in (0, 100]"
                print (f"Using {args.data_pct}% of training data for Subject: {subj}, ROI: {roi}", flush=True)
                rng = np.random.default_rng(seed=args.seed)
                indices = rng.permutation(X_train.shape[0])
                n_used = int(X_train.shape[0] * args.data_pct / 100)
                selected_indices = indices[:n_used]
                X_train = X_train[selected_indices]
                y_train = y_train[selected_indices]
            
            
            if y_test.shape[1] == 0 or y_train.shape[1] == 0 or args.debug_mode:
                # print(f'Skipping Layer: {layer_name}, Subject: {subj}, ROI: {roi} -- No valid neural data.')
                all_results.append(null_results)
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
                    
                
            try:
                pearsonr_nc = lambda y_true, y_pred: pearsonr_score(y_true, y_pred, noise_ceiling=noise_ceiling)
                scorer = make_scorer(pearsonr_nc, greater_is_better=True)

                cross_val_score = None  # Placeholder since cross-validation is skipped
                if not args.skip_cv:
                    cross_val_scores = cross_validate(
                        get_pipeline(use_gpu=args.use_gpu, use_wide_range_alphas=args.use_wide_range_alphas), 
                        X_train, y_train, cv=5, scoring=scorer, n_jobs=1
                    )
                    cross_val_score = cross_val_scores['test_score'].mean()

                pipeline = get_pipeline(use_gpu=args.use_gpu, use_wide_range_alphas=args.use_wide_range_alphas)
                if args.probes_input_dir is not None:
                    probe_model_path = Path(args.probes_input_dir) / args.model_id / benchmark_name / f'subject_{subj}' / f'roi_{roi}.pkl'
                    assert probe_model_path.exists(), f"Probe model file does not exist: {probe_model_path}"
                    print(f"Loading probe model for Subject: {subj}, ROI: {roi} from {probe_model_path}")
                    with open(probe_model_path, 'rb') as f:
                        probe_data = pickle.load(f)
                    W = probe_data['W']
                    b = probe_data['b']
                    pipeline['regressor'].coef_ = W
                    pipeline['regressor'].intercept_ = b
                else:
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
                    
                    
                results = {
                    'benchmark_name': benchmark_name,
                    'model_id': model_name,
                    'layer_name': layer_name,
                    'layer_position': layer_pos,
                    'layer_position_normalized': layer_pos_norm,
                    'subject': subj,
                    'roi': roi,
                    'cv_score': float(cross_val_score) if cross_val_score is not None else None,
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
                }

                if args.probes_output_dir is not None:
                    probe_model_path = Path(args.probes_output_dir) / args.model_id / benchmark_name / f'subject_{subj}' / f'roi_{roi}.pkl'
                    if not probe_model_path.parent.exists():
                        probe_model_path.parent.mkdir(parents=True, exist_ok=False)
                    W = pipeline['regressor'].coef_
                    b = pipeline['regressor'].intercept_
                    print(f"Saving probe model for Subject: {subj}, ROI: {roi} at {probe_model_path}, Weights shape: {W.shape}, Bias shape: {b.shape}")
                    with open(probe_model_path, 'wb') as f:
                        pickle.dump({
                            'W': W,
                            'b': b,
                            'metadata': results
                        }, f)
            except Exception as e:
                print(f'Error processing Layer: {layer_name}, Subject: {subj}, ROI: {roi} -- {e}')
                results = null_results
                
            
            all_results.append(results)


        # print(f'Layer: {layer_name}, Subject: {subj}, ROI: {roi} -- CV PearsonR (NC): {cross_val_score:.4f}, Test R2: {r2:.4f}, Test EVS: {evs:.4f}, Test MAE: {mae:.4f}, Test MSE: {mse:.4f}, Test PearsonR: {pearsonr:.4f}, Test PearsonR (NC): {pearsonr_nc:.4f}')
        
            
    save_results(output_file_path, config, all_results)
            
    print("Evaluation completed.")

def cli():
    args = parse_args()
    print(args)
    main(args)


if __name__ == "__main__":
    cli()
