"""Per-time-step Ridge regression for audio features vs temporal EEG.

For each layer and each time step t:
    RidgeCV fit on features_train[:, t, :] → eeg_train[:, t, :]
    Pearson r on test set, optionally noise-ceiling corrected

Output per layer/subject/roi: scores[T, n_ch]  (saved to HDF5 + JSON summary)
"""

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import scipy.stats
from sklearn.linear_model import RidgeCV
from tqdm.auto import tqdm

from mbs.core import str2bool
from .utils.evaluation_helpers import (
    ALPHA_LIST_SHORT,
    load_layer_features,
    load_neural_metadata,
    load_neural_data,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-time-step Ridge regression for temporal audio features vs EEG."
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model identifier (e.g. whisper-base).")
    parser.add_argument("--target_feature_layers", type=str, required=True,
                        help="Path to JSON file listing layer names to evaluate.")
    parser.add_argument("--data_hdf5_path", type=str, required=True,
                        help="Path to neural HDF5 file (temporal schema, T_model attr set).")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing per-stimulus feature HDF5 files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write scores HDF5 and summary JSON.")
    parser.add_argument("--noise_ceiling_correct", type=str2bool, default=True,
                        help="Divide Pearson r by noise ceiling.")
    parser.add_argument("--use_wide_range_alphas", type=str2bool, default=False,
                        help="Use extended alpha grid for RidgeCV.")
    parser.add_argument("--overwrite", type=str2bool, default=False,
                        help="Overwrite existing output directory.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _decode_ids(ids):
    return [x.decode() if hasattr(x, "decode") else x for x in ids]


def evaluate_layer_temporal(feat_train, feat_test, eeg_train, eeg_test, noise_ceiling, alphas):
    """Fit per-time-step RidgeCV; return scores[T, n_ch] (raw Pearson r).

    Args:
        feat_train: [n_train, T, d]
        feat_test:  [n_test,  T, d]
        eeg_train:  [n_train, T, n_ch]
        eeg_test:   [n_test,  T, n_ch]
        noise_ceiling: [T, n_ch] or [n_ch] or None
        alphas: Ridge alpha grid
    """
    T = feat_train.shape[1]
    n_ch = eeg_train.shape[2]
    scores = np.full((T, n_ch), np.nan, dtype=np.float32)

    for t in range(T):
        model = RidgeCV(alphas=alphas)
        model.fit(feat_train[:, t, :], eeg_train[:, t, :])
        y_pred = model.predict(feat_test[:, t, :])
        y_ref = eeg_test[:, t, :]
        y_pred = y_pred.reshape(y_ref.shape)  # sklearn squeezes n_ch=1 to 1D
        r = scipy.stats.pearsonr(y_ref, y_pred, axis=0)[0]
        scores[t] = r.astype(np.float32)

    return scores


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_list_path = Path(args.target_feature_layers)
    assert layer_list_path.exists(), f"Layer list not found: {layer_list_path}"
    with open(layer_list_path) as f:
        layer_list_raw = json.load(f)
    layer_list = [e["name"] if isinstance(e, dict) else e for e in layer_list_raw]

    data_hdf5_path = Path(args.data_hdf5_path)
    assert data_hdf5_path.exists(), f"Neural HDF5 not found: {data_hdf5_path}"

    subjects, rois, splits, nc_max = load_neural_metadata(data_hdf5_path)
    features_dir = Path(args.features_dir)

    alphas = ALPHA_LIST_SHORT

    scores_path = output_dir / "temporal_scores.h5"
    summary_path = output_dir / "temporal_scores_summary.json"

    # Load existing summary entries so we can resume after a job timeout.
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {"model_id": args.model_id, "layers": []}

    # Append mode: keeps completed datasets intact if the job was killed mid-run.
    with h5py.File(scores_path, "a") as out_h5:
        for layer_name in tqdm(layer_list, desc="Layers"):
            layer_feats, id_map = load_layer_features(layer_name, features_folder=features_dir)

            if layer_feats.ndim != 3:
                print(f"Layer {layer_name}: features are {layer_feats.ndim}D, expected 3D [n,T,d] — skipping.")
                continue

            for subject in subjects:
                for roi in rois:
                    key = f"{layer_name.replace('.', '-')}/{subject}/{roi}"
                    if key in out_h5:
                        continue  # already computed in a previous run

                    train_ids_raw, eeg_train, nc = load_neural_data(
                        data_hdf5_path, subject, roi, "train"
                    )
                    test_ids_raw, eeg_test, _ = load_neural_data(
                        data_hdf5_path, subject, roi, "test"
                    )
                    train_ids = _decode_ids(train_ids_raw)
                    test_ids = _decode_ids(test_ids_raw)

                    train_idx_raw = [id_map.get(sid) for sid in train_ids]
                    test_idx_raw = [id_map.get(sid) for sid in test_ids]

                    # Filter to IDs present in both the feature set and the EEG set.
                    # Missing IDs arise when a SLURM chunk wrote fewer stimuli than expected
                    # (e.g. the last chunk); we drop those rows and proceed with the rest.
                    train_eeg_mask = [i for i, v in enumerate(train_idx_raw) if v is not None]
                    test_eeg_mask  = [i for i, v in enumerate(test_idx_raw)  if v is not None]
                    train_feat_idx = [v for v in train_idx_raw if v is not None]
                    test_feat_idx  = [v for v in test_idx_raw  if v is not None]

                    n_miss_train = len(train_idx_raw) - len(train_feat_idx)
                    n_miss_test  = len(test_idx_raw)  - len(test_feat_idx)
                    if n_miss_train or n_miss_test:
                        print(f"  Warning: {layer_name}/{subject}/{roi} — "
                              f"{len(train_feat_idx)}/{len(train_idx_raw)} train, "
                              f"{len(test_feat_idx)}/{len(test_idx_raw)} test IDs matched.")
                    if not train_feat_idx or not test_feat_idx:
                        print(f"  Skipping {layer_name}/{subject}/{roi}: no matched IDs.")
                        continue

                    feat_train = layer_feats[train_feat_idx]
                    feat_test  = layer_feats[test_feat_idx]
                    eeg_train  = eeg_train[train_eeg_mask]
                    eeg_test   = eeg_test[test_eeg_mask]

                    nc_for_correction = nc if args.noise_ceiling_correct else None
                    scores = evaluate_layer_temporal(
                        feat_train, feat_test, eeg_train, eeg_test,
                        nc_for_correction, alphas
                    )
                    if args.noise_ceiling_correct and nc is not None:
                        # nc shape: [T, n_ch] or [n_ch]; broadcast-divide
                        nc_safe = np.where(nc > 0, nc, 1e-6)
                        scores = scores / nc_safe

                    out_h5.create_dataset(key, data=scores, compression="gzip")
                    out_h5.flush()
                    summary["layers"].append({
                        "layer": layer_name,
                        "subject": subject,
                        "roi": roi,
                        "mean_score": float(np.nanmean(scores)),
                        "peak_score": float(np.nanmax(scores)),
                    })
                    # Write summary after each dataset so progress survives a kill.
                    with open(summary_path, "w") as f:
                        json.dump(summary, f, indent=2)

    print(f"Temporal scores written to {scores_path}")
    print(f"Summary written to {output_dir / 'temporal_scores_summary.json'}")


if __name__ == "__main__":
    main(parse_args())
