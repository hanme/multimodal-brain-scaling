"""
H5 I/O utilities.

"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import h5py
from tqdm.auto import tqdm


def load_layer_features(layer_name: str, features_folder: Path = None, features_file_path: Path = None):
    """
    Load precomputed layer features.

    Returns
    -------
    layer_feats : np.ndarray
        Array shaped (S_total, ...) where ... may be:
          - (C,) Flattened features
          - (N, C) Transformer features
          - (C, H, W) Conv features
    stimulus_ids_mapping : dict[str -> int]
        Maps stimulus_id -> row index into layer_feats.
    """
    assert features_folder is not None or features_file_path is not None, \
        "Either features_folder or features_file_path must be provided."
    assert not (features_folder is not None and features_file_path is not None), \
        "Only one of features_folder or features_file_path should be provided."

    if features_folder:
        h5_files = sorted(Path(features_folder).glob("*.h5"))
        assert len(h5_files) > 0, f"No h5 files found in {features_folder}"
        layer_chunks = []
        stimulus_ids: List[Any] = []
        for h5_file in tqdm(h5_files, desc=f"Loading features: {layer_name}", leave=False):
            with h5py.File(h5_file, "r") as f:
                layer_key = f"features/{layer_name.replace('.', '-')}"
                layer_chunks.append(f[layer_key][:])
                stimulus_ids.extend(f["ids"][:])
        layer_feats = np.concatenate(layer_chunks, axis=0)
    else:
        h5_file = Path(features_file_path)
        assert h5_file.exists(), f"No h5 file found at {features_file_path}"
        with h5py.File(h5_file, "r") as f:
            layer_key = f"features/{layer_name.replace('.', '-')}"
            layer_feats = f[layer_key][:]
            stimulus_ids = f["ids"][:]

    # Normalize ids to python strings
    if len(stimulus_ids) > 0 and hasattr(stimulus_ids[0], "decode"):
        stimulus_ids = [s.decode("utf-8") for s in stimulus_ids]

    stimulus_ids_mapping: Dict[str, int] = {str(stim_id): idx for idx, stim_id in enumerate(stimulus_ids)}
    return layer_feats, stimulus_ids_mapping


def load_neural_metadata(data_path: Path):
    """
    Returns attributes stored in the neural H5.
    """
    with h5py.File(Path(data_path), "r") as f:
        subjects = f.attrs["subjects"]
        rois = f.attrs["rois"]
        splits = f.attrs["splits"]
        nc_max = f.attrs.get("max_nc", 100.0)
    return subjects, rois, splits, nc_max


def load_neural_data(data_path: Path, subject: str, roi: str, split: str):
    """
    Load neural responses and noise ceilings for one (subject, roi, split).

    Returns
    -------
    stimulus_ids : np.ndarray of shape (S,)
    neural_data  : np.ndarray of shape (S, N) where N is number of voxels/channels/neuroids
    noise_ceiling: np.ndarray of shape (N,), converted to correlation units
    """
    with h5py.File(Path(data_path), "r") as f:
        subject_specific_stimulus_set = f.attrs.get("subject_specific_stimulus_set", False)

        if subject_specific_stimulus_set:
            stimulus_ids = f[split]["stimulus_ids"][subject][()]
        else:
            stimulus_ids = f[split]["stimulus_ids"][()]

        nc_max = f.attrs.get("max_nc", 100.0)
        neural_data = f[split]["neural_data"][subject][roi][()]
        noise_ceiling = f["noise_ceilings"][subject][roi][()] / nc_max + 1e-6
        noise_ceiling = np.sqrt(noise_ceiling)  # VE -> corr
        
    # neural_data = neural_data.reshape(neural_data.shape[0], -1)
    # noise_ceiling = noise_ceiling.reshape(-1)
    if neural_data.ndim > 2:
        # EEG/MEG case: flatten to (S, N)
        neural_data = neural_data.reshape(neural_data.shape[0], -1)
        noise_ceiling = noise_ceiling.reshape(-1)
        
        noise_ceiling_mask = noise_ceiling > 0.1
        neural_data = neural_data[:, noise_ceiling_mask]
        noise_ceiling = noise_ceiling[noise_ceiling_mask]
    
    print(f"Loaded neural data for subject={subject}, roi={roi}, split={split}: "
          f"responses shape={neural_data.shape}, noise ceiling shape={noise_ceiling.shape}")
    
    neural_data_mean = np.mean(neural_data, axis=0, keepdims=True)
    neural_data_std = np.std(neural_data, axis=0, keepdims=True) + 1e-6
    neural_data = (neural_data - neural_data_mean) / neural_data_std
    print(f"Normalized neural data from mean={neural_data_mean.mean():.4f}, std={neural_data_std.mean():.4f}.")

    if len(stimulus_ids) > 0 and hasattr(stimulus_ids[0], "decode"):
        stimulus_ids = np.array([s.decode("utf-8") for s in stimulus_ids], dtype=object)

    return stimulus_ids, neural_data, noise_ceiling
