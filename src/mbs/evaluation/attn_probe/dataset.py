"""
Dataset + dataloader builders for ONE ROI and ONE layer.

We keep one DataLoader per subject to avoid padding (since output dim varies by subject).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .h5io import load_neural_metadata, load_neural_data


def normalize_ids(ids: np.ndarray) -> List[str]:
    """
    Convert an array of ids (bytes/strings/ints) into a list of strings.
    """
    out: List[str] = []
    for s in ids:
        if hasattr(s, "decode"):
            out.append(s.decode("utf-8"))
        else:
            out.append(str(s))
    return out


def align_feature_indices(feature_id_map: Dict[str, int], neural_stim_ids: np.ndarray) -> np.ndarray:
    """
    For each neural stimulus id, find the row index in the feature table.

    Raises fast if any stimulus id is missing (this catches dataset mismatch early).
    """
    stim = normalize_ids(neural_stim_ids)
    idx = np.empty(len(stim), dtype=np.int64)

    missing = []
    for i, sid in enumerate(stim):
        j = feature_id_map.get(sid, None)
        if j is None:
            missing.append(sid)
        else:
            idx[i] = j

    if missing:
        raise KeyError(f"{len(missing)} neural stimulus ids missing in features. Example: {missing[:5]}")
    return idx


def _to_str_list(x) -> List[str]:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    out = normalize_ids(x)
    return out


class SingleRoiSubjectDataset(Dataset):
    """
    Dataset for a fixed (subject, roi, split).

    It aligns neural stimuli to features by stimulus ids.
    """
    def __init__(
        self,
        *,
        subject: str,
        roi: str,
        split: str,
        neural_h5_path: Path,
        layer_feats: np.ndarray,
        feature_id_map: Dict[str, int],
        seed: int = 42,
        data_pct: Optional[float] = None,
    ):
        super().__init__()
        self.subject = subject
        self.roi = roi
        self.split = split
        self.seed = seed

        stim_ids, y, noise_ceiling = load_neural_data(neural_h5_path, subject=subject, roi=roi, split=split)
        self.y = y.astype(np.float32)                 # (S, N_subject)
        self.noise_ceiling = noise_ceiling.astype(np.float32)  # (N_subject,)

        feat_idx = align_feature_indices(feature_id_map, stim_ids)
        self.layer_feats = layer_feats
        self.feat_idx = feat_idx

        assert self.y.shape[0] == self.feat_idx.shape[0], "Mismatch after alignment."
        
        if data_pct is not None:
            print(f"Using {data_pct}% of data for subject={subject}, roi={roi}, split={split}")
            assert 0 < data_pct <= 100, "data_pct must be in (0, 100]"
            n_total = self.y.shape[0]
            rng = np.random.default_rng(seed=self.seed)
            perm = rng.permutation(n_total)
            indices = perm[: int(n_total * data_pct / 100)]
            self.y = self.y[indices]
            self.feat_idx = self.feat_idx[indices]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        j = int(self.feat_idx[i])
        feats_i = np.asarray(self.layer_feats[j])  # could be (C) or (N,C) or (C,H,W)
        y_i = self.y[i]                            # (N_subject,)
        return {
            "feats": torch.from_numpy(feats_i).float(),
            "y": torch.from_numpy(y_i).float(),
        }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def infer_in_dim_from_layer_feats(layer_feats: np.ndarray) -> int:
    """
    Infer feature channel dimension from stored features.
    """
    if layer_feats.ndim == 2:   # (S, C)
        return int(layer_feats.shape[1])
    if layer_feats.ndim == 3:   # (S, N, C)
        return int(layer_feats.shape[2])
    if layer_feats.ndim == 4:   # (S, C, H, W)
        return int(layer_feats.shape[1])
    raise ValueError(f"Unsupported layer_feats.ndim={layer_feats.ndim}")


def build_subject_loaders(
    *,
    neural_h5_path: Path,
    roi: str,
    layer_feats: np.ndarray,
    feature_id_map: Dict[str, int],
    train_split: str,
    val_split: str,
    batch_size: int,
    subjects_allowlist: Optional[List[str]] = None,
    seed: int = 42,
    data_pct: Optional[float] = None,
) -> Tuple[List[str], Dict[Tuple[str, str], DataLoader], Dict[Tuple[str, str], np.ndarray]]:
    """
    Returns:
      subjects: list[str]
      loaders: dict[(split, subject)] -> DataLoader
      noise_ceilings: dict[(split, subject)] -> np.ndarray (V_subject,)
    """
    subjects_attr, rois_attr, splits_attr, _ = load_neural_metadata(neural_h5_path)
    subjects = _to_str_list(subjects_attr)
    rois = _to_str_list(rois_attr)
    splits = _to_str_list(splits_attr)

    assert roi in rois, f"ROI='{roi}' not in H5 rois={rois}"
    assert train_split in splits and val_split in splits, f"Splits not found in H5 splits={splits}"

    if subjects_allowlist is not None:
        allow = set(subjects_allowlist)
        subjects = [s for s in subjects if s in allow]
        assert len(subjects) > 0, "No subjects after allowlist filtering."

    loaders: Dict[Tuple[str, str], DataLoader] = {}
    noise_ceilings: Dict[Tuple[str, str], np.ndarray] = {}

    for s in subjects:
        for split in [train_split, val_split]:
            if 'train' in split.lower():
                # Use data_pct only for training split
                data_pct_used = data_pct
            else:
                data_pct_used = None
            
            ds = SingleRoiSubjectDataset(
                subject=s,
                roi=roi,
                split=split,
                neural_h5_path=neural_h5_path,
                layer_feats=layer_feats,
                feature_id_map=feature_id_map,
                seed=seed,
                data_pct=data_pct_used,
            )
            loaders[(split, s)] = make_loader(ds, batch_size=batch_size, shuffle=(split == train_split))
            noise_ceilings[(split, s)] = ds.noise_ceiling

    return subjects, loaders, noise_ceilings
