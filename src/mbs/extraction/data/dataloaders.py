from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .datasets import BrainScoreStimulusDataset, H5ImageDataset, THINGSImageDataset
from .datasets_audio import AudioSegmentDataset


def collate_fn(batch):
    # Each item: (x, id)
    xs = [b[0] for b in batch]
    ids = [b[1] for b in batch]

    x0 = xs[0]

    # image encoder: x is Tensor (C,H,W) -> (B,C,H,W)
    if torch.is_tensor(x0):
        x = torch.stack(xs, 0)

    # HF: x is BatchFeature/BatchEncoding/dict-like -> dict of batched tensors
    else:
        x = {}
        for k in x0.keys():
            v0 = x0[k]
            if torch.is_tensor(v0):
                x[k] = torch.stack([xi[k].squeeze(0) for xi in xs], 0)
            else:
                x[k] = [xi[k] for xi in xs]

    # ids: ints -> tensor, else keep list (strings, etc.)
    if isinstance(ids[0], (int, bool)):
        ids = torch.tensor(ids)

    return x, ids


def create_dataloader(
    data_root: str,
    dataset_type: Literal['brain_score', 'h5', 'things', 'audio'],
    transform: T.Compose,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:

    assert dataset_type in ['brain_score', 'h5', 'things', 'audio'], f"Unsupported dataset_type: {dataset_type}"

    match dataset_type:
        case 'brain_score':
            dataset = BrainScoreStimulusDataset(stimulus_set_id=data_root, transform=transform)
        case 'h5':
            x_key = kwargs.get('x_key', 'imgBrick') # key for NSD stimulus HDF5 file
            dataset = H5ImageDataset(h5_path=data_root, x_key=x_key, transform=transform)
        case 'things':
            dataset = THINGSImageDataset(root_dir=data_root, transform=transform)
        case 'audio':
            wav_files = sorted(Path(data_root).glob("*.wav"))
            if not wav_files:
                raise FileNotFoundError(f"No .wav files found in {data_root}")
            dataset = AudioSegmentDataset(
                wav_files=wav_files,
                window_duration=kwargs.get('window_duration', 30.0),
                stride=kwargs.get('window_stride', 10.0),
                transform=transform,
            )
            
            
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return dataloader