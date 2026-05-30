
import os
import warnings
from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms as T
import albumentations as A

from PIL import Image


import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from .datasets import ImageDataset, NeuralDataset
from .transforms import create_transforms_image, create_transforms_neural


def collate_fn_neural(batch):
    # neural_stimuli = torch.stack([stimulus for stimulus in batch['neural_stimuli']], dim=0)
    neural_stimuli = {
        subject: torch.stack([stimuli[subject] for stimuli in batch['neural_stimulus']], dim=0)
        for subject in batch['neural_stimulus'][0].keys()
    }
    neural_responses = { 
        subject: {
            region: torch.stack([response[subject][region] for response in batch['neural_response']], dim=0)
            for region in batch['neural_response'][0][subject].keys()
        } for subject in batch['neural_response'][0].keys()
    }
    return {
        'neural_stimulus': neural_stimuli,
        'neural_response': neural_responses,
    }

class CombinedDataModule(L.LightningDataModule):
    def __init__(self, 
                 image_data_root:Union[str, Path], 
                 neural_data_root:Union[str, Path], 
                 neural_data_filename:str,
                 neural_data_subjects:Optional[List[str]],
                 neural_data_regions:Optional[List[str]],
                 image_transform_train:Union[T.Compose, A.Compose],
                 image_transform_val:Union[T.Compose, A.Compose],
                 neural_stimuli_transform_train:Union[T.Compose, A.Compose], 
                 neural_stimuli_transform_val:Union[T.Compose, A.Compose], 
                 things_image_db_path:Optional[Union[str, Path]]=None,
                 nsd_image_h5_path:Optional[Union[str, Path]]=None,
                 batch_size_image:int=32,
                 batch_size_neural:int=32,
                 num_workers:int=4,
                 pin_memory:bool=True,
                 drop_last_train:bool=True,
                 repeated_aug:bool=False,
                 combined_loader_mode_train:Literal['min_size', 'max_size_cycle', 'max_size', 'sequential']='min_size',
                 combined_loader_mode_val:Literal['min_size', 'max_size_cycle', 'max_size', 'sequential']='sequential',
                 neural_data_pct:float=1.0,
                 neural_data_random_state:int=0,
                 neural_data_random_shuffle:bool=False,
                 ):
        super().__init__()
        self.image_data_root = image_data_root
        self.neural_data_root = neural_data_root
        self.neural_data_filename = neural_data_filename
        self.neural_data_subjects = neural_data_subjects
        self.neural_data_regions = neural_data_regions
        self.things_image_db_path = things_image_db_path
        self.nsd_image_h5_path = nsd_image_h5_path
        self.image_transform_train = image_transform_train
        self.image_transform_val = image_transform_val
        self.neural_stimuli_transform_train = neural_stimuli_transform_train
        self.neural_stimuli_transform_val = neural_stimuli_transform_val
        self.batch_size_image = batch_size_image
        self.batch_size_neural = batch_size_neural
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train
        self.repeated_aug = repeated_aug
        self.combined_loader_mode_train = combined_loader_mode_train
        self.combined_loader_mode_val = combined_loader_mode_val
        self.neural_data_pct = neural_data_pct
        self.neural_data_random_state = neural_data_random_state
        self.neural_data_random_shuffle = neural_data_random_shuffle
        
        assert (self.things_image_db_path is not None) != (self.nsd_image_h5_path is not None) , \
            "Either things_image_db_path or nsd_image_h5_path must be provided, but not both."
        
        assert self.repeated_aug is False,\
            "Repeated augmentation is not implemented yet"
        
        if not isinstance(self.image_data_root, Path):
            self.image_data_root = Path(self.image_data_root)
        if not isinstance(self.neural_data_root, Path):
            self.neural_data_root = Path(self.neural_data_root)
        
    def setup(self, stage: str = None):
        self.image_dataset_train = ImageDataset(root=self.image_data_root / 'train', transform=self.image_transform_train)
        self.neural_dataset_train = NeuralDataset(
            root=self.neural_data_root, 
            filename=self.neural_data_filename,
            things_image_db_path=self.things_image_db_path,
            nsd_image_h5_path=self.nsd_image_h5_path,
            regions=self.neural_data_regions,
            subjects=self.neural_data_subjects,
            transform=self.neural_stimuli_transform_train, 
            pct=self.neural_data_pct, 
            random_state=self.neural_data_random_state,
            split='train',
            random_shuffle=self.neural_data_random_shuffle,
        )
        
        self.image_dataset_val = ImageDataset(root=self.image_data_root / 'val', transform=self.image_transform_val)
        self.neural_dataset_val = NeuralDataset(
            root=self.neural_data_root,
            filename=self.neural_data_filename,
            things_image_db_path=self.things_image_db_path,
            nsd_image_h5_path=self.nsd_image_h5_path,
            regions=self.neural_data_regions,
            subjects=self.neural_data_subjects,
            transform=self.neural_stimuli_transform_val, 
            split='test'
        )
    
        
    def train_dataloader(self):
        image_dataloader_train = torch.utils.data.DataLoader(
            self.image_dataset_train, 
            batch_size=self.batch_size_image, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train
        )
        neural_dataloader_train = torch.utils.data.DataLoader(
            self.neural_dataset_train, 
            batch_size=self.batch_size_neural, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train
        )
        loaders = {
            'image': image_dataloader_train,
            'neural': neural_dataloader_train
        }

        combined_loader_train = CombinedLoader(
            iterables=loaders,
            mode=self.combined_loader_mode_train # 'min_size', 'max_size_cycle', 'max_size', 'sequential'
        )
        return combined_loader_train

    def val_dataloader(self):
        image_dataloader_val = torch.utils.data.DataLoader(
            self.image_dataset_val, 
            batch_size=self.batch_size_image, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        neural_dataloader_val = torch.utils.data.DataLoader(
            self.neural_dataset_val, 
            batch_size=self.batch_size_neural, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        loaders = {
            'image': image_dataloader_val,
            'neural': neural_dataloader_val
        }

        combined_loader_val = CombinedLoader(
            iterables=loaders,
            mode=self.combined_loader_mode_val # 'min_size', 'max_size_cycle', 'max_size', 'sequential'
        )
        return combined_loader_val
    
    
def create_datamodule(**kwargs) -> CombinedDataModule:
    
    
    image_transform_train, image_transform_val = create_transforms_image(**kwargs)
    neural_stimuli_transform_train, neural_stimuli_transform_val = create_transforms_neural(**kwargs)
    
    
    data_neural_regions = kwargs['data_neural_regions']
    data_neural_regions= data_neural_regions.split(',') if not (data_neural_regions in ["", "None", None]) else []

    data_module = CombinedDataModule(
        image_data_root=kwargs['data_path_image'],
        neural_data_root=kwargs['data_path_neural'],
        neural_data_filename=kwargs['data_neural_filename'],
        things_image_db_path=kwargs.get('things_image_db_path', None),
        nsd_image_h5_path=kwargs.get('nsd_image_h5_path', None),
        neural_data_regions=data_neural_regions,
        neural_data_subjects=kwargs['data_neural_subjects'],
        image_transform_train=image_transform_train,
        image_transform_val=image_transform_val,
        neural_stimuli_transform_train=neural_stimuli_transform_train,
        neural_stimuli_transform_val=neural_stimuli_transform_val,
        batch_size_image=kwargs['batch_size_image'],
        batch_size_neural=kwargs['batch_size_neural'],
        num_workers=kwargs['workers'],
        pin_memory=kwargs['pin_memory'],
        drop_last_train=kwargs['drop_last_train'],
        repeated_aug=kwargs['repeated_aug'],
        combined_loader_mode_train=kwargs['combined_loader_mode_train'],
        combined_loader_mode_val=kwargs['combined_loader_mode_val'],
        neural_data_pct=kwargs['neural_data_pct'],
        neural_data_random_state=kwargs['seed'],
        neural_data_random_shuffle=kwargs['neural_data_random_shuffle'],
    )
    
    return data_module