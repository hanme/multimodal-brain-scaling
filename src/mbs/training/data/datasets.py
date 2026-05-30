import warnings
from typing import Union, Optional, Literal, List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
import h5py

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root:Union[str, Path], file_list_path:Optional[Union[str, Path]]=None, transform=None):
        """
        Initializes the dataset by setting the root directory and loading samples based on the provided parameters.
        Parameters:
            root (Union[str, Path]): The root directory of the dataset. This can be a string path or a Path object.
            file_list_path (Optional[Union[str, Path]]): 
                If not None, represents the path to a file containing a list of samples. Each line in the file should consist of an image path 
                followed by its corresponding label (separated by a space). The samples will be loaded from this file.
                If None, the dataset is assumed to be organized in subdirectories within the root directory where each subdirectory name represents a class.
            transform (optional): A callable (e.g., a function or transform pipeline) that takes in a sample and returns a transformed version. 
                This transform will be applied to the loaded samples if provided.
        Attributes set:
            self.root: The dataset's root directory.
            self.transform: The transform callable to apply to each sample.
            self.samples: A list of tuples, where each tuple contains the file path (or Path object) to an image and its corresponding label.
            self.targets: A list containing the labels for each sample.
        Behavior:
            - If a file_list_path is provided (i.e., not None), the initializer reads the file, splits each line to extract image paths and labels, 
              and populates the samples list accordingly.
            - Otherwise, it scans the subdirectories of the root directory, assumes each subdirectory corresponds to a single class, assigns a unique 
              label per class (based on sorted order), and collects images from each subdirectory whose file extension is either '.jpeg', '.jpg', or '.png'.
            - In both cases, the targets attribute is populated with the labels extracted from the samples.
        """
        self.root = root
        self.transform = transform

        if file_list_path is not None:
            with open(file_list_path, "r") as f:
                file_list = f.readlines()
                self.samples = [(x.split(' ')[0].strip(), int(x.split(' ')[1].strip())) for x in file_list]
                print(f"Loaded {len(self.samples)} samples from {file_list_path}")
            self.targets = [x[1] for x in self.samples]
        else:
            root_path = Path(self.root)
            classes = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
            class2idx = {classes[i]: i for i in range(len(classes))}
            self.samples = []
            for class_name in classes:
                class_path = root_path / class_name
                # Image can be in jepg, jpg, and png format,
                self.samples.extend([(class_path / img, class2idx[class_name]) for img in class_path.iterdir() if img.suffix.lower() in ['.jpeg', '.jpg', '.png']])
                
            self.targets = [x[1] for x in self.samples]
            print(f"Loaded {len(self.samples)} samples from {self.root}")
        
    def apply_transform(self, image, transform):
        if isinstance(transform, T.Compose):
            image = transform(image)
            return image
        elif isinstance(transform, A.Compose):
            image = transform(image=np.array(image))["image"]
            return image
        else:
            raise NotImplementedError(f'Transforms of type {type(transform)} is not implemented')

    def __len__(self):
        return len(self.samples)
    
    def loader(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Image.open(self.root / path).convert('RGB')

    def __getitem__(self, idx):
        # Load an image and its target
        img_path, target = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.apply_transform(image, self.transform)
        
        return {
            'input': image,
            'target': target
        }
    
    # def __getitem__(self, idx):
        
    #     return idx



# 

class NeuralDataset(Dataset):
    def __init__(self, 
                 root:Union[str, Path],
                filename:Optional[str],
                regions:List[str],
                subjects:List[str],
                split:Literal["train", "test"],
                things_image_db_path:Union[str, Path]=None,
                nsd_image_h5_path:Union[str, Path]=None,
                transform=None,
                pct:float=100.0,
                random_state:int=0,
                random_shuffle:bool=False,
                 ):
        """
            Initializes the dataset by loading neural data and applying optional transformations.
                Args:
                    root (Union[str, Path]): The root directory where the neural data files are stored.
                    transform (callable, optional): A function/transform to apply to the stimuli.
                    split (str, optional): The split of the dataset to use. Default is "train".
                    filename (str, optional): The name of the file containing the neural data. Default is "TVSD_train_test.h5".
                    regions (list of str, optional): The regions of the brain to load data for. Default is ["V1", "V4", "IT"].
                Attributes:
                    root (Path): The root directory as a Path object.
                    transform (callable): The transformation function.
                    neural_responses (np.ndarray): The loaded neural responses with shape (N, num_neurons).
                    neural_stimuli (list of PIL.Image): The loaded and transformed neural stimuli.
                    neural_behaviors (np.ndarray): The loaded neural stimuli behavior with shape (N,).
                    pct (float): The percentage of the dataset to use for training. Default is 1.0 (100%).
                    random_state (int): The random seed for reproducibility. Default is 0.
                    
                    
        """
        self.root = root
        self.transform = transform
        self.pct = pct
        self.random_state = random_state
        self.split = split
        self.filename = filename
        self.subjects = subjects
        self.regions = regions
        self.things_image_db_path = things_image_db_path
        self.nsd_image_h5_path = nsd_image_h5_path
        self.random_shuffle = random_shuffle
        
        assert split in ["train", "test"], "Split should be either 'train' or 'test'"

        # Only one of Things or NSD image database path should be provided
        assert (things_image_db_path is None) != (nsd_image_h5_path is None), \
            "Either things_image_db_path or nsd_image_h5_path should be provided, but not both."

        print(f"Loading subjects: {self.subjects}")
        print(f"Loading regions: {self.regions}")
        
        assert 0 < self.pct <= 100, "Neural data percentage should be between 0 and 100"

        # Load neural data
        if isinstance(self.root, str):
            self.root = Path(self.root)
        # self.neural_responses = np.load(self.root / 'neural_responses.npy')  # Shape: (N, num_neurons)
        # self.neural_stimuli = np.load(self.root / 'stimuli.npy')  # Shape: (N, C, H, W)
        # self.neural_behaviors = np.load(self.root / 'labels.npy')  # Shape: (N,)
        
        self.neural_stimulus_ids = {}
        with h5py.File(self.root / filename, 'r') as f:
            self.all_subjects = f.attrs['subjects'][()]
            self.all_rois = f.attrs['rois'][()]
            
            if self.nsd_image_h5_path is not None:
                # For NSD, every subjects has their own stimulus ids
                
                for subject in self.all_subjects:
                    self.neural_stimulus_ids[subject] = f[split]['stimulus_ids'][subject][()]
            else:
                stimulus_ids = f[split]['stimulus_ids'][()]
                for subject in self.all_subjects:
                    self.neural_stimulus_ids[subject] = stimulus_ids
            
            # self.neural_stimuli = f[split]['stimuli'][()]
            # self.neural_behaviors = f[split]['labels'][()]
            
            assert all(region in self.all_rois for region in self.regions),\
                f"Some regions are not in the dataset. Available regions: {self.all_rois}"
            assert all(subject in self.all_subjects for subject in self.subjects),\
                f"Some subjects are not in the dataset. Available subjects: {self.all_subjects}"
            

                
        self._compute_available_indices()
        
        if self.random_shuffle:
            rng = np.random.default_rng(self.random_state)
            for subject in self.subjects:
                random_indices = rng.permutation(len(self.neural_stimulus_ids[subject]))
                self.neural_stimulus_ids[subject] = self.neural_stimulus_ids[subject][random_indices]
            print(f"Randomly shuffled neural data to break image-neural correspondence.")
        
        if self.pct < 100:
            rng = np.random.default_rng(self.random_state)
            total_neural_samples = len(self.available_indices)
            indices = np.arange(total_neural_samples)
            indices = rng.choice(indices, size=int(self.pct * total_neural_samples / 100), replace=False)
            print(f"Randomly selected {len(indices)} samples from {total_neural_samples} samples")
            self.available_indices = self.available_indices[indices]
            
            for subject in self.subjects:
                self.neural_stimulus_ids[subject] = self.neural_stimulus_ids[subject][self.available_indices]
            
            
        with h5py.File(self.root / filename, 'r') as f:
            subject = self.subjects[0]
            region = self.regions[0]
            neural_data_shape = f[split]['neural_data'][subject][region].shape
        
        if len(neural_data_shape) == 2:
            print(f"Neural data shape for subject {subject}, region {region}: {neural_data_shape} (2D)")
            self.is_temporal = False
        elif len(neural_data_shape) == 3:
            print(f"Neural data shape for subject {subject}, region {region}: {neural_data_shape} (3D - temporal)")
            self.is_temporal = True
        else:
            raise ValueError(f"Neural data shape for subject {subject}, region {region} has unsupported number of dimensions: {len(neural_data_shape)}")

        
        self.noise_ceiling_masks = {}
        if self.is_temporal:
            print("Temporal neural data detected. Creating noise ceiling masks.")
            with h5py.File(self.root / filename, 'r') as f:
                max_nc = f.attrs.get('max_nc', 100.0)
                for subject in self.subjects:
                    for region in self.regions:
                        noise_ceiling = f['noise_ceilings'][subject][region][()]
                        noise_ceiling = noise_ceiling.flatten() / max_nc + 1e-6  # to avoid division by zero
                        noise_ceiling= np.sqrt(noise_ceiling)
                        
                        noise_ceiling_mask = noise_ceiling > 0.1
                        self.noise_ceiling_masks[f"{subject}_{region}"] = noise_ceiling_mask
                        
        # self.neural_responses = {}
        # with h5py.File(self.root / filename, 'r') as f:
        #     for subject in self.all_subjects:
        #         self.neural_responses[subject] = {}
        #         for region in self.regions:
        #             neural_data = f[split]['neural_data'][subject][region][()]
        #             if self.is_temporal:
        #                 mask = self.noise_ceiling_masks[f"{subject}_{region}"]
        #                 neural_data = neural_data[:, mask]
                    
        #             neural_data = torch.reshape(neural_data, (neural_data.shape[0], -1))
        #             neural_data_mean = torch.mean(neural_data, axis=0, keepdims=True)
        #             neural_data_std = torch.std(neural_data, axis=0, keepdims=True) + 1e-6
        #             neural_data = (neural_data - neural_data_mean) / neural_data_std
        #             print(f"Normalized neural data from mean={neural_data_mean.mean():.4f}, std={neural_data_std.mean():.4f}.")
                    
        #             self.neural_responses[subject][region] = neural_data
                
        
        # self.neural_responses = torch.tensor(self.neural_responses, dtype=torch.float32)
        # self.neural_behaviors = torch.tensor(self.neural_behaviors, dtype=torch.long)
        # # self.neural_stimuli = torch.tensor(self.neural_stimuli, dtype=torch.float32)
        # self.neural_stimuli = self.neural_stimuli.transpose((0, 2, 3, 1)) # NCHW -> NHWC
        # self.neural_stimuli = [Image.fromarray((stimulus * 255).astype(np.uint8)) for stimulus in self.neural_stimuli]  # Convert to PIL

        if len(self.regions) > 0:
            # print(f"Loaded {len(self.neural_responses[self.subjects[0]][self.regions[0]])} neural responses from {self.root}")
            print(f"Loaded {len(self.available_indices)} neural responses from {self.root/self.filename}")
        else:
            print(f"Loaded 0 neural responses from {self.root}")
        # print(f"Loaded {len(self.neural_stimuli)} neural stimuli from {self.root}")
        # print(f"Loaded {len(self.neural_behaviors)} neural behaviors from {self.root}")
        
    def _compute_available_indices(self):
        # Since subjects in NSD can have different number of stimuli, we return the minimum length
        available_lengths = [len(self.neural_stimulus_ids[subject]) for subject in self.subjects]
        self._length = min(available_lengths)
        self.available_indices = np.arange(self._length)
        
    def apply_transform(self, image, transform):
        if isinstance(transform, T.Compose):
            image = transform(image)
            return image
        elif isinstance(transform, A.Compose):
            image = transform(image=np.array(image))["image"]
            return image
        else:
            raise NotImplementedError(f'Transforms of type {type(transform)} is not implemented')

    def __len__(self):
        return len(self.available_indices)
        # return len(self.neural_stimuli)

    def __getitem__(self, idx):
        
        # Load neural data
        # neural_responses = {
        #     subject: {
        #         region: torch.tensor(self.neural_responses[subject][region][idx], dtype=torch.float32)
        #         for region in self.regions
        #     } for subject in self.subjects
        # }
        
        with h5py.File(self.root / self.filename, 'r') as f:
            neural_responses = {
                subject: {
                    region: torch.tensor(f[self.split]['neural_data'][subject][region][self.available_indices[idx]], dtype=torch.float32).flatten()
                    for region in self.regions
                } for subject in self.subjects
            }
            
        if self.is_temporal:
            assert len(self.noise_ceiling_masks) > 0, "Noise ceiling masks not found for temporal data"
            
            # Apply noise ceiling masks
            for subject in self.subjects:
                for region in self.regions:
                    mask = self.noise_ceiling_masks[f"{subject}_{region}"]
                    neural_responses[subject][region] = neural_responses[subject][region][mask]
        
        
        # neural_behavior = torch.tensor(self.neural_behaviors[idx], dtype=torch.long)
        
        # neural_stimulus = self.neural_stimuli[idx]
        neural_stimuli = {}
        if self.nsd_image_h5_path is not None:
            # Load stimuluifrom NSD h5 file
            with h5py.File(self.nsd_image_h5_path, 'r') as f:
                for subject in self.subjects:
                    stim_idx = self.neural_stimulus_ids[subject][idx]
                
                    neural_stimulus = f['imgBrick'][stim_idx]
                    neural_stimulus = Image.fromarray(neural_stimulus)
                    neural_stimuli[subject] = neural_stimulus
        
        else:
            # Load stimulus from Things image database
            for subject in self.subjects:
                stim_filename = self.neural_stimulus_ids[subject][idx].decode('utf-8')
                # print(stim_filename)
                # folder_name = stim_filename.split('.')[0]
                # neural_stimulus_path = Path(self.things_image_db_path) / folder_name / stim_filename
                neural_stimulus_path = Path(self.things_image_db_path) / stim_filename
                neural_stimulus = Image.open(neural_stimulus_path).convert("RGB")
                neural_stimuli[subject] = neural_stimulus

        
        # neural_stimulus = neural_stimulus.transpose((1, 2, 0))  # CHW -> HWC
        # neural_stimulus = Image.fromarray((neural_stimulus * 255).astype(np.uint8))  # Convert to PIL
        
        # neural_response = self.neural_responses[idx]
        # neural_behavior = self.neural_behaviors[idx]
        # neural_stimulus = self.neural_stimuli[idx]
        
        # print(self.transform)
        # print(self.neural_stimulus_paths[idx])
        if self.transform:
            # neural_stimulus = self.apply_transform(neural_stimulus, self.transform)
            for subject in self.subjects:
                neural_stimuli[subject] = self.apply_transform(neural_stimuli[subject], self.transform)

        return {
            'neural_response': neural_responses,
            # 'neural_behavior': neural_behavior,
            'neural_stimulus': neural_stimuli
        }
        
    # def __getitem__(self, idx):
    #     # Load neural data
    #     return idx

