from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

import h5py

class THINGSImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(list(self.root_dir.glob('*/*.jpg')))
        self.transform = transform
        self.default_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = self.default_transform(img)

        img_loc = str(img_path.relative_to(self.root_dir))

        return x, img_loc
    
class H5ImageDataset(Dataset):
    """
    Memory-efficient dataset for images stored in a single HDF5 file.

    Expected HDF5 layout (example):

        /${x_key}  -> shape (N, H, W, C)
        /${y_key}  -> shape (N, ... )  - optional

    Only the requested index is read into memory on each __getitem__ call.
    """

    def __init__(
        self,
        h5_path: str,
        x_key: str = "imgBrick",
        y_key: str | None = None,
        transform=None,
        target_transform=None,
    ):
        self.h5_path = h5_path
        self.x_key = x_key
        self.y_key = y_key
        self.transform = transform
        self.target_transform = target_transform
        self.default_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

        # We only open the file briefly here to get the length.
        # The long-lived handle will be opened lazily in each process.
        with h5py.File(self.h5_path, "r") as f:
            self._length = f[self.x_key].shape[0]

        # Long-lived handles (per process), opened on first access
        self._file = None
        self._x = None
        self._y = None

    # ----------------- internal helpers -----------------

    def _ensure_file_open(self):
        """Open HDF5 file and datasets on first use in this process."""
        if self._file is None:
            # read-only is enough; one handle per process/worker
            self._file = h5py.File(self.h5_path, "r")
            self._x = self._file[self.x_key]
            if self.y_key is not None:
                self._y = self._file[self.y_key]

    def _close_file(self):
        """Close HDF5 file if it's open."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._x = None
            self._y = None

    # ----------------- PyTorch Dataset API -----------------

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._ensure_file_open()

        # HDF5 slice -> NumPy array
        x = self._x[idx]

        # Convert to PIL.Image
        x = Image.fromarray(x)

        if self.transform is not None:
            x = self.transform(x)
        else:
            x = self.default_transform(x)

        if self._y is not None:
            y = self._y[idx]
            y = torch.from_numpy(y)
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y, idx

        return x, idx

    # Make sure file handle is closed when dataset is garbage-collected
    def __del__(self):
        self._close_file()
        
class BrainScoreStimulusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimulus_set_id: str,
        transform=None,
    ):
        try:
            from brainscore_vision import load_stimulus_set
        except ImportError as exc:
            raise ImportError(
                "Brain-Score stimulus sets require the 'brainscore-vision' package. "
                "Install the evaluation extra or use dataset_type='h5'/'things'."
            ) from exc

        self.stimulus_set_id = stimulus_set_id
        self.stimulus_set = load_stimulus_set(stimulus_set_id)
        self.stimulus_set = self.stimulus_set.sort_values('stimulus_id').reset_index(drop=True)
        self.transform = transform
        self.default_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.stimulus_set)

    def __getitem__(self, idx):
        stimulus = self.stimulus_set.iloc[idx]
        stimulus_id = stimulus.stimulus_id
        img_path = self.stimulus_set.get_stimulus(stimulus_id)
        
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            x = self.transform(img)
        else:
            x = self.default_transform(img)
            

        return x, stimulus_id
