from math import gcd
from pathlib import Path
from typing import List, Union

import numpy as np


class AudioPreprocessor:
    """Load a WAV file, convert to mono float32, and resample to target_sr."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def __call__(self, wav_path: Union[str, Path]) -> np.ndarray:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(wav_path))

        # Normalise to float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # Stereo → mono
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Resample if needed
        if sr != self.target_sr:
            from scipy.signal import resample_poly
            g = gcd(int(sr), int(self.target_sr))
            data = resample_poly(data, self.target_sr // g, sr // g).astype(np.float32)

        return data


class AudioSegmentDataset:
    """Sliding-window dataset over a list of WAV files.

    Returns (waveform_or_transformed, window_id) per item.  The window_id encodes the source
    filename stem, so files from disjoint lists produce disjoint window-ID sets — required
    for a clean train/test split at the segment (file) level.
    """

    def __init__(
        self,
        wav_files: List[Union[str, Path]],
        window_duration: float = 30.0,
        stride: float = 0.5,
        target_sr: int = 16000,
        transform=None,
    ):
        self.wav_files = [str(p) for p in wav_files]
        self.window_duration = window_duration
        self.stride = stride
        self.target_sr = target_sr
        self.transform = transform
        self._preprocessor = AudioPreprocessor(target_sr)
        self._waveforms: dict = {}
        self._index = self._build_index()

    def _build_index(self) -> list:
        index = []
        window_samples = int(self.window_duration * self.target_sr)
        stride_samples = int(self.stride * self.target_sr)
        for path in self.wav_files:
            waveform = self._preprocessor(path)
            self._waveforms[path] = waveform
            start = 0
            while start + window_samples <= len(waveform):
                index.append((path, start))
                start += stride_samples
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        path, start = self._index[idx]
        window_samples = int(self.window_duration * self.target_sr)
        window = self._waveforms[path][start : start + window_samples].copy()
        window_id = f"{Path(path).stem}_{start:07d}"
        if self.transform is not None:
            window = self.transform(window)
        return window, window_id
