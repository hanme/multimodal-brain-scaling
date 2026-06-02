from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class WhisperBackboneWrapper(nn.Module):
    """Wraps a Whisper AudioEncoder so backbone.blocks.{i} paths work with HookedEncoder."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.backbone = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.backbone(mel)


class WhisperTransform:
    """Converts a 16 kHz float32 mono waveform to a Whisper log-mel spectrogram [n_mels, 3000]."""

    def __init__(self, n_mels: int = 80):
        self.n_mels = n_mels

    def __call__(self, waveform) -> torch.Tensor:
        import whisper

        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = whisper.pad_or_trim(waveform.float())  # → 480000 samples (30 s)
        mel = whisper.log_mel_spectrogram(waveform, n_mels=self.n_mels)  # [n_mels, 3000]
        return mel


def load_whisper(model_id: str, **kwargs) -> Tuple[WhisperBackboneWrapper, WhisperTransform]:
    """Load a Whisper encoder wrapped for hook-based feature extraction.

    Args:
        model_id: "whisper-tiny" | "whisper-base" | "whisper-small" | "whisper-medium" | "whisper-large"

    Returns:
        (WhisperBackboneWrapper, WhisperTransform)
        The wrapper exposes backbone.blocks.{i} for HookedEncoder layer paths.
    """
    import whisper

    size = model_id.removeprefix("whisper-")
    download_root = kwargs.get("model_cache_dir", None)
    model = whisper.load_model(size, download_root=download_root)
    encoder = model.encoder
    encoder.eval()

    wrapped = WhisperBackboneWrapper(encoder)
    transform = WhisperTransform(n_mels=model.dims.n_mels)
    return wrapped, transform


def load_model_audio(model_id: str, **kwargs):
    """Dispatcher for audio backbone loaders."""
    if model_id.startswith("whisper-"):
        return load_whisper(model_id, **kwargs)
    raise ValueError(
        f"Unknown audio model_id: '{model_id}'. "
        f"Supported: whisper-{{tiny,base,small,medium,large}}"
    )
