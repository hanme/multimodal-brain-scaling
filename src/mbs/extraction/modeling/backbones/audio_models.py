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
    # Pin whisper-large to large-v3 (chosen version); tiny/base/small/medium are unchanged so
    # their existing extracted features stay reproducible. large-v3 has 32 encoder blocks
    # (matching whisper_large_layers.json) and 128 mel bins (WhisperTransform reads n_mels dynamically).
    if size == "large":
        size = "large-v3"
    download_root = kwargs.get("model_cache_dir", None)
    model = whisper.load_model(size, download_root=download_root)
    encoder = model.encoder
    encoder.eval()

    wrapped = WhisperBackboneWrapper(encoder)
    transform = WhisperTransform(n_mels=model.dims.n_mels)
    return wrapped, transform


# ---------------------------------------------------------------------------
# wav2vec2 (raw-waveform transformer; NO mel front-end)
# ---------------------------------------------------------------------------

# Which HF checkpoint each logical size maps to. "medium"/"large" are OUR names;
# the pinned ids are the pretrained (self-supervised, no ASR fine-tuning) releases.
WAV2VEC2_CHECKPOINTS = {
    "wav2vec2-medium": "facebook/wav2vec2-base",    # 12 transformer layers, 768-d
    "wav2vec2-large":  "facebook/wav2vec2-large",   # 24 transformer layers, 1024-d
}


class Wav2Vec2BackboneWrapper(nn.Module):
    """Wraps a HF Wav2Vec2Model so backbone.encoder.layers.{i} paths work with HookedEncoder.

    forward() takes a batch of raw 16 kHz waveforms [B, samples] and returns the final
    hidden state. Only the hooked encoder-layer outputs are consumed downstream, so the
    return value is incidental (HookedEncoder is built with include_output=False).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_values).last_hidden_state


class Wav2Vec2Transform:
    """Raw-waveform front-end for wav2vec2.

    Unlike WhisperTransform there is NO mel spectrogram: wav2vec2 consumes the raw 16 kHz
    mono waveform directly. __call__ applies the checkpoint's own zero-mean/unit-variance
    normalization (when its feature extractor sets do_normalize) to a full window and returns
    a 1-D float32 tensor of model-ready input_values.

    The delta-t extractor also reads ``do_normalize``, ``norm_eps`` and ``conv_stride`` to
    reproduce the same normalization on causally-truncated waveforms (see extract_features_delta_t).
    """

    def __init__(self, feature_extractor, conv_stride: int, sampling_rate: int = 16000):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate
        self.conv_stride = int(conv_stride)                       # total conv downsampling (e.g. 320)
        self.do_normalize = bool(getattr(feature_extractor, "do_normalize", True))
        self.norm_eps = 1e-7                                      # HF zero_mean_unit_var_norm epsilon

    def __call__(self, waveform) -> torch.Tensor:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        out = self.feature_extractor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt",
        )
        return out.input_values[0].float()                       # [samples]


def load_wav2vec2(model_id: str, **kwargs) -> Tuple[Wav2Vec2BackboneWrapper, Wav2Vec2Transform]:
    """Load a pretrained wav2vec2 encoder wrapped for hook-based feature extraction.

    Args:
        model_id: "wav2vec2-medium" (facebook/wav2vec2-base) | "wav2vec2-large"
                  (facebook/wav2vec2-large).

    Returns:
        (Wav2Vec2BackboneWrapper, Wav2Vec2Transform)
        The wrapper exposes backbone.encoder.layers.{i} for HookedEncoder layer paths.
    """
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    if model_id not in WAV2VEC2_CHECKPOINTS:
        raise ValueError(
            f"Unknown wav2vec2 model_id: '{model_id}'. "
            f"Supported: {sorted(WAV2VEC2_CHECKPOINTS)}"
        )
    hf_id = WAV2VEC2_CHECKPOINTS[model_id]
    cache_dir = kwargs.get("model_cache_dir", None)

    model = Wav2Vec2Model.from_pretrained(hf_id, cache_dir=cache_dir)
    model.eval()  # disables SpecAugment time/feature masking (train-only), so extraction is deterministic

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_id, cache_dir=cache_dir)
    conv_stride = int(np.prod(model.config.conv_stride))         # 5*2*2*2*2*2*2 = 320

    wrapped = Wav2Vec2BackboneWrapper(model)
    transform = Wav2Vec2Transform(feature_extractor, conv_stride=conv_stride)
    return wrapped, transform


def load_model_audio(model_id: str, **kwargs):
    """Dispatcher for audio backbone loaders."""
    if model_id.startswith("whisper-"):
        return load_whisper(model_id, **kwargs)
    if model_id.startswith("wav2vec2-"):
        return load_wav2vec2(model_id, **kwargs)
    raise ValueError(
        f"Unknown audio model_id: '{model_id}'. "
        f"Supported: whisper-{{tiny,base,small,medium,large}}, "
        f"wav2vec2-{{medium,large}}"
    )
