"""
Tests for the audio backbone loader (Phase 1).

Fast tests use synthetic WAV files; slow tests (marked @pytest.mark.slow)
download Whisper-tiny weights and run a full forward pass.
"""

import numpy as np
import pytest


def _make_wav(path, sr, duration_s, stereo=False, seed=42):
    from scipy.io import wavfile
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)
    data = rng.uniform(-0.5, 0.5, (n, 2) if stereo else (n,)).astype(np.float32)
    wavfile.write(str(path), sr, data)
    return path


# --- Backbone registry ---

def test_model_loaders_includes_audio():
    from mbs.extraction.modeling.backbones import MODEL_LOADERS
    assert "audio" in MODEL_LOADERS


def test_create_backbone_audio_returns_model_and_transform():
    pytest.importorskip("whisper")
    from mbs.extraction.modeling.backbones import create_backbone
    model, transform = create_backbone(model_id="whisper-tiny", backbone_source="audio")
    assert model is not None
    assert callable(transform)


def test_create_feature_extractor_dispatches_audio_to_hooks():
    """backbone_source='audio' must go through the hook-based path."""
    pytest.importorskip("whisper")
    from mbs.extraction.modeling import create_feature_extractor
    from mbs.extraction.modeling.encoder_hooks import HookFeatureExtractor
    feat_layers = ["blocks.0"]
    fe, _ = create_feature_extractor(
        model_id="whisper-tiny",
        backbone_source="audio",
        feat_layers=feat_layers,
        max_feature_dim=0,
    )
    assert isinstance(fe, HookFeatureExtractor)


# --- AudioPreprocessor ---

def test_preprocessor_converts_stereo_to_mono(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "s.wav", 44100, 1.0, stereo=True)
    from mbs.extraction.data.datasets_audio import AudioPreprocessor
    waveform = AudioPreprocessor(target_sr=16000)(str(wav))
    assert waveform.ndim == 1, "output must be 1-D (mono)"


def test_preprocessor_resamples_to_target_sr(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "s.wav", 44100, 1.0)
    from mbs.extraction.data.datasets_audio import AudioPreprocessor
    waveform = AudioPreprocessor(target_sr=16000)(str(wav))
    # 1s at 16 kHz: allow ±1 sample for resampling rounding
    assert abs(len(waveform) - 16000) <= 1


def test_preprocessor_30s_at_16k(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "s.wav", 16000, 30.0)
    from mbs.extraction.data.datasets_audio import AudioPreprocessor
    waveform = AudioPreprocessor(target_sr=16000)(str(wav))
    assert len(waveform) == 16000 * 30


def test_preprocessor_output_dtype_is_float32(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "s.wav", 16000, 1.0)
    from mbs.extraction.data.datasets_audio import AudioPreprocessor
    waveform = AudioPreprocessor(target_sr=16000)(str(wav))
    assert waveform.dtype == np.float32


# --- Slow: actual Whisper forward pass ---

@pytest.mark.slow
def test_whisper_tiny_block_output_shape(tmp_path):
    """30 s of noise → each Whisper-tiny block outputs shape [..., 1500, 384]."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("whisper")
    pytest.importorskip("scipy")

    wav = _make_wav(tmp_path / "a.wav", 16000, 30.0)

    from mbs.extraction.modeling.backbones import create_backbone
    from mbs.extraction.modeling.encoder_hooks import HookedEncoder
    from mbs.extraction.data.datasets_audio import AudioPreprocessor

    # whisper-tiny: 4 encoder blocks, d_model=384
    model, transform = create_backbone(model_id="whisper-tiny", backbone_source="audio")
    model.eval()

    # Paths relative to BackboneWrapperHF: backbone.blocks.{i}
    n_blocks = 4
    feat_layers = {f"backbone.blocks.{i}": f"block_{i}" for i in range(n_blocks)}
    hooked = HookedEncoder(model, feat_layers, include_output=False)

    waveform = AudioPreprocessor(target_sr=16000)(str(wav))
    mel = transform(waveform)          # [80, 3000] or [1, 80, 3000]
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)         # add batch dim → [1, 80, 3000]

    with torch.no_grad():
        features = hooked(mel)

    assert len(features) == n_blocks
    for alias, feat in features.items():
        assert feat.shape[-2] == 1500, f"{alias}: expected T=1500, got {feat.shape}"
        assert feat.shape[-1] == 384,  f"{alias}: expected d_model=384, got {feat.shape}"


@pytest.mark.slow
def test_whisper_base_has_6_encoder_blocks():
    pytest.importorskip("torch")
    pytest.importorskip("whisper")
    from mbs.extraction.modeling.backbones import create_backbone
    model, _ = create_backbone(model_id="whisper-base", backbone_source="audio")
    assert len(model.backbone.blocks) == 6
