"""
Tests for AudioSegmentDataset (Phase 2).

All tests use synthetic WAV files generated in tmp_path — no real data needed.
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


# --- Window count ---

def test_window_count_single_file(tmp_path):
    """177 s audio, 30 s window, 0.5 s stride → floor((177-30)/0.5)+1 windows."""
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "a.wav", 44100, 177.0)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds = AudioSegmentDataset([str(wav)], window_duration=30.0, stride=0.5, target_sr=16000)

    expected = int((177.0 - 30.0) / 0.5) + 1
    assert len(ds) == expected


def test_window_count_multiple_files(tmp_path):
    """Dataset over two files sums window counts."""
    pytest.importorskip("scipy")
    wav1 = _make_wav(tmp_path / "a.wav", 44100, 60.0, seed=1)
    wav2 = _make_wav(tmp_path / "b.wav", 44100, 60.0, seed=2)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds1 = AudioSegmentDataset([str(wav1)], window_duration=30.0, stride=0.5, target_sr=16000)
    ds2 = AudioSegmentDataset([str(wav2)], window_duration=30.0, stride=0.5, target_sr=16000)
    ds_both = AudioSegmentDataset([str(wav1), str(wav2)], window_duration=30.0, stride=0.5, target_sr=16000)
    assert len(ds_both) == len(ds1) + len(ds2)


# --- Window content ---

def test_window_length_is_exact(tmp_path):
    """Every returned waveform must be exactly window_duration * target_sr samples."""
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "a.wav", 44100, 60.0)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds = AudioSegmentDataset([str(wav)], window_duration=30.0, stride=0.5, target_sr=16000)

    expected = int(30.0 * 16000)
    for i in range(min(5, len(ds))):
        waveform, _ = ds[i]
        assert len(waveform) == expected, f"window {i}: expected {expected} samples, got {len(waveform)}"


def test_waveform_is_1d_float32(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "a.wav", 44100, 35.0)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds = AudioSegmentDataset([str(wav)], window_duration=30.0, stride=0.5, target_sr=16000)
    waveform, _ = ds[0]

    assert waveform.ndim == 1, "waveform must be 1-D (mono)"
    assert waveform.dtype == np.float32


def test_stereo_source_returns_mono(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "a.wav", 44100, 35.0, stereo=True)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds = AudioSegmentDataset([str(wav)], window_duration=30.0, stride=0.5, target_sr=16000)
    waveform, _ = ds[0]
    assert waveform.ndim == 1


# --- Window IDs ---

def test_window_ids_are_unique_single_file(tmp_path):
    pytest.importorskip("scipy")
    wav = _make_wav(tmp_path / "a.wav", 44100, 60.0)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds = AudioSegmentDataset([str(wav)], window_duration=30.0, stride=0.5, target_sr=16000)
    ids = [ds[i][1] for i in range(len(ds))]
    assert len(ids) == len(set(ids)), "window IDs must be globally unique"


def test_window_ids_are_unique_across_files(tmp_path):
    pytest.importorskip("scipy")
    wav1 = _make_wav(tmp_path / "audio01.wav", 44100, 35.0, seed=1)
    wav2 = _make_wav(tmp_path / "audio02.wav", 44100, 35.0, seed=2)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds = AudioSegmentDataset([str(wav1), str(wav2)], window_duration=30.0, stride=0.5, target_sr=16000)
    ids = [ds[i][1] for i in range(len(ds))]
    assert len(ids) == len(set(ids)), "IDs across files must be unique"


def test_window_id_encodes_source_file(tmp_path):
    """Window IDs from different source files must not be interchangeable."""
    pytest.importorskip("scipy")
    wav1 = _make_wav(tmp_path / "audio01.wav", 44100, 35.0, seed=1)
    wav2 = _make_wav(tmp_path / "audio02.wav", 44100, 35.0, seed=2)

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds1 = AudioSegmentDataset([str(wav1)], window_duration=30.0, stride=0.5, target_sr=16000)
    ds2 = AudioSegmentDataset([str(wav2)], window_duration=30.0, stride=0.5, target_sr=16000)

    ids1 = {ds1[i][1] for i in range(len(ds1))}
    ids2 = {ds2[i][1] for i in range(len(ds2))}
    assert ids1.isdisjoint(ids2), "IDs from different source files must not overlap"


# --- Train/test split integrity ---

def test_train_test_split_no_leakage(tmp_path):
    """Windows from test segments (audio17-20) must not appear in the train set."""
    pytest.importorskip("scipy")
    train_wavs = [_make_wav(tmp_path / f"audio{i:02d}.wav", 44100, 35.0, seed=i) for i in range(1, 17)]
    test_wavs  = [_make_wav(tmp_path / f"audio{i:02d}.wav", 44100, 35.0, seed=i) for i in range(17, 21)]

    from mbs.extraction.data.datasets_audio import AudioSegmentDataset
    ds_train = AudioSegmentDataset([str(w) for w in train_wavs], window_duration=30.0, stride=0.5, target_sr=16000)
    ds_test  = AudioSegmentDataset([str(w) for w in test_wavs],  window_duration=30.0, stride=0.5, target_sr=16000)

    train_ids = {ds_train[i][1] for i in range(len(ds_train))}
    test_ids  = {ds_test[i][1]  for i in range(len(ds_test))}
    assert train_ids.isdisjoint(test_ids), "train and test window IDs must be disjoint"
