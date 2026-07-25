"""Tests for extract_features_delta_t.py's output naming and resume.

Why this matters more than it looks. load_layer_features() globs *.h5 in a feature directory,
concatenates them, and builds an id->row map that keeps only the LAST occurrence of each id. So a
directory whose contents do not line up with its ids does not raise -- it silently returns
misaligned features, and every downstream number is quietly wrong.

Two ways to get there, both of which these tests pin shut:
  * legacy start/batch filenames carry no stimulus identity, so a directory that gains clips
    between runs reuses a name for a different stimulus and clobbers the earlier file. That is
    exactly the novel search's Phase 1 -> Phase 2 top-up.
  * mixing the two naming schemes in one directory double-loads every clip.

The forward pass needs a real model, so these exercise the naming, the guards and the flag
plumbing -- all of which run before any model is loaded. The end-to-end resume (2 clips extracted,
directory grown to 16, exactly 14 re-extracted) was verified separately against whisper-tiny.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mbs.extraction.extract_features_delta_t import _batch_path  # noqa: E402

MODULE = "mbs.extraction.extract_features_delta_t"


def _run(tmp_path, data_root, output_dir, *extra):
    """Invoke the CLI far enough to hit the pre-model guards."""
    layers = tmp_path / "layers.json"
    if not layers.exists():
        layers.write_text('[{"name": "blocks.0", "position": 0.0}]')
    return subprocess.run(
        [sys.executable, "-m", MODULE, "--model_id", "whisper-tiny",
         "--data_root", str(data_root), "--target_feature_layers", str(layers),
         "--output_dir", str(output_dir), *extra],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(SRC), "PATH": __import__("os").environ["PATH"],
             "HOME": __import__("os").environ["HOME"]})


# ──────────────────────────────────────────────────────────
# Filenames
# ──────────────────────────────────────────────────────────

def test_stimulus_named_files_are_unique_per_clip(tmp_path):
    """The property the Phase-2 top-up depends on: a name identifies its stimulus, so a clip
    added later cannot take a name an earlier clip already owns."""
    ids = ["method_1001_N3_var1_deviant_0000000",
           "method_1001_N7_var1_deviant_0000000",
           "method_1001_standard_0000000"]
    paths = [_batch_path(tmp_path, 0, 0, [i], True) for i in ids]
    assert len({p.name for p in paths}) == 3
    for i, p in zip(ids, paths):
        assert p.name == f"feats_delta_t-{i}-seed_42.h5"
    # ...and the name does not depend on WHERE in the directory listing the clip fell
    assert _batch_path(tmp_path, 0, 0, [ids[0]], True) == \
           _batch_path(tmp_path, 9, 11, [ids[0]], True)


def test_legacy_names_collide_across_runs_of_different_sizes(tmp_path):
    """Documents the failure this feature exists to avoid, so nobody 'simplifies' it away.
    Phase 1's 2 clips and Phase 2's 14 both start at batch 0 of start 0."""
    phase1 = _batch_path(tmp_path, 0, 0, ["method_1001_N7_var1_deviant_0000000"], False)
    phase2 = _batch_path(tmp_path, 0, 0, ["method_1001_N3_var1_deviant_0000000"], False)
    assert phase1 == phase2, "legacy naming is position-based, not identity-based"
    assert phase1.name == "feats_delta_t-start_00000-batch_0-seed_42.h5"


def test_stimulus_naming_refuses_a_multi_stimulus_batch(tmp_path):
    with pytest.raises(AssertionError, match="save_every 1"):
        _batch_path(tmp_path, 0, 0, ["a", "b"], True)


# ──────────────────────────────────────────────────────────
# Guards (all fire before the model loads)
# ──────────────────────────────────────────────────────────

def test_stimulus_naming_requires_save_every_one(tmp_path):
    data = tmp_path / "wavs"
    data.mkdir()
    r = _run(tmp_path, data, tmp_path / "out", "--name_by_stim_id", "true", "--save_every", "8")
    assert r.returncode != 0
    assert "--save_every 1" in r.stdout + r.stderr


def test_refuses_stimulus_naming_into_a_legacy_directory(tmp_path):
    data, out = tmp_path / "wavs", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    (out / "feats_delta_t-start_00000-batch_0-seed_42.h5").touch()
    r = _run(tmp_path, data, out, "--name_by_stim_id", "true", "--save_every", "1")
    assert r.returncode != 0
    assert "already holds start/batch-named features" in r.stdout + r.stderr


def test_refuses_legacy_naming_into_a_stimulus_named_directory(tmp_path):
    data, out = tmp_path / "wavs", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    (out / "feats_delta_t-method_1001_standard_0000000-seed_42.h5").touch()
    r = _run(tmp_path, data, out, "--name_by_stim_id", "false")
    assert r.returncode != 0
    assert "already holds stimulus-named features" in r.stdout + r.stderr


def test_a_consistent_directory_passes_the_guard(tmp_path):
    """Same scheme both times must NOT be blocked -- that is the resume path. It gets past the
    guards and fails later, on the empty wav directory."""
    data, out = tmp_path / "wavs", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    (out / "feats_delta_t-method_1001_standard_0000000-seed_42.h5").touch()
    r = _run(tmp_path, data, out, "--name_by_stim_id", "true", "--save_every", "1")
    combined = r.stdout + r.stderr
    assert "already holds" not in combined
    assert "No .wav files found" in combined


def test_flags_are_accepted_and_recorded(tmp_path):
    """--overwrite used to be declared and never read; both flags must reach args."""
    data = tmp_path / "wavs"
    data.mkdir()
    r = _run(tmp_path, data, tmp_path / "out",
             "--name_by_stim_id", "true", "--save_every", "1", "--overwrite", "true")
    assert "'name_by_stim_id': True" in r.stdout
    assert "'overwrite': True" in r.stdout
