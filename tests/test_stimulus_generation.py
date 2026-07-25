"""Tests for 00aa_generate_audio_stimuli.py's deviant-grid flags.

The load-bearing property here is REGENERATION STABILITY. The novel search generates each method
directory twice -- Phase 1 writes 1 standard + the N7/var1 deviant, Phase 2 later fills in the
other 14 deviants into the same directory -- and reuses Phase 1's extracted features for the two
overlapping clips. That reuse is only valid if the second generation reproduces those two files
byte-for-byte. If it ever stops doing so, the ~30 CHF Phase-2 saving silently becomes ~30 CHF of
wrong features, with no error anywhere.

generate_deviant_sequence seeds on (method_id, N, v) alone, so nothing about the grid size, the
row count or the worker count should perturb an already-generated file. These tests pin that.

wav2vec2 (10 s clips) is used where one family suffices, to keep the suite fast.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEN = SCRIPTS / "00aa_generate_audio_stimuli.py"

HEADER = ("source,method_id,paradigm,change_type,standard_id,standard_freq,standard_dur,"
          "standard_isi,standard_soa,standard_int,deviant_id,deviant_freq,deviant_int,"
          "deviant_dur,deviant_isi,deviant_soa,p_deviant_pc\n")
ROW = "novel_grid,1001,oddball,Frequency,1001,200,80,500,580,80,1001,225,80,80,500,580,10\n"


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _generate(out_dir, csv_path, *extra, workers=1):
    r = subprocess.run(
        [sys.executable, str(GEN), "--metadata_csv", str(csv_path),
         "--output_dir", str(out_dir), "--n_workers", str(workers), *extra],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture
def metadata(tmp_path):
    p = tmp_path / "grid.csv"
    p.write_text(HEADER + ROW)
    return p


# ──────────────────────────────────────────────────────────
# The regeneration invariant the Phase-2 feature reuse rests on
# ──────────────────────────────────────────────────────────

def test_phase2_regeneration_is_byte_identical_to_phase1(tmp_path, metadata):
    """1 standard + N7/var1 must come out identical when the full deviant grid is generated
    later into the same method, at a different worker count."""
    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    _generate(p1, metadata, "--trial_levels", "7", "--num_variations", "1",
              "--models", "wav2vec2", workers=1)
    _generate(p2, metadata, "--trial_levels", "3,5,7", "--num_variations", "5",
              "--models", "wav2vec2", workers=4)

    for cfg in ("regular", "counter"):
        a = p1 / f"audio_outputs_{cfg}" / "wav2vec2"
        b = p2 / f"audio_outputs_{cfg}" / "wav2vec2"
        overlapping = sorted(x.name for x in a.glob("*.wav"))
        assert overlapping == ["method_1001_N7_var1_deviant.wav", "method_1001_standard.wav"]
        for name in overlapping:
            assert _md5(a / name) == _md5(b / name), f"{cfg}/{name} changed between phases"


def test_phase1_emits_exactly_the_two_clips_it_pays_for(tmp_path, metadata):
    out = tmp_path / "p1"
    _generate(out, metadata, "--trial_levels", "7", "--num_variations", "1",
              "--models", "whisper,wav2vec2")
    for cfg in ("regular", "counter"):
        for fam in ("whisper", "wav2vec2"):
            wavs = sorted(p.name for p in (out / f"audio_outputs_{cfg}" / fam).glob("*.wav"))
            assert wavs == ["method_1001_N7_var1_deviant.wav", "method_1001_standard.wav"], \
                f"{cfg}/{fam}"


def test_default_flags_still_emit_the_full_16_clip_grid(tmp_path, metadata):
    """Back-compat: the literature runs pass no grid flags and must be unaffected."""
    out = tmp_path / "full"
    _generate(out, metadata, "--models", "wav2vec2")
    wavs = list((out / "audio_outputs_regular" / "wav2vec2").glob("*.wav"))
    assert len(wavs) == 16                                  # 1 standard + 3 N x 5 variations


def test_deviant_filenames_carry_the_substrings_analyze_method_dispatches_on(tmp_path, metadata):
    """insilico_mmn.analyze_method splits standard from deviant on these substrings, and
    finalize_method finds the N7/var1 trace by 'n7' + 'var1'. Renaming breaks scoring silently."""
    out = tmp_path / "g"
    _generate(out, metadata, "--models", "wav2vec2")
    names = [p.name.lower() for p in (out / "audio_outputs_regular" / "wav2vec2").glob("*.wav")]
    assert sum("standard" in n and "deviant" not in n for n in names) == 1
    assert sum("deviant" in n for n in names) == 15
    assert sum("n7" in n and "var1" in n for n in names) == 1


def test_counterbalancing_swaps_only_the_frequencies(tmp_path, metadata):
    """The two directions must differ, but share the tone-identity sequence -- the counter
    direction is the same design with standard and deviant frequencies exchanged."""
    out = tmp_path / "g"
    _generate(out, metadata, "--trial_levels", "7", "--num_variations", "1",
              "--models", "wav2vec2")
    reg = out / "audio_outputs_regular" / "wav2vec2" / "method_1001_standard.wav"
    ctr = out / "audio_outputs_counter" / "wav2vec2" / "method_1001_standard.wav"
    assert _md5(reg) != _md5(ctr), "counter direction must not duplicate the regular one"

    import pandas as pd
    mr = pd.read_csv(out / "audio_outputs_regular_metadata" / "metadata.csv").iloc[0]
    mc = pd.read_csv(out / "audio_outputs_counter_metadata" / "metadata.csv").iloc[0]
    assert (mr.standard_freq, mr.deviant_freq) == (200, 225)
    assert (mc.standard_freq, mc.deviant_freq) == (225, 200)
    assert mr.sequence == mc.sequence


# ──────────────────────────────────────────────────────────
# Flag plumbing
# ──────────────────────────────────────────────────────────

def test_models_flag_restricts_which_families_are_synthesized(tmp_path, metadata):
    out = tmp_path / "g"
    _generate(out, metadata, "--trial_levels", "7", "--num_variations", "1",
              "--models", "wav2vec2")
    made = sorted(p.name for p in (out / "audio_outputs_regular").iterdir())
    assert made == ["wav2vec2"], "vggish/ast/whisper should not be generated"


def test_grid_flags_survive_multiprocessing_workers(tmp_path):
    """The grid travels in a picklable StimulusGrid rather than module state -- macOS spawns
    workers rather than forking, so mutated module constants would not reach them."""
    csv_path = tmp_path / "many.csv"
    csv_path.write_text(HEADER + "".join(
        ROW.replace("1001", str(1001 + i)).replace(",200,", f",{200 + 10 * i},")
        for i in range(6)))
    out = tmp_path / "g"
    _generate(out, csv_path, "--trial_levels", "7", "--num_variations", "1",
              "--models", "wav2vec2", workers=4)
    wavs = list((out / "audio_outputs_regular" / "wav2vec2").glob("*.wav"))
    assert len(wavs) == 12, "6 methods x (1 standard + 1 deviant); a worker lost the grid"


@pytest.mark.parametrize("bad,msg", [
    (["--trial_levels", "x"], "comma-separated integers"),
    (["--trial_levels", ""], "empty"),
    (["--num_variations", "0"], "must be >= 1"),
    (["--models", "nope"], "unknown model"),
])
def test_bad_grid_flags_are_rejected_with_a_message(tmp_path, metadata, bad, msg):
    r = subprocess.run(
        [sys.executable, str(GEN), "--metadata_csv", str(metadata),
         "--output_dir", str(tmp_path / "out"), *bad],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0
    assert msg in r.stdout + r.stderr
