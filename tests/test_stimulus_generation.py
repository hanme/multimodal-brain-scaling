"""Tests for 00aa_generate_audio_stimuli.py's deviant-grid flags and trailing-silence floor.

The load-bearing property here is REGENERATION STABILITY. The novel search generates each method
directory twice -- Phase 1 writes 1 standard + the N7/var1 deviant, Phase 2 later fills in the
other 14 deviants into the same directory -- and reuses Phase 1's extracted features for the two
overlapping clips. That reuse is only valid if the second generation reproduces those two files
byte-for-byte. If it ever stops doing so, the ~30 CHF Phase-2 saving silently becomes ~30 CHF of
wrong features, with no error anywhere.

generate_deviant_sequence seeds on (method_id, N, v) alone, so nothing about the grid size, the
row count or the worker count should perturb an already-generated file. These tests pin that.

The second property, pinned lower down, is the TRAILING-SILENCE FLOOR. compute_tone_slots reserves
max(trailing_floor_ms, SOA) of audio after the final tone's onset; the default 0.0 collapses to the
historical formula, so the flag is a strict opt-in and no already-generated set can drift.

wav2vec2 (10 s clips) is used where one family suffices, to keep the suite fast.
"""

import csv
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEN = SCRIPTS / "00aa_generate_audio_stimuli.py"


def _load_generator():
    """Import the generator for direct calls. The filename starts with digits, so it cannot be
    imported by name -- same spec_from_file_location dance as tests/test_insilico_mmn.py."""
    spec = importlib.util.spec_from_file_location("gen_audio_stimuli", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()

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
    (["--trailing_floor_ms", "-1"], "must be >= 0"),
])
def test_bad_grid_flags_are_rejected_with_a_message(tmp_path, metadata, bad, msg):
    r = subprocess.run(
        [sys.executable, str(GEN), "--metadata_csv", str(metadata),
         "--output_dir", str(tmp_path / "out"), *bad],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0
    assert msg in r.stdout + r.stderr


# ──────────────────────────────────────────────────────────
# The trailing-silence floor
# ──────────────────────────────────────────────────────────
#
# The historical layout ends a clip exactly one SOA after the final tone's onset. The MMN criteria
# read out to 360 ms past that tone (S2's recovery search runs 120 ms past a trough that may sit as
# late as 240 ms), so short-SOA literature rows were scored on a truncated epoch -- and where the
# trough landed on the final sample, S2's unguarded `z[imin+1:imin+1+6]` slice came back empty and
# returned False without being tested. --trailing_floor_ms 400 fixes that by reserving
# max(400, SOA) of audio after the final onset. 400 rather than 360 because the epoch is one
# feature frame shorter than the audio: 19 ms on whisper, 39 ms on wav2vec2.

LITERATURE_CSV = REPO / "data/metadata/literature_frequency_intensity_duration_metadata.csv"
FLOOR = 400.0

# (tone_duration_ms, isi_ms) per literature method_id, plus the novel grid's single fixed design.
# Both families run the same design; only total_duration_ms differs (whisper 30 s, wav2vec2 10 s).
NOVEL_ROW = (80.0, 500.0)          # SOA 580 -- scripts/build_novel_grid_csv.py

# Ids whose SOA >= 400 already: the floor must leave them bit-identical, or the 24 unchanged
# conditions in the re-screen stop being comparable to the committed baseline.
UNCHANGED_IDS = {9, 12, 20, 21, 27, 33, 43, 44, 55, 72, 74, 75}

# Ids the floor rebuilds, with the expected (K, leftover) per family. Only three of the ten
# (SOA, family) pairs drop a tone -- SOA 200 both families and SOA 350 whisper, where the front had
# no room to give; the rest keep every tone and just shrink the leading silence.
EXPECTED_CHANGED = {                       # id -> {family: (K, leftover_ms)}
    17: {"whisper": (148, 50.0), "wav2vec2": (48, 50.0)},      # SOA 200
    18: {"whisper": (148, 50.0), "wav2vec2": (48, 50.0)},
    19: {"whisper": (148, 50.0), "wav2vec2": (48, 50.0)},
    10: {"whisper": (99, 0.0), "wav2vec2": (32, 100.0)},       # SOA 300
    60: {"whisper": (99, 0.0), "wav2vec2": (32, 100.0)},
    37: {"whisper": (95, 200.0), "wav2vec2": (31, 40.0)},      # SOA 310
    53: {"whisper": (89, 63.0), "wav2vec2": (29, 43.0)},       # SOA 333
    28: {"whisper": (84, 300.0), "wav2vec2": (27, 250.0)},     # SOA 350
    29: {"whisper": (84, 300.0), "wav2vec2": (27, 250.0)},
    30: {"whisper": (84, 300.0), "wav2vec2": (27, 250.0)},
    31: {"whisper": (84, 300.0), "wav2vec2": (27, 250.0)},
    32: {"whisper": (84, 300.0), "wav2vec2": (27, 250.0)},
}


def _literature_designs():
    """{method_id: (tone_duration_ms, isi_ms)} for the 24 change_type=="Frequency" rows -- the set
    the generator itself filters to."""
    with open(LITERATURE_CSV) as f:
        return {int(r["method_id"]): (float(r["standard_dur"]), float(r["standard_isi"]))
                for r in csv.DictReader(f) if r["change_type"] == "Frequency"}


LITERATURE = _literature_designs()
FAMILIES = [("whisper", 30000), ("wav2vec2", 10000)]
# Literature rows x both families, plus the novel grid's row -- the full set the floor must hold on.
ALL_DESIGNS = [(mid, fam, total, tone, isi)
               for mid, (tone, isi) in sorted(LITERATURE.items())
               for fam, total in FAMILIES]
ALL_DESIGNS += [("novel", fam, total, *NOVEL_ROW) for fam, total in FAMILIES]


def _historical(total, tone, isi):
    """The committed pre-flag formula, transcribed. The default path must reproduce it exactly."""
    cycle = tone + isi
    K = int((total - isi) / cycle)
    return K, total - isi - K * cycle


def test_the_literature_sheet_still_has_the_24_frequency_rows_these_tests_pin():
    assert set(LITERATURE) == UNCHANGED_IDS | set(EXPECTED_CHANGED), (
        "the Frequency rows moved; the changed/unchanged partition below is stale")


@pytest.mark.parametrize("mid,fam,total,tone,isi", ALL_DESIGNS)
def test_default_floor_reproduces_the_historical_layout_exactly(mid, fam, total, tone, isi):
    """0.0 must be a no-op. This is what makes the flag safe to add to a script that has already
    generated ~1 TB of novel-grid audio nobody wants to re-render."""
    assert gen.compute_tone_slots(total, tone, isi) == _historical(total, tone, isi)
    assert gen.compute_tone_slots(total, tone, isi, 0.0) == _historical(total, tone, isi)


@pytest.mark.parametrize("mid,fam,total,tone,isi", ALL_DESIGNS)
def test_floor_only_moves_rows_whose_soa_is_below_it(mid, fam, total, tone, isi):
    """SOA >= 400 -- the 12 unchanged literature ids and the novel grid's 580 -- must come out
    identical in both K and leftover, since max(400, SOA) == SOA collapses the formula."""
    if tone + isi < FLOOR:
        pytest.skip(f"{mid}/{fam} has SOA {tone + isi:g} < {FLOOR:g}")
    assert gen.compute_tone_slots(total, tone, isi, FLOOR) == _historical(total, tone, isi)


@pytest.mark.parametrize("mid", sorted(EXPECTED_CHANGED))
@pytest.mark.parametrize("fam,total", FAMILIES)
def test_changed_ids_land_on_the_expected_slot_counts(mid, fam, total):
    tone, isi = LITERATURE[mid]
    K, leftover = gen.compute_tone_slots(total, tone, isi, FLOOR)
    assert (K, leftover) == pytest.approx(EXPECTED_CHANGED[mid][fam])


@pytest.mark.parametrize("mid,fam,total,tone,isi", ALL_DESIGNS)
def test_trailing_audio_is_exactly_max_floor_soa(mid, fam, total, tone, isi):
    """The property the whole change exists for, measured the way the audio is actually laid out:
    [leftover][ISI][tone][ISI] ... [tone K][ISI][pad]. Everything after the FINAL tone's onset is
    total - (leftover + isi + (K-1)*cycle). No overshoot -- the rule takes the minimum necessary."""
    cycle = tone + isi
    K, leftover = gen.compute_tone_slots(total, tone, isi, FLOOR)
    trailing = total - (leftover + isi + (K - 1) * cycle)
    assert trailing == pytest.approx(max(FLOOR, cycle))


@pytest.mark.parametrize("mid,fam,total,tone,isi", ALL_DESIGNS)
def test_floor_never_starves_the_lead_in_or_the_tone_budget(mid, fam, total, tone, isi):
    """leftover >= 0 keeps the mandatory leading ISI intact, and K >= N+2 keeps room for the
    [standard, deviant x N, standard] suffix validate_tone_slots requires at every trial level."""
    K, leftover = gen.compute_tone_slots(total, tone, isi, FLOOR)
    assert leftover >= 0, "leading silence fell below the mandatory ISI"
    for N in gen.TRIAL_LEVELS:
        assert K >= N + 2, f"K={K} cannot hold the N={N} suffix"
    gen.validate_tone_slots(K, 0, fam, gen.TRIAL_LEVELS)


def test_trailing_floor_flag_reaches_the_generator_and_is_recorded(tmp_path, metadata):
    """The floor is the one layout parameter not derivable from the metadata row, so every
    generated set records the value it was built under."""
    import pandas as pd
    out = tmp_path / "g"
    _generate(out, metadata, "--trial_levels", "7", "--num_variations", "1",
              "--models", "wav2vec2", "--trailing_floor_ms", "400")
    m = pd.read_csv(out / "audio_outputs_regular_metadata" / "metadata.csv")
    assert (m.trailing_floor_ms == 400.0).all()


def test_floor_at_or_below_soa_leaves_the_audio_byte_identical(tmp_path, metadata):
    """The novel grid's SOA is 580, so a 400 ms floor must not re-render a single sample of it.
    Proven on real waveforms, not just on (K, leftover)."""
    base, floored = tmp_path / "base", tmp_path / "floored"
    for out, extra in ((base, []), (floored, ["--trailing_floor_ms", "400"])):
        _generate(out, metadata, "--trial_levels", "7", "--num_variations", "1",
                  "--models", "wav2vec2", *extra)
    a = base / "audio_outputs_regular" / "wav2vec2"
    b = floored / "audio_outputs_regular" / "wav2vec2"
    names = sorted(p.name for p in a.glob("*.wav"))
    assert names, "nothing generated"
    for name in names:
        assert _md5(a / name) == _md5(b / name), f"{name} changed under a floor below its SOA"
