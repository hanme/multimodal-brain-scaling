"""Tests for the SLURM wrappers' condition addressing and argument plumbing.

These scripts cannot be submitted locally, but the parts that decide WHICH condition a task
processes -- and whether extra flags survive at all -- are plain shell and are exactly where a
silent mistake is expensive:

  * an off-by-one or a wrapped negative index extracts the wrong method into the right directory,
    which no downstream check would catch;
  * the novel grid's 1056-entry list needs two submissions per model with TASK_OFFSET, so the
    offset arithmetic is load-bearing;
  * slurm_generate_stimuli.sh silently dropping "$@" would synthesize the full 16-clip grid
    instead of the 2-clip screen -- an 8x cost overshoot with no error.

The SBATCH directives are also asserted, since a stale walltime only shows up as a job killed
hours in.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXTRACT = REPO / "scripts/slurm_mmn_extract_batch.sh"
GENERATE = REPO / "scripts/slurm_generate_stimuli.sh"
INSILICO = REPO / "scripts/slurm_insilico_mmn_electrodes.sh"


def _select(method_list=None, task_id=0, offset=None):
    """Run just the condition-selection block of the extract wrapper."""
    block = re.search(r"^# Condition list.*?^METHOD=\"\$\{METHODS\[\$TASK_ID\]\}\"",
                      EXTRACT.read_text(), re.S | re.M)
    assert block, "condition-selection block not found -- did the script get restructured?"
    env = {"SLURM_ARRAY_TASK_ID": str(task_id), "PATH": "/usr/bin:/bin"}
    if method_list is not None:
        env["METHOD_LIST"] = str(method_list)
    if offset is not None:
        env["TASK_OFFSET"] = str(offset)
    return subprocess.run(["bash", "-c", block.group(0) + '\necho "SELECTED:$METHOD"'],
                          capture_output=True, text=True, env=env)


@pytest.fixture
def method_list(tmp_path):
    p = tmp_path / "methods.txt"
    p.write_text("\n".join(f"method_{1000 + i}{s}"
                           for i in range(1, 4) for s in ("", "_counter")) + "\n")
    return p                       # 6 entries: 1001, 1001_counter, ... 1003_counter


def _selected(r):
    assert r.returncode == 0, r.stdout + r.stderr
    return re.search(r"SELECTED:(\S+)", r.stdout).group(1)


# ──────────────────────────────────────────────────────────
# Condition addressing
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("task,expected", [
    (0, "method_1001"), (1, "method_1001_counter"), (5, "method_1003_counter")])
def test_method_list_indexes_by_array_task_id(method_list, task, expected):
    assert _selected(_select(method_list, task)) == expected


def test_task_offset_addresses_the_second_half_of_a_split_array(method_list):
    """The 1056-entry novel list exceeds a 1000-task cap, so each model is submitted twice."""
    assert _selected(_select(method_list, task_id=0, offset=3)) == "method_1002_counter"
    assert _selected(_select(method_list, task_id=2, offset=3)) == "method_1003_counter"


def test_index_past_the_end_fails_loudly(method_list):
    r = _select(method_list, task_id=6)
    assert r.returncode != 0
    assert "no method for index 6" in r.stdout + r.stderr


def test_negative_index_fails_instead_of_wrapping(method_list):
    """Bash indexes negatively from the END of an array, so an unguarded negative index would
    silently extract the last condition."""
    r = _select(method_list, task_id=-1)
    assert r.returncode != 0
    assert "method_1003_counter" not in _s(r)
    assert "no method for index -1" in _s(r)


def _s(r):
    return r.stdout + r.stderr


def test_offset_past_the_end_fails_loudly(method_list):
    r = _select(method_list, task_id=0, offset=99)
    assert r.returncode != 0
    assert "no method for index 99" in _s(r)


def test_falls_back_to_the_48_literature_conditions_without_a_method_list():
    """Literature runs pass no METHOD_LIST and must keep their previous addressing."""
    assert _selected(_select(task_id=0)) == "method_09"
    assert _selected(_select(task_id=1)) == "method_09_counter"
    assert _selected(_select(task_id=47)) == "method_75_counter"
    r = _select(task_id=48)
    assert r.returncode != 0 and "48 entries" in _s(r)


# ──────────────────────────────────────────────────────────
# Argument and override plumbing
# ──────────────────────────────────────────────────────────

def test_generate_wrapper_forwards_extra_args():
    """Without "$@" the deviant-grid flags vanish and the full 16-clip grid is synthesized."""
    text = GENERATE.read_text()
    call = text[text.index("python scripts/00aa_generate_audio_stimuli.py"):]
    assert '"$@"' in call.split("EXIT_CODE")[0]


def test_extract_wrapper_passes_the_naming_and_resume_flags():
    text = EXTRACT.read_text()
    assert '--name_by_stim_id       "${MMN_NAME_BY_STIM_ID:-false}"' in text
    assert '--overwrite             "${MMN_OVERWRITE:-false}"' in text


def test_extract_wrapper_exposes_a_features_root_override():
    """whisper-base defaults to the bare outputs/features, which a second screen must not share."""
    text = EXTRACT.read_text()
    assert 'if [ -n "${MMN_FEATURES_ROOT:-}" ]; then MMN_ROOT="$MMN_FEATURES_ROOT"' in text


@pytest.mark.parametrize("var", ["MMN_FEATURES_ROOT", "STIM_ROOT", "METADATA_CSV",
                                 "DATA_DIR", "OUT_DIR"])
def test_insilico_wrapper_exposes_every_path_override(var):
    """Missing any one of these sends a non-literature screen at the literature features, or
    overwrites the committed literature predictions."""
    assert f"${{{var}:-" in INSILICO.read_text() or f'"${var}"' in INSILICO.read_text()


def test_insilico_wrapper_passes_metadata_csv_and_forwards_extra_args():
    text = INSILICO.read_text()
    assert '--metadata_csv "$METADATA_CSV"' in text
    assert '--data_dir "$DATA_DIR"' in text
    assert text.rstrip().index('"$@"') > text.index("insilico_mmn_electrodes.py")


def test_insilico_walltime_is_sized_for_the_full_grid():
    """30 min sized the 48-condition literature screen; 1056 conditions need hours."""
    m = re.search(r"#SBATCH --time=(\d+):", INSILICO.read_text())
    assert m and int(m.group(1)) >= 12


def test_extract_walltime_and_resources_are_unchanged():
    text = EXTRACT.read_text()
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --cpus-per-task=8" in text


@pytest.mark.parametrize("script", [EXTRACT, GENERATE, INSILICO])
def test_wrappers_are_syntactically_valid(script):
    assert subprocess.run(["bash", "-n", str(script)],
                          capture_output=True).returncode == 0
