"""Tests for the two cluster helper scripts: staging and array submission.

Both replace loops that were previously inline in the runbook. That matters because they run on a
cluster the operator drives by copy-paste, twice (Phase 1 and Phase 2), over ~1000 directories --
the worst possible place for a hand-retyped loop.

slurm_stage_novel_stimuli.sh bridges the generator's flat output tree to the per-condition directories
the extractor and insilico_mmn read; nothing else in the repo does. submit_novel_extraction.sh
encodes the per-model differences (10 s vs 30 s window and stimulus root, per-model feature root)
that would otherwise be retyped per submission.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "scripts/slurm_slurm_stage_novel_stimuli.sh"
SUBMIT = REPO / "scripts/submit_novel_extraction.sh"

HEADER = ("source,method_id,paradigm,change_type,standard_id,standard_freq,standard_dur,"
          "standard_isi,standard_soa,standard_int,deviant_id,deviant_freq,deviant_int,"
          "deviant_dur,deviant_isi,deviant_soa,p_deviant_pc\n")


def _csv(path, ids, change_type="Frequency"):
    rows = "".join(f"novel_grid,{i},oddball,{change_type},{i},200,80,500,580,80,"
                   f"{i},225,80,80,500,580,10\n" for i in ids)
    path.write_text(HEADER + rows)
    return path


def _fake_generated_tree(root, ids, clips=("standard", "N7_var1_deviant")):
    """The flat layout 00aa_generate_audio_stimuli.py produces."""
    for cfg in ("regular", "counter"):
        for fam in ("whisper", "wav2vec2"):
            d = root / f"audio_outputs_{cfg}" / fam
            d.mkdir(parents=True, exist_ok=True)
            for i in ids:
                for c in clips:
                    (d / f"method_{i}_{c}.wav").write_bytes(b"RIFF" + bytes(64))
    return root


def _stage(tmp_path, ids, **env_over):
    src = _fake_generated_tree(tmp_path / "gen", ids)
    csv_path = _csv(tmp_path / "grid.csv", ids)
    env = dict(os.environ)
    env.update(METADATA_CSV=str(csv_path), SRC=str(src),
               WHISPER_ROOT=str(tmp_path / "whisper"),
               WAV2VEC2_ROOT=str(tmp_path / "wav2vec2"),
               METHOD_LIST_OUT=str(tmp_path / "methods.txt"))
    env.update({k: str(v) for k, v in env_over.items()})
    r = subprocess.run(["bash", str(STAGE)], capture_output=True, text=True, env=env, cwd=REPO)
    return r, env


# ──────────────────────────────────────────────────────────
# Staging
# ──────────────────────────────────────────────────────────

def test_staging_builds_both_direction_dirs_for_both_families(tmp_path):
    ids = [1001, 1002, 1003]
    r, env = _stage(tmp_path, ids)
    assert r.returncode == 0, r.stdout + r.stderr
    for root in (Path(env["WHISPER_ROOT"]), Path(env["WAV2VEC2_ROOT"])):
        dirs = sorted(p.name for p in root.iterdir())
        assert dirs == ["method_1001", "method_1001_counter", "method_1002",
                        "method_1002_counter", "method_1003", "method_1003_counter"]
        for d in root.iterdir():
            assert len(list(d.glob("*.wav"))) == 2, d


def test_staging_keeps_the_two_directions_distinct(tmp_path):
    """method_<id>/ must hold the regular tree's wavs and method_<id>_counter/ the counter
    tree's -- the direction lives in the source subdirectory, not the filename."""
    ids = [1001]
    src = _fake_generated_tree(tmp_path / "gen", ids)
    (src / "audio_outputs_regular" / "whisper" / "method_1001_standard.wav").write_bytes(b"REG")
    (src / "audio_outputs_counter" / "whisper" / "method_1001_standard.wav").write_bytes(b"CTR")
    csv_path = _csv(tmp_path / "grid.csv", ids)
    env = dict(os.environ)
    env.update(METADATA_CSV=str(csv_path), SRC=str(src),
               WHISPER_ROOT=str(tmp_path / "w"), WAV2VEC2_ROOT=str(tmp_path / "v"),
               METHOD_LIST_OUT=str(tmp_path / "m.txt"))
    assert subprocess.run(["bash", str(STAGE)], capture_output=True, text=True,
                          env=env, cwd=REPO).returncode == 0
    w = tmp_path / "w"
    assert (w / "method_1001" / "method_1001_standard.wav").read_bytes() == b"REG"
    assert (w / "method_1001_counter" / "method_1001_standard.wav").read_bytes() == b"CTR"


def test_method_list_addresses_every_staged_dir_in_order(tmp_path):
    ids = [1001, 1002]
    r, env = _stage(tmp_path, ids)
    entries = Path(env["METHOD_LIST_OUT"]).read_text().split()
    assert entries == ["method_1001", "method_1001_counter",
                       "method_1002", "method_1002_counter"]
    for root in (Path(env["WHISPER_ROOT"]), Path(env["WAV2VEC2_ROOT"])):
        for e in entries:
            assert (root / e).is_dir(), f"{e} listed but not staged"


def test_staging_is_idempotent_and_tops_up_a_grown_method(tmp_path):
    """The Phase-2 flow re-stages into directories that already hold Phase 1's 2 wavs."""
    ids = [1001]
    r, env = _stage(tmp_path, ids)
    assert r.returncode == 0
    d = Path(env["WHISPER_ROOT"]) / "method_1001"
    assert len(list(d.glob("*.wav"))) == 2

    _fake_generated_tree(Path(env["SRC"]), ids,
                         clips=("standard", "N7_var1_deviant", "N3_var1_deviant"))
    r2 = subprocess.run(["bash", str(STAGE)], capture_output=True, text=True,
                        env=env, cwd=REPO)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert len(list(d.glob("*.wav"))) == 3


def test_staging_only_takes_frequency_rows(tmp_path):
    """The same filter the generator and build_methods_from_csv apply, so the staged set and the
    condition registry cannot drift apart."""
    src = _fake_generated_tree(tmp_path / "gen", [1001, 1002])
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text(
        HEADER
        + "novel_grid,1001,oddball,Frequency,1001,200,80,500,580,80,1001,225,80,80,500,580,10\n"
        + "novel_grid,1002,oddball,Duration,1002,200,80,500,580,80,1002,225,80,80,500,580,10\n")
    env = dict(os.environ)
    env.update(METADATA_CSV=str(csv_path), SRC=str(src),
               WHISPER_ROOT=str(tmp_path / "w"), WAV2VEC2_ROOT=str(tmp_path / "v"),
               METHOD_LIST_OUT=str(tmp_path / "m.txt"))
    r = subprocess.run(["bash", str(STAGE)], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    assert Path(tmp_path / "m.txt").read_text().split() == ["method_1001", "method_1001_counter"]


def test_staging_fails_loudly_on_missing_source_wavs(tmp_path):
    """A partial generation must not quietly produce a short grid -- the in-silico driver skips
    a method whose feature dir is missing without erroring."""
    ids = [1001, 1002]
    src = _fake_generated_tree(tmp_path / "gen", [1001])          # 1002 never generated
    csv_path = _csv(tmp_path / "grid.csv", ids)
    env = dict(os.environ)
    env.update(METADATA_CSV=str(csv_path), SRC=str(src),
               WHISPER_ROOT=str(tmp_path / "w"), WAV2VEC2_ROOT=str(tmp_path / "v"),
               METHOD_LIST_OUT=str(tmp_path / "m.txt"))
    r = subprocess.run(["bash", str(STAGE)], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode != 0
    assert "MISSING" in r.stdout and "method_1002" in r.stdout


@pytest.mark.parametrize("missing", ["METADATA_CSV", "SRC"])
def test_staging_refuses_missing_inputs(tmp_path, missing):
    ids = [1001]
    _fake_generated_tree(tmp_path / "gen", ids)
    _csv(tmp_path / "grid.csv", ids)
    env = dict(os.environ)
    env.update(METADATA_CSV=str(tmp_path / "grid.csv"), SRC=str(tmp_path / "gen"),
               WHISPER_ROOT=str(tmp_path / "w"), WAV2VEC2_ROOT=str(tmp_path / "v"),
               METHOD_LIST_OUT=str(tmp_path / "m.txt"))
    env[missing] = str(tmp_path / "does_not_exist")
    r = subprocess.run(["bash", str(STAGE)], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode != 0
    assert "not found" in r.stdout + r.stderr


# ──────────────────────────────────────────────────────────
# Submission
# ──────────────────────────────────────────────────────────

def _dry_run(tmp_path, n_conditions, max_array=1001, **over):
    ml = tmp_path / "methods.txt"
    ml.write_text("\n".join(f"method_{1000 + i}" for i in range(n_conditions)) + "\n")
    for d in ("mmn_stimuli_novel", "mmn_stimuli_novel_wav2vec2"):
        (tmp_path / d).mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(DRY_RUN="1", METHOD_LIST=str(ml), MAX_ARRAY_FALLBACK=str(max_array),
               WHISPER_STIM=str(tmp_path / "mmn_stimuli_novel"),
               WAV2VEC2_STIM=str(tmp_path / "mmn_stimuli_novel_wav2vec2"))
    env.update({k: str(v) for k, v in over.items()})
    env.pop("PATH_TO_SCONTROL", None)
    r = subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_submission_covers_every_condition_exactly_once(tmp_path):
    """1056 conditions against a 1001 cap must split into two arrays whose index ranges plus
    offsets tile 0..1055 with no gap and no repeat."""
    out = _dry_run(tmp_path, 1056, max_array=1001, MODELS="whisper-tiny")
    covered = []
    for line in out.splitlines():
        if "sbatch" not in line:
            continue
        off = int(re.search(r"TASK_OFFSET=(\d+)", line).group(1))
        lo, hi = map(int, re.search(r"--array=(\d+)-(\d+)%", line).groups())
        covered += list(range(off + lo, off + hi + 1))
    assert sorted(covered) == list(range(1056))


def test_submission_uses_one_array_when_the_list_fits(tmp_path):
    out = _dry_run(tmp_path, 528, max_array=1001, MODELS="whisper-tiny")
    assert out.count("sbatch") == 1
    assert "--array=0-527%" in out


def test_wav2vec2_gets_the_10s_window_and_its_own_stimulus_root(tmp_path):
    """The single most consequential per-model difference; getting it wrong extracts
    whisper-length clips against 10 s features."""
    out = _dry_run(tmp_path, 4, MODELS="whisper-small wav2vec2-large")
    lines = {l.split("MODEL_ID=")[1].split(",")[0]: l for l in out.splitlines() if "sbatch" in l}
    assert "WINDOW_DUR=30.0" in lines["whisper-small"]
    assert "mmn_stimuli_novel," in lines["whisper-small"] or \
           "mmn_stimuli_novel " in lines["whisper-small"]
    assert "WINDOW_DUR=10.0" in lines["wav2vec2-large"]
    assert "mmn_stimuli_novel_wav2vec2" in lines["wav2vec2-large"]


def test_every_model_gets_its_own_feature_root_and_stimulus_naming(tmp_path):
    """whisper-base otherwise defaults to the bare outputs/features the literature run owns."""
    out = _dry_run(tmp_path, 4)
    for model in ("whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
                  "wav2vec2-medium", "wav2vec2-large"):
        line = next(l for l in out.splitlines() if f"MODEL_ID={model}," in l)
        assert f"MMN_FEATURES_ROOT=outputs/features/{model}-mmn-novel" in line
        assert "MMN_NAME_BY_STIM_ID=true" in line


def test_dry_run_submits_nothing(tmp_path):
    out = _dry_run(tmp_path, 4, MODELS="whisper-tiny")
    assert "DRY RUN" in out and "submitting" not in out


def test_submission_refuses_a_missing_method_list(tmp_path):
    env = dict(os.environ)
    env.update(DRY_RUN="1", METHOD_LIST=str(tmp_path / "absent.txt"))
    r = subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode != 0 and "METHOD_LIST not found" in r.stdout + r.stderr


def test_submission_refuses_a_missing_stimulus_root(tmp_path):
    ml = tmp_path / "m.txt"
    ml.write_text("method_1001\n")
    env = dict(os.environ)
    env.update(DRY_RUN="1", METHOD_LIST=str(ml), MODELS="whisper-tiny",
               WHISPER_STIM=str(tmp_path / "absent"))
    r = subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode != 0 and "stimulus root missing" in r.stdout + r.stderr


@pytest.mark.parametrize("script", [STAGE, SUBMIT])
def test_scripts_are_syntactically_valid(script):
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


# ──────────────────────────────────────────────────────────
# In-silico submission
# ──────────────────────────────────────────────────────────

INSILICO_SUBMIT = REPO / "scripts/submit_novel_insilico.sh"


def _insilico_dry(tmp_path, ids=(1001, 1002), **over):
    csv_path = _csv(tmp_path / "grid.csv", ids)
    for d in ("mmn_stimuli_novel", "mmn_stimuli_novel_wav2vec2"):
        (tmp_path / d).mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(DRY_RUN="1", METADATA_CSV=str(csv_path),
               WHISPER_STIM=str(tmp_path / "mmn_stimuli_novel"),
               WAV2VEC2_STIM=str(tmp_path / "mmn_stimuli_novel_wav2vec2"))
    env.update({k: str(v) for k, v in over.items()})
    return subprocess.run(["bash", str(INSILICO_SUBMIT)], capture_output=True,
                          text=True, env=env, cwd=REPO)


def test_insilico_submits_one_job_per_model_with_every_path_redirected(tmp_path):
    r = _insilico_dry(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith("sbatch")]
    assert len(lines) == 6
    for model in ("whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
                  "wav2vec2-medium", "wav2vec2-large"):
        line = next(l for l in lines if f"MODEL_ID={model}," in l)
        assert f"MMN_FEATURES_ROOT=outputs/features/{model}-mmn-novel" in line
        assert f"DATA_DIR=outputs/insilico_mmn_predictions_novel/{model}" in line
        assert "METADATA_CSV=" in line and "STIM_ROOT=" in line
        assert "--save_plots false" in line


def test_insilico_refuses_to_write_into_the_literature_predictions_root(tmp_path):
    """The failure this script exists to prevent: the wrapper defaults DATA_DIR to the
    literature root, and overwriting it destroys the comparison baseline."""
    for root in ("outputs/insilico_mmn_predictions",
                 "outputs/insilico_mmn_predictions/whisper-tiny"):
        r = _insilico_dry(tmp_path, PREDICTIONS_ROOT=root)
        assert r.returncode != 0, root
        assert "REFUSING" in r.stdout + r.stderr


def test_insilico_allows_the_novel_roots(tmp_path):
    for root in ("outputs/insilico_mmn_predictions_novel",
                 "outputs/insilico_mmn_predictions_novel_phase2"):
        r = _insilico_dry(tmp_path, PREDICTIONS_ROOT=root)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"DATA_DIR={root}/whisper-tiny" in r.stdout


def test_insilico_wav2vec2_gets_the_10s_stimulus_root(tmp_path):
    r = _insilico_dry(tmp_path, MODELS="whisper-small wav2vec2-large")
    lines = {l.split("MODEL_ID=")[1].split(",")[0]: l
             for l in r.stdout.splitlines() if l.startswith("sbatch")}
    assert "mmn_stimuli_novel_wav2vec2" in lines["wav2vec2-large"]
    assert "mmn_stimuli_novel_wav2vec2" not in lines["whisper-small"]


def test_insilico_can_re_run_the_winners_with_plots(tmp_path):
    r = _insilico_dry(tmp_path, MODELS="whisper-tiny", SAVE_PLOTS="true",
                      METHODS="method_1042,method_1042_counter",
                      PREDICTIONS_ROOT="outputs/insilico_mmn_predictions_novel_figs")
    assert r.returncode == 0, r.stdout + r.stderr
    line = next(l for l in r.stdout.splitlines() if l.startswith("sbatch"))
    assert "--save_plots true" in line
    assert "--methods method_1042,method_1042_counter" in line


def test_insilico_reports_the_condition_count_from_the_metadata(tmp_path):
    r = _insilico_dry(tmp_path, ids=(1001, 1002, 1003))
    assert "(6 conditions incl. counter)" in r.stdout      # 3 pairs x 2 directions


def test_insilico_refuses_a_missing_metadata_csv(tmp_path):
    r = _insilico_dry(tmp_path, METADATA_CSV=str(tmp_path / "absent.csv"))
    assert r.returncode != 0 and "not found" in r.stdout + r.stderr
