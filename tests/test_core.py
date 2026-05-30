from pathlib import Path

import pytest

from mbs.core import deep_update, get_md5_hash, load_yaml, str2bool


def test_str2bool_accepts_common_values():
    assert str2bool("true") is True
    assert str2bool("0") is False


def test_str2bool_rejects_unknown_value():
    with pytest.raises(Exception):
        str2bool("maybe")


def test_deep_update_preserves_nested_defaults():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    update = {"a": {"b": 4}}
    assert deep_update(base, update) == {"a": {"b": 4, "c": 2}, "d": 3}


def test_load_yaml_base_config(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("a:\n  b: 1\n  c: 2\n", encoding="utf-8")
    child.write_text("base_config: base.yaml\na:\n  b: 4\n", encoding="utf-8")
    assert load_yaml(child) == {"base_config": "base.yaml", "a": {"b": 4, "c": 2}}


def test_get_md5_hash_returns_string():
    assert isinstance(get_md5_hash({"a": 1}), str)
