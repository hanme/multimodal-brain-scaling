import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_EXTS = {".py", ".yaml", ".yml", ".json", ".sh", ".sbatch"}
SCANNED_DIRS = ("src/mbs", "configs", "scripts")


def iter_source_py_files():
    for path in (REPO_ROOT / "src" / "mbs").rglob("*.py"):
        if path.is_file():
            yield path


def iter_text_files():
    for relative in SCANNED_DIRS:
        base = REPO_ROOT / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in CONFIG_EXTS:
                yield path


def test_no_old_src_imports():
    """No module should import from the pre-reorg ``src`` package layout.

    Uses AST so we don't false-positive on docstrings or unrelated identifiers
    (and so this file itself doesn't need to dodge its own keywords).
    """
    offenders = []
    for path in iter_source_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".", 1)[0] == "src":
                    offenders.append((path.relative_to(REPO_ROOT), node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] == "src":
                        offenders.append((path.relative_to(REPO_ROOT), node.lineno))
    assert offenders == [], offenders


def test_no_private_cluster_paths_or_usernames():
    """Scan source, configs, and scripts for leaked private paths or identifiers."""
    banned = [
        "/" + "Users/",
        "/" + "mnt/scratch",
        "/" + "work/upschrimpf",
        "/" + "scratch/izar",
        "/" + "capstor/",
        "ak" + "gokce",
        "jed" + ".epfl.ch",
        "clar" + "iden:",
    ]
    offenders = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in banned:
            if token in text:
                offenders.append((path.relative_to(REPO_ROOT), token))
    assert offenders == [], offenders
