from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the checkout root for workflows that use top-level research assets.

    Walks upward from ``start`` (or the current working directory) and returns
    the first directory containing both ``pyproject.toml`` and ``src/mbs``.
    Raises ``FileNotFoundError`` if no such ancestor exists, so callers do not
    silently write to the wrong directory.
    """
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "mbs").exists():
            return path
    raise FileNotFoundError(
        "Could not locate the multimodal-brain-scaling checkout root from "
        f"{current}. Run from a cloned checkout or pass an explicit path."
    )
