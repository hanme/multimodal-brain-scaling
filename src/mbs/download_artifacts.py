#!/usr/bin/env python
"""Download Hugging Face result artifacts and restore local CSV files.

Run from a cloned checkout:

    uv run mbs-download-artifacts --artifacts-dir artifacts
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from mbs.core import find_repo_root


REPO_ID = "epfl-neuroai/multimodal-brain-scaling"

DOWNLOAD_PATTERNS = (
    "data/*.parquet",
    "metadata/*",
    "README.md",
)


@dataclass(frozen=True)
class ResultTable:
    csv_name: str
    parquet_name: str
    optional: bool = False

    @property
    def hub_path(self) -> Path:
        return Path("data") / self.parquet_name


RESULT_TABLES = (
    ResultTable(
        csv_name="pretraining_results_with_metadata.csv",
        parquet_name="pretraining_results_with_metadata.parquet",
    ),
    ResultTable(
        csv_name="layer_search_results.csv",
        parquet_name="layer_search_results.parquet",
    ),
    ResultTable(
        csv_name="finetuning_results.csv",
        parquet_name="finetuning_results.parquet",
    ),
    ResultTable(
        csv_name="mapping_results.csv",
        parquet_name="mapping_results.parquet",
    ),
    ResultTable(
        csv_name="pretraining_results.csv",
        parquet_name="pretraining_results.parquet",
        optional=True,
    ),
)


def download_dataset(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=list(DOWNLOAD_PATTERNS),
    )
    snapshot_dir = Path(local_dir)

    for table in RESULT_TABLES:
        parquet_path = snapshot_dir / table.hub_path
        if not parquet_path.exists():
            if table.optional:
                continue
            raise FileNotFoundError(f"Missing table in HF snapshot: {parquet_path}")

        csv_path = artifacts_dir / table.csv_name
        print(f"Restoring {parquet_path} -> {csv_path}")
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        df.to_csv(csv_path, index=False)

    metadata_dir = snapshot_dir / "metadata"
    if metadata_dir.exists():
        for metadata_path in metadata_dir.iterdir():
            if metadata_path.is_file():
                dst = artifacts_dir / metadata_path.name
                print(f"Restoring {metadata_path} -> {dst}")
                shutil.copy2(metadata_path, dst)

    print(f"Restored artifacts to {artifacts_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Parquet tables from epfl-neuroai/multimodal-brain-scaling "
            "and restore local CSV artifacts."
        )
    )
    parser.add_argument("--repo-id", default=REPO_ID, help="HF dataset repo id.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Local CSV artifact directory. Defaults to <repo_root>/artifacts.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF revision, branch, or tag to download.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.artifacts_dir is None:
        args.artifacts_dir = find_repo_root() / "artifacts"
    download_dataset(args)


if __name__ == "__main__":
    main()
