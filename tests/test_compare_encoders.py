"""TDD for the encoder comparison reader (D1/D2/D3 × mTRF/probe held-out r).

Defines mbs.analysis.compare_encoders.{read_parcel_heldout, to_records, format_markdown}.
Both result-h5 schemas must be readable:
  * per-dataset:  <layer>/heldout_r__<split>      (mtrf_parcel + new probe driver)
  * legacy:       <layer>/heldout_r               (original D1 group-probe run)
"""

import h5py
import numpy as np
import pytest

from mbs.analysis.compare_encoders import (
    read_parcel_heldout, to_records, format_markdown,
)

PARCELS = ["frontal", "temporal", "parietal", "occipital"]


def _write_persplit(path, layers, splits):
    with h5py.File(path, "w") as f:
        for li, layer in enumerate(layers):
            g = f.create_group(layer.replace(".", "-"))
            g.create_dataset("parcels", data=np.array(PARCELS, dtype="S"))
            g.create_dataset("parcel_nc_r", data=np.full(4, 0.5, np.float32))
            for si, split in enumerate(splits):
                r = np.array([0.1 + 0.01 * li + 0.001 * si + 0.0001 * p for p in range(4)], np.float32)
                g.create_dataset(f"heldout_r__{split}", data=r)
                g.create_dataset(f"heldout_r_nc__{split}", data=(r / 0.5).astype(np.float32))


def _write_legacy(path, layers):
    with h5py.File(path, "w") as f:
        for li, layer in enumerate(layers):
            g = f.create_group(layer.replace(".", "-"))
            g.create_dataset("parcels", data=np.array(PARCELS, dtype="S"))
            g.create_dataset("parcel_nc_r", data=np.full(4, 0.5, np.float32))
            g.create_dataset("heldout_r", data=np.full(4, 0.2 + 0.01 * li, np.float32))
            g.create_dataset("heldout_r_nc", data=np.full(4, 0.4 + 0.02 * li, np.float32))


@pytest.fixture
def d3_scores(tmp_path):
    p = tmp_path / "mtrf_parcel_scores.h5"
    _write_persplit(p, ["blocks.0", "blocks.1"], ["test_d1", "test_d2"])
    return p


@pytest.fixture
def d1_scores(tmp_path):
    p = tmp_path / "d1_scores.h5"
    _write_persplit(p, ["blocks.0", "blocks.1"], ["test"])
    return p


@pytest.fixture
def legacy_scores(tmp_path):
    p = tmp_path / "legacy.h5"
    _write_legacy(p, ["blocks.0", "blocks.1"])
    return p


def test_read_persplit_d3(d3_scores):
    data = read_parcel_heldout(d3_scores)
    assert set(data.keys()) == {"blocks.0", "blocks.1"}
    assert set(data["blocks.0"].keys()) == {"test_d1", "test_d2"}
    cell = data["blocks.0"]["test_d1"]
    assert cell["parcels"] == PARCELS
    assert cell["r"].shape == (4,)
    assert cell["r_nc"] is not None


def test_read_single_split(d1_scores):
    data = read_parcel_heldout(d1_scores)
    assert set(data["blocks.0"].keys()) == {"test"}


def test_read_legacy_maps_to_test(legacy_scores):
    data = read_parcel_heldout(legacy_scores)
    # legacy heldout_r (no split suffix) is surfaced under the canonical "test" split
    assert set(data["blocks.0"].keys()) == {"test"}
    np.testing.assert_allclose(data["blocks.0"]["test"]["r"], np.full(4, 0.2), atol=1e-5)


def test_to_records_flattens(d3_scores):
    recs = to_records("D3-mTRF", read_parcel_heldout(d3_scores))
    # 2 layers × 2 splits × 4 parcels = 16 records
    assert len(recs) == 16
    r0 = recs[0]
    assert set(r0) >= {"run", "layer", "split", "parcel", "r", "r_nc"}
    assert r0["run"] == "D3-mTRF"


def test_format_markdown_contains_values(d1_scores, d3_scores):
    recs = (to_records("D1", read_parcel_heldout(d1_scores))
            + to_records("D3", read_parcel_heldout(d3_scores)))
    md = format_markdown(recs, value="r")
    assert "frontal" in md and "blocks.0" in md
    assert "D1" in md and "D3" in md
    assert "|" in md  # it's a markdown table
