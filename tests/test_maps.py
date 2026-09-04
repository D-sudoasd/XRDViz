from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xrdviz.maps import MapData, load_map_csv, parse_map_csv


def test_map_data_validates_shape_and_round_trips_json_serializable_state():
    data = MapData(
        kind="rsm",
        x=[1.0, 2.0],
        y=[10.0, 20.0],
        intensity=[[1.0, 2.0], [3.0, 4.0]],
        counts=[[5, 6], [7, 8]],
        labels={"x": "q_parallel", "y": "q_perp", "intensity": "I"},
        units={"x": "A^-1", "y": "A^-1", "intensity": "a.u."},
        source_path="map.csv",
        metadata={"calibrated": True, "corrections": ["mask"]},
    )

    assert data.kind == "rsm"
    assert data.intensity.shape == (2, 2)
    assert data.x_label == "q_parallel"
    assert data.y_unit == "A^-1"
    assert data.source_path == "map.csv"
    assert data.metadata["calibrated"] is True
    assert data.z is data.intensity
    assert not data.x.flags.writeable
    assert not data.intensity.flags.writeable

    restored = MapData.from_dict(json.loads(json.dumps(data.to_dict())))
    assert restored.kind == data.kind
    np.testing.assert_array_equal(restored.x, data.x)
    np.testing.assert_array_equal(restored.y, data.y)
    np.testing.assert_array_equal(restored.intensity, data.intensity)
    np.testing.assert_array_equal(restored.counts, data.counts)
    assert dict(restored.labels) == dict(data.labels)
    assert dict(restored.units) == dict(data.units)
    assert dict(restored.metadata) == dict(data.metadata)


@pytest.mark.parametrize("kind", ["detector", "cake", "rsm", "pole_figure"])
def test_map_data_accepts_only_supported_kinds(kind):
    data = MapData(kind, [0, 1], [0, 1], [[1, 2], [3, 4]])
    assert data.kind == kind


def test_map_data_rejects_invalid_kind_shape_coordinates_counts_and_nonfinite_values():
    base = dict(kind="rsm", x=[0, 1], y=[0, 1], intensity=[[1, 2], [3, 4]])
    invalid = (
        {**base, "kind": "unknown"},
        {**base, "intensity": [[1, 2, 3], [4, 5, 6]]},
        {**base, "x": [0, 0]},
        {**base, "y": [0, np.nan]},
        {**base, "intensity": [[1, np.inf], [3, 4]]},
        {**base, "counts": [[1, -1], [3, 4]]},
        {**base, "counts": [[1, 2, 3], [4, 5, 6]]},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            MapData(**kwargs)


def test_parse_map_csv_builds_sorted_complete_regular_grid_and_preserves_counts():
    # Rows are intentionally not in grid order.  Matrix rows correspond to y
    # and columns to x after import.
    text = """q_parallel,q_perp,intensity,counts
1,20,4,40
0,10,1,10
1,10,2,20
0,20,3,30
"""
    data = parse_map_csv(
        text,
        kind="rsm",
        labels={"intensity": "Integrated intensity"},
        units={"x": "A^-1", "y": "A^-1"},
        source_path="scan_rsm.csv",
    )

    np.testing.assert_array_equal(data.x, [0.0, 1.0])
    np.testing.assert_array_equal(data.y, [10.0, 20.0])
    np.testing.assert_array_equal(data.intensity, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(data.counts, [[10.0, 20.0], [30.0, 40.0]])
    assert data.x_label == "q_parallel"
    assert data.y_label == "q_perp"
    assert data.intensity_label == "Integrated intensity"
    assert data.source_path == "scan_rsm.csv"


def test_parse_map_csv_recognizes_phi_chi_and_tab_delimiter():
    text = "phi\tchi\tI\n0\t-10\t1\n90\t-10\t2\n0\t10\t3\n90\t10\t4\n"
    data = parse_map_csv(text, kind="pole_figure")

    np.testing.assert_array_equal(data.x, [0.0, 90.0])
    np.testing.assert_array_equal(data.y, [-10.0, 10.0])
    np.testing.assert_array_equal(data.intensity, [[1.0, 2.0], [3.0, 4.0]])
    assert data.x_label == "phi"
    assert data.y_label == "chi"


def test_parse_map_csv_consumes_exported_kind_labels_units_and_source_metadata():
    text = (
        "kind,x,y,intensity,counts,x_label,y_label,intensity_label,x_unit,y_unit,intensity_unit,source_file\n"
        "cake,10,0,1,2,2θ,χ,Integrated,deg,deg,a.u.,cake.npz\n"
        "cake,20,0,2,3,2θ,χ,Integrated,deg,deg,a.u.,cake.npz\n"
        "cake,10,5,3,4,2θ,χ,Integrated,deg,deg,a.u.,cake.npz\n"
        "cake,20,5,4,5,2θ,χ,Integrated,deg,deg,a.u.,cake.npz\n"
    )

    data = parse_map_csv(text)

    assert data.kind == "cake"
    assert data.x_label == "2θ"
    assert data.y_label == "χ"
    assert data.intensity_label == "Integrated"
    assert data.x_unit == "deg"
    assert data.y_unit == "deg"
    assert data.intensity_unit == "a.u."
    assert data.source_path == "cake.npz"


def test_parse_map_csv_rejects_conflicting_export_metadata():
    text = (
        "kind,x,y,intensity,x_label,y_label,x_unit,y_unit,source_file\n"
        "cake,10,0,1,2θ,χ,deg,deg,cake.npz\n"
        "rsm,20,0,2,2θ,χ,deg,deg,cake.npz\n"
        "cake,10,5,3,2θ,χ,deg,deg,cake.npz\n"
        "cake,20,5,4,2θ,χ,deg,deg,cake.npz\n"
    )

    with pytest.raises(ValueError, match="kind metadata"):
        parse_map_csv(text)


def test_parse_map_csv_enforces_row_and_grid_budgets_before_allocating():
    with pytest.raises(ValueError, match="row limit"):
        parse_map_csv("x,y,intensity\n0,0,1\n1,0,2\n", max_rows=1)

    with pytest.raises(ValueError, match="grid limit"):
        parse_map_csv(
            "x,y,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n",
            max_cells=3,
        )


@pytest.mark.parametrize(
    ("text", "require_regular"),
    [
        ("qx,qz,intensity\n0,0,1\n1,0,2\n0,1,3\n", False),  # missing (1, 1)
        ("qx,qz,intensity\n0,0,1\n0,0,2\n", False),  # duplicate point
        (
            "qx,qz,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n0,3,5\n1,3,6\n",
            True,
        ),  # non-regular y when regular spacing is required
        ("qx,qz,intensity\n0,0,nan\n1,0,2\n0,1,3\n1,1,4\n", False),
        ("qx,qz,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,\n", False),
    ],
)
def test_parse_map_csv_rejects_invalid_grid(text, require_regular):
    with pytest.raises(ValueError):
        parse_map_csv(text, require_regular=require_regular)


def test_load_map_csv_uses_file_path_as_provenance(tmp_path):
    path = tmp_path / "cake.csv"
    path.write_text(
        "two_theta,chi,intensity\n10,0,1\n20,0,2\n10,5,3\n20,5,4\n", encoding="utf-8"
    )

    data = load_map_csv(path, kind="cake")

    assert data.source_path == str(path)
    assert data.x_label == "two_theta"
    assert data.y_label == "chi"


def test_load_map_csv_streams_from_file_without_read_text(tmp_path, monkeypatch):
    path = tmp_path / "map.csv"
    path.write_text(
        "x,y,intensity\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n",
        encoding="utf-8",
    )

    def reject_read_text(*_args, **_kwargs):
        raise AssertionError("read_text used")

    monkeypatch.setattr(Path, "read_text", reject_read_text)

    data = load_map_csv(path)

    assert data.shape == (2, 2)


def test_map_data_rejects_conflicting_intensity_and_z_json_fields():
    payload = {
        "kind": "rsm",
        "x": [0, 1],
        "y": [0, 1],
        "intensity": [[1, 2], [3, 4]],
        "z": [[1, 2], [3, 4]],
    }

    with pytest.raises(ValueError, match="both intensity and z"):
        MapData.from_dict(payload)


def test_detector_raw_helper_keeps_pixel_coordinates_without_physical_assumptions():
    image = np.arange(6, dtype=float).reshape(2, 3)
    data = MapData.from_detector_raw(image, source_path="detector.npy")

    assert data.kind == "detector"
    np.testing.assert_array_equal(data.x, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(data.y, [0.0, 1.0])
    np.testing.assert_array_equal(data.intensity, image)
    assert data.x_unit == "pixel"
    assert data.y_unit == "pixel"
    assert data.source_path == "detector.npy"


def test_cake_helper_maps_detector_cake_result_and_keeps_counts():
    cake = SimpleNamespace(
        two_theta=np.array([10.0, 20.0]),
        chi=np.array([-5.0, 5.0]),
        intensity=np.array([[1.0, 2.0], [3.0, 4.0]]),
        counts=np.array([[2, 3], [4, 5]]),
    )

    data = MapData.from_cake(cake, source_path="cake.npz")

    assert data.kind == "cake"
    np.testing.assert_array_equal(data.x, cake.two_theta)
    np.testing.assert_array_equal(data.y, cake.chi)
    np.testing.assert_array_equal(data.intensity, cake.intensity)
    np.testing.assert_array_equal(data.counts, cake.counts)
    assert data.x_unit == "deg"
    assert data.y_unit == "deg"
    assert data.x_label == r"$2\theta$"
    assert data.y_label == r"$\chi$"


def test_cake_helper_rejects_missing_intensity_instead_of_inventing_data():
    cake = SimpleNamespace(two_theta=[10, 20], chi=[0, 5], matrix=None)
    with pytest.raises(ValueError):
        MapData.from_cake(cake)


def test_cake_helper_preserves_empty_bins_via_counts_without_treating_nan_as_zero_data():
    cake = SimpleNamespace(
        two_theta=np.array([10.0, 20.0]),
        chi=np.array([-5.0, 5.0]),
        intensity=np.array([[np.nan, 2.0], [3.0, np.nan]]),
        counts=np.array([[0, 2], [1, 0]]),
    )

    data = MapData.from_cake(cake)

    np.testing.assert_array_equal(data.counts, cake.counts)
    np.testing.assert_array_equal(data.intensity, [[0.0, 2.0], [3.0, 0.0]])
