"""Tests for the skidpad track builder's start_xy override (P0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fast_lto.tracks.skidpad import _load_reference_xy, build_skidpad_track

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAP_CSV = REPO_ROOT / "data" / "tracks" / "skidpad" / "skidpad_map.csv"
REF_CSV = REPO_ROOT / "data" / "tracks" / "skidpad" / "skidpad_reference.csv"


def _require_fixtures() -> None:
    if not MAP_CSV.exists() or not REF_CSV.exists():
        pytest.skip("skidpad map/reference fixtures not present in this checkout")


def test_start_xy_none_keeps_reference_first_point() -> None:
    _require_fixtures()
    track = build_skidpad_track(map_csv=MAP_CSV, ref_csv=REF_CSV, ds_m=0.5)
    ref_xy = _load_reference_xy(REF_CSV)
    assert track["skidpad"]["P0"] == pytest.approx(ref_xy[0].tolist(), abs=1e-6)
    assert track["positions"][0] == pytest.approx(ref_xy[0].tolist(), abs=1e-6)


def test_start_xy_override_moves_p0_and_shortens_track() -> None:
    _require_fixtures()
    baseline = build_skidpad_track(map_csv=MAP_CSV, ref_csv=REF_CSV, ds_m=0.5)
    shifted = build_skidpad_track(
        map_csv=MAP_CSV, ref_csv=REF_CSV, ds_m=0.5, start_xy=(4.0, 0.0)
    )

    assert shifted["skidpad"]["P0"] == pytest.approx([4.0, 0.0], abs=1e-6)
    # Only the entry straight changes; everything downstream of the gate is
    # untouched (same circles, same exit point).
    assert shifted["skidpad"]["gate"] == pytest.approx(baseline["skidpad"]["gate"], abs=1e-6)
    assert shifted["skidpad"]["P1"] == pytest.approx(baseline["skidpad"]["P1"], abs=1e-6)

    p0_base = np.array(baseline["skidpad"]["P0"])
    gate = np.array(baseline["skidpad"]["gate"])
    expected_shrink = np.linalg.norm(gate - p0_base) - np.linalg.norm(gate - np.array([4.0, 0.0]))
    actual_shrink = baseline["total_length_m"] - shifted["total_length_m"]
    assert actual_shrink == pytest.approx(expected_shrink, abs=1e-3)


def test_start_xy_matches_manually_trimmed_reference(tmp_path: Path) -> None:
    """The config-driven override must reproduce trimming the raw CSV by hand."""
    _require_fixtures()
    ref_xy = _load_reference_xy(REF_CSV)
    # First row at/after x=4.0 (entry straight runs along y=0).
    idx = int(np.searchsorted(ref_xy[:, 0], 4.0))
    trimmed_ref = tmp_path / "trimmed_reference.csv"
    header = REF_CSV.read_text().splitlines()[0]
    rows = REF_CSV.read_text().splitlines()[1 + idx :]
    trimmed_ref.write_text("\n".join([header, *rows]) + "\n")

    via_trim = build_skidpad_track(map_csv=MAP_CSV, ref_csv=trimmed_ref, ds_m=0.5)
    via_param = build_skidpad_track(
        map_csv=MAP_CSV,
        ref_csv=REF_CSV,
        ds_m=0.5,
        start_xy=tuple(ref_xy[idx].tolist()),
    )

    assert via_param["skidpad"]["P0"] == pytest.approx(via_trim["skidpad"]["P0"], abs=1e-6)
    assert via_param["total_length_m"] == pytest.approx(via_trim["total_length_m"], abs=1e-6)


def test_start_xy_beyond_gate_raises() -> None:
    _require_fixtures()
    baseline = build_skidpad_track(map_csv=MAP_CSV, ref_csv=REF_CSV, ds_m=0.5)
    gate = baseline["skidpad"]["gate"]
    with pytest.raises(ValueError):
        build_skidpad_track(map_csv=MAP_CSV, ref_csv=REF_CSV, ds_m=0.5, start_xy=tuple(gate))
