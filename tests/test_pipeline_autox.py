"""Tests for autox track extension and post-finish untimed weighting.

``pipeline`` pulls in the OCP solver (casadi) at import time, so these tests
are skipped where casadi is unavailable -- see
``tests/export/test_autox_leadin.py`` for the same pattern.
"""

from __future__ import annotations

import numpy as np
import pytest


def _closed_track(N: int = 200, ds_m: float = 0.5) -> dict:
    """A minimal closed-loop track dict, straight enough that geometry
    (positions/headings/curvatures) doesn't matter for these tests."""
    arc_lengths = (np.arange(N) * ds_m).tolist()
    return {
        "positions": [[float(i) * ds_m, 0.0] for i in range(N)],
        "headings": [0.0] * N,
        "curvatures": [0.0] * N,
        "curvatures_half": [0.0] * N,
        "arc_lengths": arc_lengths,
        "w_left": [1.5] * N,
        "w_right": [1.5] * N,
        "ds_m": ds_m,
        "num_points": N,
        "total_length_m": N * ds_m,
    }


def _circular_track(N: int = 40, radius: float = 20.0) -> dict:
    """A closed circular loop -- genuinely curved, unlike ``_closed_track`` --
    so wrap-seam continuity checks are meaningful (position spacing alone
    can't catch a heading/curvature discontinuity on a degenerate straight
    fixture)."""
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    ds_m = 2 * np.pi * radius / N
    positions = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    headings = theta + np.pi / 2.0
    kappa = 1.0 / radius
    return {
        "positions": positions.tolist(),
        "headings": headings.tolist(),
        "curvatures": [kappa] * N,
        "curvatures_half": [kappa] * N,
        "arc_lengths": (np.arange(N) * ds_m).tolist(),
        "w_left": [1.5] * N,
        "w_right": [1.5] * N,
        "ds_m": ds_m,
        "num_points": N,
        "total_length_m": N * ds_m,
    }


def test_extension_measured_from_timing_gate():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _extend_track_for_autox

    ds_m = 0.5
    N = 200
    track = _closed_track(N=N, ds_m=ds_m)
    base_length_m = track["total_length_m"]

    extension_m = 50.0
    timing_offset_m = 6.0
    # start_node_offset=0 keeps this test's arithmetic anchored at index 0
    # (positions[0] == (0, 0), the default start_x/start_y) -- production
    # default is 1; see test_resolve_start_index_* / test_extend_track_*
    # below for coverage of the anchor-resolution itself.
    extended = _extend_track_for_autox(
        track, extension_m, timing_offset_m=timing_offset_m, start_node_offset=0
    )

    arc = np.asarray(extended["arc_lengths"])
    gate2 = base_length_m + timing_offset_m

    # The horizon must reach at least extension_m past the second gate
    # crossing (allow one ds of slack from rounding to the mesh).
    assert arc[-1] >= gate2 + extension_m - ds_m
    # And not run substantially further than that (same forward point count
    # as timing_offset_m + extension_m).
    assert arc[-1] < gate2 + extension_m + ds_m

    assert extended["autox_base_length_m"] == pytest.approx(base_length_m)


def test_extension_no_longer_requires_offset_le_extension():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _extend_track_for_autox

    track = _closed_track()
    # Previously this raised ValueError; now extension_m is measured past the
    # gate, so an offset larger than the extension is no longer a conflict.
    # start_node_offset=0: see comment in test_extension_measured_from_timing_gate.
    extended = _extend_track_for_autox(
        track, extension_m=5.0, timing_offset_m=6.0, start_node_offset=0
    )
    assert extended["arc_lengths"][-1] > track["total_length_m"] + 6.0


class TestAutoxTimeWeights:
    def test_weight_one_through_timed_lap(self):
        pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
        from pipeline import _autox_time_weights

        arc = np.array([-1.0, 0.0, 3.0, 6.0, 50.0, 55.9, 56.0, 60.0, 100.0])
        base_length_m = 50.0
        timing_offset_m = 6.0
        # gate2 = 56.0
        weights = _autox_time_weights(
            arc, base_length_m, timing_offset_m, eps_time=0.1, decel_hold_m=0.0
        )
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        assert np.allclose(weights, expected)

    def test_decel_hold_extends_full_weight_past_gate(self):
        pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
        from pipeline import _autox_time_weights

        arc = np.array([50.0, 56.0, 58.0, 59.9, 60.0, 65.0])
        base_length_m = 50.0
        timing_offset_m = 6.0  # gate2 = 56.0
        weights = _autox_time_weights(
            arc, base_length_m, timing_offset_m, eps_time=0.1, decel_hold_m=4.0
        )
        # Held at 1.0 through [56, 60), eps_time from 60 onward.
        expected = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
        assert np.allclose(weights, expected)


def test_resolve_start_index_nearest_plus_offset():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _resolve_autox_start_index

    positions = np.array([[float(i), float(i)] for i in range(10)])
    x, y = positions[4]

    idx0, snap0 = _resolve_autox_start_index(positions, x, y, node_offset=0)
    idx1, snap1 = _resolve_autox_start_index(positions, x, y, node_offset=1)

    assert idx0 == 4
    assert idx1 == 5
    # Snap distance is to the *nearest* sample, before node_offset is applied.
    assert snap0 == pytest.approx(0.0)
    assert snap1 == pytest.approx(0.0)


def test_resolve_start_index_wraps_at_array_boundary():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _resolve_autox_start_index

    positions = np.array([[float(i), float(i)] for i in range(10)])
    x, y = positions[-1]

    idx1, _ = _resolve_autox_start_index(positions, x, y, node_offset=1)
    idx2, _ = _resolve_autox_start_index(positions, x, y, node_offset=2)

    assert idx1 == 0
    assert idx2 == 1


def test_extend_track_pins_launch_at_resolved_anchor():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _extend_track_for_autox

    track = _closed_track(N=200, ds_m=0.5)
    positions = np.asarray(track["positions"])
    k = 50
    start_x, start_y = positions[k]

    extended = _extend_track_for_autox(
        track,
        extension_m=10.0,
        timing_offset_m=6.0,
        ocp_lead_m=0.0,
        start_x=start_x,
        start_y=start_y,
        start_node_offset=0,
    )

    # The pinned launch node (first node of the horizon, since ocp_lead_m=0)
    # is the physically-requested point, not the CSV's array index 0.
    assert np.allclose(extended["positions"][0], positions[k])


def test_extend_track_invariant_under_rotation():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _extend_track_for_autox

    track = _closed_track(N=200, ds_m=0.5)
    positions = np.asarray(track["positions"])

    extended_0 = _extend_track_for_autox(
        track,
        extension_m=10.0,
        timing_offset_m=6.0,
        start_x=positions[0][0],
        start_y=positions[0][1],
        start_node_offset=0,
    )
    extended_7 = _extend_track_for_autox(
        track,
        extension_m=10.0,
        timing_offset_m=6.0,
        start_x=positions[7][0],
        start_y=positions[7][1],
        start_node_offset=0,
    )

    # Loop circumference and point count are anchor-invariant...
    assert extended_0["autox_base_length_m"] == pytest.approx(extended_7["autox_base_length_m"])
    assert extended_0["num_points"] == extended_7["num_points"]
    # ...but the actual geometry differs, since the anchor moved.
    assert not np.allclose(extended_0["positions"][0], extended_7["positions"][0])


def test_extend_track_no_discontinuity_at_wrap_seam():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _extend_track_for_autox

    track = _circular_track(N=40, radius=20.0)
    ds_m = track["ds_m"]
    positions = np.asarray(track["positions"])
    k = 13
    start_x, start_y = positions[k]

    extended = _extend_track_for_autox(
        track,
        extension_m=10.0,
        lead_in_m=5.0,
        ocp_lead_m=3.0,
        timing_offset_m=6.0,
        start_x=start_x,
        start_y=start_y,
        start_node_offset=0,
    )

    # No jump in consecutive-point spacing anywhere in the OCP horizon,
    # including the backward-run-in<->core and core<->run-off seams that the
    # rotation introduces. Points are sampled directly on the circle, so
    # consecutive *chord* length is uniformly slightly under ds_m (the arc
    # length) by construction -- compare spacings against each other, not
    # against ds_m, so this only fails on a genuine seam discontinuity.
    ext_pos = np.asarray(extended["positions"])
    spacing = np.linalg.norm(np.diff(ext_pos, axis=0), axis=1)
    assert np.allclose(spacing, spacing[0], atol=1e-9)
    assert spacing[0] < ds_m  # sanity: chord < arc, as expected for a circle

    # Same for the prescribed lead-in, and its junction into the horizon.
    lead_in_pos = np.asarray(extended["autox_lead_in"]["positions"])
    lead_in_spacing = np.linalg.norm(np.diff(lead_in_pos, axis=0), axis=1)
    assert np.allclose(lead_in_spacing, spacing[0], atol=1e-9)
    junction = np.linalg.norm(lead_in_pos[-1] - ext_pos[0])
    assert junction == pytest.approx(spacing[0], abs=1e-9)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
