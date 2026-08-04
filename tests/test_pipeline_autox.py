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


def test_extension_measured_from_timing_gate():
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from pipeline import _extend_track_for_autox

    ds_m = 0.5
    N = 200
    track = _closed_track(N=N, ds_m=ds_m)
    base_length_m = track["total_length_m"]

    extension_m = 50.0
    timing_offset_m = 6.0
    extended = _extend_track_for_autox(
        track, extension_m, timing_offset_m=timing_offset_m
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
    extended = _extend_track_for_autox(track, extension_m=5.0, timing_offset_m=6.0)
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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
