"""Regression test for the autox lead-in curvature export (four_wheel).

Guards the autox lead-in fill in ``_splice_segment``: the prescribed constant-speed
lead-in is borrowed from the (curved) tail of the closed loop, so its exported
reference curvature must reflect that geometry instead of being zeroed out. The
exporter derives ``kappa = yaw_rate / v_path``, so the fill sets
``yaw_rate = kappa * initial_speed`` over the lead-in.

Skipped where casadi is unavailable: ``_splice_segment`` lives in ``pipeline``,
whose import chain pulls in the solver, though the exporter itself does not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fast_lto.export.trajectory import export_reference_trajectory

# four_wheel reduced state / input names (see FourWheelModel).
_STATE_NAMES = [
    "d",
    "psi_err",
    "v_long",
    "v_lat",
    "yaw_rate",
    "Fx_fl",
    "Fx_fr",
    "Fx_rr",
    "Fx_rl",
    "delta",
]
_INPUT_NAMES = [
    "Fx_fl_dot_norm",
    "Fx_fr_dot_norm",
    "Fx_rr_dot_norm",
    "Fx_rl_dot_norm",
    "delta_dot_norm",
]


def _menger_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed three-point (Menger) curvature; exact for points on a circle."""
    x0, x1, x2 = x[:-2], x[1:-1], x[2:]
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]
    # Twice the signed triangle area (positive = left turn).
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    a = np.hypot(x1 - x0, y1 - y0)
    b = np.hypot(x2 - x1, y2 - y1)
    c = np.hypot(x2 - x0, y2 - y0)
    return 2.0 * cross / np.maximum(a * b * c, 1e-12)


def _circular_lead_in(k0: float, ds: float, K: int):
    """A K-point circular arc of constant curvature ``k0`` (left turn)."""
    R = 1.0 / k0
    phi = np.linspace(-K * ds, -ds, K) / R
    x = R * np.sin(phi)
    y = R * (1.0 - np.cos(phi))
    headings = phi + np.pi / 2.0
    return {
        "positions": np.column_stack((x, y)).tolist(),
        "headings": headings.tolist(),
        "curvatures": [float(k0)] * K,
        "arc_lengths": (np.arange(-K, 0) * ds).tolist(),
        "w_left": [1.75] * K,
        "w_right": [1.75] * K,
    }


def _four_wheel_solution(tail_xy, tail_heading, ds: float, v0: float, n: int):
    """A minimal solved four_wheel autox body continuing straight from the tail."""
    x0, y0 = tail_xy
    xs = x0 + np.arange(1, n + 1) * ds * np.cos(tail_heading)
    ys = y0 + np.arange(1, n + 1) * ds * np.sin(tail_heading)
    sol = {
        "mode": "autox",
        "path_xy": np.column_stack((xs, ys)).tolist(),
        "arc_lengths": (np.arange(n) * ds).tolist(),
        "w_left": [1.75] * n,
        "w_right": [1.75] * n,
        "kappa": [0.0] * n,
        "headings": [float(tail_heading)] * n,
        "state_names": _STATE_NAMES,
        "input_names": _INPUT_NAMES,
        "model_params": {},
        "run_config": {"model_name": "four_wheel", "mode": "autox"},
    }
    for name in _STATE_NAMES:
        sol[name] = [0.0] * n
    for name in _INPUT_NAMES:
        sol[name] = [0.0] * n
    sol["v_long"] = [v0] * n
    return sol


def test_leadin_kappa_matches_path_geometry(tmp_path):
    # Imported lazily so this module still collects without casadi.
    pytest.importorskip("casadi", reason="pipeline import chain requires casadi")
    from fast_lto.pipeline import _splice_segment

    k0, ds, v0, K, N = 0.12, 0.5, 3.0, 10, 4

    lead_in = _circular_lead_in(k0, ds, K)
    tail_xy = lead_in["positions"][-1]
    tail_heading = lead_in["headings"][-1]
    sol = _four_wheel_solution(tail_xy, tail_heading, ds, v0, N)

    sol = _splice_segment(sol, lead_in, v0, side="before", timed=1)

    json_path = tmp_path / "solution.json"
    csv_path = tmp_path / "reference.csv"
    json.dump(sol, json_path.open("w"))
    export_reference_trajectory(json_path, csv_path)

    import csv as _csv

    rows = list(_csv.DictReader(csv_path.open()))
    kappa = np.array([float(r["kappa"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])

    # The lead-in must not export as straight.
    assert np.all(np.abs(kappa[:K]) > 1e-6)

    # A circular arc of curvature k0, so the exported path's three-point
    # curvature must equal k0 inside the lead-in. Menger index i is point i+1.
    kappa_geo = _menger_curvature(x, y)
    interior = slice(1, K - 1)
    assert np.allclose(kappa[interior], k0, atol=1e-3)
    assert np.allclose(kappa_geo[: K - 2], k0, atol=1e-3)
    assert np.allclose(kappa[1 : K - 1], kappa_geo[: K - 2], atol=1e-3)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
