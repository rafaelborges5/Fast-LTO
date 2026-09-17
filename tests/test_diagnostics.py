"""Model diagnostics: the single source of physics the plots read from.

A plot that re-derives tyre loads or friction usage in NumPy can drift from the
model, which makes the figure you would use to catch a physics regression
capable of being wrong itself.

Two things are pinned here: the diagnostics obey physics that holds regardless
of the trajectory, and -- the important one -- the friction usage they report
agrees with the constraint the solver actually enforced.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

pytest.importorskip("casadi", reason="vehicle models require casadi")

from fast_lto.vehicle_models import DynamicBicycleModel, FourWheelModel  # noqa: E402
from fast_lto.vehicle_models.diagnostics import evaluate_diagnostics  # noqa: E402
from fast_lto.vehicle_models.point_mass import PointMassModel  # noqa: E402

N_NODES = 12


def _ramp(lo: float, hi: float) -> list:
    return list(np.linspace(lo, hi, N_NODES))


def _four_wheel_solution() -> Dict:
    """A plausible cornering trajectory. Diagnostics are pointwise, so this
    does not need to satisfy the dynamics — only to be in a sane range."""
    return {
        "d": _ramp(-0.5, 0.5),
        "psi_err": _ramp(-0.05, 0.05),
        "v_long": _ramp(8.0, 14.0),
        "v_lat": _ramp(-0.3, 0.3),
        "yaw_rate": _ramp(0.2, 1.1),
        "Fx_fl": _ramp(-200.0, 200.0),
        "Fx_fr": _ramp(-200.0, 200.0),
        "Fx_rr": _ramp(-150.0, 350.0),
        "Fx_rl": _ramp(-150.0, 350.0),
        "delta": _ramp(-0.15, 0.15),
        "Fx_fl_dot_norm": _ramp(-0.5, 0.5),
        "Fx_fr_dot_norm": _ramp(-0.5, 0.5),
        "Fx_rr_dot_norm": _ramp(-0.5, 0.5),
        "Fx_rl_dot_norm": _ramp(-0.5, 0.5),
        "delta_dot_norm": _ramp(-0.3, 0.3),
    }


def _dynamic_bicycle_solution() -> Dict:
    return {
        "d": _ramp(-0.5, 0.5),
        "psi_err": _ramp(-0.05, 0.05),
        "v": _ramp(8.0, 14.0),
        "v_lat": _ramp(-0.3, 0.3),
        "yaw_rate": _ramp(0.2, 1.1),
        "a_long": _ramp(-3.0, 3.0),
        "delta": _ramp(-0.15, 0.15),
    }


# ---------------------------------------------------------------------------
# Four-wheel
# ---------------------------------------------------------------------------


def test_four_wheel_reports_every_quantity_the_panels_draw() -> None:
    diagnostics = evaluate_diagnostics(FourWheelModel(), _four_wheel_solution())

    for wheel in ("fl", "fr", "rr", "rl"):
        for prefix in ("alpha", "Fy", "Fz", "util"):
            assert f"{prefix}_{wheel}" in diagnostics
    for name in ("F_drag", "F_roll", "Mz_Fx", "Mz_total", "a_long_body", "a_lat_body"):
        assert name in diagnostics

    for name, values in diagnostics.items():
        assert values.shape == (N_NODES,), f"{name} is not one value per node"
        assert np.all(np.isfinite(values)), f"{name} is not finite"


def test_vertical_loads_carry_exactly_the_weight_plus_downforce() -> None:
    """Load transfer moves load between wheels; it must not create any."""
    model = FourWheelModel()
    solution = _four_wheel_solution()
    diagnostics = evaluate_diagnostics(model, solution)

    p = model.params
    v = np.asarray(solution["v_long"], dtype=float)
    downforce = 0.5 * p["rho"] * p["C_l"] * p["A_f"] * v**2
    expected = p["m"] * p["g"] + downforce

    total = sum(diagnostics[f"Fz_{w}"] for w in ("fl", "fr", "rr", "rl"))
    assert total == pytest.approx(expected, rel=1e-9)


def test_friction_usage_matches_the_constraint_the_solver_enforces() -> None:
    """The strongest available check that plot and solver share one physics.

    ``util_*`` is the fraction of the friction ellipse in use. The solver's own
    constraint caps exactly that quantity, so on a real solve the peak sits at
    100% and never meaningfully above. A plot computing it differently would
    drift off that ceiling.
    """
    model = FourWheelModel()
    solution = _four_wheel_solution()
    diagnostics = evaluate_diagnostics(model, solution)

    for wheel in ("fl", "fr", "rr", "rl"):
        util = diagnostics[f"util_{wheel}"]
        Fz = diagnostics[f"Fz_{wheel}"]
        Fy = diagnostics[f"Fy_{wheel}"]
        Fx = np.asarray(solution[f"Fx_{wheel}"], dtype=float)

        cap = np.maximum(model.params[f"D_{wheel}"] * Fz, 1.0)
        expected = np.sqrt(Fx**2 + Fy**2) / cap * 100.0
        assert util == pytest.approx(expected, rel=1e-9)


def test_diagnostics_follow_the_parameters_they_were_built_with() -> None:
    """A heavier car must load its tires more — no cached or default values."""
    solution = _four_wheel_solution()
    light = evaluate_diagnostics(FourWheelModel({"m": 150.0}), solution)
    heavy = evaluate_diagnostics(FourWheelModel({"m": 250.0}), solution)

    light_total = sum(light[f"Fz_{w}"] for w in ("fl", "fr", "rr", "rl"))
    heavy_total = sum(heavy[f"Fz_{w}"] for w in ("fl", "fr", "rr", "rl"))
    assert np.all(heavy_total > light_total)


# ---------------------------------------------------------------------------
# Dynamic bicycle
# ---------------------------------------------------------------------------


def test_dynamic_bicycle_reports_axle_tire_state() -> None:
    diagnostics = evaluate_diagnostics(DynamicBicycleModel(), _dynamic_bicycle_solution())

    for name in ("alpha_f", "alpha_r", "Fy_f", "Fy_r", "a_lat_tires"):
        assert name in diagnostics
        assert diagnostics[name].shape == (N_NODES,)
        assert np.all(np.isfinite(diagnostics[name]))


def test_dynamic_bicycle_axle_loads_carry_the_weight() -> None:
    model = DynamicBicycleModel()
    diagnostics = evaluate_diagnostics(model, _dynamic_bicycle_solution())

    total = diagnostics["Fz_f"] + diagnostics["Fz_r"]
    assert total == pytest.approx(model.params["m"] * model.params["g"], rel=1e-9)


def test_lateral_acceleration_comes_from_the_tire_forces() -> None:
    model = DynamicBicycleModel()
    solution = _dynamic_bicycle_solution()
    diagnostics = evaluate_diagnostics(model, solution)

    delta = np.asarray(solution["delta"], dtype=float)
    expected = (diagnostics["Fy_f"] * np.cos(delta) + diagnostics["Fy_r"]) / model.params["m"]
    assert diagnostics["a_lat_tires"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_a_model_with_nothing_to_report_returns_nothing() -> None:
    assert evaluate_diagnostics(PointMassModel(), {"d": [0.0], "psi_err": [0.0], "v": [5.0]}) == {}


def test_a_solution_missing_a_state_is_an_error_not_a_default() -> None:
    solution = _four_wheel_solution()
    del solution["v_lat"]

    with pytest.raises(KeyError, match="v_lat"):
        evaluate_diagnostics(FourWheelModel(), solution)
