"""The corner-corridor geometry has two implementations. They must agree.

Whether the car fits through a corner is written twice on purpose: symbolically
in ``VehicleModel.get_corner_constraints``, because the solver differentiates
it, and numerically in ``warm_start.corner_slacks``, because a seed has to be
screened without building a CasADi graph. Two implementations of one formula
drift, and these had -- in how each guarded the Frenet Jacobian near its
singularity, not in the algebra.

Proving they compute the same geometry catches more than merging them into one
kernel would, which would only prove they run the same code.

``utils.corridor`` stays out of it: it answers a different question, the widest
feasible interval in ``d`` with no known ``d`` to evaluate at.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from fast_lto.optimization.warm_start import corner_slacks
from fast_lto.vehicle_models import DynamicBicycleModel, FourWheelModel, PointMassModel

MODELS = [PointMassModel, DynamicBicycleModel, FourWheelModel]


def _symbolic_slacks(model, d, psi_err, kappa, w_left, w_right):
    """Evaluate ``get_corner_constraints`` node by node, negated to slacks.

    The symbolic side returns ``g <= 0``; the numeric side returns room to
    spare. One is the negation of the other.
    """
    sym = [ca.MX.sym(name) for name in ("d", "psi", "kappa", "wl", "wr")]
    constraints = model.get_corner_constraints(*sym)
    func = ca.Function("g", sym, [ca.vertcat(*constraints)])

    n_corners = len(model.get_corner_offsets())
    out = np.empty((n_corners, len(d)))
    for i in range(len(d)):
        values = func(d[i], psi_err[i], kappa[i], w_left[i], w_right[i])
        out[:, i] = -np.asarray(values).reshape(-1)
    return [out[j] for j in range(n_corners)]


@pytest.fixture(scope="module")
def sample():
    """States away from the singularity, where both sides are unambiguous.

    ``|kappa * d|`` is kept well under 1 so no clamp engages: the guards
    legitimately differ (the OCP constrains ``D_kappa >= eps_D_kappa``, so the
    symbolic form needs none), and this test is about the geometry they share,
    not about behaviour past the point where the frame stops being invertible.
    """
    rng = np.random.default_rng(20260907)
    n = 200
    d = rng.uniform(-2.0, 2.0, n)
    kappa = rng.uniform(-0.15, 0.15, n)
    # Reject the few draws that approach the singularity.
    keep = np.abs(kappa * d) < 0.5
    return {
        "d": d[keep],
        "psi_err": rng.uniform(-0.6, 0.6, n)[keep],
        "kappa": kappa[keep],
        "w_left": rng.uniform(1.5, 3.0, n)[keep],
        "w_right": rng.uniform(1.5, 3.0, n)[keep],
    }


@pytest.mark.parametrize("model_cls", MODELS, ids=lambda c: c.__name__)
def test_numeric_corner_slacks_match_the_symbolic_constraints(model_cls, sample) -> None:
    model = model_cls()
    corners = model.get_corner_offsets()
    assert corners, f"{model_cls.__name__} declares no corners; this test would be vacuous"

    numeric = corner_slacks(
        sample["d"],
        sample["psi_err"],
        sample["kappa"],
        sample["w_left"],
        sample["w_right"],
        corners,
    )
    symbolic = _symbolic_slacks(
        model,
        sample["d"],
        sample["psi_err"],
        sample["kappa"],
        sample["w_left"],
        sample["w_right"],
    )

    assert len(numeric) == len(symbolic) == len(corners)
    for corner, num, sym in zip(corners, numeric, symbolic):
        np.testing.assert_allclose(
            num,
            sym,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{model_cls.__name__} corner {corner.name} disagrees with the solver's own "
            "constraint; the numeric and symbolic corner geometry have drifted",
        )


def test_the_curvature_term_costs_the_outside_of_a_corner() -> None:
    """Sanity on the sign: a left turn eats room at the front-right corner.

    A straight-line projection would ignore curvature entirely. The
    ``0.5 * kappa / D_kappa * long_proj^2`` term accounts for the centerline
    bending away from a point held ``dx`` ahead of the CoG: on a left turn the
    track curves left, so that point ends up further right of the centerline
    than ``dy`` alone suggests, and every corner's ``d`` shifts right.

    So the corner that loses room is the front one on the *outside* of the
    turn -- the car runs wide, it does not cut in. Getting this backwards is
    easy, which is why it is pinned here.
    """
    model = FourWheelModel()
    corners = model.get_corner_offsets()
    zero = np.array([0.0])
    width = np.array([2.0])

    straight = corner_slacks(zero, zero, zero, width, width, corners)
    left_turn = corner_slacks(zero, zero, zero + 0.1, width, width, corners)

    front_right = next(i for i, c in enumerate(corners) if c.dx > 0 and c.dy < 0)
    front_left = next(i for i, c in enumerate(corners) if c.dx > 0 and c.dy > 0)

    assert left_turn[front_right][0] < straight[front_right][0], "outside corner must lose room"
    assert left_turn[front_left][0] > straight[front_left][0], "inside corner gains it"


def test_the_jacobian_clamp_keeps_its_sign() -> None:
    """Past the singularity the correction must not silently change direction.

    ``D_kappa = 1 - kappa*d`` goes negative beyond the centre of the osculating
    circle. Clamping its magnitude to a positive floor regardless of sign -- the
    previous behaviour -- flipped the curvature term there, so a seed that had
    crossed the singularity was scored as though it had not.
    """
    corners = FourWheelModel().get_corner_offsets()
    kappa = np.array([0.5])
    w = np.array([2.0])

    # d slightly inside and slightly outside 1/kappa = 2.0 m.
    inside = corner_slacks(np.array([1.999]), np.array([0.0]), kappa, w, w, corners)
    outside = corner_slacks(np.array([2.001]), np.array([0.0]), kappa, w, w, corners)

    # D_kappa flips sign between the two, so the curvature correction must too;
    # with an unsigned clamp both landed on the same side.
    front_left = next(i for i, c in enumerate(corners) if c.dx > 0 and c.dy > 0)
    assert np.sign(inside[front_left][0]) != np.sign(outside[front_left][0])
