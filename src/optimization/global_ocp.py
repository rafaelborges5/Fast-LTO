from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import casadi as ca
import numpy as np

if __name__ == "__main__":
    # Running as script - add repo/src to path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vehicle_models import PointMassModel
else:
    # Imported as module
    from vehicle_models import PointMassModel


def load_track_with_widths(path: Path) -> Dict:
    with Path(path).open("r") as f:
        return json.load(f)


def build_space_dynamics(model: PointMassModel):
    """
    Return a function for x_{i+1} = x_i + ds * x'_i (space-domain Euler).
    States: [d, psi_err, v]
    Inputs: [a_long, a_lat]
    """
    def step(x_i: ca.MX, u_i: ca.MX, kappa_i: ca.MX, ds: float):
        # Rebuild full state with s dummy (not used explicitly in reduced state)
        full_state = ca.vertcat(ca.MX(0), x_i[0], x_i[1], x_i[2])
        x_dot = model.get_dynamics(full_state, u_i, kappa_i)
        # Extract derivatives of [d, psi_err, v] (indices 1,2,3 in full state)
        d_dot = x_dot[1]
        pe_dot = x_dot[2]
        v_dot = x_dot[3]
        return x_i + ds * ca.vertcat(d_dot, pe_dot, v_dot), x_dot, full_state

    return step


def build_ocp(track: Dict, model: PointMassModel, reg_u: float = 1e-3):
    """
    Build a space-domain OCP over the full lap.
    """
    positions = np.array(track["positions"], dtype=np.float64)
    headings = np.array(track["headings"], dtype=np.float64)
    curv = np.array(track["curvatures"], dtype=np.float64)
    arc_lengths = np.array(track["arc_lengths"], dtype=np.float64)
    w_left = np.array(track["w_left"], dtype=np.float64)
    w_right = np.array(track["w_right"], dtype=np.float64)
    ds = float(track["ds_m"])
    N = len(arc_lengths)

    opti = ca.Opti()

    nx = 3  # d, psi_err, v
    nu = 2  # a_long, a_lat

    X = opti.variable(N, nx)
    U = opti.variable(N, nu)

    # Parameters
    kappa_param = opti.parameter(N)
    w_left_param = opti.parameter(N)
    w_right_param = opti.parameter(N)
    x0_param = opti.parameter(nx)

    opti.set_value(kappa_param, curv)
    opti.set_value(w_left_param, w_left)
    opti.set_value(w_right_param, w_right)

    # Initial condition
    opti.subject_to(X[0, :] == x0_param.T)

    # Dynamics and constraints
    step = build_space_dynamics(model)
    total_time = 0
    eps = 1e-3

    for i in range(N - 1):
        x_i = X[i, :].T
        u_i = U[i, :].T
        kappa_i = kappa_param[i]
        x_next, x_dot_full, full_state = step(x_i, u_i, kappa_i, ds)
        opti.subject_to(X[i + 1, :].T == x_next)

        # Model constraints g <= 0
        g_list = model.get_constraints(full_state, u_i)
        for g in g_list:
            opti.subject_to(g <= 0)

        # Track width bounds
        opti.subject_to(-w_right_param[i] <= x_i[0])
        opti.subject_to(x_i[0] <= w_left_param[i])

        # Time increment
        # Recompute s_dot: x_dot_full[0] is s_dot from full state dynamics
        s_dot = x_dot_full[0]
        total_time += ds / (s_dot + eps)

    # Objective: minimize time + small input regularization
    obj = total_time + reg_u * ca.sumsqr(U)
    opti.minimize(obj)

    # Bounds from model helpers if provided
    state_bounds = model.state_bounds()
    if state_bounds is not None:
        lbx_full, ubx_full = state_bounds  # full-state bounds [s,d,psi_err,v]
        # Reduced state here is [d, psi_err, v] (indices 1,2,3)
        lbx = [lbx_full[1], lbx_full[2], lbx_full[3]]
        ubx = [ubx_full[1], ubx_full[2], ubx_full[3]]
        for j in range(nx):
            opti.subject_to(opti.bounded(lbx[j], X[:, j], ubx[j]))

    input_bounds = model.input_bounds()
    if input_bounds is not None:
        lbu, ubu = input_bounds  # inputs [a_long, a_lat]
        for j in range(nu):
            opti.subject_to(opti.bounded(lbu[j], U[:, j], ubu[j]))

    # IPOPT settings
    opti.solver(
        "ipopt",
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
        },
        {},
    )

    return opti, X, U, {"kappa": kappa_param, "w_left": w_left_param, "w_right": w_right_param, "x0": x0_param}, obj


def _demo() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    track_path = repo_root / "data" / "discretized" / "ellipse_with_widths.json"
    track = load_track_with_widths(track_path)

    model = PointMassModel()
    opti, X, U, params, obj = build_ocp(track, model)

    # Simple initialization
    N = len(track["arc_lengths"])
    x0 = np.array([0.0, 0.0, 5.0])  # d, psi_err, v
    opti.set_value(params["x0"], x0)
    opti.set_initial(X, 0)
    opti.set_initial(U, 0)
    opti.set_initial(X[:, 2], 5.0)

    sol = opti.solve()

    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))
    obj_val = float(sol.value(obj))
    print(f"Solved full-lap OCP. Objective (approx lap time): {obj_val:.2f} s")
    print(f"v min/max: {X_sol[:,2].min():.2f} / {X_sol[:,2].max():.2f} m/s")


if __name__ == "__main__":
    _demo()

