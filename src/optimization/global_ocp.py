from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import casadi as ca
import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vehicle_models import VehicleModel, PointMassModel
    from utils.track_bounds import load_boundaries
    from optimization.integrators import SpaceIntegrator, EulerIntegrator, RK4Integrator
else:
    from vehicle_models import VehicleModel, PointMassModel
    from utils.track_bounds import load_boundaries
    from optimization.integrators import SpaceIntegrator, EulerIntegrator, RK4Integrator


def load_track_with_widths(path: Path) -> Dict:
    with Path(path).open("r") as f:
        return json.load(f)


def build_space_dynamics(
    model: VehicleModel,
) -> Tuple[Callable, Callable]:
    """
    Build the space-domain dynamics from any VehicleModel.

    Convention: state[0] = s (progress) is removed; the reduced state is
    state[1:].  Space derivatives are obtained via  dx/ds = (dx/dt) / (ds/dt).

    Returns
    -------
    f_space : callable(x_reduced, u, kappa) -> ca.MX
        Space-domain RHS:  dx_reduced / ds.
    eval_at_point : callable(x_reduced, u, kappa) -> (full_state, s_dot)
        Reconstruct the full state (with s=0) and compute s_dot = ds/dt.
    """

    def f_space(x_reduced: ca.MX, u: ca.MX, kappa: ca.MX) -> ca.MX:
        full_state = ca.vertcat(ca.MX(0), x_reduced)
        x_dot = model.get_dynamics(full_state, u, kappa)
        s_dot = x_dot[0]
        return x_dot[1:] / s_dot

    def eval_at_point(
        x_reduced: ca.MX, u: ca.MX, kappa: ca.MX
    ) -> Tuple[ca.MX, ca.MX]:
        full_state = ca.vertcat(ca.MX(0), x_reduced)
        x_dot = model.get_dynamics(full_state, u, kappa)
        return full_state, x_dot[0]  # (full_state, s_dot)

    return f_space, eval_at_point


def build_ocp(
    track: Dict,
    model: VehicleModel,
    integrator: SpaceIntegrator | None = None,
    reg_u: float = 1e-4,
):
    """
    Build a space-domain OCP over the full lap.

    Parameters
    ----------
    track : dict
        Discretized track data (positions, headings, curvatures, widths, …).
    model : VehicleModel
        Any vehicle model following the state convention [s, d, ...].
    integrator : SpaceIntegrator, optional
        Spatial integration scheme.  Defaults to ``EulerIntegrator()``.
    reg_u : float
        Input regularisation weight in the objective.
    """
    if integrator is None:
        integrator = EulerIntegrator()

    curv = np.array(track["curvatures"], dtype=np.float64)
    arc_lengths = np.array(track["arc_lengths"], dtype=np.float64)
    w_left = np.array(track["w_left"], dtype=np.float64)
    w_right = np.array(track["w_right"], dtype=np.float64)
    ds = float(track["ds_m"])
    N = len(arc_lengths)

    nx = model.nx_reduced  # reduced state (no s)
    nu = model.nu

    opti = ca.Opti()

    X = opti.variable(N, nx)
    U = opti.variable(N, nu)

    kappa_param = opti.parameter(N)
    w_left_param = opti.parameter(N)
    w_right_param = opti.parameter(N)
    x0_param = opti.parameter(nx)

    opti.set_value(kappa_param, curv)
    opti.set_value(w_left_param, w_left)
    opti.set_value(w_right_param, w_right)

    opti.subject_to(X[0, :] == x0_param.T)

    f_space, eval_at_point = build_space_dynamics(model)
    total_time = 0
    eps = 1e-3

    for i in range(N - 1):
        x_i = X[i, :].T
        u_i = U[i, :].T
        kappa_i = kappa_param[i]

        x_next = integrator.step(f_space, x_i, u_i, kappa_i, ds)
        opti.subject_to(X[i + 1, :].T == x_next)

        full_state, s_dot = eval_at_point(x_i, u_i, kappa_i)
        g_list = model.get_constraints(full_state, u_i)
        for g in g_list:
            opti.subject_to(g <= 0)

        opti.subject_to(-w_right_param[i] <= x_i[0])
        opti.subject_to(x_i[0] <= w_left_param[i])

        total_time += ds / (s_dot + eps)

    opti.subject_to(-w_right_param[N - 1] <= X[N - 1, 0])
    opti.subject_to(X[N - 1, 0] <= w_left_param[N - 1])

    opti.subject_to(X[N - 1, :].T == X[0, :].T)

    # minimise lap time + input regularisation
    obj = total_time + reg_u * ca.sumsqr(U)
    opti.minimize(obj)

    reduced_bounds = model.reduced_state_bounds()
    if reduced_bounds is not None:
        lbx, ubx = reduced_bounds
        for j in range(nx):
            opti.subject_to(opti.bounded(lbx[j], X[:, j], ubx[j]))

    input_bounds = model.input_bounds()
    if input_bounds is not None:
        lbu, ubu = input_bounds
        for j in range(nu):
            opti.subject_to(opti.bounded(lbu[j], U[:, j], ubu[j]))

    opti.solver(
        "ipopt",
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
        },
        {},
    )

    return (
        opti,
        X,
        U,
        {"kappa": kappa_param, "w_left": w_left_param, "w_right": w_right_param, "x0": x0_param},
        obj,
    )


def solve_ocp_and_save(
    track: Dict,
    model: VehicleModel,
    solution_path: Path,
    integrator: SpaceIntegrator | None = None,
    initial_speed: float = 5.0,
    reg_u: float = 1e-4,
) -> Dict:
    """
    Build and solve OCP, then save solution to JSON.

    Parameters
    ----------
    track : dict
        Track data with widths (from load_track_with_widths).
    model : VehicleModel
        Vehicle model instance.
    solution_path : Path
        Path to save solution JSON.
    integrator : SpaceIntegrator, optional
        Spatial integrator. Defaults to EulerIntegrator().
    initial_speed : float
        Initial speed guess (m/s). Default: 5.0.
    reg_u : float
        Input regularization weight. Default: 1e-4.

    Returns
    -------
    dict
        Solution dictionary (same as saved JSON).
    """
    if integrator is None:
        integrator = EulerIntegrator()

    opti, X, U, params, obj = build_ocp(track, model, integrator=integrator, reg_u=reg_u)

    reduced_names = model.reduced_state_names()
    N = len(track["arc_lengths"])
    x0 = np.zeros(model.nx_reduced)
    v_idx = reduced_names.index("v")
    x0[v_idx] = initial_speed

    opti.set_value(params["x0"], x0)
    opti.set_initial(X, 0)
    opti.set_initial(U, 0)
    opti.set_initial(X[:, v_idx], initial_speed)

    sol = opti.solve()

    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))
    obj_val = float(sol.value(obj))

    print(f"Solved full-lap OCP.  Objective (approx lap time): {obj_val:.2f} s")
    print(f"v min/max: {X_sol[:, v_idx].min():.2f} / {X_sol[:, v_idx].max():.2f} m/s")

    positions = np.array(track["positions"], dtype=np.float64)
    headings = np.array(track["headings"], dtype=np.float64)
    normals = np.column_stack((-np.sin(headings), np.cos(headings)))
    d = X_sol[:, 0]
    path_xy = positions + d[:, None] * normals

    input_names = model.get_input_names()

    sol_dict = {
        "path_xy": path_xy.tolist(),
        "obj_val": obj_val,
        "state_names": reduced_names,
        "input_names": input_names,
        "arc_lengths": track["arc_lengths"],
        "w_left": track["w_left"],
        "w_right": track["w_right"],
        "kappa": track["curvatures"],
        "headings": track["headings"],
        "model_params": model.params,
    }
    for j, name in enumerate(reduced_names):
        sol_dict[name] = X_sol[:, j].tolist()
    for j, name in enumerate(input_names):
        sol_dict[name] = U_sol[:, j].tolist()

    solution_path.parent.mkdir(parents=True, exist_ok=True)
    with solution_path.open("w") as f:
        json.dump(sol_dict, f, indent=2)
    print(f"Saved solution to {solution_path}")

    return sol_dict


def _demo() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    track_path = repo_root / "data" / "discretized" / "fsg_random_with_widths.json"
    solution_out = repo_root / "data" / "solutions" / "fsg_random_point_mass.json"
    track = load_track_with_widths(track_path)

    model = PointMassModel()
    solve_ocp_and_save(
        track=track,
        model=model,
        solution_path=solution_out,
        integrator=EulerIntegrator(),
        initial_speed=5.0,
        reg_u=1e-4,
    )


if __name__ == "__main__":
    _demo()
