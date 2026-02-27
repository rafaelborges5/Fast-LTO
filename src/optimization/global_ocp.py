from __future__ import annotations

import json
import time
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


def build_space_dynamics_normalized(
    model: VehicleModel,
) -> Tuple[Callable, Callable]:
    """
    Build the space-domain dynamics in the NORMALISED reduced state.

    Returns
    -------
    f_space_norm : callable(x_reduced_norm, u_norm, kappa) -> ca.MX
        Space-domain RHS in normalised coordinates: dx_reduced_norm / ds.
    eval_at_point_norm : callable(x_reduced_norm, u_norm, kappa)
        Returns (full_state_phys, s_dot) for the given normalised state.
    """

    def f_space_norm(x_reduced_norm: ca.MX, u_norm: ca.MX, kappa: ca.MX) -> ca.MX:
        return model.get_dynamics_normalized(x_reduced_norm, u_norm, kappa)

    def eval_at_point_norm(
        x_reduced_norm: ca.MX, u_norm: ca.MX, kappa: ca.MX
    ) -> Tuple[ca.MX, ca.MX]:
        x_red_phys = model.reduced_state_norm_to_phys(x_reduced_norm)
        u_phys = model.input_norm_to_phys(u_norm)
        full_state = ca.vertcat(ca.MX(0), x_red_phys)
        x_dot = model.get_dynamics(full_state, u_phys, kappa)
        return full_state, x_dot[0]  # (full_state_phys, s_dot)

    return f_space_norm, eval_at_point_norm


def build_ocp(
    track: Dict,
    model: VehicleModel,
    integrator: SpaceIntegrator | None = None,
    reg_u: float = 1e-4,
    use_normalization: bool = True,
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
    curv_half = np.array(track["curvatures_half"], dtype=np.float64)
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
    kappa_half_param = opti.parameter(N)
    w_left_param = opti.parameter(N)
    w_right_param = opti.parameter(N)
    x0_param = opti.parameter(nx)

    opti.set_value(kappa_param, curv)
    opti.set_value(kappa_half_param, curv_half)
    opti.set_value(w_left_param, w_left)
    opti.set_value(w_right_param, w_right)

    opti.subject_to(X[0, :] == x0_param.T)

    if use_normalization:
        f_space, eval_at_point = build_space_dynamics_normalized(model)
    else:
        f_space, eval_at_point = build_space_dynamics(model)
    total_time = 0

    for i in range(N - 1):
        x_i = X[i, :].T
        u_i = U[i, :].T
        kappa_i = kappa_param[i]
        kappa_half_i = kappa_half_param[i]
        kappa_next_i = kappa_param[i + 1]

        x_next = integrator.step(
            f_space,
            x_i,
            u_i,
            kappa_i,
            ds,
            kappa_half=kappa_half_i,
            kappa_next=kappa_next_i,
        )
        opti.subject_to(X[i + 1, :].T == x_next)

        if use_normalization:
            g_list = model.get_constraints_normalized(x_i, u_i)
        else:
            full_state, _ = eval_at_point(x_i, u_i, kappa_i)
            g_list = model.get_constraints(full_state, u_i)
        for g in g_list:
            opti.subject_to(g <= 0)

        if use_normalization:
            # d_norm = (d_phys - shift) / scale
            x_scale, x_shift = model.get_reduced_state_scaling()
            if x_scale is None or x_shift is None:
                raise RuntimeError("Normalization scales not defined.")
            
            d_scale = x_scale[0]
            d_shift = x_shift[0]
            
            ub_d_norm = (w_left_param[i] - d_shift) / d_scale
            lb_d_norm = (-w_right_param[i] - d_shift) / d_scale
            
            opti.subject_to(lb_d_norm <= x_i[0])
            opti.subject_to(x_i[0] <= ub_d_norm)
        else:
            opti.subject_to(-w_right_param[i] <= x_i[0])
            opti.subject_to(x_i[0] <= w_left_param[i])

        total_time += integrator.time_step(
            f_space, eval_at_point, x_i, u_i, kappa_i, ds,
            kappa_half=kappa_half_i, kappa_next=kappa_next_i,
        )

    if use_normalization:
        x_scale, x_shift = model.get_reduced_state_scaling()
        if x_scale is None or x_shift is None:
            raise RuntimeError("Normalization scales not defined.")
        d_scale = x_scale[0]
        d_shift = x_shift[0]
        
        ub_d_norm_last = (w_left_param[N - 1] - d_shift) / d_scale
        lb_d_norm_last = (-w_right_param[N - 1] - d_shift) / d_scale
        
        opti.subject_to(lb_d_norm_last <= X[N - 1, 0])
        opti.subject_to(X[N - 1, 0] <= ub_d_norm_last)
    else:
        opti.subject_to(-w_right_param[N - 1] <= X[N - 1, 0])
        opti.subject_to(X[N - 1, 0] <= w_left_param[N - 1])

    opti.subject_to(X[N - 1, :].T == X[0, :].T)

    # minimise lap time + input regularisation (reg_u normalised by N)
    obj = total_time + (reg_u / N) * ca.sumsqr(U)
    opti.minimize(obj)

    if use_normalization:
        reduced_bounds = model.reduced_state_bounds_normalized()
    else:
        reduced_bounds = model.reduced_state_bounds()
    if reduced_bounds is not None:
        lbx, ubx = reduced_bounds
        for j in range(nx):
            opti.subject_to(opti.bounded(lbx[j], X[:, j], ubx[j]))

    if use_normalization:
        input_bounds = model.input_bounds_normalized()
    else:
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
            "ipopt.nlp_scaling_method": "none",  # IPOPT internal scaling deactivated
        },
        {},
    )

    return (
        opti,
        X,
        U,
        {
            "kappa": kappa_param,
            "kappa_half": kappa_half_param,
            "w_left": w_left_param,
            "w_right": w_right_param,
            "x0": x0_param,
        },
        obj,
        total_time,
    )


def solve_ocp_and_save(
    track: Dict,
    model: VehicleModel,
    solution_path: Path,
    integrator: SpaceIntegrator | None = None,
    initial_speed: float = 5.0,
    reg_u: float = 1e-4,
    run_config: Dict | None = None,
    use_normalization: bool = True,
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

    opti, X, U, params, obj, total_time_expr = build_ocp(
        track,
        model,
        integrator=integrator,
        reg_u=reg_u,
        use_normalization=use_normalization,
    )

    reduced_names = model.reduced_state_names()
    N = len(track["arc_lengths"])
    x0_phys = np.zeros(model.nx_reduced)
    v_idx = reduced_names.index("v")
    x0_phys[v_idx] = initial_speed

    if use_normalization:
        x0_norm = model.reduced_state_phys_to_norm(x0_phys)
        opti.set_value(params["x0"], x0_norm)
        v_norm = model.reduced_state_phys_to_norm(x0_phys)[v_idx]
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        opti.set_initial(X[:, v_idx], v_norm)
    else:
        opti.set_value(params["x0"], x0_phys)
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        opti.set_initial(X[:, v_idx], initial_speed)

    start_time = time.perf_counter()
    sol = opti.solve()
    solve_time_s = time.perf_counter() - start_time

    stats = opti.stats()
    iter_count = stats.get("iter_count")
    return_status = stats.get("return_status")

    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))
    obj_val = float(sol.value(obj))
    lap_time_s = float(sol.value(total_time_expr))
    reg_term = obj_val - lap_time_s

    N = len(track["arc_lengths"])
    ds_m = (
        float(track.get("ds_m", track["arc_lengths"][1] - track["arc_lengths"][0]))
        if N > 1
        else float(track.get("ds_m", 0.0))
    )
    time_per_point_ms = solve_time_s / N * 1e3 if N > 0 else None
    time_per_iter_ms = solve_time_s / iter_count * 1e3 if iter_count not in (None, 0) else None

    print(f"Solved full-lap OCP.  Objective value: {obj_val:.2f}")
    print(f"  Lap-time term: {lap_time_s:.2f} s")
    print(f"  Regularisation term: {reg_term:.3f}")
    print(
        f"Solve stats: N={N}, ds={ds_m:.3f} m, "
        f"time={solve_time_s:.3f} s, "
        f"iters={iter_count if iter_count is not None else 'N/A'}, "
        f"status={return_status}"
    )
    if use_normalization:
        # Map solver variables back to physical reduced states/inputs.
        x_scale, x_shift = model.get_reduced_state_scaling()
        u_scale, u_shift = model.get_input_scaling()
        if x_scale is None or x_shift is None or u_scale is None or u_shift is None:
            raise RuntimeError(
                "Normalisation scales/shifts are not defined for solution "
                "post-processing."
            )
        # x_phys = x_norm * scale + shift  (broadcast over samples)
        x_scale_np = np.asarray(x_scale).astype(float).reshape(1, -1)
        x_shift_np = np.asarray(x_shift).astype(float).reshape(1, -1)
        u_scale_np = np.asarray(u_scale).astype(float).reshape(1, -1)
        u_shift_np = np.asarray(u_shift).astype(float).reshape(1, -1)

        X_phys = X_sol * x_scale_np + x_shift_np
        U_phys = U_sol * u_scale_np + u_shift_np
    else:
        X_phys = X_sol
        U_phys = U_sol

    print(
        f"v min/max: {X_phys[:, v_idx].min():.2f} / "
        f"{X_phys[:, v_idx].max():.2f} m/s"
    )

    positions = np.array(track["positions"], dtype=np.float64)
    headings = np.array(track["headings"], dtype=np.float64)
    normals = np.column_stack((-np.sin(headings), np.cos(headings)))
    d = X_phys[:, 0]
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
        "profiling": {
            "N": N,
            "ds_m": ds_m,
            "solve_time_s": solve_time_s,
            "iter_count": iter_count,
            "return_status": return_status,
            "time_per_point_ms": time_per_point_ms,
            "time_per_iter_ms": time_per_iter_ms,
            "lap_time_s": lap_time_s,
            "reg_term": reg_term,
            "reg_term_relative": reg_term / obj_val if obj_val != 0.0 else None,
        },
        "run_config": run_config,
    }
    for j, name in enumerate(reduced_names):
        sol_dict[name] = X_phys[:, j].tolist()
    for j, name in enumerate(input_names):
        sol_dict[name] = U_phys[:, j].tolist()

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
