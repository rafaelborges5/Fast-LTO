from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Sequence, Tuple

import casadi as ca
import numpy as np

from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator, SpaceIntegrator
from fast_lto.utils.smooth import smoothmax
from fast_lto.vehicle_models import VehicleModel

# Terminal state tolerances (physical units). Slightly loose so dynamics stay feasible.
#: Events ``build_ocp`` supports. Trackdrive leaves node 0 free (closed-loop).
OCP_MODES = ("trackdrive", "autox", "skidpad")

_TERMINAL_D_TOL_M = 0.05
_TERMINAL_PSI_TOL_RAD = 0.03
_TERMINAL_YAW_RATE_TOL = 0.15
_TERMINAL_V_LAT_TOL = 0.15


def _terminal_state_bounds(
    model: VehicleModel, use_normalization: bool
) -> list[tuple[int, float, float]]:
    """``(state_idx, lo, hi)`` terminal tolerances in solver units."""
    reduced_names = model.reduced_state_names()
    if use_normalization:
        x_scale, x_shift = model.get_reduced_state_scaling()

    def _phys_to_norm(idx: int, tol: float) -> tuple[float, float]:
        if not use_normalization:
            return -tol, tol
        s = float(np.array(x_scale).reshape(-1)[idx])
        sh = float(np.array(x_shift).reshape(-1)[idx])
        return (-tol - sh) / s, (tol - sh) / s

    state_tols = [(0, _TERMINAL_D_TOL_M), (1, _TERMINAL_PSI_TOL_RAD)]
    for name, tol in (("yaw_rate", _TERMINAL_YAW_RATE_TOL), ("v_lat", _TERMINAL_V_LAT_TOL)):
        if name in reduced_names:
            state_tols.append((reduced_names.index(name), tol))
    return [(idx, *_phys_to_norm(idx, tol)) for idx, tol in state_tols]


def _apply_terminal_window(
    opti: ca.Opti,
    X: ca.MX,
    N: int,
    n_window: int,
    bounds: list[tuple[int, float, float]],
) -> None:
    """Apply ``bounds`` over the last ``n_window`` nodes (d, heading, yaw_rate, v_lat)."""
    n_window = min(N, max(1, int(n_window)))
    for i in range(N - n_window, N):
        for idx, lo, hi in bounds:
            opti.subject_to(lo <= X[i, idx])
            opti.subject_to(X[i, idx] <= hi)


def _autox_gate_lap_time(
    track: Dict,
    time_at_node_s: np.ndarray,
    timing_offset_m: float,
) -> Tuple[float | None, str | None]:
    """Elapsed time between the two crossings of the autox timing gate.

    Gate is ``timing_offset_m`` past the start. Returns ``(lap_time_s, warning)``.
    """
    base_length_m = track.get("autox_base_length_m")
    if base_length_m is None:
        return None, "track has no 'autox_base_length_m' (not an autox-extended track)"

    arc_arr = np.asarray(track["arc_lengths"], dtype=np.float64)
    s_start = timing_offset_m
    s_end = float(base_length_m) + timing_offset_m
    if s_end > arc_arr[-1] + 1e-6 or s_start < arc_arr[0] - 1e-6:
        return None, (
            f"autox_extension_m too short to reach the timing gate "
            f"(need s={s_end:.1f} m, horizon ends at {arc_arr[-1]:.1f} m)"
        )

    t_start = float(np.interp(s_start, arc_arr, time_at_node_s))
    t_end = float(np.interp(s_end, arc_arr, time_at_node_s))
    return t_end - t_start, None


def _safe_debug_value(opti: ca.Opti, expr: ca.MX) -> np.ndarray | float | None:
    """Return opti.debug.value(expr) as plain Python/numpy, or None on failure."""
    try:
        val = opti.debug.value(expr)
    except Exception:
        return None
    arr = np.array(val)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


class BuiltOcp(NamedTuple):
    """Everything ``build_ocp`` hands back."""

    opti: ca.Opti
    X: ca.MX
    U: ca.MX
    params: Dict[str, ca.MX]
    objective: ca.MX
    total_time: ca.MX
    objective_time: ca.MX
    pure_timed_time: ca.MX
    cumulative_time: ca.MX


def _per_input_weights(
    weight: float | Sequence[float] | np.ndarray | None,
    nu: int,
    name: str,
    *,
    default: float,
) -> np.ndarray:
    """Broadcast a regularisation weight to one value per input.

    Accepts a scalar, a sequence of length ``nu``, or None for the default.
    """
    if weight is None:
        return np.full(nu, float(default))

    arr = np.atleast_1d(np.asarray(weight, dtype=float)).reshape(-1)
    if arr.size == 1:
        return np.full(nu, float(arr[0]))
    if arr.size != nu:
        raise ValueError(f"{name} must be a scalar or have length {nu}, got {arr.size}")
    return arr


class _BrakeZone(NamedTuple):
    """An alternative model, and the nodes it applies to."""

    model: VehicleModel
    f_space: Callable
    eval_at_point: Callable
    mask: np.ndarray


def _build_brake_zone(
    model: VehicleModel,
    track: Dict,
    N: int,
    D_safe_braking: float,
    use_normalization: bool,
) -> _BrakeZone:
    """Copy of ``model`` with tyre ``D`` overridden on untimed nodes (``timed_mask``)."""
    timed_mask_raw = track.get("timed_mask")
    if timed_mask_raw is None:
        raise ValueError(
            "D_safe_braking requires track['timed_mask'] to know which nodes "
            "have no timing objective."
        )
    timed_arr = np.asarray(timed_mask_raw, dtype=float)
    if timed_arr.size != N:
        raise ValueError(f"track['timed_mask'] has {timed_arr.size} entries, expected {N}")

    d_keys = ("D_fl", "D_fr", "D_rr", "D_rl")
    missing = [k for k in d_keys if k not in model.params]
    if missing:
        raise ValueError(
            f"D_safe_braking requires per-wheel D params {d_keys}, but "
            f"{type(model).__name__} is missing {missing}."
        )

    brake_params = dict(model.params)
    for k in d_keys:
        brake_params[k] = float(D_safe_braking)
    brake_model = type(model)(brake_params)

    builder = build_space_dynamics_normalized if use_normalization else build_space_dynamics
    f_space_brake, eval_at_point_brake = builder(brake_model)
    return _BrakeZone(brake_model, f_space_brake, eval_at_point_brake, timed_arr < 0.5)


def _apply_final_node_constraints(
    opti: ca.Opti,
    X: ca.MX,
    U: ca.MX,
    N: int,
    kappa_last: ca.MX,
    model_last: VehicleModel,
    eval_at_point_last: Callable,
    use_normalization: bool,
) -> None:
    """Apply model constraints at the last node (node loop covers ``i < N-1`` only)."""
    x_last = X[N - 1, :].T
    u_last = U[N - 1, :].T
    if use_normalization:
        g_list = model_last.get_constraints_normalized(x_last, u_last, kappa_last)
    else:
        full_state_last, _ = eval_at_point_last(x_last, u_last, kappa_last)
        g_list = model_last.get_constraints(full_state_last, u_last, kappa_last)
    for g in g_list:
        opti.subject_to(g <= 0)


def _apply_terminal_speed(
    opti: ca.Opti,
    X: ca.MX,
    N: int,
    model: VehicleModel,
    terminal_speed: float,
    use_normalization: bool,
) -> None:
    """Inequality ``v <= terminal_speed`` at the final node."""
    reduced_names = model.reduced_state_names()
    v_idx = reduced_names.index("v") if "v" in reduced_names else reduced_names.index("v_long")
    if use_normalization:
        x_scale, x_shift = model.get_reduced_state_scaling()
        v_scale = float(np.array(x_scale).reshape(-1)[v_idx])
        v_shift = float(np.array(x_shift).reshape(-1)[v_idx])
        opti.subject_to(X[N - 1, v_idx] <= (terminal_speed - v_shift) / v_scale)
    else:
        opti.subject_to(X[N - 1, v_idx] <= terminal_speed)


def load_track_with_widths(path: Path) -> Dict:
    with Path(path).open("r") as f:
        return json.load(f)


def _s_dot_guard_params(model: VehicleModel) -> Tuple[float, float]:
    return (
        float(model.params.get("eps_s_dot", 1e-3)),
        float(model.params.get("smoothmax_eps", 1e-3)),
    )


def _safe_s_dot(model: VehicleModel, s_dot: ca.MX) -> ca.MX:
    floor, smooth_eps = _s_dot_guard_params(model)
    return smoothmax(s_dot, ca.MX(floor), smooth_eps)


def _enforce_rk4_mesh_limit(integrator: SpaceIntegrator, model: VehicleModel, ds: float) -> None:
    max_ds = model.params.get("rk4_max_ds_m")
    if isinstance(integrator, RK4Integrator) and max_ds is not None and ds > float(max_ds):
        raise ValueError(
            f"RK4 mesh too coarse for {type(model).__name__}: ds={ds:.3f} m, "
            f"limit={float(max_ds):.3f} m. Refine ds or use Euler."
        )


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
        return x_dot[1:] / _safe_s_dot(model, s_dot)

    def eval_at_point(x_reduced: ca.MX, u: ca.MX, kappa: ca.MX) -> Tuple[ca.MX, ca.MX]:
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


def _constraint_eval_points(
    integrator: SpaceIntegrator,
    f_space: Callable,
    x: ca.MX,
    u: ca.MX,
    kappa: ca.MX,
    ds: float,
    kappa_half: ca.MX | None = None,
    kappa_next: ca.MX | None = None,
) -> list[tuple[ca.MX, ca.MX]]:
    points = [(x, kappa)]
    if isinstance(integrator, RK4Integrator):
        kh = kappa_half if kappa_half is not None else kappa
        kn = kappa_next if kappa_next is not None else kappa
        k1 = f_space(x, u, kappa)
        x2 = x + ds / 2 * k1
        k2 = f_space(x2, u, kh)
        x3 = x + ds / 2 * k2
        k3 = f_space(x3, u, kh)
        x4 = x + ds * k3
        points.extend(((x2, kh), (x3, kh), (x4, kn)))
    return points


def build_ocp(
    track: Dict,
    model: VehicleModel,
    integrator: SpaceIntegrator | None = None,
    reg_du: float | Sequence[float] | np.ndarray | None = None,
    reg_u_l2: float | Sequence[float] | np.ndarray | None = None,
    use_normalization: bool = True,
    solver_verbose: bool = False,
    boundary_margin: float = 0.0,
    mode: str = "trackdrive",
    time_weights: np.ndarray | None = None,
    terminal_speed: float | None = None,
    enforce_terminal_constraints: bool = True,
    terminal_straight_m: float | None = None,
    terminal_state_constraint: bool = False,
    terminal_window_nodes: int = 2,
    D_safe_braking: float | None = None,
) -> BuiltOcp:
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
    reg_du : float or array-like, optional
        Input rate-regularisation weight(s) in the objective. If a scalar is
        given then make isotropic matrix.
    """
    if mode not in OCP_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Must be one of {sorted(OCP_MODES)}.")
    if integrator is None:
        integrator = EulerIntegrator()

    curv = np.array(track["curvatures"], dtype=np.float64)
    curv_half = np.array(track["curvatures_half"], dtype=np.float64)
    arc_lengths = np.array(track["arc_lengths"], dtype=np.float64)
    w_left = np.array(track["w_left"], dtype=np.float64) - boundary_margin
    w_right = np.array(track["w_right"], dtype=np.float64) - boundary_margin
    ds = float(track["ds_m"])
    N = len(arc_lengths)
    _enforce_rk4_mesh_limit(integrator, model, ds)

    nx = model.nx_reduced  # reduced state (no s)
    nu = model.nu

    reg_du_arr = _per_input_weights(reg_du, nu, "reg_du", default=1e-4)

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

    if mode != "trackdrive":
        opti.subject_to(X[0, :] == x0_param.T)

    if time_weights is not None:
        time_weights = np.asarray(time_weights, dtype=float).reshape(-1)
        if time_weights.size not in (N - 1, N):
            raise ValueError(
                f"time_weights must have length {N - 1} or {N}, got {time_weights.size}"
            )
        time_weights = time_weights[: N - 1]

    if use_normalization:
        f_space, eval_at_point = build_space_dynamics_normalized(model)
    else:
        f_space, eval_at_point = build_space_dynamics(model)
    s_dot_floor, s_dot_smooth_eps = _s_dot_guard_params(model)

    brake_zone: _BrakeZone | None = None
    if mode == "autox" and D_safe_braking is not None:
        brake_zone = _build_brake_zone(model, track, N, D_safe_braking, use_normalization)

    def _node_model(i: int) -> Tuple[VehicleModel, Callable, Callable]:
        if brake_zone is not None and brake_zone.mask[i]:
            return brake_zone.model, brake_zone.f_space, brake_zone.eval_at_point
        return model, f_space, eval_at_point

    total_time = 0
    timed_time = 0
    pure_timed_time = 0
    # Per node, so a gate-to-gate time can be measured after the solve.
    cumulative_time = [total_time]

    use_corner_constraints = len(model.get_corner_offsets()) > 0
    if use_normalization:
        x_scale, x_shift = model.get_reduced_state_scaling()
        if x_scale is None or x_shift is None:
            raise RuntimeError("Normalization scales not defined.")
        d_scale = x_scale[0]
        d_shift = x_shift[0]
        psi_scale = x_scale[1]
        psi_shift = x_shift[1]

    for i in range(N - 1):
        x_i = X[i, :].T
        u_i = U[i, :].T
        kappa_i = kappa_param[i]
        kappa_half_i = kappa_half_param[i]
        kappa_next_i = kappa_param[i + 1]
        model_i, f_space_i, eval_at_point_i = _node_model(i)

        x_next = integrator.step(
            f_space_i,
            x_i,
            u_i,
            kappa_i,
            ds,
            kappa_half=kappa_half_i,
            kappa_next=kappa_next_i,
        )
        opti.subject_to(X[i + 1, :].T == x_next)

        for x_c, kappa_c in _constraint_eval_points(
            integrator,
            f_space_i,
            x_i,
            u_i,
            kappa_i,
            ds,
            kappa_half=kappa_half_i,
            kappa_next=kappa_next_i,
        ):
            if use_normalization:
                g_list = model_i.get_constraints_normalized(x_c, u_i, kappa_c)
            else:
                full_state, _ = eval_at_point_i(x_c, u_i, kappa_c)
                g_list = model_i.get_constraints(full_state, u_i, kappa_c)
            for g in g_list:
                opti.subject_to(g <= 0)

        if use_corner_constraints:
            if use_normalization:
                d_phys_i = x_i[0] * d_scale + d_shift
                psi_phys_i = x_i[1] * psi_scale + psi_shift
            else:
                d_phys_i = x_i[0]
                psi_phys_i = x_i[1]
            for g in model.get_corner_constraints(
                d_phys_i,
                psi_phys_i,
                kappa_i,
                w_left_param[i],
                w_right_param[i],
            ):
                opti.subject_to(g <= 0)
        elif use_normalization:
            ub_d_norm = (w_left_param[i] - d_shift) / d_scale
            lb_d_norm = (-w_right_param[i] - d_shift) / d_scale
            opti.subject_to(lb_d_norm <= x_i[0])
            opti.subject_to(x_i[0] <= ub_d_norm)
        else:
            opti.subject_to(-w_right_param[i] <= x_i[0])
            opti.subject_to(x_i[0] <= w_left_param[i])

        dt_i = integrator.time_step(
            f_space_i,
            eval_at_point_i,
            x_i,
            u_i,
            kappa_i,
            ds,
            kappa_half=kappa_half_i,
            kappa_next=kappa_next_i,
            eps=s_dot_floor,
            smooth_eps=s_dot_smooth_eps,
        )
        total_time += dt_i
        cumulative_time.append(total_time)
        if time_weights is not None:
            w_i = float(time_weights[i])
            timed_time += w_i * dt_i
            if w_i >= 1.0 - 1e-9:
                pure_timed_time += dt_i

    if use_corner_constraints:
        x_last = X[N - 1, :].T
        if use_normalization:
            d_phys_last = x_last[0] * d_scale + d_shift
            psi_phys_last = x_last[1] * psi_scale + psi_shift
        else:
            d_phys_last = x_last[0]
            psi_phys_last = x_last[1]
        for g in model.get_corner_constraints(
            d_phys_last,
            psi_phys_last,
            kappa_param[N - 1],
            w_left_param[N - 1],
            w_right_param[N - 1],
        ):
            opti.subject_to(g <= 0)
    elif use_normalization:
        ub_d_norm_last = (w_left_param[N - 1] - d_shift) / d_scale
        lb_d_norm_last = (-w_right_param[N - 1] - d_shift) / d_scale
        opti.subject_to(lb_d_norm_last <= X[N - 1, 0])
        opti.subject_to(X[N - 1, 0] <= ub_d_norm_last)
    else:
        opti.subject_to(-w_right_param[N - 1] <= X[N - 1, 0])
        opti.subject_to(X[N - 1, 0] <= w_left_param[N - 1])

    if mode != "trackdrive" and enforce_terminal_constraints:
        model_last, _, eval_at_point_last = _node_model(N - 1)
        _apply_final_node_constraints(
            opti,
            X,
            U,
            N,
            kappa_param[N - 1],
            model_last,
            eval_at_point_last,
            use_normalization,
        )

    if mode == "trackdrive":
        opti.subject_to(X[N - 1, :].T == X[0, :].T)

    if terminal_speed is not None:
        _apply_terminal_speed(opti, X, N, model, terminal_speed, use_normalization)

    if mode == "skidpad" and terminal_straight_m is not None and terminal_straight_m > 0.0:
        n_window = int(round(terminal_straight_m / ds))
        _apply_terminal_window(
            opti, X, N, n_window, _terminal_state_bounds(model, use_normalization)
        )

    if mode == "autox" and terminal_state_constraint:
        _apply_terminal_window(
            opti, X, N, terminal_window_nodes, _terminal_state_bounds(model, use_normalization)
        )

    if N > 1:
        dU = U[1:, :] - U[:-1, :]
        penalty = 0
        for j in range(nu):
            w_j = float(reg_du_arr[j])
            if w_j != 0.0:
                penalty += w_j * ca.sumsqr(dU[:, j])
    else:
        penalty = 0

    if reg_u_l2 is not None and N > 0:
        reg_u_l2_arr = _per_input_weights(reg_u_l2, nu, "reg_u_l2", default=0.0)
        for j in range(nu):
            w_j = float(reg_u_l2_arr[j])
            if w_j != 0.0:
                penalty += w_j * ca.sumsqr(U[:, j])

    objective_time = timed_time if time_weights is not None else total_time
    obj = objective_time + penalty
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
            "ipopt.print_level": 5 if solver_verbose else 0,
            "print_time": 1 if solver_verbose else 0,
            "ipopt.sb": "yes",
            "ipopt.nlp_scaling_method": "none",  # IPOPT internal scaling deactivated
            "ipopt.max_cpu_time": 600.0,  # fail fast rather than hang
        },
        {},
    )

    return BuiltOcp(
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
        objective_time,
        pure_timed_time,
        ca.vertcat(*cumulative_time),
    )


def solve_ocp_and_save(
    track: Dict,
    model: VehicleModel,
    solution_path: Path,
    integrator: SpaceIntegrator | None = None,
    initial_speed: float = 5.0,
    reg_du: float | Sequence[float] | np.ndarray | None = None,
    reg_u_l2: float | Sequence[float] | np.ndarray | None = None,
    run_config: Dict | None = None,
    use_normalization: bool = True,
    solver_verbose: bool = False,
    boundary_margin: float = 0.0,
    mode: str = "trackdrive",
    time_weights: np.ndarray | None = None,
    terminal_speed: float | None = None,
    enforce_terminal_constraints: bool = True,
    autox_timing_offset_m: float | None = None,
    initial_guess: Dict[str, np.ndarray] | None = None,
    terminal_straight_m: float | None = None,
    terminal_state_constraint: bool = False,
    terminal_window_nodes: int = 2,
    D_safe_braking: float | None = None,
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
    reg_du : float or array-like, optional
        Input rate-regularization weight(s). If one scalar is given
        make isotropic matrix.
    initial_guess : dict, optional
        ``{"X": (N, nx), "U": (N, nu)}`` in the same units the solver uses
        (normalised when ``use_normalization``). Replaces the default
        centreline / constant-speed guess, which is what makes margin
        continuation possible on corridors too tight to start from d = 0.

    Returns
    -------
    dict
        Solution dictionary (same as saved JSON).
    """
    if integrator is None:
        integrator = EulerIntegrator()

    (
        opti,
        X,
        U,
        params,
        obj,
        total_time_expr,
        timed_time_expr,
        pure_timed_expr,
        cumulative_time_expr,
    ) = build_ocp(
        track,
        model,
        integrator=integrator,
        reg_du=reg_du,
        reg_u_l2=reg_u_l2,
        use_normalization=use_normalization,
        solver_verbose=solver_verbose,
        boundary_margin=boundary_margin,
        mode=mode,
        time_weights=time_weights,
        terminal_speed=terminal_speed,
        enforce_terminal_constraints=enforce_terminal_constraints,
        terminal_straight_m=terminal_straight_m,
        terminal_state_constraint=terminal_state_constraint,
        terminal_window_nodes=terminal_window_nodes,
        D_safe_braking=D_safe_braking,
    )

    reduced_names = model.reduced_state_names()
    N = len(track["arc_lengths"])
    x0_phys = np.zeros(model.nx_reduced)
    v_idx = reduced_names.index("v") if "v" in reduced_names else reduced_names.index("v_long")
    x0_phys[v_idx] = initial_speed

    if use_normalization:
        x0_norm = model.reduced_state_phys_to_norm(x0_phys)
        opti.set_value(params["x0"], x0_norm)
    else:
        opti.set_value(params["x0"], x0_phys)

    if use_normalization:
        v_norm = x0_norm[v_idx]
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        opti.set_initial(X[:, v_idx], v_norm)
    else:
        opti.set_initial(X, 0)
        opti.set_initial(U, 0)
        opti.set_initial(X[:, v_idx], initial_speed)

    if initial_guess is not None:
        if "X" in initial_guess:
            opti.set_initial(X, np.asarray(initial_guess["X"], dtype=float))
        if "U" in initial_guess:
            opti.set_initial(U, np.asarray(initial_guess["U"], dtype=float))

    start_time = time.perf_counter()
    try:
        sol = opti.solve()
    except RuntimeError as exc:
        solve_time_s = time.perf_counter() - start_time
        stats = opti.stats()
        return_status = stats.get("return_status")
        iter_count = stats.get("iter_count")

        debug_solution_path = solution_path.with_name(f"{solution_path.stem}_debug_failure.json")
        debug_payload = {
            "error": str(exc),
            "return_status": return_status,
            "iter_count": iter_count,
            "solve_time_s": solve_time_s,
            "use_normalization": bool(use_normalization),
            "state_names": model.reduced_state_names(),
            "input_names": model.get_input_names(),
            "track_summary": {
                "num_points": len(track.get("arc_lengths", [])),
                "ds_m": float(track.get("ds_m", 0.0)),
            },
            "X_debug": None,
            "U_debug": None,
            "objective_debug": None,
            "total_time_debug": None,
        }

        x_dbg = _safe_debug_value(opti, X)
        u_dbg = _safe_debug_value(opti, U)
        obj_dbg = _safe_debug_value(opti, obj)
        t_dbg = _safe_debug_value(opti, total_time_expr)

        if isinstance(x_dbg, np.ndarray):
            debug_payload["X_debug"] = x_dbg.tolist()
        if isinstance(u_dbg, np.ndarray):
            debug_payload["U_debug"] = u_dbg.tolist()
        if isinstance(obj_dbg, float):
            debug_payload["objective_debug"] = obj_dbg
        if isinstance(t_dbg, float):
            debug_payload["total_time_debug"] = t_dbg

        debug_solution_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_solution_path.open("w") as f:
            json.dump(debug_payload, f, indent=2)

        print("OCP solve failed.", file=sys.stderr, flush=True)
        print(
            f"  IPOPT status: {return_status}, "
            f"iterations: {iter_count if iter_count is not None else 'N/A'}",
            file=sys.stderr,
            flush=True,
        )
        print(f"  Runtime: {solve_time_s:.3f} s", file=sys.stderr, flush=True)
        print(
            f"  Debug snapshot saved to: {debug_solution_path}",
            file=sys.stderr,
            flush=True,
        )
        try:
            print("  Infeasibilities at latest iterate:", file=sys.stderr, flush=True)
            opti.debug.show_infeasibilities()
        except Exception:
            print(
                "  Could not print infeasibilities from opti.debug.",
                file=sys.stderr,
                flush=True,
            )
        raise
    solve_time_s = time.perf_counter() - start_time

    stats = opti.stats()
    iter_count = stats.get("iter_count")
    return_status = stats.get("return_status")

    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))
    obj_val = float(sol.value(obj))
    lap_time_s = float(sol.value(total_time_expr))
    timed_time_s = float(sol.value(timed_time_expr))
    pure_timed_s = float(sol.value(pure_timed_expr)) if time_weights is not None else 0.0
    reg_term = obj_val - timed_time_s

    autox_lap_time_s: float | None = None
    autox_timing_warning: str | None = None
    if autox_timing_offset_m is not None:
        time_at_node_s = np.asarray(sol.value(cumulative_time_expr), dtype=np.float64).reshape(-1)
        autox_lap_time_s, autox_timing_warning = _autox_gate_lap_time(
            track, time_at_node_s, float(autox_timing_offset_m)
        )

    N = len(track["arc_lengths"])
    ds_m = (
        float(track.get("ds_m", track["arc_lengths"][1] - track["arc_lengths"][0]))
        if N > 1
        else float(track.get("ds_m", 0.0))
    )
    time_per_point_ms = solve_time_s / N * 1e3 if N > 0 else None
    time_per_iter_ms = solve_time_s / iter_count * 1e3 if iter_count not in (None, 0) else None

    print(f"Solved full-lap OCP.  Objective value: {obj_val:.2f}")
    print(f"  Full-maneuver time: {lap_time_s:.2f} s")
    if autox_lap_time_s is not None:
        print(
            f"  Autox lap time (timing gate @ {autox_timing_offset_m:.1f} m): "
            f"{autox_lap_time_s:.3f} s"
        )
    elif autox_timing_warning is not None:
        print(f"  [WARN] Could not compute autox timing-gate lap time: {autox_timing_warning}")
    if time_weights is not None:
        n_timed = int(np.sum(np.asarray(time_weights) >= 1.0 - 1e-9))
        score = pure_timed_s / 2.0 if n_timed > 0 else float("nan")
        print(f"  Timed laps total: {pure_timed_s:.3f} s  (score avg: {score:.3f} s)")
        print(f"  Weighted objective time: {timed_time_s:.2f} s")
    print(
        f"  Regularisation term: {reg_term:.4f} "
        f"({100.0 * reg_term / obj_val:.2f}% of objective)"
    )
    print(
        f"Solve stats: N={N}, ds={ds_m:.3f} m, "
        f"time={solve_time_s:.3f} s, "
        f"iters={iter_count if iter_count is not None else 'N/A'}, "
        f"status={return_status}"
    )
    if use_normalization:
        x_scale, x_shift = model.get_reduced_state_scaling()
        u_scale, u_shift = model.get_input_scaling()
        if x_scale is None or x_shift is None or u_scale is None or u_shift is None:
            raise RuntimeError(
                "Normalisation scales/shifts are not defined for solution " "post-processing."
            )
        x_scale_np = np.asarray(x_scale).astype(float).reshape(1, -1)
        x_shift_np = np.asarray(x_shift).astype(float).reshape(1, -1)
        u_scale_np = np.asarray(u_scale).astype(float).reshape(1, -1)
        u_shift_np = np.asarray(u_shift).astype(float).reshape(1, -1)

        X_phys = X_sol * x_scale_np + x_shift_np
        U_phys = U_sol * u_scale_np + u_shift_np
    else:
        X_phys = X_sol
        U_phys = U_sol

    print(f"v min/max: {X_phys[:, v_idx].min():.2f} / " f"{X_phys[:, v_idx].max():.2f} m/s")

    positions = np.array(track["positions"], dtype=np.float64)
    headings = np.array(track["headings"], dtype=np.float64)
    normals = np.column_stack((-np.sin(headings), np.cos(headings)))
    d = X_phys[:, 0]
    path_xy = positions + d[:, None] * normals

    input_names = model.get_input_names()

    sol_dict = {
        "mode": mode,
        "path_xy": path_xy.tolist(),
        "obj_val": obj_val,
        "state_names": reduced_names,
        "input_names": input_names,
        "arc_lengths": track["arc_lengths"],
        "w_left": track["w_left"],
        "w_right": track["w_right"],
        "kappa": track["curvatures"],
        "headings": track["headings"],
        "timed_mask": track.get("timed_mask"),
        "decel_mask": track.get("decel_mask"),
        "skidpad": track.get("skidpad"),
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
            "timed_time_s": timed_time_s,
            "pure_timed_time_s": pure_timed_s,
            "skidpad_score_s": (pure_timed_s / 2.0) if time_weights is not None else None,
            "autox_lap_time_s": autox_lap_time_s,
            "autox_timing_offset_m": (
                float(autox_timing_offset_m) if autox_timing_offset_m is not None else None
            ),
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
