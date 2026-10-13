"""The OCP of ``build_ocp`` as a sequence of stages, for the stage-wise backends.

Stage ``k`` owns the state ``x_k`` and, except for the last stage, the input
``u_k``. Everything a stage needs beyond those — curvatures, corridor widths,
time weight, launch state — is a per-stage parameter vector ``p_k``, so one set
of compiled functions serves every track.

Stages that share their maths share a :class:`StageType`: the first stage,
ordinary stages, brake-zone stages, terminal-window stages and the last stage,
created only for the combinations that occur. A backend turns each type into
code once and dispatches stages to it.

The closed-loop wrap ``x_{N-1} = x_0`` is not part of any stage (it couples the
first and the last). ``StageProblem.periodic`` flags it and each backend
encodes it in its own way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from fast_lto.optimization.integrators import SpaceIntegrator
from fast_lto.vehicle_models import VehicleModel

#: Parameter layout shared by every stage: geometry, spacing, time weight, launch state.
P_KAPPA, P_KAPPA_HALF, P_KAPPA_NEXT, P_W_LEFT, P_W_RIGHT, P_DS, P_TIME_WEIGHT = range(7)
P_X0 = 7


@dataclass
class StageType:
    """The maths of one kind of stage, as SX expressions in ``(x, u, p)``.

    ``x_next`` and ``dt`` are ``None`` on the last stage, which has no input.
    """

    name: str
    kind: str  # "first", "middle" or "last"
    has_input: bool
    x_next: Optional[ca.SX]
    dt: Optional[ca.SX]
    cost: ca.SX
    eq: ca.SX
    ineq: ca.SX
    ineq_lb: np.ndarray
    ineq_ub: np.ndarray


@dataclass
class StageProblem:
    """A stage-wise OCP: symbols, stage types, and per-stage data."""

    x: ca.SX
    u: ca.SX
    p: ca.SX
    types: List[StageType]
    stage_type: np.ndarray  # (N,) index into types
    params: np.ndarray  # (N, n_p)
    x_lb: np.ndarray
    x_ub: np.ndarray
    u_lb: np.ndarray
    u_ub: np.ndarray
    periodic: bool

    @property
    def N(self) -> int:
        return len(self.stage_type)

    @property
    def nx(self) -> int:
        return int(self.x.numel())

    @property
    def nu(self) -> int:
        return int(self.u.numel())

    def evaluate_dt(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Per-interval ``dt`` (length ``N - 1``) at a solution in solver units."""
        dt = np.zeros(self.N - 1)
        for t_idx, t in enumerate(self.types):
            if t.dt is None:
                continue
            nodes = np.flatnonzero(self.stage_type[: self.N - 1] == t_idx)
            if nodes.size == 0:
                continue
            f = ca.Function("dt", [self.x, self.u, self.p], [t.dt]).map(nodes.size)
            dt[nodes] = np.asarray(f(X[nodes].T, U[nodes].T, self.params[nodes].T)).reshape(-1)
        return dt


def build_stage_problem(
    track: Dict,
    model: VehicleModel,
    integrator: SpaceIntegrator,
    *,
    mode: str,
    boundary_margin: float,
    time_weights: Optional[np.ndarray],
    reg_u_l2: float | Sequence[float] | np.ndarray | None,
    x0: np.ndarray,
    terminal_speed: Optional[float] = None,
    enforce_terminal_constraints: bool = True,
    terminal_straight_m: Optional[float] = None,
    terminal_state_constraint: bool = False,
    terminal_window_nodes: int = 2,
    D_safe_braking: Optional[float] = None,
) -> StageProblem:
    """The same problem as ``build_ocp(..., use_normalization=True)``, stage by stage."""
    # Local import: global_ocp imports this module.
    from fast_lto.optimization import global_ocp as go

    kappa = np.asarray(track["curvatures"], dtype=float)
    kappa_half = np.asarray(track["curvatures_half"], dtype=float)
    w_left = np.asarray(track["w_left"], dtype=float) - boundary_margin
    w_right = np.asarray(track["w_right"], dtype=float) - boundary_margin
    ds = float(track["ds_m"])
    N = len(kappa)
    nx, nu = model.nx_reduced, model.nu

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p", P_X0 + nx)

    f_space, eval_at_point = go.build_space_dynamics_normalized(model)
    stage_fns = {False: go._build_stage_function(model, f_space, eval_at_point, integrator, True)}
    models = {False: model}
    in_brake_zone = np.zeros(N, dtype=bool)
    if mode == "autox" and D_safe_braking is not None:
        zone = go._build_brake_zone(model, track, N, D_safe_braking, True)
        stage_fns[True] = go._build_stage_function(
            zone.model, zone.f_space, zone.eval_at_point, integrator, True
        )
        models[True] = zone.model
        in_brake_zone = zone.mask.astype(bool)

    in_window = np.zeros(N, dtype=bool)
    n_window = 0
    if mode == "skidpad" and terminal_straight_m is not None and terminal_straight_m > 0.0:
        n_window = int(round(terminal_straight_m / ds))
    elif mode == "autox" and terminal_state_constraint:
        n_window = int(terminal_window_nodes)
    if n_window > 0:
        in_window[N - min(N, max(1, n_window)) :] = True
    window_bounds = go._terminal_state_bounds(model, True)

    reg = go._per_input_weights(reg_u_l2, nu, "reg_u_l2", default=0.0)
    x_scale, x_shift = (
        np.asarray(a, dtype=float).reshape(-1) for a in model.get_reduced_state_scaling()
    )
    names = model.reduced_state_names()
    v_idx = names.index("v") if "v" in names else names.index("v_long")

    types: Dict[Tuple[str, bool, bool], StageType] = {}

    def stage_type(kind: str, brake: bool, window: bool) -> int:
        key = (kind, brake, window)
        if key not in types:
            types[key] = _make_type(kind, brake, window)
        return list(types).index(key)

    def _make_type(kind: str, brake: bool, window: bool) -> StageType:
        eq: List[ca.SX] = []
        ineq: List[Tuple[ca.SX, np.ndarray, np.ndarray]] = []

        def upper(expr: ca.SX, ub: float | np.ndarray = 0.0) -> None:
            n = expr.numel()
            ineq.append((expr, np.full(n, -np.inf), np.broadcast_to(ub, (n,)).astype(float)))

        x_next = dt = None
        cost = ca.SX(0)
        last_has_input = False
        if kind != "last":
            x_next, g, dt = stage_fns[brake](
                x,
                u,
                p[P_KAPPA],
                p[P_KAPPA_HALF],
                p[P_KAPPA_NEXT],
                p[P_W_LEFT],
                p[P_W_RIGHT],
                p[P_DS],
            )
            upper(g)
            cost = p[P_TIME_WEIGHT] * dt + ca.sum1(ca.DM(reg) * u**2)
            if kind == "first" and mode != "trackdrive":
                eq.append(x - p[P_X0:])
        else:
            d_phys = x[0] * x_scale[0] + x_shift[0]
            psi_phys = x[1] * x_scale[1] + x_shift[1]
            corners = model.get_corner_constraints(
                d_phys, psi_phys, p[P_KAPPA], p[P_W_LEFT], p[P_W_RIGHT]
            )
            if corners:
                upper(ca.vertcat(*corners))
            else:
                upper(
                    ca.vertcat(
                        (-p[P_W_RIGHT] - x_shift[0]) / x_scale[0] - x[0],
                        x[0] - (p[P_W_LEFT] - x_shift[0]) / x_scale[0],
                    )
                )
            if mode != "trackdrive" and enforce_terminal_constraints:
                x_mx, u_mx, kappa_mx = ca.MX.sym("x", nx), ca.MX.sym("u", nu), ca.MX.sym("kappa")
                g_mx = models[brake].get_constraints_normalized(x_mx, u_mx, kappa_mx)
                g_last = ca.Function("g_last", [x_mx, u_mx, kappa_mx], [ca.vertcat(*g_mx)])
                g_x = g_last.expand()(x, u, p[P_KAPPA])
                upper(g_x)
                # Models whose constraints read the input keep a last input, as in build_ocp.
                last_has_input = bool(ca.depends_on(g_x, u))
                if last_has_input:
                    cost = ca.sum1(ca.DM(reg) * u**2)
            if terminal_speed is not None:
                upper(x[v_idx], (terminal_speed - x_shift[v_idx]) / x_scale[v_idx])
        if window:
            for idx, lo, hi in window_bounds:
                ineq.append((x[idx], np.array([lo]), np.array([hi])))

        return StageType(
            name=f"{kind}{'_brake' if brake else ''}{'_window' if window else ''}",
            kind=kind,
            has_input=kind != "last" or last_has_input,
            x_next=x_next,
            dt=dt,
            cost=cost,
            eq=ca.vertcat(*eq) if eq else ca.SX(0, 1),
            ineq=ca.vertcat(*[e for e, _, _ in ineq]),
            ineq_lb=np.concatenate([lo for _, lo, _ in ineq]),
            ineq_ub=np.concatenate([hi for _, _, hi in ineq]),
        )

    tw = np.ones(N - 1) if time_weights is None else np.asarray(time_weights, float)[: N - 1]
    type_of = np.zeros(N, dtype=np.int32)
    params = np.zeros((N, P_X0 + nx))
    for k in range(N):
        kind = "first" if k == 0 else ("last" if k == N - 1 else "middle")
        type_of[k] = stage_type(kind, bool(in_brake_zone[k]), bool(in_window[k]))
        k_next = min(k + 1, N - 1)
        params[k, :P_X0] = [
            kappa[k],
            kappa_half[k],
            kappa[k_next],
            w_left[k],
            w_right[k],
            ds,
            tw[k] if k < N - 1 else 0.0,
        ]
        params[k, P_X0:] = x0

    state_bounds = model.reduced_state_bounds_normalized()
    input_bounds = model.input_bounds_normalized()
    if state_bounds is None or input_bounds is None:
        raise ValueError("Stage-wise solvers need finite state and input bounds.")
    x_lb, x_ub = (np.asarray(b, dtype=float) for b in state_bounds)
    u_lb, u_ub = (np.asarray(b, dtype=float) for b in input_bounds)
    return StageProblem(
        x=x,
        u=u,
        p=p,
        types=list(types.values()),
        stage_type=type_of,
        params=params,
        x_lb=x_lb,
        x_ub=x_ub,
        u_lb=u_lb,
        u_ub=u_ub,
        periodic=mode == "trackdrive",
    )
