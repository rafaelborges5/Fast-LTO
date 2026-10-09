"""IPOPT on a :class:`StageProblem`, with derivatives built per stage.

Each stage type gets three functions of ``(z, p)`` with ``z = [x; u]``: values,
Jacobian blocks, and the Hessian of its share of the Lagrangian. They are
compiled once (see ``codegen``) and mapped over the stages of that type. The
NLP Jacobian and Hessian are a fixed permutation of the block nonzeros: the
Hessian is block-diagonal because ``x_{k+1}`` enters the dynamics linearly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import casadi as ca
import numpy as np

from fast_lto.optimization.codegen import CompiledLibrary, compile_library
from fast_lto.optimization.stage_problem import StageProblem, StageType


class StageSolution(NamedTuple):
    X: np.ndarray  # (N, nx), solver units
    U: np.ndarray  # (N, nu), zero where a stage has no input
    success: bool
    return_status: str
    iter_count: Optional[int]
    solve_time_s: float


@dataclass
class _TypeFunctions:
    vals: ca.Function
    jac: ca.Function
    hess: ca.Function
    n_eq: int
    n_ineq: int
    has_next: bool


def _type_functions(t: StageType, problem: StageProblem, tag: str) -> _TypeFunctions:
    z = ca.vertcat(problem.x, problem.u) if t.has_input else problem.x
    p = problem.p
    has_next = t.x_next is not None
    x_next = t.x_next if t.x_next is not None else ca.SX(0, 1)
    outs = [x_next, t.eq, t.ineq, t.cost]
    jacs = [
        ca.jacobian(x_next, z),
        ca.jacobian(t.eq, z),
        ca.jacobian(t.ineq, z),
        ca.gradient(t.cost, z),
    ]
    lam_next = ca.SX.sym("lam_next", x_next.numel())
    lam_eq = ca.SX.sym("lam_eq", t.eq.numel())
    lam_ineq = ca.SX.sym("lam_ineq", t.ineq.numel())
    sigma = ca.SX.sym("sigma")
    lag = (
        sigma * t.cost - ca.dot(lam_next, x_next) + ca.dot(lam_eq, t.eq) + ca.dot(lam_ineq, t.ineq)
    )
    hess = ca.triu(ca.hessian(lag, z)[0])
    return _TypeFunctions(
        vals=ca.Function(f"stage_vals_{tag}", [z, p], outs),
        jac=ca.Function(f"stage_jac_{tag}", [z, p], outs + jacs),
        hess=ca.Function(f"stage_hess_{tag}", [z, p, lam_next, lam_eq, lam_ineq, sigma], [hess]),
        n_eq=int(t.eq.numel()),
        n_ineq=int(t.ineq.numel()),
        has_next=has_next,
    )


class _Scatter:
    """Rows/columns plus nonzero vectors; ``build`` permutes them into one sparse MX."""

    def __init__(self, nrow: int, ncol: int) -> None:
        self.shape = (nrow, ncol)
        self.rows: List[np.ndarray] = []
        self.cols: List[np.ndarray] = []
        self.vals: List[ca.MX] = []

    def add(self, rows: np.ndarray, cols: np.ndarray, vals: ca.MX) -> None:
        # (n, stages) index arrays flatten column-major: mapped outputs are stage-major.
        rows = np.asarray(rows, dtype=np.int64).ravel(order="F")
        if rows.size:
            self.rows.append(rows)
            self.cols.append(np.asarray(cols, dtype=np.int64).ravel(order="F"))
            self.vals.append(ca.vec(vals))

    def build(self) -> ca.MX:
        rows = np.concatenate(self.rows).tolist()
        cols = np.concatenate(self.cols).tolist()
        sp, mapping = ca.Sparsity.triplet(*self.shape, rows, cols, True)
        mapping = np.asarray(mapping)
        if np.unique(mapping).size != mapping.size:
            raise AssertionError("overlapping entries in sparse assembly")
        perm = np.empty(mapping.size, dtype=np.int64)
        perm[mapping] = np.arange(mapping.size)
        return ca.MX(sp, ca.vertcat(*self.vals)[perm.tolist()])


def _block(sp: ca.Sparsity) -> Tuple[np.ndarray, np.ndarray]:
    r, c = sp.get_triplet()
    return np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64)


class _Layout:
    """Variable and constraint offsets for x0, u0, x1, u1, ..., x_{N-1}."""

    def __init__(self, problem: StageProblem, fns: List[_TypeFunctions]) -> None:
        N, nx, nu = problem.N, problem.nx, problem.nu
        self.x_off = np.zeros(N, dtype=np.int64)
        self.u_off = np.full(N, -1, dtype=np.int64)
        self.dyn_row = np.full(N, -1, dtype=np.int64)
        self.eq_row = np.zeros(N, dtype=np.int64)
        self.ineq_row = np.zeros(N, dtype=np.int64)
        w = row = 0
        lbg: List[np.ndarray] = []
        ubg: List[np.ndarray] = []
        for k in range(N):
            t = problem.types[problem.stage_type[k]]
            f = fns[problem.stage_type[k]]
            self.x_off[k] = w
            w += nx
            if t.has_input:
                self.u_off[k] = w
                w += nu
            if f.has_next:
                self.dyn_row[k] = row
                row += nx
                lbg.append(np.zeros(nx))
                ubg.append(np.zeros(nx))
            self.eq_row[k] = row
            row += f.n_eq
            lbg.append(np.zeros(f.n_eq))
            ubg.append(np.zeros(f.n_eq))
            self.ineq_row[k] = row
            row += f.n_ineq
            lbg.append(t.ineq_lb)
            ubg.append(t.ineq_ub)
        self.periodic_row = row
        if problem.periodic:
            row += nx
            lbg.append(np.zeros(nx))
            ubg.append(np.zeros(nx))
        self.nw, self.ng = w, row
        self.lbg, self.ubg = np.concatenate(lbg), np.concatenate(ubg)

    def z_index(self, k: int, has_input: bool, nx: int, nu: int) -> np.ndarray:
        idx = np.arange(self.x_off[k], self.x_off[k] + nx)
        if has_input:
            idx = np.concatenate([idx, np.arange(self.u_off[k], self.u_off[k] + nu)])
        return idx


def _ranges(starts: np.ndarray, n: int) -> np.ndarray:
    """(n, m) matrix whose column j is ``starts[j] + arange(n)``."""
    return starts[None, :] + np.arange(n)[:, None]


def _compile(fns: List[_TypeFunctions]) -> CompiledLibrary:
    return compile_library("fast_lto_stage", [f for tf in fns for f in (tf.vals, tf.jac, tf.hess)])


def prepare(problem: StageProblem) -> CompiledLibrary:
    """Compile (or find in the cache) the stage functions for ``problem``."""
    return _compile([_type_functions(t, problem, str(i)) for i, t in enumerate(problem.types)])


def build_nlp(problem: StageProblem, compile: bool = True) -> Tuple[Dict, Dict, Dict, _Layout]:
    """NLP expressions, custom derivative functions, bounds and layout."""
    N, nx, nu = problem.N, problem.nx, problem.nu
    fns = [_type_functions(t, problem, str(i)) for i, t in enumerate(problem.types)]
    if compile:
        lib = _compile(fns)
        for tf in fns:
            tf.vals = ca.external(tf.vals.name(), str(lib.path))
            tf.jac = ca.external(tf.jac.name(), str(lib.path))
            tf.hess = ca.external(tf.hess.name(), str(lib.path))
    L = _Layout(problem, fns)
    w = ca.MX.sym("w", L.nw)
    p_dummy = ca.MX.sym("p", 0)
    lam_f = ca.MX.sym("lam_f")
    lam_g = ca.MX.sym("lam_g", L.ng)

    g_vals, g_jac = _Scatter(L.ng, 1), _Scatter(L.ng, 1)
    J = _Scatter(L.ng, L.nw)
    grad = _Scatter(L.nw, 1)
    H = _Scatter(L.nw, L.nw)
    f_vals: ca.MX = 0
    f_jac: ca.MX = 0

    for t_idx, (t, tf) in enumerate(zip(problem.types, fns)):
        nodes = np.flatnonzero(problem.stage_type == t_idx)
        if nodes.size == 0:
            continue
        m = nodes.size
        zi = np.stack([L.z_index(k, t.has_input, nx, nu) for k in nodes], axis=1)
        Z = ca.reshape(w[zi.ravel(order="F").tolist()], zi.shape[0], m)
        P = ca.DM(problem.params[nodes].T)
        rows: Dict[str, np.ndarray] = {
            "eq": _ranges(L.eq_row[nodes], tf.n_eq),
            "ineq": _ranges(L.ineq_row[nodes], tf.n_ineq),
        }
        if tf.has_next:
            rows["dyn"] = _ranges(L.dyn_row[nodes], nx)
            next_cols = _ranges(L.x_off[nodes + 1], nx)

        for target, fn in ((g_vals, tf.vals), (g_jac, tf.jac)):
            out = fn.map(m)(Z, P)
            x_next, eq, ineq, cost = out[0], out[1], out[2], out[3]
            if tf.has_next:
                target.add(
                    rows["dyn"],
                    np.zeros(rows["dyn"].size),
                    w[next_cols.ravel(order="F").tolist()] - ca.vec(x_next),
                )
            target.add(rows["eq"], np.zeros(rows["eq"].size), eq)
            target.add(rows["ineq"], np.zeros(rows["ineq"].size), ineq)
            if fn is tf.vals:
                f_vals = f_vals + ca.sum2(cost)
            else:
                f_jac = f_jac + ca.sum2(cost)
                J_next, J_eq, J_ineq, g_cost = out[4], out[5], out[6], out[7]
                if tf.has_next:
                    br, bc = _block(tf.jac.sparsity_out(4))
                    J.add(rows["dyn"][br], zi[bc], -J_next.nz[:])
                    J.add(rows["dyn"], next_cols, ca.DM.ones(rows["dyn"].size))
                for key, Jb, o in (("eq", J_eq, 5), ("ineq", J_ineq, 6)):
                    br, bc = _block(tf.jac.sparsity_out(o))
                    J.add(rows[key][br], zi[bc], Jb.nz[:])
                br, _ = _block(tf.jac.sparsity_out(7))
                grad.add(zi[br], np.zeros(zi[br].size), g_cost.nz[:])

        def lam(key: str) -> ca.MX:
            r = rows[key]
            return ca.reshape(lam_g[r.ravel(order="F").tolist()], r.shape[0], m)

        lam_next = lam("dyn") if tf.has_next else ca.MX(0, m)
        Hk = tf.hess.map(m)(Z, P, lam_next, lam("eq"), lam("ineq"), ca.repmat(lam_f, 1, m))
        hr, hc = _block(tf.hess.sparsity_out(0))
        H.add(zi[hr], zi[hc], Hk.nz[:])

    if problem.periodic:
        prow = L.periodic_row + np.arange(nx)
        x_last = L.x_off[N - 1] + np.arange(nx)
        x_first = L.x_off[0] + np.arange(nx)
        for target in (g_vals, g_jac):
            target.add(prow, np.zeros(nx), w[x_last.tolist()] - w[x_first.tolist()])
        J.add(prow, x_last, ca.DM.ones(nx))
        J.add(prow, x_first, -ca.DM.ones(nx))

    g = ca.densify(g_vals.build())
    gj = ca.densify(g_jac.build())
    grad_f = ca.densify(grad.build())
    nlp = {"x": w, "p": p_dummy, "f": f_vals, "g": g}
    custom = {
        "grad_f": ca.Function(
            "nlp_grad_f", [w, p_dummy], [f_jac, grad_f], ["x", "p"], ["f", "grad_f_x"]
        ),
        "jac_g": ca.Function(
            "nlp_jac_g", [w, p_dummy], [gj, J.build()], ["x", "p"], ["g", "jac_g_x"]
        ),
        "hess_lag": ca.Function(
            "nlp_hess_l",
            [w, p_dummy, lam_f, lam_g],
            [H.build()],
            ["x", "p", "lam_f", "lam_g"],
            ["triu_hess_gamma_x_x"],
        ),
    }
    lbw = np.empty(L.nw)
    ubw = np.empty(L.nw)
    for k in range(N):
        lbw[L.x_off[k] : L.x_off[k] + nx] = problem.x_lb
        ubw[L.x_off[k] : L.x_off[k] + nx] = problem.x_ub
        if L.u_off[k] >= 0:
            lbw[L.u_off[k] : L.u_off[k] + nu] = problem.u_lb
            ubw[L.u_off[k] : L.u_off[k] + nu] = problem.u_ub
    bounds = {"lbx": lbw, "ubx": ubw, "lbg": L.lbg, "ubg": L.ubg}
    return nlp, custom, bounds, L


def solve(
    problem: StageProblem,
    X0: np.ndarray,
    U0: np.ndarray,
    ipopt_options: Dict,
    compile: bool = True,
) -> StageSolution:
    nlp, custom, bounds, L = build_nlp(problem, compile=compile)
    opts = dict(ipopt_options)
    opts.update(custom)
    # IPOPT returns the bound multipliers itself; skip CasADi's own AD pass,
    # which cannot differentiate through compiled functions.
    opts.update({"calc_lam_p": False, "no_nlp_grad": True})
    solver = ca.nlpsol("stage_ipopt", "ipopt", nlp, opts)

    N, nx, nu = problem.N, problem.nx, problem.nu
    w0 = np.zeros(L.nw)
    for k in range(N):
        w0[L.x_off[k] : L.x_off[k] + nx] = X0[k]
        if L.u_off[k] >= 0:
            w0[L.u_off[k] : L.u_off[k] + nu] = U0[k]
    t0 = time.perf_counter()
    res = solver(x0=w0, p=np.zeros(0), **bounds)
    elapsed = time.perf_counter() - t0
    stats = solver.stats()

    wopt = np.asarray(res["x"]).reshape(-1)
    X = np.stack([wopt[L.x_off[k] : L.x_off[k] + nx] for k in range(N)])
    U = np.zeros((N, nu))
    for k in range(N):
        if L.u_off[k] >= 0:
            U[k] = wopt[L.u_off[k] : L.u_off[k] + nu]
    return StageSolution(
        X,
        U,
        bool(stats.get("success")),
        str(stats.get("return_status")),
        stats.get("iter_count"),
        elapsed,
    )
