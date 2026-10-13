"""FATROP on a :class:`StageProblem`, through FATROP's native C interface.

CasADi's own FATROP plugin differentiates the whole NLP itself and cannot take
per-stage derivatives. FATROP's C interface asks for exactly those, stage by
stage, so each stage type is generated in FATROP's layout and compiled together
with a small, model-independent adapter (``fatrop_glue.c``).

FATROP couples consecutive stages only. The trackdrive wrap ``x_{N-1} = x_0``
is encoded by carrying a copy ``c`` of ``x_0`` through every stage
(``c_{k+1} = c_k``) and constraining ``x_0 = c_0`` and ``x_{N-1} = c_{N-1}``.
"""

from __future__ import annotations

import ctypes
import time
from importlib import resources
from typing import Dict, List, Tuple

import casadi as ca
import numpy as np

from fast_lto.optimization.codegen import CompiledLibrary, compile_library
from fast_lto.optimization.stage_ipopt import StageSolution
from fast_lto.optimization.stage_problem import StageProblem

_FUNCTION_SLOTS = ("BAbt", "b", "RSQrqt", "Ggt", "g", "Ggti", "gi", "rq", "L")


class _FatropType:
    def __init__(self, nu: int, nx: int, nx_next: int, ng: int, ngi: int, n_p: int) -> None:
        self.nu, self.nx, self.nx_next, self.ng, self.ngi, self.n_p = nu, nx, nx_next, ng, ngi, n_p
        self.functions: Dict[str, ca.Function] = {}
        self.lb = np.zeros(0)
        self.ub = np.zeros(0)


def _fatrop_types(problem: StageProblem) -> List[_FatropType]:
    """Each stage type in FATROP's layout: variables ``(u, x)``, bounds as inequalities."""
    nx, periodic = problem.nx, problem.periodic
    x, u, p = problem.x, problem.u, problem.p
    out = []
    for i, t in enumerate(problem.types):
        c = ca.SX.sym("c", nx) if periodic else ca.SX(0, 1)
        xa = ca.vertcat(x, c)
        ua = u if t.has_input else ca.SX(0, 1)
        ux = ca.vertcat(ua, xa)

        eq = [t.eq]
        if periodic and t.kind in ("first", "last"):
            eq.append(x - c)
        eq_e = ca.vertcat(*eq)
        ineq = [t.ineq, x] + ([u] if t.has_input else [])
        lb = [t.ineq_lb, problem.x_lb] + ([problem.u_lb] if t.has_input else [])
        ub = [t.ineq_ub, problem.x_ub] + ([problem.u_ub] if t.has_input else [])
        ineq_e = ca.vertcat(*ineq)

        has_next = t.x_next is not None
        ft = _FatropType(
            int(ua.numel()),
            int(xa.numel()),
            int(xa.numel()) if has_next else 0,
            int(eq_e.numel()),
            int(ineq_e.numel()),
            int(p.numel()),
        )
        ft.lb, ft.ub = np.concatenate(lb), np.concatenate(ub)

        s = ca.SX.sym("s")
        lam_dyn = ca.SX.sym("lam_dyn", ft.nx_next)
        lam_g = ca.SX.sym("lam_g", ft.ng)
        lam_gi = ca.SX.sym("lam_gi", ft.ngi)
        lag = s * t.cost + ca.dot(lam_g, eq_e) + ca.dot(lam_gi, ineq_e)
        fns: Dict[str, ca.Function] = {}
        tag = f"{i}"
        if has_next:
            # FATROP's dynamics residual b = f(u, x) - x_{k+1}, multiplier added.
            xkp1 = ca.SX.sym("xkp1", ft.nx_next)
            b = ca.vertcat(t.x_next, c) - xkp1
            lag += ca.dot(lam_dyn, b)
            fns["BAbt"] = ca.Function(
                f"BAbt_{tag}",
                [xkp1, ua, xa, p],
                [ca.densify(ca.vertcat(ca.jacobian(b, ux).T, b.T))],
            )
            fns["b"] = ca.Function(f"b_{tag}", [xkp1, ua, xa, p], [ca.densify(b)])
        # Last row: gradient of the Lagrangian, which is what FATROP's Riccati
        # recursion expects alongside the Hessian.
        H = ca.hessian(lag, ux)[0]
        fns["RSQrqt"] = ca.Function(
            f"RSQrqt_{tag}",
            [s, ua, xa, lam_dyn, lam_g, lam_gi, p],
            [ca.densify(ca.vertcat(H, ca.gradient(lag, ux).T))],
        )
        if ft.ng:
            fns["Ggt"] = ca.Function(
                f"Ggt_{tag}", [ua, xa, p], [ca.densify(ca.vertcat(ca.jacobian(eq_e, ux).T, eq_e.T))]
            )
            fns["g"] = ca.Function(f"g_{tag}", [ua, xa, p], [ca.densify(eq_e)])
        fns["Ggti"] = ca.Function(
            f"Ggti_{tag}",
            [ua, xa, p],
            [ca.densify(ca.vertcat(ca.jacobian(ineq_e, ux).T, ineq_e.T))],
        )
        fns["gi"] = ca.Function(f"gi_{tag}", [ua, xa, p], [ca.densify(ineq_e)])
        fns["rq"] = ca.Function(
            f"rq_{tag}", [s, ua, xa, p], [ca.densify(ca.gradient(s * t.cost, ux))]
        )
        fns["L"] = ca.Function(f"L_{tag}", [s, ua, xa, p], [ca.densify(s * t.cost)])
        ft.functions = fns
        out.append(ft)
    return out


def _type_table(types: List[_FatropType]) -> str:
    """C table the adapter dispatches through (matches ``fl_type`` in fatrop_glue.c)."""
    proto = "const double**, double**, long long int*, double*, int"
    work = "long long int*, long long int*, long long int*, long long int*"
    lines = []
    rows = []
    for t in types:
        slots = []
        for slot in _FUNCTION_SLOTS:
            f = t.functions.get(slot)
            if f is None:
                slots.append("{0, 0}")
                continue
            lines.append(f"int {f.name()}({proto});")
            lines.append(f"int {f.name()}_work({work});")
            slots.append(f"{{{f.name()}, {f.name()}_work}}")
        rows.append(
            f"  {{{t.nu}, {t.nx}, {t.nx_next}, {t.ng}, {t.ngi}, {t.n_p}, {{{', '.join(slots)}}}}}"
        )
    lines += [
        f"typedef int (*cas_fn)({proto});",
        f"typedef int (*cas_work)({work});",
        "typedef struct { cas_fn f; cas_work work; } fl_fun;",
        f"typedef struct {{ int nu, nx, nx_next, ng, ngi, np; fl_fun fn[{len(_FUNCTION_SLOTS)}]; }} fl_type;",
        "const fl_type fl_types[] = {",
        ",\n".join(rows),
        "};",
        f"const int fl_n_types = {len(types)};",
    ]
    return "\n".join(lines) + "\n"


class _Library:
    def __init__(self, path: str) -> None:
        lib = ctypes.CDLL(path)
        dbl = ctypes.POINTER(ctypes.c_double)
        lib.fl_create.restype = ctypes.c_void_p
        lib.fl_create.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            dbl,
            dbl,
            dbl,
            dbl,
            dbl,
            ctypes.c_int,
        ]
        lib.fl_solve.argtypes = [ctypes.c_void_p]
        lib.fl_get_solution.argtypes = [ctypes.c_void_p, dbl, dbl]
        lib.fl_get_stats.argtypes = [ctypes.c_void_p, dbl]
        lib.fl_set_option_double.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        lib.fl_set_option_int.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.fl_destroy.argtypes = [ctypes.c_void_p]
        self.lib = lib


def _as(a: np.ndarray, ctype: type) -> "ctypes._Pointer":
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def prepare(problem: StageProblem) -> Tuple[List[_FatropType], CompiledLibrary]:
    """Generate and compile the stage types (cached). Returns the types and the library."""
    types = _fatrop_types(problem)
    glue = resources.files("fast_lto.optimization").joinpath("fatrop_glue.c").read_text()
    functions = [f for t in types for slot in _FUNCTION_SLOTS if (f := t.functions.get(slot))]
    lib = compile_library(
        "fast_lto_fatrop",
        functions,
        extra_sources={"fl_table.c": _type_table(types), "fl_glue.c": glue},
        link_fatrop=True,
    )
    return types, lib


def solve(
    problem: StageProblem,
    X0: np.ndarray,
    U0: np.ndarray,
    options: Dict,
    verbose: bool = False,
) -> StageSolution:
    types, compiled = prepare(problem)
    lib = _Library(str(compiled.path))
    N, nx = problem.N, problem.nx
    st = np.ascontiguousarray(problem.stage_type, dtype=np.int32)
    params = np.ascontiguousarray(problem.params, dtype=np.float64).ravel()
    lb = np.concatenate([types[i].lb for i in st])
    ub = np.concatenate([types[i].ub for i in st])
    has_u = np.array([types[i].nu > 0 for i in st])
    copy = [X0[0]] if problem.periodic else []
    x_init = np.concatenate([np.concatenate([X0[k], *copy]) for k in range(N)])
    u_init = np.concatenate([U0[k] for k in range(N) if has_u[k]] or [np.zeros(0)])

    handle = lib.lib.fl_create(
        N,
        _as(st, ctypes.c_int),
        _as(params, ctypes.c_double),
        _as(lb, ctypes.c_double),
        _as(ub, ctypes.c_double),
        _as(x_init, ctypes.c_double),
        _as(u_init, ctypes.c_double),
        int(verbose),
    )
    try:
        for name, value in options.items():
            key = name.encode()
            if isinstance(value, int) and not isinstance(value, bool):
                lib.lib.fl_set_option_int(handle, key, value)
            else:
                lib.lib.fl_set_option_double(handle, key, float(value))
        t0 = time.perf_counter()
        flag = lib.lib.fl_solve(handle)
        elapsed = time.perf_counter() - t0
        x_all = np.zeros(x_init.size)
        u_all = np.zeros(max(u_init.size, 1))
        lib.lib.fl_get_solution(handle, _as(x_all, ctypes.c_double), _as(u_all, ctypes.c_double))
        stats = np.zeros(11)
        lib.lib.fl_get_stats(handle, _as(stats, ctypes.c_double))
    finally:
        lib.lib.fl_destroy(handle)

    nxa = types[0].nx
    X = x_all.reshape(N, nxa)[:, :nx]
    U = np.zeros((N, problem.nu))
    U[has_u] = u_all[: u_init.size].reshape(-1, problem.nu)
    success = flag == 0
    status = "Solve_Succeeded" if success else f"FATROP_return_flag_{flag}"
    return StageSolution(X, U, success, status, int(stats[9]), elapsed)
