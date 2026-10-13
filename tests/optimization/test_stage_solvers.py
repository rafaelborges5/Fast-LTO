"""The stage-wise solvers solve the same problem as the default IPOPT path.

``solver`` / ``derivatives`` only change how the OCP is solved. IPOPT with
compiled derivatives runs the same algorithm on the same maths, so it must land
on the reference trajectory. FATROP is a different algorithm on a nonconvex
problem and may find a better local optimum, so it must be feasible for the
same NLP and no worse. All with ``reg_u = 0`` (the input-rate penalty couples
stages, which the stage-wise solvers reject).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

import casadi as ca
import numpy as np
import pytest

from fast_lto.config import build_pipeline_config
from fast_lto.modes import get_mode
from fast_lto.optimization import stage_ipopt
from fast_lto.optimization.codegen import find_compiler
from fast_lto.optimization.global_ocp import load_track_with_widths
from fast_lto.optimization.integrators import EulerIntegrator
from fast_lto.optimization.stage_problem import build_stage_problem
from fast_lto.pipeline import PipelineConfig, _make_model, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    find_compiler()
    HAS_COMPILER = True
except RuntimeError:
    HAS_COMPILER = False

needs_compiler = pytest.mark.skipif(not HAS_COMPILER, reason="no C compiler")

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "trackdrive": {"track_id": "ellipse", "track_type": "ellipse", "mode": "trackdrive"},
    "autox": {
        "track_id": "ellipse",
        "track_type": "ellipse",
        "mode": "autox",
        "autox": {"extension_m": 10.0, "D_safe_braking": 1.0, "terminal_state_constraint": True},
    },
    "skidpad": {
        "track_id": "skidpad",
        "track_type": "skidpad",
        "mode": "skidpad",
        "skidpad": {
            "map_csv": str(REPO_ROOT / "data/tracks/skidpad/skidpad_map.csv"),
            "reference_csv": str(REPO_ROOT / "data/tracks/skidpad/skidpad_reference.csv"),
        },
    },
}

COMBOS = [("ipopt", "compiled"), ("fatrop", "compiled")]


def _config(tmp: Path, scenario: str, model: str, **overrides: Any) -> PipelineConfig:
    settings = dict(SCENARIOS[scenario])
    if "autox" in settings and model != "four_wheel":
        # The brake zone overrides per-wheel grip, which only four_wheel has.
        settings["autox"] = {**settings["autox"], "D_safe_braking": None}
    return build_pipeline_config(
        {
            **settings,
            "ds_m": 1.5,
            "integrator_name": "euler",
            "model_name": model,
            "reg_u": 0.0,
            "reg_u_l2": 1e-3,
            "repo_root": tmp,
            "generate_track": True,
            "warm_start": "off",
            "export_trajectory": False,
            "plot_results": False,
            "show_plots": False,
            **overrides,
        }
    )


def _solve(config: PipelineConfig, start_from: str = "track") -> Dict[str, Any]:
    with redirect_stdout(io.StringIO()):
        results = run_pipeline(config, start_from=start_from, end_at="ocp")  # type: ignore[arg-type]
    return json.loads(Path(results["ocp"]).read_text())


def _cases():
    for scenario in SCENARIOS:
        for model in ("point_mass", "four_wheel"):
            marks = [needs_compiler]
            if model == "four_wheel":
                marks.append(pytest.mark.slow)
            yield pytest.param(scenario, model, id=f"{scenario}-{model}", marks=marks)


def _max_violation(config: PipelineConfig, sol: Dict[str, Any]) -> float:
    """Worst constraint or bound violation of ``sol`` on the stage-wise NLP of ``config``."""
    mode = get_mode(config.mode)
    track = mode.prepare_track(load_track_with_widths(config.track_with_widths_path), config)
    model = _make_model(config.model_name)
    names = model.reduced_state_names()
    v_idx = names.index("v") if "v" in names else names.index("v_long")
    x0 = np.zeros(model.nx_reduced)
    x0[v_idx] = config.launch_speed
    kwargs = {k: v for k, v in mode.solver_kwargs(config).items() if k != "autox_timing_offset_m"}
    problem = build_stage_problem(
        track,
        model,
        EulerIntegrator(),
        mode=config.mode,
        boundary_margin=config.boundary_margin,
        time_weights=mode.time_weights(track, config),
        reg_u_l2=config.reg_u_l2,
        x0=np.asarray(model.reduced_state_phys_to_norm(x0)).reshape(-1),
        **kwargs,
    )
    nlp, _, bounds, layout = stage_ipopt.build_nlp(problem, compile=False)
    x_scale, x_shift = (np.asarray(a).reshape(-1) for a in model.get_reduced_state_scaling())
    u_scale, u_shift = (np.asarray(a).reshape(-1) for a in model.get_input_scaling())
    X = (np.column_stack([sol[n] for n in names]) - x_shift) / x_scale
    U = (np.column_stack([sol[n] for n in model.get_input_names()]) - u_shift) / u_scale
    w = np.zeros(layout.nw)
    for k in range(problem.N):
        w[layout.x_off[k] : layout.x_off[k] + problem.nx] = X[k]
        if layout.u_off[k] >= 0:
            w[layout.u_off[k] : layout.u_off[k] + problem.nu] = U[k]
    g = np.asarray(ca.Function("g", [nlp["x"], nlp["p"]], [nlp["g"]])(w, np.zeros(0))).ravel()
    viol_g = np.maximum(bounds["lbg"] - g, 0) + np.maximum(g - bounds["ubg"], 0)
    viol_x = np.maximum(bounds["lbx"] - w, 0) + np.maximum(w - bounds["ubx"], 0)
    return float(max(viol_g.max(), viol_x.max()))


@pytest.mark.parametrize("scenario,model", list(_cases()))
def test_solvers_agree(tmp_path: Path, scenario: str, model: str) -> None:
    reference = _solve(_config(tmp_path, scenario, model))
    assert reference["profiling"]["return_status"] == "Solve_Succeeded"
    for solver, derivatives in COMBOS:
        config = _config(tmp_path, scenario, model, solver=solver, derivatives=derivatives)
        sol = _solve(config, start_from="ocp")
        prof = sol["profiling"]
        assert prof["return_status"] == "Solve_Succeeded", (solver, derivatives)
        assert prof["solver"] == solver and prof["derivatives"] == derivatives
        if solver == "ipopt":
            np.testing.assert_allclose(sol["obj_val"], reference["obj_val"], rtol=1e-6)
            np.testing.assert_allclose(
                prof["lap_time_s"], reference["profiling"]["lap_time_s"], rtol=1e-6
            )
        else:
            assert sol["obj_val"] <= reference["obj_val"] * (1 + 1e-5)
            assert _max_violation(config, sol) < 1e-6


@pytest.mark.parametrize("model", ["point_mass", "dynamic_bicycle", "four_wheel"])
@pytest.mark.parametrize("mode", ["trackdrive", "autox"])
def test_assembled_derivatives_match_casadi(model: str, mode: str) -> None:
    """The permutation-assembled Jacobian and Hessian equal CasADi's own AD."""
    rng = np.random.default_rng(0)
    vehicle = _make_model(model)
    N = 12
    s = np.arange(N) * 1.5
    track = {
        "curvatures": (0.05 * np.sin(s / 5)).tolist(),
        "curvatures_half": (0.05 * np.sin((s + 0.75) / 5)).tolist(),
        "w_left": np.full(N, 2.0).tolist(),
        "w_right": np.full(N, 2.0).tolist(),
        "ds_m": 1.5,
        "timed_mask": [1.0] * (N - 4) + [0.0] * 4,
    }
    problem = build_stage_problem(
        track,
        vehicle,
        EulerIntegrator(),
        mode=mode,
        boundary_margin=0.2,
        time_weights=rng.uniform(0.5, 1.0, N),
        reg_u_l2=1e-3,
        x0=np.zeros(vehicle.nx_reduced),
        terminal_speed=None if mode == "trackdrive" else 2.0,
        terminal_state_constraint=mode == "autox",
        D_safe_braking=1.0 if (mode == "autox" and model == "four_wheel") else None,
    )
    nlp, custom, _, layout = stage_ipopt.build_nlp(problem, compile=False)
    x, p = nlp["x"], nlp["p"]
    w = 0.1 * rng.standard_normal(layout.nw)
    lam = rng.standard_normal(layout.ng)
    none = np.zeros(0)

    def value(expr: ca.MX) -> np.ndarray:
        return np.asarray(ca.Function("f", [x, p], [expr])(w, none))

    jac = np.asarray(custom["jac_g"](w, none)[1])
    np.testing.assert_allclose(jac, value(ca.jacobian(nlp["g"], x)), atol=1e-10)
    grad = np.asarray(custom["grad_f"](w, none)[1])
    np.testing.assert_allclose(grad, value(ca.gradient(nlp["f"], x)), atol=1e-10)
    lag = 0.7 * nlp["f"] + ca.dot(ca.DM(lam), nlp["g"])
    hess = np.asarray(custom["hess_lag"](w, none, 0.7, lam))
    np.testing.assert_allclose(hess, value(ca.triu(ca.hessian(lag, x)[0])), atol=1e-9)


def test_fatrop_needs_compiled_derivatives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fatrop needs derivatives: compiled"):
        _config(tmp_path, "trackdrive", "point_mass", solver="fatrop")


def test_compiled_rejects_input_rate_penalty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reg_u: 0"):
        _config(tmp_path, "trackdrive", "point_mass", derivatives="compiled", reg_u=1.0)
