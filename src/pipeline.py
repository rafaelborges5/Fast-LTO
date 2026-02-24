"""
Pipeline orchestrator for Fast-LTO.

This module provides a structured way to run the full pipeline from track generation
to visualization, with the ability to start from any intermediate step.

Pipeline Steps
--------------
1. Track generation        -> data/tracks/{track_id}.csv
2. Spline fitting          -> data/discretized/{track_id}.json
3. Bounds computation      -> data/discretized/{track_id}_with_widths.json
4. OCP solving             -> data/solutions/{track_id}_{model_name}.json
5. Visualization           -> ocp_plots/{timestamp}/panels.png

Each step can be run independently, and intermediate results are saved to disk
for reuse in subsequent runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

from optimization.global_ocp import load_track_with_widths, solve_ocp_and_save
from optimization.integrators import EulerIntegrator, RK4Integrator, SpaceIntegrator
from splines.discretized_track import DiscretizedTrack
from splines.spline_fitter import ContinuityType, fit_and_discretize
from tracks.bean import generate_bean_track
from tracks.ellipse import generate_ellipse_track
from tracks.fsg_trackdrive import generate_fsg_track
from utils.track_bounds import (
    LateralBoundsResult,
    apply_savgol_to_widths,
    compute_lateral_bounds,
    load_boundaries,
    save_track_with_widths,
)
from vehicle_models import PointMassModel, VehicleModel


StepName = Literal["track", "spline", "bounds", "ocp", "plot"]


@dataclass
class PipelineConfig:
    """
    Configuration for the Fast-LTO pipeline.

    Parameters are grouped roughly by pipeline step; most have sensible defaults.
    """

    track_id: str = "fsg_random"  # used for file naming
    track_type: Literal["fsg", "ellipse", "bean"] = "fsg"

    repo_root: Optional[Path] = None
    track_csv_path: Optional[Path] = None  # auto-derived if None

    generate_track: bool = False  # if False and CSV exists, reuse existing

    ds_m: float = 0.5
    continuity: ContinuityType = "C2"

    compute_bounds: bool = True

    use_savgol_bounds: bool = True
    savgol_window_length: int = 41
    savgol_polyorder: int = 2

    model_name: str = "point_mass"
    integrator_name: Literal["euler", "rk4"] = "euler"
    reg_u: float = 1e-4
    initial_speed: float = 1.0  # Initial speed guess (m/s). Must be > 0 for numerical stability.

    plot_results: bool = True
    show_plots: bool = True  # Whether to display plots interactively

    def __post_init__(self) -> None:
        if self.repo_root is None:
            self.repo_root = Path(__file__).resolve().parent.parent
        else:
            self.repo_root = Path(self.repo_root)

        if self.track_csv_path is None:
            self.track_csv_path = (
                self.repo_root / "data" / "tracks" / f"{self.track_id}.csv"
            )
        else:
            self.track_csv_path = Path(self.track_csv_path)

        self.discretized_dir = self.repo_root / "data" / "discretized"
        self.solutions_dir = self.repo_root / "data" / "solutions"
        self.plots_dir = self.repo_root / "ocp_plots"

    @property
    def discretized_track_path(self) -> Path:
        return self.discretized_dir / f"{self.track_id}.json"

    @property
    def track_with_widths_path(self) -> Path:
        return self.discretized_dir / f"{self.track_id}_with_widths.json"

    @property
    def solution_path(self) -> Path:
        return self.solutions_dir / f"{self.track_id}_{self.model_name}.json"


def step_generate_track(config: PipelineConfig) -> Path:
    csv_path = config.track_csv_path
    assert csv_path is not None  # for type checkers

    print(f"[Step 1] Track generation")
    print(f"  Track type: {config.track_type}")
    print(f"  Output CSV: {csv_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if config.track_type == "fsg":
        generate_fsg_track(output_csv=csv_path)
    elif config.track_type == "ellipse":
        generate_ellipse_track(output_csv=csv_path)
    elif config.track_type == "bean":
        generate_bean_track(output_csv=csv_path)
    else:
        raise ValueError(f"Unknown track_type: {config.track_type}")

    print("  Track CSV generated.")
    return csv_path


def step_fit_spline(
    config: PipelineConfig,
    csv_path: Optional[Path] = None,
) -> DiscretizedTrack:
    if csv_path is None:
        csv_path = config.track_csv_path
    assert csv_path is not None

    print(f"[Step 2] Spline fitting and discretization")
    print(f"  Input CSV: {csv_path}")
    print(f"  ds: {config.ds_m} m, continuity: {config.continuity}")

    config.discretized_dir.mkdir(parents=True, exist_ok=True)

    track = fit_and_discretize(
        csv_path=csv_path,
        ds_m=config.ds_m,
        continuity=config.continuity,
        viz=False,
        save_path=config.discretized_track_path,
    )

    print(f"  Discretized track: {track}")
    print(f"  Saved to: {config.discretized_track_path}")
    return track


def step_compute_bounds(
    config: PipelineConfig,
    track: Optional[DiscretizedTrack] = None,
    csv_path: Optional[Path] = None,
) -> LateralBoundsResult:
    if csv_path is None:
        csv_path = config.track_csv_path
    assert csv_path is not None

    if track is None:
        print(f"[Step 3] Loading discretized track from {config.discretized_track_path}")
        track = DiscretizedTrack.load(config.discretized_track_path)
    else:
        print("[Step 3] Computing lateral bounds")

    print(f"  Boundaries CSV: {csv_path}")
    boundaries = load_boundaries(csv_path)
    left = boundaries["left"]
    right = boundaries["right"]

    result = compute_lateral_bounds(track, left=left, right=right)
    
    if config.use_savgol_bounds:
        w_left_s, w_right_s = apply_savgol_to_widths(
            result.w_left,
            result.w_right,
            window_length=config.savgol_window_length,
            polyorder=config.savgol_polyorder,
        )
        result.w_left = w_left_s
        result.w_right = w_right_s

    print(
        f"  Computed widths: misses left/right: "
        f"{result.misses_left}/{result.misses_right}"
    )

    config.discretized_dir.mkdir(parents=True, exist_ok=True)
    bounds_config = {
        "use_savgol_bounds": bool(config.use_savgol_bounds),
        "savgol_window_length": int(config.savgol_window_length),
        "savgol_polyorder": int(config.savgol_polyorder),
    }
    save_track_with_widths(
        config.track_with_widths_path,
        track=track,
        csv_source=csv_path,
        result=result,
        bounds_config=bounds_config,
    )
    print(f"  Saved track with widths to: {config.track_with_widths_path}")

    return result


def _make_model(model_name: str) -> VehicleModel:
    if model_name == "point_mass":
        return PointMassModel()
    raise ValueError(f"Unknown model_name: {model_name!r}")


def _make_integrator(name: Literal["euler", "rk4"]) -> SpaceIntegrator:
    if name == "euler":
        return EulerIntegrator()
    if name == "rk4":
        return RK4Integrator()
    raise ValueError(f"Unknown integrator_name: {name!r}")


def step_solve_ocp(
    config: PipelineConfig,
    track_with_widths_path: Optional[Path] = None,
) -> Path:
    if track_with_widths_path is None:
        track_with_widths_path = config.track_with_widths_path

    print("[Step 4] OCP solving")
    print(f"  Track with widths: {track_with_widths_path}")

    track_data: Dict = load_track_with_widths(track_with_widths_path)

    model = _make_model(config.model_name)
    integrator = _make_integrator(config.integrator_name)

    print(f"  Model: {config.model_name}")
    print(f"  Integrator: {config.integrator_name}")
    print(f"  Input regularization: {config.reg_u}")

    config.solutions_dir.mkdir(parents=True, exist_ok=True)
    solution_path = config.solution_path

    # Define the OCP run configuration that uniquely characterises a solution.
    run_config = {
        "track_id": config.track_id,
        "model_name": config.model_name,
        "ds_m": float(config.ds_m),
        "continuity": str(config.continuity),
        "integrator_name": config.integrator_name,
        "reg_u": float(config.reg_u),
        "initial_speed": float(config.initial_speed),
        "use_savgol_bounds": bool(config.use_savgol_bounds),
        "savgol_window_length": int(config.savgol_window_length),
        "savgol_polyorder": int(config.savgol_polyorder),
    }

    sol_dict = solve_ocp_and_save(
        track=track_data,
        model=model,
        solution_path=solution_path,
        integrator=integrator,
        initial_speed=config.initial_speed,
        reg_u=config.reg_u,
        run_config=run_config,
    )

    # Optional concise profiling summary (single line)
    profiling = sol_dict.get("profiling", {})
    N = profiling.get("N")
    ds_m = profiling.get("ds_m")
    solve_time_s = profiling.get("solve_time_s")
    iter_count = profiling.get("iter_count")
    return_status = profiling.get("return_status")
    if solve_time_s is not None and N is not None:
        time_per_point_ms = profiling.get("time_per_point_ms", solve_time_s / N * 1e3)
        print(
            f"[OCP profiling] N={N}, ds={ds_m:.3f} m, "
            f"time={solve_time_s:.3f} s, "
            f"time/N={time_per_point_ms:.3f} ms, "
            f"iters={iter_count if iter_count is not None else 'N/A'}, "
            f"status={return_status}"
        )

    return solution_path


def step_visualize(
    config: PipelineConfig,
    solution_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> Path:
    from datetime import datetime

    from visualization.ocp_plots import plot_all_panels, _compute_constraint_activity

    if solution_path is None:
        solution_path = config.solution_path
    if csv_path is None:
        csv_path = config.track_csv_path
    assert csv_path is not None

    print("[Step 5] Visualization")
    print(f"  Solution: {solution_path}")
    print(f"  Boundaries CSV: {csv_path}")

    with solution_path.open("r") as f:
        data = json.load(f)

    boundaries = load_boundaries(csv_path)
    cones_left = boundaries["left"]
    cones_right = boundaries["right"]

    path_xy = np.array(data["path_xy"], dtype=np.float64)
    v = np.array(data["v"], dtype=np.float64)
    s = np.array(data["arc_lengths"], dtype=np.float64)
    d = np.array(data["d"], dtype=np.float64)
    w_left = np.array(data["w_left"], dtype=np.float64)
    w_right = np.array(data["w_right"], dtype=np.float64)
    a_long = np.array(data["a_long"], dtype=np.float64)
    a_lat = np.array(data["a_lat"], dtype=np.float64)
    params = data.get("model_params", {})
    mu = params.get("mu", 1.2)
    g_val = params.get("g", 9.81)
    mu_g = mu * g_val

    profiling = data.get("profiling")
    constraint_activity = _compute_constraint_activity(
        d=d,
        w_left=w_left,
        w_right=w_right,
        a_long=a_long,
        a_lat=a_lat,
        v=v,
        params=params,
    )

    timestamp_dir = config.plots_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    plot_path = timestamp_dir / "panels.png"
    plot_all_panels(
        cones_left,
        cones_right,
        path_xy,
        v,
        s,
        d,
        w_left,
        w_right,
        a_long,
        a_lat,
        mu_g,
        profiling=profiling,
        constraint_activity=constraint_activity,
        out_path=plot_path,
        show=config.show_plots,
    )

    print(f"  Saved plots to: {timestamp_dir}")
    return timestamp_dir


def run_pipeline(
    config: PipelineConfig,
    start_from: StepName = "track",
    end_at: Optional[StepName] = None,
) -> Dict[str, Path]:
    results: Dict[str, Path] = {}

    if end_at is None:
        end_at = "plot"

    # Track whether we just generated a new track CSV
    track_just_generated = False

    if start_from == "track":
        need_generate = (
            config.generate_track
            or config.track_csv_path is None
            or not config.track_csv_path.exists()
        )
        if need_generate:
            csv_path = step_generate_track(config)
            track_just_generated = True
        else:
            csv_path = config.track_csv_path
            print(f"[Step 1] Using existing track CSV: {csv_path}")
        results["track"] = csv_path
    else:
        csv_path = config.track_csv_path
        if csv_path is None or not csv_path.exists():
            raise FileNotFoundError(
                f"Track CSV not found at {csv_path}. "
                f"Run with start_from='track' first."
            )
        results["track"] = csv_path

    if end_at == "track":
        return results

    # If track was just generated, force recomputation of downstream steps
    if start_from in ("track", "spline"):
        if start_from == "spline" or track_just_generated or not config.discretized_track_path.exists():
            track = step_fit_spline(config, csv_path=csv_path)
        else:
            print(
                f"[Step 2] Using existing discretized track: "
                f"{config.discretized_track_path}"
            )
            track = DiscretizedTrack.load(config.discretized_track_path)
        results["spline"] = config.discretized_track_path
    else:
        if not config.discretized_track_path.exists():
            raise FileNotFoundError(
                f"Discretized track not found at {config.discretized_track_path}. "
                f"Run with start_from='spline' first."
            )
        results["spline"] = config.discretized_track_path
        track = None

    if end_at == "spline":
        return results

    if start_from in ("track", "spline", "bounds"):
        if config.compute_bounds:
            if track is not None:
                result = step_compute_bounds(
                    config,
                    track=track,
                    csv_path=csv_path,
                )
                _ = result  # currently unused
            else:
                need_bounds = False
                if (
                    start_from == "bounds"
                    or track_just_generated
                    or not config.track_with_widths_path.exists()
                ):
                    need_bounds = True
                else:
                    try:
                        with config.track_with_widths_path.open("r") as f:
                            existing_bounds = json.load(f)
                        stored_cfg = existing_bounds.get("bounds_config")
                    except Exception:
                        stored_cfg = None

                    current_cfg = {
                        "use_savgol_bounds": bool(config.use_savgol_bounds),
                        "savgol_window_length": int(config.savgol_window_length),
                        "savgol_polyorder": int(config.savgol_polyorder),
                    }
                    if stored_cfg != current_cfg:
                        need_bounds = True

                if need_bounds:
                    result = step_compute_bounds(
                        config,
                        track=None,
                        csv_path=csv_path,
                    )
                    _ = result  # currently unused
                else:
                    print(
                        f"[Step 3] Using existing track with widths: "
                        f"{config.track_with_widths_path}"
                    )
            results["bounds"] = config.track_with_widths_path
    else:
        if not config.track_with_widths_path.exists():
            raise FileNotFoundError(
                f"Track with widths not found at {config.track_with_widths_path}. "
                f"Run with start_from='bounds' first."
            )
        results["bounds"] = config.track_with_widths_path

    if end_at == "bounds":
        return results

    if start_from in ("track", "spline", "bounds", "ocp"):
        solution_path = config.solution_path

        # Decide whether we can safely reuse an existing solution or must re-solve.
        need_solve = False

        if start_from == "ocp" or track_just_generated or not solution_path.exists():
            need_solve = True
        else:
            # Build current run signature.
            current_sig = {
                "track_id": config.track_id,
                "model_name": config.model_name,
                "ds_m": float(config.ds_m),
                "continuity": str(config.continuity),
                "integrator_name": config.integrator_name,
                "reg_u": float(config.reg_u),
                "initial_speed": float(config.initial_speed),
                "use_savgol_bounds": bool(config.use_savgol_bounds),
                "savgol_window_length": int(config.savgol_window_length),
                "savgol_polyorder": int(config.savgol_polyorder),
            }

            # Load stored signature from existing solution, if any.
            try:
                with solution_path.open("r") as f:
                    existing_data = json.load(f)
                stored_sig = existing_data.get("run_config")
            except Exception:
                stored_sig = None

            if stored_sig != current_sig:
                need_solve = True

        if need_solve:
            solution_path = step_solve_ocp(config)
        else:
            print(f"[Step 4] Using existing solution: {solution_path}")

        results["ocp"] = solution_path
    else:
        if not config.solution_path.exists():
            raise FileNotFoundError(
                f"Solution not found at {config.solution_path}. "
                f"Run with start_from='ocp' first."
            )
        results["ocp"] = config.solution_path

    if end_at == "ocp":
        return results

    # Visualization.
    if config.plot_results:
        plot_dir = step_visualize(config, solution_path=results["ocp"], csv_path=csv_path)
        results["plot"] = plot_dir

    return results


__all__ = [
    "PipelineConfig",
    "StepName",
    "step_generate_track",
    "step_fit_spline",
    "step_compute_bounds",
    "step_solve_ocp",
    "step_visualize",
    "run_pipeline",
]

