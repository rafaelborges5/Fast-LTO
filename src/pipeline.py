"""
Pipeline orchestrator for Fast-LTO.

This module provides a structured way to run the full pipeline from track generation
to visualization, with the ability to start from any intermediate step.

Pipeline Steps
--------------
1. Track generation        -> data/tracks/{track_id}.csv
2. Spline fitting          -> data/discretized/{track_id}.json
3. Bounds computation      -> data/discretized/{track_id}_with_widths.json
4. OCP solving             -> data/solutions/{track_id}_{model_name}_{integrator_name}_{mode}.json
5. Trajectory export       -> data/output_trajectories/{track_id}_{model}_{integrator}_{timestamp}.csv
6. Visualization           -> ocp_plots/{timestamp}/panels.png

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
from vehicle_models import DynamicBicycleModel, PointMassModel, VehicleModel


StepName = Literal["track", "spline", "bounds", "ocp", "export", "plot"]


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

    use_savgol_bounds: bool = False
    savgol_window_length: int = 41
    savgol_polyorder: int = 2

    mode: Literal["autox", "trackdrive"] = "trackdrive"

    model_name: str = "point_mass"
    integrator_name: Literal["euler", "rk4"] = "euler"
    # Input rate-regularization weight on changes in inputs (du).
    reg_u: float = 600.0
    initial_speed: Optional[float] = None  # None → mode default (3.0 autox, 5.0 trackdrive)
    boundary_margin: float = 0.0  # Shrink lateral bounds by this amount (m) during optimization
    autox_extension_m: float = 50.0  # Extra track beyond finish line for autox mode

    export_trajectory: bool = True

    plot_results: bool = True
    show_plots: bool = True  # Whether to display plots interactively
    normalize_states_and_inputs: bool = True
    solver_verbose: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("autox", "trackdrive"):
            raise ValueError(f"Unknown mode: {self.mode!r}. Must be 'autox' or 'trackdrive'.")

        if self.initial_speed is None:
            self.initial_speed = 3.0 if self.mode == "autox" else 5.0

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
        self.output_trajectories_dir = self.repo_root / "data" / "output_trajectories"
        self.plots_dir = self.repo_root / "ocp_plots"

    @property
    def discretized_track_path(self) -> Path:
        return self.discretized_dir / f"{self.track_id}.json"

    @property
    def track_with_widths_path(self) -> Path:
        return self.discretized_dir / f"{self.track_id}_with_widths.json"

    @property
    def solution_path(self) -> Path:
        return self.solutions_dir / f"{self.track_id}_{self.model_name}_{self.integrator_name}_{self.mode}.json"

    @property
    def export_trajectory_path(self) -> Path:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.output_trajectories_dir / f"{self.track_id}_{self.model_name}_{self.integrator_name}_{ts}.csv"


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
        "bounds_method": "kdtree_spline",
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
    if model_name == "dynamic_bicycle":
        return DynamicBicycleModel()
    raise ValueError(f"Unknown model_name: {model_name!r}")


def _make_integrator(name: Literal["euler", "rk4"]) -> SpaceIntegrator:
    if name == "euler":
        return EulerIntegrator()
    if name == "rk4":
        return RK4Integrator()
    raise ValueError(f"Unknown integrator_name: {name!r}")


def _extend_track_for_autox(track_data: Dict, extension_m: float) -> Dict:
    """Extend a closed-loop track by wrapping points beyond the finish line."""
    ds_m = float(track_data["ds_m"])
    N_orig = len(track_data["arc_lengths"])
    M_pts = min(max(1, round(extension_m / ds_m)), N_orig - 1)

    positions = np.array(track_data["positions"], dtype=np.float64)
    headings = np.array(track_data["headings"], dtype=np.float64)
    curvatures = np.array(track_data["curvatures"], dtype=np.float64)
    curvatures_half = np.array(track_data["curvatures_half"], dtype=np.float64)
    arc_lengths = np.array(track_data["arc_lengths"], dtype=np.float64)
    w_left = np.array(track_data["w_left"], dtype=np.float64)
    w_right = np.array(track_data["w_right"], dtype=np.float64)

    total_length = arc_lengths[-1] + ds_m

    extended = dict(track_data)
    extended["positions"] = np.concatenate([positions, positions[:M_pts]], axis=0).tolist()
    extended["headings"] = np.concatenate([headings, headings[:M_pts]]).tolist()
    extended["curvatures"] = np.concatenate([curvatures, curvatures[:M_pts]]).tolist()
    extended["curvatures_half"] = np.concatenate([curvatures_half, curvatures_half[:M_pts]]).tolist()
    extended["arc_lengths"] = np.concatenate([arc_lengths, arc_lengths[:M_pts] + total_length]).tolist()
    extended["w_left"] = np.concatenate([w_left, w_left[:M_pts]]).tolist()
    extended["w_right"] = np.concatenate([w_right, w_right[:M_pts]]).tolist()
    extended["num_points"] = N_orig + M_pts
    extended["total_length_m"] = float(arc_lengths[-1] + ds_m * M_pts + ds_m)

    return extended


def step_solve_ocp(
    config: PipelineConfig,
    track_with_widths_path: Optional[Path] = None,
) -> Path:
    if track_with_widths_path is None:
        track_with_widths_path = config.track_with_widths_path

    print("[Step 4] OCP solving")
    print(f"  Track with widths: {track_with_widths_path}")

    track_data: Dict = load_track_with_widths(track_with_widths_path)

    if config.mode == "autox":
        track_data = _extend_track_for_autox(track_data, config.autox_extension_m)
        print(f"  Autox: extended track by {config.autox_extension_m:.0f} m "
              f"({track_data['num_points']} points total)")

    model = _make_model(config.model_name)
    integrator = _make_integrator(config.integrator_name)

    print(f"  Mode: {config.mode}")
    print(f"  Model: {config.model_name}")
    print(f"  Integrator: {config.integrator_name}")
    print(f"  Input rate regularization (reg_u): {config.reg_u}")

    config.solutions_dir.mkdir(parents=True, exist_ok=True)
    solution_path = config.solution_path

    # Define the OCP run configuration that uniquely characterises a solution.
    track_ds_m = float(
        track_data.get(
            "ds_m",
            (
                track_data["arc_lengths"][1] - track_data["arc_lengths"][0]
                if len(track_data.get("arc_lengths", [])) > 1
                else config.ds_m
            ),
        )
    )
    track_num_points = int(track_data.get("num_points", len(track_data.get("arc_lengths", []))))

    # Note: reg_u may be a scalar or a sequence (for per-input weights).
    if isinstance(config.reg_u, (list, tuple, np.ndarray)):
        reg_du_for_sig = [float(v) for v in config.reg_u]
    else:
        reg_du_for_sig = float(config.reg_u)

    run_config = {
        "track_id": config.track_id,
        "mode": config.mode,
        "model_name": config.model_name,
        "ds_m": float(track_ds_m),
        "num_points": int(track_num_points),
        "continuity": str(config.continuity),
        "integrator_name": config.integrator_name,
        "reg_du": reg_du_for_sig,
        "initial_speed": float(config.initial_speed),
        "normalize_states_and_inputs": bool(config.normalize_states_and_inputs),
        "solver_verbose": bool(config.solver_verbose),
        "use_savgol_bounds": bool(config.use_savgol_bounds),
        "savgol_window_length": int(config.savgol_window_length),
        "savgol_polyorder": int(config.savgol_polyorder),
        "boundary_margin": float(config.boundary_margin),
    }

    sol_dict = solve_ocp_and_save(
        track=track_data,
        model=model,
        solution_path=solution_path,
        integrator=integrator,
        initial_speed=config.initial_speed,
        reg_du=config.reg_u,
        run_config=run_config,
        use_normalization=config.normalize_states_and_inputs,
        solver_verbose=config.solver_verbose,
        boundary_margin=config.boundary_margin,
        mode=config.mode,
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


def step_export_trajectory(
    config: PipelineConfig,
    solution_path: Optional[Path] = None,
) -> Path:
    from export.trajectory import export_reference_trajectory

    if solution_path is None:
        solution_path = config.solution_path

    output_path = config.export_trajectory_path

    print("[Step 5] Exporting reference trajectory CSV")
    print(f"  Solution: {solution_path}")
    print(f"  Output:   {output_path}")

    export_reference_trajectory(solution_path, output_path)

    print(f"  Exported {output_path.name}")
    return output_path


def step_visualize(
    config: PipelineConfig,
    solution_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> Path:
    from datetime import datetime

    if solution_path is None:
        solution_path = config.solution_path
    if csv_path is None:
        csv_path = config.track_csv_path
    assert csv_path is not None

    print("[Step 6] Visualization")
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
    params = data.get("model_params", {})
    profiling = data.get("profiling")

    timestamp_dir = config.plots_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    plot_path = timestamp_dir / "panels.png"
    if config.model_name == "point_mass":
        from visualization.ocp_plots import plot_all_panels, _compute_constraint_activity

        a_long = np.array(data["a_long"], dtype=np.float64)
        a_lat = np.array(data["a_lat"], dtype=np.float64)

        mu = params.get("mu", 1.2)
        g_val = params.get("g", 9.81)
        mu_g = mu * g_val

        constraint_activity = _compute_constraint_activity(
            d=d,
            w_left=w_left,
            w_right=w_right,
            a_long=a_long,
            a_lat=a_lat,
            v=v,
            params=params,
        )

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
    elif config.model_name == "dynamic_bicycle":
        from visualization.ocp_plots_dynamic_bicycle import plot_all_panels_dynamic_bicycle

        a_long = np.array(data["a_long"], dtype=np.float64)
        delta = np.array(data["delta"], dtype=np.float64)
        v_lat = np.array(data["v_lat"], dtype=np.float64)
        yaw_rate = np.array(data["yaw_rate"], dtype=np.float64)

        plot_all_panels_dynamic_bicycle(
            cones_left,
            cones_right,
            path_xy,
            v,
            s,
            d,
            w_left,
            w_right,
            a_long,
            delta,
            v_lat,
            yaw_rate,
            params=params,
            profiling=profiling,
            out_path=plot_path,
            show=config.show_plots,
        )
    else:
        raise ValueError(
            f"No visualization available for model_name={config.model_name!r}. "
            "Supported: 'point_mass', 'dynamic_bicycle'."
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

        try:
            track_for_sig = load_track_with_widths(config.track_with_widths_path)
            track_ds_m = float(
                track_for_sig.get(
                    "ds_m",
                    (
                        track_for_sig["arc_lengths"][1]
                        - track_for_sig["arc_lengths"][0]
                        if len(track_for_sig.get("arc_lengths", [])) > 1
                        else config.ds_m
                    ),
                )
            )
            track_num_points = int(track_for_sig.get("num_points", len(track_for_sig.get("arc_lengths", []))))
        except Exception:
            track_ds_m = float(config.ds_m)
            track_num_points = -1

        if start_from == "ocp" or track_just_generated or not solution_path.exists():
            need_solve = True
        else:
            # Build current run signature.
            if isinstance(config.reg_u, (list, tuple, np.ndarray)):
                reg_du_sig = [float(v) for v in config.reg_u]
            else:
                reg_du_sig = float(config.reg_u)

            current_sig = {
                "track_id": config.track_id,
                "mode": config.mode,
                "model_name": config.model_name,
                "ds_m": float(track_ds_m),
                "num_points": int(track_num_points),
                "continuity": str(config.continuity),
                "integrator_name": config.integrator_name,
                "reg_du": reg_du_sig,
                "initial_speed": float(config.initial_speed),
                "use_savgol_bounds": bool(config.use_savgol_bounds),
                "savgol_window_length": int(config.savgol_window_length),
                "savgol_polyorder": int(config.savgol_polyorder),
                "boundary_margin": float(config.boundary_margin),
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

    # Trajectory export.
    if config.export_trajectory:
        export_path = step_export_trajectory(config, solution_path=results["ocp"])
        results["export"] = export_path

    if end_at == "export":
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
    "step_export_trajectory",
    "step_visualize",
    "run_pipeline",
]

