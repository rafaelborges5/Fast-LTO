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
from vehicle_models import DynamicBicycleModel, FourWheelModel, PointMassModel, VehicleModel


StepName = Literal["track", "spline", "bounds", "ocp", "export", "plot"]


@dataclass
class PipelineConfig:
    """
    Configuration for the Fast-LTO pipeline.

    Parameters are grouped roughly by pipeline step; most have sensible defaults.
    """

    track_id: str = "fsg_random"
    track_type: Literal["fsg", "ellipse", "bean", "skidpad"] = "fsg"

    repo_root: Optional[Path] = None
    track_csv_path: Optional[Path] = None

    generate_track: bool = False

    ds_m: float = 0.5
    continuity: ContinuityType = "C2"
    smooth_centerline: int = 0

    compute_bounds: bool = True

    use_savgol_bounds: bool = False
    savgol_window_length: int = 41
    savgol_polyorder: int = 2

    mode: Literal["autox", "trackdrive", "skidpad"] = "trackdrive"

    model_name: str = "point_mass"
    integrator_name: Literal["euler", "rk4"] = "euler"
    reg_u: float = 600.0
    reg_u_l2: float | None = None
    initial_speed: Optional[float] = None
    boundary_margin: float = 0.0
    autox_extension_m: float = 50.0
    autox_lead_in_m: float = 0.0

    skidpad_map_csv: Optional[str] = None
    skidpad_reference_csv: Optional[str] = None
    eps_time: float = 0.1
    entry_exit_halfwidth: float = 1.5
    kappa_blend_m: float = 1.5
    terminal_speed: Optional[float] = None

    export_trajectory: bool = True

    plot_results: bool = True
    show_plots: bool = True
    normalize_states_and_inputs: bool = True
    solver_verbose: bool = False

    vehicle_config: Optional[object] = None

    def __post_init__(self) -> None:
        if self.mode not in ("autox", "trackdrive", "skidpad"):
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Must be 'autox', 'trackdrive' or 'skidpad'."
            )

        if self.initial_speed is None:
            self.initial_speed = 5.0 if self.mode == "trackdrive" else 3.0

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
        smooth_centerline=config.smooth_centerline,
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


def _make_model(model_name: str, vehicle_config=None) -> VehicleModel:
    if vehicle_config is not None:
        params = vehicle_config.build_model_params(model_name)
    else:
        params = None
    if model_name == "point_mass":
        return PointMassModel(params=params)
    if model_name == "dynamic_bicycle":
        return DynamicBicycleModel(params=params)
    if model_name == "four_wheel":
        return FourWheelModel(params=params)
    raise ValueError(f"Unknown model_name: {model_name!r}")


def _make_integrator(name: Literal["euler", "rk4"]) -> SpaceIntegrator:
    if name == "euler":
        return EulerIntegrator()
    if name == "rk4":
        return RK4Integrator()
    raise ValueError(f"Unknown integrator_name: {name!r}")


def _extend_track_for_autox(
    track_data: Dict, extension_m: float, lead_in_m: float = 0.0
) -> Dict:
    """Extend a closed-loop track by wrapping points beyond the finish line.

    The OCP horizon itself (``positions``/``headings``/... fed to
    ``build_ocp``) only ever gets the forward ``extension_m`` run-off — it is
    unaffected by ``lead_in_m`` and reproduces the original single-lap autox
    solve exactly. If ``lead_in_m`` > 0, the geometry for a lead-in stretch
    *before* the start/finish line is also computed (borrowed from the tail
    of the same closed loop, since the point just "before" s=0 on a closed
    track is, geometrically, the end of the loop) and returned under
    ``"autox_lead_in"``. It is not part of the optimization: see
    ``_prepend_autox_lead_in``, which stitches it onto the solved trajectory
    afterwards as a prescribed, constant-velocity segment. Solving for it
    jointly with the OCP would force models with rate-limited actuator states
    (e.g. four_wheel's tire forces/steering) to hit an exact speed target
    while ramping those actuators up from a standing start on a coarse mesh,
    which can make the problem infeasible.
    """
    ds_m = float(track_data["ds_m"])
    N_orig = len(track_data["arc_lengths"])
    M_pts = min(max(1, round(extension_m / ds_m)), N_orig - 1)
    K_pts = min(max(0, round(lead_in_m / ds_m)), N_orig - 1)

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

    if K_pts > 0:
        extended["autox_lead_in"] = {
            "positions": positions[-K_pts:].tolist(),
            "headings": headings[-K_pts:].tolist(),
            "curvatures": curvatures[-K_pts:].tolist(),
            "arc_lengths": (arc_lengths[-K_pts:] - total_length).tolist(),
            "w_left": w_left[-K_pts:].tolist(),
            "w_right": w_right[-K_pts:].tolist(),
        }

    return extended


def _prepend_autox_lead_in(sol_dict: Dict, lead_in: Dict, initial_speed: float) -> Dict:
    """Stitch a prescribed, constant-velocity lead-in onto a solved autox trajectory.

    The lead-in is not part of the OCP: it's a straight run along the
    centerline at exactly ``initial_speed``, giving the physical car a
    stretch of track before the true start/finish line. Lateral dynamics
    (yaw rate, tire forces, steering, ...) are simply held at the same
    zero/rest values as the pinned initial condition, since the trajectory
    tracker mostly consumes speed and lateral-deviation references.
    """
    K = len(lead_in["arc_lengths"])
    positions = lead_in["positions"]

    sol_dict["path_xy"] = positions + list(sol_dict["path_xy"])
    sol_dict["arc_lengths"] = lead_in["arc_lengths"] + list(sol_dict["arc_lengths"])
    sol_dict["w_left"] = lead_in["w_left"] + list(sol_dict["w_left"])
    sol_dict["w_right"] = lead_in["w_right"] + list(sol_dict["w_right"])
    sol_dict["kappa"] = lead_in["curvatures"] + list(sol_dict["kappa"])
    sol_dict["headings"] = lead_in["headings"] + list(sol_dict["headings"])

    for name in sol_dict["state_names"]:
        fill = float(initial_speed) if name in ("v", "v_long") else 0.0
        sol_dict[name] = [fill] * K + list(sol_dict[name])

    for name in sol_dict["input_names"]:
        sol_dict[name] = [0.0] * K + list(sol_dict[name])

    return sol_dict


def _resolve_path(root: Optional[Path], p: str | Path) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    path = Path(p)
    if path.is_absolute() or root is None:
        return path
    return root / path


def step_build_skidpad_track(config: PipelineConfig) -> Path:
    """Build the skidpad track-with-widths JSON from the cone map + reference."""
    from tracks.skidpad import build_skidpad_track

    if config.skidpad_map_csv is None or config.skidpad_reference_csv is None:
        raise ValueError(
            "mode='skidpad' requires skidpad_map_csv and skidpad_reference_csv."
        )

    map_csv = _resolve_path(config.repo_root, config.skidpad_map_csv)
    ref_csv = _resolve_path(config.repo_root, config.skidpad_reference_csv)

    print("[Skidpad] Building track from cone map + reference trajectory")
    print(f"  Map:       {map_csv}")
    print(f"  Reference: {ref_csv}")

    track = build_skidpad_track(
        map_csv=map_csv,
        ref_csv=ref_csv,
        ds_m=config.ds_m,
        entry_exit_halfwidth=config.entry_exit_halfwidth,
        kappa_blend_m=config.kappa_blend_m,
    )

    config.discretized_dir.mkdir(parents=True, exist_ok=True)
    with config.track_with_widths_path.open("w") as f:
        json.dump(track, f, indent=2)

    sk = track["skidpad"]
    timed = int(np.sum(track["timed_mask"]))
    print(
        f"  N={track['num_points']} total={track['total_length_m']:.1f} m "
        f"R_c={sk['R_c']:.2f} R_in={sk['R_in']:.2f} R_out={sk['R_out']:.2f} "
        f"timed_nodes={timed}"
    )
    print(f"  Saved track with widths to: {config.track_with_widths_path}")
    return config.track_with_widths_path


def _run_skidpad_pipeline(
    config: PipelineConfig, end_at: Optional[StepName]
) -> Dict[str, Path]:
    """Dedicated skidpad flow: build track -> OCP -> export -> visualize."""
    results: Dict[str, Path] = {}

    results["bounds"] = step_build_skidpad_track(config)
    if end_at == "bounds":
        return results

    solution_path = step_solve_ocp(config)
    results["ocp"] = solution_path
    if end_at == "ocp":
        return results

    if config.export_trajectory:
        results["export"] = step_export_trajectory(config, solution_path=solution_path)
    if end_at == "export":
        return results

    if config.plot_results:
        map_csv = _resolve_path(config.repo_root, config.skidpad_map_csv)
        results["plot"] = step_visualize(
            config, solution_path=solution_path, csv_path=map_csv
        )

    return results


def step_solve_ocp(
    config: PipelineConfig,
    track_with_widths_path: Optional[Path] = None,
) -> Path:
    if track_with_widths_path is None:
        track_with_widths_path = config.track_with_widths_path

    print("[Step 4] OCP solving")
    print(f"  Track with widths: {track_with_widths_path}")

    track_data: Dict = load_track_with_widths(track_with_widths_path)

    time_weights = None
    if config.mode == "skidpad":
        mask = np.asarray(track_data["timed_mask"], dtype=float)
        decel = np.asarray(track_data.get("decel_mask", np.zeros_like(mask)), dtype=float)
        time_weights = np.where(mask > 0.5, 1.0, float(config.eps_time))
        time_weights = np.where(decel > 0.5, 0.0, time_weights)
        print(f"  Skidpad: {int(mask.sum())}/{len(mask)} timed nodes, "
              f"{int(decel.sum())} exit (decel) nodes, "
              f"un-timed weight eps_time={config.eps_time}, "
              f"terminal_speed={config.terminal_speed}")
    elif config.mode == "autox":
        track_data = _extend_track_for_autox(
            track_data, config.autox_extension_m, config.autox_lead_in_m
        )
        print(f"  Autox: extended track by {config.autox_extension_m:.0f} m "
              f"({track_data['num_points']} points total, OCP horizon)")

    model = _make_model(config.model_name, vehicle_config=config.vehicle_config)
    integrator = _make_integrator(config.integrator_name)

    print(f"  Mode: {config.mode}")
    print(f"  Model: {config.model_name}")
    print(f"  Integrator: {config.integrator_name}")
    print(f"  Input rate regularization (reg_u): {config.reg_u}")
    print(f"  Input L2 regularization (reg_u_l2): {config.reg_u_l2}")

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
        reg_u_l2=config.reg_u_l2,
        run_config=run_config,
        use_normalization=config.normalize_states_and_inputs,
        solver_verbose=config.solver_verbose,
        boundary_margin=config.boundary_margin,
        mode=config.mode,
        time_weights=time_weights,
        terminal_speed=config.terminal_speed if config.mode == "skidpad" else None,
    )

    autox_lead_in = track_data.get("autox_lead_in") if config.mode == "autox" else None
    if autox_lead_in:
        sol_dict = _prepend_autox_lead_in(sol_dict, autox_lead_in, config.initial_speed)
        with solution_path.open("w") as f:
            json.dump(sol_dict, f, indent=2)
        print(
            f"  Autox: prepended {config.autox_lead_in_m:.0f} m constant-speed "
            f"lead-in ({len(autox_lead_in['arc_lengths'])} points, not part of the OCP solve)"
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

    if config.mode == "skidpad":
        from datetime import datetime
        from tracks.skidpad import load_skidpad_cones
        from visualization.skidpad_plots import plot_skidpad

        cones = load_skidpad_cones(csv_path)
        timestamp_dir = config.plots_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        plot_path = timestamp_dir / "panels.png"
        summary = plot_skidpad(
            data, cones, out_path=plot_path, show=config.show_plots
        )
        laps = ", ".join(f"{t:.3f}s" for t in summary["timed_lap_times"])
        print(f"  Timed laps: {laps}  |  score (avg): {summary['score']:.3f} s")
        print(f"  Saved plots to: {timestamp_dir}")
        return timestamp_dir

    boundaries = load_boundaries(csv_path)
    cones_left = boundaries["left"]
    cones_right = boundaries["right"]

    path_xy = np.array(data["path_xy"], dtype=np.float64)
    v = np.array(data.get("v", data.get("v_long")), dtype=np.float64)
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
    elif config.model_name == "four_wheel":
        from visualization.ocp_plots_four_wheel import plot_all_panels_four_wheel

        Fx_fl = np.array(data["Fx_fl"], dtype=np.float64)
        Fx_fr = np.array(data["Fx_fr"], dtype=np.float64)
        Fx_rr = np.array(data["Fx_rr"], dtype=np.float64)
        Fx_rl = np.array(data["Fx_rl"], dtype=np.float64)
        delta = np.array(data["delta"], dtype=np.float64)
        v_lat = np.array(data["v_lat"], dtype=np.float64)
        yaw_rate = np.array(data["yaw_rate"], dtype=np.float64)

        input_names = data.get("input_names", [])
        input_data = {name: data[name] for name in input_names if name in data}

        plot_all_panels_four_wheel(
            cones_left, cones_right, path_xy, v, s, d,
            w_left, w_right,
            Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
            v_lat, yaw_rate,
            params=params, profiling=profiling,
            input_data=input_data,
            out_path=plot_path, show=config.show_plots,
        )
    else:
        raise ValueError(
            f"No visualization available for model_name={config.model_name!r}. "
            "Supported: 'point_mass', 'dynamic_bicycle', 'four_wheel'."
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

    # Skidpad uses a dedicated builder (overlapping path can't go through the
    # generic spline/bounds machinery); the OCP/export/plot steps are reused.
    if config.mode == "skidpad" or config.track_type == "skidpad":
        return _run_skidpad_pipeline(config, end_at)

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
        need_refit = (
            start_from == "spline"
            or track_just_generated
            or not config.discretized_track_path.exists()
        )
        if not need_refit:
            try:
                cached = DiscretizedTrack.load(config.discretized_track_path)
                if abs(cached.ds_m - config.ds_m) > 1e-6 or cached.continuity != config.continuity:
                    need_refit = True
            except Exception:
                need_refit = True

        if need_refit:
            track = step_fit_spline(config, csv_path=csv_path)
        else:
            print(
                f"[Step 2] Using existing discretized track: "
                f"{config.discretized_track_path}"
            )
            track = cached
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

