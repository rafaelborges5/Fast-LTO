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
    # Metres before s=0 that the OCP's own optimized horizon begins (instead of
    # the flat autox_lead_in_m hold). The pinned launch condition moves back by
    # this much; the flat hold is trimmed to sit immediately before it.
    autox_ocp_lead_m: float = 0.0
    # Distance (m) from the car's start position to the real timing gate; the
    # accurate autox lap time is measured between this point and the same point
    # one lap later, not from s=0 through the run-off extension.
    autox_timing_offset_m: float = 6.0

    skidpad_map_csv: Optional[str] = None
    skidpad_reference_csv: Optional[str] = None
    eps_time: float = 0.1
    entry_exit_halfwidth: float = 1.5
    kappa_blend_m: float = 1.5
    # Overrides the entry point (P0), otherwise taken from the reference's first
    # row. skidpad_start_x=None keeps the original behaviour.
    skidpad_start_x: Optional[float] = None
    skidpad_start_y: float = 0.0
    # Metres of straight, prescribed constant-speed run-in prepended before the
    # OCP's s=0 (at initial_speed), not part of the optimization. 0.0 = off.
    skidpad_lead_in_m: float = 0.0
    terminal_speed: Optional[float] = None
    # Metres of the exit/decel zone (measured from the finish gate) that keep the
    # heavy timed time-weight, so the terminal brake starts AFTER the finish line
    # instead of bleeding back before it. 0.0 = original behaviour.
    decel_hold_m: float = 0.0
    # Metres before the finish that must stay centered (d) and heading-aligned
    # (psi_err) within a tight tolerance, so the trajectory ends straight
    # instead of at a residual angle. 0.0 = off.
    skidpad_terminal_straight_m: float = 0.0

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

        if self.terminal_speed is not None and self.mode == "trackdrive":
            raise ValueError(
                "terminal_speed is not supported for mode='trackdrive': the "
                "closed-loop constraint (X[N-1] == X[0]) would silently pin "
                "the free launch speed at node 0 too. Use 'autox' or "
                "'skidpad'."
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
    track_data: Dict,
    extension_m: float,
    lead_in_m: float = 0.0,
    ocp_lead_m: float = 0.0,
    timing_offset_m: float = 0.0,
) -> Dict:
    """Extend a closed-loop track by wrapping points beyond the finish line.

    The OCP horizon itself (``positions``/``headings``/... fed to
    ``build_ocp``) gets a forward run-off plus, if ``ocp_lead_m`` > 0,
    ``ocp_lead_m`` metres of backward run-in before ``s=0`` (also borrowed
    from the tail of the closed loop, since the point just "before" s=0 on a
    closed track is, geometrically, the end of the loop). The OCP's pinned
    launch condition then lands ``ocp_lead_m`` metres earlier, so the solver
    optimizes the speed/steering profile through that stretch instead of it
    being a flat hold. With ``ocp_lead_m=0`` this reproduces the original
    single-lap autox solve exactly.

    ``extension_m`` is measured from the *timing gate*, not from the track's
    nominal (but physically arbitrary) ``s = autox_base_length_m`` wrap
    point: the real timing gate sits ``timing_offset_m`` downstream of s=0,
    and is crossed a second time one lap later at
    ``s = autox_base_length_m + timing_offset_m`` (see
    ``solve_ocp_and_save``'s ``autox_lap_time_s``, which measures elapsed
    time between those two crossings). So the forward run-off appended here
    covers ``timing_offset_m + extension_m`` metres past
    ``s = autox_base_length_m``, guaranteeing ``extension_m`` metres of
    horizon remain *after* the real finish line — e.g. for braking down to a
    terminal speed once the timed lap is over.

    If ``lead_in_m`` > 0, the geometry for a further lead-in stretch *before*
    the (possibly moved-back) OCP horizon start is also computed, from the
    same tail-of-the-loop wraparound, and returned under
    ``"autox_lead_in"``. It is not part of the optimization: see
    ``_prepend_autox_lead_in``, which stitches it onto the solved trajectory
    afterwards as a prescribed, constant-velocity segment. Solving for it
    jointly with the OCP would force models with rate-limited actuator states
    (e.g. four_wheel's tire forces/steering) to hit an exact speed target
    while ramping those actuators up from a standing start on a coarse mesh,
    which can make the problem infeasible -- this is also the practical limit
    on ``ocp_lead_m``: pinning that same zero-actuator launch condition too
    far into a real corner is itself infeasible, and the solver will raise
    accordingly.

    The true single-lap length is stashed on the returned dict as
    ``"autox_base_length_m"``, used both for the lap-time measurement above
    and for building the post-finish untimed weighting in ``step_solve_ocp``.
    """
    ds_m = float(track_data["ds_m"])
    N_orig = len(track_data["arc_lengths"])
    M_pts = min(max(1, round((timing_offset_m + extension_m) / ds_m)), N_orig - 1)
    K_pts = min(max(0, round(lead_in_m / ds_m)), N_orig - 1)
    J_pts = min(max(0, round(ocp_lead_m / ds_m)), N_orig - 1)
    if K_pts + J_pts > N_orig - 1:
        raise ValueError(
            f"autox_lead_in_m + autox_ocp_lead_m ({lead_in_m:.1f} + "
            f"{ocp_lead_m:.1f} m = {K_pts + J_pts} points) exceeds the "
            f"track's available run-in ({N_orig - 1} points); reduce one or "
            "both."
        )

    positions = np.array(track_data["positions"], dtype=np.float64)
    headings = np.array(track_data["headings"], dtype=np.float64)
    curvatures = np.array(track_data["curvatures"], dtype=np.float64)
    curvatures_half = np.array(track_data["curvatures_half"], dtype=np.float64)
    arc_lengths = np.array(track_data["arc_lengths"], dtype=np.float64)
    w_left = np.array(track_data["w_left"], dtype=np.float64)
    w_right = np.array(track_data["w_right"], dtype=np.float64)

    total_length = arc_lengths[-1] + ds_m

    # Tail-of-the-loop layout, in order: [..lead_in K_pts..][..ocp_lead J_pts..][s=0..]
    ocp_lead_start = N_orig - J_pts
    lead_in_start = N_orig - J_pts - K_pts

    extended = dict(track_data)
    extended["positions"] = np.concatenate(
        [positions[ocp_lead_start:], positions, positions[:M_pts]], axis=0
    ).tolist()
    extended["headings"] = np.concatenate(
        [headings[ocp_lead_start:], headings, headings[:M_pts]]
    ).tolist()
    extended["curvatures"] = np.concatenate(
        [curvatures[ocp_lead_start:], curvatures, curvatures[:M_pts]]
    ).tolist()
    extended["curvatures_half"] = np.concatenate(
        [curvatures_half[ocp_lead_start:], curvatures_half, curvatures_half[:M_pts]]
    ).tolist()
    extended["arc_lengths"] = np.concatenate([
        arc_lengths[ocp_lead_start:] - total_length,
        arc_lengths,
        arc_lengths[:M_pts] + total_length,
    ]).tolist()
    extended["w_left"] = np.concatenate(
        [w_left[ocp_lead_start:], w_left, w_left[:M_pts]]
    ).tolist()
    extended["w_right"] = np.concatenate(
        [w_right[ocp_lead_start:], w_right, w_right[:M_pts]]
    ).tolist()
    extended["num_points"] = N_orig + M_pts + J_pts
    extended["total_length_m"] = float(arc_lengths[-1] + ds_m * M_pts + ds_m)
    extended["autox_base_length_m"] = float(total_length)

    if K_pts > 0:
        extended["autox_lead_in"] = {
            "positions": positions[lead_in_start:ocp_lead_start].tolist(),
            "headings": headings[lead_in_start:ocp_lead_start].tolist(),
            "curvatures": curvatures[lead_in_start:ocp_lead_start].tolist(),
            "arc_lengths": (arc_lengths[lead_in_start:ocp_lead_start] - total_length).tolist(),
            "w_left": w_left[lead_in_start:ocp_lead_start].tolist(),
            "w_right": w_right[lead_in_start:ocp_lead_start].tolist(),
        }

    return extended


def _prepend_autox_lead_in(sol_dict: Dict, lead_in: Dict, initial_speed: float) -> Dict:
    """Stitch a prescribed, constant-velocity lead-in onto a solved autox trajectory.

    The lead-in is not part of the OCP: it's a constant-speed run along the
    centerline at exactly ``initial_speed``, giving the physical car a
    stretch of track before the true start/finish line.

    The lead-in geometry is borrowed from the tail of the closed loop, so it
    genuinely curves. The exporter derives the reference curvature from the
    ``yaw_rate`` state (``kappa = yaw_rate / v_path``), so we set
    ``yaw_rate = kappa * initial_speed`` from the borrowed centerline
    curvature rather than zero-filling it -- otherwise the first few metres
    would export as straight while the path bends (up to ~0.26 1/m), feeding
    the tracker a wrong curvature reference at launch. Speed is held at
    ``initial_speed``; the remaining dynamic states (tire forces, steering,
    ...) stay at rest -- the tracker consumes speed, lateral deviation and
    curvature, none of which depend on them.
    """
    K = len(lead_in["arc_lengths"])
    positions = lead_in["positions"]
    curvatures = lead_in["curvatures"]

    sol_dict["path_xy"] = positions + list(sol_dict["path_xy"])
    sol_dict["arc_lengths"] = lead_in["arc_lengths"] + list(sol_dict["arc_lengths"])
    sol_dict["w_left"] = lead_in["w_left"] + list(sol_dict["w_left"])
    sol_dict["w_right"] = lead_in["w_right"] + list(sol_dict["w_right"])
    sol_dict["kappa"] = curvatures + list(sol_dict["kappa"])
    sol_dict["headings"] = lead_in["headings"] + list(sol_dict["headings"])

    # yaw_rate = kappa * v so the exporter recovers the true lead-in curvature.
    yaw_rate_lead_in = [float(k) * float(initial_speed) for k in curvatures]

    for name in sol_dict["state_names"]:
        if name in ("v", "v_long"):
            fill = [float(initial_speed)] * K
        elif name == "yaw_rate":
            fill = yaw_rate_lead_in
        else:
            fill = [0.0] * K
        sol_dict[name] = fill + list(sol_dict[name])

    for name in sol_dict["input_names"]:
        sol_dict[name] = [0.0] * K + list(sol_dict[name])

    if sol_dict.get("timed_mask") is not None:
        # Lead-in sits before the timing gate, so it counts as "timed" under
        # the same convention as the rest of the run-up (see
        # _autox_time_weights): only the post-finish tail is untimed.
        sol_dict["timed_mask"] = [1] * K + list(sol_dict["timed_mask"])

    return sol_dict


def _build_skidpad_lead_in(track_data: Dict, lead_in_m: float) -> Optional[Dict]:
    """Geometry for a straight, prescribed run-in before the skidpad OCP's s=0.

    Unlike autox (a closed loop, so the run-in has to be borrowed from the tail
    of the lap), the skidpad centerline already starts on a straight (see
    ``tracks/skidpad.py``), so the lead-in is just that same straight
    extrapolated backward from node 0 by ``lead_in_m``. Requires curvature[0]
    to be exactly 0 -- true as long as ``skidpad_start_x/y`` (if set) still
    leaves the node before the corner's kappa-blend zone.
    """
    if lead_in_m <= 0.0:
        return None

    ds_m = float(track_data["ds_m"])
    K = max(1, int(round(lead_in_m / ds_m)))

    kappa0 = float(track_data["curvatures"][0])
    if abs(kappa0) > 1e-6:
        raise ValueError(
            f"skidpad_lead_in_m requires the track to start on a straight "
            f"(curvature[0]={kappa0:.4f} != 0); move skidpad_start_x/y "
            "further from the gate or shorten the lead-in."
        )

    x0, y0 = track_data["positions"][0]
    heading0 = float(track_data["headings"][0])
    dir_x, dir_y = float(np.cos(heading0)), float(np.sin(heading0))
    w_left0 = float(track_data["w_left"][0])
    w_right0 = float(track_data["w_right"][0])

    offsets = ds_m * np.arange(K, 0, -1)
    return {
        "positions": [[x0 - dir_x * off, y0 - dir_y * off] for off in offsets],
        "headings": [heading0] * K,
        "curvatures": [0.0] * K,
        "arc_lengths": (-offsets).tolist(),
        "w_left": [w_left0] * K,
        "w_right": [w_right0] * K,
    }


def _prepend_skidpad_lead_in(sol_dict: Dict, lead_in: Dict, speed: float) -> Dict:
    """Stitch a prescribed, constant-speed straight lead-in onto a solved
    skidpad trajectory (mirrors ``_prepend_autox_lead_in``). Simpler than the
    autox version since the lead-in is a straight line: curvature and
    yaw_rate are exactly zero throughout, not just held at launch.
    """
    K = len(lead_in["arc_lengths"])

    sol_dict["path_xy"] = lead_in["positions"] + list(sol_dict["path_xy"])
    sol_dict["arc_lengths"] = lead_in["arc_lengths"] + list(sol_dict["arc_lengths"])
    sol_dict["w_left"] = lead_in["w_left"] + list(sol_dict["w_left"])
    sol_dict["w_right"] = lead_in["w_right"] + list(sol_dict["w_right"])
    sol_dict["kappa"] = lead_in["curvatures"] + list(sol_dict["kappa"])
    sol_dict["headings"] = lead_in["headings"] + list(sol_dict["headings"])

    for name in sol_dict["state_names"]:
        fill = [float(speed)] * K if name in ("v", "v_long") else [0.0] * K
        sol_dict[name] = fill + list(sol_dict[name])

    for name in sol_dict["input_names"]:
        sol_dict[name] = [0.0] * K + list(sol_dict[name])

    # Lead-in sits before the gate, same as the rest of the entry straight it
    # extends, so it stays untimed/decel-free under the existing masks.
    if sol_dict.get("timed_mask") is not None:
        sol_dict["timed_mask"] = [0] * K + list(sol_dict["timed_mask"])
    if sol_dict.get("decel_mask") is not None:
        sol_dict["decel_mask"] = [0] * K + list(sol_dict["decel_mask"])

    return sol_dict


def _autox_time_weights(
    arc_lengths,
    base_length_m: float,
    timing_offset_m: float,
    eps_time: float,
    decel_hold_m: float,
) -> np.ndarray:
    """Per-node objective time weights for autox: full weight through the
    timed lap, ``eps_time`` after the finish line.

    Mirrors skidpad's timed/untimed masking (``step_solve_ocp``'s
    ``mode == "skidpad"`` branch), but the "finish line" here is the timing
    gate's second crossing, ``gate2 = base_length_m + timing_offset_m`` (see
    ``_extend_track_for_autox``), not a track-provided mask. ``decel_hold_m``
    keeps the heavy timed weight for that many extra metres past the gate,
    so the terminal brake starts after crossing rather than bleeding back
    onto the timed lap.
    """
    arc = np.asarray(arc_lengths, dtype=float)
    gate2 = float(base_length_m) + float(timing_offset_m)
    weights = np.where(arc < gate2, 1.0, float(eps_time))
    if decel_hold_m > 0.0:
        hold_end = gate2 + float(decel_hold_m)
        weights = np.where((arc >= gate2) & (arc < hold_end), 1.0, weights)
    return weights


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

    start_xy = None
    if config.skidpad_start_x is not None:
        start_xy = (config.skidpad_start_x, config.skidpad_start_y)

    track = build_skidpad_track(
        map_csv=map_csv,
        ref_csv=ref_csv,
        ds_m=config.ds_m,
        entry_exit_halfwidth=config.entry_exit_halfwidth,
        kappa_blend_m=config.kappa_blend_m,
        start_xy=start_xy,
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
        # Keep the heavy timed weight for the first `decel_hold_m` metres of the exit
        # (from the finish gate), so the terminal brake starts after the finish line
        # rather than bleeding back onto the last timed circle.
        n_hold = 0
        if config.decel_hold_m > 0.0:
            ds_hold = float(track_data.get("ds_m", config.ds_m))
            decel_idx = np.where(decel > 0.5)[0]
            n_hold = min(int(round(config.decel_hold_m / ds_hold)), decel_idx.size)
            if n_hold > 0:
                time_weights[decel_idx[:n_hold]] = 1.0
        print(f"  Skidpad: {int(mask.sum())}/{len(mask)} timed nodes, "
              f"{int(decel.sum())} exit (decel) nodes, "
              f"un-timed weight eps_time={config.eps_time}, "
              f"decel_hold={config.decel_hold_m} m ({n_hold} exit nodes held), "
              f"terminal_speed={config.terminal_speed}")
    elif config.mode == "autox":
        track_data = _extend_track_for_autox(
            track_data,
            config.autox_extension_m,
            config.autox_lead_in_m,
            config.autox_ocp_lead_m,
            timing_offset_m=config.autox_timing_offset_m,
        )
        print(f"  Autox: extended track by {config.autox_timing_offset_m + config.autox_extension_m:.0f} m "
              f"({track_data['num_points']} points total, OCP horizon, "
              f"{config.autox_ocp_lead_m:.1f} m of which is backward run-in, "
              f"{config.autox_extension_m:.0f} m of which is post-finish run-off)")
        time_weights = _autox_time_weights(
            track_data["arc_lengths"],
            track_data["autox_base_length_m"],
            config.autox_timing_offset_m,
            config.eps_time,
            config.decel_hold_m,
        )
        n_timed = int(np.sum(time_weights >= 1.0 - 1e-9))
        # Flows through to the solution JSON via the generic
        # `track.get("timed_mask")` in solve_ocp_and_save (same field skidpad
        # uses), so visualization can shade the post-finish untimed zone.
        track_data["timed_mask"] = (time_weights >= 1.0 - 1e-9).astype(int).tolist()
        print(f"  Autox: {n_timed}/{len(time_weights)} timed nodes, "
              f"un-timed weight eps_time={config.eps_time}, "
              f"decel_hold={config.decel_hold_m} m, "
              f"terminal_speed={config.terminal_speed}")

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
        "autox_timing_offset_m": float(config.autox_timing_offset_m),
        "autox_ocp_lead_m": float(config.autox_ocp_lead_m),
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
        terminal_speed=config.terminal_speed if config.mode in ("skidpad", "autox") else None,
        autox_timing_offset_m=config.autox_timing_offset_m if config.mode == "autox" else None,
        terminal_straight_m=config.skidpad_terminal_straight_m if config.mode == "skidpad" else None,
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

    skidpad_lead_in = (
        _build_skidpad_lead_in(track_data, config.skidpad_lead_in_m)
        if config.mode == "skidpad" else None
    )
    if skidpad_lead_in:
        sol_dict = _prepend_skidpad_lead_in(sol_dict, skidpad_lead_in, config.initial_speed)
        with solution_path.open("w") as f:
            json.dump(sol_dict, f, indent=2)
        print(
            f"  Skidpad: prepended {config.skidpad_lead_in_m:.1f} m constant-speed "
            f"({config.initial_speed:.1f} m/s) lead-in "
            f"({len(skidpad_lead_in['arc_lengths'])} points, not part of the OCP solve)"
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
    timed_mask = data.get("timed_mask")

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
            timed_mask=timed_mask,
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
            timed_mask=timed_mask,
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
            timed_mask=timed_mask,
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
                "autox_timing_offset_m": float(config.autox_timing_offset_m),
                "autox_ocp_lead_m": float(config.autox_ocp_lead_m),
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

