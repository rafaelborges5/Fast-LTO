"""
Pipeline orchestrator: track generation through to plots.

Six steps, each writing one artifact under the data root:

1. ``track``   -> data/tracks/{track_id}.csv
2. ``spline``  -> data/discretized/{track_id}.json
3. ``bounds``  -> data/discretized/{track_id}_with_widths.json
4. ``ocp``     -> data/solutions/{track_id}_{model}_{integrator}_{mode}.json
5. ``export``  -> data/output_trajectories/{track_id}_{model}_{integrator}_{stamp}.csv
6. ``plot``    -> ocp_plots/{stamp}/panels.png

``start_from`` resumes at any step, reading the previous step's artifact off
disk. See ``README.md`` in this directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import numpy as np

from fast_lto.modes import AutoxConfig, EventMode, SkidpadConfig, get_mode
from fast_lto.optimization.global_ocp import load_track_with_widths, solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator, SpaceIntegrator
from fast_lto.paths import default_data_root
from fast_lto.splines.discretized_track import DiscretizedTrack
from fast_lto.splines.spline_fitter import ContinuityType, fit_and_discretize
from fast_lto.tracks.bean import generate_bean_track
from fast_lto.tracks.ellipse import generate_ellipse_track
from fast_lto.tracks.fsg_trackdrive import generate_fsg_track
from fast_lto.utils.track_bounds import (
    LateralBoundsResult,
    apply_savgol_to_widths,
    compute_lateral_bounds,
    load_boundaries,
    save_track_with_widths,
)
from fast_lto.vehicle_models import (
    DynamicBicycleModel,
    FourWheelModel,
    PointMassModel,
    VehicleModel,
)
from fast_lto.visualization.panels import render_panels

if TYPE_CHECKING:
    # Typing only: config imports pipeline, so a runtime import is circular.
    from fast_lto.config import VehicleConfig

StepName = Literal["track", "spline", "bounds", "ocp", "export", "plot"]
WarmStartPolicy = Literal["off", "auto", "ladder"]
WARM_START_POLICIES = get_args(WarmStartPolicy)

TrackType = Literal["fsg", "ellipse", "bean", "skidpad"]
# Derived, not hand-listed: the CLI's --track-type choices read this.
TRACK_TYPES = get_args(TrackType)


@dataclass
class PipelineConfig:
    """
    Configuration for the Fast-LTO pipeline.

    Parameters are grouped roughly by pipeline step; most have sensible defaults.
    """

    track_id: str = "fsg_random"
    track_type: TrackType = "fsg"

    repo_root: Optional[Path] = None
    #: Reads a CSV somewhere other than data/tracks/{track_id}.csv.
    track_csv_override: Optional[Path] = None

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
    #: A scalar weight, or one per input (see --reg-du-vec).
    reg_u: Union[float, Sequence[float]] = 600.0
    reg_u_l2: float | None = None
    #: None follows the mode; the ``launch_speed`` property resolves it.
    initial_speed: Optional[float] = None
    boundary_margin: float = 0.0
    #: Per-event settings, each owned by its ``EventMode``. Naming the block of
    #: an event other than ``mode`` is an error rather than a silent no-op.
    autox: AutoxConfig = field(default_factory=AutoxConfig)
    skidpad: SkidpadConfig = field(default_factory=SkidpadConfig)

    export_trajectory: bool = True

    plot_results: bool = True
    show_plots: bool = True
    normalize_states_and_inputs: bool = True
    solver_verbose: bool = False

    #: ``"off"`` reads and writes no seed, so the solve is bit for bit the cold
    #: one. ``"auto"`` seeds from the store, and solves one easier problem first
    #: when nothing fits and a cold start would begin infeasible. ``"ladder"``
    #: always walks up from a safe margin, ignoring the store.
    warm_start: WarmStartPolicy = "auto"
    warm_start_max_margin_gap: float = 0.15
    warm_start_ladder_step: float = 0.05
    warm_start_max_seeds: int = 50
    warm_start_seed: Optional[str] = None

    vehicle_config: Optional["VehicleConfig"] = None

    def __post_init__(self) -> None:
        if self.mode not in ("autox", "trackdrive", "skidpad"):
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Must be 'autox', 'trackdrive' or 'skidpad'."
            )

        if self.warm_start not in WARM_START_POLICIES:
            raise ValueError(
                f"Unknown warm_start: {self.warm_start!r}. "
                f"Must be one of {list(WARM_START_POLICIES)}."
            )

        if self.repo_root is None:
            self.repo_root = default_data_root()
        else:
            self.repo_root = Path(self.repo_root)

        self.discretized_dir = self.repo_root / "data" / "discretized"
        self.solutions_dir = self.repo_root / "data" / "solutions"
        self.output_trajectories_dir = self.repo_root / "data" / "output_trajectories"
        self.plots_dir = self.repo_root / "ocp_plots"

    @property
    def launch_speed(self) -> float:
        """Speed the trajectory starts at, in m/s.

        ``initial_speed`` is the request: ``None`` means "whatever this event
        starts at". This is the answer, and unlike the field it is never
        ``None``, so callers do not have to re-narrow it at every use.

        Derived on access for the same reason ``track_csv_path`` is. It used to
        be written into ``initial_speed`` by ``__post_init__``, which needed a
        private flag and a setter to survive being re-run after ``mode``
        changed -- three pieces of machinery for what is one expression.
        """
        if self.initial_speed is not None:
            return float(self.initial_speed)
        return 5.0 if self.mode == "trackdrive" else 3.0

    @property
    def track_csv_path(self) -> Path:
        """The boundary CSV this run reads.

        Derived on every access rather than resolved once in ``__post_init__``,
        which runs more than once: ``RunConfig.to_pipeline_config`` re-runs it
        so a CLI flag can override the YAML. Anything derived there has to be
        re-derivable, or a ``--track-id`` override renames the outputs while
        still reading the previous id's CSV. ``launch_speed`` is the same.
        """
        if self.track_csv_override is not None:
            return Path(self.track_csv_override)
        assert self.repo_root is not None  # always resolved in __post_init__
        return self.repo_root / "data" / "tracks" / f"{self.track_id}.csv"

    @property
    def discretized_track_path(self) -> Path:
        return self.discretized_dir / f"{self.track_id}.json"

    @property
    def track_with_widths_path(self) -> Path:
        return self.discretized_dir / f"{self.track_id}_with_widths.json"

    @property
    def solution_path(self) -> Path:
        return (
            self.solutions_dir
            / f"{self.track_id}_{self.model_name}_{self.integrator_name}_{self.mode}.json"
        )

    @property
    def export_trajectory_path(self) -> Path:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return (
            self.output_trajectories_dir
            / f"{self.track_id}_{self.model_name}_{self.integrator_name}_{ts}.csv"
        )


def step_generate_track(config: PipelineConfig) -> Path:
    csv_path = config.track_csv_path

    print("[Step 1] Track generation")
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

    print("[Step 2] Spline fitting and discretization")
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

    print(f"  Computed widths: misses left/right: " f"{result.misses_left}/{result.misses_right}")

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


def _make_model(model_name: str, vehicle_config: Optional["VehicleConfig"] = None) -> VehicleModel:
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


def _splice_segment(
    sol_dict: Dict,
    segment: Dict,
    speed: float,
    *,
    side: Literal["before", "after"],
    timed: int,
) -> Dict:
    """Stitch a prescribed, constant-speed segment onto a solved trajectory.

    ``yaw_rate`` is filled as ``kappa * speed`` from the segment's own
    curvature, because the exporter derives the reference curvature back out of
    it. Zeros would export a curving lead-in as straight. The remaining dynamic
    states stay at rest; nothing downstream reads them.

    Parameters
    ----------
    side:
        ``"before"`` prepends the segment (a lead-in), ``"after"`` appends it
        (a terminal pad).
    timed:
        Value to extend ``timed_mask`` with. ``decel_mask``, where present, is
        always extended with ``0``: no prescribed segment is a braking zone.
    """
    n = len(segment["arc_lengths"])

    def splice(existing: Sequence[Any], addition: Sequence[Any]) -> List[Any]:
        if side == "before":
            return list(addition) + list(existing)
        return list(existing) + list(addition)

    for sol_key, segment_key in (
        ("path_xy", "positions"),
        ("arc_lengths", "arc_lengths"),
        ("w_left", "w_left"),
        ("w_right", "w_right"),
        ("kappa", "curvatures"),
        ("headings", "headings"),
    ):
        sol_dict[sol_key] = splice(sol_dict[sol_key], segment[segment_key])

    yaw_rate_fill = [float(k) * float(speed) for k in segment["curvatures"]]

    for name in sol_dict["state_names"]:
        if name in ("v", "v_long"):
            fill = [float(speed)] * n
        elif name == "yaw_rate":
            fill = yaw_rate_fill
        else:
            fill = [0.0] * n
        sol_dict[name] = splice(sol_dict[name], fill)

    for name in sol_dict["input_names"]:
        sol_dict[name] = splice(sol_dict[name], [0.0] * n)

    for mask_name, mask_value in (("timed_mask", timed), ("decel_mask", 0)):
        if sol_dict.get(mask_name) is not None:
            sol_dict[mask_name] = splice(sol_dict[mask_name], [mask_value] * n)

    return sol_dict


def _resolve_path(root: Optional[Path], p: str | Path) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    path = Path(p)
    if path.is_absolute() or root is None:
        return path
    return root / path


def step_build_skidpad_track(config: PipelineConfig) -> Path:
    """Build the skidpad track-with-widths JSON from the cone map + reference."""
    from fast_lto.tracks.skidpad import build_skidpad_track

    skidpad = config.skidpad
    if skidpad.map_csv is None or skidpad.reference_csv is None:
        raise ValueError("mode='skidpad' requires skidpad.map_csv and skidpad.reference_csv.")

    map_csv = _resolve_path(config.repo_root, skidpad.map_csv)
    ref_csv = _resolve_path(config.repo_root, skidpad.reference_csv)

    print("[Skidpad] Building track from cone map + reference trajectory")
    print(f"  Map:       {map_csv}")
    print(f"  Reference: {ref_csv}")

    start_xy = None
    if skidpad.start_x is not None:
        start_xy = (skidpad.start_x, skidpad.start_y)

    track = build_skidpad_track(
        map_csv=map_csv,
        ref_csv=ref_csv,
        ds_m=config.ds_m,
        entry_exit_halfwidth=skidpad.entry_exit_halfwidth,
        kappa_blend_m=skidpad.kappa_blend_m,
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


def _seed_signature_for(
    config: PipelineConfig,
    track_data: Dict,
    model: VehicleModel,
    boundary_margin: float,
    mode: EventMode,
) -> Dict:
    from fast_lto.optimization.warm_start import seed_signature

    return seed_signature(
        track_id=config.track_id,
        mode=config.mode,
        model_name=config.model_name,
        integrator_name=config.integrator_name,
        continuity=config.continuity,
        normalize_states_and_inputs=config.normalize_states_and_inputs,
        boundary_margin=boundary_margin,
        track=track_data,
        model=model,
        reg_u=config.reg_u,
        reg_u_l2=config.reg_u_l2,
        initial_speed=config.launch_speed,
        **mode.seed_kwargs(config),
    )


def _plan_ladder(
    config: PipelineConfig,
    track_data: Dict,
    model: VehicleModel,
) -> List[float]:
    """Margins to solve on the way to the target, target included.

    A single value means "solve the target directly". The starting rung is the
    largest margin at which the default centreline guess is still feasible, so
    the first (cold) solve of the ladder is an easy one.
    """
    from fast_lto.utils.corridor import critical_margin, describe_critical_margin

    target = float(config.boundary_margin)
    corners = model.get_corner_offsets()
    if not corners:
        return [target]

    crit = critical_margin(track_data, corners)
    print(f"  Warm start: {describe_critical_margin(crit, target)}")

    if crit.already_closed:
        # No margin makes the centreline guess feasible; a ladder cannot help.
        return [target]

    start = max(crit.margin - 0.02, 0.0)
    if start >= target:
        return [target]

    if config.warm_start == "auto":
        return [start, target]

    step = max(float(config.warm_start_ladder_step), 1e-3)
    # Drop a rung sitting on top of the target; solving it twice buys nothing.
    rungs = [float(m) for m in np.arange(start, target, step) if target - m > 0.5 * step]
    rungs.append(target)
    return rungs


def _solve_once(
    config: PipelineConfig,
    track_data: Dict,
    model: VehicleModel,
    integrator: SpaceIntegrator,
    time_weights: Optional[np.ndarray],
    solution_path: Path,
    run_config: Optional[Dict],
    boundary_margin: float,
    mode: EventMode,
    initial_guess: Optional[Dict] = None,
) -> Dict:
    return solve_ocp_and_save(
        track=track_data,
        model=model,
        solution_path=solution_path,
        integrator=integrator,
        initial_speed=config.launch_speed,
        reg_du=config.reg_u,
        reg_u_l2=config.reg_u_l2,
        run_config=run_config,
        use_normalization=config.normalize_states_and_inputs,
        solver_verbose=config.solver_verbose,
        boundary_margin=boundary_margin,
        mode=config.mode,
        time_weights=time_weights,
        initial_guess=initial_guess,
        **mode.solver_kwargs(config),
    )


def _save_seed_quietly(
    ws: ModuleType,
    seeds_root: Path,
    signature: Dict,
    solution: Dict,
    max_seeds: int,
) -> None:
    """Store a seed, but never let a cache write throw away a good solve."""
    try:
        ws.save_seed(seeds_root, signature, solution, max_seeds)
    except Exception as exc:  # noqa: BLE001
        print(f"  Warm start: could not store seed ({type(exc).__name__}: {exc})")


def _solve_with_warm_start(
    config: PipelineConfig,
    track_data: Dict,
    model: VehicleModel,
    integrator: SpaceIntegrator,
    time_weights: Optional[np.ndarray],
    solution_path: Path,
    run_config: Dict,
    mode: EventMode,
) -> Dict:
    """Solve the target problem, seeded from the store when that helps.

    Never decides *whether* to solve — only what the solver starts from. With
    ``warm_start='off'`` nothing here touches the disk and the solve is the cold
    one.
    """
    import tempfile

    from fast_lto.optimization import warm_start as ws

    provenance: Dict = {
        "policy": str(config.warm_start),
        "seed_file": None,
        "seed_margin": None,
        "seed_vehicle_distance": None,
        "ladder": [],
        "fell_back_cold": False,
    }

    def finish(sol: Dict) -> Dict:
        """Record how the solve was seeded, in the file as well as the dict."""
        sol["warm_start"] = provenance
        with solution_path.open("w") as f:
            json.dump(sol, f, indent=2)
        return sol

    if config.warm_start == "off":
        return finish(
            _solve_once(
                config,
                track_data,
                model,
                integrator,
                time_weights,
                solution_path,
                run_config,
                config.boundary_margin,
                mode,
            )
        )

    seeds_root = config.solutions_dir / ws.SEEDS_DIRNAME
    signature = _seed_signature_for(config, track_data, model, config.boundary_margin, mode)

    def guess_from(solution: Dict, source: str) -> Optional[Dict]:
        try:
            guess = ws.resample_guess(
                solution, track_data, model, config.normalize_states_and_inputs
            )
        except Exception as exc:  # noqa: BLE001 - a bad seed must never be fatal
            print(f"  Warm start: ignoring seed ({source}): {exc}")
            return None
        ok, why = ws.validate_guess(guess, track_data, model, float(config.boundary_margin))
        if not ok:
            print(f"  Warm start: ignoring seed ({source}): {why}")
            return None
        return guess

    guess: Optional[Dict] = None

    # 1. An explicitly requested seed always wins.
    if config.warm_start_seed:
        seed_path = _resolve_path(config.repo_root, config.warm_start_seed)
        if seed_path is not None and Path(seed_path).is_file():
            solution = json.loads(Path(seed_path).read_text())
            guess = guess_from(solution, str(seed_path))
            if guess is not None:
                print(f"  Warm start: seeded from {seed_path}")
                provenance["seed_file"] = str(seed_path)
        else:
            print(f"  Warm start: seed file not found: {config.warm_start_seed}")

    # 2. Otherwise take the closest compatible solve out of the store.
    if guess is None and config.warm_start != "ladder":
        match = ws.find_seed(seeds_root, signature, float(config.warm_start_max_margin_gap))
        if match is not None:
            solution = json.loads(match.path.read_text())
            guess = guess_from(solution, match.path.name)
            if guess is not None:
                print(
                    f"  Warm start: seeded from {match.path.name} "
                    f"(margin {match.margin:.2f}, gap {match.margin_gap:.2f} m, "
                    f"vehicle distance {match.vehicle_distance:.3f})"
                )
                provenance.update(match.as_provenance())

    # 3. No seed: walk up to the target when a cold start would begin infeasible.
    if guess is None:
        ladder = _plan_ladder(config, track_data, model)
        if len(ladder) > 1:
            print(
                "  Warm start: no compatible seed, solving "
                + " -> ".join(f"{m:.2f}" for m in ladder)
            )
            with tempfile.TemporaryDirectory() as tmp:
                for rung in ladder[:-1]:
                    print(f"  Warm start: intermediate solve at margin {rung:.2f}")
                    rung_path = Path(tmp) / f"ladder_m{round(rung * 1000):04d}.json"
                    try:
                        rung_sol = _solve_once(
                            config,
                            track_data,
                            model,
                            integrator,
                            time_weights,
                            rung_path,
                            None,
                            rung,
                            mode,
                            initial_guess=guess,
                        )
                    except Exception as exc:  # noqa: BLE001
                        # An optimisation, not a requirement: fall through to
                        # the target with whatever guess we have.
                        print(
                            f"  Warm start: intermediate solve at {rung:.2f} failed "
                            f"({type(exc).__name__}), continuing to the target"
                        )
                        provenance["fell_back_cold"] = guess is None
                        break
                    provenance["ladder"].append(float(rung))
                    _save_seed_quietly(
                        ws,
                        seeds_root,
                        _seed_signature_for(config, track_data, model, rung, mode),
                        rung_sol,
                        int(config.warm_start_max_seeds),
                    )
                    guess = guess_from(rung_sol, f"ladder rung {rung:.2f}")
                    if guess is None:
                        break
        else:
            provenance["fell_back_cold"] = True

    sol = finish(
        _solve_once(
            config,
            track_data,
            model,
            integrator,
            time_weights,
            solution_path,
            run_config,
            config.boundary_margin,
            mode,
            initial_guess=guess,
        )
    )
    _save_seed_quietly(ws, seeds_root, signature, sol, int(config.warm_start_max_seeds))
    return sol


def _print_mode_summary(
    config: PipelineConfig,
    track_data: Dict,
    time_weights: Optional[np.ndarray],
) -> None:
    """Print what the mode did to the problem, if it had anything to say."""
    for line in get_mode(config.mode).summary(config, track_data, time_weights):
        print(line)


def step_solve_ocp(
    config: PipelineConfig,
    track_with_widths_path: Optional[Path] = None,
) -> Path:
    if track_with_widths_path is None:
        track_with_widths_path = config.track_with_widths_path

    print("[Step 4] OCP solving")
    print(f"  Track with widths: {track_with_widths_path}")

    track_data: Dict = load_track_with_widths(track_with_widths_path)

    mode = get_mode(config.mode)
    track_data = mode.prepare_track(track_data, config)
    time_weights = mode.time_weights(track_data, config)
    _print_mode_summary(config, track_data, time_weights)

    model = _make_model(config.model_name, vehicle_config=config.vehicle_config)
    integrator = _make_integrator(config.integrator_name)

    print(f"  Mode: {config.mode}")
    print(f"  Model: {config.model_name}")
    print(f"  Integrator: {config.integrator_name}")
    print(f"  Input rate regularization (reg_u): {config.reg_u}")
    print(f"  Input L2 regularization (reg_u_l2): {config.reg_u_l2}")

    config.solutions_dir.mkdir(parents=True, exist_ok=True)
    solution_path = config.solution_path

    # The run configuration that uniquely characterises a solution.
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

    # reg_u may be a scalar or one weight per input. Tested for the scalar
    # case: Sequence is open-ended, so excluding the obvious types proves
    # nothing.
    reg_du_for_sig: Union[float, List[float]]
    if isinstance(config.reg_u, (int, float)):
        reg_du_for_sig = float(config.reg_u)
    else:
        reg_du_for_sig = [float(v) for v in config.reg_u]

    run_config = {
        "track_id": config.track_id,
        "mode": config.mode,
        "model_name": config.model_name,
        "ds_m": float(track_ds_m),
        "num_points": int(track_num_points),
        "continuity": str(config.continuity),
        "integrator_name": config.integrator_name,
        "reg_du": reg_du_for_sig,
        "initial_speed": config.launch_speed,
        "normalize_states_and_inputs": bool(config.normalize_states_and_inputs),
        "solver_verbose": bool(config.solver_verbose),
        "use_savgol_bounds": bool(config.use_savgol_bounds),
        "savgol_window_length": int(config.savgol_window_length),
        "savgol_polyorder": int(config.savgol_polyorder),
        "boundary_margin": float(config.boundary_margin),
        "autox_timing_offset_m": float(config.autox.timing_offset_m),
        "autox_ocp_lead_m": float(config.autox.ocp_lead_m),
        "autox_start_x": float(config.autox.start_x),
        "autox_start_y": float(config.autox.start_y),
        "autox_start_node_offset": int(config.autox.start_node_offset),
        "autox_idx_ref": int(track_data.get("autox_idx_ref", 0)),
        "D_safe_braking": (
            float(config.autox.D_safe_braking) if config.autox.D_safe_braking is not None else None
        ),
    }

    sol_dict = _solve_with_warm_start(
        config=config,
        track_data=track_data,
        model=model,
        integrator=integrator,
        mode=mode,
        time_weights=time_weights,
        solution_path=solution_path,
        run_config=run_config,
    )

    for plan in mode.splices(config, track_data, sol_dict):
        sol_dict = _splice_segment(
            sol_dict, plan.segment, plan.speed, side=plan.side, timed=plan.timed
        )
        with solution_path.open("w") as f:
            json.dump(sol_dict, f, indent=2)
        print(plan.message)

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
    from fast_lto.export.trajectory import export_reference_trajectory

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

    print("[Step 6] Visualization")
    print(f"  Solution: {solution_path}")
    print(f"  Boundaries CSV: {csv_path}")

    with solution_path.open("r") as f:
        data = json.load(f)

    if config.mode == "skidpad":
        from datetime import datetime

        from fast_lto.tracks.skidpad import load_skidpad_cones
        from fast_lto.visualization.skidpad_plots import plot_skidpad

        cones = load_skidpad_cones(csv_path)
        timestamp_dir = config.plots_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        plot_path = timestamp_dir / "panels.png"
        summary = plot_skidpad(data, cones, out_path=plot_path, show=config.show_plots)
        laps = ", ".join(f"{t:.3f}s" for t in summary["timed_lap_times"])
        print(f"  Timed laps: {laps}  |  score (avg): {summary['score']:.3f} s")
        print(f"  Saved plots to: {timestamp_dir}")
        return timestamp_dir

    boundaries = load_boundaries(csv_path)

    timestamp_dir = config.plots_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    render_panels(
        config.model_name,
        data,
        boundaries["left"],
        boundaries["right"],
        out_path=timestamp_dir / "panels.png",
        show=config.show_plots,
    )

    print(f"  Saved plots to: {timestamp_dir}")
    return timestamp_dir


# --------------------------------------------------------------------------- #
#  The run plan -- every step runs every time; `start_from` is the only reuse
# --------------------------------------------------------------------------- #

STEP_ORDER: Tuple[StepName, ...] = ("track", "spline", "bounds", "ocp", "export", "plot")


@dataclass(frozen=True)
class _Step:
    """One stage of the pipeline: what it runs, and where its output lands."""

    name: StepName
    run: Callable[[PipelineConfig, Dict[str, Path]], Path]
    artifact: Callable[[PipelineConfig], Path]
    enabled: Callable[[PipelineConfig], bool] = lambda config: True


def _plot_source_csv(config: PipelineConfig) -> Path:
    """The cone CSV the plots are drawn against."""
    if config.mode == "skidpad":
        if config.skidpad.map_csv is None:
            raise ValueError("mode='skidpad' requires skidpad_map_csv.")
        return _resolve_path(config.repo_root, config.skidpad.map_csv)
    return config.track_csv_path


def _step_track(config: PipelineConfig, results: Dict[str, Path]) -> Path:
    if config.generate_track or not config.track_csv_path.exists():
        return step_generate_track(config)
    print(f"[Step 1] Using existing track CSV: {config.track_csv_path}")
    return config.track_csv_path


def _step_spline(config: PipelineConfig, results: Dict[str, Path]) -> Path:
    step_fit_spline(config, csv_path=results["track"])
    return config.discretized_track_path


def _step_bounds(config: PipelineConfig, results: Dict[str, Path]) -> Path:
    # Read back from disk, so this step behaves the same whether or not the
    # spline ran in this process.
    step_compute_bounds(config, track=None, csv_path=results["track"])
    return config.track_with_widths_path


def _step_ocp(config: PipelineConfig, results: Dict[str, Path]) -> Path:
    return step_solve_ocp(config)


def _step_export(config: PipelineConfig, results: Dict[str, Path]) -> Path:
    return step_export_trajectory(config, solution_path=results["ocp"])


def _step_plot(config: PipelineConfig, results: Dict[str, Path]) -> Path:
    return step_visualize(config, solution_path=results["ocp"], csv_path=_plot_source_csv(config))


_SOLVE_AND_AFTER = (
    _Step("ocp", _step_ocp, lambda c: c.solution_path),
    _Step(
        "export", _step_export, lambda c: c.output_trajectories_dir, lambda c: c.export_trajectory
    ),
    _Step("plot", _step_plot, lambda c: c.plots_dir, lambda c: c.plot_results),
)


def _plan(config: PipelineConfig) -> List[_Step]:
    """The steps this configuration runs, in order.

    Skidpad reaches the same track-with-widths JSON by a different route -- its
    path overlaps itself, so it cannot go through the generic spline and bounds
    machinery -- and used to be a parallel copy of the whole function that took
    ``end_at`` but quietly ignored ``start_from``. It is one entry in the plan
    instead; everything from the solve onward is shared.
    """
    if config.mode == "skidpad" or config.track_type == "skidpad":
        upstream: List[_Step] = [
            _Step(
                "bounds",
                lambda c, r: step_build_skidpad_track(c),
                lambda c: c.track_with_widths_path,
            )
        ]
    else:
        upstream = [
            _Step("track", _step_track, lambda c: c.track_csv_path),
            _Step("spline", _step_spline, lambda c: c.discretized_track_path),
            _Step(
                "bounds",
                _step_bounds,
                lambda c: c.track_with_widths_path,
                lambda c: c.compute_bounds,
            ),
        ]
    return [*upstream, *_SOLVE_AND_AFTER]


def run_pipeline(
    config: PipelineConfig,
    start_from: StepName = "track",
    end_at: Optional[StepName] = None,
) -> Dict[str, Path]:
    """Run the pipeline from ``start_from`` through ``end_at`` (both inclusive).

    Steps before ``start_from`` are not run; their outputs must already exist,
    and are reported in the result so the caller sees the full set of paths
    either way.
    """
    for name, value in (("start_from", start_from), ("end_at", end_at)):
        if value is not None and value not in STEP_ORDER:
            raise ValueError(f"Unknown {name}: {value!r}. Must be one of {list(STEP_ORDER)}.")

    first = STEP_ORDER.index(start_from)
    last = STEP_ORDER.index(end_at) if end_at is not None else len(STEP_ORDER) - 1
    if last < first:
        raise ValueError(f"end_at={end_at!r} comes before start_from={start_from!r}.")

    results: Dict[str, Path] = {}
    for step in _plan(config):
        position = STEP_ORDER.index(step.name)
        if position > last or not step.enabled(config):
            continue

        if position < first:
            # Skipped by request, so its output has to be there already.
            artifact = step.artifact(config)
            if not artifact.exists():
                raise FileNotFoundError(
                    f"{step.name} output not found at {artifact}. "
                    f"Run with start_from={step.name!r} first."
                )
            results[step.name] = artifact
            continue

        results[step.name] = step.run(config, results)

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
