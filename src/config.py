"""
Unified YAML configuration for Fast-LTO.

Provides RunConfig (top-level) and VehicleConfig (shared + per-model vehicle
parameters).  Load with ``RunConfig.from_yaml("path.yaml")``, save with
``run_config.to_yaml("path.yaml")``.

Every key in the YAML is validated against known fields — typos and unused
parameters raise ``ValueError`` rather than being silently ignored.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from vehicle_models.point_mass import PointMassModel
from vehicle_models.dynamic_bicycle import DynamicBicycleModel
from vehicle_models.four_wheel import FourWheelModel

_MODEL_CLASSES = {
    "point_mass": PointMassModel,
    "dynamic_bicycle": DynamicBicycleModel,
    "four_wheel": FourWheelModel,
}

_MODEL_NAMES = set(_MODEL_CLASSES.keys())


def _get_model_default_keys(model_name: str) -> set:
    cls = _MODEL_CLASSES[model_name]
    return set(cls(params=None).get_default_params().keys())


# ---------------------------------------------------------------------------
# VehicleConfig
# ---------------------------------------------------------------------------

_VEHICLE_SHARED_FIELDS = {
    "m", "g", "v_min", "v_max", "d_max", "psi_err_max", "mu",
    "lf", "lr", "corners",
    "v_eps", "smoothmax_eps", "eps_s_dot", "eps_D_kappa",
}


@dataclass
class VehicleConfig:
    """Shared vehicle parameters plus per-model overrides."""

    m: float = 160.0
    g: float = 9.81
    v_min: float = 0.4
    v_max: float = 20.0
    d_max: float = 3.0
    psi_err_max: float = 1.2
    mu: float = 1.2
    lf: float = 0.842
    lr: float = 0.689
    corners: Optional[List] = None

    v_eps: float = 0.5
    smoothmax_eps: float = 1e-3
    eps_s_dot: float = 1.0
    eps_D_kappa: float = 0.05

    point_mass: Optional[Dict[str, Any]] = None
    dynamic_bicycle: Optional[Dict[str, Any]] = None
    four_wheel: Optional[Dict[str, Any]] = None

    _yaml_keys: Optional[set] = field(default=None, repr=False)

    def _shared_as_dict(self) -> Dict[str, Any]:
        """Return shared fields as a flat dict (only non-None values)."""
        result = {}
        for name in _VEHICLE_SHARED_FIELDS:
            val = getattr(self, name)
            if val is not None:
                result[name] = val
        return result

    def build_model_params(self, model_name: str) -> Dict[str, Any]:
        """Build a flat params dict for the given model.

        Merge order: model defaults ← shared vehicle ← model-specific overrides.
        Raises ValueError if any override key is unknown to the model.

        Shared vehicle params that aren't in the model's defaults are silently
        skipped (not all models use every shared param, e.g. ``mu``).  But
        model-specific overrides must exactly match known params — a typo there
        is always an error.
        """
        if model_name not in _MODEL_CLASSES:
            raise ValueError(
                f"Unknown model_name: {model_name!r}. "
                f"Known models: {sorted(_MODEL_CLASSES)}"
            )

        default_keys = _get_model_default_keys(model_name)
        defaults = copy.deepcopy(
            _MODEL_CLASSES[model_name](params=None).get_default_params()
        )

        # Layer shared vehicle params (only those the model knows about)
        shared = self._shared_as_dict()
        for key, val in shared.items():
            if key in default_keys:
                defaults[key] = val

        # Layer model-specific overrides (strict — all must be known)
        overrides = getattr(self, model_name) or {}
        override_unknown = set(overrides.keys()) - default_keys
        if override_unknown:
            raise ValueError(
                f"Model-specific overrides for '{model_name}' contain "
                f"unknown params: {sorted(override_unknown)}. "
                f"Known params: {sorted(default_keys)}"
            )
        defaults.update(overrides)

        return defaults

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VehicleConfig":
        """Construct from a raw YAML dict, validating all keys."""
        known_fields = {f.name for f in fields(cls) if f.name != "_yaml_keys"}
        allowed_keys = known_fields | _MODEL_NAMES
        unknown = set(d.keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"Unknown keys in 'vehicle' section: {sorted(unknown)}. "
                f"Allowed keys: {sorted(allowed_keys)}"
            )

        # Validate model-specific sub-dicts
        for model_name in _MODEL_NAMES:
            sub = d.get(model_name)
            if sub is not None and not isinstance(sub, dict):
                raise ValueError(
                    f"vehicle.{model_name} must be a dict, got {type(sub).__name__}"
                )
            if isinstance(sub, dict):
                model_keys = _get_model_default_keys(model_name)
                unknown_model = set(sub.keys()) - model_keys
                if unknown_model:
                    raise ValueError(
                        f"Unknown keys in vehicle.{model_name}: "
                        f"{sorted(unknown_model)}. "
                        f"Known params: {sorted(model_keys)}"
                    )

        vc = cls(**d)
        vc._yaml_keys = set(d.keys()) & _VEHICLE_SHARED_FIELDS
        return vc

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val
        return result


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

_PIPELINE_FIELDS = {
    "track_id", "track_type", "ds_m", "continuity", "smooth_centerline",
    "mode", "model_name", "integrator_name",
    "reg_u", "reg_u_l2", "initial_speed",
    "boundary_margin", "autox_extension_m", "autox_start_x", "autox_start_y",
    "autox_start_node_offset", "autox_lead_in_m", "autox_ocp_lead_m",
    "autox_timing_offset_m",
    "use_savgol_bounds", "savgol_window_length", "savgol_polyorder",
    "normalize_states_and_inputs", "solver_verbose",
    "export_trajectory", "plot_results", "show_plots",
    # Warm start
    "warm_start", "warm_start_max_margin_gap", "warm_start_ladder_step",
    "warm_start_max_seeds", "warm_start_seed",
    # Skidpad-specific
    "skidpad_map_csv", "skidpad_reference_csv",
    "eps_time", "entry_exit_halfwidth", "kappa_blend_m", "terminal_speed",
    "decel_hold_m", "skidpad_start_x", "skidpad_start_y", "skidpad_lead_in_m",
    "skidpad_terminal_straight_m",
}


@dataclass
class RunConfig:
    """Top-level run configuration loaded from YAML."""

    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    # Pipeline / solver settings
    track_id: str = "fsg_random"
    track_type: str = "fsg"
    ds_m: float = 0.5
    continuity: str = "C2"
    smooth_centerline: int = 0
    mode: str = "trackdrive"
    model_name: str = "point_mass"
    integrator_name: str = "euler"
    reg_u: float = 600.0
    reg_u_l2: Optional[float] = None
    initial_speed: Optional[float] = None
    boundary_margin: float = 0.0
    autox_extension_m: float = 50.0
    # Car's real start position in the map frame (x, y), used to find the autox
    # horizon's anchor node instead of trusting the track CSV's arbitrary array
    # index 0 (see pipeline._resolve_autox_start_index). Default (0.0, 0.0):
    # this stack's SLAM pose-graph anchors the first pose at the origin, so
    # this is reliably close to the car's actual start regardless of which CSV
    # row the boundary-estimation tool happened to emit first.
    autox_start_x: float = 0.0
    autox_start_y: float = 0.0
    # Nodes to step forward (direction of travel) from the sample nearest
    # (autox_start_x, autox_start_y) before pinning it as the OCP's launch
    # node -- a small mesh-scale safety margin so the pin sits slightly ahead
    # of, not behind, the car. Default 1 (~0.5 m at ds_m=0.5).
    autox_start_node_offset: int = 1
    autox_lead_in_m: float = 0.0
    # Metres before the anchor node (see autox_start_x/y above) that the OCP's
    # own optimized horizon begins (instead of the flat autox_lead_in_m hold).
    # With the anchor now genuinely at the car's position, there's usually
    # nothing to gain by pushing the pin further back -- 0.0 is the normal
    # setting; only raise this if part of the approach itself needs to be
    # optimized rather than held flat.
    autox_ocp_lead_m: float = 0.0
    autox_timing_offset_m: float = 6.0

    # Skidpad-specific (only used when mode == "skidpad")
    skidpad_map_csv: Optional[str] = None
    skidpad_reference_csv: Optional[str] = None
    eps_time: float = 0.1
    entry_exit_halfwidth: float = 1.5
    kappa_blend_m: float = 1.5
    terminal_speed: Optional[float] = None
    # Metres of the exit/decel zone (from the finish gate) kept at the heavy timed
    # weight, delaying the terminal brake until after the finish line. 0.0 = off.
    decel_hold_m: float = 0.0
    # Overrides the entry point (P0), otherwise taken from the reference's first
    # row. skidpad_start_x=None keeps the original behaviour.
    skidpad_start_x: Optional[float] = None
    skidpad_start_y: float = 0.0
    # Metres of straight, prescribed constant-speed run-in prepended before the
    # OCP's s=0 (at initial_speed), not part of the optimization. 0.0 = off.
    skidpad_lead_in_m: float = 0.0
    # Metres before the finish that must stay centered (d) and heading-aligned
    # (psi_err) within a tight tolerance, so the trajectory ends straight
    # instead of at a residual angle. 0.0 = off.
    skidpad_terminal_straight_m: float = 0.0

    use_savgol_bounds: bool = False
    savgol_window_length: int = 41
    savgol_polyorder: int = 2
    normalize_states_and_inputs: bool = True
    solver_verbose: bool = False

    export_trajectory: bool = True
    plot_results: bool = True
    show_plots: bool = True

    # Warm start: "off" (no seed read, none written, no extra solves),
    # "auto" (seed when a compatible one exists, plus one intermediate solve
    # when the target margin is past the critical margin), or "ladder"
    # (always walk up from a safe margin, ignoring the store).
    warm_start: str = "auto"
    warm_start_max_margin_gap: float = 0.15
    warm_start_ladder_step: float = 0.05
    warm_start_max_seeds: int = 50
    warm_start_seed: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Load and validate a YAML config file."""
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")

        allowed_top = {"vehicle", "pipeline"}
        unknown_top = set(raw.keys()) - allowed_top
        if unknown_top:
            raise ValueError(
                f"Unknown top-level YAML sections: {sorted(unknown_top)}. "
                f"Allowed: {sorted(allowed_top)}"
            )

        # Parse vehicle section
        vehicle_raw = raw.get("vehicle", {})
        vehicle = VehicleConfig.from_dict(vehicle_raw) if vehicle_raw else VehicleConfig()

        # Parse pipeline section
        pipeline_raw = raw.get("pipeline", {})
        unknown_pipeline = set(pipeline_raw.keys()) - _PIPELINE_FIELDS
        if unknown_pipeline:
            raise ValueError(
                f"Unknown keys in 'pipeline' section: {sorted(unknown_pipeline)}. "
                f"Allowed keys: {sorted(_PIPELINE_FIELDS)}"
            )

        return cls(vehicle=vehicle, **pipeline_raw)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {}

        # Vehicle section
        data["vehicle"] = self.vehicle.to_dict()

        # Pipeline section
        pipeline = {}
        for name in _PIPELINE_FIELDS:
            val = getattr(self, name)
            if val is not None:
                pipeline[name] = val
        data["pipeline"] = pipeline

        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_pipeline_config(self):
        """Convert to a PipelineConfig for backwards compatibility."""
        from pipeline import PipelineConfig

        kwargs = {}
        for name in _PIPELINE_FIELDS:
            val = getattr(self, name, None)
            if val is not None:
                kwargs[name] = val

        # Attach vehicle config so _make_model can use it
        config = PipelineConfig(**kwargs)
        config.vehicle_config = self.vehicle
        return config

    def validate_for_model(self) -> None:
        """Validate that all vehicle params are consumed by the selected model.

        Call this before running the pipeline to catch misconfigurations early.
        Only checks params that were explicitly set in the YAML (not defaults).
        """
        yaml_keys = self.vehicle._yaml_keys
        if yaml_keys:
            default_keys = _get_model_default_keys(self.model_name)
            unused = yaml_keys - default_keys
            if unused:
                raise ValueError(
                    f"Vehicle params {sorted(unused)} are set in the config "
                    f"but model '{self.model_name}' does not use them. "
                    f"Move them to the model-specific section or remove them."
                )
        self.vehicle.build_model_params(self.model_name)
