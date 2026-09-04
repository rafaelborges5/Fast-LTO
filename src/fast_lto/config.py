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
from typing import Any, Dict, List, Optional

import yaml

from fast_lto.pipeline import PipelineConfig
from fast_lto.vehicle_models.dynamic_bicycle import DynamicBicycleModel
from fast_lto.vehicle_models.four_wheel import FourWheelModel
from fast_lto.vehicle_models.point_mass import PointMassModel

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
    "m",
    "g",
    "v_min",
    "v_max",
    "d_max",
    "psi_err_max",
    "mu",
    "lf",
    "lr",
    "corners",
    "v_eps",
    "smoothmax_eps",
    "eps_s_dot",
    "eps_D_kappa",
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
                f"Unknown model_name: {model_name!r}. " f"Known models: {sorted(_MODEL_CLASSES)}"
            )

        default_keys = _get_model_default_keys(model_name)
        defaults = copy.deepcopy(_MODEL_CLASSES[model_name](params=None).get_default_params())

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
                raise ValueError(f"vehicle.{model_name} must be a dict, got {type(sub).__name__}")
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

# PipelineConfig fields that are runtime plumbing rather than YAML settings:
# the caller supplies them (paths, the already-parsed vehicle config), so they
# are not accepted in — nor written to — a config file.
_NON_YAML_PIPELINE_FIELDS = frozenset(
    {
        "repo_root",
        "track_csv_override",
        "generate_track",
        "compute_bounds",
        "vehicle_config",
    }
)


def _pipeline_yaml_fields() -> set:
    """The YAML-settable keys of the ``pipeline`` section.

    Derived from ``PipelineConfig`` rather than hand-listed, so a new option is
    declared exactly once — next to its own documentation and default — and can
    never be silently dropped on load because someone forgot a second list.
    """
    return {f.name for f in fields(PipelineConfig) if not f.name.startswith("_")} - (
        _NON_YAML_PIPELINE_FIELDS
    )


@dataclass
class RunConfig:
    """A parsed config file: the ``vehicle`` section plus the ``pipeline`` one.

    The pipeline settings *are* a :class:`~fast_lto.pipeline.PipelineConfig`,
    not a parallel copy of its fields.  Reach them through ``config.pipeline``::

        run_config = RunConfig.from_yaml("configs/autox.yaml")
        run_config.pipeline.boundary_margin = 0.5
        pipeline_config = run_config.to_pipeline_config()
    """

    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

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

        vehicle_raw = raw.get("vehicle", {})
        vehicle = VehicleConfig.from_dict(vehicle_raw) if vehicle_raw else VehicleConfig()

        pipeline_raw = raw.get("pipeline", {})
        allowed_pipeline = _pipeline_yaml_fields()
        unknown_pipeline = set(pipeline_raw.keys()) - allowed_pipeline
        if unknown_pipeline:
            raise ValueError(
                f"Unknown keys in 'pipeline' section: {sorted(unknown_pipeline)}. "
                f"Allowed keys: {sorted(allowed_pipeline)}"
            )

        return cls(vehicle=vehicle, pipeline=PipelineConfig(**pipeline_raw))

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pipeline: Dict[str, Any] = {}
        for name in sorted(_pipeline_yaml_fields()):
            val = getattr(self.pipeline, name)
            if val is not None:
                pipeline[name] = val

        data: Dict[str, Any] = {"vehicle": self.vehicle.to_dict(), "pipeline": pipeline}
        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_pipeline_config(self) -> PipelineConfig:
        """Attach the vehicle config and return the pipeline settings.

        Re-runs ``__post_init__`` so paths and mode-dependent defaults reflect
        any field changed since load (a CLI flag overriding the YAML, say).
        Returns the live object, not a copy: later edits to it are edits to
        ``self.pipeline``.
        """
        self.pipeline.vehicle_config = self.vehicle
        self.pipeline.__post_init__()
        return self.pipeline

    def validate_for_model(self) -> None:
        """Validate that all vehicle params are consumed by the selected model.

        Call this before running the pipeline to catch misconfigurations early.
        Only checks params that were explicitly set in the YAML (not defaults).
        """
        yaml_keys = self.vehicle._yaml_keys
        if yaml_keys:
            default_keys = _get_model_default_keys(self.pipeline.model_name)
            unused = yaml_keys - default_keys
            if unused:
                raise ValueError(
                    f"Vehicle params {sorted(unused)} are set in the config "
                    f"but model '{self.pipeline.model_name}' does not use them. "
                    f"Move them to the model-specific section or remove them."
                )
        self.vehicle.build_model_params(self.pipeline.model_name)
