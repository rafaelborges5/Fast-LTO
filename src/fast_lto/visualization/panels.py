"""Which panel figure to draw for a solved trajectory, and with what.

Each vehicle model has its own state vector, so each needs its own panels and
its own fields out of the solution JSON. A renderer owns that knowledge, and
``PANEL_RENDERERS`` maps a ``model_name`` to one -- mirroring how
``pipeline._make_model`` maps the same string to a model class.

Adding a model means adding a renderer here and an entry in the registry. The
pipeline does not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np


def _as_array(values: Optional[list]) -> Optional[np.ndarray]:
    """A solution field as an array, or None when the solve did not set it."""
    return None if values is None else np.asarray(values)


@dataclass(frozen=True)
class PanelInputs:
    """The solved trajectory, plus everything shared by every panel figure."""

    data: Dict
    cones_left: np.ndarray
    cones_right: np.ndarray
    path_xy: np.ndarray
    v: np.ndarray
    s: np.ndarray
    d: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray
    params: Dict
    profiling: Optional[Dict]
    timed_mask: Optional[np.ndarray]
    out_path: Path
    show: bool

    def array(self, name: str) -> np.ndarray:
        """A named state or input from the solution, as a float array."""
        return np.array(self.data[name], dtype=np.float64)


def _render_point_mass(p: PanelInputs) -> None:
    from fast_lto.visualization.ocp_plots import _compute_constraint_activity, plot_all_panels

    a_long = p.array("a_long")
    a_lat = p.array("a_lat")
    mu_g = p.params.get("mu", 1.2) * p.params.get("g", 9.81)

    constraint_activity = _compute_constraint_activity(
        d=p.d,
        w_left=p.w_left,
        w_right=p.w_right,
        a_long=a_long,
        a_lat=a_lat,
        v=p.v,
        params=p.params,
    )

    plot_all_panels(
        p.cones_left,
        p.cones_right,
        p.path_xy,
        p.v,
        p.s,
        p.d,
        p.w_left,
        p.w_right,
        a_long,
        a_lat,
        mu_g,
        profiling=p.profiling,
        constraint_activity=constraint_activity,
        timed_mask=p.timed_mask,
        out_path=p.out_path,
        show=p.show,
    )


def _render_dynamic_bicycle(p: PanelInputs) -> None:
    from fast_lto.vehicle_models.diagnostics import evaluate_diagnostics
    from fast_lto.vehicle_models.dynamic_bicycle import DynamicBicycleModel
    from fast_lto.visualization.ocp_plots_dynamic_bicycle import plot_all_panels_dynamic_bicycle

    diagnostics = evaluate_diagnostics(DynamicBicycleModel(params=dict(p.params)), p.data)

    plot_all_panels_dynamic_bicycle(
        p.cones_left,
        p.cones_right,
        p.path_xy,
        p.v,
        p.s,
        p.d,
        p.w_left,
        p.w_right,
        p.array("a_long"),
        p.array("delta"),
        p.array("v_lat"),
        p.array("yaw_rate"),
        params=p.params,
        diagnostics=diagnostics,
        profiling=p.profiling,
        timed_mask=p.timed_mask,
        out_path=p.out_path,
        show=p.show,
    )


def _render_four_wheel(p: PanelInputs) -> None:
    from fast_lto.vehicle_models.diagnostics import evaluate_diagnostics
    from fast_lto.vehicle_models.four_wheel import FourWheelModel
    from fast_lto.visualization.ocp_plots_four_wheel import plot_all_panels_four_wheel

    input_names = p.data.get("input_names", [])
    input_data = {name: p.data[name] for name in input_names if name in p.data}

    # Rebuild the model from the parameters the solution was produced with, so
    # the tire loads and slip angles plotted are the ones the solver used.
    diagnostics = evaluate_diagnostics(FourWheelModel(params=dict(p.params)), p.data)

    plot_all_panels_four_wheel(
        p.cones_left,
        p.cones_right,
        p.path_xy,
        p.v,
        p.s,
        p.d,
        p.w_left,
        p.w_right,
        p.array("Fx_fl"),
        p.array("Fx_fr"),
        p.array("Fx_rr"),
        p.array("Fx_rl"),
        p.array("delta"),
        p.array("v_lat"),
        p.array("yaw_rate"),
        params=p.params,
        diagnostics=diagnostics,
        profiling=p.profiling,
        input_data=input_data,
        timed_mask=p.timed_mask,
        out_path=p.out_path,
        show=p.show,
    )


PANEL_RENDERERS: Dict[str, Callable[[PanelInputs], None]] = {
    "point_mass": _render_point_mass,
    "dynamic_bicycle": _render_dynamic_bicycle,
    "four_wheel": _render_four_wheel,
}


def render_panels(
    model_name: str,
    data: Dict,
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    out_path: Path,
    show: bool,
) -> None:
    """Draw the panel figure for a solved trajectory."""
    try:
        renderer = PANEL_RENDERERS[model_name]
    except KeyError:
        raise ValueError(
            f"No visualization available for model_name={model_name!r}. "
            f"Supported: {', '.join(repr(n) for n in sorted(PANEL_RENDERERS))}."
        ) from None

    renderer(
        PanelInputs(
            data=data,
            cones_left=cones_left,
            cones_right=cones_right,
            path_xy=np.array(data["path_xy"], dtype=np.float64),
            v=np.array(data.get("v", data.get("v_long")), dtype=np.float64),
            s=np.array(data["arc_lengths"], dtype=np.float64),
            d=np.array(data["d"], dtype=np.float64),
            w_left=np.array(data["w_left"], dtype=np.float64),
            w_right=np.array(data["w_right"], dtype=np.float64),
            params=data.get("model_params", {}),
            profiling=data.get("profiling"),
            timed_mask=_as_array(data.get("timed_mask")),
            out_path=out_path,
            show=show,
        )
    )
