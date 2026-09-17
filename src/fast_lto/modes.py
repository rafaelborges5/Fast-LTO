"""Per-event behaviour for the pipeline.

Trackdrive, autox and skidpad are the same optimisation over different track
setups. Each ``EventMode`` answers what its own event needs -- how to reshape
the track, how to weight the time objective, which solver arguments to pass,
which prescribed segments to stitch on afterwards -- so no call site branches on
``config.mode``. Trackdrive inherits every default and is empty on purpose.

See ``README.md`` in this directory for what each event asks of the solver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from fast_lto.pipeline import PipelineConfig


# ---------------------------------------------------------------------------
# Per-event settings
# ---------------------------------------------------------------------------


@dataclass
class TimedEventConfig:
    """Settings shared by the events that are timed against a gate.

    Trackdrive has none of these, which is why it has no config class at all:
    its closed-loop constraint already ties the lap together.
    """

    #: Objective weight on time past the finish, so the run-off costs little.
    eps_time: float = 0.1
    #: Metres past the gate that keep full time weight, so the terminal brake
    #: starts after the finish line instead of bleeding back before it.
    decel_hold_m: float = 0.0
    #: Speed the trajectory must be at or under by the final node.
    terminal_speed: Optional[float] = None


@dataclass
class AutoxConfig(TimedEventConfig):
    """One timed lap from a standing start, with run-off to brake into."""

    #: Metres of run-off appended past the timing gate's second crossing.
    extension_m: float = 50.0
    #: The car's real start position in the map frame, which anchors ``s = 0``
    #: instead of the track CSV's arbitrary first row.
    start_x: float = 0.0
    start_y: float = 0.0
    #: Nodes to step forward from the sample nearest ``(start_x, start_y)``, so
    #: the pinned launch node sits just ahead of the car rather than behind it.
    start_node_offset: int = 1
    #: Metres of prescribed constant-speed run-in before the OCP horizon.
    lead_in_m: float = 0.0
    #: Metres of the approach to optimise rather than hold flat, moving the
    #: pinned launch node back from ``s = 0``. Normally 0.
    ocp_lead_m: float = 0.0
    #: Metres from the start position to the real timing gate, which the lap
    #: time is measured between on consecutive crossings.
    timing_offset_m: float = 6.0
    #: Require the end of the horizon to be centred and heading-aligned, over
    #: the last ``terminal_window_nodes`` only; the approach stays free.
    terminal_state_constraint: bool = False
    #: Nodes that constraint is spread over. One is infeasible for rate-limited
    #: actuators, which cannot snap to the target in zero steps.
    terminal_window_nodes: int = 2
    #: Metres of prescribed constant-speed pad appended after the horizon, held
    #: at the solved terminal speed. 0 = off.
    terminal_pad_m: float = 0.0
    #: Replaces the tyre ``D`` outright (not a scale) on the untimed nodes
    #: past the gate. None = nominal grip everywhere.
    D_safe_braking: Optional[float] = None


@dataclass
class SkidpadConfig(TimedEventConfig):
    """Two timed circles each way, built from a cone map and a reference line."""

    #: Cone map and reference trajectory the track is built from. Paths are
    #: resolved against the data root when relative.
    map_csv: Optional[str] = None
    reference_csv: Optional[str] = None
    #: Half-width of the entry and exit straights, in metres.
    entry_exit_halfwidth: float = 1.5
    #: Metres over which curvature blends between straight and circle.
    kappa_blend_m: float = 1.5
    #: Entry point, otherwise taken from the reference's first row.
    start_x: Optional[float] = None
    start_y: float = 0.0
    #: Metres of prescribed constant-speed run-in before the OCP's ``s = 0``.
    lead_in_m: float = 0.0
    #: Metres before the finish that must stay centred and heading-aligned, so
    #: the trajectory ends straight instead of at a residual angle. 0 = off.
    terminal_straight_m: float = 0.0


@dataclass(frozen=True)
class SplicePlan:
    """One prescribed segment to stitch onto a solved trajectory.

    ``speed`` is resolved when the plan is built, which is why ``splices`` is
    handed the solution: the autox terminal pad runs at the speed the solver
    finished at, not the ``terminal_speed`` it was aiming for.
    """

    segment: Dict
    speed: float
    side: Literal["before", "after"]
    timed: int
    message: str


# ---------------------------------------------------------------------------
# Mode-specific track geometry and weighting
# ---------------------------------------------------------------------------


def _resolve_autox_start_index(
    positions: np.ndarray,
    start_x: float,
    start_y: float,
    node_offset: int = 1,
) -> Tuple[int, float]:
    """Resolve the autox horizon's anchor node from the car's start position.

    Returns ``(idx_ref, snap_distance_m)``. A large snap distance means the
    wrong track or the wrong coordinates were used.
    """
    n = positions.shape[0]
    d2 = (positions[:, 0] - start_x) ** 2 + (positions[:, 1] - start_y) ** 2
    nearest = int(np.argmin(d2))
    idx_ref = (nearest + int(node_offset)) % n
    return idx_ref, float(np.sqrt(d2[nearest]))


def _extend_track_for_autox(
    track_data: Dict,
    extension_m: float,
    lead_in_m: float = 0.0,
    ocp_lead_m: float = 0.0,
    timing_offset_m: float = 0.0,
    terminal_pad_m: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 0.0,
    start_node_offset: int = 1,
) -> Dict:
    """Extend a closed-loop track by wrapping points beyond the finish line.

    ``s = 0``, and every offset measured from it, is anchored at the car's
    start position rather than the CSV's first row. The run-off covers
    ``timing_offset_m + extension_m`` metres so that ``extension_m`` of it
    remains past the *gate*, which is where the timed lap actually ends.

    The lead-in and terminal pad are returned alongside the horizon rather
    than made part of it: they are prescribed constant-speed segments that
    ``_splice_segment`` stitches on after the solve. See ``README.md``.
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
    w_left = np.array(track_data["w_left"], dtype=np.float64)
    w_right = np.array(track_data["w_right"], dtype=np.float64)

    # A roll re-anchors s=0 and preserves every adjacency. Local copies only.
    idx_ref, start_snap_m = _resolve_autox_start_index(
        positions, start_x, start_y, start_node_offset
    )
    if idx_ref != 0:
        positions = np.roll(positions, -idx_ref, axis=0)
        headings = np.roll(headings, -idx_ref, axis=0)
        curvatures = np.roll(curvatures, -idx_ref, axis=0)
        curvatures_half = np.roll(curvatures_half, -idx_ref, axis=0)
        w_left = np.roll(w_left, -idx_ref, axis=0)
        w_right = np.roll(w_right, -idx_ref, axis=0)
    arc_lengths = np.arange(N_orig, dtype=np.float64) * ds_m

    total_length = arc_lengths[-1] + ds_m

    # Tail of the loop: [..lead_in K_pts..][..ocp_lead J_pts..][s=0..]
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
    extended["arc_lengths"] = np.concatenate(
        [
            arc_lengths[ocp_lead_start:] - total_length,
            arc_lengths,
            arc_lengths[:M_pts] + total_length,
        ]
    ).tolist()
    extended["w_left"] = np.concatenate([w_left[ocp_lead_start:], w_left, w_left[:M_pts]]).tolist()
    extended["w_right"] = np.concatenate(
        [w_right[ocp_lead_start:], w_right, w_right[:M_pts]]
    ).tolist()
    extended["num_points"] = N_orig + M_pts + J_pts
    extended["total_length_m"] = float(arc_lengths[-1] + ds_m * M_pts + ds_m)
    extended["autox_base_length_m"] = float(total_length)
    extended["autox_idx_ref"] = int(idx_ref)
    extended["autox_start_snap_m"] = float(start_snap_m)

    if K_pts > 0:
        extended["autox_lead_in"] = {
            "positions": positions[lead_in_start:ocp_lead_start].tolist(),
            "headings": headings[lead_in_start:ocp_lead_start].tolist(),
            "curvatures": curvatures[lead_in_start:ocp_lead_start].tolist(),
            "arc_lengths": (arc_lengths[lead_in_start:ocp_lead_start] - total_length).tolist(),
            "w_left": w_left[lead_in_start:ocp_lead_start].tolist(),
            "w_right": w_right[lead_in_start:ocp_lead_start].tolist(),
        }

    if terminal_pad_m > 0.0:
        P_pts = max(1, round(terminal_pad_m / ds_m))
        # Continues past the run-off, wrapping again if the pad is long enough.
        pad_idx = (M_pts + np.arange(P_pts)) % N_orig
        last_arc = extended["arc_lengths"][-1]
        extended["autox_terminal_pad"] = {
            "positions": positions[pad_idx].tolist(),
            "headings": headings[pad_idx].tolist(),
            "curvatures": curvatures[pad_idx].tolist(),
            "arc_lengths": [last_arc + ds_m * (i + 1) for i in range(P_pts)],
            "w_left": w_left[pad_idx].tolist(),
            "w_right": w_right[pad_idx].tolist(),
        }

    return extended


def _build_skidpad_lead_in(track_data: Dict, lead_in_m: float) -> Optional[Dict]:
    """Geometry for a straight, prescribed run-in before the skidpad's ``s = 0``.

    The skidpad centerline already starts on a straight, so the lead-in is that
    straight extrapolated backwards. Requires ``curvature[0] == 0``.
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


def _autox_time_weights(
    arc_lengths: Sequence[float] | np.ndarray,
    base_length_m: float,
    timing_offset_m: float,
    eps_time: float,
    decel_hold_m: float,
) -> np.ndarray:
    """Per-node time weights for autox: full through the lap, ``eps_time`` after.

    The finish line is the timing gate's second crossing rather than a
    track-provided mask, which is what skidpad uses instead.
    """
    arc = np.asarray(arc_lengths, dtype=float)
    gate2 = float(base_length_m) + float(timing_offset_m)
    weights = np.where(arc < gate2, 1.0, float(eps_time))
    if decel_hold_m > 0.0:
        hold_end = gate2 + float(decel_hold_m)
        weights = np.where((arc >= gate2) & (arc < hold_end), 1.0, weights)
    return weights


def _skidpad_time_weights(
    track_data: Dict,
    eps_time: float,
    decel_hold_m: float,
    fallback_ds_m: float,
) -> np.ndarray:
    """Per-node time weights for skidpad, from the track's own masks.

    Timed nodes keep full weight, untimed ones drop to ``eps_time``, and the
    exit zone goes to zero so braking after the finish costs nothing.
    """
    mask = np.asarray(track_data["timed_mask"], dtype=float)
    decel = np.asarray(track_data.get("decel_mask", np.zeros_like(mask)), dtype=float)

    weights = np.where(mask > 0.5, 1.0, float(eps_time))
    weights = np.where(decel > 0.5, 0.0, weights)

    if decel_hold_m > 0.0:
        ds_hold = float(track_data.get("ds_m", fallback_ds_m))
        decel_idx = np.where(decel > 0.5)[0]
        n_hold = min(int(round(decel_hold_m / ds_hold)), decel_idx.size)
        if n_hold > 0:
            weights[decel_idx[:n_hold]] = 1.0

    return weights


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class EventMode(ABC):
    """What one Formula Student event needs from the pipeline."""

    #: The ``PipelineConfig`` attribute holding this event's settings, and the
    #: YAML key it is read from. None for an event that has no settings.
    config_attr: Optional[str] = None

    #: The dataclass that block is parsed into.
    config_type: Optional[type] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """The ``mode`` string this class implements."""

    def config_of(self, config: "PipelineConfig") -> Any:
        """This event's settings block, or None if it has no settings."""
        if self.config_attr is None:
            return None
        return getattr(config, self.config_attr)

    def prepare_track(self, track_data: Dict, config: "PipelineConfig") -> Dict:
        """Return the track the OCP should be built over."""
        return track_data

    def time_weights(self, track_data: Dict, config: "PipelineConfig") -> Optional[np.ndarray]:
        """Per-node weights on the objective's time term."""
        return None

    def solver_kwargs(self, config: "PipelineConfig") -> Dict[str, Any]:
        """Mode-specific arguments for ``solve_ocp_and_save``."""
        return {}

    def seed_kwargs(self, config: "PipelineConfig") -> Dict[str, Any]:
        """Mode-specific arguments for the warm-start seed signature.

        Every knob that changes how a solution was produced has to appear here,
        or two different solves would share a cache entry.
        """
        kwargs = {
            k: v for k, v in self.solver_kwargs(config).items() if k != "autox_timing_offset_m"
        }
        event = self.config_of(config)
        if isinstance(event, TimedEventConfig):
            kwargs.setdefault("eps_time", event.eps_time)
            kwargs.setdefault("decel_hold_m", event.decel_hold_m)
        return kwargs

    def splices(
        self, config: "PipelineConfig", track_data: Dict, sol_dict: Dict
    ) -> List[SplicePlan]:
        """Prescribed segments to stitch onto the solved trajectory."""
        return []

    def summary(
        self,
        config: "PipelineConfig",
        track_data: Dict,
        time_weights: Optional[np.ndarray],
    ) -> List[str]:
        """Lines describing what this mode did to the problem, for the caller
        to print."""
        return []


class TrackdriveMode(EventMode):
    """A closed flying lap: no run-in, no run-off, no terminal condition.

    Deliberately empty -- the closed-loop constraint ``X[N-1] == X[0]`` already
    ties the lap together.
    """

    name = "trackdrive"


class AutoxMode(EventMode):
    """One timed lap from a standing start, with run-off to brake into."""

    name = "autox"
    config_attr = "autox"
    config_type = AutoxConfig

    def prepare_track(self, track_data: Dict, config: "PipelineConfig") -> Dict:
        autox = config.autox
        return _extend_track_for_autox(
            track_data,
            autox.extension_m,
            autox.lead_in_m,
            autox.ocp_lead_m,
            timing_offset_m=autox.timing_offset_m,
            terminal_pad_m=autox.terminal_pad_m,
            start_x=autox.start_x,
            start_y=autox.start_y,
            start_node_offset=autox.start_node_offset,
        )

    def time_weights(self, track_data: Dict, config: "PipelineConfig") -> Optional[np.ndarray]:
        autox = config.autox
        weights = _autox_time_weights(
            track_data["arc_lengths"],
            track_data["autox_base_length_m"],
            autox.timing_offset_m,
            autox.eps_time,
            autox.decel_hold_m,
        )
        # The same field skidpad's track carries; see the package README.
        track_data["timed_mask"] = (weights >= 1.0 - 1e-9).astype(int).tolist()
        return weights

    def solver_kwargs(self, config: "PipelineConfig") -> Dict[str, Any]:
        autox = config.autox
        return {
            "terminal_speed": autox.terminal_speed,
            "autox_timing_offset_m": autox.timing_offset_m,
            "terminal_state_constraint": autox.terminal_state_constraint,
            "terminal_window_nodes": autox.terminal_window_nodes,
            "D_safe_braking": autox.D_safe_braking,
        }

    def summary(
        self,
        config: "PipelineConfig",
        track_data: Dict,
        time_weights: Optional[np.ndarray],
    ) -> List[str]:
        autox = config.autox
        lines = [
            f"  Autox: extended track by "
            f"{autox.timing_offset_m + autox.extension_m:.0f} m "
            f"({track_data['num_points']} points total, OCP horizon, "
            f"{autox.ocp_lead_m:.1f} m of which is backward run-in, "
            f"{autox.extension_m:.0f} m of which is post-finish run-off)",
            f"  Autox: start anchored at idx_ref={track_data['autox_idx_ref']} "
            f"(snapped {track_data['autox_start_snap_m']:.2f} m from "
            f"requested ({autox.start_x:.2f}, {autox.start_y:.2f}))",
        ]
        if time_weights is not None:
            n_timed = int(np.sum(time_weights >= 1.0 - 1e-9))
            lines.append(
                f"  Autox: {n_timed}/{len(time_weights)} timed nodes, "
                f"un-timed weight eps_time={autox.eps_time}, "
                f"decel_hold={autox.decel_hold_m} m, "
                f"terminal_speed={autox.terminal_speed}"
            )
        return lines

    def splices(
        self, config: "PipelineConfig", track_data: Dict, sol_dict: Dict
    ) -> List[SplicePlan]:
        plans: List[SplicePlan] = []

        lead_in = track_data.get("autox_lead_in")
        if lead_in:
            plans.append(
                SplicePlan(
                    segment=lead_in,
                    speed=config.launch_speed,
                    side="before",
                    timed=1,  # before the gate, like the rest of the run-up
                    message=(
                        f"  Autox: prepended {config.autox.lead_in_m:.0f} m constant-speed "
                        f"lead-in ({len(lead_in['arc_lengths'])} points, "
                        "not part of the OCP solve)"
                    ),
                )
            )

        pad = track_data.get("autox_terminal_pad")
        if pad:
            v_name = "v" if "v" in sol_dict["state_names"] else "v_long"
            pad_speed = float(sol_dict[v_name][-1])
            plans.append(
                SplicePlan(
                    segment=pad,
                    speed=pad_speed,
                    side="after",
                    timed=0,
                    message=(
                        f"  Autox: appended {config.autox.terminal_pad_m:.0f} m constant-speed "
                        f"({pad_speed:.2f} m/s) terminal pad "
                        f"({len(pad['arc_lengths'])} points, not part of the OCP solve)"
                    ),
                )
            )

        return plans


class SkidpadMode(EventMode):
    """Two timed circles each way, with the timing taken from the track's mask."""

    name = "skidpad"
    config_attr = "skidpad"
    config_type = SkidpadConfig

    def time_weights(self, track_data: Dict, config: "PipelineConfig") -> Optional[np.ndarray]:
        skidpad = config.skidpad
        return _skidpad_time_weights(
            track_data, skidpad.eps_time, skidpad.decel_hold_m, config.ds_m
        )

    def solver_kwargs(self, config: "PipelineConfig") -> Dict[str, Any]:
        return {
            "terminal_speed": config.skidpad.terminal_speed,
            "terminal_straight_m": config.skidpad.terminal_straight_m,
        }

    def summary(
        self,
        config: "PipelineConfig",
        track_data: Dict,
        time_weights: Optional[np.ndarray],
    ) -> List[str]:
        if time_weights is None:
            return []
        mask = np.asarray(track_data["timed_mask"], dtype=float)
        decel = np.asarray(track_data.get("decel_mask", np.zeros_like(mask)), dtype=float)
        # Read back off the weights instead of recomputing the hold rule.
        n_hold = int(np.sum((decel > 0.5) & (time_weights >= 1.0 - 1e-9)))
        return [
            f"  Skidpad: {int(mask.sum())}/{len(mask)} timed nodes, "
            f"{int(decel.sum())} exit (decel) nodes, "
            f"un-timed weight eps_time={config.skidpad.eps_time}, "
            f"decel_hold={config.skidpad.decel_hold_m} m ({n_hold} exit nodes held), "
            f"terminal_speed={config.skidpad.terminal_speed}"
        ]

    def splices(
        self, config: "PipelineConfig", track_data: Dict, sol_dict: Dict
    ) -> List[SplicePlan]:
        lead_in = _build_skidpad_lead_in(track_data, config.skidpad.lead_in_m)
        if not lead_in:
            return []
        return [
            SplicePlan(
                segment=lead_in,
                speed=config.launch_speed,
                side="before",
                timed=0,  # before the gate, like the entry straight it extends
                message=(
                    f"  Skidpad: prepended {config.skidpad.lead_in_m:.1f} m constant-speed "
                    f"({config.launch_speed:.1f} m/s) lead-in "
                    f"({len(lead_in['arc_lengths'])} points, not part of the OCP solve)"
                ),
            )
        ]


_MODES: Dict[str, EventMode] = {
    mode.name: mode for mode in (TrackdriveMode(), AutoxMode(), SkidpadMode())
}

MODE_NAMES = tuple(_MODES)


def get_mode(name: str) -> EventMode:
    """Look up the mode implementation for a ``config.mode`` string."""
    try:
        return _MODES[name]
    except KeyError:
        raise ValueError(f"Unknown mode: {name!r}. Must be one of {sorted(_MODES)}.") from None
