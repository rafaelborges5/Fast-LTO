"""Per-event behaviour for the pipeline.

Trackdrive, autox and skidpad are the same optimisation over different track
setups, so they used to be told apart by ``if config.mode == ...`` at every
call site that cared: the track preparation, the objective's time weights, five
solver arguments repeated in two places, and the prescribed segments stitched on
after the solve. Adding an event meant finding all of them.

Each mode now answers those questions itself:

``prepare_track``
    Reshape the discretised track before it reaches the OCP (autox extends the
    closed loop past the finish line; the others leave it alone).
``time_weights``
    Per-node weights on the objective's time term, or ``None`` for a plain
    minimum-time lap.
``solver_kwargs`` / ``seed_kwargs``
    The mode-specific arguments to ``solve_ocp_and_save`` and to the warm-start
    seed signature. Only keys the mode actually uses are returned, so the
    callee's own defaults cover the rest.
``splices``
    Prescribed constant-speed segments to stitch onto the solved trajectory
    (lead-ins, terminal pads). They are not part of the optimisation.

A mode that does none of these — trackdrive — inherits every default and is
empty on purpose.
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
#
# These used to be flat fields on PipelineConfig, prefixed by hand. Twenty-one
# of its fifty-two settings belonged to exactly one event, and three of those
# did not say so in their names -- `entry_exit_halfwidth` and `kappa_blend_m`
# read like general track settings and are skidpad-only, `D_safe_braking` like
# a vehicle limit and is autox-only. Nothing stopped an autox config from
# setting a skidpad option either: every field was accepted regardless of mode
# and the irrelevant ones were silently ignored.
#
# Each event now owns its settings, and `EventMode.config_of` reaches the block
# that belongs to it. A block that does not match the configured mode is an
# error rather than a no-op, and a setting has one obvious home.


@dataclass
class TimedEventConfig:
    """Settings shared by the events that are timed against a gate.

    Trackdrive has no such settings, which is why it has no config class at
    all: its closed-loop constraint already ties the lap together. That is also
    what used to make ``terminal_speed`` a special case -- it had to be
    rejected for trackdrive in ``PipelineConfig.__post_init__``, because a
    terminal speed on a closed lap silently pins the free launch speed at node
    0 as well. There is now no trackdrive field to set, so the check is gone.
    """

    #: Objective weight on time for nodes past the finish, so the run-off costs
    #: almost nothing compared to the timed section.
    eps_time: float = 0.1
    # Metres of the exit/decel zone (measured from the finish gate) that keep the
    # heavy timed time-weight, so the terminal brake starts AFTER the finish line
    # instead of bleeding back before it. 0.0 = original behaviour.
    decel_hold_m: float = 0.0
    #: Speed the trajectory must be at or under by the final node.
    terminal_speed: Optional[float] = None


@dataclass
class AutoxConfig(TimedEventConfig):
    """One timed lap from a standing start, with run-off to brake into."""

    #: Metres of run-off appended past the timing gate's second crossing.
    extension_m: float = 50.0
    # Car's real start position in the map frame (x, y), used to find the autox
    # horizon's anchor node instead of trusting the track CSV's arbitrary array
    # index 0 (see _resolve_autox_start_index). Default (0.0, 0.0): this
    # stack's SLAM pose-graph anchors the first pose at the origin, so this is
    # reliably close to the car's actual start regardless of which CSV row the
    # boundary-estimation tool happened to emit first.
    start_x: float = 0.0
    start_y: float = 0.0
    # Nodes to step forward (direction of travel) from the sample nearest
    # (start_x, start_y) before pinning it as the OCP's launch node -- a small
    # mesh-scale safety margin so the pin sits slightly ahead of, not behind,
    # the car. Default 1 (~0.5 m at ds_m=0.5).
    start_node_offset: int = 1
    #: Metres of prescribed constant-speed run-in before the OCP horizon.
    lead_in_m: float = 0.0
    # Metres before the anchor node (see start_x/y above) that the OCP's own
    # optimized horizon begins (instead of the flat lead_in_m hold). The pinned
    # launch condition moves back by this much; the flat hold is trimmed to sit
    # immediately before it. With the anchor now genuinely at the car's
    # position, there's usually nothing to gain by pushing the pin further back
    # -- 0.0 is the normal setting; only raise this if part of the approach
    # itself needs to be optimized rather than held flat.
    ocp_lead_m: float = 0.0
    # Distance (m) from the car's start position to the real timing gate; the
    # accurate autox lap time is measured between this point and the same point
    # one lap later, not from s=0 through the run-off extension.
    timing_offset_m: float = 6.0
    # If True, only the last terminal_window_nodes of the OCP horizon must land
    # centered (d) and heading-aligned (psi_err) within a tight tolerance --
    # unlike skidpad's terminal_straight_m, the path leading up to that short
    # window is left free.
    terminal_state_constraint: bool = False
    # Discrete steps the terminal-state constraint above is enforced over.
    # 1 was tried first and found infeasible for rate-limited actuator models
    # (e.g. four_wheel): the state can't snap to the target in zero steps, so
    # a couple of nodes of slack let the dynamics actually converge into it.
    terminal_window_nodes: int = 2
    # Metres of straight, prescribed constant-speed pad appended after the OCP
    # horizon (held at the solved terminal speed on the real centerline), not
    # part of the optimization -- reference margin in case the controller
    # tracks past the solved end. 0.0 = off.
    terminal_pad_m: float = 0.0
    # If set, replaces D_fl/D_fr/D_rr/D_rl (absolute override, not a scale)
    # with this single value for every OCP node with no timing objective --
    # i.e. the untimed tail after the timing gate's second crossing, same
    # boundary as `timed_mask`/_autox_time_weights. Everything else (B, C,
    # aero, mass, ...) stays at the nominal, racing-line value. Does not reach
    # the constant-speed pad appended after the OCP solve (terminal_pad_m) --
    # that pad isn't part of the OCP at all. None = off (nominal D used
    # everywhere).
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
    # Overrides the entry point (P0), otherwise taken from the reference's first
    # row. start_x=None keeps the original behaviour.
    start_x: Optional[float] = None
    start_y: float = 0.0
    # Metres of straight, prescribed constant-speed run-in prepended before the
    # OCP's s=0 (at initial_speed), not part of the optimization. 0.0 = off.
    lead_in_m: float = 0.0
    # Metres before the finish that must stay centered (d) and heading-aligned
    # (psi_err) within a tight tolerance, so the trajectory ends straight
    # instead of at a residual angle. 0.0 = off.
    terminal_straight_m: float = 0.0


@dataclass(frozen=True)
class SplicePlan:
    """One prescribed segment to stitch onto a solved trajectory.

    ``speed`` is resolved when the plan is built, which is why ``splices`` is
    handed the solution: the autox terminal pad runs at the speed the solver
    actually finished at, not at the ``terminal_speed`` target it was aiming for.
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
    """Resolve the autox horizon's anchor node ("s=0") from the car's real
    start position, instead of trusting the track CSV's array index 0.

    Index 0 is just whichever ``M`` row an upstream mapping/boundary-
    estimation tool happened to write first -- an artifact of that tool's
    internal conventions, not the car's actual position. It has been
    observed to drift by several metres between mapping sessions on the same
    physical track. Anchoring instead on the sample nearest ``(start_x,
    start_y)`` -- the car's real start pose in the map frame -- fixes that.

    ``node_offset`` steps the anchor forward (direction of travel) by that
    many additional samples past the nearest one, as a small safety margin
    so the OCP's pinned launch condition sits slightly ahead of the car
    rather than behind it.

    Returns ``(idx_ref, snap_distance_m)``: the resolved index, and the
    distance from ``(start_x, start_y)`` to the nearest sample (before
    applying ``node_offset``) -- a large snap distance is a sign the wrong
    track or wrong coordinates were used.
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

    ``s = 0`` -- and every offset measured "from s=0" below (``ocp_lead_m``,
    ``lead_in_m``, ``timing_offset_m``) -- is anchored at ``idx_ref``: the
    track sample nearest ``(start_x, start_y)``, stepped ``start_node_offset``
    nodes forward (see ``_resolve_autox_start_index``). This is deliberately
    *not* the track CSV's array index 0, which is an artifact of the
    upstream boundary-estimation tool's row ordering and has been observed
    to drift several metres between mapping sessions on the same physical
    track -- anchoring on the car's real start position instead removes that
    drift from the OCP's launch point, the lead-in, and the timing gate all
    at once, since they all move together.

    The OCP horizon itself (``positions``/``headings``/... fed to
    ``build_ocp``) gets a forward run-off plus, if ``ocp_lead_m`` > 0,
    ``ocp_lead_m`` metres of backward run-in before ``s=0`` (also borrowed
    from the tail of the closed loop, since the point just "before" s=0 on a
    closed track is, geometrically, the end of the loop). The OCP's pinned
    launch condition then lands ``ocp_lead_m`` metres earlier, so the solver
    optimizes the speed/steering profile through that stretch instead of it
    being a flat hold. With ``ocp_lead_m=0`` (the normal setting now that
    ``s=0`` sits at the car's real position) the pin lands exactly at
    ``idx_ref``.

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
    ``_splice_segment``, which stitches it onto the solved trajectory
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

    If ``terminal_pad_m`` > 0, geometry for a further constant-speed pad
    *after* the OCP horizon is also computed (from the same wraparound,
    continuing past the run-off) and returned under
    ``"autox_terminal_pad"``. Like ``autox_lead_in``, it is not part of the
    optimization: see ``_splice_segment``, which stitches it onto
    the solved trajectory at the solved terminal speed, purely as reference
    margin in case the controller tracks a little past the solved end.
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

    # Re-anchor s=0 at idx_ref (the car's real start) instead of array index
    # 0, by rotating local copies of every per-point array. This never
    # touches track_data itself -- the map/CSV/JSON on disk stay untouched --
    # only this function's disposable, per-solve working copy is reordered.
    # A circular roll preserves every adjacency relationship (including the
    # one wrap seam), so everything below this point -- the tail-of-the-loop
    # wraparound math, curvatures_half's inter-node midpoints, etc. -- is
    # unchanged and operates correctly on the rotated copies.
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
    # Uniform arc-length spacing survives a circular roll exactly.
    arc_lengths = np.arange(N_orig, dtype=np.float64) * ds_m

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
        # Continue past the run-off (index M_pts in the *original* closed
        # loop), wrapping around again if the pad is long enough to need it.
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


def _autox_time_weights(
    arc_lengths: Sequence[float] | np.ndarray,
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


def _skidpad_time_weights(
    track_data: Dict,
    eps_time: float,
    decel_hold_m: float,
    fallback_ds_m: float,
) -> np.ndarray:
    """Per-node objective time weights for skidpad, from the track's own masks.

    Timed nodes keep full weight, untimed ones drop to ``eps_time``, and the
    exit (decel) zone goes to zero so braking after the finish costs nothing.
    ``decel_hold_m`` then keeps full weight for that many metres of the exit, so
    the terminal brake starts after the finish line instead of bleeding back
    onto the last timed circle.
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

    #: The ``PipelineConfig`` attribute holding this event's settings, or None
    #: for an event that has none. Also the YAML key: a config naming a block
    #: that no mode claims, or one claimed by a different mode, is an error.
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

        The same knobs as ``solver_kwargs`` minus the ones the signature does
        not take, plus the shared timing weights. All of them describe how a
        solution was produced, so leaving one out would let two different
        solves share a cache entry.
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
        """Lines describing what this mode did to the problem.

        Returned rather than printed so the caller owns the output stream.
        """
        return []


class TrackdriveMode(EventMode):
    """A closed flying lap: no run-in, no run-off, no terminal condition.

    Deliberately empty. The closed-loop constraint ``X[N-1] == X[0]`` already
    ties the lap together, and ``terminal_speed`` is rejected for this mode in
    ``PipelineConfig.__post_init__`` because it would silently pin the free
    launch speed at node 0 as well.
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
        # Flows through to the solution JSON via the generic
        # `track.get("timed_mask")` in solve_ocp_and_save (the same field
        # skidpad uses), so visualization can shade the post-finish untimed
        # zone -- and so D_safe_braking knows which nodes are untimed.
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
                    # Sits before the timing gate, so it counts as timed under
                    # the same convention as the rest of the run-up.
                    timed=1,
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
        # Held nodes are exactly the decel nodes the weighting left at full
        # weight -- read back off the result instead of recomputing the rule.
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
                # Sits before the gate, same as the entry straight it extends.
                timed=0,
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
