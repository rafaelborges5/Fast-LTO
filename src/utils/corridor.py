"""
Geometric feasibility of the lateral corridor, for a given ``boundary_margin``.

Mirrors ``VehicleModel.get_corner_constraints`` (see
``src/vehicle_models/vehicle_base.py``) on a discretized track, so the pipeline
can answer two questions before the solver ever runs:

* how wide is the feasible interval in ``d`` at each station, and
* at which margin does the default initial guess (``d = 0``, ``psi_err = 0``)
  stop being feasible.

The second one matters because ``solve_ocp_and_save`` starts every cold solve
from exactly that point: above the critical margin the solve begins outside the
feasible set, which is slow and tends to land in a worse local minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from ..vehicle_models.vehicle_base import CornerOffset  # type: ignore
except ImportError:  # script-style imports without package context
    from vehicle_models.vehicle_base import CornerOffset  # type: ignore


@dataclass(frozen=True)
class CriticalMargin:
    """Where the ``psi_err = 0`` corridor closes."""

    margin: float
    index: int
    arc_length_m: float
    kappa: float
    width_at_zero_margin: float

    @property
    def already_closed(self) -> bool:
        """True when the car does not fit even with no margin at all."""
        return self.width_at_zero_margin <= 0.0


def corridor_at(
    kappa: float,
    w_left: float,
    w_right: float,
    corners: Sequence[CornerOffset],
    psi_err: float = 0.0,
    iters: int = 3,
) -> Tuple[float, float]:
    """Feasible ``[d_lo, d_hi]`` at one station with all corners inside.

    ``w_left`` / ``w_right`` are the margin-shrunk half widths. The curvature
    term in the corner constraint divides by ``D_kappa = 1 - kappa * d``, which
    depends on the unknown ``d``; a few fixed-point passes around the interval
    centre are enough (the term is a small correction).
    """
    sin_p, cos_p = np.sin(psi_err), np.cos(psi_err)
    d_mid = 0.0
    lo = hi = 0.0
    for _ in range(max(iters, 1)):
        d_kappa = 1.0 - kappa * d_mid
        if abs(d_kappa) < 1e-9:
            d_kappa = 1e-9 if d_kappa >= 0.0 else -1e-9
        hi, lo = np.inf, -np.inf
        for corner in corners:
            long_proj = corner.dx * cos_p - corner.dy * sin_p
            shift = (
                corner.dx * sin_p
                + corner.dy * cos_p
                - 0.5 * kappa / d_kappa * long_proj**2
            )
            if corner.dy >= 0.0:
                hi = min(hi, w_left - shift)
            else:
                lo = max(lo, -w_right - shift)
        d_mid = 0.5 * (lo + hi)
    return lo, hi


def corridor_widths(
    track: Dict,
    corners: Sequence[CornerOffset],
    margin: float,
    psi_err: float = 0.0,
) -> np.ndarray:
    """Feasible ``d`` interval width at every station (negative = infeasible)."""
    kappa = np.asarray(track["curvatures"], dtype=float)
    w_left = np.asarray(track["w_left"], dtype=float) - margin
    w_right = np.asarray(track["w_right"], dtype=float) - margin

    widths = np.empty(len(kappa), dtype=float)
    for i in range(len(kappa)):
        lo, hi = corridor_at(kappa[i], w_left[i], w_right[i], corners, psi_err)
        widths[i] = hi - lo
    return widths


def critical_margin(
    track: Dict,
    corners: Sequence[CornerOffset],
    psi_err: float = 0.0,
    margin_max: float = 1.0,
    tol: float = 1e-4,
) -> CriticalMargin:
    """Largest margin for which every station still admits ``psi_err``.

    The corridor shrinks monotonically with the margin (it is subtracted from
    both sides), so a bisection on ``min_station(width) = 0`` is exact.
    """
    if not corners:
        raise ValueError("critical_margin needs corner offsets; the model has none")

    def min_width(margin: float) -> float:
        return float(corridor_widths(track, corners, margin, psi_err).min())

    width_0 = min_width(0.0)
    arc = np.asarray(track["arc_lengths"], dtype=float)
    kappa = np.asarray(track["curvatures"], dtype=float)

    if width_0 <= 0.0:
        idx = int(np.argmin(corridor_widths(track, corners, 0.0, psi_err)))
        return CriticalMargin(0.0, idx, float(arc[idx]), float(kappa[idx]), width_0)

    if min_width(margin_max) > 0.0:
        idx = int(np.argmin(corridor_widths(track, corners, margin_max, psi_err)))
        return CriticalMargin(
            margin_max, idx, float(arc[idx]), float(kappa[idx]), width_0
        )

    lo, hi = 0.0, margin_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if min_width(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    widths = corridor_widths(track, corners, hi, psi_err)
    idx = int(np.argmin(widths))
    return CriticalMargin(lo, idx, float(arc[idx]), float(kappa[idx]), width_0)


def describe_critical_margin(crit: CriticalMargin, target_margin: float) -> str:
    """One-line summary for the pipeline log."""
    if crit.already_closed:
        return (
            f"centreline initial guess is infeasible at any margin "
            f"(station s = {crit.arc_length_m:.1f} m, kappa = {crit.kappa:+.3f}, "
            f"corridor {crit.width_at_zero_margin:+.3f} m at margin 0)"
        )
    verdict = (
        "continuation will be used"
        if target_margin > crit.margin
        else "cold start is fine"
    )
    return (
        f"centreline initial guess feasible up to boundary_margin "
        f"{crit.margin:.2f} m (limiting station s = {crit.arc_length_m:.1f} m, "
        f"kappa = {crit.kappa:+.3f}) — target {target_margin:.2f}, {verdict}"
    )
