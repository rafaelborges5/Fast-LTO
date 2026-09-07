"""Smooth stand-ins for non-smooth operations, for use inside the OCP.

IPOPT builds a quadratic model of the problem from first and second
derivatives, so an expression with a kink in it is a place the solver can stall
or oscillate. Where the physics wants a hard ``max`` -- flooring a denominator,
or refusing a negative tyre load -- the expression handed to the solver has to
be a rounded version of it instead.
"""

from __future__ import annotations

import casadi as ca


def smoothmax(a: ca.MX, b: ca.MX, eps: float) -> ca.MX:
    """A C1 approximation of ``max(a, b)``.

    Exactly ``max(a, b) = 0.5 * (a + b + |a - b|)`` with the absolute value
    replaced by ``sqrt((a - b)^2 + eps^2)``: a hyperbola that tracks ``|x|``
    closely but rounds off the corner at zero, where the derivative of ``|x|``
    jumps from -1 to +1.

    The approximation costs a small bias. At ``a == b`` it returns
    ``max + eps/2``, and the error falls off as roughly ``eps^2 / (4|a - b|)``
    away from the crossing, so it is negligible wherever the two arguments are
    not nearly equal -- which is everywhere the answer matters.

    Used in three places, all of them the same job (keep a denominator away
    from zero without giving the solver a corner to trip on):

    * the space-domain transform divides by ``ds/dt``, which must not reach
      zero (:mod:`fast_lto.optimization.integrators`);
    * slip angles divide by speed, floored at ``v_eps``
      (:class:`~fast_lto.vehicle_models.dynamic_bicycle.DynamicBicycleModel`);
    * per-wheel vertical loads are floored at ``Fz_min``, which is physical as
      well as numerical -- under enough load transfer an inside wheel lifts,
      and the tyre model must not be handed a negative load
      (:class:`~fast_lto.vehicle_models.four_wheel.FourWheelModel`).
    """
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps**2))
