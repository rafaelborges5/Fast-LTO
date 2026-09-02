"""
Space-domain integrators for the OCP.

All integrators advance the reduced state (excluding *s*) by one spatial step:

    x_{i+1} = step(f_space, x_i, u_i, kappa_i, ds)

and provide a matching time-quadrature estimate for the same interval:

    dt_i = time_step(f_space, eval_at_point, x_i, u_i, kappa_i, ds)

where ``f_space(x, u, kappa)`` returns dx/ds (space derivatives of the reduced
state) and ``eval_at_point(x, u, kappa)`` returns ``(full_state, s_dot)``.

Both methods accept optional ``kappa_half`` and ``kappa_next`` for integrators
(like RK4) that evaluate the dynamics at intermediate spatial positions:
  - kappa_half : curvature at s_i + ds/2   (midpoint of the interval)
  - kappa_next : curvature at s_{i+1}      (right endpoint of the interval)
When omitted they default to ``kappa`` (left-endpoint zero-order hold).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import casadi as ca


def _smoothmax(a: ca.MX, b: ca.MX, eps: float) -> ca.MX:
    """Smooth C1 approximation of max(a, b)."""
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps**2))


class SpaceIntegrator(ABC):

    @abstractmethod
    def step(
        self,
        f_space: callable,
        x: ca.MX,
        u: ca.MX,
        kappa: ca.MX,
        ds: float,
        kappa_half: ca.MX | None = None,
        kappa_next: ca.MX | None = None,
    ) -> ca.MX:
        """
        Advance one spatial step.

        Parameters
        ----------
        f_space : callable(x, u, kappa) -> ca.MX
            Space-domain RHS returning dx/ds.
        x : ca.MX
            Current reduced state.
        u : ca.MX
            Current input.
        kappa : ca.MX
            Curvature at the left endpoint s_i.
        ds : float
            Spatial step size.
        kappa_half : ca.MX, optional
            Curvature at the interval midpoint s_i + ds/2.
        kappa_next : ca.MX, optional
            Curvature at the right endpoint s_{i+1}.

        Returns
        -------
        x_next : ca.MX
            State at next spatial step.
        """
        ...

    @abstractmethod
    def time_step(
        self,
        f_space: callable,
        eval_at_point: callable,
        x: ca.MX,
        u: ca.MX,
        kappa: ca.MX,
        ds: float,
        kappa_half: ca.MX | None = None,
        kappa_next: ca.MX | None = None,
        eps: float = 1e-3,
        smooth_eps: float | None = None,
    ) -> ca.MX:
        """
        Estimate the time increment dt = ∫ ds / s_dot for one spatial interval

        Parameters
        ----------
        f_space : callable(x, u, kappa) -> ca.MX
            Space-domain RHS
        eval_at_point : callable(x, u, kappa) -> (full_state, s_dot)
            Evaluates the full time-domain s_dot = ds/dt at a given state.
        x : ca.MX
            Current reduced state.
        u : ca.MX
            Current input (held constant across the interval).
        kappa : ca.MX
            Curvature at the left endpoint s_i.
        ds : float
            Spatial step size.
        kappa_half : ca.MX, optional
            Curvature at the interval midpoint s_i + ds/2.
        kappa_next : ca.MX, optional
            Curvature at the right endpoint s_{i+1}.
        eps : float
            Lower floor for s_dot to avoid division by zero.
        smooth_eps : float, optional
            Smoothing width for the C1 max approximation. Defaults to ``eps``.

        Returns
        -------
        dt : ca.MX
            Estimated time to traverse the interval [s_i, s_{i+1}].
        """
        ...


class EulerIntegrator(SpaceIntegrator):

    def step(self, f_space, x, u, kappa, ds,
             kappa_half=None, kappa_next=None):
        return x + ds * f_space(x, u, kappa)

    def time_step(self, f_space, eval_at_point, x, u, kappa, ds,
                  kappa_half=None, kappa_next=None, eps=1e-3,
                  smooth_eps=None):
        _, s_dot = eval_at_point(x, u, kappa)
        smooth_eps = eps if smooth_eps is None else smooth_eps
        s_dot_safe = _smoothmax(s_dot, ca.MX(eps), smooth_eps)
        return ds / s_dot_safe


class RK4Integrator(SpaceIntegrator):

    def step(self, f_space, x, u, kappa, ds,
             kappa_half=None, kappa_next=None):
        kh = kappa_half if kappa_half is not None else kappa
        kn = kappa_next if kappa_next is not None else kappa
        k1 = f_space(x,              u, kappa)
        k2 = f_space(x + ds / 2 * k1, u, kh)
        k3 = f_space(x + ds / 2 * k2, u, kh)
        k4 = f_space(x + ds * k3,     u, kn)
        return x + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def time_step(self, f_space, eval_at_point, x, u, kappa, ds,
                  kappa_half=None, kappa_next=None, eps=1e-3,
                  smooth_eps=None):
        kh = kappa_half if kappa_half is not None else kappa
        kn = kappa_next if kappa_next is not None else kappa
        smooth_eps = eps if smooth_eps is None else smooth_eps

        k1 = f_space(x,               u, kappa)
        x2 = x + ds / 2 * k1
        k2 = f_space(x2,              u, kh)
        x3 = x + ds / 2 * k2
        k3 = f_space(x3,              u, kh)
        x4 = x + ds * k3

        _, sd1 = eval_at_point(x,  u, kappa)
        _, sd2 = eval_at_point(x2, u, kh)
        _, sd3 = eval_at_point(x3, u, kh)
        _, sd4 = eval_at_point(x4, u, kn)

        sd1_safe = _smoothmax(sd1, ca.MX(eps), smooth_eps)
        sd2_safe = _smoothmax(sd2, ca.MX(eps), smooth_eps)
        sd3_safe = _smoothmax(sd3, ca.MX(eps), smooth_eps)
        sd4_safe = _smoothmax(sd4, ca.MX(eps), smooth_eps)

        # Simpson-3/8 weighted time integral (RK4-consistent quadrature).
        return (ds / 6) * (
            1 / sd1_safe
            + 2 / sd2_safe
            + 2 / sd3_safe
            + 1 / sd4_safe
        )
