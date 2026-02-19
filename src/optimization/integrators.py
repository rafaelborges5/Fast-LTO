"""
Space-domain integrators for the OCP.

All integrators advance the reduced state (excluding *s*) by one spatial step:

    x_{i+1} = Step(f_space, x_i, u_i, kappa_i, ds)

where ``f_space(x, u, kappa)`` returns dx/ds (space derivatives of the reduced
state).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import casadi as ca


class SpaceIntegrator(ABC):

    @abstractmethod
    def step(
        self,
        f_space: callable,
        x: ca.MX,
        u: ca.MX,
        kappa: ca.MX,
        ds: float,
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
            Curvature at current point.
        ds : float
            Spatial step size.

        Returns
        -------
        x_next : ca.MX
            State at next spatial step.
        """
        ...


class EulerIntegrator(SpaceIntegrator):

    def step(self, f_space, x, u, kappa, ds):
        return x + ds * f_space(x, u, kappa)


class RK4Integrator(SpaceIntegrator):

    def step(self, f_space, x, u, kappa, ds):
        k1 = f_space(x, u, kappa)
        k2 = f_space(x + ds / 2 * k1, u, kappa)
        k3 = f_space(x + ds / 2 * k2, u, kappa)
        k4 = f_space(x + ds * k3, u, kappa)
        return x + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
