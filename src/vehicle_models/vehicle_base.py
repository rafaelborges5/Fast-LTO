"""
Abstract interface for vehicle models used in the optimizer.

Convention
----------
state[0] = s     arc-length progress along the centerline
state[1] = d     lateral deviation from the centerline
state[2:] = ...  model-specific (e.g. psi_err, v, yaw_rate, ...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import casadi as ca


class VehicleModel(ABC):
    def __init__(self, params: dict):
        self.params = params

    @property
    def nx(self) -> int:
        """Number of states in the full state vector (including *s*)."""
        return len(self.get_state_names())

    @property
    def nu(self) -> int:
        """Number of inputs."""
        return len(self.get_input_names())

    @property
    def nx_reduced(self) -> int:
        """Number of states in the space-domain formulation (full minus *s*)."""
        return self.nx - 1

    def reduced_state_names(self) -> List[str]:
        """State names excluding *s* (the space-domain reduced states)."""
        return self.get_state_names()[1:]

    @abstractmethod
    def get_state_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_input_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_dynamics(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> ca.MX:
        """
        Symbolic expression for the TIME derivatives: x_dot = f(x, u, kappa).
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
    ) -> List[ca.MX]:
        """
        Inequality constraints g(x, u) <= 0.
        """
        raise NotImplementedError

    def get_default_params(self) -> dict:
        return {}

    def state_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Bounds on the FULL state vector [s, d, ...].  Return None if unbounded."""
        return None

    def reduced_state_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Bounds on the reduced state vector [d, ...] (everything except *s*)."""
        bounds = self.state_bounds()
        if bounds is None:
            return None
        lb, ub = bounds
        return lb[1:], ub[1:]

    def input_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        return None
