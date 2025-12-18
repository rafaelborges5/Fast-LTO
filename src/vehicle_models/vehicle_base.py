"""
Abstract interface for vehicle models used in the optimizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import casadi as ca


class VehicleModel(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def get_state_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_input_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_dynamics(self, states: Sequence[ca.MX], inputs: Sequence[ca.MX], curvature: ca.MX) -> ca.MX:
        """
        symbolic expression for x_dot = f(x, u, kappa).
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self, states: Sequence[ca.MX], inputs: Sequence[ca.MX]) -> List[ca.MX]:
        """
        inequality constraints g(x,u) <= 0.
        """
        raise NotImplementedError

    def get_default_params(self) -> dict:
        return {}

    def state_bounds(self):
        return None

    def input_bounds(self):
        return None

