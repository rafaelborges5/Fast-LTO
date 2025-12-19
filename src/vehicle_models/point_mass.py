"""
Point-mass / kinematic bicycle-like model using lateral acceleration input.

States (Frenet):
    [s, d, psi_err, v]

Inputs:
    [a_long, a_lat]  # longitudinal accel, lateral accel command
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import casadi as ca

from .vehicle_base import VehicleModel


class PointMassModel(VehicleModel):
    def __init__(self, params: dict | None = None):
        defaults = self.get_default_params()
        if params is not None:
            defaults.update(params)
        super().__init__(defaults)

    def get_default_params(self) -> dict:
        return {
            "L": 1.8,  # wheelbase
            "mu": 1.2,
            "g": 9.81,
            "a_long_min": -6.0,
            "a_long_max": 4.0,
            "a_lat_min": -8.0,
            "a_lat_max": 8.0,
            "v_min": 0.1,
            "v_max": 40.0,
            "v_eps": 0.1,  # guard to avoid / by 0
        }

    def get_state_names(self) -> List[str]:
        return ["s", "d", "psi_err", "v"]

    def get_input_names(self) -> List[str]:
        return ["a_long", "a_lat"]

    def get_dynamics(self, states: Sequence[ca.MX], inputs: Sequence[ca.MX], curvature: ca.MX) -> ca.MX:
        """
        x_dot = f(x, u, kappa)
        """
        s = states[0]
        d = states[1]
        psi_err = states[2]
        v = states[3]
        a_long = inputs[0]
        a_lat = inputs[1]
        kappa = curvature

        denom = (1 - kappa * d)
        # Avoid divide-by-zero in psi_err_dot when v is near zero.
        v_safe = ca.fmax(v, self.params["v_eps"])

        s_dot = v * ca.cos(psi_err) / denom
        d_dot = v * ca.sin(psi_err)
        psi_err_dot = a_lat / v_safe - kappa * s_dot
        v_dot = a_long

        return ca.vertcat(s_dot, d_dot, psi_err_dot, v_dot)

    def get_constraints(self, states: Sequence[ca.MX], inputs: Sequence[ca.MX]) -> List[ca.MX]:
        """
        Return inequalities g(x,u) <= 0:
          a_long - a_long_max <= 0
          a_long_min - a_long <= 0
          a_lat   - a_lat_max  <= 0
          a_lat_min - a_lat    <= 0
          friction circle: (a_long/(mu*g))^2 + (a_lat/(mu*g))^2 - 1 <= 0
          speed bounds as soft/path constraints: v - v_max <=0, v_min - v <=0
        """
        a_long = inputs[0]
        a_lat = inputs[1]
        v = states[3]

        p = self.params
        mu_g = p["mu"] * p["g"]

        g_list = [
            a_long - p["a_long_max"],
            p["a_long_min"] - a_long,
            a_lat - p["a_lat_max"],
            p["a_lat_min"] - a_lat,
            (a_long / mu_g) ** 2 + (a_lat / mu_g) ** 2 - 1.0,
            v - p["v_max"],
            p["v_min"] - v,
        ]
        return g_list

    def state_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        lb = [-ca.inf, -ca.inf, -ca.inf, p["v_min"]]
        ub = [ca.inf, ca.inf, ca.inf, p["v_max"]]
        return lb, ub

    def input_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        lb = [p["a_long_min"], p["a_lat_min"]]
        ub = [p["a_long_max"], p["a_lat_max"]]
        return lb, ub

