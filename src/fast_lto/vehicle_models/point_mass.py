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
    def __init__(self, params: dict | None = None) -> None:
        defaults = self.get_default_params()
        if params is not None:
            defaults.update(params)
        super().__init__(defaults)

    def get_default_params(self) -> dict:
        return {
            "m": 170.0,
            # Unused here, but declared so all three describe the same car.
            "lf": 0.842,
            "lr": 0.689,
            "mu": 1.4,
            "g": 9.81,
            "a_long_min": -15.0,
            "a_long_max": 15.0,
            "a_lat_min": -9.0,
            "a_lat_max": 9.0,
            "v_min": 0.1,
            "v_max": 40.0,
            "d_max": 2.0,
            "psi_err_max": 0.8,
            "v_eps": 0.1,
            "smoothmax_eps": 1e-3,
            "eps_s_dot": 1e-3,
            "eps_D_kappa": 0.05,
            "corners": [
                ("FL", 1.809, 0.750),
                ("FR", 1.809, -0.750),
                ("RL", -0.842, 0.630),
                ("RR", -0.842, -0.630),
            ],
        }

    def get_state_names(self) -> List[str]:
        return ["s", "d", "psi_err", "v"]

    def get_input_names(self) -> List[str]:
        return ["a_long", "a_lat"]

    def get_dynamics(
        self, states: Sequence[ca.MX], inputs: Sequence[ca.MX], curvature: ca.MX
    ) -> ca.MX:
        """
        x_dot = f(x, u, kappa)
        """
        _s = states[0]
        d = states[1]
        psi_err = states[2]
        v = states[3]
        a_long = inputs[0]
        a_lat = inputs[1]
        kappa = curvature

        denom = 1 - kappa * d
        # Avoid divide-by-zero in psi_err_dot when v is near zero.
        v_safe = ca.fmax(v, self.params["v_eps"])

        s_dot = v * ca.cos(psi_err) / denom  # ds/dt
        d_dot = v * ca.sin(psi_err)  # dd/dt
        psi_err_dot = a_lat / v_safe - kappa * s_dot  # d(psi_err)/dt
        v_dot = a_long  # dv/dt

        return ca.vertcat(s_dot, d_dot, psi_err_dot, v_dot)

    def get_constraints(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> List[ca.MX]:
        """The friction circle, ``(a_long/mu g)^2 + (a_lat/mu g)^2 <= 1``.

        The acceleration and speed limits are box bounds, not constraints, so
        they are handled by ``input_bounds`` and ``state_bounds``.
        """
        a_long = inputs[0]
        a_lat = inputs[1]

        p = self.params
        mu_g = p["mu"] * p["g"]

        g_list = [
            (a_long / mu_g) ** 2 + (a_lat / mu_g) ** 2 - 1.0,
        ]
        return g_list

    def state_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        lb = [-ca.inf, -float(p["d_max"]), -float(p["psi_err_max"]), p["v_min"]]
        ub = [ca.inf, float(p["d_max"]), float(p["psi_err_max"]), p["v_max"]]
        return lb, ub

    def input_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        lb = [p["a_long_min"], p["a_lat_min"]]
        ub = [p["a_long_max"], p["a_lat_max"]]
        return lb, ub
