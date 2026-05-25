from __future__ import annotations

from typing import List, Sequence, Tuple

import casadi as ca

from .vehicle_base import VehicleModel


def _smoothmax(a: ca.MX, b: ca.MX, eps: float) -> ca.MX:
    """
    Smooth approximation of max(a, b) with C1 continuity.
    """
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps**2))


class DynamicBicycleModel(VehicleModel):
    """
    Dynamic single-track (bicycle) model in Frenet coordinates.

    States (Frenet):
        [s, d, psi_err, v, v_lat, yaw_rate]

    Inputs:
        [a_long, delta]
    """

    def __init__(self, params: dict | None = None):
        defaults = self.get_default_params()
        if params is not None:
            defaults.update(params)
        super().__init__(defaults)

    def get_default_params(self) -> dict:
        return {
            # Vehicle
            # (Using AMZ parameter file values where applicable)
            "m": 180.0,
            "Iz": 190.0,
            "lf": 0.78,
            "lr": 0.74,
            "g": 9.81,
            # Tires (Magic Formula, simplified)
            # Your parameter file provides a single lateral set (used for both axles here).
            "Bf": 10.0,
            "Cf": 1.3,
            "Dmf_f": 1.0,
            "Br": 10.0,
            "Cr": 1.3,
            "Dmf_r": 1.0,
            # Friction / safety envelope
            "mu": 0.9,
            "gamma_ellipse": 1.2,  # start loose; tighten later
            "use_friction_ellipse": True,
            # Guards
            "v_eps": 0.5,
            "smoothmax_eps": 1e-3,
            "eps_s_dot": 1.0,
            "eps_D_kappa": 0.05,
            # Resistance / aero parameters (not yet used in v_dot; keep for v2)
            "C_d": 1.55,
            "C_r": 0.12,
            "C_l": 3.4,
            "rho": 1.225,
            "a_front": 1.2,
            # Bounds (finite for normalization)
            "d_max": 3.0,
            "psi_err_max": 1.2,
            "v_min": 0.4,
            "v_max": 20.0,
            "v_lat_max": 4.0,
            "yaw_rate_max": 3.0,
            "a_long_min": -15.0,
            "a_long_max": 10.0,
            "delta_max": 0.45,
        }

    def get_state_names(self) -> List[str]:
        return ["s", "d", "psi_err", "v", "v_lat", "yaw_rate"]

    def get_input_names(self) -> List[str]:
        return ["a_long", "delta"]

    def _normal_loads(self) -> Tuple[ca.MX, ca.MX]:
        p = self.params
        m = float(p["m"])
        g = float(p["g"])
        lf = float(p["lf"])
        lr = float(p["lr"])
        L = lf + lr
        # Static distribution
        Fz_f = m * g * (lr / L)
        Fz_r = m * g * (lf / L)
        return ca.MX(Fz_f), ca.MX(Fz_r)

    def _pacejka_lateral_force(self, alpha: ca.MX, Fz: ca.MX, B: float, C: float, Dmf: float) -> ca.MX:
        # Simplified Magic Formula; peak force = Fz * Dmf
        return -Fz * Dmf * ca.sin(C * ca.atan(B * alpha))

    def get_dynamics(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> ca.MX:
        p = self.params

        d = states[1]
        psi_err = states[2]
        v = states[3]
        v_lat = states[4]
        yaw_rate = states[5]

        a_long = inputs[0]
        delta = inputs[1]
        kappa = curvature

        # Smooth guard on v to avoid division by zero in slip angles.
        v_safe = _smoothmax(v, ca.MX(float(p["v_eps"])), float(p["smoothmax_eps"]))

        D_kappa = 1 - kappa * d
        # In dynamics we keep the raw D_kappa; constraints enforce positivity.
        s_dot = (v * ca.cos(psi_err) - v_lat * ca.sin(psi_err)) / D_kappa
        d_dot = v * ca.sin(psi_err) + v_lat * ca.cos(psi_err)
        psi_err_dot = yaw_rate - kappa * s_dot

        lf = float(p["lf"])
        lr = float(p["lr"])
        m = float(p["m"])
        Iz = float(p["Iz"])

        alpha_f = ca.atan((v_lat + lf * yaw_rate) / v_safe) - delta
        alpha_r = ca.atan((v_lat - lr * yaw_rate) / v_safe)

        Fz_f, Fz_r = self._normal_loads()
        Fy_f = self._pacejka_lateral_force(alpha_f, Fz_f, float(p["Bf"]), float(p["Cf"]), float(p["Dmf_f"]))
        Fy_r = self._pacejka_lateral_force(alpha_r, Fz_r, float(p["Br"]), float(p["Cr"]), float(p["Dmf_r"]))

        v_dot = a_long + yaw_rate * v_lat
        v_lat_dot = (Fy_f * ca.cos(delta) + Fy_r) / m - yaw_rate * v
        yaw_rate_dot = (lf * Fy_f * ca.cos(delta) - lr * Fy_r) / Iz

        return ca.vertcat(s_dot, d_dot, psi_err_dot, v_dot, v_lat_dot, yaw_rate_dot)

    def get_constraints(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> List[ca.MX]:
        p = self.params

        d = states[1]
        psi_err = states[2]
        v = states[3]
        v_lat = states[4]
        yaw_rate = states[5]

        a_long = inputs[0]
        delta = inputs[1]
        kappa = curvature

        # Recompute key kinematics for guard constraints.
        v_safe = _smoothmax(v, ca.MX(float(p["v_eps"])), float(p["smoothmax_eps"]))
        D_kappa = 1 - kappa * d
        s_dot = (v * ca.cos(psi_err) - v_lat * ca.sin(psi_err)) / D_kappa

        g_list: List[ca.MX] = []

        # Enforce D_kappa >= eps_D_kappa
        g_list.append(ca.MX(float(p["eps_D_kappa"])) - D_kappa)

        # Enforce s_dot >= eps_s_dot
        g_list.append(ca.MX(float(p["eps_s_dot"])) - s_dot)

        if bool(p.get("use_friction_ellipse", True)):
            lf = float(p["lf"])
            lr = float(p["lr"])
            m = float(p["m"])

            alpha_f = ca.atan((v_lat + lf * yaw_rate) / v_safe) - delta
            alpha_r = ca.atan((v_lat - lr * yaw_rate) / v_safe)
            Fz_f, Fz_r = self._normal_loads()
            Fy_f = self._pacejka_lateral_force(alpha_f, Fz_f, float(p["Bf"]), float(p["Cf"]), float(p["Dmf_f"]))
            Fy_r = self._pacejka_lateral_force(alpha_r, Fz_r, float(p["Br"]), float(p["Cr"]), float(p["Dmf_r"]))
            a_lat = (Fy_f * ca.cos(delta) + Fy_r) / m

            mu_g = float(p["mu"]) * float(p["g"])
            gamma = float(p["gamma_ellipse"])
            g_list.append((a_long / mu_g) ** 2 + (a_lat / mu_g) ** 2 - gamma**2)

        return g_list

    def state_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        lb = [
            -ca.inf,
            -float(p["d_max"]),
            -float(p["psi_err_max"]),
            float(p["v_min"]),
            -float(p["v_lat_max"]),
            -float(p["yaw_rate_max"]),
        ]
        ub = [
            ca.inf,
            float(p["d_max"]),
            float(p["psi_err_max"]),
            float(p["v_max"]),
            float(p["v_lat_max"]),
            float(p["yaw_rate_max"]),
        ]
        return lb, ub

    def input_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        lb = [float(p["a_long_min"]), -float(p["delta_max"])]
        ub = [float(p["a_long_max"]), float(p["delta_max"])]
        return lb, ub

