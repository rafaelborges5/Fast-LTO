from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import casadi as ca

from fast_lto.utils.smooth import smoothmax

from .vehicle_base import VehicleModel


class DynamicBicycleModel(VehicleModel):
    """
    Dynamic single-track (bicycle) model in Frenet coordinates.

    States (Frenet):
        [s, d, psi_err, v, v_lat, yaw_rate]

    Inputs:
        [a_long, delta]
    """

    def __init__(self, params: dict | None = None) -> None:
        defaults = self.get_default_params()
        if params is not None:
            defaults.update(params)
        super().__init__(defaults)

    def get_default_params(self) -> dict:
        return {
            "m": 170.0,
            "Iz": 250.0,
            # lf/lr: distances to front / rear axle (same convention as other models).
            "lf": 0.842,
            "lr": 0.689,
            "g": 9.81,
            "Bf": 9.0,
            "Cf": 1.3,
            "Dmf_f": 1.4,
            "Br": 9.0,
            "Cr": 1.3,
            "Dmf_r": 1.4,
            # Friction envelope
            "mu": 1.4,
            "gamma_ellipse": 1.0,
            "use_friction_ellipse": True,
            # Guards
            "v_eps": 0.5,
            "smoothmax_eps": 1e-3,
            "eps_s_dot": 1.0,
            "eps_D_kappa": 0.05,
            "rk4_max_ds_m": 2.5,
            # Aero (unused in dynamics)
            "C_d": 1.55,
            "C_r": 0.12,
            "C_l": 3.4,
            "rho": 1.225,
            "a_front": 1.2,
            # Bounds
            "d_max": 3.0,
            "psi_err_max": 1.2,
            "v_min": 0.4,
            "v_max": 20.0,
            "v_lat_max": 4.0,
            "yaw_rate_max": 3.0,
            "a_long_min": -15.0,
            "a_long_max": 15.0,
            "delta_max": 0.4,
            "corners": [
                ("FL", 1.809, 0.750),
                ("FR", 1.809, -0.750),
                ("RL", -0.842, 0.630),
                ("RR", -0.842, -0.630),
            ],
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
        Fz_f = m * g * (lr / L)
        Fz_r = m * g * (lf / L)
        return ca.MX(Fz_f), ca.MX(Fz_r)

    def _pacejka_lateral_force(
        self, alpha: ca.MX, Fz: ca.MX, B: float, C: float, Dmf: float
    ) -> ca.MX:
        # Magic Formula (peak = Fz * Dmf).
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

        v_safe = smoothmax(v, ca.MX(float(p["v_eps"])), float(p["smoothmax_eps"]))

        D_kappa = 1 - kappa * d
        # Raw D_kappa here; positivity is enforced in get_constraints.
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
        Fy_f = self._pacejka_lateral_force(
            alpha_f, Fz_f, float(p["Bf"]), float(p["Cf"]), float(p["Dmf_f"])
        )
        Fy_r = self._pacejka_lateral_force(
            alpha_r, Fz_r, float(p["Br"]), float(p["Cr"]), float(p["Dmf_r"])
        )

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

        v_safe = smoothmax(v, ca.MX(float(p["v_eps"])), float(p["smoothmax_eps"]))
        D_kappa = 1 - kappa * d
        s_dot = (v * ca.cos(psi_err) - v_lat * ca.sin(psi_err)) / D_kappa

        g_list: List[ca.MX] = []

        g_list.append(ca.MX(float(p["eps_D_kappa"])) - D_kappa)

        g_list.append(ca.MX(float(p["eps_s_dot"])) - s_dot)

        if bool(p.get("use_friction_ellipse", True)):
            lf = float(p["lf"])
            lr = float(p["lr"])
            m = float(p["m"])

            alpha_f = ca.atan((v_lat + lf * yaw_rate) / v_safe) - delta
            alpha_r = ca.atan((v_lat - lr * yaw_rate) / v_safe)
            Fz_f, Fz_r = self._normal_loads()
            Fy_f = self._pacejka_lateral_force(
                alpha_f, Fz_f, float(p["Bf"]), float(p["Cf"]), float(p["Dmf_f"])
            )
            Fy_r = self._pacejka_lateral_force(
                alpha_r, Fz_r, float(p["Br"]), float(p["Cr"]), float(p["Dmf_r"])
            )
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

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def diagnostics(self, x_red: ca.MX, u: ca.MX) -> Dict[str, ca.MX]:
        """Axle slip angles, lateral forces, and lateral accel (same as dynamics).

        Steering is an input, so it is read from ``u``.
        """
        p = self.params
        v = x_red[2]
        v_lat = x_red[3]
        yaw_rate = x_red[4]
        delta = u[1]

        lf = float(p["lf"])
        lr = float(p["lr"])
        m = float(p["m"])

        v_safe = smoothmax(v, ca.MX(float(p["v_eps"])), float(p["smoothmax_eps"]))

        alpha_f = ca.atan((v_lat + lf * yaw_rate) / v_safe) - delta
        alpha_r = ca.atan((v_lat - lr * yaw_rate) / v_safe)

        Fz_f, Fz_r = self._normal_loads()
        Fy_f = self._pacejka_lateral_force(
            alpha_f, Fz_f, float(p["Bf"]), float(p["Cf"]), float(p["Dmf_f"])
        )
        Fy_r = self._pacejka_lateral_force(
            alpha_r, Fz_r, float(p["Br"]), float(p["Cr"]), float(p["Dmf_r"])
        )

        return {
            "alpha_f": alpha_f,
            "alpha_r": alpha_r,
            "Fy_f": Fy_f,
            "Fy_r": Fy_r,
            "Fz_f": Fz_f + 0 * v,  # keep every entry node-shaped
            "Fz_r": Fz_r + 0 * v,
            "a_lat_tires": (Fy_f * ca.cos(delta) + Fy_r) / m,
        }
