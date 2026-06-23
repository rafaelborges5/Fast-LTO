from __future__ import annotations

from typing import List, Sequence, Tuple

import casadi as ca

from .vehicle_base import VehicleModel


def _smoothmax(a: ca.MX, b: ca.MX, eps: float) -> ca.MX:
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps**2))


class FourWheelModel(VehicleModel):
    """
    Four-wheel vehicle model in Frenet coordinates.

    States:
        [s, d, psi_err, v_long, v_lat, yaw_rate, Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta]

    Inputs (normalized rates):
        [Fx_fl_dot_norm, Fx_fr_dot_norm, Fx_rr_dot_norm, Fx_rl_dot_norm, delta_dot_norm]

    Internal wheel ordering: FL=1, FR=2, RR=3, RL=4 (cyclic, matching branch 141).
    """

    def __init__(self, params: dict | None = None):
        defaults = self.get_default_params()
        if params is not None:
            defaults.update(params)
        super().__init__(defaults)

    def get_default_params(self) -> dict:
        return {
            "m": 170.0,
            "Iz": 250.0,
            "g": 9.81,
            "lf": 0.689,
            "lr": 0.842,
            "a_l": 0.630,
            "a_r": 0.630,
            "h": 0.246,
            # Aero
            "rho": 1.225,
            "C_l": 5.54,
            "C_d": 1.58,
            "C_r": 0.15,
            "A_f": 1.2,
            # Per-wheel Pacejka tire params
            "B_fl": 9.0, "C_fl": 1.3, "D_fl": 1.4,
            "B_fr": 9.0, "C_fr": 1.3, "D_fr": 1.4,
            "B_rr": 9.0, "C_rr": 1.3, "D_rr": 1.4,
            "B_rl": 9.0, "C_rl": 1.3, "D_rl": 1.4,
            # Actuator rate limits (physical)
            "dFxmax": 1000.0,
            "ddeltamax": 1.3,
            # State bounds
            "Fx_max": 1500.0,
            "delta_max": 0.4,
            # Guards
            "v_eps": 0.5,
            "slip_vx_eps": 0.2,
            "eps_D_kappa": 0.05,
            "eps_s_dot": 1.0,
            "eps_friction_den": 5.0,
            "smoothmax_eps": 1e-3,
            "rk4_max_ds_m": 2.5,
            # CG-level acceleration limits (None = disabled)
            "a_long_max": None,
            "a_lat_max": None,
            # Bounds
            "d_max": 3.0,
            "psi_err_max": 1.2,
            "v_min": 0.4,
            "v_max": 20.0,
            "v_lat_max": 4.0,
            "yaw_rate_max": 3.0,
            # Load transfer mode
            "load_transfer_mode": "current_quasistatic",
            # Body corners for lateral constraints (total vehicle envelope)
            "corners": [
                ("FL", 1.809, 0.750),
                ("FR", 1.809, -0.750),
                ("RL", -0.842, 0.630),
                ("RR", -0.842, -0.630),
            ],
        }

    def get_state_names(self) -> List[str]:
        return [
            "s", "d", "psi_err", "v_long", "v_lat", "yaw_rate",
            "Fx_fl", "Fx_fr", "Fx_rr", "Fx_rl", "delta",
        ]

    def get_input_names(self) -> List[str]:
        return [
            "Fx_fl_dot_norm", "Fx_fr_dot_norm",
            "Fx_rr_dot_norm", "Fx_rl_dot_norm",
            "delta_dot_norm",
        ]

    # ------------------------------------------------------------------ #
    #  Aerodynamics
    # ------------------------------------------------------------------ #

    def _aero_forces(self, v_long: ca.MX):
        p = self.params
        q = 0.5 * float(p["rho"]) * float(p["A_f"]) * v_long**2
        F_down = float(p["C_l"]) * q
        F_drag = float(p["C_d"]) * q
        F_roll = float(p["m"]) * float(p["g"]) * float(p["C_r"])
        return F_down, F_drag, F_roll

    # ------------------------------------------------------------------ #
    #  Slip angles
    # ------------------------------------------------------------------ #

    def _slip_angles(self, v_long, v_lat, yaw_rate, delta):
        p = self.params
        l_f = float(p["lf"])
        l_r = float(p["lr"])
        a_l = float(p["a_l"])
        a_r = float(p["a_r"])
        eps = float(p["slip_vx_eps"])

        vx_fl = v_long - a_l * yaw_rate
        vx_fr = v_long + a_r * yaw_rate
        vx_rr = v_long + a_r * yaw_rate
        vx_rl = v_long - a_l * yaw_rate

        def _guard(vx):
            return ca.sign(vx) * ca.sqrt(vx**2 + eps)

        vy_front = v_lat + l_f * yaw_rate
        vy_rear = v_lat - l_r * yaw_rate

        alpha_fl = ca.atan2(vy_front, _guard(vx_fl)) - delta
        alpha_fr = ca.atan2(vy_front, _guard(vx_fr)) - delta
        alpha_rr = ca.atan2(vy_rear, _guard(vx_rr))
        alpha_rl = ca.atan2(vy_rear, _guard(vx_rl))

        return alpha_fl, alpha_fr, alpha_rr, alpha_rl

    # ------------------------------------------------------------------ #
    #  Pacejka lateral coefficient  f_i = D*sin(C*atan(B*alpha))
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pacejka_coeff(alpha, B, C, D):
        return D * ca.sin(C * ca.atan(B * alpha))

    def _all_pacejka_coeffs(self, alpha_fl, alpha_fr, alpha_rr, alpha_rl):
        p = self.params
        f_fl = self._pacejka_coeff(alpha_fl, float(p["B_fl"]), float(p["C_fl"]), float(p["D_fl"]))
        f_fr = self._pacejka_coeff(alpha_fr, float(p["B_fr"]), float(p["C_fr"]), float(p["D_fr"]))
        f_rr = self._pacejka_coeff(alpha_rr, float(p["B_rr"]), float(p["C_rr"]), float(p["D_rr"]))
        f_rl = self._pacejka_coeff(alpha_rl, float(p["B_rl"]), float(p["C_rl"]), float(p["D_rl"]))
        return f_fl, f_fr, f_rr, f_rl

    # ------------------------------------------------------------------ #
    #  Vertical loads
    # ------------------------------------------------------------------ #

    def _static_loads(self):
        p = self.params
        m, g = float(p["m"]), float(p["g"])
        l_f, l_r = float(p["lf"]), float(p["lr"])
        a_l, a_r = float(p["a_l"]), float(p["a_r"])
        L = l_f + l_r
        W = a_l + a_r
        W_total = m * g
        Fw_fl = W_total * (l_r / L) * (a_r / W)
        Fw_fr = W_total * (l_r / L) * (a_l / W)
        Fw_rr = W_total * (l_f / L) * (a_l / W)
        Fw_rl = W_total * (l_f / L) * (a_r / W)
        return Fw_fl, Fw_fr, Fw_rr, Fw_rl

    def _compute_vertical_loads(
        self, v_long, v_lat, yaw_rate,
        Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
    ):
        mode = self.params.get("load_transfer_mode", "static")

        Fw_fl, Fw_fr, Fw_rr, Fw_rl = self._static_loads()

        if mode == "static":
            return ca.MX(Fw_fl), ca.MX(Fw_fr), ca.MX(Fw_rr), ca.MX(Fw_rl)

        F_down, _, _ = self._aero_forces(v_long)
        Fz_fl = Fw_fl + F_down / 4
        Fz_fr = Fw_fr + F_down / 4
        Fz_rr = Fw_rr + F_down / 4
        Fz_rl = Fw_rl + F_down / 4

        if mode == "static_aero":
            return Fz_fl, Fz_fr, Fz_rr, Fz_rl

        # ---- current_quasistatic: solve A * Fz = b ----
        p = self.params
        h = float(p["h"])
        l_f = float(p["lf"])
        l_r = float(p["lr"])
        a_l = float(p["a_l"])
        a_r = float(p["a_r"])
        L = l_f + l_r
        W = a_l + a_r
        h1 = 0.5 * h / L
        h2 = 0.5 * h / W

        alpha_fl, alpha_fr, alpha_rr, alpha_rl = self._slip_angles(
            v_long, v_lat, yaw_rate, delta,
        )
        f_fl, f_fr, f_rr, f_rl = self._all_pacejka_coeffs(
            alpha_fl, alpha_fr, alpha_rr, alpha_rl,
        )

        _, F_drag, F_roll = self._aero_forces(v_long)

        cd = ca.cos(delta)
        sd = ca.sin(delta)

        Fx_front = Fx_fl + Fx_fr
        P_known = Fx_front * cd + Fx_rr + Fx_rl - F_roll - F_drag
        R_known = Fx_front * sd

        b1 = Fw_fl + F_down / 4 - h1 * P_known - h2 * R_known
        b2 = Fw_fr + F_down / 4 - h1 * P_known + h2 * R_known
        b3 = Fw_rr + F_down / 4 + h1 * P_known
        b4 = Fw_rl + F_down / 4 + h1 * P_known

        c_fl = h1 * sd * f_fl
        c_fr = h1 * sd * f_fr

        A = ca.MX(4, 4)
        # Row 1 (FL)
        A[0, 0] = 1 + f_fl * (h1 * sd - h2 * cd)
        A[0, 1] = f_fr * (h1 * sd - h2 * cd)
        # Row 2 (FR)
        A[1, 0] = f_fl * (h1 * sd + h2 * cd)
        A[1, 1] = 1 + f_fr * (h1 * sd + h2 * cd)
        # Row 3 (RR)
        A[2, 0] = -c_fl
        A[2, 1] = -c_fr
        A[2, 2] = 1 + h2 * f_rr
        A[2, 3] = h2 * f_rl
        # Row 4 (RL)
        A[3, 0] = -c_fl
        A[3, 1] = -c_fr
        A[3, 2] = -h2 * f_rr
        A[3, 3] = 1 - h2 * f_rl

        b_vec = ca.vertcat(b1, b2, b3, b4)
        Fz_vec = ca.solve(A, b_vec)

        eps_fz = float(p.get("smoothmax_eps", 1e-3))
        Fz_min = ca.MX(10.0)
        Fz_fl_s = _smoothmax(Fz_vec[0], Fz_min, eps_fz)
        Fz_fr_s = _smoothmax(Fz_vec[1], Fz_min, eps_fz)
        Fz_rr_s = _smoothmax(Fz_vec[2], Fz_min, eps_fz)
        Fz_rl_s = _smoothmax(Fz_vec[3], Fz_min, eps_fz)

        return Fz_fl_s, Fz_fr_s, Fz_rr_s, Fz_rl_s

    # ------------------------------------------------------------------ #
    #  Body forces & yaw moment
    # ------------------------------------------------------------------ #

    def _body_forces_and_moment(
        self,
        Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
        Fy_fl, Fy_fr, Fy_rr, Fy_rl,
        F_drag, F_roll,
    ):
        p = self.params
        l_f = float(p["lf"])
        l_r = float(p["lr"])
        a_l = float(p["a_l"])
        a_r = float(p["a_r"])

        cd = ca.cos(delta)
        sd = ca.sin(delta)

        Fx_total = (
            (Fx_fl + Fx_fr) * cd
            - (Fy_fl + Fy_fr) * sd
            + Fx_rr + Fx_rl
            - F_roll - F_drag
        )

        Fy_total = (
            (Fx_fl + Fx_fr) * sd
            + (Fy_fl + Fy_fr) * cd
            + Fy_rr + Fy_rl
        )

        Mz = (
            Fx_fl * (-a_l * cd + l_f * sd)
            + Fy_fl * (l_f * cd + a_l * sd)
            + Fx_fr * (a_r * cd + l_f * sd)
            + Fy_fr * (l_f * cd - a_r * sd)
            + Fx_rr * a_r - Fy_rr * l_r
            - Fx_rl * a_l - Fy_rl * l_r
        )

        return Fx_total, Fy_total, Mz

    # ------------------------------------------------------------------ #
    #  Dynamics
    # ------------------------------------------------------------------ #

    def get_dynamics(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> ca.MX:
        p = self.params

        d = states[1]
        psi_err = states[2]
        v_long = states[3]
        v_lat = states[4]
        yaw_rate = states[5]
        Fx_fl = states[6]
        Fx_fr = states[7]
        Fx_rr = states[8]
        Fx_rl = states[9]
        delta = states[10]

        Fx_fl_dot_n = inputs[0]
        Fx_fr_dot_n = inputs[1]
        Fx_rr_dot_n = inputs[2]
        Fx_rl_dot_n = inputs[3]
        delta_dot_n = inputs[4]

        kappa = curvature
        m = float(p["m"])
        Iz = float(p["Iz"])

        # Kinematics
        D_kappa = 1 - kappa * d
        s_dot = (v_long * ca.cos(psi_err) - v_lat * ca.sin(psi_err)) / D_kappa
        d_dot = v_long * ca.sin(psi_err) + v_lat * ca.cos(psi_err)
        psi_err_dot = yaw_rate - kappa * s_dot

        # Vertical loads
        Fz_fl, Fz_fr, Fz_rr, Fz_rl = self._compute_vertical_loads(
            v_long, v_lat, yaw_rate,
            Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
        )

        # Slip angles & lateral forces
        alpha_fl, alpha_fr, alpha_rr, alpha_rl = self._slip_angles(
            v_long, v_lat, yaw_rate, delta,
        )
        f_fl, f_fr, f_rr, f_rl = self._all_pacejka_coeffs(
            alpha_fl, alpha_fr, alpha_rr, alpha_rl,
        )
        Fy_fl = -Fz_fl * f_fl
        Fy_fr = -Fz_fr * f_fr
        Fy_rr = -Fz_rr * f_rr
        Fy_rl = -Fz_rl * f_rl

        # Aero resistance
        _, F_drag, F_roll = self._aero_forces(v_long)

        # Body forces and moment
        Fx_total, Fy_total, Mz = self._body_forces_and_moment(
            Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
            Fy_fl, Fy_fr, Fy_rr, Fy_rl,
            F_drag, F_roll,
        )

        # Vehicle dynamics
        v_long_dot = Fx_total / m + yaw_rate * v_lat
        v_lat_dot = Fy_total / m - yaw_rate * v_long
        yaw_rate_dot = Mz / Iz

        # Actuator dynamics
        dFxmax = float(p["dFxmax"])
        ddeltamax = float(p["ddeltamax"])

        Fx_fl_dot = Fx_fl_dot_n * dFxmax
        Fx_fr_dot = Fx_fr_dot_n * dFxmax
        Fx_rr_dot = Fx_rr_dot_n * dFxmax
        Fx_rl_dot = Fx_rl_dot_n * dFxmax
        delta_dot = delta_dot_n * ddeltamax

        return ca.vertcat(
            s_dot, d_dot, psi_err_dot,
            v_long_dot, v_lat_dot, yaw_rate_dot,
            Fx_fl_dot, Fx_fr_dot, Fx_rr_dot, Fx_rl_dot,
            delta_dot,
        )

    # ------------------------------------------------------------------ #
    #  Constraints
    # ------------------------------------------------------------------ #

    def get_constraints(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> List[ca.MX]:
        p = self.params

        d = states[1]
        psi_err = states[2]
        v_long = states[3]
        v_lat = states[4]
        yaw_rate = states[5]
        Fx_fl = states[6]
        Fx_fr = states[7]
        Fx_rr = states[8]
        Fx_rl = states[9]
        delta = states[10]

        kappa = curvature

        D_kappa = 1 - kappa * d
        s_dot = (v_long * ca.cos(psi_err) - v_lat * ca.sin(psi_err)) / D_kappa

        g_list: List[ca.MX] = []

        g_list.append(ca.MX(float(p["eps_D_kappa"])) - D_kappa)
        g_list.append(ca.MX(float(p["eps_s_dot"])) - s_dot)

        # Per-wheel friction circles
        Fz_fl, Fz_fr, Fz_rr, Fz_rl = self._compute_vertical_loads(
            v_long, v_lat, yaw_rate,
            Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
        )

        alpha_fl, alpha_fr, alpha_rr, alpha_rl = self._slip_angles(
            v_long, v_lat, yaw_rate, delta,
        )
        f_fl, f_fr, f_rr, f_rl = self._all_pacejka_coeffs(
            alpha_fl, alpha_fr, alpha_rr, alpha_rl,
        )
        Fy_fl = -Fz_fl * f_fl
        Fy_fr = -Fz_fr * f_fr
        Fy_rr = -Fz_rr * f_rr
        Fy_rl = -Fz_rl * f_rl

        eps_den = float(p["eps_friction_den"])

        for Fx_i, Fy_i, Fz_i, D_i in [
            (Fx_fl, Fy_fl, Fz_fl, float(p["D_fl"])),
            (Fx_fr, Fy_fr, Fz_fr, float(p["D_fr"])),
            (Fx_rr, Fy_rr, Fz_rr, float(p["D_rr"])),
            (Fx_rl, Fy_rl, Fz_rl, float(p["D_rl"])),
        ]:
            cap_sq = (D_i * Fz_i) ** 2 + eps_den
            g_list.append(Fx_i**2 / cap_sq + Fy_i**2 / cap_sq - 1)

        # Optional CG-level acceleration constraint
        a_long_max = p.get("a_long_max")
        a_lat_max = p.get("a_lat_max")
        if a_long_max is not None or a_lat_max is not None:
            _, F_drag, F_roll = self._aero_forces(v_long)
            Fx_total, Fy_total, _ = self._body_forces_and_moment(
                Fx_fl, Fx_fr, Fx_rr, Fx_rl, delta,
                Fy_fl, Fy_fr, Fy_rr, Fy_rl,
                F_drag, F_roll,
            )
            m = float(p["m"])
            if a_long_max is not None and a_lat_max is not None:
                g_list.append(
                    (Fx_total / (m * float(a_long_max))) ** 2
                    + (Fy_total / (m * float(a_lat_max))) ** 2
                    - 1
                )
            elif a_long_max is not None:
                g_list.append(
                    (Fx_total / (m * float(a_long_max))) ** 2 - 1
                )
            else:
                g_list.append(
                    (Fy_total / (m * float(a_lat_max))) ** 2 - 1
                )

        return g_list

    # ------------------------------------------------------------------ #
    #  Bounds
    # ------------------------------------------------------------------ #

    def state_bounds(self) -> Tuple[List[float], List[float]]:
        p = self.params
        Fx_max = float(p["Fx_max"])
        delta_max = float(p["delta_max"])
        lb = [
            -ca.inf,
            -float(p["d_max"]),
            -float(p["psi_err_max"]),
            float(p["v_min"]),
            -float(p["v_lat_max"]),
            -float(p["yaw_rate_max"]),
            -Fx_max, -Fx_max, -Fx_max, -Fx_max,
            -delta_max,
        ]
        ub = [
            ca.inf,
            float(p["d_max"]),
            float(p["psi_err_max"]),
            float(p["v_max"]),
            float(p["v_lat_max"]),
            float(p["yaw_rate_max"]),
            Fx_max, Fx_max, Fx_max, Fx_max,
            delta_max,
        ]
        return lb, ub

    def input_bounds(self) -> Tuple[List[float], List[float]]:
        lb = [-1.0, -1.0, -1.0, -1.0, -1.0]
        ub = [1.0, 1.0, 1.0, 1.0, 1.0]
        return lb, ub
