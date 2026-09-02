"""
Visualization helpers for four-wheel OCP solutions.

Main panels (3x2):
  1. Track + speed   2. Speed profile
  3. Lateral offset  4. Per-wheel Fx
  5. GG diagram      6. Profiling

Diagnostics page (4x2):
  1. Steering angle & rate  2. Per-wheel slip angles
  3. Yaw moments (TV+total) 4. Wheel force rates
  5. Friction utilization   6. Vertical loads
  7. Lateral forces         8. Lateral velocity & yaw rate
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from fast_lto.visualization.ocp_plots import (
    plot_offsets,
    plot_path_with_speed,
    plot_speed_profile,
)

WHEEL_COLORS = {
    "FL": "tab:blue",
    "FR": "tab:orange",
    "RR": "tab:green",
    "RL": "tab:red",
}


def _compute_four_wheel_forces(
    *,
    v: np.ndarray,
    v_lat: np.ndarray,
    yaw_rate: np.ndarray,
    Fx_fl: np.ndarray,
    Fx_fr: np.ndarray,
    Fx_rr: np.ndarray,
    Fx_rl: np.ndarray,
    delta: np.ndarray,
    params: Dict,
):
    l_f = float(params.get("lf", 0.689))
    l_r = float(params.get("lr", 0.842))
    a_l = float(params.get("a_l", 0.62))
    a_r = float(params.get("a_r", 0.62))
    m = float(params.get("m", 160.0))
    g = float(params.get("g", 9.81))
    L = l_f + l_r
    W = a_l + a_r
    rho = float(params.get("rho", 1.225))
    C_l = float(params.get("C_l", 5.54))
    C_d = float(params.get("C_d", 1.58))
    C_r = float(params.get("C_r", 0.15))
    A_f = float(params.get("A_f", 1.2))

    slip_eps = float(params.get("slip_vx_eps", 0.2))
    v_safe = np.maximum(np.abs(v), 0.5)

    # Static loads
    Fw_fl = m * g * (l_r / L) * (a_r / W)
    Fw_fr = m * g * (l_r / L) * (a_l / W)
    Fw_rr = m * g * (l_f / L) * (a_l / W)
    Fw_rl = m * g * (l_f / L) * (a_r / W)

    F_down = 0.5 * rho * C_l * A_f * v**2
    F_drag = 0.5 * rho * C_d * A_f * v**2
    F_roll = m * g * C_r

    mode = params.get("load_transfer_mode", "static")

    if mode == "static":
        Fz_fl = np.full_like(v, Fw_fl)
        Fz_fr = np.full_like(v, Fw_fr)
        Fz_rr = np.full_like(v, Fw_rr)
        Fz_rl = np.full_like(v, Fw_rl)
    elif mode == "static_aero":
        Fz_fl = Fw_fl + F_down / 4
        Fz_fr = Fw_fr + F_down / 4
        Fz_rr = Fw_rr + F_down / 4
        Fz_rl = Fw_rl + F_down / 4
    else:
        h = float(params.get("h", 0.246))
        h1 = 0.5 * h / L
        h2 = 0.5 * h / W

        # Slip angles (no low-speed guard for visualization)
        vx_fl = v - a_l * yaw_rate
        vx_fr = v + a_r * yaw_rate
        vx_rr = v + a_r * yaw_rate
        vx_rl = v - a_l * yaw_rate

        vy_front = v_lat + l_f * yaw_rate
        vy_rear = v_lat - l_r * yaw_rate

        alpha_fl = (
            np.arctan2(
                vy_front, np.maximum(np.abs(vx_fl), 0.3) * np.sign(np.where(vx_fl == 0, 1.0, vx_fl))
            )
            - delta
        )
        alpha_fr = (
            np.arctan2(
                vy_front, np.maximum(np.abs(vx_fr), 0.3) * np.sign(np.where(vx_fr == 0, 1.0, vx_fr))
            )
            - delta
        )
        alpha_rr = np.arctan2(
            vy_rear, np.maximum(np.abs(vx_rr), 0.3) * np.sign(np.where(vx_rr == 0, 1.0, vx_rr))
        )
        alpha_rl = np.arctan2(
            vy_rear, np.maximum(np.abs(vx_rl), 0.3) * np.sign(np.where(vx_rl == 0, 1.0, vx_rl))
        )

        def _pacejka(alpha, B, C, D):
            return D * np.sin(C * np.arctan(B * alpha))

        f_fl = _pacejka(
            alpha_fl,
            float(params.get("B_fl", 9.0)),
            float(params.get("C_fl", 1.3)),
            float(params.get("D_fl", 1.2)),
        )
        f_fr = _pacejka(
            alpha_fr,
            float(params.get("B_fr", 9.0)),
            float(params.get("C_fr", 1.3)),
            float(params.get("D_fr", 1.2)),
        )
        f_rr = _pacejka(
            alpha_rr,
            float(params.get("B_rr", 9.0)),
            float(params.get("C_rr", 1.3)),
            float(params.get("D_rr", 1.2)),
        )
        f_rl = _pacejka(
            alpha_rl,
            float(params.get("B_rl", 9.0)),
            float(params.get("C_rl", 1.3)),
            float(params.get("D_rl", 1.2)),
        )

        cd = np.cos(delta)
        sd = np.sin(delta)
        Fx_front = Fx_fl + Fx_fr
        P_known = Fx_front * cd + Fx_rr + Fx_rl - F_roll - F_drag
        R_known = Fx_front * sd

        b1 = Fw_fl + F_down / 4 - h1 * P_known - h2 * R_known
        b2 = Fw_fr + F_down / 4 - h1 * P_known + h2 * R_known
        b3 = Fw_rr + F_down / 4 + h1 * P_known
        b4 = Fw_rl + F_down / 4 + h1 * P_known

        N = len(v)
        Fz_fl = np.zeros(N)
        Fz_fr = np.zeros(N)
        Fz_rr = np.zeros(N)
        Fz_rl = np.zeros(N)

        for i in range(N):
            A = np.array(
                [
                    [
                        1 + f_fl[i] * (h1 * sd[i] - h2 * cd[i]),
                        f_fr[i] * (h1 * sd[i] - h2 * cd[i]),
                        0,
                        0,
                    ],
                    [
                        f_fl[i] * (h1 * sd[i] + h2 * cd[i]),
                        1 + f_fr[i] * (h1 * sd[i] + h2 * cd[i]),
                        0,
                        0,
                    ],
                    [-h1 * sd[i] * f_fl[i], -h1 * sd[i] * f_fr[i], 1 + h2 * f_rr[i], h2 * f_rl[i]],
                    [-h1 * sd[i] * f_fl[i], -h1 * sd[i] * f_fr[i], -h2 * f_rr[i], 1 - h2 * f_rl[i]],
                ]
            )
            b_vec = np.array([b1[i], b2[i], b3[i], b4[i]])
            try:
                Fz_sol = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                Fz_sol = np.array([Fw_fl, Fw_fr, Fw_rr, Fw_rl])
            Fz_fl[i] = max(Fz_sol[0], 10.0)
            Fz_fr[i] = max(Fz_sol[1], 10.0)
            Fz_rr[i] = max(Fz_sol[2], 10.0)
            Fz_rl[i] = max(Fz_sol[3], 10.0)

    # Slip angles (raw, for visualization)
    vx_fl = v - a_l * yaw_rate
    vx_fr = v + a_r * yaw_rate
    vx_rr = v + a_r * yaw_rate
    vx_rl = v - a_l * yaw_rate

    vy_front = v_lat + l_f * yaw_rate
    vy_rear = v_lat - l_r * yaw_rate

    alpha_fl = np.arctan2(vy_front, vx_fl) - delta
    alpha_fr = np.arctan2(vy_front, vx_fr) - delta
    alpha_rr = np.arctan2(vy_rear, vx_rr)
    alpha_rl = np.arctan2(vy_rear, vx_rl)

    def _pacejka_np(alpha, B, C, D):
        return D * np.sin(C * np.arctan(B * alpha))

    f_fl_vis = _pacejka_np(
        alpha_fl,
        float(params.get("B_fl", 9.0)),
        float(params.get("C_fl", 1.3)),
        float(params.get("D_fl", 1.2)),
    )
    f_fr_vis = _pacejka_np(
        alpha_fr,
        float(params.get("B_fr", 9.0)),
        float(params.get("C_fr", 1.3)),
        float(params.get("D_fr", 1.2)),
    )
    f_rr_vis = _pacejka_np(
        alpha_rr,
        float(params.get("B_rr", 9.0)),
        float(params.get("C_rr", 1.3)),
        float(params.get("D_rr", 1.2)),
    )
    f_rl_vis = _pacejka_np(
        alpha_rl,
        float(params.get("B_rl", 9.0)),
        float(params.get("C_rl", 1.3)),
        float(params.get("D_rl", 1.2)),
    )

    Fy_fl = -Fz_fl * f_fl_vis
    Fy_fr = -Fz_fr * f_fr_vis
    Fy_rr = -Fz_rr * f_rr_vis
    Fy_rl = -Fz_rl * f_rl_vis

    # Friction utilization
    D_fl_p = float(params.get("D_fl", 1.2))
    D_fr_p = float(params.get("D_fr", 1.2))
    D_rr_p = float(params.get("D_rr", 1.2))
    D_rl_p = float(params.get("D_rl", 1.2))

    def _fric_util(Fx, Fy, Fz, D_val):
        cap = np.maximum(D_val * Fz, 1.0)
        return np.sqrt(Fx**2 + Fy**2) / cap * 100.0

    util_fl = _fric_util(Fx_fl, Fy_fl, Fz_fl, D_fl_p)
    util_fr = _fric_util(Fx_fr, Fy_fr, Fz_fr, D_fr_p)
    util_rr = _fric_util(Fx_rr, Fy_rr, Fz_rr, D_rr_p)
    util_rl = _fric_util(Fx_rl, Fy_rl, Fz_rl, D_rl_p)

    return {
        "alpha_fl": alpha_fl,
        "alpha_fr": alpha_fr,
        "alpha_rr": alpha_rr,
        "alpha_rl": alpha_rl,
        "Fy_fl": Fy_fl,
        "Fy_fr": Fy_fr,
        "Fy_rr": Fy_rr,
        "Fy_rl": Fy_rl,
        "Fz_fl": Fz_fl,
        "Fz_fr": Fz_fr,
        "Fz_rr": Fz_rr,
        "Fz_rl": Fz_rl,
        "util_fl": util_fl,
        "util_fr": util_fr,
        "util_rr": util_rr,
        "util_rl": util_rl,
        "F_drag": F_drag,
        "F_roll": F_roll,
    }


def _compute_yaw_moments(
    *,
    Fx_fl,
    Fx_fr,
    Fx_rr,
    Fx_rl,
    Fy_fl,
    Fy_fr,
    Fy_rr,
    Fy_rl,
    delta,
    params,
):
    l_f = float(params.get("lf", 0.689))
    l_r = float(params.get("lr", 0.842))
    a_l = float(params.get("a_l", 0.62))
    a_r = float(params.get("a_r", 0.62))

    cd = np.cos(delta)
    sd = np.sin(delta)

    Mz_Fx = (
        Fx_fl * (-a_l * cd + l_f * sd) + Fx_fr * (a_r * cd + l_f * sd) + Fx_rr * a_r - Fx_rl * a_l
    )

    Mz_Fy = (
        Fy_fl * (l_f * cd + a_l * sd) + Fy_fr * (l_f * cd - a_r * sd) - Fy_rr * l_r - Fy_rl * l_r
    )

    return Mz_Fx, Mz_Fx + Mz_Fy


def _compute_body_accels(
    *,
    Fx_fl,
    Fx_fr,
    Fx_rr,
    Fx_rl,
    Fy_fl,
    Fy_fr,
    Fy_rr,
    Fy_rl,
    delta,
    F_drag,
    F_roll,
    params,
):
    m = float(params.get("m", 160.0))
    cd = np.cos(delta)
    sd = np.sin(delta)

    Fx_total = (Fx_fl + Fx_fr) * cd - (Fy_fl + Fy_fr) * sd + Fx_rr + Fx_rl - F_roll - F_drag
    Fy_total = (Fx_fl + Fx_fr) * sd + (Fy_fl + Fy_fr) * cd + Fy_rr + Fy_rl
    return Fx_total / m, Fy_total / m


def _plot_profiling_panel_four_wheel(
    profiling: Optional[Dict],
    constraint_activity: Optional[Dict[str, float]],
    ax: plt.Axes,
) -> None:
    ax.axis("off")
    if profiling is None:
        ax.text(0.0, 0.5, "No profiling data.", transform=ax.transAxes, fontsize=10, va="center")
        return

    lines = []
    lines.append("Solver profiling")
    N = profiling.get("N")
    ds_m = profiling.get("ds_m")
    if N is not None and ds_m is not None:
        lines.append(f"N = {N}, ds = {ds_m:.3f} m")
    solve_time_s = profiling.get("solve_time_s")
    if solve_time_s is not None:
        lines.append(f"Time = {solve_time_s:.3f} s")
    iter_count = profiling.get("iter_count")
    if iter_count is not None:
        lines.append(f"Iterations = {iter_count}")
    return_status = profiling.get("return_status")
    if return_status is not None:
        lines.append(f"Status = {return_status}")
    lap_time_s = profiling.get("lap_time_s")
    reg_term = profiling.get("reg_term")
    if lap_time_s is not None:
        lines.append("")
        lines.append(f"Lap time = {lap_time_s:.3f} s")
    if reg_term is not None:
        rel = profiling.get("reg_term_relative")
        if rel:
            lines.append(f"Reg term = {reg_term:.4f} ({rel:.2%})")
        else:
            lines.append(f"Reg term = {reg_term:.4f}")

    if constraint_activity:
        lines.append("")
        lines.append("Constraint activity:")

        def pct(key):
            val = constraint_activity.get(key, 0.0)
            return float(val * 100.0)

        lines.append(f"Track bounds  ~ {pct('track_bounds_active'):.1f}%")
        lines.append(f"Friction FL   ~ {pct('friction_fl_active'):.1f}%")
        lines.append(f"Friction FR   ~ {pct('friction_fr_active'):.1f}%")
        lines.append(f"Friction RR   ~ {pct('friction_rr_active'):.1f}%")
        lines.append(f"Friction RL   ~ {pct('friction_rl_active'):.1f}%")
        lines.append(f"Force bounds  ~ {pct('force_bounds_active'):.1f}%")
        lines.append(f"Speed bounds  ~ {pct('speed_bounds_active'):.1f}%")

    ax.text(
        0.0, 1.0, "\n".join(lines), transform=ax.transAxes, fontsize=9, va="top", family="monospace"
    )


def _compute_constraint_activity_four_wheel(
    *,
    d,
    w_left,
    w_right,
    v,
    Fx_fl,
    Fx_fr,
    Fx_rr,
    Fx_rl,
    util_fl,
    util_fr,
    util_rr,
    util_rl,
    params,
):
    n = len(d)
    if n == 0:
        return {}

    tol_d = 0.05
    near_left = np.abs(d - w_left) < tol_d
    near_right = np.abs(d + w_right) < tol_d
    track_active = float(np.count_nonzero(near_left | near_right) / n)

    tol_fric = 5.0
    fric_fl = float(np.count_nonzero(util_fl > 100.0 - tol_fric) / n)
    fric_fr = float(np.count_nonzero(util_fr > 100.0 - tol_fric) / n)
    fric_rr = float(np.count_nonzero(util_rr > 100.0 - tol_fric) / n)
    fric_rl = float(np.count_nonzero(util_rl > 100.0 - tol_fric) / n)

    Fx_max = float(params.get("Fx_max", 1500.0))
    tol_f = 50.0
    all_fx = np.concatenate([Fx_fl, Fx_fr, Fx_rr, Fx_rl])
    force_active = float(np.count_nonzero(np.abs(np.abs(all_fx) - Fx_max) < tol_f) / (4 * n))

    v_min = float(params.get("v_min", 0.4))
    v_max = float(params.get("v_max", 20.0))
    tol_v = 0.5
    v_active = float(
        np.count_nonzero((np.abs(v - v_min) < tol_v) | (np.abs(v - v_max) < tol_v)) / n
    )

    return {
        "track_bounds_active": track_active,
        "friction_fl_active": fric_fl,
        "friction_fr_active": fric_fr,
        "friction_rr_active": fric_rr,
        "friction_rl_active": fric_rl,
        "force_bounds_active": force_active,
        "speed_bounds_active": v_active,
    }


def plot_all_panels_four_wheel(
    cones_left,
    cones_right,
    path_xy,
    v,
    s,
    d,
    w_left,
    w_right,
    Fx_fl,
    Fx_fr,
    Fx_rr,
    Fx_rl,
    delta,
    v_lat,
    yaw_rate,
    params: Dict,
    profiling: Optional[Dict] = None,
    input_data: Optional[Dict] = None,
    timed_mask: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
):
    forces = _compute_four_wheel_forces(
        v=v,
        v_lat=v_lat,
        yaw_rate=yaw_rate,
        Fx_fl=Fx_fl,
        Fx_fr=Fx_fr,
        Fx_rr=Fx_rr,
        Fx_rl=Fx_rl,
        delta=delta,
        params=params,
    )

    a_x, a_y = _compute_body_accels(
        Fx_fl=Fx_fl,
        Fx_fr=Fx_fr,
        Fx_rr=Fx_rr,
        Fx_rl=Fx_rl,
        Fy_fl=forces["Fy_fl"],
        Fy_fr=forces["Fy_fr"],
        Fy_rr=forces["Fy_rr"],
        Fy_rl=forces["Fy_rl"],
        delta=delta,
        F_drag=forces["F_drag"],
        F_roll=forces["F_roll"],
        params=params,
    )

    constraint_activity = _compute_constraint_activity_four_wheel(
        d=d,
        w_left=w_left,
        w_right=w_right,
        v=v,
        Fx_fl=Fx_fl,
        Fx_fr=Fx_fr,
        Fx_rr=Fx_rr,
        Fx_rl=Fx_rl,
        util_fl=forces["util_fl"],
        util_fr=forces["util_fr"],
        util_rr=forces["util_rr"],
        util_rl=forces["util_rl"],
        params=params,
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    axes = axes.flatten()

    # 1. Track + speed
    plot_path_with_speed(
        cones_left, cones_right, path_xy, v, out_path=None, show=False, fig=fig, ax=axes[0]
    )
    axes[0].set_title("Path (speed colormap)")

    # 2. Speed profile
    plot_speed_profile(
        s, v, v_max=None, out_path=None, show=False, fig=fig, ax=axes[1], timed_mask=timed_mask
    )
    axes[1].set_title("Speed profile")

    # 3. Lateral offset
    plot_offsets(s, d, w_left, w_right, out_path=None, show=False, fig=fig, ax=axes[2])
    axes[2].set_title("Lateral offset")

    # 4. Per-wheel Fx
    ax4 = axes[3]
    for name, arr in [("FL", Fx_fl), ("FR", Fx_fr), ("RR", Fx_rr), ("RL", Fx_rl)]:
        ax4.plot(s, arr, color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    Fx_max = float(params.get("Fx_max", 1500.0))
    ax4.axhline(Fx_max, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax4.axhline(-Fx_max, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax4.set_xlabel("s [m]")
    ax4.set_ylabel("Fx [N]")
    ax4.legend(fontsize=8)
    ax4.grid(True, ls="--", alpha=0.4)
    ax4.set_title("Per-wheel Fx")

    # 5. GG diagram (a_lat on x, a_long on y — standard convention)
    ax5 = axes[4]
    ax5.scatter(a_y, a_x, s=6, alpha=0.6)
    ax5.set_xlabel("a_lat [m/s^2]")
    ax5.set_ylabel("a_long [m/s^2]")
    ax5.set_aspect("equal", adjustable="box")
    ax5.grid(True, ls="--", alpha=0.4)
    ax5.set_title("GG diagram")

    # 6. Profiling
    _plot_profiling_panel_four_wheel(profiling, constraint_activity, axes[5])
    axes[5].set_title("Profiling & activity")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)

        plot_diagnostics_four_wheel(
            s=s,
            v=v,
            v_lat=v_lat,
            yaw_rate=yaw_rate,
            Fx_fl=Fx_fl,
            Fx_fr=Fx_fr,
            Fx_rr=Fx_rr,
            Fx_rl=Fx_rl,
            delta=delta,
            forces=forces,
            params=params,
            input_data=input_data,
            out_path=out_path.parent / "diagnostics.png",
            show=show,
        )

    if show:
        plt.show()
    plt.close(fig)


def plot_diagnostics_four_wheel(
    *,
    s,
    v,
    v_lat,
    yaw_rate,
    Fx_fl,
    Fx_fr,
    Fx_rr,
    Fx_rl,
    delta,
    forces,
    params,
    input_data: Optional[Dict] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
):
    Mz_Fx, Mz_total = _compute_yaw_moments(
        Fx_fl=Fx_fl,
        Fx_fr=Fx_fr,
        Fx_rr=Fx_rr,
        Fx_rl=Fx_rl,
        Fy_fl=forces["Fy_fl"],
        Fy_fr=forces["Fy_fr"],
        Fy_rr=forces["Fy_rr"],
        Fy_rl=forces["Fy_rl"],
        delta=delta,
        params=params,
    )

    dFxmax = float(params.get("dFxmax", 1000.0))
    ddeltamax = float(params.get("ddeltamax", 1.3))

    # Physical rates from normalised inputs
    if input_data is not None:
        Fx_fl_dot = np.array(input_data.get("Fx_fl_dot_norm", np.zeros_like(s))) * dFxmax
        Fx_fr_dot = np.array(input_data.get("Fx_fr_dot_norm", np.zeros_like(s))) * dFxmax
        Fx_rr_dot = np.array(input_data.get("Fx_rr_dot_norm", np.zeros_like(s))) * dFxmax
        Fx_rl_dot = np.array(input_data.get("Fx_rl_dot_norm", np.zeros_like(s))) * dFxmax
        delta_dot = np.array(input_data.get("delta_dot_norm", np.zeros_like(s))) * ddeltamax
    else:
        Fx_fl_dot = Fx_fr_dot = Fx_rr_dot = Fx_rl_dot = np.zeros_like(s)
        delta_dot = np.zeros_like(s)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    # 1. Steering angle (rad) + steering rate (rad/s) on secondary axis
    ax1 = axes[0]
    ax1.plot(s, delta, color="tab:purple", label="delta")
    ax1.set_ylabel("delta [rad]", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.set_xlabel("s [m]")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1r = ax1.twinx()
    ax1r.plot(s, delta_dot, color="tab:gray", label="delta_dot", linewidth=0.8, alpha=0.7)
    ax1r.set_ylabel("delta_dot [rad/s]", color="tab:gray")
    ax1r.tick_params(axis="y", labelcolor="tab:gray")
    ax1.set_title("Steering angle & rate")

    # 2. Per-wheel slip angles
    for name, key in [
        ("FL", "alpha_fl"),
        ("FR", "alpha_fr"),
        ("RR", "alpha_rr"),
        ("RL", "alpha_rl"),
    ]:
        axes[1].plot(
            s, np.degrees(forces[key]), color=WHEEL_COLORS[name], label=name, linewidth=0.8
        )
    axes[1].set_ylabel("alpha [deg]")
    axes[1].set_xlabel("s [m]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", alpha=0.4)
    axes[1].set_title("Slip angles")

    # 3. Yaw moments (TV + total merged)
    axes[2].plot(s, Mz_Fx, color="tab:cyan", label="Mz (Fx only)", linewidth=0.8)
    axes[2].plot(s, Mz_total, color="tab:brown", label="Mz total", linewidth=0.8)
    axes[2].set_ylabel("Mz [Nm]")
    axes[2].set_xlabel("s [m]")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, ls="--", alpha=0.4)
    axes[2].set_title("Yaw moments")

    # 4. Force rates of change
    for name, arr in [("FL", Fx_fl_dot), ("FR", Fx_fr_dot), ("RR", Fx_rr_dot), ("RL", Fx_rl_dot)]:
        axes[3].plot(s, arr, color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[3].axhline(dFxmax, color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[3].axhline(-dFxmax, color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[3].set_ylabel("dFx/dt [N/s]")
    axes[3].set_xlabel("s [m]")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, ls="--", alpha=0.4)
    axes[3].set_title("Wheel force rates")

    # 5. Friction utilization
    for name, key in [("FL", "util_fl"), ("FR", "util_fr"), ("RR", "util_rr"), ("RL", "util_rl")]:
        axes[4].plot(s, forces[key], color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[4].axhline(100, color="gray", ls="--", lw=0.7)
    axes[4].set_ylabel("utilization [%]")
    axes[4].set_xlabel("s [m]")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, ls="--", alpha=0.4)
    axes[4].set_title("Friction utilization")

    # 6. Vertical loads
    for name, key in [("FL", "Fz_fl"), ("FR", "Fz_fr"), ("RR", "Fz_rr"), ("RL", "Fz_rl")]:
        axes[5].plot(s, forces[key], color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[5].set_ylabel("Fz [N]")
    axes[5].set_xlabel("s [m]")
    axes[5].legend(fontsize=8)
    axes[5].grid(True, ls="--", alpha=0.4)
    axes[5].set_title("Vertical loads")

    # 7. Lateral forces
    for name, key in [("FL", "Fy_fl"), ("FR", "Fy_fr"), ("RR", "Fy_rr"), ("RL", "Fy_rl")]:
        axes[6].plot(s, forces[key], color=WHEEL_COLORS[name], label=name, linewidth=0.8)
    axes[6].set_ylabel("Fy [N]")
    axes[6].set_xlabel("s [m]")
    axes[6].legend(fontsize=8)
    axes[6].grid(True, ls="--", alpha=0.4)
    axes[6].set_title("Lateral forces")

    # 8. v_lat and yaw_rate
    ax8 = axes[7]
    ax8.plot(s, v_lat, color="tab:blue", label="v_lat")
    ax8.set_ylabel("v_lat [m/s]", color="tab:blue")
    ax8.tick_params(axis="y", labelcolor="tab:blue")
    ax8r = ax8.twinx()
    ax8r.plot(s, yaw_rate, color="tab:orange", label="yaw_rate")
    ax8r.set_ylabel("yaw_rate [rad/s]", color="tab:orange")
    ax8r.tick_params(axis="y", labelcolor="tab:orange")
    ax8.set_xlabel("s [m]")
    ax8.grid(True, ls="--", alpha=0.4)
    ax8.set_title("Lateral velocity & yaw rate")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
