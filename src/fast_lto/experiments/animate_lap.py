"""
Animated lap visualization for the four-wheel model.

Layout:
  Left:  Track with racecar + cones
  Right: Dashboard
    Top 2x2 grid (tall thin bars):
      [Fx Front | Fz Front]
      [Fx Rear  | Fz Rear ]
    Bottom strip: Steering wheels | TV moment | Speed
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.transforms import Affine2D

REPO = Path(__file__).resolve().parents[3]


# ── Racecar shape ────────────────────────────────────────────────────


def _make_car_polygon(front=0.95, rear=0.658, half_w=0.75):
    nose_w = 0.30
    rw = half_w * 0.55
    pts = [
        (front, 0),
        (front - 0.20, nose_w),
        (front - 0.40, nose_w + 0.05),
        (front - 0.15, half_w),
        (front - 0.25, half_w),
        (front * 0.35, rw + 0.08),
        (0.0, rw + 0.05),
        (-0.05, half_w * 0.65),
        (-rear * 0.3, half_w * 0.62),
        (-rear * 0.7, half_w * 0.58),
        (-rear, half_w * 0.55),
        (-rear - 0.05, half_w),
        (-rear - 0.12, half_w),
        (-rear - 0.12, -half_w),
        (-rear - 0.05, -half_w),
        (-rear, -half_w * 0.55),
        (-rear * 0.7, -half_w * 0.58),
        (-rear * 0.3, -half_w * 0.62),
        (-0.05, -half_w * 0.65),
        (0.0, -(rw + 0.05)),
        (front * 0.35, -(rw + 0.08)),
        (front - 0.25, -half_w),
        (front - 0.15, -half_w),
        (front - 0.40, -(nose_w + 0.05)),
        (front - 0.20, -nose_w),
    ]
    return np.array(pts)


def _make_wheel_rect(cx, cy, length=0.22, width=0.12):
    return np.array(
        [
            (cx - length / 2, cy - width / 2),
            (cx + length / 2, cy - width / 2),
            (cx + length / 2, cy + width / 2),
            (cx - length / 2, cy + width / 2),
        ]
    )


def _rotate_translate(pts, angle, tx, ty):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (pts @ R.T) + np.array([tx, ty])


# ── Viz data ─────────────────────────────────────────────────────────


def _compute_viz_data(sol, params):
    path_xy = np.array(sol["path_xy"])
    headings = np.array(sol["headings"])
    psi_err = np.array(sol["psi_err"])
    s = np.array(sol["arc_lengths"])
    v = np.array(sol.get("v", sol.get("v_long")))

    Fx_fl = np.array(sol["Fx_fl"])
    Fx_fr = np.array(sol["Fx_fr"])
    Fx_rr = np.array(sol["Fx_rr"])
    Fx_rl = np.array(sol["Fx_rl"])
    delta = np.array(sol["delta"])

    l_f = params["lf"]
    l_r = params["lr"]
    a_l = params["a_l"]
    a_r = params["a_r"]
    m = params["m"]
    g_val = params["g"]
    L = l_f + l_r
    W = a_l + a_r

    rho = params.get("rho", 1.225)
    C_l = params.get("C_l", 5.54)
    A_f = params.get("A_f", 1.2)
    F_down = 0.5 * rho * C_l * A_f * v**2
    Fw = [
        m * g_val * (l_r / L) * (a_r / W),
        m * g_val * (l_r / L) * (a_l / W),
        m * g_val * (l_f / L) * (a_l / W),
        m * g_val * (l_f / L) * (a_r / W),
    ]
    Fz = np.column_stack([Fw[i] + F_down / 4 for i in range(4)])

    cd = np.cos(delta)
    sd = np.sin(delta)
    Mz_Fx = (
        Fx_fl * (-a_l * cd + l_f * sd) + Fx_fr * (a_r * cd + l_f * sd) + Fx_rr * a_r - Fx_rl * a_l
    )

    return {
        "path_xy": path_xy,
        "vehicle_heading": headings + psi_err,
        "v": v,
        "s": s,
        "delta": delta,
        "Fx": np.column_stack([Fx_fl, Fx_fr, Fx_rr, Fx_rl]),
        "Fz": Fz,
        "Mz_Fx": Mz_Fx,
    }


# ── Tall paired bars ─────────────────────────────────────────────────


def _setup_tall_pair(
    ax, left_label, right_label, left_color, right_color, ylim, title, centered=True
):
    """Two thin vertical bars side by side in a tall axis."""
    ax.set_xlim(-0.2, 2.2)
    if centered:
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="#bbb", lw=0.5)
    else:
        ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([0.4, 1.6])
    ax.set_xticklabels([left_label, right_label], fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", colors="#666", length=0)
    ax.tick_params(axis="y", labelsize=6)

    bw = 0.65
    bar_l = ax.bar(0.4, 0, width=bw, color=left_color, alpha=0.8)[0]
    bar_r = ax.bar(1.6, 0, width=bw, color=right_color, alpha=0.8)[0]

    txt_l = ax.text(0.4, 0, "", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    txt_r = ax.text(1.6, 0, "", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    return bar_l, bar_r, txt_l, txt_r


def _update_tall_pair(val_l, val_r, bar_l, bar_r, txt_l, txt_r, centered):
    bar_l.set_height(val_l)
    bar_r.set_height(val_r)
    txt_l.set_text(f"{val_l:.0f}")
    txt_r.set_text(f"{val_r:.0f}")
    if centered:
        txt_l.set_y(val_l)
        txt_l.set_va("bottom" if val_l >= 0 else "top")
        txt_r.set_y(val_r)
        txt_r.set_va("bottom" if val_r >= 0 else "top")
    else:
        txt_l.set_y(val_l)
        txt_l.set_va("bottom")
        txt_r.set_y(val_r)
        txt_r.set_va("bottom")


# ── Animation ────────────────────────────────────────────────────────


def animate_lap(solution_path, output_path, stride=3, fps=20, track_csv=None):
    with Path(solution_path).open() as f:
        sol = json.load(f)
    params = sol["model_params"]
    viz = _compute_viz_data(sol, params)

    cones_left = cones_right = None
    if track_csv is not None:
        from fast_lto.utils.track_bounds import load_boundaries

        bd = load_boundaries(Path(track_csv))
        cones_left = bd["left"]
        cones_right = bd["right"]

    N = len(viz["s"])
    frames = list(range(0, N, stride))
    if frames[-1] != N - 1:
        frames.append(N - 1)

    l_f = params["lf"]
    l_r = params["lr"]
    a_l = params["a_l"]
    a_r = params["a_r"]

    car_body = _make_car_polygon(front=0.95, rear=l_r, half_w=0.75)
    wheel_locals = [
        _make_wheel_rect(l_f, a_l),
        _make_wheel_rect(l_f, -a_r),
        _make_wheel_rect(-l_r, -a_r),
        _make_wheel_rect(-l_r, a_l),
    ]

    Fx_max = float(params.get("Fx_max", 1500))
    Fz_max = float(np.max(viz["Fz"]) * 1.15)
    Mz_max = float(max(np.abs(viz["Mz_Fx"]).max() * 1.15, 100))
    v_max = float(viz["v"].max() * 1.15)

    # ── Layout ──
    fig = plt.figure(figsize=(20, 11), facecolor="#f5f5f5")
    gs_top = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.06)
    ax_track = fig.add_subplot(gs_top[0])

    # Right panel: bars grid on top, controls strip on bottom
    gs_r = gs_top[1].subgridspec(2, 1, height_ratios=[3.5, 1.3], hspace=0.25)

    # Top: 2x2 grid of tall bar panels  [Fx_F  Fz_F]
    #                                     [Fx_R  Fz_R]
    gs_bars = gs_r[0].subgridspec(2, 2, hspace=0.35, wspace=0.35)
    ax_fx_f = fig.add_subplot(gs_bars[0, 0])
    ax_fz_f = fig.add_subplot(gs_bars[0, 1])
    ax_fx_r = fig.add_subplot(gs_bars[1, 0])
    ax_fz_r = fig.add_subplot(gs_bars[1, 1])

    # Bottom strip: steering | TV | speed
    gs_bot = gs_r[1].subgridspec(1, 4, width_ratios=[1.5, 1, 1, 0.6], wspace=0.4)
    ax_steer = fig.add_subplot(gs_bot[0])
    ax_tv = fig.add_subplot(gs_bot[1])
    ax_speed = fig.add_subplot(gs_bot[2])
    ax_info = fig.add_subplot(gs_bot[3])

    # ── Track ──
    path_xy = viz["path_xy"]
    ax_track.plot(path_xy[:, 0], path_xy[:, 1], color="#ddd", lw=1.0)
    if cones_left is not None:
        ax_track.scatter(
            cones_left[:, 0],
            cones_left[:, 1],
            s=8,
            color="#2980b9",
            marker="^",
            alpha=0.6,
            zorder=2,
        )
    if cones_right is not None:
        ax_track.scatter(
            cones_right[:, 0],
            cones_right[:, 1],
            s=8,
            color="#f39c12",
            marker="^",
            alpha=0.6,
            zorder=2,
        )
    (trail_line,) = ax_track.plot([], [], "-", color="#e74c3c", lw=2.0, alpha=0.5)
    ax_track.set_aspect("equal")
    ax_track.grid(True, ls="--", alpha=0.1)
    ax_track.set_title("Maisach — Four-Wheel LTO", fontsize=13, pad=8)
    pad = 5
    ax_track.set_xlim(path_xy[:, 0].min() - pad, path_xy[:, 0].max() + pad)
    ax_track.set_ylim(path_xy[:, 1].min() - pad, path_xy[:, 1].max() + pad)

    car_patch = plt.Polygon(car_body, closed=True, fc="#2c3e50", ec="#1a252f", lw=0.8, zorder=10)
    ax_track.add_patch(car_patch)
    track_wps = []
    for wl in wheel_locals:
        wp = plt.Polygon(wl, closed=True, fc="#444", ec="#222", lw=0.5, zorder=11)
        ax_track.add_patch(wp)
        track_wps.append(wp)

    # ── Tall bar panels ──
    fx_f = _setup_tall_pair(
        ax_fx_f, "FL", "FR", "tab:blue", "tab:orange", Fx_max, "Fx Front [N]", centered=True
    )
    fx_r = _setup_tall_pair(
        ax_fx_r, "RL", "RR", "tab:red", "tab:green", Fx_max, "Fx Rear [N]", centered=True
    )
    fz_f = _setup_tall_pair(
        ax_fz_f, "FL", "FR", "tab:blue", "tab:orange", Fz_max, "Fz Front [N]", centered=False
    )
    fz_r = _setup_tall_pair(
        ax_fz_r, "RL", "RR", "tab:red", "tab:green", Fz_max, "Fz Rear [N]", centered=False
    )

    # ── Steering wheels ──
    ax_steer.set_xlim(-2.2, 2.2)
    ax_steer.set_ylim(-2.0, 1.4)
    ax_steer.set_aspect("equal")
    ax_steer.set_title("Steering", fontsize=9, pad=2)
    ax_steer.set_xticks([])
    ax_steer.set_yticks([])
    ax_steer.plot([-1.2, 1.2], [0, 0], color="#bbb", lw=2, zorder=1)

    sw_w, sw_h = 0.35, 0.9
    steer_wl = mpatches.FancyBboxPatch(
        (-sw_w / 2, -sw_h / 2),
        sw_w,
        sw_h,
        boxstyle="round,pad=0.05",
        fc="#333",
        ec="#111",
        lw=1,
        zorder=5,
    )
    steer_wl.set_transform(Affine2D().translate(-0.9, 0) + ax_steer.transData)
    ax_steer.add_patch(steer_wl)

    steer_wr = mpatches.FancyBboxPatch(
        (-sw_w / 2, -sw_h / 2),
        sw_w,
        sw_h,
        boxstyle="round,pad=0.05",
        fc="#333",
        ec="#111",
        lw=1,
        zorder=5,
    )
    steer_wr.set_transform(Affine2D().translate(0.9, 0) + ax_steer.transData)
    ax_steer.add_patch(steer_wr)

    steer_text = ax_steer.text(0, -1.6, "", ha="center", va="top", fontsize=9, fontweight="bold")

    # ── TV yaw moment (vertical bar) ──
    ax_tv.set_xlim(0, 1)
    ax_tv.set_ylim(-Mz_max, Mz_max)
    ax_tv.axhline(0, color="#bbb", lw=0.5)
    ax_tv.set_title("TV [Nm]", fontsize=9, pad=2)
    ax_tv.set_xticks([])
    ax_tv.tick_params(axis="y", labelsize=6)
    tv_bar = ax_tv.bar(0.5, 0, width=0.6, color="tab:cyan", alpha=0.8)[0]
    tv_text = ax_tv.text(0.5, 0, "", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Speed (vertical bar) ──
    ax_speed.set_xlim(0, 1)
    ax_speed.set_ylim(0, v_max)
    ax_speed.set_title("Speed", fontsize=9, pad=2)
    ax_speed.set_xticks([])
    ax_speed.tick_params(axis="y", labelsize=6)
    speed_bar = ax_speed.bar(0.5, 0, width=0.6, color="#e74c3c", alpha=0.8)[0]
    speed_text = ax_speed.text(
        0.5, 0, "", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#222"
    )

    # ── Info ──
    ax_info.axis("off")
    info_text = ax_info.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=9,
        transform=ax_info.transAxes,
        family="monospace",
    )

    # ── Update ──
    def _update(frame_idx):
        i = frames[frame_idx]
        x, y = viz["path_xy"][i]
        heading = viz["vehicle_heading"][i]

        car_patch.set_xy(_rotate_translate(car_body, heading, x, y))
        for wp, wl in zip(track_wps, wheel_locals):
            wp.set_xy(_rotate_translate(wl, heading, x, y))

        t0 = max(0, i - 80)
        trail_line.set_data(viz["path_xy"][t0 : i + 1, 0], viz["path_xy"][t0 : i + 1, 1])

        # Fx: [FL, FR, RR, RL]
        _update_tall_pair(viz["Fx"][i, 0], viz["Fx"][i, 1], *fx_f, True)
        _update_tall_pair(viz["Fx"][i, 3], viz["Fx"][i, 2], *fx_r, True)

        # Fz
        _update_tall_pair(viz["Fz"][i, 0], viz["Fz"][i, 1], *fz_f, False)
        _update_tall_pair(viz["Fz"][i, 3], viz["Fz"][i, 2], *fz_r, False)

        # Steering
        delta_i = viz["delta"][i]
        steer_wl.set_transform(Affine2D().rotate(delta_i).translate(-0.9, 0) + ax_steer.transData)
        steer_wr.set_transform(Affine2D().rotate(delta_i).translate(0.9, 0) + ax_steer.transData)
        steer_text.set_text(f"{np.degrees(delta_i):+.1f}°")

        # TV moment
        mz = viz["Mz_Fx"][i]
        tv_bar.set_height(mz)
        tv_bar.set_color("tab:cyan" if mz >= 0 else "#e67e22")
        tv_text.set_text(f"{mz:+.0f}")
        tv_text.set_y(mz)
        tv_text.set_va("bottom" if mz >= 0 else "top")

        # Speed
        v_i = viz["v"][i]
        speed_bar.set_height(v_i)
        speed_text.set_text(f"{v_i:.1f}\n{v_i*3.6:.0f} km/h")
        speed_text.set_y(v_i)

        info_text.set_text(f"s={viz['s'][i]:.0f}/{viz['s'][-1]:.0f}m")
        return ()

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps, blit=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(frames)} frames at {fps} fps ({len(frames)/fps:.1f}s)...")
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    plt.close(fig)


if __name__ == "__main__":
    sol_path = REPO / "data" / "solutions" / "track_boundary_maisach_four_wheel_euler.json"
    csv_path = REPO / "data" / "tracks" / "track_boundary_maisach.csv"
    out_dir = REPO / "ocp_plots" / "maisach_extras"

    # Fast: stride=4, 24fps
    animate_lap(sol_path, out_dir / "lap_animation_fast.gif", stride=4, fps=24, track_csv=csv_path)

    # Slow: stride=1, 12fps
    animate_lap(sol_path, out_dir / "lap_animation_slow.gif", stride=1, fps=12, track_csv=csv_path)
