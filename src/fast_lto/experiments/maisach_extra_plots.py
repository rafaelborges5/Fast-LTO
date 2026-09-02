"""
Extra plots for the Maisach four-wheel solution:
  1. CoG lateral deviation + four corner lateral positions vs track bounds
  2. Steering rate panel (delta_dot)
  3. Clean trajectory-only comparison (no boundary lines)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "ocp_plots" / "maisach_extras"


def _load_maisach_4w():
    with (REPO / "data" / "solutions" / "track_boundary_maisach_four_wheel_euler.json").open() as f:
        return json.load(f)


def plot_corner_lateral_deviation():
    """Plot 1: CoG d(s) and four corner d(s) vs track bounds."""
    sol = _load_maisach_4w()
    params = sol["model_params"]

    s = np.array(sol["arc_lengths"])
    d = np.array(sol["d"])
    psi_err = np.array(sol["psi_err"])
    kappa = np.array(sol["kappa"])
    w_left = np.array(sol["w_left"])
    w_right = np.array(sol["w_right"])

    corners = params["corners"]
    margin = 0.15

    sin_psi = np.sin(psi_err)
    cos_psi = np.cos(psi_err)
    D_kappa = 1 - kappa * d

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(s, w_left, -w_right, alpha=0.08, color="gray", label="track bounds")
    ax.fill_between(
        s,
        w_left - margin,
        -(w_right - margin),
        alpha=0.06,
        color="orange",
        label=f"bounds - {margin}m margin",
    )

    ax.plot(s, d, "k-", lw=1.5, label="CoG", zorder=5)

    colors = {"FL": "tab:blue", "FR": "tab:orange", "RL": "tab:red", "RR": "tab:green"}
    for c_name, dx, dy in corners:
        long_proj = dx * cos_psi - dy * sin_psi
        d_corner = d + dx * sin_psi + dy * cos_psi - 0.5 * kappa / D_kappa * long_proj**2
        ax.plot(s, d_corner, color=colors.get(c_name, "gray"), lw=0.8, alpha=0.8, label=c_name)

    ax.set_xlabel("s [m]")
    ax.set_ylabel("lateral position [m]")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_title("CoG + four corners lateral deviation (Maisach, four-wheel model)")
    fig.tight_layout()
    fig.savefig(OUT / "corner_lateral_deviation.png", dpi=200)
    print("Saved: corner_lateral_deviation.png")
    plt.close(fig)


def plot_steering_rate():
    """Plot 2: Steering angle and steering rate vs s."""
    sol = _load_maisach_4w()
    params = sol["model_params"]

    s = np.array(sol["arc_lengths"])
    delta = np.array(sol["delta"])
    delta_dot_norm = np.array(sol["delta_dot_norm"])
    ddeltamax = float(params["ddeltamax"])
    delta_rate = delta_dot_norm * ddeltamax

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(s, np.degrees(delta), color="tab:purple", lw=1.0)
    delta_max = float(params.get("delta_max", 0.4))
    axes[0].axhline(np.degrees(delta_max), color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[0].axhline(-np.degrees(delta_max), color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[0].set_ylabel("delta [deg]")
    axes[0].grid(True, ls="--", alpha=0.3)
    axes[0].set_title("Steering angle and rate (Maisach, four-wheel)")

    axes[1].plot(s, delta_rate, color="tab:cyan", lw=1.0)
    axes[1].axhline(
        ddeltamax, color="gray", ls="--", lw=0.7, alpha=0.5, label=f"limit: +/-{ddeltamax} rad/s"
    )
    axes[1].axhline(-ddeltamax, color="gray", ls="--", lw=0.7, alpha=0.5)
    axes[1].set_xlabel("s [m]")
    axes[1].set_ylabel("delta_dot [rad/s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "steering_rate.png", dpi=200)
    print("Saved: steering_rate.png")
    plt.close(fig)


def plot_trajectory_comparison_clean():
    """Plot 4: Clean trajectory-only comparison (no boundary scatter)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Load all solutions
    models = {
        "point_mass": {
            "file": REPO / "data" / "experiments" / "maisach_fair_point_mass_ds1.0.json",
            "label": "Point Mass (47.0s)",
            "color": "tab:green",
            "lw": 1.2,
        },
        "dynamic_bicycle": {
            "file": REPO / "data" / "experiments" / "maisach_fair_dynamic_bicycle_ds1.0.json",
            "label": "Dyn. Bicycle (49.4s)",
            "color": "tab:orange",
            "lw": 1.2,
        },
        "four_wheel_no_aero": {
            "file": REPO / "data" / "experiments" / "maisach_fair_four_wheel_no_aero_ds1.0.json",
            "label": "4W no aero (46.3s)",
            "color": "tab:blue",
            "lw": 1.2,
        },
        "four_wheel_full": {
            "file": REPO / "data" / "experiments" / "maisach_full_four_wheel_ds1.0.json",
            "label": "4W full aero (42.1s)",
            "color": "tab:red",
            "lw": 1.5,
            "ls": "--",
        },
    }

    # Left panel: full track overview
    ax = axes[0]
    for cfg in models.values():
        d = json.load(cfg["file"].open())
        xy = np.array(d["path_xy"])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=cfg["color"],
            ls=cfg.get("ls", "-"),
            lw=cfg["lw"],
            label=cfg["label"],
        )
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best")
    ax.set_title("Trajectory comparison — full track")
    ax.grid(True, ls="--", alpha=0.2)

    # Right panel: zoomed into a tight section
    # Find a high-curvature section
    sol_4w = json.load(models["four_wheel_full"]["file"].open())
    kappa = np.array(sol_4w["kappa"])
    abs_kappa = np.abs(kappa)
    # Find the tightest corner region
    window = 30
    conv = np.convolve(abs_kappa, np.ones(window) / window, mode="valid")
    peak_idx = np.argmax(conv) + window // 2
    xy_4w = np.array(sol_4w["path_xy"])
    cx, cy = xy_4w[peak_idx]

    ax2 = axes[1]
    for cfg in models.values():
        d = json.load(cfg["file"].open())
        xy = np.array(d["path_xy"])
        ax2.plot(
            xy[:, 0],
            xy[:, 1],
            color=cfg["color"],
            ls=cfg.get("ls", "-"),
            lw=cfg["lw"] * 1.5,
            label=cfg["label"],
        )
    zoom = 12
    ax2.set_xlim(cx - zoom, cx + zoom)
    ax2.set_ylim(cy - zoom, cy + zoom)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.set_title("Zoomed — tightest corner")
    ax2.grid(True, ls="--", alpha=0.2)

    fig.suptitle("Trajectory Comparison — Maisach Track", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "trajectory_comparison_clean.png", dpi=200)
    print("Saved: trajectory_comparison_clean.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_corner_lateral_deviation()
    plot_steering_rate()
    plot_trajectory_comparison_clean()
    print("Done.")
