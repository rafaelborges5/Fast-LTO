"""
Visualization helpers for OCP solutions.

Functions accept numpy arrays and can be used from scripts or the main pipeline.
Run directly to plot the last saved solution (ellipse_point_mass.npz) if present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Running as script - add repo/src to path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.track_bounds import load_boundaries
else:
    from utils.track_bounds import load_boundaries


def plot_path_with_speed(
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    path_xy: np.ndarray,
    v: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(cones_left[:, 0], cones_left[:, 1], c="tab:blue", s=12, label="left cones")
    ax.scatter(cones_right[:, 0], cones_right[:, 1], c="tab:orange", s=12, label="right cones")
    sc = ax.scatter(path_xy[:, 0], path_xy[:, 1], c=v, cmap="viridis", s=8, label="path (v-colored)")
    cbar = plt.colorbar(sc, ax=ax, label="speed [m/s]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Path vs cones (speed colored)")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_speed_profile(
    s: np.ndarray,
    v: np.ndarray,
    v_max: Optional[float] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, v, label="v(s)")
    if v_max is not None:
        ax.axhline(v_max, color="r", linestyle="--", label="v_max")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("v [m/s]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Speed profile")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_gg(
    a_long: np.ndarray,
    a_lat: np.ndarray,
    mu_g: float,
    a_long_bounds: Optional[tuple[float, float]] = None,
    a_lat_bounds: Optional[tuple[float, float]] = None,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(a_long, a_lat, s=8, alpha=0.7, label="samples")
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_x = mu_g * np.cos(theta)
    circ_y = mu_g * np.sin(theta)
    ax.plot(circ_x, circ_y, "r--", label="friction circle")
    if a_long_bounds is not None and a_lat_bounds is not None:
        ax.plot([a_long_bounds[0], a_long_bounds[1], a_long_bounds[1], a_long_bounds[0], a_long_bounds[0]],
                [a_lat_bounds[0], a_lat_bounds[0], a_lat_bounds[1], a_lat_bounds[1], a_lat_bounds[0]],
                "k--", alpha=0.5, label="a box")
    ax.set_xlabel("a_long [m/s^2]")
    ax.set_ylabel("a_lat [m/s^2]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("GG diagram")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_offsets(
    s: np.ndarray,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, d, label="d(s)")
    ax.plot(s, w_left, "g--", label="w_left")
    ax.plot(s, -w_right, "r--", label="-(w_right)")
    ax.fill_between(s, -w_right, w_left, color="gray", alpha=0.1)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("d [m]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Lateral offset and bounds")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_inputs(
    s: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    out_path: Optional[Path] = None,
    show: bool = True,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s, a_long, label="a_long")
    ax.plot(s, a_lat, label="a_lat")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("accel [m/s^2]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Inputs vs s")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    if fig is None:
        plt.close("all")


def plot_all_panels(
    cones_left: np.ndarray,
    cones_right: np.ndarray,
    path_xy: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    d: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    a_long: np.ndarray,
    a_lat: np.ndarray,
    mu_g: float,
    out_path: Optional[Path] = None,
    show: bool = True,
):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    plot_path_with_speed(cones_left, cones_right, path_xy, v, out_path=None, show=False, fig=fig, ax=axes[0])
    axes[0].set_title("Path vs cones")

    plot_speed_profile(s, v, v_max=None, out_path=None, show=False, fig=fig, ax=axes[1])
    axes[1].set_title("Speed profile")

    plot_offsets(s, d, w_left, w_right, out_path=None, show=False, fig=fig, ax=axes[2])
    axes[2].set_title("Lateral offset")

    plot_inputs(s, a_long, a_lat, out_path=None, show=False, fig=fig, ax=axes[3])
    axes[3].set_title("Inputs")

    plot_gg(a_long, a_lat, mu_g, out_path=None, show=False, fig=fig, ax=axes[4])
    axes[4].set_title("GG diagram")

    fig.delaxes(axes[5])
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _demo() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    solution_path = repo_root / "data" / "solutions" / "ellipse_point_mass.json"
    cones_csv = repo_root / "data" / "tracks" / "ellipse.csv"
    timestamp_dir = repo_root / "ocp_plots" / datetime.now().strftime("%Y%m%d-%H%M%S")
    if not solution_path.exists():
        print(f"Solution not found at {solution_path}, run global_ocp first.")
        return

    with solution_path.open("r") as f:
        data = json.load(f)
    boundaries = load_boundaries(cones_csv)
    cones_left = boundaries["left"]
    cones_right = boundaries["right"]

    path_xy = np.array(data["path_xy"], dtype=np.float64)
    v = np.array(data["v"], dtype=np.float64)
    s = np.array(data["arc_lengths"], dtype=np.float64)
    d = np.array(data["d"], dtype=np.float64)
    w_left = np.array(data["w_left"], dtype=np.float64)
    w_right = np.array(data["w_right"], dtype=np.float64)
    a_long = np.array(data["a_long"], dtype=np.float64)
    a_lat = np.array(data["a_lat"], dtype=np.float64)
    params = data.get("model_params", {})
    mu = params.get("mu", 1.2)
    g_val = params.get("g", 9.81)
    mu_g = mu * g_val
    a_long_bounds = (params.get("a_long_min", -np.inf), params.get("a_long_max", np.inf))
    a_lat_bounds = (params.get("a_lat_min", -np.inf), params.get("a_lat_max", np.inf))

    timestamp_dir.mkdir(parents=True, exist_ok=True)

    plot_all_panels(
        cones_left,
        cones_right,
        path_xy,
        v,
        s,
        d,
        w_left,
        w_right,
        a_long,
        a_lat,
        mu_g,
        out_path=timestamp_dir / "panels.png",
        show=True,
    )
    print(f"Saved plots to {timestamp_dir}")


if __name__ == "__main__":
    _demo()

