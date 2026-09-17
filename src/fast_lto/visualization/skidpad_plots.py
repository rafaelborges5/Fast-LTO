"""Skidpad LTO solution plots.

The cone map is unlabeled rings (not L/R/M polylines) and the path
self-overlaps, so this is separate from the generic track panels.

Draws: XY overview (cones, centres/gate, path by speed, timed laps
highlighted); speed and lateral deviation vs arc length; per-wheel
longitudinal forces or other available inputs. Annotates timed-lap times
and the FS score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _s_dot(data: Dict) -> np.ndarray:
    """Progress rate ds/dt, matching the OCP time quadrature.

    s_dot = (v_long*cos(psi_err) - v_lat*sin(psi_err)) / (1 - kappa*d)
    """
    v_long = np.array(data.get("v_long", data.get("v")), dtype=np.float64)
    psi_err = np.array(data.get("psi_err", np.zeros_like(v_long)), dtype=np.float64)
    v_lat = np.array(data.get("v_lat", np.zeros_like(v_long)), dtype=np.float64)
    d = np.array(data["d"], dtype=np.float64)
    kappa = np.array(data["kappa"], dtype=np.float64)
    D_kappa = 1.0 - kappa * d
    return (v_long * np.cos(psi_err) - v_lat * np.sin(psi_err)) / D_kappa


def _node_dt(data: Dict) -> np.ndarray:
    """Per-interval traversal time dt[i] = ds[i] / s_dot[i] (left-node Euler)."""
    s = np.array(data["arc_lengths"], dtype=np.float64)
    s_dot = _s_dot(data)
    ds = np.diff(s)
    return ds / np.maximum(s_dot[:-1], 1e-6)


def timed_blocks(timed: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end) index ranges of contiguous timed==1 nodes."""
    idx = np.where(timed == 1)[0]
    if idx.size == 0:
        return []
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(int(sp[0]), int(sp[-1])) for sp in splits]


def compute_timed_lap_times(data: Dict, timed: np.ndarray) -> List[float]:
    """Integrate traversal time over each contiguous timed block (ds/s_dot)."""
    dt = _node_dt(data)
    times = []
    for a, b in timed_blocks(timed):
        hi = min(b + 1, len(dt))
        times.append(float(np.sum(dt[a:hi])))
    return times


def plot_skidpad(
    data: Dict,
    cones: np.ndarray,
    out_path: Path | None = None,
    show: bool = False,
) -> Dict:
    """Render the skidpad diagnostic panels. Returns a summary dict."""
    path_xy = np.array(data["path_xy"], dtype=np.float64)
    v = np.array(data.get("v_long", data.get("v")), dtype=np.float64)
    s = np.array(data["arc_lengths"], dtype=np.float64)
    d = np.array(data["d"], dtype=np.float64)
    w_left = np.array(data["w_left"], dtype=np.float64)
    w_right = np.array(data["w_right"], dtype=np.float64)
    timed = np.asarray(data.get("timed_mask") or np.zeros(len(s)), dtype=int)
    sk = data.get("skidpad", {}) or {}
    profiling = data.get("profiling", {}) or {}

    lap_times = compute_timed_lap_times(data, timed)
    score = float(np.mean(lap_times)) if lap_times else float("nan")
    full_time = profiling.get("lap_time_s")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.scatter(cones[:, 0], cones[:, 1], c="dimgray", s=16, label="cones", zorder=2)
    sc = ax.scatter(path_xy[:, 0], path_xy[:, 1], c=v, cmap="viridis", s=8, zorder=3)
    tm = timed == 1
    ax.scatter(
        path_xy[tm, 0],
        path_xy[tm, 1],
        facecolors="none",
        edgecolors="red",
        s=22,
        linewidths=0.6,
        label="timed laps",
        zorder=4,
    )
    for key, mark in (("c_first", "C1"), ("c_second", "C3")):
        if key in sk:
            c = sk[key]
            ax.plot(c[0], c[1], "x", color=mark, markersize=12, mew=2)
    if "gate" in sk:
        g = sk["gate"]
        ax.plot(g[0], g[1], "k+", markersize=14, mew=2, label="gate")
    fig.colorbar(sc, ax=ax, label="v [m/s]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Optimal skidpad path (colored by speed)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    ax = axes[0, 1]
    ax.plot(s, v, color="tab:blue", lw=1.2)
    for a, b in timed_blocks(timed):
        ax.axvspan(s[a], s[b], color="red", alpha=0.12)
    ax.set_title("Speed vs arc length (timed laps shaded)")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("v [m/s]")
    ax.grid(True, ls="--", alpha=0.3)

    ax = axes[1, 0]
    ax.plot(s, d, color="tab:green", lw=1.0, label="d (line offset)")
    ax.plot(s, w_left, color="tab:blue", lw=0.8, ls="--", label="+w_left")
    ax.plot(s, -w_right, color="tab:orange", lw=0.8, ls="--", label="-w_right")
    for a, b in timed_blocks(timed):
        ax.axvspan(s[a], s[b], color="red", alpha=0.12)
    ax.set_title("Lateral deviation d(s) and corridor")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("d [m]")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    ax = axes[1, 1]
    wheel_keys = ["Fx_fl", "Fx_fr", "Fx_rr", "Fx_rl"]
    if all(k in data for k in wheel_keys):
        for k in wheel_keys:
            ax.plot(s, np.array(data[k], dtype=np.float64), lw=0.9, label=k)
        ax.set_ylabel("Fx [N]")
        ax.set_title("Per-wheel longitudinal forces")
    elif "delta" in data:
        ax.plot(s, np.array(data["delta"], dtype=np.float64), lw=1.0, label="delta")
        ax.set_ylabel("delta [rad]")
        ax.set_title("Steering angle")
    for a, b in timed_blocks(timed):
        ax.axvspan(s[a], s[b], color="red", alpha=0.12)
    ax.set_xlabel("s [m]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls="--", alpha=0.3)

    lap_str = ", ".join(f"{t:.3f}s" for t in lap_times) if lap_times else "n/a"
    full_str = f"{full_time:.2f}s" if full_time is not None else "n/a"
    fig.suptitle(
        f"Skidpad LTO  |  timed laps: {lap_str}  |  score (avg): {score:.3f}s  "
        f"|  full maneuver: {full_str}  |  v_max={v.max():.2f} m/s",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    return {"timed_lap_times": lap_times, "score": score, "full_time": full_time}


__all__ = ["plot_skidpad", "compute_timed_lap_times", "timed_blocks"]
