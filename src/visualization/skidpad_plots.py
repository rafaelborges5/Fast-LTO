"""
Visualization for the partially-timed skidpad LTO solution.

The skidpad cone map is a set of unlabeled ring cones (not L/R/M polylines), and
the optimal path self-overlaps (each circle driven twice), so this plotter is
separate from the generic track panels. It draws a 3×2 grid:

  A) XY overview: cones + fitted centers/gate, the optimal path colored by speed,
     with the two *timed* laps highlighted.
  B) speed vs arc length, timed laps shaded.
  C) lateral deviation d(s) within the corridor.
  D) per-wheel longitudinal forces (four_wheel) or steering angle.
  E) per-wheel friction utilisation √(Fx²+Fy²) / (D·Fz).
  F) GG diagram (lateral on x-axis, longitudinal on y-axis), coloured by phase.

It also computes and annotates the two timed-lap times and the FS score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _compute_tyre_derived(data: Dict) -> Optional[Dict]:
    """Reconstruct Fy and friction utilisation from stored states.

    Returns None when the required four-wheel model outputs are absent.
    """
    wheel_keys = ["Fx_fl", "Fx_fr", "Fx_rr", "Fx_rl"]
    if not all(k in data for k in wheel_keys):
        return None

    try:
        from visualization.ocp_plots_four_wheel import _compute_four_wheel_forces
    except ImportError:
        return None

    mp = data.get("model_params", {})
    v = np.array(data.get("v_long", data.get("v")), dtype=np.float64)
    v_lat = np.array(data.get("v_lat", np.zeros_like(v)), dtype=np.float64)
    yaw = np.array(data.get("yaw_rate", np.zeros_like(v)), dtype=np.float64)
    delta = np.array(data.get("delta", np.zeros_like(v)), dtype=np.float64)
    Fx_fl = np.array(data["Fx_fl"], dtype=np.float64)
    Fx_fr = np.array(data["Fx_fr"], dtype=np.float64)
    Fx_rr = np.array(data["Fx_rr"], dtype=np.float64)
    Fx_rl = np.array(data["Fx_rl"], dtype=np.float64)

    return _compute_four_wheel_forces(
        v=v, v_lat=v_lat, yaw_rate=yaw, delta=delta,
        Fx_fl=Fx_fl, Fx_fr=Fx_fr, Fx_rr=Fx_rr, Fx_rl=Fx_rl,
        params=mp,
    )


def plot_skidpad(
    data: Dict,
    cones: np.ndarray,
    out_path: Path | None = None,
    show: bool = False,
) -> Dict:
    """Render a 3×2 grid of skidpad diagnostic panels. Returns a summary dict."""
    path_xy = np.array(data["path_xy"], dtype=np.float64)
    v = np.array(data.get("v_long", data.get("v")), dtype=np.float64)
    s = np.array(data["arc_lengths"], dtype=np.float64)
    d_lat = np.array(data["d"], dtype=np.float64)
    w_left = np.array(data["w_left"], dtype=np.float64)
    w_right = np.array(data["w_right"], dtype=np.float64)
    yaw = np.array(data.get("yaw_rate", np.zeros_like(v)), dtype=np.float64)
    timed = np.asarray(data.get("timed_mask") or np.zeros(len(s)), dtype=int)
    decel = np.asarray(data.get("decel_mask") or np.zeros(len(s)), dtype=int)
    sk = data.get("skidpad", {}) or {}
    profiling = data.get("profiling", {}) or {}
    mp = data.get("model_params", {}) or {}

    lap_times = compute_timed_lap_times(data, timed)
    score = float(np.mean(lap_times)) if lap_times else float("nan")
    full_time = profiling.get("lap_time_s")

    # Tyre-derived quantities (may be None for non-four-wheel models)
    tyre = _compute_tyre_derived(data)

    # GG accelerations
    Fx_fl = np.array(data.get("Fx_fl", np.zeros_like(v)), dtype=np.float64)
    Fx_fr = np.array(data.get("Fx_fr", np.zeros_like(v)), dtype=np.float64)
    Fx_rr = np.array(data.get("Fx_rr", np.zeros_like(v)), dtype=np.float64)
    Fx_rl = np.array(data.get("Fx_rl", np.zeros_like(v)), dtype=np.float64)
    Fx_total = Fx_fl + Fx_fr + Fx_rr + Fx_rl
    m = float(mp.get("m", 200.0))
    g_acc = float(mp.get("g", 9.81))
    rho = float(mp.get("rho", 1.225))
    Cd = float(mp.get("C_d", 1.58))
    Af = float(mp.get("A_f", 1.2))
    F_drag = tyre["F_drag"] if tyre is not None else 0.5 * rho * Cd * Af * v**2
    a_long = (Fx_total - F_drag) / m
    a_lat = v * yaw  # centripetal approx

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    _C_TIMED = "#2563eb"
    _C_EXIT  = "#dc2626"
    _C_UNTIME = "#d97706"

    def _shade(ax: plt.Axes) -> None:
        for a, b in timed_blocks(timed):
            ax.axvspan(s[a], s[b], color=_C_TIMED, alpha=0.10, zorder=0)
        di = np.where(decel == 1)[0]
        if di.size:
            ax.axvspan(s[di[0]], s[di[-1]], color=_C_EXIT, alpha=0.10, zorder=0)

    # ── A: XY overview ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(cones[:, 0], cones[:, 1], c="dimgray", s=16, label="cones", zorder=2)
    sc = ax.scatter(path_xy[:, 0], path_xy[:, 1], c=v, cmap="viridis", s=8, zorder=3)
    tm = timed == 1
    ax.scatter(
        path_xy[tm, 0], path_xy[tm, 1],
        facecolors="none", edgecolors=_C_TIMED, s=22, linewidths=0.6,
        label="timed laps", zorder=4,
    )
    for key, col in (("c_first", "C1"), ("c_second", "C3")):
        if key in sk:
            c = sk[key]
            ax.plot(c[0], c[1], "x", color=col, markersize=12, mew=2)
    if "gate" in sk:
        g = sk["gate"]
        ax.plot(g[0], g[1], "k+", markersize=14, mew=2, label="gate")
    fig.colorbar(sc, ax=ax, label="v [m/s]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A  Optimal path (coloured by speed)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

    # ── B: Speed vs arc length ────────────────────────────────────────────────
    ax = axes[0, 1]
    _shade(ax)
    ax.plot(s, v, color=_C_TIMED, lw=1.4)
    ax.set_title("B  Speed vs arc length")
    ax.set_xlabel("s [m]"); ax.set_ylabel("v [m/s]")
    ax.grid(True, ls="--", alpha=0.3)

    # ── C: Lateral deviation ─────────────────────────────────────────────────
    ax = axes[1, 0]
    _shade(ax)
    ax.plot(s, d_lat, color="#16a34a", lw=1.2, label="d (offset)")
    ax.plot(s,  w_left,  color=_C_TIMED, lw=0.9, ls="--", label="+w_left")
    ax.plot(s, -w_right, color=_C_EXIT,  lw=0.9, ls="--", label="−w_right")
    ax.set_title("C  Lateral deviation d(s) and corridor")
    ax.set_xlabel("s [m]"); ax.set_ylabel("d [m]")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.3)

    # ── D: Per-wheel longitudinal forces ──────────────────────────────────────
    ax = axes[1, 1]
    _shade(ax)
    wheel_keys = ["Fx_fl", "Fx_fr", "Fx_rr", "Fx_rl"]
    WC = {"Fx_fl": "#2563eb", "Fx_fr": "#7c3aed", "Fx_rr": "#dc2626", "Fx_rl": "#d97706"}
    if all(k in data for k in wheel_keys):
        for k in wheel_keys:
            ax.plot(s, np.array(data[k], dtype=np.float64), lw=0.9,
                    label=k.replace("Fx_", ""), color=WC[k])
        ax.set_ylabel("Fx [N]")
        ax.set_title("D  Per-wheel longitudinal forces")
    elif "delta" in data:
        ax.plot(s, np.array(data["delta"], dtype=np.float64), lw=1.0, color=_C_TIMED)
        ax.set_ylabel("δ [rad]")
        ax.set_title("D  Steering angle")
    ax.axhline(0, color="#6b7280", lw=0.7)
    ax.set_xlabel("s [m]")
    ax.legend(fontsize=8, ncol=4); ax.grid(True, ls="--", alpha=0.3)

    # ── E: Per-wheel friction utilisation ─────────────────────────────────────
    ax = axes[2, 0]
    _shade(ax)
    if tyre is not None:
        for lbl, ut, col, ls_ in [
            ("FL", tyre["util_fl"], "#2563eb", "-"),
            ("FR", tyre["util_fr"], "#7c3aed", "--"),
            ("RR", tyre["util_rr"], "#dc2626", "-"),
            ("RL", tyre["util_rl"], "#d97706", "--"),
        ]:
            ax.plot(s, ut, color=col, lw=1.2, ls=ls_, label=lbl)
        ax.axhline(100, color="#6b7280", lw=1.0, ls=":", label="100% limit")
        ax.set_ylim(0, 115)
    else:
        ax.text(0.5, 0.5, "Tyre data not available\n(non-four-wheel model)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_title("E  Per-wheel friction utilisation  √(Fx²+Fy²) / (D·Fz)")
    ax.set_xlabel("s [m]"); ax.set_ylabel("Utilisation [%]")
    ax.legend(fontsize=8, ncol=5); ax.grid(True, ls="--", alpha=0.3)

    # ── F: GG diagram (lateral on x, longitudinal on y) ───────────────────────
    ax = axes[2, 1]
    for mask, col, lab, sz, z in [
        ((timed == 0) & (decel == 0), _C_UNTIME, "Untimed",    7, 2),
        (decel == 1,                  _C_EXIT,    "Exit/decel", 9, 3),
        (timed == 1,                  _C_TIMED,   "Timed laps", 9, 4),
    ]:
        ax.scatter(
            a_lat[mask] / g_acc, a_long[mask] / g_acc,
            c=col, s=sz, alpha=0.80, label=lab, zorder=z, linewidths=0,
        )
    ax.axhline(0, color="#6b7280", lw=0.7)
    ax.axvline(0, color="#6b7280", lw=0.7)
    ax.set_xlabel("Lateral acceleration [g]  (v·ω approx)")
    ax.set_ylabel("Longitudinal acceleration [g]")
    ax.set_title("F  GG diagram")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, ls="--", alpha=0.3)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    lap_str = ", ".join(f"{t:.3f}s" for t in lap_times) if lap_times else "n/a"
    full_str = f"{full_time:.2f}s" if full_time is not None else "n/a"
    reg_pct = profiling.get("reg_term_relative", 0.0) * 100
    fig.suptitle(
        f"Skidpad LTO  |  timed laps: {lap_str}  |  score: {score:.3f}s"
        f"  |  full: {full_str}  |  v_max={v.max():.2f} m/s"
        f"  |  reg={reg_pct:.2f}% of obj",
        fontsize=12,
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
