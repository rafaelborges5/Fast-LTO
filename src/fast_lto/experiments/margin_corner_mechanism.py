"""
Why one fast corner ignores boundary margin (v_max=27 clean solves).

Shows the corner at s~115-122 is a local speed MAXIMUM between two grip-limited
corners, run at only a fraction of available lateral grip -> not cornering-
limited, so margin (track width) does not change its speed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]

SOL = REPO / "data" / "experiments" / "margin_sweep_corner_v27_clean"
OUT = REPO / "ocp_plots" / "maisach_extras"
MARGINS = [0.12, 0.25, 0.37, 0.50]
COL = {0.12: "tab:blue", 0.25: "tab:green", 0.37: "tab:orange", 0.50: "tab:red"}
LO, HI = 100.0, 145.0


def path_curvature(xy):
    dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)


def grip_g(v, p):
    Fz = p["m"] * p["g"] + 0.5 * p["rho"] * p["C_l"] * p.get("A_f", 1.2) * v**2
    return p["D_fl"] * Fz / (p["m"] * p["g"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = {m: json.load(open(SOL / f"margin_{int(m*100):03d}.json")) for m in MARGINS}
    arc = np.array(S[0.12]["arc_lengths"])
    p = S[0.12]["model_params"]
    vmax = p["v_max"]
    sel = (arc >= LO) & (arc <= HI)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: speed profiles (all margins) — show the invariant peak
    ax = axes[0]
    for m in MARGINS:
        ax.plot(
            arc[sel],
            np.array(S[m]["v_long"])[sel],
            lw=2.0,
            color=COL[m],
            label=f"{int(m*100)} cm  ({S[m]['profiling']['lap_time_s']:.2f}s)",
        )
    ax.axhline(vmax, color="gray", ls="--", lw=0.8, label=f"v_max={vmax:.0f}")
    ax.axvspan(113, 124, color="gold", alpha=0.18, label="margin-invariant peak")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Speed — all margins overlap at the fast bend (a local maximum, not an apex)")
    ax.legend(fontsize=8, ncol=3, loc="lower center")
    ax.grid(True, ls="--", alpha=0.3)

    # Panel 2: lateral grip utilization (margin 12cm)
    ax = axes[1]
    xy = np.array(S[0.12]["path_xy"])
    v = np.array(S[0.12]["v_long"])
    kpath = path_curvature(xy)
    util = (v**2 * kpath) / 9.81 / grip_g(v, p) * 100.0
    ax.plot(arc[sel], util[sel], color="tab:purple", lw=2.0)
    ax.axhline(100, color="red", ls="--", lw=0.8, label="grip limit")
    ax.axvspan(113, 124, color="gold", alpha=0.18)
    ax.fill_between(
        arc[sel],
        0,
        util[sel],
        where=util[sel] > 85,
        color="red",
        alpha=0.2,
        label="grip-limited (margin bites here)",
    )
    ax.set_ylabel("lateral grip used [%]")
    ax.set_xlabel("arc length s [m]")
    ax.set_title("Grip utilization: the gold bend uses only ~20–65% — it has speed to spare")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_ylim(0, 120)

    fig.suptitle("Why the fast corner ignores margin — Maisach (four-wheel, v_max=27)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "margin_corner_mechanism.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
