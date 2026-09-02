"""
Deep analysis of the Maisach margin sweep.

Reads the persisted per-margin four-wheel solves (from margin_corner_plot.py)
and quantifies WHERE boundary margin affects speed and where it does not.

Key question: why does one fast corner (s~115m) show no speed change with margin?
Hypothesis: margin only costs speed where the car is (a) below v_max AND
(b) in a corner gentle enough that extra width meaningfully increases the
racing-line radius.  Where the car is pinned at v_max, or in a tight corner
with little geometric leverage, margin does ~nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]

SOL_DIR = REPO / "data" / "experiments" / "margin_sweep_corner_v27"
REF_SOLUTION = REPO / "data" / "solutions" / "track_boundary_maisach_four_wheel_euler.json"
OUT_DIR = REPO / "ocp_plots" / "maisach_extras"

MARGINS = [0.12, 0.25, 0.37, 0.50]
MARGIN_COLORS = {0.12: "tab:blue", 0.25: "tab:green", 0.37: "tab:orange", 0.50: "tab:red"}

CORNER_S_LO = 90.0
CORNER_S_HI = 145.0


def load_all():
    sols = {}
    for m in MARGINS:
        p = SOL_DIR / f"margin_{int(m*100):03d}.json"
        with p.open() as f:
            sols[m] = json.load(f)
    with REF_SOLUTION.open() as f:
        ref = json.load(f)
    return sols, ref


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sols, ref = load_all()
    v_max = float(ref["model_params"]["v_max"])

    # Common arc grid (all solves share the same track mesh)
    arc = np.array(sols[MARGINS[0]]["arc_lengths"])
    kappa = np.abs(np.array(sols[MARGINS[0]]["kappa"]))

    V = np.vstack([np.array(sols[m]["v_long"]) for m in MARGINS])  # (n_margin, N)
    v_spread = V.max(axis=0) - V.min(axis=0)  # how much margin moves speed
    v_mean = V.mean(axis=0)

    # Fraction of track pinned at v_max (within 0.1 m/s)
    at_vmax = V[0] > (v_max - 0.1)
    print(f"v_max = {v_max:.1f} m/s")
    print(f"Fraction of track at v_max (tightest margin): {at_vmax.mean()*100:.1f}%")
    print(f"Mean speed spread across margins: {v_spread.mean():.3f} m/s")
    print(
        f"Max  speed spread across margins: {v_spread.max():.3f} m/s "
        f"at s={arc[v_spread.argmax()]:.1f} m"
    )

    # Corner of interest stats
    cm = (arc >= 108) & (arc <= 130)
    print("\nFast corner plateau s∈[108,130]:")
    print(f"  mean speed   = {v_mean[cm].mean():.2f} m/s")
    print(f"  mean spread  = {v_spread[cm].mean():.3f} m/s  (≈ margin-invariant)")
    print(f"  frac at vmax = {at_vmax[cm].mean()*100:.0f}%")

    # ---- Boundary-constraint activity ---------------------------------------
    # Is the racing line riding the edge (margin bites) or sitting interior
    # (margin is slack → no effect)?  slack = distance from line to nearest
    # *effective* boundary (w - margin).
    print("\nBoundary slack (min distance line→effective edge), by region:")
    regions = [
        ("flat corner s[108,130]", 108, 130),
        ("hi-sens corner s[295,310]", 295, 310),
        ("whole lap", 0, 1e9),
    ]
    for name, lo, hi in regions:
        rm = (arc >= lo) & (arc <= hi)
        line = []
        for mgn in MARGINS:
            d = np.array(sols[mgn]["d"])
            wl = np.array(sols[mgn]["w_left"]) - mgn
            wr = np.array(sols[mgn]["w_right"]) - mgn
            slack = np.minimum(wl - d, d + wr)  # ≥0; ~0 means against an edge
            line.append(slack[rm].min())
        print(
            f"  {name:28s}: min slack over margins = {min(line):.3f} m "
            f"(mean {np.mean(line):.2f} m)"
        )

    # =====================================================================
    # Figure: 3 panels
    # =====================================================================
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # --- Panel 1: full-track speed overlay ---
    ax = axes[0]
    for m in MARGINS:
        ax.plot(
            arc,
            sols[m]["v_long"],
            lw=1.1,
            color=MARGIN_COLORS[m],
            label=f"{int(m*100)} cm  ({sols[m]['profiling']['lap_time_s']:.2f}s)",
        )
    ax.axhline(v_max, color="gray", ls="--", lw=0.8, alpha=0.7, label=f"v_max={v_max:.0f}")
    ax.axvspan(108, 130, color="gold", alpha=0.15, label="margin-invariant corner")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Speed profile vs boundary margin — full lap")
    ax.legend(fontsize=8, ncol=3, loc="lower center")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Panel 2: speed spread (sensitivity to margin) ---
    ax = axes[1]
    ax.fill_between(arc, 0, v_spread, color="tab:purple", alpha=0.5)
    ax.plot(arc, v_spread, color="tab:purple", lw=1.0)
    ax.axvspan(108, 130, color="gold", alpha=0.15)
    ax.set_ylabel("speed spread\n(max−min over margins) [m/s]")
    ax.set_title("Where margin actually changes speed (high = sensitive)")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Panel 3: curvature + v_max mask ---
    ax = axes[2]
    ax.plot(arc, kappa, color="tab:gray", lw=0.9, label="|curvature| [1/m]")
    ax.fill_between(
        arc,
        0,
        kappa.max() * 1.05,
        where=at_vmax,
        color="tab:red",
        alpha=0.15,
        label="pinned at v_max",
    )
    ax.axvspan(108, 130, color="gold", alpha=0.15)
    ax.set_ylabel("|curvature| [1/m]")
    ax.set_xlabel("arc length s [m]")
    ax.set_title("Curvature & v_max-saturated regions")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle("Margin sensitivity analysis — Maisach (four-wheel)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "margin_sensitivity_analysis.png"
    fig.savefig(out, dpi=200)
    print(f"\nSaved: {out}")
    plt.close(fig)

    # =====================================================================
    # Scatter: spread vs speed, colored by curvature
    # =====================================================================
    fig2, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(v_mean, v_spread, c=kappa, cmap="viridis", s=12, alpha=0.7)
    ax.axvline(v_max, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.text(v_max - 0.2, ax.get_ylim()[1] * 0.9, "v_max", ha="right", color="gray")
    cb = fig2.colorbar(sc, ax=ax)
    cb.set_label("|curvature| [1/m]")
    ax.set_xlabel("mean speed at point [m/s]")
    ax.set_ylabel("speed spread over margins [m/s]")
    ax.set_title(
        "Margin sensitivity vs local speed\n(low at v_max cap AND at tight low-speed corners)"
    )
    ax.grid(True, ls="--", alpha=0.3)
    fig2.tight_layout()
    out2 = OUT_DIR / "margin_sensitivity_scatter.png"
    fig2.savefig(out2, dpi=200)
    print(f"Saved: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
