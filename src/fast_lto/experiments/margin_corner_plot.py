"""
Plot trajectories at different boundary margins zoomed into a specific corner.

Runs the four-wheel OCP for several margin values on the Maisach track and
overlays the resulting trajectories on the corner at arc_length ~115m.

Usage:
    cd Fast-LTO && python src/experiments/margin_corner_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

from fast_lto.optimization.global_ocp import solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator
from fast_lto.splines.spline_fitter import fit_and_discretize
from fast_lto.utils.track_bounds import compute_lateral_bounds, load_boundaries
from fast_lto.vehicle_models import FourWheelModel

TRACK_CSV = REPO / "data" / "tracks" / "track_boundary_maisach.csv"
# Reference solution that generated the margin-sweep panel.  We pull the exact
# vehicle params + reg_du from here so the reproduction matches (the working-tree
# trackdrive.yaml has since been changed to a slower, more conservative car:
# v_max=12, reg_du=600, lower grip).
REF_SOLUTION = REPO / "data" / "solutions" / "track_boundary_maisach_four_wheel_euler.json"
OUT_DIR = REPO / "ocp_plots" / "maisach_extras"
# Persisted per-margin solves so analysis can re-read without re-solving.
SOL_DIR = REPO / "data" / "experiments" / "margin_sweep_corner_v27_clean"

# Locked margin-sweep config (per Rafael): ds=1.0, v_max=27, D=1.2, C_l=5.54,
# reg_u_l2=4.0 (→ reg term ~3.5%).  We start from the reference model_params
# (which already carry D=1.2, C_l=5.54, the measured corners, m=160) and only
# override v_max; regularisation is reg_u_l2, not reg_du.
V_MAX_OVERRIDE = 27.0
# reg_u_l2=4.0 over-regularizes THIS setup (degenerate ~10 m/s car); the value
# doesn't transfer from the FSG calibration.  For the physics diagnosis we run
# with regularization off to isolate where speed is grip- vs cap-limited.
REG_U_L2 = None

CORNER_S_LO = 90.0
CORNER_S_HI = 145.0

MARGINS = [0.12, 0.25, 0.37, 0.50]
MARGIN_COLORS = {0.12: "tab:blue", 0.25: "tab:green", 0.37: "tab:orange", 0.50: "tab:red"}

DS = 1.0  # coarser mesh → ~50s per solve (ref was 0.4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fitting track spline at ds=1.0m …")
    disc_track = fit_and_discretize(TRACK_CSV, ds_m=DS, continuity="C2")

    print("Computing lateral bounds …")
    bd = load_boundaries(TRACK_CSV)
    bounds_result = compute_lateral_bounds(disc_track, bd["left"], bd["right"])

    positions = np.array(disc_track.positions)
    headings = np.array(disc_track.headings)
    w_left = np.array(bounds_result.w_left)
    w_right = np.array(bounds_result.w_right)

    track = {
        "positions": positions.tolist(),
        "headings": headings.tolist(),
        "curvatures": disc_track.curvatures.tolist(),
        "curvatures_half": disc_track.curvatures_half.tolist(),
        "arc_lengths": disc_track.arc_lengths.tolist(),
        "w_left": w_left.tolist(),
        "w_right": w_right.tolist(),
        "ds_m": DS,
    }

    print("Loading vehicle model from reference solution …")
    with REF_SOLUTION.open() as f:
        ref = json.load(f)
    params = dict(ref["model_params"])
    params["v_max"] = V_MAX_OVERRIDE  # locked sweep used 27, ref solution had 20
    reg_du = float(ref["run_config"].get("reg_du", 0.0))
    print(
        f"  v_max={params['v_max']}  m={params['m']}  D={params['D_fl']}  "
        f"C_l={params['C_l']}  reg_du={reg_du}  reg_u_l2={REG_U_L2}"
    )
    model = FourWheelModel(params=params)

    integrator = EulerIntegrator()

    results = {}
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    for margin in MARGINS:
        sol_path = SOL_DIR / f"margin_{int(margin*100):03d}.json"
        if sol_path.exists():
            print(f"\nReusing cached solve: margin={margin:.2f}m ({sol_path.name})")
            with sol_path.open() as f:
                sol = json.load(f)
        else:
            print(f"\nSolving OCP: margin={margin:.2f}m …")
            sol = solve_ocp_and_save(
                track=track,
                model=model,
                solution_path=sol_path,
                integrator=integrator,
                initial_speed=5.0,
                reg_du=reg_du,
                reg_u_l2=REG_U_L2,
                use_normalization=True,
                solver_verbose=False,
                boundary_margin=margin,
                mode="trackdrive",
            )
        lap = sol["profiling"]["lap_time_s"]
        v = np.array(sol["v_long"])
        print(f"  lap={lap:.3f}s  v_min={v.min():.2f}m/s")
        results[margin] = {
            "path_xy": np.array(sol["path_xy"]),
            "v_long": v,
            "arc_lengths": np.array(sol["arc_lengths"]),
            "lap_time": lap,
        }

    # ---- Compute boundary XY lines -------------------------------------------
    normals = np.column_stack((-np.sin(headings), np.cos(headings)))
    left_bnd_xy = positions + w_left[:, None] * normals
    right_bnd_xy = positions - w_right[:, None] * normals

    # Find apex XY (lowest speed in corner) from the tightest-margin solution
    arc_ref = results[MARGINS[0]]["arc_lengths"]
    mask = (arc_ref >= CORNER_S_LO) & (arc_ref <= CORNER_S_HI)
    v_corner = results[MARGINS[0]]["v_long"][mask]
    xy_corner = results[MARGINS[0]]["path_xy"][mask]
    apex_xy = xy_corner[np.argmin(v_corner)]
    cx, cy = apex_xy

    zoom = 20.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: XY corner view ---
    ax = axes[0]
    ax.plot(left_bnd_xy[:, 0], left_bnd_xy[:, 1], "k-", lw=1.8, label="track edge", zorder=2)
    ax.plot(right_bnd_xy[:, 0], right_bnd_xy[:, 1], "k-", lw=1.8, zorder=2)

    # Effective (shrunk) boundaries for extreme margins
    for margin, ls in [(MARGINS[0], "--"), (MARGINS[-1], ":")]:
        leff = positions + (w_left - margin)[:, None] * normals
        reff = positions - (w_right - margin)[:, None] * normals
        col = MARGIN_COLORS[margin]
        ax.plot(leff[:, 0], leff[:, 1], ls=ls, lw=0.8, color=col, alpha=0.5, zorder=2)
        ax.plot(reff[:, 0], reff[:, 1], ls=ls, lw=0.8, color=col, alpha=0.5, zorder=2)

    for margin in MARGINS:
        xy = results[margin]["path_xy"]
        label = f"{int(margin*100)} cm  ({results[margin]['lap_time']:.2f}s)"
        ax.plot(xy[:, 0], xy[:, 1], lw=2.2, color=MARGIN_COLORS[margin], label=label, zorder=3)

    ax.set_xlim(cx - zoom, cx + zoom)
    ax.set_ylim(cy - zoom, cy + zoom)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, title="margin", loc="best")
    ax.set_title(f"Trajectory zoom — corner ~s={CORNER_S_LO:.0f}–{CORNER_S_HI:.0f}m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, ls="--", alpha=0.3)

    # --- Right: speed vs arc_length ---
    ax2 = axes[1]
    for margin in MARGINS:
        arc = results[margin]["arc_lengths"]
        v = results[margin]["v_long"]
        mask = (arc >= CORNER_S_LO) & (arc <= CORNER_S_HI)
        ax2.plot(
            arc[mask],
            v[mask],
            lw=2.0,
            color=MARGIN_COLORS[margin],
            label=f"{int(margin*100)} cm  ({results[margin]['lap_time']:.2f}s)",
        )

    ax2.set_xlabel("arc length s [m]")
    ax2.set_ylabel("speed [m/s]")
    ax2.set_title(f"Speed profile — s ∈ [{CORNER_S_LO:.0f}, {CORNER_S_HI:.0f}] m")
    ax2.legend(fontsize=9, title="margin")
    ax2.grid(True, ls="--", alpha=0.3)

    fig.suptitle(
        "Margin sweep — Maisach corner (four-wheel, v_max=27, reg_u_l2=4.0, ds=1.0m)", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / "margin_corner_zoom.png"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
