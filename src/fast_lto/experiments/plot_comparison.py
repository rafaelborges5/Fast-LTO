"""Plot side-by-side comparison of vehicle models on Maisach — fair + full."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "data" / "experiments"
OUT_DIR = REPO / "ocp_plots" / "model_comparison"

MODELS_FAIR = {
    "point_mass":          {"file": "maisach_fair_point_mass",       "label": "Point Mass",       "color": "tab:green",  "ls": "-"},
    "dynamic_bicycle":     {"file": "maisach_fair_dynamic_bicycle",  "label": "Dyn. Bicycle",     "color": "tab:orange", "ls": "-"},
    "four_wheel_no_aero":  {"file": "maisach_fair_four_wheel_no_aero", "label": "4W (no aero)",   "color": "tab:blue",   "ls": "-"},
    "four_wheel_full":     {"file": "maisach_full_four_wheel",       "label": "4W (full aero)",   "color": "tab:red",    "ls": "--"},
}


def load_sol(file_stem, ds):
    with (EXP_DIR / f"{file_stem}_ds{ds}.json").open() as f:
        return json.load(f)


def _get_v(d):
    return np.array(d.get("v", d.get("v_long")), dtype=np.float64)


def main(ds: float = 1.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {k: load_sol(v["file"], ds) for k, v in MODELS_FAIR.items()}

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # --- 1. Track overlay ---
    ax = axes[0, 0]
    for k, cfg in MODELS_FAIR.items():
        xy = np.array(data[k]["path_xy"])
        ax.plot(xy[:, 0], xy[:, 1], color=cfg["color"], ls=cfg["ls"], lw=1.0, label=cfg["label"])
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("Optimal paths")
    ax.grid(True, ls="--", alpha=0.3)

    # --- 2. Speed profiles ---
    ax = axes[0, 1]
    for k, cfg in MODELS_FAIR.items():
        s = np.array(data[k]["arc_lengths"])
        v = _get_v(data[k])
        ax.plot(s, v, color=cfg["color"], ls=cfg["ls"], lw=0.9, label=cfg["label"])
    ax.set_xlabel("s [m]")
    ax.set_ylabel("v [m/s]")
    ax.legend(fontsize=8)
    ax.set_title("Speed profiles")
    ax.grid(True, ls="--", alpha=0.3)

    # --- 3. Lateral offsets ---
    ax = axes[1, 0]
    s0 = np.array(data["four_wheel_full"]["arc_lengths"])
    ax.fill_between(s0,
                     np.array(data["four_wheel_full"]["w_left"]),
                     -np.array(data["four_wheel_full"]["w_right"]),
                     alpha=0.08, color="gray", label="bounds")
    for k, cfg in MODELS_FAIR.items():
        s = np.array(data[k]["arc_lengths"])
        ax.plot(s, np.array(data[k]["d"]), color=cfg["color"], ls=cfg["ls"], lw=0.9, label=cfg["label"])
    ax.set_xlabel("s [m]")
    ax.set_ylabel("d [m]")
    ax.legend(fontsize=8)
    ax.set_title("Lateral offset")
    ax.grid(True, ls="--", alpha=0.3)

    # --- 4. GG diagrams ---
    ax = axes[1, 1]
    for k, cfg in MODELS_FAIR.items():
        d = data[k]
        params = d.get("model_params", {})
        m = float(params.get("m", 160.0))
        v_arr = _get_v(d)

        if "a_lat" in d:
            a_long = np.array(d["a_long"])
            a_lat = np.array(d["a_lat"])
        elif "a_long" in d:
            a_long = np.array(d["a_long"])
            a_lat = v_arr * np.array(d["yaw_rate"])
        else:
            yr = np.array(d["yaw_rate"])
            vl = np.array(d["v_lat"])
            Fx_fl = np.array(d["Fx_fl"]); Fx_fr = np.array(d["Fx_fr"])
            Fx_rr = np.array(d["Fx_rr"]); Fx_rl = np.array(d["Fx_rl"])
            delta = np.array(d["delta"])
            cd = np.cos(delta)
            C_d_v = float(params.get("C_d", 0)); C_r_v = float(params.get("C_r", 0))
            rho = float(params.get("rho", 1.225)); A_f = float(params.get("A_f", 1.2))
            F_drag = 0.5 * rho * C_d_v * A_f * v_arr**2
            F_roll = m * float(params.get("g", 9.81)) * C_r_v
            Fx_tot = (Fx_fl+Fx_fr)*cd + Fx_rr+Fx_rl - F_roll - F_drag
            a_long = Fx_tot / m + yr * vl
            a_lat = v_arr * yr

        ax.scatter(a_lat, a_long, s=3, alpha=0.4, color=cfg["color"], label=cfg["label"])

    D_g = 1.2 * 9.81
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(D_g*np.cos(th), D_g*np.sin(th), "r--", lw=0.8, alpha=0.5, label="D*g")
    ax.set_xlabel("a_lat [m/s^2]")
    ax.set_ylabel("a_long [m/s^2]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title("GG diagrams")
    ax.grid(True, ls="--", alpha=0.3)

    # --- 5. Bar chart ---
    ax = axes[2, 0]
    names = list(MODELS_FAIR.keys())
    labels = [MODELS_FAIR[n]["label"] for n in names]
    colors_bar = [MODELS_FAIR[n]["color"] for n in names]
    lap_times = [data[n]["profiling"]["lap_time_s"] for n in names]
    solve_times = [data[n]["profiling"]["solve_time_s"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, lap_times, w, color=colors_bar, alpha=0.85, label="Lap time")
    b2 = ax.bar(x + w/2, solve_times, w, color=colors_bar, alpha=0.35,
                edgecolor=colors_bar, linewidth=1.5, label="Solve time")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Time [s]")
    ax.legend(fontsize=9)
    ax.set_title("Lap time vs solve time")
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    for bar, val in zip(b1, lap_times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(b2, solve_times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    # --- 6. Summary ---
    ax = axes[2, 1]
    ax.axis("off")

    pm  = data["point_mass"]["profiling"]
    db  = data["dynamic_bicycle"]["profiling"]
    fwn = data["four_wheel_no_aero"]["profiling"]
    fwf = data["four_wheel_full"]["profiling"]

    lines = [
        f"Track: Maisach (512.6m), ds={ds}m, N={pm['N']}",
        f"Matched: m=160kg, D=1.2, v_max=20 m/s",
        f"4 corner constraints + 15cm margin",
        f"",
        f"{'Model':<18} {'Lap':>7} {'Solve':>7} {'Iters':>6} {'nx':>4} {'nu':>4}",
        f"{'-'*48}",
        f"{'Point Mass':<18} {pm['lap_time_s']:>6.2f}s {pm['solve_time_s']:>6.1f}s {pm['iter_count']:>5}    4    2",
        f"{'Dyn. Bicycle':<18} {db['lap_time_s']:>6.2f}s {db['solve_time_s']:>6.1f}s {db['iter_count']:>5}    6    2",
        f"{'4W (no aero)':<18} {fwn['lap_time_s']:>6.2f}s {fwn['solve_time_s']:>6.1f}s {fwn['iter_count']:>5}   11    5",
        f"{'4W (full aero)':<18} {fwf['lap_time_s']:>6.2f}s {fwf['solve_time_s']:>6.1f}s {fwf['iter_count']:>5}   11    5",
        f"",
        f"FAIR comparison (all at D*g = {1.2*9.81:.1f} m/s^2):",
        f"  4W-noaero vs PM:      {pm['lap_time_s']-fwn['lap_time_s']:+.2f}s",
        f"  4W-noaero vs Bicycle: {db['lap_time_s']-fwn['lap_time_s']:+.2f}s",
        f"",
        f"FULL aero advantage:",
        f"  4W-full vs 4W-noaero: {fwn['lap_time_s']-fwf['lap_time_s']:+.2f}s",
        f"  4W-full vs PM:        {pm['lap_time_s']-fwf['lap_time_s']:+.2f}s",
        f"",
        f"Note: PM and Bicycle have no aero drag",
        f"in dynamics -> optimistic at high speed.",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            fontsize=9.5, va="top", family="monospace")

    fig.suptitle(f"Vehicle Model Comparison — Maisach Track (ds={ds}m)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / f"comparison_ds{ds}.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    ds = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    main(ds)
