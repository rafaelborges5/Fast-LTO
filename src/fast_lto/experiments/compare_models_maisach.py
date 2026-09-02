"""
Compare three vehicle models on the Maisach track with matched grip limits.

Two scenarios:
  A) "Fair" — four-wheel with NO aero (C_l=C_d=C_r=0, static loads).
     All three models have the same peak grip: D*g = 11.77 m/s^2.
     Isolates the pure modeling advantage (per-wheel allocation, yaw dynamics).

  B) "Full" — four-wheel with full aero and quasi-static load transfer.
     Shows the real-world advantage from downforce and load transfer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fast_lto.optimization.global_ocp import load_track_with_widths, solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator
from fast_lto.vehicle_models import DynamicBicycleModel, FourWheelModel, PointMassModel

REPO = Path(__file__).resolve().parents[3]
TRACK_PATH = REPO / "data" / "discretized" / "track_boundary_maisach_with_widths.json"
SOLUTIONS_DIR = REPO / "data" / "experiments"

COMMON_CORNERS = [
    ("FL", 0.95, 0.75),
    ("FR", 0.95, -0.75),
    ("RL", -0.658, 0.75),
    ("RR", -0.658, -0.75),
]

BOUNDARY_MARGIN = 0.15


def make_four_wheel_no_aero():
    return FourWheelModel(
        params={
            "C_l": 0.0,
            "C_d": 0.0,
            "C_r": 0.0,
            "load_transfer_mode": "static",
        }
    )


def make_four_wheel_full():
    return FourWheelModel()


def make_dynamic_bicycle():
    return DynamicBicycleModel(
        params={
            "m": 160.0,
            "Iz": 250.0,
            "lf": 0.872,
            "lr": 0.658,
            "g": 9.81,
            "Bf": 9.0,
            "Cf": 1.3,
            "Dmf_f": 1.2,
            "Br": 9.0,
            "Cr": 1.3,
            "Dmf_r": 1.2,
            "mu": 1.2,
            "gamma_ellipse": 1.0,
            "use_friction_ellipse": True,
            "v_min": 0.4,
            "v_max": 20.0,
            "v_lat_max": 4.0,
            "yaw_rate_max": 3.0,
            "a_long_min": -15.0,
            "a_long_max": 15.0,
            "delta_max": 0.4,
            "d_max": 3.0,
            "psi_err_max": 1.2,
            "eps_s_dot": 1.0,
            "eps_D_kappa": 0.05,
            "v_eps": 0.5,
            "smoothmax_eps": 1e-3,
            "corners": COMMON_CORNERS,
        }
    )


def make_point_mass():
    D_g = 1.2 * 9.81
    return PointMassModel(
        params={
            "m": 160.0,
            "g": 9.81,
            "lf": 0.872,
            "lr": 0.658,
            "mu": 1.2,
            "v_min": 0.4,
            "v_max": 20.0,
            "a_long_min": -D_g,
            "a_long_max": D_g,
            "a_lat_min": -D_g,
            "a_lat_max": D_g,
            "d_max": 3.0,
            "psi_err_max": 1.2,
            "eps_s_dot": 1.0,
            "v_eps": 0.5,
            "smoothmax_eps": 1e-3,
            "corners": COMMON_CORNERS,
        }
    )


def _solve_one(name, model, track_data, sol_path, reg_du, reg_u_l2):
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        sol = solve_ocp_and_save(
            track=track_data,
            model=model,
            solution_path=sol_path,
            integrator=EulerIntegrator(),
            initial_speed=5.0,
            reg_du=reg_du,
            reg_u_l2=reg_u_l2,
            use_normalization=True,
            solver_verbose=False,
            boundary_margin=BOUNDARY_MARGIN,
        )
        wall = time.perf_counter() - t0
        prof = sol["profiling"]
        return {
            "lap_time": prof["lap_time_s"],
            "solve_time": prof["solve_time_s"],
            "wall_time": wall,
            "iters": prof["iter_count"],
            "status": prof["return_status"],
            "N": prof["N"],
        }
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"  FAILED: {e}")
        return {"status": "FAILED", "error": str(e), "wall_time": wall}


def run_experiment(ds_m: float = 1.0):
    from fast_lto.splines.spline_fitter import fit_and_discretize
    from fast_lto.utils.track_bounds import (
        compute_lateral_bounds,
        load_boundaries,
        save_track_with_widths,
    )

    csv_path = REPO / "data" / "tracks" / "track_boundary_maisach.csv"
    disc_path = REPO / "data" / "discretized" / "track_boundary_maisach.json"
    widths_path = REPO / "data" / "discretized" / "track_boundary_maisach_with_widths.json"

    print(f"Fitting spline at ds={ds_m}m...")
    track_obj = fit_and_discretize(
        csv_path, ds_m=ds_m, continuity="C2", viz=False, save_path=disc_path
    )
    boundaries = load_boundaries(csv_path)
    result = compute_lateral_bounds(track_obj, left=boundaries["left"], right=boundaries["right"])
    save_track_with_widths(
        widths_path, track=track_obj, csv_source=csv_path, result=result, bounds_config={}
    )
    N = track_obj.num_points
    print(f"  N={N}, ds={track_obj.ds_m:.3f}m, total={track_obj.total_length_m:.1f}m")

    track_data = load_track_with_widths(widths_path)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Scenario A: Fair (no aero) ----
    print(f"\n{'#'*60}")
    print("  SCENARIO A: FAIR COMPARISON (no aero, static loads)")
    print(f"  All models have peak grip = D*g = {1.2*9.81:.2f} m/s^2")
    print(f"{'#'*60}")

    fair = {}
    tag = f"ds{ds_m}"
    fair["point_mass"] = _solve_one(
        "point_mass",
        make_point_mass(),
        track_data,
        SOLUTIONS_DIR / f"maisach_fair_point_mass_{tag}.json",
        600.0,
        None,
    )
    fair["dynamic_bicycle"] = _solve_one(
        "dynamic_bicycle",
        make_dynamic_bicycle(),
        track_data,
        SOLUTIONS_DIR / f"maisach_fair_dynamic_bicycle_{tag}.json",
        600.0,
        None,
    )
    fair["four_wheel_no_aero"] = _solve_one(
        "four_wheel (no aero)",
        make_four_wheel_no_aero(),
        track_data,
        SOLUTIONS_DIR / f"maisach_fair_four_wheel_no_aero_{tag}.json",
        0.0,
        2.0,
    )

    # ---- Scenario B: Full (with aero) ----
    print(f"\n{'#'*60}")
    print("  SCENARIO B: FULL FOUR-WHEEL (with aero + load transfer)")
    print(f"{'#'*60}")

    full = {}
    full["four_wheel_full"] = _solve_one(
        "four_wheel (full aero)",
        make_four_wheel_full(),
        track_data,
        SOLUTIONS_DIR / f"maisach_full_four_wheel_{tag}.json",
        0.0,
        2.0,
    )

    # ---- Summary ----
    all_results = {**fair, **full}

    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (ds={ds_m}m, N={N}, Maisach track)")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'Lap (s)':>8} {'Solve (s)':>10} {'Iters':>6}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        if r.get("status") == "FAILED":
            print(f"{name:<28} {'FAILED':>8} {r['wall_time']:>10.1f}")
        else:
            print(f"{name:<28} {r['lap_time']:>8.2f} {r['solve_time']:>10.1f} {r['iters']:>6}")

    pm_t = fair["point_mass"].get("lap_time", 0)
    db_t = fair["dynamic_bicycle"].get("lap_time", 0)
    fw_na_t = fair["four_wheel_no_aero"].get("lap_time", 0)
    fw_full_t = full["four_wheel_full"].get("lap_time", 0)
    if pm_t and fw_na_t:
        print("\nFair comparison (no aero):")
        print(f"  4W vs PM:      {pm_t - fw_na_t:+.2f}s ({(pm_t - fw_na_t)/pm_t*100:+.1f}%)")
        print(f"  4W vs Bicycle:  {db_t - fw_na_t:+.2f}s ({(db_t - fw_na_t)/db_t*100:+.1f}%)")
    if fw_full_t and pm_t:
        print("\nFull aero advantage:")
        print(f"  4W-full vs PM: {pm_t - fw_full_t:+.2f}s ({(pm_t - fw_full_t)/pm_t*100:+.1f}%)")
        print(f"  4W-full vs 4W-no-aero: {fw_na_t - fw_full_t:+.2f}s (pure aero gain)")

    return all_results


if __name__ == "__main__":
    ds = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_experiment(ds_m=ds)
