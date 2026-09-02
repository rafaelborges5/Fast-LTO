"""
Validate the trajectory renormalization by checking invariants and plotting comparisons.

Usage:
    python scripts/validate_renormalization.py <solution_json>

Example:
    python scripts/validate_renormalization.py data/solutions/fsg_random_point_mass_euler.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from fast_lto.export.trajectory import export_reference_trajectory


def load_original(solution_path: Path) -> dict:
    with solution_path.open("r") as f:
        data = json.load(f)

    path_xy = np.array(data["path_xy"], dtype=np.float64)
    return {
        "path_xy": path_xy,
        "w_left": np.array(data["w_left"], dtype=np.float64),
        "w_right": np.array(data["w_right"], dtype=np.float64),
        "kappa": np.array(data["kappa"], dtype=np.float64),
        "headings": np.array(data["headings"], dtype=np.float64),
        "psi_err": np.array(data["psi_err"], dtype=np.float64),
        "d": np.array(data["d"], dtype=np.float64),
        "arc_lengths": np.array(data["arc_lengths"], dtype=np.float64),
        "v": np.array(data.get("v", data.get("v_long")), dtype=np.float64),
        "model_name": data["run_config"]["model_name"],
        "mode": data["run_config"].get("mode", "trackdrive"),
    }


def _legacy_spline_kappa(path_xy: np.ndarray, periodic: bool, smooth: bool) -> np.ndarray:
    """Reimplementation of the old (pre-fix) spline-differentiation kappa.

    This logic used to live in ``export.trajectory._compute_path_geometry``
    and has been replaced there by an analytic formula. Kept here only as a
    one-time "before" baseline for the chatter-reduction comparison below.
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    x = path_xy[:, 0]
    y = path_xy[:, 1]

    diffs = np.diff(path_xy, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(x))
    t[1:] = np.cumsum(segment_lengths)

    if periodic:
        wrap_distance = np.linalg.norm(path_xy[0] - path_xy[-1])
        t_periodic = np.append(t, t[-1] + wrap_distance)
        x_periodic = np.append(x, x[0])
        y_periodic = np.append(y, y[0])
        spline_x = CubicSpline(t_periodic, x_periodic, bc_type="periodic")
        spline_y = CubicSpline(t_periodic, y_periodic, bc_type="periodic")
    else:
        spline_x = CubicSpline(t, x, bc_type="not-a-knot")
        spline_y = CubicSpline(t, y, bc_type="not-a-knot")

    dx_dt = spline_x(t, 1)
    dy_dt = spline_y(t, 1)
    d2x_dt2 = spline_x(t, 2)
    d2y_dt2 = spline_y(t, 2)

    numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
    denominator = (dx_dt**2 + dy_dt**2) ** 1.5
    curvatures = numerator / denominator

    if smooth and len(curvatures) > 17:
        curvatures = savgol_filter(
            curvatures,
            window_length=17,
            polyorder=2,
            mode="wrap" if periodic else "interp",
        )
    return curvatures


def _chatter_rms(kappa: np.ndarray) -> float:
    """RMS of the 2nd finite difference of kappa -- a simple chatter/jerk metric."""
    return float(np.sqrt(np.mean(np.diff(kappa, n=2) ** 2)))


def report_chatter_comparison(orig: dict, exported: dict) -> dict:
    print("\n=== Curvature chatter comparison ===\n")
    periodic = orig["mode"] == "trackdrive"

    legacy_raw = _legacy_spline_kappa(orig["path_xy"], periodic, smooth=False)
    legacy_savgol = _legacy_spline_kappa(orig["path_xy"], periodic, smooth=True)
    analytic = exported["kappa"]

    rms_raw = _chatter_rms(legacy_raw)
    rms_savgol = _chatter_rms(legacy_savgol)
    rms_analytic = _chatter_rms(analytic)

    print(f"  legacy (unfiltered spline-diff): chatter RMS = {rms_raw:.6f}")
    print(f"  legacy + savgol (current hotfix): chatter RMS = {rms_savgol:.6f}")
    print(f"  analytic (new fix):               chatter RMS = {rms_analytic:.6f}")
    if rms_analytic > 0:
        print(f"  improvement vs unfiltered: {rms_raw / rms_analytic:.1f}x")
        print(f"  improvement vs savgol:     {rms_savgol / rms_analytic:.1f}x")

    return {
        "legacy_raw": legacy_raw,
        "legacy_savgol": legacy_savgol,
    }


def load_exported_csv(csv_path: Path) -> dict:
    import csv as csv_mod

    with csv_path.open("r") as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    return {
        "x": col("x"),
        "y": col("y"),
        "boundary_left": col("boundary_left"),
        "boundary_right": col("boundary_right"),
        "kappa": col("kappa"),
        "yaw_angle": col("yaw_angle"),
        "yaw_angle_error": col("yaw_angle_error"),
        "lat_deviation": col("lat_deviation"),
        "arc_progress": col("arc_progress"),
        "time": col("time"),
    }


def check_invariants(orig: dict, exported: dict) -> bool:
    all_ok = True

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal all_ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_ok = False
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)

    print("\n=== Invariant Checks ===\n")

    # 1. lat_deviation is all zeros
    check(
        "lat_deviation == 0",
        np.allclose(exported["lat_deviation"], 0.0),
        f"max |lat_dev| = {np.max(np.abs(exported['lat_deviation'])):.2e}",
    )

    # 2. Total track width preserved
    total_orig = orig["w_left"] + orig["w_right"]
    total_new = exported["boundary_left"] + np.abs(exported["boundary_right"])
    check(
        "Total track width preserved",
        np.allclose(total_orig, total_new, atol=1e-10),
        f"max diff = {np.max(np.abs(total_orig - total_new)):.2e}",
    )

    # 3. Both boundaries non-negative (small tolerance for floating-point)
    check(
        "boundary_left >= 0",
        np.all(exported["boundary_left"] >= -1e-6),
        f"min = {np.min(exported['boundary_left']):.6f}",
    )
    check(
        "|boundary_right| >= 0 (always true)",
        np.all(np.abs(exported["boundary_right"]) >= -1e-6),
        f"min |br| = {np.min(np.abs(exported['boundary_right'])):.6f}",
    )

    # 4. Vehicle heading preserved
    vehicle_heading_orig = orig["headings"] + orig["psi_err"]
    vehicle_heading_new = exported["yaw_angle"] + exported["yaw_angle_error"]
    cos_diff = np.abs(np.cos(vehicle_heading_orig) - np.cos(vehicle_heading_new))
    sin_diff = np.abs(np.sin(vehicle_heading_orig) - np.sin(vehicle_heading_new))
    check(
        "Vehicle heading preserved",
        np.max(cos_diff) < 1e-6 and np.max(sin_diff) < 1e-6,
        f"max cos_diff = {np.max(cos_diff):.2e}, max sin_diff = {np.max(sin_diff):.2e}",
    )

    # 5. Curvature magnitude reasonable (exclude seam point at index 0)
    max_kappa_orig = np.max(np.abs(orig["kappa"]))
    max_kappa_new = np.max(np.abs(exported["kappa"][1:]))
    max_kappa_seam = np.abs(exported["kappa"][0])
    check(
        "Curvature magnitude reasonable (excl. seam)",
        max_kappa_new < max_kappa_orig * 3.0,
        f"centerline max |κ| = {max_kappa_orig:.4f}, optimal max |κ| = {max_kappa_new:.4f}, seam |κ| = {max_kappa_seam:.4f}",
    )

    # 6. Spline periodicity (heading smooth at wrap)
    heading_jump = np.abs(exported["yaw_angle"][-1] - exported["yaw_angle"][-2])
    typical_jump = np.median(np.abs(np.diff(exported["yaw_angle"])))
    check(
        "Heading smooth at wrap-around",
        heading_jump < 10 * typical_jump + 1e-6,
        f"last jump = {heading_jump:.4f}, median jump = {typical_jump:.4f}",
    )

    # 7. x, y positions unchanged
    check(
        "x positions unchanged",
        np.allclose(orig["path_xy"][:, 0], exported["x"]),
    )
    check(
        "y positions unchanged",
        np.allclose(orig["path_xy"][:, 1], exported["y"]),
    )

    # 8. Arc lengths monotonically increasing
    check(
        "Arc lengths monotonically increasing",
        np.all(np.diff(exported["arc_progress"]) > 0),
        f"min step = {np.min(np.diff(exported['arc_progress'])):.6f}",
    )

    # 9. Total length within ~5% of centerline length
    total_opt = exported["arc_progress"][-1]
    total_center = orig["arc_lengths"][-1]
    pct_diff = abs(total_opt - total_center) / total_center * 100
    check(
        "Total length within 10% of centerline",
        pct_diff < 10.0,
        f"centerline = {total_center:.2f}m, optimal = {total_opt:.2f}m ({pct_diff:.2f}%)",
    )

    # 10. Time monotonically increasing
    check(
        "Time monotonically increasing",
        np.all(np.diff(exported["time"]) > 0),
    )

    # 11. Closed-loop total turning == +-2*pi (periodic/trackdrive only).
    # This is an exact geometric invariant of a closed, non-self-crossing
    # path -- it depends only on kappa being *correct*, not merely smooth,
    # so it catches sign errors or dropped terms that a smoothness check
    # alone would miss.
    if orig["mode"] == "trackdrive":
        x_e, y_e = exported["x"], exported["y"]
        seg = np.linalg.norm(np.diff(np.column_stack([x_e, y_e]), axis=0), axis=1)
        wrap_len = np.linalg.norm([x_e[0] - x_e[-1], y_e[0] - y_e[-1]])
        ds_path = np.append(seg, wrap_len)
        total_turning = np.sum(exported["kappa"] * ds_path)
        check(
            "Closed-loop total turning == 2*pi",
            abs(abs(total_turning) - 2 * np.pi) < 0.05,
            f"total turning = {total_turning:.4f} rad (target +-{2*np.pi:.4f})",
        )
    else:
        print(f"  [SKIP] Closed-loop total turning -- mode={orig['mode']!r} is not periodic")

    return all_ok


def plot_comparisons(orig: dict, exported: dict, out_dir: Path, legacy: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    s_orig = orig["arc_lengths"]
    s_new = exported["arc_progress"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Curvature comparison: centerline vs legacy (spline-diff, +-savgol) vs new analytic
    ax = axes[0, 0]
    ax.plot(s_orig, orig["kappa"], label="centerline κ", alpha=0.5, color="gray")
    if legacy is not None:
        ax.plot(
            s_new,
            legacy["legacy_raw"],
            label="legacy spline-diff (unfiltered)",
            alpha=0.5,
            color="tab:red",
            lw=0.8,
        )
        ax.plot(
            s_new,
            legacy["legacy_savgol"],
            label="legacy + savgol (old hotfix)",
            alpha=0.8,
            color="tab:orange",
            lw=1.2,
        )
    ax.plot(s_new, exported["kappa"], label="analytic κ (new fix)", alpha=0.9, color="tab:blue")
    ax.set_xlabel("arc length [m]")
    ax.set_ylabel("curvature [1/m]")
    ax.set_title("Curvature: centerline vs optimal path")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Heading comparison
    ax = axes[0, 1]
    ax.plot(s_orig, orig["headings"], label="centerline heading", alpha=0.7)
    ax.plot(s_new, exported["yaw_angle"], label="optimal path heading", alpha=0.7)
    ax.set_xlabel("arc length [m]")
    ax.set_ylabel("heading [rad]")
    ax.set_title("Heading: centerline vs optimal path")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Boundaries (asymmetric)
    ax = axes[1, 0]
    ax.plot(s_orig, orig["w_left"], label="original w_left", alpha=0.5, linestyle="--")
    ax.plot(s_orig, orig["w_right"], label="original w_right", alpha=0.5, linestyle="--")
    ax.plot(s_new, exported["boundary_left"], label="renorm boundary_left", alpha=0.8)
    ax.plot(s_new, np.abs(exported["boundary_right"]), label="renorm |boundary_right|", alpha=0.8)
    ax.set_xlabel("arc length [m]")
    ax.set_ylabel("width [m]")
    ax.set_title("Track boundaries (now asymmetric)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Original lateral deviation
    ax = axes[1, 1]
    ax.plot(s_orig, orig["d"], label="original d", color="tab:purple")
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("arc length [m]")
    ax.set_ylabel("lateral deviation [m]")
    ax.set_title("Original lateral deviation (now zeroed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Renormalization validation — {orig['model_name']}", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / f"renormalization_{orig['model_name']}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/validate_renormalization.py <solution_json> [solution_json2 ...]"
        )
        sys.exit(1)

    for solution_arg in sys.argv[1:]:
        solution_path = Path(solution_arg).resolve()
        if not solution_path.exists():
            print(f"Solution not found: {solution_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Validating: {solution_path.name}")
        print(f"{'='*60}")

        orig = load_original(solution_path)

        csv_out = REPO_ROOT / "data" / "validation" / f"renorm_{solution_path.stem}.csv"
        export_reference_trajectory(solution_path, csv_out)
        exported = load_exported_csv(csv_out)

        ok = check_invariants(orig, exported)
        legacy = report_chatter_comparison(orig, exported)

        plot_comparisons(
            orig,
            exported,
            out_dir=REPO_ROOT / "data" / "validation",
            legacy=legacy,
        )

        if ok:
            print("\nAll checks PASSED.")
        else:
            print("\nSome checks FAILED.")

    print()


if __name__ == "__main__":
    main()
