from __future__ import annotations

"""
Dense ds-scaling study for the full-lap OCP.

Preferred usage (from repo root, after `pip install -e .`):
    python -m fast_lto.experiments.ds_scaling \
        --track-id fsg_random \
        --ds-min 0.1 \
        --ds-max 5.0 \
        --num-ds 20

For ad-hoc use without installing, you can also run:
    python src/experiments/ds_scaling.py
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from fast_lto.pipeline import PipelineConfig, run_pipeline


def _get_repo_root() -> Path:
    """Infer repository root from this file location."""
    return Path(__file__).resolve().parents[3]


def _scaled_reg_u_l2(
    ds: float,
    reg_u_l2_ref: float | None,
    reg_ds_ref: float,
    reg_scale_mode: str,
) -> float | None:
    """
    Return the L2 input-regularisation weight to use at a given ds.

    With ``linear_ds`` the weight is scaled proportionally to ds.  Because the
    L2 penalty ``reg_u_l2 * sum_i u_i^2`` sums over all N grid points while the
    lap-time term stays ~constant, the *relative* regularisation contribution
    grows as 1/ds for a fixed weight.  Scaling the weight ∝ ds keeps that
    relative contribution roughly constant across the whole ds sweep (so it can
    be held under the 2-3% budget at every resolution).

    ``reg_ds_ref`` is the ds at which ``reg_u_l2_ref`` was calibrated.
    """
    if reg_u_l2_ref is None:
        return None
    if reg_scale_mode == "fixed":
        return float(reg_u_l2_ref)
    if reg_scale_mode == "linear_ds":
        return float(reg_u_l2_ref) * (float(ds) / float(reg_ds_ref))
    raise ValueError(f"Unknown reg_scale_mode: {reg_scale_mode!r}")


def _ds_key(ds: float) -> str:
    """Stable string key for a ds value (for resume matching)."""
    return f"{float(ds):.4f}"


def _write_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """(Re)write the full results CSV with a stable, sorted header."""
    if not results:
        return
    fieldnames: List[str] = sorted({k for row in results for k in row.keys()})
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    tmp_path.replace(csv_path)  # atomic, so an interrupt can't truncate the CSV


def run_ds_scaling_experiment(
    track_id: str = "fsg_random",
    track_type: str = "fsg",
    model_name: str = "point_mass",
    integrator_name: str = "euler",
    continuity: str = "C4",
    reg_u: float = 600.0,
    initial_speed: float = 5.0,
    ds_min: float = 0.1,
    ds_max: float = 5.0,
    num_ds: int = 20,
    ds_list: List[float] | None = None,
    make_plots: bool = True,
    vehicle_config: Any = None,
    reg_u_l2_ref: float | None = None,
    reg_ds_ref: float = 1.0,
    reg_scale_mode: str = "linear_ds",
    reg_relative_max: float = 0.03,
    boundary_margin: float = 0.0,
    smooth_centerline: int = 0,
    mode: str = "trackdrive",
    resume: bool = True,
) -> Dict[str, Path]:
    """
    Sweep ds in [ds_min, ds_max] and record OCP scaling metrics.

    Returns
    -------
    dict
        Paths to the CSV and any generated plots.
    """
    repo_root = _get_repo_root()
    out_dir = repo_root / "data" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ds_list is not None:
        ds_values = np.array(sorted(float(d) for d in ds_list), dtype=float)
    else:
        ds_values = np.geomspace(ds_min, ds_max, num=num_ds)

    # CSV path is deterministic per config, so a re-run resumes the same file.
    csv_path = out_dir / f"ds_scaling_{track_id}_{model_name}_{integrator_name}_{continuity}.csv"

    # Resume: load already-completed ds rows (any row present = a finished attempt;
    # deterministic failures such as the RK4 mesh limit are not retried).
    done_by_ds: Dict[str, Dict[str, Any]] = {}
    if resume and csv_path.exists():
        try:
            for row in _load_results_from_csv(csv_path):
                ds_v = row.get("ds_m")
                if ds_v not in (None, "") and not (isinstance(ds_v, float) and np.isnan(ds_v)):
                    done_by_ds[_ds_key(float(ds_v))] = row
        except Exception as exc:  # noqa: BLE001
            print(f"  Could not read existing CSV for resume ({exc}); starting fresh.")
            done_by_ds = {}

    results: List[Dict[str, Any]] = []

    print("Running ds-scaling experiment")
    print(f"  Track ID: {track_id} (type={track_type}, mode={mode})")
    print(f"  Model: {model_name}, integrator: {integrator_name}, continuity: {continuity}")
    if ds_list is not None:
        print(f"  ds values: {[round(float(d), 3) for d in ds_values]} m (explicit)")
    else:
        print(f"  ds range: [{ds_min:.3f}, {ds_max:.3f}] m with {num_ds} points (log-spaced)")
    print(
        f"  reg_u (du): {reg_u}, reg_u_l2_ref: {reg_u_l2_ref} @ ds_ref={reg_ds_ref} "
        f"(mode={reg_scale_mode}), reg budget: {reg_relative_max:.1%}"
    )
    print(f"  boundary_margin: {boundary_margin}, smooth_centerline: {smooth_centerline}")
    if done_by_ds:
        print(
            f"  Resume: {len(done_by_ds)} ds value(s) already in {csv_path.name}; "
            f"these will be skipped."
        )
    print()

    for ds in ds_values:
        ds_float = float(ds)
        print("=" * 60)
        print(f"ds = {ds_float:.4f} m")

        cached = done_by_ds.get(_ds_key(ds_float))
        if cached is not None:
            print(f"  Already done (status={cached.get('return_status')}); skipping.")
            results.append(cached)
            continue

        metrics: Dict[str, Any] = {
            "track_id": track_id,
            "model_name": model_name,
            "integrator_name": integrator_name,
            "ds_m": ds_float,
        }

        try:
            # Scale L2 input regularisation with ds so its relative contribution
            # to the objective stays within budget at every resolution, then
            # halve-and-retry if a solve still exceeds it.
            reg_u_l2_ds = _scaled_reg_u_l2(ds_float, reg_u_l2_ref, reg_ds_ref, reg_scale_mode)

            max_reg_retries = 3
            profiling: Dict[str, Any] = {}
            for attempt in range(max_reg_retries + 1):
                config = PipelineConfig(
                    track_id=track_id,
                    track_type=track_type,
                    generate_track=False,
                    repo_root=repo_root,
                    ds_m=ds_float,
                    continuity=continuity,
                    smooth_centerline=smooth_centerline,
                    mode=mode,
                    model_name=model_name,
                    integrator_name=integrator_name,
                    reg_u=reg_u,
                    reg_u_l2=reg_u_l2_ds,
                    initial_speed=initial_speed,
                    boundary_margin=boundary_margin,
                    plot_results=False,
                    show_plots=False,
                )
                config.vehicle_config = vehicle_config

                # Force recomputation of spline + bounds for each ds, reuse CSV.
                pipeline_results = run_pipeline(
                    config=config,
                    start_from="spline",
                    end_at="ocp",
                )

                solution_path = pipeline_results["ocp"]
                with solution_path.open("r") as f:
                    sol_data = json.load(f)
                profiling = sol_data.get("profiling", {}) or {}

                rr = profiling.get("reg_term_relative")
                status_ok = profiling.get("return_status") == "Solve_Succeeded"
                if (
                    reg_u_l2_ds is None
                    or rr is None
                    or not status_ok
                    or rr <= reg_relative_max
                    or attempt == max_reg_retries
                ):
                    if rr is not None and rr > reg_relative_max and status_ok:
                        print(
                            f"  WARNING: reg fraction {rr:.2%} still over budget "
                            f"after {attempt} retries (reg_u_l2={reg_u_l2_ds:.5g})."
                        )
                    break

                old = reg_u_l2_ds
                reg_u_l2_ds = reg_u_l2_ds * 0.5
                print(
                    f"  reg fraction {rr:.2%} > budget {reg_relative_max:.1%}; "
                    f"halving reg_u_l2 {old:.5g} -> {reg_u_l2_ds:.5g} and re-solving."
                )

            N = profiling.get("N")
            solve_time_s = profiling.get("solve_time_s")
            iter_count = profiling.get("iter_count")
            time_per_point_ms = profiling.get("time_per_point_ms")
            time_per_iter_ms = profiling.get("time_per_iter_ms")
            lap_time_s = profiling.get("lap_time_s")
            reg_term = profiling.get("reg_term")
            reg_term_relative = profiling.get("reg_term_relative")
            return_status = profiling.get("return_status")

            metrics.update(
                {
                    "N": N,
                    "solve_time_s": solve_time_s,
                    "iter_count": iter_count,
                    "time_per_point_ms": time_per_point_ms,
                    "time_per_iter_ms": time_per_iter_ms,
                    "lap_time_s": lap_time_s,
                    "reg_term": reg_term,
                    "reg_term_relative": reg_term_relative,
                    "reg_u_l2_applied": reg_u_l2_ds,
                    "return_status": return_status,
                    "error": "",
                }
            )

            if N not in (None, 0) and iter_count not in (None, 0):
                metrics["iters_per_point"] = iter_count / N
            else:
                metrics["iters_per_point"] = None

            if lap_time_s not in (None, 0.0) and solve_time_s is not None:
                metrics["time_per_lap_solve_ratio"] = solve_time_s / lap_time_s
            else:
                metrics["time_per_lap_solve_ratio"] = None

            if time_per_point_ms is not None and iter_count not in (None, 0) and N not in (None, 0):
                metrics["time_per_iter_per_point_ms"] = time_per_point_ms / iter_count
            else:
                metrics["time_per_iter_per_point_ms"] = None

            print(
                f"Solved: N={N}, time={solve_time_s:.3f} s, "
                f"iters={iter_count}, lap_time={lap_time_s:.3f} s, "
                f"status={return_status}"
            )

        except Exception as exc:  # noqa: BLE001
            print(f"FAILED for ds={ds_float:.4f}: {exc}")
            metrics["error"] = str(exc)

        results.append(metrics)
        # Checkpoint after every ds so an interrupt loses at most the in-flight point.
        _write_csv(results, csv_path)

    # Preserve any previously-computed ds rows that aren't on the current grid,
    # so changing the grid never discards already-solved (expensive) points.
    iterated_keys = {_ds_key(float(r["ds_m"])) for r in results if r.get("ds_m") is not None}
    leftovers = [row for k, row in done_by_ds.items() if k not in iterated_keys]
    if leftovers:
        print(f"  Keeping {len(leftovers)} previously-solved off-grid ds row(s).")
        results.extend(leftovers)
    results.sort(key=lambda r: float(r["ds_m"]) if r.get("ds_m") not in (None, "") else np.inf)
    _write_csv(results, csv_path)

    print()
    print(f"Saved ds-scaling results to: {csv_path}")

    plot_paths: Dict[str, Path] = {}
    if make_plots:
        plot_paths.update(
            _make_plots(
                results,
                out_dir,
                track_id,
                model_name=model_name,
                integrator_name=integrator_name,
                continuity=continuity,
            )
        )
        plot_paths.update(
            _make_lap_asymptote_plot(
                results,
                out_dir,
                track_id,
                model_name=model_name,
                integrator_name=integrator_name,
                continuity=continuity,
            )
        )

    out_paths: Dict[str, Path] = {"csv": csv_path}
    out_paths.update(plot_paths)
    return out_paths


def _make_plots(
    results: List[Dict[str, Any]],
    out_dir: Path,
    track_id: str,
    model_name: str = "point_mass",
    integrator_name: str = "euler",
    continuity: str = "C2",
) -> Dict[str, Path]:
    """Generate summary plots from the collected results."""
    # Only use runs without errors and with basic metrics present.
    valid = [
        r
        for r in results
        if not r.get("error")
        and r.get("ds_m") is not None
        and r.get("N") is not None
        and r.get("solve_time_s") is not None
    ]
    if len(valid) < 2:
        print("Not enough valid runs for plotting. Skipping plots.")
        return {}

    ds_arr = np.array([r["ds_m"] for r in valid], dtype=float)
    N_arr = np.array([r.get("N", np.nan) for r in valid], dtype=float)

    def _to_array(key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in valid], dtype=float)

    solve_time_arr = _to_array("solve_time_s")
    iter_arr = _to_array("iter_count")
    t_per_iter_arr = _to_array("time_per_iter_ms")
    t_per_point_arr = _to_array("time_per_point_ms")
    lap_time_arr = _to_array("lap_time_s")
    iters_per_point_arr = _to_array("iters_per_point")
    ratio_arr = _to_array("time_per_lap_solve_ratio")

    # ------------------------------------------------------------------
    # Single 3x3 figure: top 2 rows vs ds (log-scale x), bottom row vs N.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    # Row 0: vs ds
    axes[0, 0].plot(ds_arr, t_per_iter_arr, "o-")
    axes[0, 0].set_ylabel("time per iter [ms]")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    axes[0, 1].plot(ds_arr, iter_arr, "o-")
    axes[0, 1].set_ylabel("total iterations")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    axes[0, 2].plot(ds_arr, solve_time_arr, "o-")
    axes[0, 2].set_ylabel("solve time [s]")
    axes[0, 2].grid(True, linestyle="--", alpha=0.4)

    # Row 1: vs ds
    axes[1, 0].plot(ds_arr, lap_time_arr, "o-")
    axes[1, 0].set_ylabel("racing lap time [s]")
    axes[1, 0].set_xlabel("ds [m]")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    axes[1, 1].plot(ds_arr, t_per_point_arr, "o-")
    axes[1, 1].set_ylabel("time per point [ms]")
    axes[1, 1].set_xlabel("ds [m]")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    axes[1, 2].plot(ds_arr, iters_per_point_arr, "o-")
    axes[1, 2].set_ylabel("iters per point")
    axes[1, 2].set_xlabel("ds [m]")
    axes[1, 2].grid(True, linestyle="--", alpha=0.4)

    # Apply log scale to ds-plots (first two rows).
    for row in range(2):
        for col in range(3):
            axes[row, col].set_xscale("log")

    # Row 2: vs N
    axes[2, 0].plot(N_arr, solve_time_arr, "o-")
    axes[2, 0].set_xlabel("N")
    axes[2, 0].set_ylabel("solve time [s]")
    axes[2, 0].grid(True, linestyle="--", alpha=0.4)

    axes[2, 1].plot(N_arr, t_per_point_arr, "o-")
    axes[2, 1].set_xlabel("N")
    axes[2, 1].set_ylabel("time per point [ms]")
    axes[2, 1].grid(True, linestyle="--", alpha=0.4)

    axes[2, 2].plot(N_arr, ratio_arr, "o-")
    axes[2, 2].set_xlabel("N")
    axes[2, 2].set_ylabel("solve_time / lap_time")
    axes[2, 2].grid(True, linestyle="--", alpha=0.4)

    tag = f"{track_id}_{model_name}_{integrator_name}_{continuity}"
    fig.suptitle(
        f"ds-scaling (track={track_id}, model={model_name}, integrator={integrator_name}, continuity={continuity})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    all_path = out_dir / f"ds_scaling_all_{tag}.png"
    fig.savefig(all_path, dpi=200)
    plt.close(fig)

    print(f"Saved ds-scaling plots to: {all_path}")

    return {"plot_all": all_path}


def _make_lap_asymptote_plot(
    results: List[Dict[str, Any]],
    out_dir: Path,
    track_id: str,
    model_name: str = "point_mass",
    integrator_name: str = "euler",
    continuity: str = "C2",
) -> Dict[str, Path]:
    """
    Lap-time convergence vs ds.

    Left: lap time vs ds with the fine-grid value drawn as the asymptote.
    Right: |lap time - asymptote| vs ds on log-log (discretisation error).

    The finest ds that succeeded is taken as the reference ("truth"); this
    shows how coarse the grid can get before lap time degrades meaningfully.
    """
    valid = [
        r
        for r in results
        if not r.get("error")
        and r.get("ds_m") is not None
        and r.get("lap_time_s") is not None
        and r.get("return_status") == "Solve_Succeeded"
    ]
    if len(valid) < 2:
        print("Not enough valid runs for lap-asymptote plot. Skipping.")
        return {}

    valid = sorted(valid, key=lambda r: float(r["ds_m"]))
    ds_arr = np.array([float(r["ds_m"]) for r in valid], dtype=float)
    lap_arr = np.array([float(r["lap_time_s"]) for r in valid], dtype=float)

    # Reference = finest ds (smallest ds_m).
    asymptote = float(lap_arr[0])
    err = np.abs(lap_arr - asymptote)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(ds_arr, lap_arr, "o-", color="C0")
    axes[0].axhline(
        asymptote,
        color="k",
        ls="--",
        alpha=0.7,
        label=f"asymptote = {asymptote:.3f} s (ds={ds_arr[0]:.2f} m)",
    )
    axes[0].set_xlabel("ds [m]")
    axes[0].set_ylabel("racing lap time [s]")
    axes[0].set_title("Lap-time convergence")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    # Annotate each point with % deviation from the asymptote.
    for ds_i, lap_i in zip(ds_arr, lap_arr):
        if asymptote != 0.0:
            pct = (lap_i - asymptote) / asymptote * 100.0
            axes[0].annotate(
                f"{pct:+.1f}%",
                (ds_i, lap_i),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=8,
                ha="center",
            )

    # Error plot (skip the reference point itself where err == 0).
    mask = err > 0
    if np.any(mask):
        axes[1].loglog(ds_arr[mask], err[mask], "s-", color="C3")
    axes[1].set_xlabel("ds [m]")
    axes[1].set_ylabel("|lap time - asymptote| [s]")
    axes[1].set_title("Discretisation error (vs finest ds)")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    tag = f"{track_id}_{model_name}_{integrator_name}_{continuity}"
    fig.suptitle(
        f"Lap-time vs ds (track={track_id}, model={model_name}, "
        f"integrator={integrator_name}, continuity={continuity})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    lap_path = out_dir / f"ds_scaling_lap_asymptote_{tag}.png"
    fig.savefig(lap_path, dpi=200)
    plt.close(fig)

    print(f"Saved lap-time asymptote plot to: {lap_path}")
    return {"plot_lap_asymptote": lap_path}


def _load_results_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load ds-scaling results from a CSV produced by this script."""
    rows: List[Dict[str, Any]] = []
    numeric_keys = {
        "ds_m",
        "N",
        "solve_time_s",
        "iter_count",
        "time_per_point_ms",
        "time_per_iter_ms",
        "lap_time_s",
        "iters_per_point",
        "time_per_lap_solve_ratio",
        "reg_term",
        "reg_term_relative",
    }

    with csv_path.open("r", newline="") as f_csv:
        reader = csv.DictReader(f_csv)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, val in row.items():
                if key in numeric_keys:
                    if val in ("", None):
                        parsed[key] = np.nan
                    else:
                        try:
                            parsed[key] = float(val)
                        except ValueError:
                            parsed[key] = np.nan
                else:
                    parsed[key] = val
            rows.append(parsed)

    return rows


def _make_combined_plots(
    config_results: List[Dict[str, Any]],
    out_dir: Path,
    track_id: str,
    model_name: str = "point_mass",
) -> Dict[str, Path]:
    """
    Generate combined summary plots overlaying multiple configurations.

    Parameters
    ----------
    config_results : list of dicts
        Each dict should have keys:
            - 'label': legend label
            - 'results': list of result dicts (same format as in _make_plots)
    """
    # Prepare colors/markers for up to four configurations.
    colors = ["C0", "C1", "C2", "C3"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    any_valid = False

    for idx, cfg in enumerate(config_results):
        label = cfg["label"]
        results = cfg["results"]

        valid = [
            r
            for r in results
            if not r.get("error")
            and r.get("ds_m") not in (None, "")
            and r.get("N") not in (None, "")
            and r.get("solve_time_s") not in (None, "")
        ]
        if len(valid) < 2:
            continue

        any_valid = True

        ds_arr = np.array([float(r["ds_m"]) for r in valid], dtype=float)
        N_arr = np.array([float(r.get("N", np.nan)) for r in valid], dtype=float)

        def _to_array(key: str) -> np.ndarray:
            return np.array(
                [float(r.get(key, np.nan)) for r in valid],
                dtype=float,
            )

        solve_time_arr = _to_array("solve_time_s")
        iter_arr = _to_array("iter_count")
        t_per_iter_arr = _to_array("time_per_iter_ms")
        t_per_point_arr = _to_array("time_per_point_ms")
        lap_time_arr = _to_array("lap_time_s")
        iters_per_point_arr = _to_array("iters_per_point")
        ratio_arr = _to_array("time_per_lap_solve_ratio")

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        style = marker + "-"

        # Row 0: vs ds
        axes[0, 0].plot(ds_arr, t_per_iter_arr, style, color=color, label=label)
        axes[0, 1].plot(ds_arr, iter_arr, style, color=color, label=label)
        axes[0, 2].plot(ds_arr, solve_time_arr, style, color=color, label=label)

        # Row 1: vs ds
        axes[1, 0].plot(ds_arr, lap_time_arr, style, color=color, label=label)
        axes[1, 1].plot(ds_arr, t_per_point_arr, style, color=color, label=label)
        axes[1, 2].plot(ds_arr, iters_per_point_arr, style, color=color, label=label)

        # Row 2: vs N
        axes[2, 0].plot(N_arr, solve_time_arr, style, color=color, label=label)
        axes[2, 1].plot(N_arr, t_per_point_arr, style, color=color, label=label)
        axes[2, 2].plot(N_arr, ratio_arr, style, color=color, label=label)

    if not any_valid:
        print("Not enough valid runs across configurations for plotting. Skipping combined plots.")
        plt.close(fig)
        return {}

    # Labels / grids.
    axes[0, 0].set_ylabel("time per iter [ms]")
    axes[0, 1].set_ylabel("total iterations")
    axes[0, 2].set_ylabel("solve time [s]")

    axes[1, 0].set_ylabel("racing lap time [s]")
    axes[1, 0].set_xlabel("ds [m]")
    axes[1, 1].set_ylabel("time per point [ms]")
    axes[1, 1].set_xlabel("ds [m]")
    axes[1, 2].set_ylabel("iters per point")
    axes[1, 2].set_xlabel("ds [m]")

    for row in range(2):
        for col in range(3):
            axes[row, col].set_xscale("log")
            axes[row, col].grid(True, linestyle="--", alpha=0.4)

    axes[2, 0].set_xlabel("N")
    axes[2, 0].set_ylabel("solve time [s]")
    axes[2, 1].set_xlabel("N")
    axes[2, 1].set_ylabel("time per point [ms]")
    axes[2, 2].set_xlabel("N")
    axes[2, 2].set_ylabel("solve_time / lap_time")

    for col in range(3):
        axes[2, col].grid(True, linestyle="--", alpha=0.4)

    # Use a single legend (top-left subplot).
    axes[0, 0].legend()

    fig.suptitle(
        f"ds-scaling multi-config (track={track_id}, model={model_name})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_path = out_dir / f"ds_scaling_all_{track_id}_{model_name}_multi.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)

    print(f"Saved combined ds-scaling plots to: {combined_path}")

    return {"plot_all_multi": combined_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense ds-scaling experiment for Fast-LTO OCP.")

    parser.add_argument("--track-id", type=str, default="fsg_random", help="Track identifier.")
    parser.add_argument(
        "--track-type",
        type=str,
        default="fsg",
        choices=["fsg", "ellipse", "bean"],
        help="Track type (only used if track needs to be generated).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="point_mass",
        help="Vehicle model name (must match PipelineConfig.model_name).",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="Spatial integrator.",
    )
    parser.add_argument(
        "--continuity",
        type=str,
        default="C4",
        choices=["C2", "C4"],
        help="Spline continuity for track discretization.",
    )
    parser.add_argument(
        "--reg-u",
        type=float,
        default=600.0,
        help="Input rate regularisation weight on changes in inputs (du).",
    )
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=5.0,
        help="Initial speed guess [m/s].",
    )
    parser.add_argument(
        "--ds-min",
        type=float,
        default=0.1,
        help="Minimum ds [m] (log-space lower bound).",
    )
    parser.add_argument(
        "--ds-max",
        type=float,
        default=5.0,
        help="Maximum ds [m] (log-space upper bound).",
    )
    parser.add_argument(
        "--num-ds",
        type=int,
        default=20,
        help="Number of ds samples (log-spaced).",
    )
    parser.add_argument(
        "--ds-list",
        type=float,
        nargs="+",
        default=None,
        help="Explicit ds values [m] to sweep (overrides --ds-min/--ds-max/--num-ds).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML run config. All non-swept params (vehicle/model, "
            "reg, margin, smoothing, v_max, initial speed, mode) are taken from "
            "it; ds, continuity and integrator remain the swept variables."
        ),
    )
    parser.add_argument(
        "--reg-l2-ref",
        type=float,
        default=None,
        help=(
            "Reference L2 input-reg weight (reg_u_l2). Scaled with ds so its "
            "relative cost stays in budget. Defaults to the YAML reg_u_l2."
        ),
    )
    parser.add_argument(
        "--reg-ds-ref",
        type=float,
        default=1.0,
        help="ds [m] at which --reg-l2-ref is calibrated (for linear_ds scaling).",
    )
    parser.add_argument(
        "--reg-scale-mode",
        type=str,
        default="linear_ds",
        choices=["linear_ds", "fixed"],
        help="How reg_u_l2 scales with ds. linear_ds keeps relative reg ~constant.",
    )
    parser.add_argument(
        "--reg-relative-max",
        type=float,
        default=0.03,
        help="Max allowed reg fraction of the objective; halve-and-retry if exceeded.",
    )
    parser.add_argument(
        "--multi-config",
        action="store_true",
        help=(
            "Run four predefined configurations " "(euler/rk4 × C2/C4) and generate combined plots."
        ),
    )
    parser.add_argument(
        "--euler-only",
        action="store_true",
        help=(
            "When combined with --multi-config, run only Euler × (C2/C4) "
            "instead of all four integrator/continuity combinations."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation (still writes CSV).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore any existing CSV and recompute every ds from scratch. "
            "By default the experiment resumes: already-solved ds values are "
            "skipped and the CSV is checkpointed after each solve."
        ),
    )

    return parser.parse_args()


def _resolve_non_swept_params(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Resolve the non-swept (fixed) experiment params.

    When ``--config`` is given, vehicle/model params, reg, margin, smoothing,
    speed and mode come from the YAML; ds, continuity and integrator stay as
    the swept variables.  CLI flags still override the scalar reg/speed knobs.
    """
    params: Dict[str, Any] = {
        "model_name": args.model,
        "reg_u": args.reg_u,
        "initial_speed": args.initial_speed,
        "reg_u_l2_ref": args.reg_l2_ref,
        "reg_ds_ref": args.reg_ds_ref,
        "reg_scale_mode": args.reg_scale_mode,
        "reg_relative_max": args.reg_relative_max,
        "boundary_margin": 0.0,
        "smooth_centerline": 0,
        "mode": "trackdrive",
        "vehicle_config": None,
    }

    if args.config is not None:
        from fast_lto.config import RunConfig

        rc = RunConfig.from_yaml(args.config)
        params["vehicle_config"] = rc.vehicle
        params["model_name"] = rc.pipeline.model_name
        params["reg_u"] = rc.pipeline.reg_u
        params["boundary_margin"] = rc.pipeline.boundary_margin
        params["smooth_centerline"] = rc.pipeline.smooth_centerline
        params["mode"] = rc.pipeline.mode
        if rc.pipeline.initial_speed is not None:
            params["initial_speed"] = rc.pipeline.initial_speed
        # Use the YAML reg_u_l2 as the reference weight unless overridden.
        if args.reg_l2_ref is None:
            params["reg_u_l2_ref"] = rc.pipeline.reg_u_l2
        rc.validate_for_model()
        print(
            f"Loaded config from {args.config}: model={rc.pipeline.model_name}, "
            f"v_max={rc.vehicle.v_max}, margin={rc.pipeline.boundary_margin}, "
            f"smooth={rc.pipeline.smooth_centerline}, reg_u={rc.pipeline.reg_u}, "
            f"reg_u_l2_ref={params['reg_u_l2_ref']}"
        )

    return params


def main() -> None:
    args = _parse_args()
    fixed = _resolve_non_swept_params(args)
    if args.multi_config:
        # Run predefined configurations and generate combined plots.
        if args.euler_only:
            configs = [
                ("euler", "C2", "Euler C2"),
                ("euler", "C4", "Euler C4"),
            ]
            print("Running multi-config ds-scaling experiment (Euler × C2/C4).")
        else:
            configs = [
                ("euler", "C2", "Euler C2"),
                ("euler", "C4", "Euler C4"),
                ("rk4", "C2", "RK4 C2"),
                ("rk4", "C4", "RK4 C4"),
            ]
            print("Running multi-config ds-scaling experiment (euler/rk4 × C2/C4).")

        csv_entries: List[Dict[str, Any]] = []

        for integrator_name, continuity, label in configs:
            print()
            print(f"=== Configuration: {label} ===")
            out_paths = run_ds_scaling_experiment(
                track_id=args.track_id,
                track_type=args.track_type,
                model_name=fixed["model_name"],
                integrator_name=integrator_name,
                continuity=continuity,
                reg_u=fixed["reg_u"],
                initial_speed=fixed["initial_speed"],
                ds_min=args.ds_min,
                ds_max=args.ds_max,
                num_ds=args.num_ds,
                ds_list=args.ds_list,
                make_plots=False,  # plots handled by combined plot function
                vehicle_config=fixed["vehicle_config"],
                reg_u_l2_ref=fixed["reg_u_l2_ref"],
                reg_ds_ref=fixed["reg_ds_ref"],
                reg_scale_mode=fixed["reg_scale_mode"],
                reg_relative_max=fixed["reg_relative_max"],
                boundary_margin=fixed["boundary_margin"],
                smooth_centerline=fixed["smooth_centerline"],
                mode=fixed["mode"],
                resume=not args.no_resume,
            )
            csv_entries.append(
                {
                    "label": label,
                    "csv": out_paths["csv"],
                }
            )

        if not args.no_plots:
            # Load CSVs and generate combined plots.
            config_results: List[Dict[str, Any]] = []
            for entry in csv_entries:
                label = entry["label"]
                csv_path = entry["csv"]
                results = _load_results_from_csv(csv_path)
                config_results.append({"label": label, "results": results})

            # All CSVs are in the same directory by construction.
            if csv_entries:
                out_dir = csv_entries[0]["csv"].parent
                _make_combined_plots(
                    config_results, out_dir, track_id=args.track_id, model_name=args.model
                )
        else:
            print("Skipping combined plot generation due to --no-plots.")

    else:
        # Single-configuration mode (original behavior).
        run_ds_scaling_experiment(
            track_id=args.track_id,
            track_type=args.track_type,
            model_name=fixed["model_name"],
            integrator_name=args.integrator,
            continuity=args.continuity,
            reg_u=fixed["reg_u"],
            initial_speed=fixed["initial_speed"],
            ds_min=args.ds_min,
            ds_max=args.ds_max,
            num_ds=args.num_ds,
            ds_list=args.ds_list,
            make_plots=not args.no_plots,
            vehicle_config=fixed["vehicle_config"],
            reg_u_l2_ref=fixed["reg_u_l2_ref"],
            reg_ds_ref=fixed["reg_ds_ref"],
            reg_scale_mode=fixed["reg_scale_mode"],
            reg_relative_max=fixed["reg_relative_max"],
            boundary_margin=fixed["boundary_margin"],
            smooth_centerline=fixed["smooth_centerline"],
            mode=fixed["mode"],
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
