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

if __name__ == "__main__":
    # Allow running as a standalone script from the repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline import PipelineConfig, run_pipeline
else:
    # Normal package-based import when used as fast_lto.experiments.ds_scaling.
    from pipeline import PipelineConfig, run_pipeline


def _get_repo_root() -> Path:
    """Infer repository root from this file location."""
    return Path(__file__).resolve().parents[2]


def run_ds_scaling_experiment(
    track_id: str = "fsg_random",
    track_type: str = "fsg",
    model_name: str = "point_mass",
    integrator_name: str = "euler",
    reg_u: float = 1e-4,
    initial_speed: float = 5.0,
    ds_min: float = 0.1,
    ds_max: float = 5.0,
    num_ds: int = 20,
    make_plots: bool = True,
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

    ds_values = np.geomspace(ds_min, ds_max, num=num_ds)

    results: List[Dict[str, Any]] = []

    print("Running ds-scaling experiment")
    print(f"  Track ID: {track_id} (type={track_type})")
    print(f"  Model: {model_name}, integrator: {integrator_name}")
    print(f"  ds range: [{ds_min:.3f}, {ds_max:.3f}] m with {num_ds} points (log-spaced)")
    print()

    for ds in ds_values:
        ds_float = float(ds)
        print("=" * 60)
        print(f"ds = {ds_float:.4f} m")

        metrics: Dict[str, Any] = {
            "track_id": track_id,
            "model_name": model_name,
            "integrator_name": integrator_name,
            "ds_m": ds_float,
        }

        try:
            config = PipelineConfig(
                track_id=track_id,
                track_type=track_type,
                generate_track=False,
                repo_root=repo_root,
                ds_m=ds_float,
                continuity="C4",
                model_name=model_name,
                integrator_name=integrator_name,
                reg_u=reg_u,
                initial_speed=initial_speed,
                plot_results=False,
                show_plots=False,
            )

            # Force recomputation of spline + bounds for each ds, but reuse track CSV.
            pipeline_results = run_pipeline(
                config=config,
                start_from="spline",
                end_at="ocp",
            )

            solution_path = pipeline_results["ocp"]

            with solution_path.open("r") as f:
                sol_data = json.load(f)

            profiling = sol_data.get("profiling", {}) or {}

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

            if (
                time_per_point_ms is not None
                and iter_count not in (None, 0)
                and N not in (None, 0)
            ):
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

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / f"ds_scaling_{track_id}.csv"
    if results:
        # Collect all keys across results to keep CSV header stable.
        fieldnames: List[str] = sorted({k for row in results for k in row.keys()})
        with csv_path.open("w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print()
    print(f"Saved ds-scaling results to: {csv_path}")

    plot_paths: Dict[str, Path] = {}
    if make_plots:
        plot_paths.update(_make_plots(results, out_dir, track_id))

    out_paths: Dict[str, Path] = {"csv": csv_path}
    out_paths.update(plot_paths)
    return out_paths


def _make_plots(
    results: List[Dict[str, Any]],
    out_dir: Path,
    track_id: str,
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
    # Plots vs ds (log-scale x)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()

    axes[0].plot(ds_arr, t_per_iter_arr, "o-")
    axes[0].set_ylabel("time per iter [ms]")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(ds_arr, iter_arr, "o-")
    axes[1].set_ylabel("total iterations")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(ds_arr, solve_time_arr, "o-")
    axes[2].set_ylabel("solve time [s]")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    axes[3].plot(ds_arr, lap_time_arr, "o-")
    axes[3].set_ylabel("racing lap time [s]")
    axes[3].set_xlabel("ds [m]")
    axes[3].grid(True, linestyle="--", alpha=0.4)

    axes[4].plot(ds_arr, t_per_point_arr, "o-")
    axes[4].set_ylabel("time per point [ms]")
    axes[4].set_xlabel("ds [m]")
    axes[4].grid(True, linestyle="--", alpha=0.4)

    axes[5].plot(ds_arr, iters_per_point_arr, "o-")
    axes[5].set_ylabel("iters per point")
    axes[5].set_xlabel("ds [m]")
    axes[5].grid(True, linestyle="--", alpha=0.4)

    for ax in axes:
        ax.set_xscale("log")

    fig.suptitle(f"ds-scaling (track={track_id})", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    summary_path = out_dir / f"ds_scaling_vs_ds_{track_id}.png"
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plots vs N (problem size)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    axes2[0].plot(N_arr, solve_time_arr, "o-")
    axes2[0].set_xlabel("N")
    axes2[0].set_ylabel("solve time [s]")
    axes2[0].grid(True, linestyle="--", alpha=0.4)

    axes2[1].plot(N_arr, t_per_point_arr, "o-")
    axes2[1].set_xlabel("N")
    axes2[1].set_ylabel("time per point [ms]")
    axes2[1].grid(True, linestyle="--", alpha=0.4)

    axes2[2].plot(N_arr, ratio_arr, "o-")
    axes2[2].set_xlabel("N")
    axes2[2].set_ylabel("solve_time / lap_time")
    axes2[2].grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle(f"Scaling vs N (track={track_id})", fontsize=14)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    vs_n_path = out_dir / f"ds_scaling_vs_N_{track_id}.png"
    fig2.savefig(vs_n_path, dpi=200)
    plt.close(fig2)

    print(f"Saved ds-scaling plots to: {summary_path} and {vs_n_path}")

    return {"plot_vs_ds": summary_path, "plot_vs_N": vs_n_path}


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
        "--reg-u",
        type=float,
        default=1e-4,
        help="Input regularisation weight.",
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
        "--no-plots",
        action="store_true",
        help="Disable plot generation (still writes CSV).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ds_scaling_experiment(
        track_id=args.track_id,
        track_type=args.track_type,
        model_name=args.model,
        integrator_name=args.integrator,
        reg_u=args.reg_u,
        initial_speed=args.initial_speed,
        ds_min=args.ds_min,
        ds_max=args.ds_max,
        num_ds=args.num_ds,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

