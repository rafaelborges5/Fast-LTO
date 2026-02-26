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
    continuity: str = "C4",
    reg_u: float = 1e-4,
    initial_speed: float = 1.0,
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
    print(f"  Model: {model_name}, integrator: {integrator_name}, continuity: {continuity}")
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
                continuity=continuity,
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
    csv_path = out_dir / f"ds_scaling_{track_id}_{integrator_name}_{continuity}.csv"
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
        plot_paths.update(_make_plots(results, out_dir, track_id, integrator_name, continuity))

    out_paths: Dict[str, Path] = {"csv": csv_path}
    out_paths.update(plot_paths)
    return out_paths


def _make_plots(
    results: List[Dict[str, Any]],
    out_dir: Path,
    track_id: str,
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

    tag = f"{track_id}_{integrator_name}_{continuity}"
    fig.suptitle(
        f"ds-scaling (track={track_id}, integrator={integrator_name}, continuity={continuity})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    all_path = out_dir / f"ds_scaling_all_{tag}.png"
    fig.savefig(all_path, dpi=200)
    plt.close(fig)

    print(f"Saved ds-scaling plots to: {all_path}")

    return {"plot_all": all_path}


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
        f"ds-scaling multi-config (track={track_id})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_path = out_dir / f"ds_scaling_all_{track_id}_multi.png"
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
        "--multi-config",
        action="store_true",
        help=(
            "Run four predefined configurations "
            "(euler/rk4 × C2/C4) and generate combined plots."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation (still writes CSV).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.multi_config:
        # Run four predefined configurations and generate combined plots.
        configs = [
            ("euler", "C2", "Euler C2"),
            ("euler", "C4", "Euler C4"),
            ("rk4", "C2", "RK4 C2"),
            ("rk4", "C4", "RK4 C4"),
        ]

        csv_entries: List[Dict[str, Any]] = []
        print("Running multi-config ds-scaling experiment (euler/rk4 × C2/C4).")

        for integrator_name, continuity, label in configs:
            print()
            print(f"=== Configuration: {label} ===")
            out_paths = run_ds_scaling_experiment(
                track_id=args.track_id,
                track_type=args.track_type,
                model_name=args.model,
                integrator_name=integrator_name,
                continuity=continuity,
                reg_u=args.reg_u,
                initial_speed=args.initial_speed,
                ds_min=args.ds_min,
                ds_max=args.ds_max,
                num_ds=args.num_ds,
                make_plots=False,  # plots handled by combined plot function
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
                _make_combined_plots(config_results, out_dir, track_id=args.track_id)
        else:
            print("Skipping combined plot generation due to --no-plots.")

    else:
        # Single-configuration mode (original behavior).
        run_ds_scaling_experiment(
            track_id=args.track_id,
            track_type=args.track_type,
            model_name=args.model,
            integrator_name=args.integrator,
            continuity=args.continuity,
            reg_u=args.reg_u,
            initial_speed=args.initial_speed,
            ds_min=args.ds_min,
            ds_max=args.ds_max,
            num_ds=args.num_ds,
            make_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()

