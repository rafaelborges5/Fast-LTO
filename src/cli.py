"""
Command-line interface for Fast-LTO pipeline.

Usage:
    python -m fast_lto.cli --start-from track --track-id fsg_random
    python -m fast_lto.cli --start-from ocp --track-id ellipse --integrator rk4
    python -m fast_lto.cli --start-from plot --track-id bean
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import PipelineConfig, StepName, run_pipeline


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fast-LTO: Lap Time Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline from scratch
  python -m fast_lto.cli --start-from track --track-id fsg_random

  # Re-solve OCP with different settings (reuses existing track)
  python -m fast_lto.cli --start-from ocp --track-id fsg_random --integrator rk4

  # Only visualize existing solution
  python -m fast_lto.cli --start-from plot --track-id fsg_random

  # Generate new track and fit spline, but skip OCP
  python -m fast_lto.cli --start-from track --end-at bounds --track-id my_track
        """,
    )

    # Track selection
    parser.add_argument(
        "--track-id",
        type=str,
        default="fsg_random",
        help="Track identifier (used for file naming). Default: fsg_random",
    )
    parser.add_argument(
        "--track-type",
        type=str,
        choices=["fsg", "ellipse", "bean"],
        default="fsg",
        help="Type of track to generate. Default: fsg",
    )
    parser.add_argument(
        "--generate-track",
        action="store_true",
        help="Force generation of new track CSV (even if it exists).",
    )

    # Pipeline control
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["track", "spline", "bounds", "ocp", "plot"],
        default="track",
        help="Step to start from. Default: track",
    )
    parser.add_argument(
        "--end-at",
        type=str,
        choices=["track", "spline", "bounds", "ocp", "plot"],
        default=None,
        help="Step to end at (inclusive). If not set, runs to completion.",
    )

    # Spline fitting options
    parser.add_argument(
        "--ds",
        type=float,
        default=None,
        help="Discretization step in meters. Default: 0.5",
    )
    parser.add_argument(
        "--continuity",
        type=str,
        choices=["C2", "C4"],
        default="C2",
        help="Spline continuity. Default: C2",
    )

    # OCP options
    parser.add_argument(
        "--model",
        type=str,
        default="point_mass",
        help="Vehicle model. Default: point_mass",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        choices=["euler", "rk4"],
        default="euler",
        help="Spatial integrator. Default: euler",
    )
    parser.add_argument(
        "--reg-u",
        type=float,
        default=None,
        help=(
            "Input rate regularization weight on changes in inputs (du). "
            "If omitted, uses PipelineConfig default."
        ),
    )
    parser.add_argument(
        "--reg-du-vec",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of input rate weights for each input "
            "(e.g. '600,300' for two inputs). Overrides --reg-u if provided."
        ),
    )
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=1.0,
        help="Initial speed guess (m/s). Default: 1.0",
    )

    parser.add_argument(
        "--no-savgol-bounds",
        action="store_true",
        help="Disable Savitzky–Golay smoothing of lateral bounds.",
    )

    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Disable state/input normalization inside the OCP (use physical units).",
    )

    # Visualization
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip visualization step.",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Save plots but don't display them interactively.",
    )

    # Paths
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory. Auto-detected if not set.",
    )

    args = parser.parse_args()

    # Create configuration
    config_kwargs = dict(
        track_id=args.track_id,
        track_type=args.track_type,
        generate_track=args.generate_track,
        repo_root=args.repo_root,
        ds_m=args.ds,
        continuity=args.continuity,
        use_savgol_bounds=not args.no_savgol_bounds,
        model_name=args.model,
        integrator_name=args.integrator,
        initial_speed=args.initial_speed,
        normalize_states_and_inputs=not args.no_normalization,
        plot_results=not args.no_plot,
        show_plots=not args.no_show_plots,
    )
    # Scalar or vector regularisation weights
    if args.reg_du_vec is not None:
        # Parse comma-separated floats into a list; validated later in OCP build
        reg_vec = [float(x) for x in args.reg_du_vec.split(",") if x.strip() != ""]
        config_kwargs["reg_u"] = reg_vec
    elif args.reg_u is not None:
        config_kwargs["reg_u"] = args.reg_u

    config = PipelineConfig(**config_kwargs)

    # Run pipeline
    print(f"Running Fast-LTO pipeline")
    print(f"  Track ID: {config.track_id}")
    print(f"  Start from: {args.start_from}")
    if args.end_at:
        print(f"  End at: {args.end_at}")
    print()

    results = run_pipeline(
        config=config,
        start_from=args.start_from,
        end_at=args.end_at,
    )

    print()
    print("Pipeline completed successfully!")
    print("Output files:")
    for step, path in results.items():
        print(f"  {step}: {path}")


if __name__ == "__main__":
    main()
