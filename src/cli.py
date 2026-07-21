"""
Command-line interface for Fast-LTO pipeline.

Usage:
    python -m fast_lto.cli --config configs/fsg_trackdrive.yaml
    python -m fast_lto.cli --config configs/fsg_trackdrive.yaml --model four_wheel --ds 1.0
    python -m fast_lto.cli --start-from ocp --track-id ellipse --integrator rk4
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
  # Run from a YAML config
  python -m fast_lto.cli --config configs/fsg_trackdrive.yaml

  # YAML config with CLI overrides
  python -m fast_lto.cli --config configs/fsg_trackdrive.yaml --model four_wheel --ds 1.0

  # Run without YAML (legacy mode)
  python -m fast_lto.cli --start-from track --track-id fsg_random

  # Re-solve OCP with different settings (reuses existing track)
  python -m fast_lto.cli --start-from ocp --track-id fsg_random --integrator rk4

  # Only visualize existing solution
  python -m fast_lto.cli --start-from plot --track-id fsg_random
        """,
    )

    # YAML config
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file. CLI flags override YAML values.",
    )

    # Track selection
    parser.add_argument(
        "--track-id",
        type=str,
        default=None,
        help="Track identifier (used for file naming). Default: fsg_random",
    )
    parser.add_argument(
        "--track-type",
        type=str,
        choices=["fsg", "ellipse", "bean"],
        default=None,
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
        choices=["track", "spline", "bounds", "ocp", "export", "plot"],
        default="track",
        help="Step to start from. Default: track",
    )
    parser.add_argument(
        "--end-at",
        type=str,
        choices=["track", "spline", "bounds", "ocp", "export", "plot"],
        default=None,
        help="Step to end at (inclusive). If not set, runs to completion.",
    )

    # Spline fitting options
    parser.add_argument(
        "--ds",
        type=float,
        default=None,
        help="Discretization step in meters.",
    )
    parser.add_argument(
        "--continuity",
        type=str,
        choices=["C2", "C4"],
        default=None,
        help="Spline continuity. Default: C2",
    )

    # Event mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["autox", "trackdrive"],
        default=None,
        help="Event mode: 'autox' or 'trackdrive'. Default: trackdrive",
    )
    parser.add_argument(
        "--autox-extension",
        type=float,
        default=None,
        help="Meters of extra track beyond finish line for autox mode. Default: 50.0",
    )
    parser.add_argument(
        "--autox-lead-in",
        type=float,
        default=None,
        help="Meters of extra track before the start line for autox mode. Default: 0.0",
    )
    parser.add_argument(
        "--autox-timing-offset",
        type=float,
        default=None,
        help=(
            "Meters from the car's start position to the real timing gate, for "
            "autox mode. The accurate lap time is measured between this point "
            "and the same point one lap later. Default: 6.0"
        ),
    )

    # OCP options
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vehicle model. Options: point_mass, dynamic_bicycle, four_wheel.",
    )
    parser.add_argument(
        "--integrator",
        type=str,
        choices=["euler", "rk4"],
        default=None,
        help="Spatial integrator. Default: euler",
    )
    parser.add_argument(
        "--reg-u",
        type=float,
        default=None,
        help="Input rate regularization weight on changes in inputs (du).",
    )
    parser.add_argument(
        "--reg-du-vec",
        type=str,
        default=None,
        help=(
            "Comma-separated list of input rate weights for each input "
            "(e.g. '600,300' for two inputs). Overrides --reg-u."
        ),
    )
    parser.add_argument(
        "--reg-u-l2",
        type=float,
        default=None,
        help="L2 regularization weight on input magnitudes.",
    )
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=None,
        help="Initial speed (m/s). Default: 3.0 for autox, 5.0 for trackdrive.",
    )

    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=None,
        help="Shrink lateral bounds by this amount (m) on each side. Default: 0.0",
    )

    parser.add_argument(
        "--savgol-bounds",
        action="store_false",
        help="Disable Savitzky–Golay smoothing of lateral bounds.",
    )

    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Disable state/input normalization inside the OCP (use physical units).",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Print IPOPT iteration output to terminal during OCP solve.",
    )

    # Export
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip trajectory CSV export step.",
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

    if args.config is not None:
        # ── YAML-based config with optional CLI overrides ──
        from config import RunConfig

        run_config = RunConfig.from_yaml(args.config)

        # Apply CLI overrides on top of YAML values
        if args.track_id is not None:
            run_config.track_id = args.track_id
        if args.track_type is not None:
            run_config.track_type = args.track_type
        if args.mode is not None:
            run_config.mode = args.mode
        if args.model is not None:
            run_config.model_name = args.model
        if args.integrator is not None:
            run_config.integrator_name = args.integrator
        if args.ds is not None:
            run_config.ds_m = args.ds
        if args.continuity is not None:
            run_config.continuity = args.continuity
        if args.initial_speed is not None:
            run_config.initial_speed = args.initial_speed
        if args.autox_extension is not None:
            run_config.autox_extension_m = args.autox_extension
        if args.autox_lead_in is not None:
            run_config.autox_lead_in_m = args.autox_lead_in
        if args.autox_timing_offset is not None:
            run_config.autox_timing_offset_m = args.autox_timing_offset
        if args.boundary_margin is not None:
            run_config.boundary_margin = args.boundary_margin
        if args.reg_du_vec is not None:
            reg_vec = [float(x) for x in args.reg_du_vec.split(",") if x.strip() != ""]
            run_config.reg_u = reg_vec
        elif args.reg_u is not None:
            run_config.reg_u = args.reg_u
        if args.reg_u_l2 is not None:
            run_config.reg_u_l2 = args.reg_u_l2
        if args.solver_verbose:
            run_config.solver_verbose = True
        if args.no_normalization:
            run_config.normalize_states_and_inputs = False
        if args.no_export:
            run_config.export_trajectory = False
        if args.no_plot:
            run_config.plot_results = False
        if args.no_show_plots:
            run_config.show_plots = False

        # Validate vehicle params against selected model before running
        run_config.validate_for_model()

        config = run_config.to_pipeline_config()
        config.generate_track = args.generate_track
        if args.repo_root is not None:
            config.repo_root = args.repo_root
        # Re-run __post_init__ to derive paths with updated fields
        config.__post_init__()

    else:
        # ── Legacy CLI-only mode ──
        config_kwargs = dict(
            track_id=args.track_id or "fsg_random",
            track_type=args.track_type or "fsg",
            generate_track=args.generate_track,
            repo_root=args.repo_root,
            continuity=args.continuity or "C2",
            use_savgol_bounds=not args.savgol_bounds,
            mode=args.mode or "trackdrive",
            model_name=args.model or "point_mass",
            integrator_name=args.integrator or "euler",
            normalize_states_and_inputs=not args.no_normalization,
            solver_verbose=args.solver_verbose,
            export_trajectory=not args.no_export,
            plot_results=not args.no_plot,
            show_plots=not args.no_show_plots,
        )
        if args.initial_speed is not None:
            config_kwargs["initial_speed"] = args.initial_speed
        if args.autox_extension is not None:
            config_kwargs["autox_extension_m"] = args.autox_extension
        if args.autox_lead_in is not None:
            config_kwargs["autox_lead_in_m"] = args.autox_lead_in
        if args.autox_timing_offset is not None:
            config_kwargs["autox_timing_offset_m"] = args.autox_timing_offset
        if args.boundary_margin is not None:
            config_kwargs["boundary_margin"] = args.boundary_margin
        if args.ds is not None:
            config_kwargs["ds_m"] = args.ds
        if args.reg_du_vec is not None:
            reg_vec = [float(x) for x in args.reg_du_vec.split(",") if x.strip() != ""]
            config_kwargs["reg_u"] = reg_vec
        elif args.reg_u is not None:
            config_kwargs["reg_u"] = args.reg_u
        if args.reg_u_l2 is not None:
            config_kwargs["reg_u_l2"] = args.reg_u_l2

        config = PipelineConfig(**config_kwargs)

    # Run pipeline
    print(f"Running Fast-LTO pipeline")
    print(f"  Mode: {config.mode}")
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
