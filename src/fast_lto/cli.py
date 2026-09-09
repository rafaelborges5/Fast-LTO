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
from typing import TYPE_CHECKING

from fast_lto.modes import MODE_NAMES
from fast_lto.pipeline import TRACK_TYPES, run_pipeline

if TYPE_CHECKING:
    from fast_lto.config import RunConfig


# argparse dest -> PipelineConfig field, for every flag that simply replaces a
# value when given. Flags needing real logic (the store_true pairs, the two
# regularisation flags) are handled explicitly in build_run_config below.
_VALUE_OVERRIDES = {
    "track_id": "track_id",
    "track_type": "track_type",
    "mode": "mode",
    "model": "model_name",
    "integrator": "integrator_name",
    "ds": "ds_m",
    "continuity": "continuity",
    "boundary_margin": "boundary_margin",
    "warm_start_seed": "warm_start_seed",
    "reg_u_l2": "reg_u_l2",
    "savgol_bounds": "use_savgol_bounds",
}

# argparse dest -> AutoxConfig field.
_AUTOX_OVERRIDES = {
    "autox_extension": "extension_m",
    "autox_lead_in": "lead_in_m",
    "autox_ocp_lead": "ocp_lead_m",
    "autox_timing_offset": "timing_offset_m",
    "autox_start_x": "start_x",
    "autox_start_y": "start_y",
    "autox_start_node_offset": "start_node_offset",
}


def build_run_config(args: argparse.Namespace) -> "RunConfig":
    """Build the run configuration from parsed arguments.

    One path, whether or not ``--config`` was given: start from the YAML (or
    from the dataclass defaults when there is none), then apply exactly the
    flags the user actually passed.  A flag left at its default never
    overwrites a value from the file.
    """
    from fast_lto.config import RunConfig

    run_config = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    pipeline = run_config.pipeline

    for arg_name, field_name in _VALUE_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(pipeline, field_name, value)

    # Autox settings live on their own block (see modes.AutoxConfig), so these
    # are applied by attribute rather than through the flat table above.
    for arg_name, field_name in _AUTOX_OVERRIDES.items():
        value = getattr(args, arg_name)
        if value is not None:
            setattr(pipeline.autox, field_name, value)

    # Not part of the tables above: an explicitly requested launch speed has to
    # be pinned, or a `--mode` on the same command line re-derives it.
    if args.initial_speed is not None:
        pipeline.set_initial_speed(args.initial_speed)

    # --reg-du-vec beats --reg-u, per its own help text.
    if args.reg_du_vec is not None:
        pipeline.reg_u = [float(x) for x in args.reg_du_vec.split(",") if x.strip() != ""]
    elif args.reg_u is not None:
        pipeline.reg_u = args.reg_u

    if args.no_warm_start:
        pipeline.warm_start = "off"
    elif args.warm_start is not None:
        pipeline.warm_start = args.warm_start

    # store_true/store_false flags carry no "unset" state, so each one is only
    # applied when it differs from its default -- otherwise simply parsing the
    # command line would silently override the config file.
    if args.solver_verbose:
        pipeline.solver_verbose = True
    if args.no_normalization:
        pipeline.normalize_states_and_inputs = False
    if args.no_export:
        pipeline.export_trajectory = False
    if args.no_plot:
        pipeline.plot_results = False
    if args.no_show_plots:
        pipeline.show_plots = False
    if args.generate_track:
        pipeline.generate_track = True

    if args.repo_root is not None:
        pipeline.repo_root = args.repo_root

    return run_config


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Separate from :func:`main` so tests can parse a command line without
    running a pipeline.
    """
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
        choices=TRACK_TYPES,
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
        choices=sorted(MODE_NAMES),
        default=None,
        help=f"Event mode: {', '.join(sorted(MODE_NAMES))}. Default: trackdrive",
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
        "--autox-ocp-lead",
        type=float,
        default=None,
        help=(
            "Meters before the start line that the OCP's own optimized horizon "
            "begins (instead of the flat autox_lead_in hold), for autox mode. "
            "Default: 0.0"
        ),
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
    parser.add_argument(
        "--autox-start-x",
        type=float,
        default=None,
        help="Car's real start x (m) in the map frame, for autox mode. Default: 0.0",
    )
    parser.add_argument(
        "--autox-start-y",
        type=float,
        default=None,
        help="Car's real start y (m) in the map frame, for autox mode. Default: 0.0",
    )
    parser.add_argument(
        "--autox-start-node-offset",
        type=int,
        default=None,
        help=(
            "Nodes to step forward from the track sample nearest "
            "(--autox-start-x, --autox-start-y) before pinning the OCP's "
            "launch node, for autox mode. Default: 1"
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

    # Tri-state on purpose. As a `store_false` flag this both read backwards
    # (passing --savgol-bounds turned smoothing *on*, against its own help text)
    # and offered no way to turn it off at all. BooleanOptionalAction gives both
    # directions, and the None default keeps an unpassed flag from silently
    # overriding the YAML.
    parser.add_argument(
        "--savgol-bounds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Smooth the lateral bounds with a Savitzky-Golay filter. "
            "Use --no-savgol-bounds to disable. Default: from the config file."
        ),
    )

    # Warm start
    parser.add_argument(
        "--warm-start",
        type=str,
        choices=["off", "auto", "ladder"],
        default=None,
        help=(
            "Seeding policy for the OCP. 'off' solves cold and touches no seed "
            "store; 'auto' seeds from the closest compatible previous solve, "
            "and inserts one easier solve when the margin is past the point "
            "where the centreline guess goes infeasible; 'ladder' always walks "
            "up from a safe margin. Default: auto."
        ),
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Alias for --warm-start off.",
    )
    parser.add_argument(
        "--warm-start-seed",
        type=str,
        default=None,
        help="Path to a solution JSON to seed this solve from, bypassing the store.",
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

    return parser


def main() -> None:
    """Main CLI entry point."""
    args = build_parser().parse_args()

    run_config = build_run_config(args)
    run_config.validate_for_model()
    config = run_config.to_pipeline_config()

    # Run pipeline
    print("Running Fast-LTO pipeline")
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
