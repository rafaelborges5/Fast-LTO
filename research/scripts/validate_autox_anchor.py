#!/usr/bin/env python3
"""Cheap, no-OCP-solve validation sweep for the autox start-anchor fix.

For every real autox-style track under ``data/tracks/``, builds the track
through the bounds step (cheap, cached to disk) and calls
``_extend_track_for_autox`` with the configured car-start anchor
(``autox_start_x/y/node_offset``), reporting:

  - idx_ref / snap distance from the requested (start_x, start_y) -- should
    be small (a fraction of ds_m) for every real track, since the car's
    actual start position is expected to be close to (0, 0) every session.
  - anchor-invariance: autox_base_length_m/num_points must match a forced
    legacy-equivalent anchor at the CSV's array index 0 -- the loop
    circumference and point count are properties of the track geometry, not
    of where s=0 is defined, so these must agree regardless of anchor.
  - wrap-seam continuity: no discontinuity in consecutive-point spacing
    across the rotation seams introduced by re-anchoring.

Usage:
    python scripts/validate_autox_anchor.py
    python scripts/validate_autox_anchor.py --config configs/autox.yaml
    python scripts/validate_autox_anchor.py --track-id alpnach_last_test_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fast_lto.paths import default_data_root

REPO = default_data_root()
from fast_lto.config import RunConfig  # noqa: E402
from fast_lto.optimization.global_ocp import load_track_with_widths  # noqa: E402
from fast_lto.pipeline import _extend_track_for_autox, run_pipeline  # noqa: E402

# Every autox-style (side,cone_id,x,y L/R/M) track CSV under data/tracks/,
# excluding accidental *.csv.csv re-saves and the tiny test_pipeline.csv
# fixture.
DEFAULT_TRACK_IDS = [
    "alpnach_aug8_first_track",
    "alpnach_last_test_2",
    "alpnach_jul_3_boundary",
    "ipz_august_3",
    "izp_autox_td_aug_6",
    "fscz_2026",
    "fscz_2025",
    "lto_test_jul_3",
    "lto_comissioning_track",
    "lto_commisioning_track",
    "track_boundary_maisach",
    "track_boundary_adaptability",
    "bean",
    "ellipse",
    "fsg_random",
]

SNAP_WARN_M = 3.0


def _build_track(config) -> dict:
    run_pipeline(config, start_from="track", end_at="bounds")
    return load_track_with_widths(config.track_with_widths_path)


def _check_track(track_id: str, config_path: Path) -> bool:
    """Returns True if every check passed (or the track was skipped cleanly)."""
    print(f"\n=== {track_id} ===")
    rc = RunConfig.from_yaml(config_path)
    rc.pipeline.track_id = track_id
    try:
        rc.validate_for_model()
    except ValueError as exc:
        print(f"  SKIP: {exc}")
        return True
    config = rc.to_pipeline_config()
    config.repo_root = REPO
    config.__post_init__()

    if not config.track_csv_path.exists():
        print(f"  SKIP: {config.track_csv_path} not present in this checkout")
        return True

    try:
        track = _build_track(config)
    except Exception as exc:  # noqa: BLE001 - report and move on to the next track
        print(f"  FAILED to build track: {type(exc).__name__}: {exc}")
        return False

    positions = np.asarray(track["positions"])
    ok = True

    try:
        extended = _extend_track_for_autox(
            track,
            config.autox_extension_m,
            config.autox_lead_in_m,
            config.autox_ocp_lead_m,
            timing_offset_m=config.autox_timing_offset_m,
            start_x=config.autox_start_x,
            start_y=config.autox_start_y,
            start_node_offset=config.autox_start_node_offset,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to extend track: {type(exc).__name__}: {exc}")
        return False

    idx_ref = extended["autox_idx_ref"]
    snap_m = extended["autox_start_snap_m"]
    flag = "  <-- large snap, worth a closer look" if snap_m > SNAP_WARN_M else ""
    print(
        f"  idx_ref={idx_ref}  snap={snap_m:.2f} m from "
        f"({config.autox_start_x:.2f}, {config.autox_start_y:.2f}){flag}"
    )

    # Anchor-invariance: loop circumference / point count must match a
    # forced legacy-equivalent anchor (array index 0, no offset).
    legacy = _extend_track_for_autox(
        track,
        config.autox_extension_m,
        config.autox_lead_in_m,
        config.autox_ocp_lead_m,
        timing_offset_m=config.autox_timing_offset_m,
        start_x=float(positions[0][0]),
        start_y=float(positions[0][1]),
        start_node_offset=0,
    )
    base_len_ok = bool(np.isclose(extended["autox_base_length_m"], legacy["autox_base_length_m"]))
    n_pts_ok = extended["num_points"] == legacy["num_points"]
    print(f"  anchor-invariant: base_length_m match={base_len_ok}, " f"num_points match={n_pts_ok}")
    ok = ok and base_len_ok and n_pts_ok

    # Wrap-seam continuity: no discontinuity in consecutive-point spacing.
    ext_pos = np.asarray(extended["positions"])
    spacing = np.linalg.norm(np.diff(ext_pos, axis=0), axis=1)
    med = float(np.median(spacing))
    max_dev = float(np.max(np.abs(spacing - med))) if len(spacing) else 0.0
    seam_ok = max_dev < 0.25 * med if med > 0 else True
    print(
        f"  wrap-seam spacing: median={med:.3f} m, max deviation={max_dev:.4f} m "
        f"({'ok' if seam_ok else 'FAILED'})"
    )
    ok = ok and seam_ok

    if not ok:
        print("  FAILED one or more checks")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/autox.yaml")
    ap.add_argument(
        "--track-id",
        action="append",
        default=None,
        help="Track id to check (repeatable). Default: every autox-style "
        "track under data/tracks/.",
    )
    args = ap.parse_args()

    track_ids = args.track_id or DEFAULT_TRACK_IDS
    config_path = REPO / args.config

    results = {tid: _check_track(tid, config_path) for tid in track_ids}

    print("\n=== summary ===")
    n_fail = sum(1 for ok in results.values() if not ok)
    for tid, ok in results.items():
        print(f"  {'ok' if ok else 'FAILED':6} {tid}")
    if n_fail:
        print(f"\n{n_fail}/{len(results)} tracks failed one or more checks")
        sys.exit(1)
    print(f"\nall {len(results)} tracks passed")


if __name__ == "__main__":
    main()
