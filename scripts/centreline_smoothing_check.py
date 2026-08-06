"""
How much of this track's peak curvature is real, and how much is the centreline
fit overshooting between unevenly spaced midline points?

For a range of `smooth_centerline` windows and `ds` values, refit the centreline
and report peak curvature, how many stations exceed the car's kinematic
turn-radius limit, and how far the fitted line drifts from the raw midline.

    python scripts/centreline_smoothing_check.py --csv data/tracks/ipz_august_3.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from splines.spline_fitter import fit_and_discretize  # noqa: E402
from utils.track_bounds import compute_lateral_bounds, load_boundaries  # noqa: E402


def raw_midline(csv_path: Path) -> np.ndarray:
    pts = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["side"].strip().upper() == "M":
                pts.append((float(row["x"]), float(row["y"])))
    return np.asarray(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/tracks/ipz_august_3.csv")
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 5, 7, 9, 11])
    ap.add_argument("--ds", type=float, nargs="+", default=[0.5])
    ap.add_argument("--wheelbase", type=float, default=0.842 + 0.689)
    ap.add_argument("--delta-max", type=float, default=0.4)
    args = ap.parse_args()

    csv_path = REPO / args.csv
    mid = raw_midline(csv_path)
    boundaries = load_boundaries(csv_path)
    kappa_ceiling = np.tan(args.delta_max) / args.wheelbase

    print(f"raw midline: {len(mid)} points, spacing "
          f"{np.linalg.norm(np.roll(mid,-1,axis=0)-mid,axis=1).min():.2f} .. "
          f"{np.linalg.norm(np.roll(mid,-1,axis=0)-mid,axis=1).max():.2f} m")
    print(f"kinematic curvature ceiling at delta_max={args.delta_max}: "
          f"{kappa_ceiling:.3f} 1/m (R={1/kappa_ceiling:.2f} m)\n")

    hdr = (f"{'ds':>5} {'smooth':>7} {'N':>5} {'len [m]':>8} {'|k|max':>7} "
           f"{'R_min':>6} {'#>ceil':>7} {'drift p95':>10} {'w_min':>6}")
    print(hdr)
    print("-" * len(hdr))
    for ds in args.ds:
        for w in args.windows:
            track = fit_and_discretize(csv_path, ds_m=ds, continuity="C2", viz=False,
                                       smooth_centerline=w)
            k = np.abs(np.asarray(track.curvatures))
            pos = np.asarray(track.positions)
            # drift: distance from each raw midline point to the fitted line
            drift = np.array([np.min(np.linalg.norm(pos - p, axis=1)) for p in mid])
            res = compute_lateral_bounds(track, left=boundaries["left"],
                                         right=boundaries["right"])
            width = np.asarray(res.w_left) + np.asarray(res.w_right)
            print(f"{ds:5.2f} {w:7d} {track.num_points:5d} {track.total_length_m:8.2f} "
                  f"{k.max():7.3f} {1/k.max():6.2f} {int((k>kappa_ceiling).sum()):7d} "
                  f"{np.percentile(drift,95):10.3f} {width.min():6.3f}")


if __name__ == "__main__":
    main()
