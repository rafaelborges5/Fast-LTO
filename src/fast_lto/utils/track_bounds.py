"""
Compute lateral box constraints (w_left, w_right) for a discretized track.

Inputs:
- DiscretizedTrack JSON (centerline samples with headings/normals)
- Track CSV with left/right boundary polylines (progress-ordered)

Outputs:
- Optional JSON with full track data plus widths (for downstream use)
- Optional visualization overlaying normals and intersections

Note: Current intersection search is O(N*M) over center samples (N) and
boundary segments (M). # TODO: accelerate with spatial indexing / local
windowing if needed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from fast_lto.splines.discretized_track import DiscretizedTrack


@dataclass
class LateralBoundsResult:
    w_left: np.ndarray
    w_right: np.ndarray
    misses_left: int
    misses_right: int


def _savgol_1d_periodic(
    arr: np.ndarray,
    window_length: int = 41,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Apply Savitzky–Golay smoothing to a 1D array representing a periodic signal.

    NaN entries are preserved: they are temporarily inpainted for filtering and
    then restored afterwards.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr.copy()

    if n <= polyorder + 2:
        return arr.copy()

    # need odd window length
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl <= polyorder:
        wl = polyorder + 2
        if wl % 2 == 0:
            wl += 1
        if wl > n:
            return arr.copy()

    mask = np.isfinite(arr)
    if not np.any(mask):
        return arr.copy()

    filled = arr.copy()
    if not np.all(mask):
        idx = np.arange(n)
        filled[~mask] = np.interp(
            idx[~mask],
            idx[mask],
            arr[mask],
            period=n,
        )

    smoothed = savgol_filter(filled, window_length=wl, polyorder=polyorder, mode="wrap")
    smoothed[~mask] = np.nan
    return smoothed


def apply_savgol_to_widths(
    w_left: np.ndarray,
    w_right: np.ndarray,
    window_length: int = 21,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    w_left_s = _savgol_1d_periodic(w_left, window_length=window_length, polyorder=polyorder)
    w_right_s = _savgol_1d_periodic(w_right, window_length=window_length, polyorder=polyorder)
    return w_left_s, w_right_s


_SIDE_TO_BOUNDARY: Dict[str, str] = {"L": "left", "M": "middle", "R": "right"}


def load_boundaries(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load left/right/middle polylines from the track CSV."""
    points: Dict[str, list[Tuple[int, float, float]]] = {"left": [], "middle": [], "right": []}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            boundary = _SIDE_TO_BOUNDARY.get(row["side"].strip().upper())
            if boundary is None:
                continue
            idx = int(row["cone_id"])
            x = float(row["x"])
            y = float(row["y"])
            points[boundary].append((idx, x, y))

    out: Dict[str, np.ndarray] = {}
    for key, arr in points.items():
        arr.sort(key=lambda p: p[0])
        out[key] = np.array([[p[1], p[2]] for p in arr], dtype=np.float64)
    return out


def _ray_segment_intersection(p: np.ndarray, n_hat: np.ndarray, q0: np.ndarray, q1: np.ndarray):
    """
    Solve p + t*n_hat = q0 + u*(q1-q0). Returns (t, u) or (None, None) if no hit.
    """
    v = q1 - q0
    A = np.array([[n_hat[0], -v[0]], [n_hat[1], -v[1]]], dtype=np.float64)
    b = q0 - p
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None, None
    inv_det = 1.0 / det
    t = inv_det * (b[0] * A[1, 1] - b[1] * A[0, 1])
    u = inv_det * (-b[0] * A[1, 0] + b[1] * A[0, 0])
    return t, u


def _compute_lateral_bounds_rays(
    track: DiscretizedTrack, left: np.ndarray, right: np.ndarray
) -> LateralBoundsResult:
    """
    Legacy implementation: for each center sample, intersect its normal with
    left/right boundary polylines using a ray–segment scan.

    Returns w_left (positive along +n) and w_right (positive along -n).
    """
    n_samples = track.num_points
    w_left = np.full(n_samples, np.nan, dtype=np.float64)
    w_right = np.full(n_samples, np.nan, dtype=np.float64)

    def _segments(poly: np.ndarray) -> list[Tuple[np.ndarray, np.ndarray]]:
        segs = list(zip(poly[:-1], poly[1:]))
        # Close the loop
        segs.append((poly[-1], poly[0]))
        return segs

    left_segs = _segments(left)
    right_segs = _segments(right)

    misses_left = 0
    misses_right = 0

    normals = np.column_stack((-np.sin(track.headings), np.cos(track.headings)))

    for i in range(n_samples):
        p = track.positions[i]
        n_hat = normals[i]

        # Left: t > 0
        t_min = None
        for q0, q1 in left_segs:
            t, u = _ray_segment_intersection(p, n_hat, q0, q1)
            if t is None or u is None:
                continue
            if u < -1e-9 or u > 1 + 1e-9:
                continue
            if t <= 1e-9:
                continue
            if (t_min is None) or (t < t_min):
                t_min = t
        if t_min is None:
            misses_left += 1
        else:
            w_left[i] = t_min

        # Right: t < 0 (we store positive magnitude)
        t_min = None
        for q0, q1 in right_segs:
            t, u = _ray_segment_intersection(p, n_hat, q0, q1)
            if t is None or u is None:
                continue
            if u < -1e-9 or u > 1 + 1e-9:
                continue
            if t >= -1e-9:
                continue
            if (t_min is None) or (-t < t_min):
                t_min = -t  # store as positive distance
        if t_min is None:
            misses_right += 1
        else:
            w_right[i] = t_min

    return LateralBoundsResult(
        w_left=w_left,
        w_right=w_right,
        misses_left=misses_left,
        misses_right=misses_right,
    )


def _make_spline(s: np.ndarray, d: np.ndarray):
    """
    Internal fun for 1D splines d(s).
    """
    # Cubic spline variant (C2 where possible)
    return CubicSpline(s, d, bc_type="natural")
    # PCHIP alternative (monotone, shape-preserving)
    # return PchipInterpolator(s, d)


def _project_boundary_points_to_frenet(
    boundary: np.ndarray,
    positions: np.ndarray,
    s: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    kappa: np.ndarray,
    tree: cKDTree,
    ds_max_factor: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project boundary points into local Frenet coordinates (s_cone, d_true).

    Uses a KD-tree to find the nearest centerline sample, then computes:
      - Δs along the tangent
      - raw lateral deviation along the normal
      - curvature-corrected lateral deviation d_true
    """
    if boundary.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    _, idx = tree.query(boundary)
    idx = np.asarray(idx, dtype=int)

    p_i = positions[idx]
    v = boundary - p_i

    t_i = tangents[idx]
    n_i = normals[idx]
    k_i = kappa[idx]

    delta_s = np.einsum("ij,ij->i", v, t_i)
    s_cone = s[idx] + delta_s

    d_raw = np.einsum("ij,ij->i", v, n_i)

    correction = 0.5 * k_i * (delta_s**2)  # 2nd order taylor correctoin
    d_true = d_raw - np.sign(d_raw) * correction

    ds_max = ds_max_factor * float(s[1] - s[0]) if s.size > 1 else np.inf
    mask = np.isfinite(s_cone) & np.isfinite(d_true) & (np.abs(delta_s) <= ds_max)

    return s_cone[mask], d_true[mask]


def _compute_lateral_bounds_kdtree(
    track: DiscretizedTrack, left: np.ndarray, right: np.ndarray
) -> LateralBoundsResult:
    """
    KD-tree based implementation:
      1) Map boundary points to nearest centerline samples via KD-tree.
      2) Project to local Frenet frame with curvature correction.
      3) Fit 1D splines d_left(s), d_right(s).
      4) Sample at discretized arc lengths to obtain w_left, w_right.
    """
    positions = track.positions
    s = track.arc_lengths
    headings = track.headings
    kappa = track.curvatures

    # Wrap length for the closed loop. total_length_m is the true arc length of
    # one full lap; the discretization ends one ds short of it, so this must
    # equal arc_lengths[-1] + ds. Guard against a future generator (e.g. a
    # skidpad-style open track) being routed through this closed-loop path.
    total_length = float(track.total_length_m)
    ds = float(s[1] - s[0]) if s.size > 1 else total_length
    assert abs(total_length - (float(s[-1]) + ds)) < 1e-6, (
        "compute_lateral_bounds expects a closed track where "
        "total_length_m == arc_lengths[-1] + ds "
        f"(got {total_length} vs {float(s[-1]) + ds})"
    )

    tangents = np.column_stack((np.cos(headings), np.sin(headings)))
    normals = np.column_stack((-np.sin(headings), np.cos(headings)))

    tree = cKDTree(positions)

    s_left_raw, d_left_raw = _project_boundary_points_to_frenet(
        left, positions, s, tangents, normals, kappa, tree
    )
    s_right_raw, d_right_raw = _project_boundary_points_to_frenet(
        right, positions, s, tangents, normals, kappa, tree
    )

    def _prepare_side(
        s_samples: np.ndarray, d_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if s_samples.size == 0:
            return s_samples, d_samples

        # Wrap into one period [0, L) BEFORE sort/dedup. The cone projections
        # s_cone = s[idx] + delta_s are unwrapped and can fall just outside
        # [0, L); wrapping first lets the seam neighbourhood dedup against the
        # CSV's repeated closing cone and guarantees the tiled sequence below is
        # strictly increasing.
        s_samples = np.mod(s_samples, total_length)

        order = np.argsort(s_samples)
        s_sorted = s_samples[order]
        d_sorted = d_samples[order]

        unique_s = [s_sorted[0]]
        unique_d = [d_sorted[0]]
        tol = 1e-3
        for sj, dj in zip(s_sorted[1:], d_sorted[1:]):
            if abs(sj - unique_s[-1]) <= tol:
                unique_d[-1] = 0.5 * (unique_d[-1] + dj)
            else:
                unique_s.append(sj)
                unique_d.append(dj)

        return np.asarray(unique_s, dtype=np.float64), np.asarray(unique_d, dtype=np.float64)

    s_left, d_left = _prepare_side(s_left_raw, d_left_raw)
    s_right, d_right = _prepare_side(s_right_raw, d_right_raw)

    def _tile_periodic(s_period: np.ndarray, d_period: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Tile the in-period samples over three periods (s-L, s, s+L) so the
        # natural spline is flanked by real data across the start/finish seam
        # instead of extrapolating linearly into the cone-free gap there.
        s_tiled = np.concatenate([s_period - total_length, s_period, s_period + total_length])
        d_tiled = np.concatenate([d_period, d_period, d_period])
        return s_tiled, d_tiled

    n_samples = track.num_points
    w_left = np.full(n_samples, np.nan, dtype=np.float64)
    w_right = np.full(n_samples, np.nan, dtype=np.float64)

    min_points = 3
    misses_left = 0
    misses_right = 0

    if s_left.size >= min_points:
        left_spline = _make_spline(*_tile_periodic(s_left, d_left))
        d_left_at_s = left_spline(s)
        w_left = np.maximum(d_left_at_s, 0.0)
        misses_left = int(np.count_nonzero(~np.isfinite(w_left)))
    else:
        misses_left = n_samples

    if s_right.size >= min_points:
        right_spline = _make_spline(*_tile_periodic(s_right, d_right))
        d_right_at_s = right_spline(s)
        w_right = np.maximum(-d_right_at_s, 0.0)
        misses_right = int(np.count_nonzero(~np.isfinite(w_right)))
    else:
        misses_right = n_samples

    return LateralBoundsResult(
        w_left=w_left,
        w_right=w_right,
        misses_left=misses_left,
        misses_right=misses_right,
    )


def compute_lateral_bounds(
    track: DiscretizedTrack, left: np.ndarray, right: np.ndarray
) -> LateralBoundsResult:
    # New KD-tree + spline implementation (default)
    return _compute_lateral_bounds_kdtree(track, left, right)
    # Legacy ray-based implementation:
    # return _compute_lateral_bounds_rays(track, left, right)


def save_bounds_json(
    path: Path, track: DiscretizedTrack, csv_source: Path, result: LateralBoundsResult
) -> None:
    data = {
        "source_boundaries_csv": str(csv_source),
        "discretized_track": str(path),
        "w_left": result.w_left.tolist(),
        "w_right": result.w_right.tolist(),
        "num_points": int(track.num_points),
        "misses_left": int(result.misses_left),
        "misses_right": int(result.misses_right),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_track_with_widths(
    path: Path,
    track: DiscretizedTrack,
    csv_source: Path,
    result: LateralBoundsResult,
    bounds_config: Dict | None = None,
) -> None:
    """
    Save a combined object that includes the discretized track plus lateral widths.
    """
    data = track.to_dict()
    payload: Dict = {
        "w_left": result.w_left.tolist(),
        "w_right": result.w_right.tolist(),
        "source_boundaries_csv": str(csv_source),
        "misses_left": int(result.misses_left),
        "misses_right": int(result.misses_right),
    }
    if bounds_config is not None:
        payload["bounds_config"] = bounds_config
    data.update(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_bounds(
    track: DiscretizedTrack,
    left: np.ndarray,
    right: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    every: int = 10,
    show: bool = True,
    out_path: Path | None = None,
) -> None:
    """Visualize centerline samples with normals clipped to their intersections."""
    fig, ax = plt.subplots(figsize=(8, 6))
    left_closed = np.vstack([left, left[0]])
    right_closed = np.vstack([right, right[0]])
    center_closed = np.vstack([track.positions, track.positions[0]])

    ax.plot(left_closed[:, 0], left_closed[:, 1], label="left boundary", color="tab:blue")
    ax.plot(right_closed[:, 0], right_closed[:, 1], label="right boundary", color="tab:orange")
    ax.plot(center_closed[:, 0], center_closed[:, 1], label="centerline", color="k", linewidth=1.2)

    normals = np.column_stack((-np.sin(track.headings), np.cos(track.headings)))
    stride_indices = np.arange(0, track.num_points, every)
    for i in stride_indices:
        p = track.positions[i]
        n_hat = normals[i]
        if np.isfinite(w_left[i]):
            left_pt = p + w_left[i] * n_hat
            ax.plot([p[0], left_pt[0]], [p[1], left_pt[1]], color="tab:green", linewidth=1.0)
            ax.scatter(left_pt[0], left_pt[1], color="tab:green", s=10)
        if np.isfinite(w_right[i]):
            right_pt = p - w_right[i] * n_hat
            ax.plot([p[0], right_pt[0]], [p[1], right_pt[1]], color="tab:red", linewidth=1.0)
            ax.scatter(right_pt[0], right_pt[1], color="tab:red", s=10)
        ax.scatter(p[0], p[1], color="k", s=5)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.set_title("Lateral bounds per center sample")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _demo() -> None:
    """Minimal runnable example for computing and visualizing bounds."""
    repo_root = Path(__file__).resolve().parents[3]
    track_path = repo_root / "data" / "discretized" / "fsg_random.json"
    csv_path = repo_root / "data" / "tracks" / "fsg_random.csv"

    track = DiscretizedTrack.load(track_path)
    boundaries = load_boundaries(csv_path)
    left = boundaries["left"]
    right = boundaries["right"]

    result = compute_lateral_bounds(track, left=left, right=right)
    print(f"Misses left/right: {result.misses_left} / {result.misses_right}")
    out_json = repo_root / "data" / "discretized" / "fsg_random_with_widths.json"
    save_track_with_widths(out_json, track=track, csv_source=csv_path, result=result)
    print(f"Saved bounds + track to {out_json}")

    out_plot = repo_root / "out" / "lateral_bounds.png"
    plot_bounds(track, left, right, result.w_left, result.w_right, every=10, show=True)
    print(f"Saved plot to {out_plot}")


if __name__ == "__main__":
    _demo()
