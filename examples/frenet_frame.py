"""The Frenet frame the optimizer works in, and where it breaks down.

Every vehicle model in this repo describes the car in curvilinear coordinates
along the track centerline rather than in the world frame:

    s        arc length travelled along the centerline
    d        signed lateral offset from it, positive to the left
    psi_err  heading of the car relative to the centerline tangent

That change of variables is what makes a minimum-time lap tractable: the track
corridor becomes a plain box constraint on ``d``, and progress along the lap is
a state rather than something to be recovered from (x, y). This example builds
the frame from a discretized track and shows both directions of the mapping --
``frenet_to_xy`` and ``project_xy_to_frenet``.

The mapping is not global, and the failure is worth seeing. A point at lateral
offset ``d`` on a centerline of curvature ``kappa`` has a Jacobian determinant

    D_kappa = 1 - kappa * d

so the frame degenerates as ``d`` approaches ``1 / kappa`` -- the centre of the
osculating circle, where every normal line meets and (s, d) stops being unique.
``visualize_frenet_singularities`` draws the normals so you can see them cross.
This is the quantity the models guard with ``eps_D_kappa``, and the reason a
tight corner plus a wide corridor is a genuinely harder problem than either
alone.

Run it::

    python examples/frenet_frame.py                  # generates a track
    python examples/frenet_frame.py --track-csv data/tracks/fsg_random.csv
    python examples/frenet_frame.py --out frenet.png # headless

This file is an example, not library code: it lives outside ``src/`` so it is
never packaged, and nothing in ``fast_lto`` imports it. The pipeline's own
Cartesian-to-Frenet projection is the cKDTree one in ``utils.track_bounds``,
which is built for a different job -- resampling boundary polylines onto the
centerline, at speed.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from fast_lto.splines.discretized_track import DiscretizedTrack


def _wrap_angle(angle: float) -> float:
    """Normalize angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FrenetFrame:
    s: float
    position_xy: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    curvature: float


@dataclass
class FrenetProjection:
    s: float
    d: float
    psi_err: Optional[float]
    kappa_s: float
    proj_xy: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    residual: float
    idx: int
    inside_bounds: Optional[bool] = None


class TrackProcessor:
    """Helper for Cartesian<->Frenet conversions on a discretized centerline."""

    def __init__(self, track: DiscretizedTrack, nominal_half_width: Optional[float] = None) -> None:
        self.track = track
        self.nominal_half_width = nominal_half_width

        self.positions = track.positions
        self.headings = track.headings
        self.curvatures = track.curvatures
        self.arc_lengths = track.arc_lengths
        self.total_length = float(track.total_length_m)
        self.ds = float(track.ds_m)
        self.n = int(track.num_points)

        self.tangents = np.column_stack((np.cos(self.headings), np.sin(self.headings)))
        self.normals = np.column_stack((-np.sin(self.headings), np.cos(self.headings)))

    def _interp_heading(self, idx: int, ratio: float) -> float:
        h0 = self.headings[idx]
        h1 = self.headings[(idx + 1) % self.n]
        delta = _wrap_angle(h1 - h0)
        return h0 + ratio * delta

    def _interp_curvature(self, idx: int, ratio: float) -> float:
        k0 = self.curvatures[idx]
        k1 = self.curvatures[(idx + 1) % self.n]
        return float(k0 + ratio * (k1 - k0))

    def _segment_arc_length(self, idx: int) -> float:
        """Arc length of segment idx -> idx+1 with wrap at the end."""
        if idx == self.n - 1:
            return self.total_length - self.arc_lengths[idx]
        return self.arc_lengths[idx + 1] - self.arc_lengths[idx]

    def sample_at_s(self, s: float) -> FrenetFrame:
        """Interpolate centerline pose at arc length s."""
        s_wrapped = s % self.total_length
        idx = int(np.floor(s_wrapped / self.ds)) % self.n
        idx_next = (idx + 1) % self.n

        s0 = self.arc_lengths[idx]
        seg_len = self._segment_arc_length(idx)
        if seg_len <= 1e-9:
            ratio = 0.0
        else:
            ratio = (
                (s_wrapped - s0) / seg_len
                if s_wrapped >= s0
                else (s_wrapped + self.total_length - s0) / seg_len
            )

        pos = self.positions[idx] + ratio * (self.positions[idx_next] - self.positions[idx])
        heading = self._interp_heading(idx, ratio)
        tangent = np.array([np.cos(heading), np.sin(heading)])
        normal = np.array([-tangent[1], tangent[0]])
        curvature = self._interp_curvature(idx, ratio)

        return FrenetFrame(
            s=s_wrapped, position_xy=pos, tangent=tangent, normal=normal, curvature=curvature
        )

    def frenet_to_xy(self, s: float, d: float) -> np.ndarray:
        """Convert Frenet (s, d) to Cartesian xy."""
        frame = self.sample_at_s(s)
        return frame.position_xy + d * frame.normal

    def project_xy_to_frenet(
        self, xy: np.ndarray, heading: Optional[float] = None
    ) -> FrenetProjection:
        """Project a Cartesian point (and optional heading) to Frenet coordinates."""
        p = np.asarray(xy, dtype=np.float64)

        # Coarse nearest point
        diffs = self.positions - p
        idx = int(np.argmin(np.sum(diffs**2, axis=1)))
        idx_next = (idx + 1) % self.n

        p0 = self.positions[idx]
        p1 = self.positions[idx_next]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))

        if seg_len_sq < 1e-12:
            t = 0.0
            proj = p0
        else:
            t = float(np.clip(np.dot(p - p0, seg) / seg_len_sq, 0.0, 1.0))
            proj = p0 + t * seg

        seg_arc = self._segment_arc_length(idx)
        s = self.arc_lengths[idx] + t * seg_arc
        s_wrapped = s % self.total_length

        heading_interp = self._interp_heading(idx, t)
        tangent = np.array([np.cos(heading_interp), np.sin(heading_interp)])
        normal = np.array([-tangent[1], tangent[0]])

        d = float(np.dot(p - proj, normal))
        residual = float(np.linalg.norm(p - proj))
        psi_err = float(_wrap_angle(heading - heading_interp)) if heading is not None else None
        kappa_s = self._interp_curvature(idx, t)

        return FrenetProjection(
            s=s_wrapped,
            d=d,
            psi_err=psi_err,
            kappa_s=kappa_s,
            proj_xy=proj,
            tangent=tangent,
            normal=normal,
            residual=residual,
            idx=idx,
            inside_bounds=None,
        )

    def geometry_check_plot(
        self,
        every: int = 10,
        normal_scale: float = 0.5,
        tangent_scale: float = 0.5,
        show: bool = True,
        out_path: Optional[Path] = None,
    ) -> None:
        """
        Plot centerline with normals (and optional nominal bounds) for sanity checks.

        Parameters
        ----------
        every : int
            Plot every N-th vector to avoid clutter.
        normal_scale : float
            Length of normal arrows.
        tangent_scale : float
            Length of tangent arrows.
        show : bool
            Whether to display the plot in a window.
        out_path : Path | None
            If provided, save the figure to this path.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.positions[:, 0], self.positions[:, 1], label="centerline")

        stride_indices = np.arange(0, self.n, every)
        for i in stride_indices:
            p = self.positions[i]
            t_hat = self.tangents[i]
            n_hat = self.normals[i]
            ax.arrow(
                p[0],
                p[1],
                tangent_scale * t_hat[0],
                tangent_scale * t_hat[1],
                head_width=0.08 * tangent_scale,
                color="tab:green",
                length_includes_head=True,
            )
            ax.arrow(
                p[0],
                p[1],
                normal_scale * n_hat[0],
                normal_scale * n_hat[1],
                head_width=0.08 * normal_scale,
                color="tab:red",
                length_includes_head=True,
            )

            if self.nominal_half_width is not None:
                left = p + self.nominal_half_width * n_hat
                right = p - self.nominal_half_width * n_hat
                ax.plot(
                    [left[0], right[0]],
                    [left[1], right[1]],
                    color="tab:orange",
                    alpha=0.6,
                    linewidth=1.0,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Frenet geometry check")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200)
        if show:
            plt.show()
        plt.close(fig)

    def visualize_frenet_singularities(
        self,
        w_left: Optional[np.ndarray] = None,
        w_right: Optional[np.ndarray] = None,
        left_boundary: Optional[np.ndarray] = None,
        right_boundary: Optional[np.ndarray] = None,
        normal_length_m: float = 3.0,
        every: int = 5,
        singularity_threshold: float = 0.4,
        show: bool = True,
        out_path: Optional[Path] = None,
    ) -> None:
        """
        Visualize Frenet geometry with normal lines and track bounds, highlighting singularities.

        Parameters
        ----------
        w_left : np.ndarray | None
            Left half-widths at each sample point (positive along +normal).
            If None, uses nominal_half_width if available.
        w_right : np.ndarray | None
            Right half-widths at each sample point (positive along -normal).
            If None, uses nominal_half_width if available.
        left_boundary : np.ndarray | None
            (N, 2) array of left boundary points. If provided, plots the boundary.
        right_boundary : np.ndarray | None
            (N, 2) array of right boundary points. If provided, plots the boundary.
        normal_length_m : float
            Length of normal lines in meters on each side of centerline (default: 3.0).
        every : int
            Plot every N-th normal line to avoid clutter (default: 5).
        singularity_threshold : float
            Threshold for singularity detection: min(width) / radius_of_curvature.
            Values above this indicate potential Frenet singularities (default: 0.4).
        show : bool
            Whether to display the plot in a window.
        out_path : Path | None
            If provided, save the figure to this path.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        centerline_closed = np.vstack([self.positions, self.positions[0]])
        ax.plot(
            centerline_closed[:, 0],
            centerline_closed[:, 1],
            label="centerline",
            color="tab:blue",
            linewidth=2.0,
        )

        if left_boundary is not None:
            left_closed = np.vstack([left_boundary, left_boundary[0]])
            ax.plot(
                left_closed[:, 0],
                left_closed[:, 1],
                label="left boundary",
                color="tab:gray",
                linewidth=1.5,
                linestyle="--",
            )
        if right_boundary is not None:
            right_closed = np.vstack([right_boundary, right_boundary[0]])
            ax.plot(
                right_closed[:, 0],
                right_closed[:, 1],
                label="right boundary",
                color="tab:gray",
                linewidth=1.5,
                linestyle="--",
            )

        if w_left is None:
            w_left = (
                np.full(self.n, self.nominal_half_width)
                if self.nominal_half_width is not None
                else np.full(self.n, normal_length_m)
            )
        if w_right is None:
            w_right = (
                np.full(self.n, self.nominal_half_width)
                if self.nominal_half_width is not None
                else np.full(self.n, normal_length_m)
            )

        # Detect singularities: where min(width) / radius_of_curvature > threshold
        # Radius of curvature R = 1 / |kappa| (for kappa != 0)
        abs_kappa = np.abs(self.curvatures)
        radius = np.where(abs_kappa > 1e-9, 1.0 / abs_kappa, np.inf)
        min_width = np.minimum(w_left, w_right)
        singularity_ratio = np.where(radius < np.inf, min_width / radius, 0.0)
        is_singular = singularity_ratio > singularity_threshold

        stride_indices = np.arange(0, self.n, every)
        normal_lines_regular = []
        normal_lines_singular = []

        for i in stride_indices:
            p = self.positions[i]
            n_hat = self.normals[i]

            p_left = p + normal_length_m * n_hat
            p_right = p - normal_length_m * n_hat

            line = np.array([[p_left[0], p_left[1]], [p_right[0], p_right[1]]])

            if is_singular[i]:
                normal_lines_singular.append(line)
            else:
                normal_lines_regular.append(line)

        if normal_lines_regular:
            lc_regular = LineCollection(
                normal_lines_regular, colors="tab:red", linewidths=1.0, alpha=0.6
            )
            ax.add_collection(lc_regular)

        if normal_lines_singular:
            lc_singular = LineCollection(
                normal_lines_singular,
                colors="tab:purple",
                linewidths=2.5,
                alpha=0.9,
                label=f"near-singular (ratio > {singularity_threshold:.2f})",
            )
            ax.add_collection(lc_singular)

        singular_indices = stride_indices[is_singular[stride_indices]]
        if len(singular_indices) > 0:
            ax.scatter(
                self.positions[singular_indices, 0],
                self.positions[singular_indices, 1],
                color="tab:purple",
                s=50,
                marker="o",
                edgecolors="black",
                linewidths=1.0,
                zorder=10,
                label=f"singular points ({len(singular_indices)} shown)",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Frenet geometry check with singularities\n"
            f"Normal lines: ±{normal_length_m}m | "
            f"Singularity threshold: {singularity_threshold:.2f}"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

        num_singular = np.sum(is_singular)
        if num_singular > 0:
            max_ratio = np.max(singularity_ratio)
            print(
                f"Frenet singularity detection: {num_singular}/{self.n} points "
                f"({100*num_singular/self.n:.1f}%) flagged as near-singular"
            )
            print(f"  Max singularity ratio: {max_ratio:.3f}")
            print(f"  Threshold: {singularity_threshold:.2f}")

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def _load_track(track_csv: Optional[Path], ds_m: float):
    """A discretized track plus its corridor, from a CSV or freshly generated.

    Returns ``(track, w_left, w_right, left_boundary, right_boundary)``.
    """
    from fast_lto.splines.spline_fitter import fit_and_discretize
    from fast_lto.tracks.fsg_trackdrive import generate_fsg_track
    from fast_lto.utils.track_bounds import compute_lateral_bounds, load_boundaries

    if track_csv is not None:
        print(f"Reading boundaries from {track_csv}")
        boundaries = load_boundaries(track_csv)
    else:
        # No input asked for, so make one. Keeps the example runnable in a
        # fresh clone, where data/ holds only what ships.
        print("No --track-csv given; generating a random FSG-style track")
        csv_path = Path(tempfile.mkdtemp()) / "frenet_example.csv"
        boundaries = generate_fsg_track(output_csv=csv_path)
        track_csv = csv_path

    left = boundaries["left"]
    right = boundaries["right"]

    track = fit_and_discretize(track_csv, ds_m=ds_m, continuity="C2")
    print(f"Discretized track: {track}")

    bounds = compute_lateral_bounds(track, left=left, right=right)
    print(f"Corridor widths: misses left/right {bounds.misses_left}/{bounds.misses_right}")

    return track, bounds.w_left, bounds.w_right, left, right


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--track-csv",
        type=Path,
        default=None,
        help="Boundary CSV to read. Default: generate a random FSG-style track.",
    )
    parser.add_argument(
        "--ds",
        type=float,
        default=4.0,
        help="Centerline discretization step in metres. Coarse by default so the "
        "normals stay far enough apart to read. Default: 4.0",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save the figure here instead of opening a window.",
    )
    parser.add_argument(
        "--singularity-threshold",
        type=float,
        default=0.4,
        help="Flag a station once |kappa * d| exceeds this fraction. Default: 0.4",
    )
    args = parser.parse_args(argv)

    track, w_left, w_right, left, right = _load_track(args.track_csv, args.ds)

    processor = TrackProcessor(track)

    # Both directions of the mapping, on a point placed half a metre to the
    # left of station 10: project it back and the offset should come out as the
    # half metre it was built from.
    offset_m = 0.5
    sample_xy = track.positions[10] + offset_m * processor.normals[10]
    projected = processor.project_xy_to_frenet(sample_xy)
    print(
        f"\nRound trip: placed a point {offset_m} m left of station 10, "
        f"projected back to s={projected.s:.2f} m, d={projected.d:.3f} m "
        f"(kappa={projected.kappa_s:+.4f} 1/m, residual={projected.residual:.3e} m)"
    )

    print("\nDrawing the frame and its singularities...")
    processor.visualize_frenet_singularities(
        w_left=w_left,
        w_right=w_right,
        left_boundary=left,
        right_boundary=right,
        normal_length_m=6.0,
        every=3,
        singularity_threshold=args.singularity_threshold,
        show=args.out is None,
        out_path=args.out,
    )
    if args.out is not None:
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
