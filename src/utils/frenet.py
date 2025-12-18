"""
Frenet projection utilities for discretized tracks.

This module lets you:
- Project Cartesian points (and headings) onto a discretized centerline
- Convert back from Frenet (s, d) to Cartesian
- Visualize tangents/normals for geometry sanity checks

It can be imported or run directly:
    python src/utils/frenet.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from splines.discretized_track import DiscretizedTrack
else:
    from ..splines.discretized_track import DiscretizedTrack


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
            ratio = (s_wrapped - s0) / seg_len if s_wrapped >= s0 else (s_wrapped + self.total_length - s0) / seg_len

        pos = self.positions[idx] + ratio * (self.positions[idx_next] - self.positions[idx])
        heading = self._interp_heading(idx, ratio)
        tangent = np.array([np.cos(heading), np.sin(heading)])
        normal = np.array([-tangent[1], tangent[0]])
        curvature = self._interp_curvature(idx, ratio)

        return FrenetFrame(s=s_wrapped, position_xy=pos, tangent=tangent, normal=normal, curvature=curvature)

    def frenet_to_xy(self, s: float, d: float) -> np.ndarray:
        """Convert Frenet (s, d) to Cartesian xy."""
        frame = self.sample_at_s(s)
        return frame.position_xy + d * frame.normal

    def project_xy_to_frenet(self, xy: np.ndarray, heading: Optional[float] = None) -> FrenetProjection:
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
                ax.plot([left[0], right[0]], [left[1], right[1]], color="tab:orange", alpha=0.6, linewidth=1.0)

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


def _demo() -> None:
    """Minimal runnable example for geometry visualization."""
    repo_root = Path(__file__).resolve().parents[2]
    track_path = repo_root / "data" / "discretized" / "ellipse.json"
    track = DiscretizedTrack.load(track_path)

    processor = TrackProcessor(track, nominal_half_width=1.5)
    print(f"Loaded track: {track}")

    # Example projection near start line
    sample_xy = np.array([0.5, 0.2])
    proj = processor.project_xy_to_frenet(sample_xy)
    print(
        f"Projection -> s={proj.s:.2f} m, d={proj.d:.2f} m, "
        f"kappa={proj.kappa_s:.4f} 1/m, residual={proj.residual:.3f} m"
    )

    # Plot geometry
    processor.geometry_check_plot(every=5, normal_scale=6.0, tangent_scale=0.5, show=True)


if __name__ == "__main__":
    _demo()

