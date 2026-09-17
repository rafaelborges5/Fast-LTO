"""
The discretised centreline: one sample per OCP node.

``DiscretizedTrack`` holds what spline fitting produced -- positions, headings,
curvatures and arc lengths -- and round-trips through JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class DiscretizedTrack:
    """
    Discretized track centerline with curvature information.

    This is the output of spline fitting + arc-length discretization.
    All arrays have the same length N (number of sample points).

    Attributes
    ----------
    positions : np.ndarray
        Shape (N, 2). Cartesian coordinates [x, y] at each sample point.
    headings : np.ndarray
        Shape (N,). Tangent angle θ in radians at each sample point.
        θ = atan2(dy/ds, dx/ds), measured from +x axis.
    curvatures : np.ndarray
        Shape (N,). Signed curvature κ at each sample point.
        κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
        Positive = turning left, Negative = turning right.
    arc_lengths : np.ndarray
        Shape (N,). Cumulative arc length s at each sample point.
        Starts at 0, ends at (N-1) * ds_m.
    ds_m : float
        Discretization step in meters.
    total_length_m : float
        Total track length in meters (full lap).
    num_points : int
        Number of sample points N.
    continuity : str
        Spline continuity used for fitting: "C2" or "C4".
    source_file : str
        Path to the source track CSV file (for provenance).
    """

    positions: np.ndarray
    headings: np.ndarray
    curvatures: np.ndarray
    curvatures_half: np.ndarray
    arc_lengths: np.ndarray
    ds_m: float
    total_length_m: float
    num_points: int
    continuity: str
    source_file: str

    def __post_init__(self) -> None:
        """Validate array shapes after initialization."""
        n = self.num_points
        assert self.positions.shape == (n, 2), f"positions shape mismatch: {self.positions.shape}"
        assert self.headings.shape == (n,), f"headings shape mismatch: {self.headings.shape}"
        assert self.curvatures.shape == (n,), f"curvatures shape mismatch: {self.curvatures.shape}"
        assert self.curvatures_half.shape == (
            n,
        ), f"curvatures_half shape mismatch: {self.curvatures_half.shape}"
        assert self.arc_lengths.shape == (
            n,
        ), f"arc_lengths shape mismatch: {self.arc_lengths.shape}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "positions": self.positions.tolist(),
            "headings": self.headings.tolist(),
            "curvatures": self.curvatures.tolist(),
            "curvatures_half": self.curvatures_half.tolist(),
            "arc_lengths": self.arc_lengths.tolist(),
            "ds_m": self.ds_m,
            "total_length_m": self.total_length_m,
            "num_points": self.num_points,
            "continuity": self.continuity,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscretizedTrack:
        """Create a DiscretizedTrack from a dictionary."""
        return cls(
            positions=np.array(data["positions"], dtype=np.float64),
            headings=np.array(data["headings"], dtype=np.float64),
            curvatures=np.array(data["curvatures"], dtype=np.float64),
            curvatures_half=np.array(data["curvatures_half"], dtype=np.float64),
            arc_lengths=np.array(data["arc_lengths"], dtype=np.float64),
            ds_m=float(data["ds_m"]),
            total_length_m=float(data["total_length_m"]),
            num_points=int(data["num_points"]),
            continuity=str(data["continuity"]),
            source_file=str(data["source_file"]),
        )

    def save(self, path: str | Path) -> None:
        """
        Save the discretized track to a JSON file.

        Parameters
        ----------
        path : str | Path
            Output file path. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DiscretizedTrack:
        """
        Load a discretized track from a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.

        Returns
        -------
        DiscretizedTrack
            The loaded track data.
        """
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"DiscretizedTrack("
            f"num_points={self.num_points}, "
            f"ds_m={self.ds_m:.3f}, "
            f"total_length_m={self.total_length_m:.2f}, "
            f"continuity={self.continuity!r}, "
            f"has_curvatures_half={self.curvatures_half is not None})"
        )


__all__ = ["DiscretizedTrack"]
