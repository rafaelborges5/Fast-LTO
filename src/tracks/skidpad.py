"""
Skidpad track generator for partially-timed lap-time optimization.

Unlike the other generators in this package (which emit cone boundaries that are
later spline-fit and bounds-computed), this module builds the *discretized
track-with-widths dict* directly — the same schema produced by
``DiscretizedTrack.to_dict()`` plus ``w_left``/``w_right`` — because the skidpad
maneuver self-overlaps in XY (each circle is driven twice) and therefore cannot
go through the generic single-pass spline/KD-tree bounds machinery.

The whole maneuver is unrolled along arc length ``s`` as one centerline:

    entry straight -> right circle x2 -> left circle x2 -> exit straight

so the OCP treats it like any other track. Two extra fields are added:

    timed_mask : per-point 0/1 flag marking the intervals that are *scored*
                 (the 2nd revolution on each circle, i.e. the FS timed laps).
    skidpad    : geometry metadata (centers, radii, gate, lap structure).

Inputs
------
map_csv : cone map, columns ``tag,x,y`` (unlabeled cones, two rings per circle).
ref_csv : a controller-reference trajectory (``export.trajectory.CSV_COLUMNS``
          layout); used to fit the circle centers, anchor the entry/exit
          endpoints, and establish the lap order.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_skidpad_cones(map_csv: str | Path) -> np.ndarray:
    """Read an ``tag,x,y`` cone map into an (M, 2) array of [x, y]."""
    pts: List[Tuple[float, float]] = []
    with Path(map_csv).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"])))
    return np.array(pts, dtype=np.float64)


def _load_reference_xy(ref_csv: str | Path) -> np.ndarray:
    """Read the controller-reference CSV and return its (N, 2) path."""
    rows = list(csv.DictReader(Path(ref_csv).open("r", newline="")))
    return np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)



def _circle_fit(P: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares circle fit. Returns (center[2], radius)."""
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    return np.array([cx, cy]), float(np.sqrt(c0 + cx**2 + cy**2))


def _signed_curvature(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete signed curvature dpsi/ds and cumulative arc length of a path."""
    d1 = np.gradient(xy, axis=0)
    heading = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    dpsi = np.gradient(heading) / np.maximum(np.gradient(s), 1e-9)
    return dpsi, s


def detect_skidpad_geometry(
    cones: np.ndarray, ref_xy: np.ndarray, kappa_thresh: float = 0.02
) -> Dict:
    """
    Infer the skidpad geometry from the reference path and the cone map.

    Returns a dict with snapped, symmetric geometry:
      c_first, c_second : (2,) circle centers in traversal order
      sign_first        : +1 if the first circle is a left turn, -1 if right
      gate              : (2,) tangent/timing point = midpoint of the centers
      R_c               : centerline radius
      R_in, R_out       : inner / outer cone-ring radii
      P0, P1            : entry start / exit end points (from the reference)
    """
    dpsi, _ = _signed_curvature(ref_xy)
    left_pts = ref_xy[dpsi > kappa_thresh]
    right_pts = ref_xy[dpsi < -kappa_thresh]
    c_left, _ = _circle_fit(left_pts)
    c_right, _ = _circle_fit(right_pts)

    gate = 0.5 * (c_left + c_right)
    R_c = 0.5 * float(np.linalg.norm(c_left - c_right))
    axis = (c_left - c_right) / np.linalg.norm(c_left - c_right)
    normal = np.array([axis[1], -axis[0]])
    c_left_snap = gate + R_c * axis
    c_right_snap = gate - R_c * axis

    dist = np.minimum(
        np.linalg.norm(cones - c_left_snap, axis=1),
        np.linalg.norm(cones - c_right_snap, axis=1),
    )
    inner = dist[dist < R_c]
    outer = dist[(dist >= R_c) & (dist < 1.3 * R_c)]
    R_in = float(np.median(inner))
    R_out = float(np.median(outer))

    R_c = (R_in + R_out) / 2.0
    c_left_snap = gate + R_c * axis
    c_right_snap = gate - R_c * axis

    first_curved = dpsi[np.abs(dpsi) > kappa_thresh]
    sign_first = 1.0 if (first_curved.size and first_curved[0] > 0) else -1.0
    if sign_first > 0:
        c_first, c_second = c_left_snap, c_right_snap
    else:
        c_first, c_second = c_right_snap, c_left_snap

    P0 = ref_xy[0].copy()
    P1 = ref_xy[-1].copy()

    return {
        "c_first": c_first,
        "c_second": c_second,
        "sign_first": sign_first,
        "gate": gate,
        "axis": normal,
        "R_c": R_c,
        "R_in": R_in,
        "R_out": R_out,
        "P0": P0,
        "P1": P1,
    }


class _Segment:
    """One centerline segment with constant curvature, parameterized by arc t."""

    def __init__(self, length: float, kappa: float, timed: bool):
        self.length = float(length)
        self.kappa = float(kappa)
        self.timed = bool(timed)

    def eval(self, t: float) -> Tuple[float, float, float]:
        raise NotImplementedError


class _Straight(_Segment):
    def __init__(self, A: np.ndarray, B: np.ndarray, timed: bool = False):
        length = float(np.linalg.norm(B - A))
        super().__init__(length, 0.0, timed)
        self.A = A
        self.dir = (B - A) / length
        self.heading = float(np.arctan2(self.dir[1], self.dir[0]))

    def eval(self, t: float) -> Tuple[float, float, float]:
        p = self.A + self.dir * t
        return float(p[0]), float(p[1]), self.heading


class _Arc(_Segment):
    def __init__(self, center: np.ndarray, R: float, theta0: float,
                 d_theta: float, timed: bool = False):
        length = R * abs(d_theta)
        super().__init__(length, np.sign(d_theta) / R, timed)
        self.center = center
        self.R = R
        self.theta0 = theta0
        self.sgn = float(np.sign(d_theta))

    def eval(self, t: float) -> Tuple[float, float, float]:
        theta = self.theta0 + self.sgn * (t / self.R)
        x = self.center[0] + self.R * np.cos(theta)
        y = self.center[1] + self.R * np.sin(theta)
        heading = theta + self.sgn * (np.pi / 2.0)
        return float(x), float(y), float(heading)


def _build_segments(geo: Dict, n_laps: int, timed_lap_index: int) -> List[_Segment]:
    """Assemble the ordered segment list: entry -> first x2 -> second x2 -> exit."""
    R_c = geo["R_c"]
    gate = geo["gate"]
    segs: List[_Segment] = []

    segs.append(_Straight(geo["P0"], gate))
    for center, sign in ((geo["c_first"], geo["sign_first"]),
                         (geo["c_second"], -geo["sign_first"])):
        theta0 = float(np.arctan2(gate[1] - center[1], gate[0] - center[0]))
        for lap in range(1, n_laps + 1):
            timed = (lap == timed_lap_index)
            segs.append(_Arc(center, R_c, theta0, sign * 2 * np.pi, timed=timed))
    segs.append(_Straight(gate, geo["P1"]))
    return segs


def _linear_blend(s: np.ndarray, s_node: float, kappa_node: np.ndarray,
                  joints: List[Tuple[float, float, float]], half: float) -> np.ndarray:
    """Apply linear ramps to a piecewise-constant signal across joints.

    joints: list of (s_joint, value_left, value_right). Within +/- half of each
    joint the signal ramps linearly from value_left to value_right.
    """
    out = kappa_node.copy()
    for s_j, vl, vr in joints:
        lo, hi = s_j - half, s_j + half
        m = (s >= lo) & (s <= hi)
        if half > 0:
            frac = (s[m] - lo) / (2 * half)
            out[m] = vl + (vr - vl) * frac
    return out


def build_skidpad_track(
    map_csv: str | Path,
    ref_csv: str | Path,
    ds_m: float = 0.5,
    entry_exit_halfwidth: float = 1.5,
    n_laps_per_side: int = 2,
    timed_lap_index: int = 2,
    kappa_blend_m: float = 1.5,
) -> Dict:
    """Build the skidpad track-with-widths dict (see module docstring)."""
    cones = load_skidpad_cones(map_csv)
    ref_xy = _load_reference_xy(ref_csv)
    geo = detect_skidpad_geometry(cones, ref_xy)

    segs = _build_segments(geo, n_laps_per_side, timed_lap_index)

    bounds = np.concatenate([[0.0], np.cumsum([s.length for s in segs])])
    total_length = float(bounds[-1])
    N = int(round(total_length / ds_m)) + 1
    s_nodes = np.linspace(0.0, total_length, N)
    ds_actual = total_length / (N - 1)

    R_c = geo["R_c"]
    w_out = geo["R_out"] - R_c
    w_in  = R_c - geo["R_in"]
    entry_exit_halfwidth = w_out

    positions = np.zeros((N, 2))
    headings = np.zeros(N)
    kappa_raw = np.zeros(N)
    wl_raw = np.zeros(N)
    wr_raw = np.zeros(N)
    timed_mask = np.zeros(N, dtype=int)
    decel_mask = np.zeros(N, dtype=int)
    exit_seg_idx = len(segs) - 1

    def _locate(s_val: float) -> Tuple[int, float]:
        k = int(np.searchsorted(bounds, s_val, side="right") - 1)
        k = min(max(k, 0), len(segs) - 1)
        return k, s_val - bounds[k]

    for i, s_val in enumerate(s_nodes):
        k, t = _locate(s_val)
        seg = segs[k]
        x, y, h = seg.eval(min(t, seg.length))
        positions[i] = (x, y)
        headings[i] = h
        kappa_raw[i] = seg.kappa
        timed_mask[i] = 1 if seg.timed else 0
        decel_mask[i] = 1 if k == exit_seg_idx else 0
        if isinstance(seg, _Straight):
            wl_raw[i] = entry_exit_halfwidth
            wr_raw[i] = entry_exit_halfwidth
        else:
            if seg.kappa < 0:
                wl_raw[i], wr_raw[i] = w_out, w_in
            else:
                wl_raw[i], wr_raw[i] = w_in, w_out

    joints_k = list(bounds[1:-1])
    half = 0.5 * kappa_blend_m
    kappa_joints = []
    wl_joints = []
    wr_joints = []
    for s_j in joints_k:
        kl, _ = _locate(s_j - 1e-6)
        kr, _ = _locate(s_j + 1e-6)
        kappa_joints.append((s_j, segs[kl].kappa, segs[kr].kappa))
        wl_left = entry_exit_halfwidth if isinstance(segs[kl], _Straight) else (
            w_out if segs[kl].kappa < 0 else w_in)
        wl_right = entry_exit_halfwidth if isinstance(segs[kr], _Straight) else (
            w_out if segs[kr].kappa < 0 else w_in)
        wr_left = entry_exit_halfwidth if isinstance(segs[kl], _Straight) else (
            w_in if segs[kl].kappa < 0 else w_out)
        wr_right = entry_exit_halfwidth if isinstance(segs[kr], _Straight) else (
            w_in if segs[kr].kappa < 0 else w_out)
        wl_joints.append((s_j, wl_left, wl_right))
        wr_joints.append((s_j, wr_left, wr_right))

    curvatures = _linear_blend(s_nodes, ds_actual, kappa_raw, kappa_joints, half)
    curvatures_half = _linear_blend(
        s_nodes + ds_actual / 2.0, ds_actual, kappa_raw, kappa_joints, half
    )
    w_left = _linear_blend(s_nodes, ds_actual, wl_raw, wl_joints, half)
    w_right = _linear_blend(s_nodes, ds_actual, wr_raw, wr_joints, half)

    return {
        "positions": positions.tolist(),
        "headings": headings.tolist(),
        "curvatures": curvatures.tolist(),
        "curvatures_half": curvatures_half.tolist(),
        "arc_lengths": s_nodes.tolist(),
        "ds_m": ds_actual,
        "total_length_m": total_length,
        "num_points": N,
        "continuity": "analytic",
        "source_file": str(map_csv),
        "w_left": w_left.tolist(),
        "w_right": w_right.tolist(),
        "timed_mask": timed_mask.tolist(),
        "decel_mask": decel_mask.tolist(),
        "misses_left": 0,
        "misses_right": 0,
        "skidpad": {
            "c_first": geo["c_first"].tolist(),
            "c_second": geo["c_second"].tolist(),
            "gate": geo["gate"].tolist(),
            "sign_first": geo["sign_first"],
            "R_c": geo["R_c"],
            "R_in": geo["R_in"],
            "R_out": geo["R_out"],
            "P0": geo["P0"].tolist(),
            "P1": geo["P1"].tolist(),
            "n_laps_per_side": n_laps_per_side,
            "timed_lap_index": timed_lap_index,
            "segment_bounds_m": bounds.tolist(),
            "entry_exit_halfwidth": entry_exit_halfwidth,
            "kappa_blend_m": kappa_blend_m,
        },
    }


__all__ = ["build_skidpad_track", "load_skidpad_cones", "detect_skidpad_geometry"]
