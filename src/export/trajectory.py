"""
Export OCP solution as a CSV matching the ControllerReferenceTrajectory ROS message.

Produces one row per discretization point with the columns expected by the car's
trajectory-tracking controller.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

CSV_COLUMNS = [
    "x",
    "y",
    "z",
    "boundary_left",
    "boundary_right",
    "velocity_long_ref",
    "kappa",
    "yaw_angle",
    "yaw_angle_error",
    "lat_deviation",
    "arc_progress",
    "time",
    "yaw_angle_dot",
    "velocity_lat",
    "acceleration_long",
    "acceleration_lat",
    "force_long_fl",
    "force_long_fr",
    "force_long_rl",
    "force_long_rr",
    "steering_angle",
    "steering_angle_dot",
]


def _compute_path_geometry(
    path_xy: np.ndarray,
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a C2 cubic spline to a path and return heading + curvature.

    Parameters
    ----------
    path_xy : np.ndarray
        Shape (N, 2) of [x, y] positions.
    periodic : bool
        If True, treat as closed loop (periodic spline). If False, open path.

    Returns
    -------
    headings : np.ndarray
        Shape (N,) tangent heading angles in radians.
    curvatures : np.ndarray
        Shape (N,) signed curvatures.
    """
    x = path_xy[:, 0]
    y = path_xy[:, 1]

    diffs = np.diff(path_xy, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(x))
    t[1:] = np.cumsum(segment_lengths)

    if periodic:
        wrap_distance = np.linalg.norm(path_xy[0] - path_xy[-1])
        t_periodic = np.append(t, t[-1] + wrap_distance)
        x_periodic = np.append(x, x[0])
        y_periodic = np.append(y, y[0])
        spline_x = CubicSpline(t_periodic, x_periodic, bc_type="periodic")
        spline_y = CubicSpline(t_periodic, y_periodic, bc_type="periodic")
    else:
        spline_x = CubicSpline(t, x, bc_type="not-a-knot")
        spline_y = CubicSpline(t, y, bc_type="not-a-knot")

    dx_dt = spline_x(t, 1)
    dy_dt = spline_y(t, 1)
    d2x_dt2 = spline_x(t, 2)
    d2y_dt2 = spline_y(t, 2)

    headings = np.arctan2(dy_dt, dx_dt)
    numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
    denominator = (dx_dt**2 + dy_dt**2) ** 1.5
    curvatures = numerator / denominator

    return headings, curvatures


def _finite_diff_periodic(arr: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Forward finite difference for a periodic signal.

    Uses forward differences everywhere; the last element wraps to the first.
    ``dt[i]`` is the time step from point *i* to point *i+1*.
    """
    # Shifted array: arr[1], arr[2], ..., arr[0]
    arr_next = np.roll(arr, -1)
    delta = arr_next - arr
    safe_dt = np.where(dt != 0, dt, 1.0)
    diff = np.where(dt != 0, delta / safe_dt, 0.0)
    return diff


def _finite_diff_open(arr: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Forward finite difference for an open (non-periodic) signal.

    Last element copies the previous derivative (backward difference).
    """
    diff = np.zeros_like(arr)
    safe_dt = np.where(dt != 0, dt, 1.0)
    diff[:-1] = np.where(dt[:-1] != 0, (arr[1:] - arr[:-1]) / safe_dt[:-1], 0.0)
    diff[-1] = diff[-2] if len(arr) > 1 else 0.0
    return diff


def export_reference_trajectory(solution_path: Path | str, output_path: Path | str) -> Path:
    """Read an OCP solution JSON and write a controller-reference CSV.

    Parameters
    ----------
    solution_path : Path
        Path to the ``*.json`` solution produced by :func:`solve_ocp_and_save`.
    output_path : Path
        Destination CSV path.

    Returns
    -------
    Path
        The written CSV path (same as *output_path*).
    """
    solution_path = Path(solution_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with solution_path.open("r") as f:
        data = json.load(f)

    run_config = data["run_config"]
    model_name = run_config["model_name"]
    mode = run_config.get("mode", "trackdrive")
    periodic = mode == "trackdrive"
    params = data.get("model_params", {})

    # -- Common arrays --
    path_xy = np.array(data["path_xy"], dtype=np.float64)
    x = path_xy[:, 0]
    y = path_xy[:, 1]
    N = len(x)

    z = np.zeros(N, dtype=np.float64)

    w_left = np.array(data["w_left"], dtype=np.float64)
    w_right = np.array(data["w_right"], dtype=np.float64)

    v = np.array(data.get("v", data.get("v_long")), dtype=np.float64)
    headings = np.array(data["headings"], dtype=np.float64)
    psi_err = np.array(data["psi_err"], dtype=np.float64)
    d = np.array(data["d"], dtype=np.float64)

    # -- Renormalize: make optimal path the new reference --
    opt_headings, opt_kappa = _compute_path_geometry(path_xy, periodic=periodic)

    boundary_left = w_left - d
    boundary_right = -(w_right + d)

    vehicle_heading = headings + psi_err
    psi_err = (vehicle_heading - opt_headings + np.pi) % (2 * np.pi) - np.pi

    kappa = opt_kappa
    headings = opt_headings
    d = np.zeros(N, dtype=np.float64)

    # -- Arc lengths along the optimal path --
    path_diffs = np.diff(path_xy, axis=0)
    segment_lengths = np.linalg.norm(path_diffs, axis=1)
    arc_lengths = np.zeros(N, dtype=np.float64)
    arc_lengths[1:] = np.cumsum(segment_lengths)

    # -- Arc-length steps --
    if periodic:
        ds = np.diff(arc_lengths, append=arc_lengths[0])
        wrap_len = np.linalg.norm(path_xy[0] - path_xy[-1])
        ds[-1] = wrap_len
    else:
        ds = np.zeros(N, dtype=np.float64)
        ds[:-1] = np.diff(arc_lengths)
        ds[-1] = ds[-2] if N > 1 else 1.0

    # -- Time: cumulative sum of ds / v --
    dt = ds / np.maximum(v, 1e-6)
    time = np.zeros(N, dtype=np.float64)
    time[1:] = np.cumsum(dt[:-1])

    # -- Finite-difference function for derivative signals --
    fdiff = _finite_diff_periodic if periodic else _finite_diff_open

    # -- Model-specific fields --
    if model_name == "four_wheel":
        v_lat_arr = np.array(data["v_lat"], dtype=np.float64)
        yaw_rate_arr = np.array(data["yaw_rate"], dtype=np.float64)
        delta_arr = np.array(data["delta"], dtype=np.float64)

        velocity_lat = v_lat_arr
        yaw_angle_dot = yaw_rate_arr
        acceleration_lat = v * yaw_rate_arr
        steering_angle = delta_arr

        delta_dot_norm = np.array(data.get("delta_dot_norm", np.zeros(N)), dtype=np.float64)
        ddeltamax = float(params.get("ddeltamax", 1.0))
        steering_angle_dot = delta_dot_norm * ddeltamax

        force_long_fl = np.array(data["Fx_fl"], dtype=np.float64)
        force_long_fr = np.array(data["Fx_fr"], dtype=np.float64)
        force_long_rl = np.array(data["Fx_rl"], dtype=np.float64)
        force_long_rr = np.array(data["Fx_rr"], dtype=np.float64)

        m = float(params.get("m", 160.0))
        g_val = float(params.get("g", 9.81))
        rho = float(params.get("rho", 1.225))
        C_d = float(params.get("C_d", 1.58))
        C_r = float(params.get("C_r", 0.15))
        A_f = float(params.get("A_f", 1.2))
        F_drag = 0.5 * rho * C_d * A_f * v**2
        F_roll_val = m * g_val * C_r
        cd = np.cos(delta_arr)
        Fx_total = ((force_long_fl + force_long_fr) * cd
                    + force_long_rr + force_long_rl - F_roll_val - F_drag)
        a_long = Fx_total / m + yaw_rate_arr * v_lat_arr

    elif model_name == "dynamic_bicycle":
        a_long = np.array(data["a_long"], dtype=np.float64)
        v_lat = np.array(data["v_lat"], dtype=np.float64)
        yaw_rate = np.array(data["yaw_rate"], dtype=np.float64)
        delta = np.array(data["delta"], dtype=np.float64)

        velocity_lat = v_lat
        yaw_angle_dot = yaw_rate
        acceleration_lat = v * yaw_rate
        steering_angle = delta
        steering_angle_dot = fdiff(delta, dt)

        m = params["m"]
        g_val = params.get("g", 9.81)
        lf = params["lf"]
        lr = params["lr"]
        L_total = lf + lr

    elif model_name == "point_mass":
        a_long = np.array(data["a_long"], dtype=np.float64)
        a_lat_arr = np.array(data["a_lat"], dtype=np.float64)
        L = params.get("L", 1.8)

        velocity_lat = np.zeros(N, dtype=np.float64)
        yaw_angle_dot = v * kappa
        acceleration_lat = a_lat_arr
        steering_angle = np.arctan(L * kappa)
        steering_angle_dot = fdiff(steering_angle, dt)

        m = params.get("m", None)
        g_val = params.get("g", 9.81)
        lf = None
        lr = None
        L_total = L

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    # -- Force distribution (static normal load, 4WD) --
    if model_name == "four_wheel":
        pass  # forces already assigned above
    elif m is not None:
        if lf is not None and lr is not None:
            pass
        else:
            lf = L_total / 2.0
            lr = L_total / 2.0

        F_x_total = m * a_long
        force_long_fl = F_x_total * lr / L_total / 2.0
        force_long_fr = F_x_total * lr / L_total / 2.0
        force_long_rl = F_x_total * lf / L_total / 2.0
        force_long_rr = F_x_total * lf / L_total / 2.0
    else:
        force_long_fl = np.zeros(N, dtype=np.float64)
        force_long_fr = np.zeros(N, dtype=np.float64)
        force_long_rl = np.zeros(N, dtype=np.float64)
        force_long_rr = np.zeros(N, dtype=np.float64)

    # -- Write CSV --
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_COLUMNS)
        for i in range(N):
            writer.writerow([
                x[i],
                y[i],
                z[i],
                boundary_left[i],
                boundary_right[i],
                v[i],
                kappa[i],
                headings[i],
                psi_err[i],
                d[i],
                arc_lengths[i],
                time[i],
                yaw_angle_dot[i],
                velocity_lat[i],
                a_long[i],
                acceleration_lat[i],
                force_long_fl[i],
                force_long_fr[i],
                force_long_rl[i],
                force_long_rr[i],
                steering_angle[i],
                steering_angle_dot[i],
            ])

    return output_path
