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
    params = data.get("model_params", {})

    # -- Common arrays --
    path_xy = np.array(data["path_xy"], dtype=np.float64)
    x = path_xy[:, 0]
    y = path_xy[:, 1]
    N = len(x)

    z = np.zeros(N, dtype=np.float64)

    w_left = np.array(data["w_left"], dtype=np.float64)
    w_right = np.array(data["w_right"], dtype=np.float64)
    boundary_left = w_left
    boundary_right = -w_right  # sign flip for ROS convention

    v = np.array(data["v"], dtype=np.float64)
    kappa = np.array(data["kappa"], dtype=np.float64)
    headings = np.array(data["headings"], dtype=np.float64)
    psi_err = np.array(data["psi_err"], dtype=np.float64)
    d = np.array(data["d"], dtype=np.float64)
    arc_lengths = np.array(data["arc_lengths"], dtype=np.float64)
    a_long = np.array(data["a_long"], dtype=np.float64)

    # -- Arc-length steps (periodic) --
    ds = np.diff(arc_lengths, append=arc_lengths[0])
    # The last step wraps around: use the typical step size as approximation
    ds[-1] = arc_lengths[1] - arc_lengths[0]

    # -- Time: cumulative sum of ds / v --
    dt = ds / np.maximum(v, 1e-6)
    time = np.zeros(N, dtype=np.float64)
    time[1:] = np.cumsum(dt[:-1])

    # -- Model-specific fields --
    if model_name == "dynamic_bicycle":
        v_lat = np.array(data["v_lat"], dtype=np.float64)
        yaw_rate = np.array(data["yaw_rate"], dtype=np.float64)
        delta = np.array(data["delta"], dtype=np.float64)

        velocity_lat = v_lat
        yaw_angle_dot = yaw_rate
        acceleration_lat = v * yaw_rate
        steering_angle = delta
        steering_angle_dot = _finite_diff_periodic(delta, dt)

        # Force calculation
        m = params["m"]
        g_val = params.get("g", 9.81)
        lf = params["lf"]
        lr = params["lr"]
        L_total = lf + lr

    elif model_name == "point_mass":
        a_lat_arr = np.array(data["a_lat"], dtype=np.float64)
        L = params.get("L", 1.8)

        velocity_lat = np.zeros(N, dtype=np.float64)
        yaw_angle_dot = v * kappa
        acceleration_lat = a_lat_arr
        steering_angle = np.arctan(L * kappa)
        steering_angle_dot = _finite_diff_periodic(steering_angle, dt)

        # Force calculation: point mass may not have mass
        m = params.get("m", None)
        g_val = params.get("g", 9.81)
        lf = None
        lr = None
        L_total = L

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    # -- Force distribution (static normal load, 4WD) --
    if m is not None:
        if lf is not None and lr is not None:
            # dynamic bicycle: explicit front/rear axle distances
            pass
        else:
            # point mass: treat L as total wheelbase with 50/50 split for lf/lr
            # but use the proper formula: Fz_f = m*g*lr/L, Fz_r = m*g*lf/L
            # With only L known, assume lf = lr = L/2
            lf = L_total / 2.0
            lr = L_total / 2.0

        F_x_total = m * a_long
        # Fx_front = F_x_total * lr / L_total
        # Fx_rear  = F_x_total * lf / L_total
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
