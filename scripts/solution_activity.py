"""
Where is an OCP solution pinned against its limits?

Reports, for a solved trajectory, how close each state/input sits to its bound
(steering, wheel forces, corridor corners), so it is visible which constraint is
the binding one before a bigger boundary_margin makes the problem unsolvable.

    python scripts/solution_activity.py data/solutions/ipz_august_3_four_wheel_euler_autox.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def corner_slacks(d, psi, kappa, w_left, w_right, corners):
    """Signed distance of every body corner to its boundary (>0 = inside)."""
    sin_p, cos_p = np.sin(psi), np.cos(psi)
    D_kappa = 1.0 - kappa * d
    out = {}
    for name, dx, dy in corners:
        long_proj = dx * cos_p - dy * sin_p
        d_corner = d + dx * sin_p + dy * cos_p - 0.5 * kappa / D_kappa * long_proj**2
        out[name] = (w_left - d_corner) if dy >= 0 else (d_corner + w_right)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--config", default="configs/autox.yaml")
    ap.add_argument(
        "--margin",
        type=float,
        default=None,
        help="margin used for the solve (default: read from run_config)",
    )
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    sol = json.loads(Path(args.solution).read_text())
    cfg = yaml.safe_load((REPO / args.config).read_text())
    fw = cfg["vehicle"]["four_wheel"]
    corners = [(str(c[0]), float(c[1]), float(c[2])) for c in cfg["vehicle"]["corners"]]

    margin = args.margin
    if margin is None:
        margin = float(sol.get("run_config", {}).get("boundary_margin", 0.0))

    s = np.asarray(sol["arc_lengths"], float)
    d = np.asarray(sol["d"], float)
    psi = np.asarray(sol["psi_err"], float)
    kappa = np.asarray(sol["kappa"], float)
    v = np.asarray(sol["v_long"], float)
    delta = np.asarray(sol["delta"], float)
    wl = np.asarray(sol["w_left"], float) - margin
    wr = np.asarray(sol["w_right"], float) - margin
    fx = np.column_stack([sol["Fx_fl"], sol["Fx_fr"], sol["Fx_rr"], sol["Fx_rl"]])

    delta_max = float(fw["delta_max"])
    fx_max = float(fw["Fx_max"])

    print(f"solution : {Path(args.solution).name}")
    prof = sol.get("profiling", {})
    print(
        f"status   : {prof.get('return_status')}  iters={prof.get('iter_count')} "
        f"time={prof.get('solve_time_s'):.1f}s  obj={sol['obj_val']:.3f}"
    )
    print(f"margin   : {margin:.3f} m   N={len(s)}   v {v.min():.2f}..{v.max():.2f} m/s")

    # steering activity
    sat = np.abs(delta) > 0.999 * delta_max
    near = np.abs(delta) > 0.95 * delta_max
    print(f"\nsteering |delta| max {np.abs(delta).max():.4f} / limit {delta_max:.3f}")
    print(f"  nodes at  >99.9% of the limit : {int(sat.sum()):4d} / {len(s)}")
    print(f"  nodes at  >95%   of the limit : {int(near.sum()):4d} / {len(s)}")
    if sat.any():
        seg = s[sat]
        print(f"  saturated arc-length range    : {seg.min():.1f} .. {seg.max():.1f} m")

    # wheel force activity
    fx_sat = np.abs(fx) > 0.999 * fx_max
    print(
        f"\nwheel force |Fx| max {np.abs(fx).max():.1f} / limit {fx_max:.0f} N; "
        f"nodes with any wheel saturated: {int(fx_sat.any(axis=1).sum())} / {len(s)}"
    )

    # corridor activity
    slack = corner_slacks(d, psi, kappa, wl, wr, corners)
    worst = np.min(np.column_stack(list(slack.values())), axis=1)
    print(
        f"\ncorner clearance (min over 4 corners): min {worst.min():+.4f} m, "
        f"nodes < 1 cm : {int((worst < 0.01).sum())}, < 5 cm : {int((worst < 0.05).sum())}"
    )
    for name, sl in slack.items():
        print(
            f"  {name}: min {sl.min():+.4f} m at s={s[int(np.argmin(sl))]:6.1f} m, "
            f"active(<1cm) at {int((sl < 0.01).sum()):4d} nodes"
        )

    # the tightest places
    idx = np.argsort(worst)[: args.top]
    print(f"\ntightest {args.top} nodes")
    print(
        f"{'s [m]':>8} {'kappa':>8} {'d':>7} {'psi':>7} {'v':>6} {'delta':>7} "
        f"{'clear':>7} {'w_l':>6} {'w_r':>6}"
    )
    for i in sorted(idx, key=lambda j: s[j]):
        print(
            f"{s[i]:8.1f} {kappa[i]:8.3f} {d[i]:7.3f} {psi[i]:7.3f} {v[i]:6.2f} "
            f"{delta[i]:7.3f} {worst[i]:7.3f} {wl[i]+margin:6.2f} {wr[i]+margin:6.2f}"
        )

    # steering demand vs achievable, around the tightest corner
    lf, lr = float(cfg["vehicle"]["lf"]), float(cfg["vehicle"]["lr"])
    kappa_path_max = np.tan(delta_max) / (lf + lr)
    print(
        f"\nkinematic path-curvature ceiling at delta_max: "
        f"{kappa_path_max:.3f} 1/m  (radius {1/kappa_path_max:.2f} m)"
    )
    over = np.abs(kappa) > kappa_path_max
    print(
        f"centreline stations above that ceiling: {int(over.sum())} / {len(s)}"
        + (f", s = {s[over].min():.1f} .. {s[over].max():.1f} m" if over.any() else "")
    )


if __name__ == "__main__":
    main()
