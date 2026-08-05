"""
Geometric feasibility of the OCP corridor as a function of ``boundary_margin``.

Replicates VehicleBase.get_corner_constraints exactly on a discretized track and
reports, per station, the widest feasible interval in ``d`` (lateral deviation)
once all four body corners must stay inside the margin-shrunk corridor.

    python scripts/margin_corridor_analysis.py \
        --track data/discretized/ipz_august_3_with_widths.json \
        --config configs/autox.yaml --margins 0.2 0.3 0.35 0.4 0.45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def corner_offsets(cfg):
    return [(str(c[0]), float(c[1]), float(c[2])) for c in cfg["vehicle"]["corners"]]


def corridor_at(psi, kappa, w_left, w_right, corners, iters=3):
    """Feasible [d_lo, d_hi] for a given heading error, all four corners inside."""
    sin_p, cos_p = np.sin(psi), np.cos(psi)
    d_mid = 0.0
    d_lo = d_hi = 0.0
    for _ in range(iters):
        D_kappa = 1.0 - kappa * d_mid
        hi = np.inf
        lo = -np.inf
        for _name, dx, dy in corners:
            long_proj = dx * cos_p - dy * sin_p
            shift = dx * sin_p + dy * cos_p - 0.5 * kappa / D_kappa * long_proj ** 2
            if dy >= 0.0:
                hi = min(hi, w_left - shift)
            else:
                lo = max(lo, -w_right - shift)
        d_lo, d_hi = lo, hi
        d_mid = 0.5 * (lo + hi)
    return d_lo, d_hi


def analyse(track, cfg, margins, psi_max, psi_grid=61):
    kappa = np.asarray(track["curvatures"], dtype=float)
    wl = np.asarray(track["w_left"], dtype=float)
    wr = np.asarray(track["w_right"], dtype=float)
    s = np.asarray(track["arc_lengths"], dtype=float)
    corners = corner_offsets(cfg)
    psis = np.linspace(-psi_max, psi_max, psi_grid)

    out = {}
    for m in margins:
        wl_m, wr_m = wl - m, wr - m
        width_aligned = np.empty(len(s))
        width_best = np.empty(len(s))
        psi_best = np.empty(len(s))
        d_best = np.empty(len(s))
        for i in range(len(s)):
            lo, hi = corridor_at(0.0, kappa[i], wl_m[i], wr_m[i], corners)
            width_aligned[i] = hi - lo
            best_w, best_psi, best_d = -np.inf, 0.0, 0.0
            for p in psis:
                lo_p, hi_p = corridor_at(p, kappa[i], wl_m[i], wr_m[i], corners)
                if hi_p - lo_p > best_w:
                    best_w, best_psi, best_d = hi_p - lo_p, p, 0.5 * (lo_p + hi_p)
            width_best[i] = best_w
            psi_best[i] = best_psi
            d_best[i] = best_d
        out[m] = dict(width_aligned=width_aligned, width_best=width_best,
                      psi_best=psi_best, d_best=d_best)
    return s, kappa, wl, wr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="data/discretized/ipz_august_3_with_widths.json")
    ap.add_argument("--config", default="configs/autox.yaml")
    ap.add_argument("--margins", type=float, nargs="*",
                    default=[0.20, 0.30, 0.35, 0.40, 0.45])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    track = json.loads((REPO / args.track).read_text())
    cfg = yaml.safe_load((REPO / args.config).read_text())
    psi_max = float(cfg["vehicle"]["psi_err_max"])
    corners = corner_offsets(cfg)

    kappa = np.asarray(track["curvatures"], dtype=float)
    wl = np.asarray(track["w_left"], dtype=float)
    wr = np.asarray(track["w_right"], dtype=float)
    width = wl + wr

    print(f"track: {track['source_file'].split('/')[-1]}  "
          f"N={track['num_points']}  ds={track['ds_m']:.3f}  L={track['total_length_m']:.1f} m")
    print(f"corners (dx, dy): {corners}")
    print(f"\ncone-line width : min {width.min():.3f}  p05 {np.percentile(width,5):.3f}  "
          f"med {np.median(width):.3f}  max {width.max():.3f} m")
    print(f"|kappa|         : med {np.median(np.abs(kappa)):.3f}  "
          f"p95 {np.percentile(np.abs(kappa),95):.3f}  max {np.abs(kappa).max():.3f} 1/m"
          f"   (min radius {1/np.abs(kappa).max():.2f} m)")

    s, kappa, wl, wr, res = analyse(track, cfg, args.margins, psi_max)

    print("\ncorridor in d after all four corner constraints")
    print(f"{'margin':>7} | {'psi=0: min':>10} {'med':>7} {'#<0':>5} | "
          f"{'best psi: min':>13} {'med':>7} {'#<0':>5} | worst station")
    summary = {}
    for m in args.margins:
        wa = res[m]["width_aligned"]
        wb = res[m]["width_best"]
        i_worst = int(np.argmin(wb))
        print(f"{m:7.2f} | {wa.min():10.3f} {np.median(wa):7.3f} {int((wa<0).sum()):5d} | "
              f"{wb.min():13.3f} {np.median(wb):7.3f} {int((wb<0).sum()):5d} | "
              f"s={s[i_worst]:6.1f} m  kappa={kappa[i_worst]:+.3f}  "
              f"w={wl[i_worst]+wr[i_worst]:.2f} m  psi*={res[m]['psi_best'][i_worst]:+.2f}")
        summary[m] = dict(width_aligned_min=float(wa.min()),
                          width_best_min=float(wb.min()),
                          n_infeasible_aligned=int((wa < 0).sum()),
                          n_infeasible_best=int((wb < 0).sum()),
                          worst_s=float(s[i_worst]))

    # where does it get tight first
    ref = args.margins[-1]
    wb = res[ref]["width_best"]
    order = np.argsort(wb)[:12]
    print(f"\ntightest 12 stations at margin {ref} (best-psi corridor):")
    print(f"{'s [m]':>8} {'kappa':>8} {'w_l':>7} {'w_r':>7} {'corridor':>9} {'psi*':>7}")
    for i in sorted(order, key=lambda j: s[j]):
        print(f"{s[i]:8.1f} {kappa[i]:8.3f} {wl[i]:7.3f} {wr[i]:7.3f} "
              f"{wb[i]:9.3f} {res[ref]['psi_best'][i]:7.2f}")

    # margin at which each station becomes infeasible
    fine = np.arange(0.0, 0.81, 0.01)
    s_fine, k_fine, wl_f, wr_f, res_f = analyse(track, cfg, list(fine), psi_max, psi_grid=41)
    crit = np.full(len(s), np.nan)
    for m in fine:
        wb_m = res_f[m]["width_best"]
        newly = np.isnan(crit) & (wb_m < 0.0)
        crit[newly] = m
    finite = crit[~np.isnan(crit)]
    print(f"\ncritical margin (corridor closes, best psi): "
          f"min {np.nanmin(crit):.2f} m at s={s[int(np.nanargmin(crit))]:.1f} m; "
          f"{len(finite)} / {len(s)} stations close below 0.80 m")
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50):
        n = int(np.nansum(crit <= thr))
        print(f"  stations infeasible at margin {thr:.2f}: {n}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            dict(summary={str(k): v for k, v in summary.items()},
                 critical_margin=[None if np.isnan(c) else float(c) for c in crit],
                 s=[float(v) for v in s]), indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
