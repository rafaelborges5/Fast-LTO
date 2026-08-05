"""
Lap time vs boundary_margin for one track, plus where the time is lost.

Reads the per-margin solution JSONs produced by scripts/margin_continuation.py
(warm) and/or the pipeline (cold) and draws a 2x2 panel:

  [0,0] lap time vs margin, warm vs cold        [0,1] speed profiles overlaid
  [1,0] paths through the limiting corner       [1,1] solver cost of each margin
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load(path):
    d = json.loads(Path(path).read_text())
    p = d["profiling"]
    return dict(margin=float(d.get("run_config", {}).get("boundary_margin", np.nan)),
                lap=float(p.get("autox_lap_time_s", np.nan)),
                obj=float(d["obj_val"]), iters=p.get("iter_count"),
                time_s=float(p.get("solve_time_s", np.nan)),
                status=p.get("return_status"),
                s=np.asarray(d["arc_lengths"], float),
                v=np.asarray(d["v_long"], float),
                d_lat=np.asarray(d["d"], float),
                delta=np.asarray(d["delta"], float),
                path=np.asarray(d["path_xy"], float))


def cones(csv_path):
    left, right = [], []
    with Path(csv_path).open() as fh:
        for row in csv.DictReader(fh):
            side = row["side"].strip().upper()
            if side == "L":
                left.append((float(row["x"]), float(row["y"])))
            elif side == "R":
                right.append((float(row["x"]), float(row["y"])))
    return np.asarray(left), np.asarray(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", nargs="*", default=[])
    ap.add_argument("--cold", nargs="*", default=[])
    ap.add_argument("--track-csv", default="data/tracks/ipz_august_3.csv")
    ap.add_argument("--corner-s", type=float, default=184.6,
                    help="arc length of the limiting corner, for the zoom panel")
    ap.add_argument("--out", default="ocp_plots/ipz_margin_sensitivity.png")
    args = ap.parse_args()

    warm = sorted([load(p) for p in args.warm], key=lambda r: r["margin"])
    cold = sorted([load(p) for p in args.cold], key=lambda r: r["margin"])
    left, right = cones(REPO / args.track_csv)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Fast-LTO autox — boundary_margin sensitivity, ipz_august_3 "
                 "(four_wheel, euler, ds = 0.5 m)", fontsize=13)

    # -- lap time vs margin ------------------------------------------------
    a = ax[0, 0]
    if warm:
        m = [r["margin"] for r in warm]
        t = [r["lap"] for r in warm]
        a.plot(m, t, "o-", color="tab:blue", label="continuation (warm start)")
        for mi, ti in zip(m, t):
            a.annotate(f"{ti:.2f}", (mi, ti), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=9, color="tab:blue")
    if cold:
        m = [r["margin"] for r in cold]
        t = [r["lap"] for r in cold]
        a.plot(m, t, "s--", color="tab:orange", label="cold start (default guess)")
        for mi, ti in zip(m, t):
            a.annotate(f"{ti:.2f}", (mi, ti), textcoords="offset points",
                       xytext=(0, -14), ha="center", fontsize=9, color="tab:orange")
    base = warm[0] if warm else cold[0]
    a.set_xlabel("boundary_margin [m]")
    a.set_ylabel("autox lap time [s]")
    a.set_title("lap time vs margin")
    a.grid(alpha=0.3)
    a.legend()

    # -- speed profiles ----------------------------------------------------
    a = ax[0, 1]
    series = warm if warm else cold
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(series)))
    for r, c in zip(series, colors):
        a.plot(r["s"], r["v"], color=c, lw=1.4, label=f"margin {r['margin']:.2f}")
    a.axvline(args.corner_s, color="k", ls=":", lw=1)
    a.annotate("limiting hairpin", (args.corner_s, a.get_ylim()[1]),
               textcoords="offset points", xytext=(-70, -14), fontsize=9)
    a.set_xlabel("s [m]")
    a.set_ylabel("v [m/s]")
    a.set_title("speed profile")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    # -- corner zoom -------------------------------------------------------
    a = ax[1, 0]
    i_c = int(np.argmin(np.abs(series[0]["s"] - args.corner_s)))
    centre = series[0]["path"][i_c]
    a.scatter(left[:, 0], left[:, 1], s=18, color="tab:blue", label="left cones")
    a.scatter(right[:, 0], right[:, 1], s=18, color="tab:orange", label="right cones")
    for r, c in zip(series, colors):
        a.plot(r["path"][:, 0], r["path"][:, 1], color=c, lw=1.6,
               label=f"margin {r['margin']:.2f}")
    a.set_xlim(centre[0] - 12, centre[0] + 12)
    a.set_ylim(centre[1] - 9, centre[1] + 9)
    a.set_aspect("equal")
    a.set_title(f"paths through the limiting corner (s = {args.corner_s:.0f} m)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="lower right")

    # -- solver cost -------------------------------------------------------
    a = ax[1, 1]
    a.axis("off")
    rows = [("margin", "start", "status", "iters", "solve [s]", "lap [s]", "delta vs 0.30")]
    ref = None
    for r in sorted(warm + cold, key=lambda r: (r["margin"], r["iters"] or 0)):
        if ref is None:
            ref = r["lap"]
    for label, group in (("warm", warm), ("cold", cold)):
        for r in group:
            rows.append((f"{r['margin']:.2f}", label,
                         "OK" if r["status"] == "Solve_Succeeded" else str(r["status"]),
                         str(r["iters"]), f"{r['time_s']:.0f}",
                         f"{r['lap']:.3f}",
                         f"{r['lap'] - ref:+.3f}"))
    tbl = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    a.set_title("solver cost and lap-time penalty", pad=18)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
