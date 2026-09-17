"""
Autox force-rate x tyre-grip x corridor-margin sweep on a real track.

Three axes on top of a YAML config: the per-wheel longitudinal force rate limit
``dFxmax``, tyre peak grip ``D`` as a front/rear staircase, and the corridor
margin. Everything else -- track, mesh, braking profile, terminal conditions --
comes from the config, so a bare run solves exactly what
``fast-lto --config <that file>`` would.

Unlike ``skidpad_batch.py``, this drives the real ``run_pipeline`` so the
warm-start ladder is available: autox needs a margin continuation that skidpad
does not, and the ladder depends only on track geometry and corner offsets, so
neighbouring combos also seed each other. Sequential for the same reason --
warm-start chaining and one shared seed store beat solve-level parallelism.

Usage, from the repo root:
    python research/experiments/autox_dfx_grip_sweep.py --track-id <track>
    python research/experiments/autox_dfx_grip_sweep.py --track-id <track> \\
        --dfx-values 1000,2000 --margins 0.45,0.50 --d-max 1.30

Outputs under data/output_trajectories:
    <track_id>_dfx<N>_F*_R*_m*.csv        one trajectory per solve
    <track_id>_dfx_grip_sweep_times.csv   lap-time table
    <track_id>_dfx_grip_sweep.png         lap-time plot
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fast_lto.config import RunConfig
from fast_lto.paths import default_data_root
from fast_lto.pipeline import run_pipeline

#: Step of the front/rear grip staircase.
D_STEP = 0.05


def _repo_root() -> Path:
    return default_data_root()


def d_sequence(d_min: float, d_max: float) -> List[Tuple[float, float]]:
    """(D_front, D_rear) pairs: the rear leads each step from d_min to d_max.

    Same staircase shape as ``skidpad_batch.d_sequence``, without the low-grip
    anchors.
    """
    seq: List[Tuple[float, float]] = []
    f = r = d_min
    seq.append((f, r))
    while f < d_max - 1e-9 or r < d_max - 1e-9:
        if r <= f + 1e-9:  # equal -> bump rear first
            r = round(r + D_STEP, 2)
        else:  # rear ahead -> front catches up
            f = round(f + D_STEP, 2)
        seq.append((round(f, 2), round(r, 2)))
    return seq


def _tag(track_id: str, dfx: float, f: float, r: float, margin: float) -> str:
    return f"{track_id}_dfx{dfx:.0f}_F{f:.2f}_R{r:.2f}_m{margin:.2f}"


def _key(dfx: float, f: float, r: float, margin: float) -> Tuple[float, float, float, float]:
    return (round(float(dfx), 1), round(float(f), 2), round(float(r), 2), round(float(margin), 2))


def _load_table(
    path: Path, out_dir: Path
) -> Dict[Tuple[float, float, float, float], Dict[str, Any]]:
    """Load prior results so completed solves can be skipped on re-run."""
    done: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                key = _key(
                    float(row["dFxmax"]),
                    float(row["D_front"]),
                    float(row["D_rear"]),
                    float(row["margin"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                row.get("status") == "Solve_Succeeded"
                and row.get("csv")
                and (out_dir / row["csv"]).exists()
            ):
                done[key] = dict(row)
    return done


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "dFxmax",
        "D_front",
        "D_rear",
        "margin",
        "csv",
        "autox_lap_time_s",
        "status",
        "solve_time_s",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _print_table(rows: List[Dict[str, Any]], track_id: str) -> None:
    margins = sorted({float(r["margin"]) for r in rows})
    dfx_values = sorted({float(r["dFxmax"]) for r in rows})
    combos = sorted({(float(r["D_front"]), float(r["D_rear"])) for r in rows})
    by_key = {_key(r["dFxmax"], r["D_front"], r["D_rear"], r["margin"]): r for r in rows}

    for margin in margins:
        print("\n" + "=" * (18 + 12 * len(dfx_values)))
        print(f"{track_id} autox lap time [s]  (margin = {margin:.2f} m)")
        print("=" * (18 + 12 * len(dfx_values)))
        hdr = f"{'D_front':>8}{'D_rear':>8}" + "".join(
            f"{'dFx='+format(d,'.0f'):>12}" for d in dfx_values
        )
        print(hdr)
        print("-" * len(hdr))
        for f, r in combos:
            line = f"{f:>8.2f}{r:>8.2f}"
            for d in dfx_values:
                res = by_key.get(_key(d, f, r, margin))
                t = res.get("autox_lap_time_s") if res else None
                line += f"{float(t):>12.4f}" if t not in (None, "") else f"{'fail':>12}"
            print(line)
        print("-" * len(hdr))


# Fixed order, not cycled: these three stay distinguishable pairwise under
# colour-vision deficiency, which a small-multiples line chart needs.
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
_TEXT_SECONDARY = "#52514e"


def _plot(rows: List[Dict[str, Any]], out_dir: Path, track_id: str) -> Optional[Path]:
    margins = sorted({float(r["margin"]) for r in rows})
    dfx_values = sorted({float(r["dFxmax"]) for r in rows})
    combos = sorted({(float(r["D_front"]), float(r["D_rear"])) for r in rows})
    by_key = {_key(r["dFxmax"], r["D_front"], r["D_rear"], r["margin"]): r for r in rows}
    labels = [f"F{f:.2f}/R{r:.2f}" for f, r in combos]
    x = list(range(len(combos)))

    # One panel per margin (small multiples), not a second y-axis.
    fig, axes = plt.subplots(1, len(margins), figsize=(6.5 * len(margins), 5), sharey=True)
    if len(margins) == 1:
        axes = [axes]

    any_plotted = False
    for ax, margin in zip(axes, margins):
        for i, dfx in enumerate(dfx_values):
            col = _SERIES_COLORS[i % len(_SERIES_COLORS)]
            ys = []
            for f, r in combos:
                res = by_key.get(_key(dfx, f, r, margin))
                t = res.get("autox_lap_time_s") if res else None
                ys.append(float(t) if t not in (None, "") else None)
            xs_valid = [xi for xi, yi in zip(x, ys) if yi is not None]
            ys_valid = [yi for yi in ys if yi is not None]
            if not ys_valid:
                continue
            any_plotted = True
            ax.plot(
                xs_valid,
                ys_valid,
                marker="o",
                markersize=7,
                color=col,
                lw=2.0,
                label=f"dFxmax={dfx:.0f}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", color=_TEXT_SECONDARY)
        ax.set_title(f"margin = {margin:.2f} m")
        ax.grid(True, ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=_TEXT_SECONDARY)
        ax.legend(frameon=False)
    if not any_plotted:
        plt.close(fig)
        return None
    axes[0].set_ylabel("autox lap time [s]", color=_TEXT_SECONDARY)
    fig.suptitle(f"{track_id}: autox lap time vs tyre grip, by dFxmax and margin")
    fig.tight_layout()
    out_png = out_dir / f"{track_id}_dfx_grip_sweep.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def _solve_one(
    base_rc: RunConfig,
    track_id: str,
    dfx: float,
    d_front: float,
    d_rear: float,
    margin: float,
    out_dir: Path,
) -> Dict[str, Any]:
    rc = copy.deepcopy(base_rc)
    rc.vehicle.four_wheel["dFxmax"] = float(dfx)
    rc.vehicle.four_wheel["D_fl"] = float(d_front)
    rc.vehicle.four_wheel["D_fr"] = float(d_front)
    rc.vehicle.four_wheel["D_rr"] = float(d_rear)
    rc.vehicle.four_wheel["D_rl"] = float(d_rear)
    rc.pipeline.boundary_margin = float(margin)

    config = rc.to_pipeline_config()
    config.plot_results = False
    config.show_plots = False

    tag = _tag(track_id, dfx, d_front, d_rear, margin)
    out_csv = out_dir / f"{tag}.csv"
    result: Dict[str, Any] = {
        "dFxmax": dfx,
        "D_front": d_front,
        "D_rear": d_rear,
        "margin": margin,
        "csv": out_csv.name,
    }

    start = time.perf_counter()
    try:
        results = run_pipeline(config, start_from="track", end_at="export")
        solve_time_s = time.perf_counter() - start

        sol = json.loads(Path(results["ocp"]).read_text())
        prof = sol.get("profiling", {})
        status = prof.get("return_status")
        lap_time = prof.get("autox_lap_time_s")
        if lap_time is None:
            lap_time = prof.get("pure_timed_time_s")

        result.update(
            autox_lap_time_s=lap_time,
            status=status,
            solve_time_s=round(solve_time_s, 2),
        )
        if status == "Solve_Succeeded":
            Path(results["export"]).replace(out_csv)
        else:
            result["status"] = f"{status}; not exported"
    except Exception as exc:  # keep the sweep alive on a single failure
        result.update(
            autox_lap_time_s=None,
            status=f"ERROR: {type(exc).__name__}: {exc}",
            solve_time_s=round(time.perf_counter() - start, 2),
        )
    return result


def run(
    config_path: Path,
    track_id: str,
    out_dir: Path,
    dfx_values: Optional[Tuple[float, ...]] = None,
    margins: Optional[Tuple[float, ...]] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    smooth_centerline: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baseline config: {config_path}, track_id={track_id}")
    base_rc = RunConfig.from_yaml(config_path)
    # The pipeline settings live on `.pipeline`; assigning to the RunConfig
    # itself would create an attribute nobody reads.
    base_rc.pipeline.track_id = track_id
    if smooth_centerline is not None:
        print(
            f"  Overriding smooth_centerline: "
            f"{base_rc.pipeline.smooth_centerline} -> {smooth_centerline}"
        )
        base_rc.pipeline.smooth_centerline = smooth_centerline

    # Anything not being swept comes from the config, so a bare run solves
    # exactly what `fast-lto --config <this>` would.
    four_wheel = base_rc.vehicle.four_wheel or {}
    grip = [float(four_wheel[k]) for k in ("D_fl", "D_fr", "D_rr", "D_rl")]
    if d_min is None:
        d_min = round(min(grip), 2)
    if d_max is None:
        d_max = round(max(grip), 2)
    if dfx_values is None:
        dfx_values = (float(four_wheel["dFxmax"]),)
    if margins is None:
        margins = (round(float(base_rc.pipeline.boundary_margin), 2),)

    combos = d_sequence(d_min=d_min, d_max=d_max)
    print(
        f"  dFxmax {', '.join(f'{v:.0f}' for v in dfx_values)} | "
        f"grip D {d_min:.2f} -> {d_max:.2f} ({len(combos)} combination(s)) | "
        f"margins {', '.join(f'{m:.2f}' for m in margins)}"
    )
    table_csv = out_dir / f"{track_id}_dfx_grip_sweep_times.csv"
    done = _load_table(table_csv, out_dir)
    results: List[Dict[str, Any]] = list(done.values())

    total = len(margins) * len(dfx_values) * len(combos)
    print(
        f"{len(done)}/{total} already done; {total - len(done)} solves remaining "
        f"(sequential, to allow warm-start chaining) ..."
    )

    for margin in margins:
        for dfx in dfx_values:
            for f, r in combos:
                key = _key(dfx, f, r, margin)
                if key in done:
                    print(f"  [{_tag(track_id, dfx, f, r, margin)}] already done, skipping")
                    continue
                print(f"  [{_tag(track_id, dfx, f, r, margin)}] solving ...")
                res = _solve_one(base_rc, track_id, dfx, f, r, margin, out_dir)
                print(
                    f"    -> lap_time={res['autox_lap_time_s']} status={res['status']} "
                    f"solve_time_s={res['solve_time_s']}"
                )
                results.append(res)
                _write_table(table_csv, results)  # persist after every solve (resumable)

    _print_table(results, track_id)
    png = _plot(results, out_dir, track_id)

    print(f"\nWrote:\n  {table_csv}")
    if png is not None:
        print(f"  {png}")
    n_ok = len([r for r in results if r.get("status") == "Solve_Succeeded"])
    print(f"  {n_ok}/{total} trajectory CSVs in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=_repo_root() / "configs" / "autox.yaml")
    ap.add_argument(
        "--track-id",
        type=str,
        required=True,
        help="Track identifier, e.g. the new track's CSV stem under data/tracks/.",
    )
    ap.add_argument("--out-dir", type=Path, default=_repo_root() / "data" / "output_trajectories")
    ap.add_argument(
        "--dfx-values",
        type=str,
        default=None,
        help="Comma-separated dFxmax values, e.g. 1000,2000. " "Default: the config's own dFxmax.",
    )
    ap.add_argument(
        "--margins",
        type=str,
        default=None,
        help="Comma-separated boundary margins, e.g. 0.45,0.50. "
        "Default: the config's own boundary_margin.",
    )
    ap.add_argument(
        "--d-min",
        type=float,
        default=None,
        help="Lower tyre-grip D for the staircase. Default: the config's own D.",
    )
    ap.add_argument(
        "--d-max",
        type=float,
        default=None,
        help="Upper tyre-grip D. Default: the config's own D, which makes the "
        "grip axis a single point.",
    )
    ap.add_argument(
        "--smooth-centerline",
        type=int,
        default=None,
        help="Override the config's Savgol centerline-smoothing window "
        "(odd int >= 3). Useful for noisy logged-path tracks where "
        "the default produces spurious sharp curvature spikes.",
    )
    args = ap.parse_args()

    def _floats(raw: Optional[str]) -> Optional[Tuple[float, ...]]:
        if raw is None:
            return None
        return tuple(float(x) for x in raw.split(",") if x.strip())

    dfx_values = _floats(args.dfx_values)
    margins = _floats(args.margins)
    run(
        args.config,
        args.track_id,
        args.out_dir,
        dfx_values,
        margins,
        args.d_min,
        args.d_max,
        smooth_centerline=args.smooth_centerline,
    )


if __name__ == "__main__":
    main()
