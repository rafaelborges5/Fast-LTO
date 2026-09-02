from __future__ import annotations

"""
Batch skidpad LTO generation.

Sweeps the tyre peak-grip D over a front/rear evolution and a set of corridor
margins, solving the four-wheel skidpad OCP for each combination.  Every run is
exported as a controller-reference trajectory CSV whose name encodes the D
parameters and the margin, e.g.::

    data/output_trajectories/fscz_skidpad_F1.30_R1.35_m0.50.csv

D evolution (rear leads each 0.05 step, 1.20 -> --d-max), plus two uniform
low-grip anchors at F=R=1.10 and F=R=1.00 (unless --no-anchors).
Margins, output prefix, and the D ceiling are all configurable; defaults
reproduce the original FSCZ sweep (prefix fscz_skidpad, margins 0.40/0.50/0.60,
d-max 1.50, anchors on).

Usage (from repo root):
    PYTHONPATH=src python src/experiments/fscz_skidpad_batch.py --jobs 4
    PYTHONPATH=src python src/experiments/fscz_skidpad_batch.py \\
        --prefix ipz_skidpad_night_aug4 --margins 0.40,0.50 --d-max 1.45 \\
        --no-anchors --jobs 4

Outputs (named after --prefix, default fscz_skidpad):
    data/output_trajectories/<prefix>_F*_R*_m*.csv   one per solve
    data/output_trajectories/<prefix>_times.csv       score table
    data/output_trajectories/<prefix>_margin_*.png    trajectory plots
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fast_lto.config import RunConfig
from fast_lto.export.trajectory import export_reference_trajectory
from fast_lto.optimization.global_ocp import solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator
from fast_lto.pipeline import _build_skidpad_lead_in, _prepend_skidpad_lead_in, _resolve_path, PipelineConfig
from fast_lto.vehicle_models.four_wheel import FourWheelModel

DEFAULT_MARGINS = (0.40, 0.50, 0.60)
DEFAULT_PREFIX = "fscz_skidpad"
DEFAULT_D_MAX = 1.50


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def d_sequence(d_min: float = 1.20, d_max: float = DEFAULT_D_MAX, include_anchors: bool = True) -> List[Tuple[float, float]]:
    """(D_front, D_rear) pairs: rear leads each 0.05 step from d_min to d_max,
    then (if include_anchors) two uniform low-grip anchors."""
    seq: List[Tuple[float, float]] = []
    f = r = d_min
    seq.append((f, r))
    while f < d_max - 1e-9 or r < d_max - 1e-9:
        if r <= f + 1e-9:  # equal -> bump rear first
            r = round(r + 0.05, 2)
        else:  # rear ahead -> front catches up
            f = round(f + 0.05, 2)
        seq.append((round(f, 2), round(r, 2)))
    if include_anchors:
        seq.append((1.10, 1.10))
        seq.append((1.00, 1.00))
    return seq


@dataclass
class Baseline:
    track: Dict[str, Any]
    time_weights: np.ndarray
    model_params: Dict[str, Any]
    integrator_name: str
    initial_speed: float
    reg_du: Any
    reg_u_l2: Any
    terminal_speed: Any
    normalize: bool
    lead_in: Optional[Dict[str, Any]]
    terminal_straight_m: float


def _build_baseline(config_path: Path) -> Baseline:
    from fast_lto.tracks.skidpad import build_skidpad_track

    rc = RunConfig.from_yaml(config_path)
    rc.validate_for_model()
    pc: PipelineConfig = rc.to_pipeline_config()
    pc.repo_root = _repo_root()
    pc.__post_init__()

    map_csv = _resolve_path(pc.repo_root, pc.skidpad_map_csv)
    ref_csv = _resolve_path(pc.repo_root, pc.skidpad_reference_csv)
    start_xy = (pc.skidpad_start_x, pc.skidpad_start_y) if pc.skidpad_start_x is not None else None
    track = build_skidpad_track(
        map_csv=map_csv,
        ref_csv=ref_csv,
        ds_m=pc.ds_m,
        entry_exit_halfwidth=pc.entry_exit_halfwidth,
        kappa_blend_m=pc.kappa_blend_m,
        start_xy=start_xy,
    )

    mask = np.asarray(track["timed_mask"], dtype=float)
    decel = np.asarray(track.get("decel_mask", np.zeros_like(mask)), dtype=float)
    time_weights = np.where(mask > 0.5, 1.0, float(pc.eps_time))
    time_weights = np.where(decel > 0.5, 0.0, time_weights)
    # Keep the heavy timed weight for the first `decel_hold_m` metres of the exit
    # (mirrors pipeline.step_solve_ocp), so the terminal brake starts after the
    # finish gate instead of the solver losing all time-pressure immediately.
    if pc.decel_hold_m > 0.0:
        ds_hold = float(track.get("ds_m", pc.ds_m))
        decel_idx = np.where(decel > 0.5)[0]
        n_hold = min(int(round(pc.decel_hold_m / ds_hold)), decel_idx.size)
        if n_hold > 0:
            time_weights[decel_idx[:n_hold]] = 1.0

    lead_in = _build_skidpad_lead_in(track, pc.skidpad_lead_in_m)

    return Baseline(
        track=track,
        time_weights=time_weights,
        model_params=rc.vehicle.build_model_params("four_wheel"),
        integrator_name=pc.integrator_name,
        initial_speed=float(pc.initial_speed),
        reg_du=pc.reg_u,
        reg_u_l2=pc.reg_u_l2,
        terminal_speed=pc.terminal_speed,
        normalize=bool(pc.normalize_states_and_inputs),
        lead_in=lead_in,
        terminal_straight_m=pc.skidpad_terminal_straight_m,
    )


def _make_integrator(name: str):
    return RK4Integrator() if name == "rk4" else EulerIntegrator()


def _tag(prefix: str, f: float, r: float, m: float) -> str:
    return f"{prefix}_F{f:.2f}_R{r:.2f}_m{m:.2f}"


def _solve_one(job: Dict[str, Any]) -> Dict[str, Any]:
    base: Baseline = job["baseline"]
    f, r, margin = job["D_front"], job["D_rear"], job["margin"]
    out_csv = Path(job["out_csv"])
    tag = out_csv.stem

    params = dict(base.model_params)
    params.update(D_fl=f, D_fr=f, D_rr=r, D_rl=r)
    model = FourWheelModel(params=params)

    sol_json = Path(job["scratch_dir"]) / f"{tag}.json"
    result: Dict[str, Any] = {
        "D_front": f,
        "D_rear": r,
        "margin": margin,
        "csv": out_csv.name,
    }
    try:
        sol = solve_ocp_and_save(
            track=base.track,
            model=model,
            solution_path=sol_json,
            integrator=_make_integrator(base.integrator_name),
            initial_speed=base.initial_speed,
            reg_du=base.reg_du,
            reg_u_l2=base.reg_u_l2,
            run_config={
                "tag": tag,
                "model_name": "four_wheel",
                "mode": "skidpad",
                "boundary_margin": margin,
            },
            use_normalization=base.normalize,
            solver_verbose=False,
            boundary_margin=margin,
            mode="skidpad",
            time_weights=base.time_weights,
            terminal_speed=base.terminal_speed,
            terminal_straight_m=base.terminal_straight_m,
        )
        prof = sol.get("profiling", {})
        status = prof.get("return_status")
        result.update(
            score_s=prof.get("skidpad_score_s"),
            full_lap_s=prof.get("lap_time_s"),
            status=status,
            solve_time_s=prof.get("solve_time_s"),
        )
        result["exported"] = False
        if status == "Solve_Succeeded":
            try:
                if base.lead_in is not None:
                    sol = _prepend_skidpad_lead_in(sol, base.lead_in, base.initial_speed)
                    with sol_json.open("w") as f:
                        json.dump(sol, f)
                export_reference_trajectory(sol_json, out_csv)
                result["exported"] = True
            except Exception as exc:  # solve is still valid; just note it
                result["status"] = f"{status}; EXPORT_ERROR: {exc}"
    except Exception as exc:  # keep the sweep alive on a single failure
        result.update(
            score_s=None, full_lap_s=None,
            status=f"ERROR: {type(exc).__name__}: {exc}",
            solve_time_s=None, exported=False,
        )
    finally:
        try:
            sol_json.unlink()
        except OSError:
            pass
    return result


def _load_table(path: Path, out_dir: Path) -> Dict[Tuple[float, float, float], Dict[str, Any]]:
    """Load prior results so completed solves can be skipped on re-run."""
    done: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                f_, r_, m_ = (round(float(row["D_front"]), 2),
                              round(float(row["D_rear"]), 2),
                              round(float(row["margin"]), 2))
            except (KeyError, TypeError, ValueError):
                continue
            # Only treat as done if the solve succeeded and its CSV is present.
            if row.get("status") == "Solve_Succeeded" and row.get("csv") \
                    and (out_dir / row["csv"]).exists():
                done[(f_, r_, m_)] = {
                    "D_front": f_, "D_rear": r_, "margin": m_,
                    "score_s": float(row["score_s"]) if row.get("score_s") else None,
                    "full_lap_s": float(row["full_lap_s"]) if row.get("full_lap_s") else None,
                    "status": row["status"], "csv": row["csv"],
                    "solve_time_s": float(row["solve_time_s"]) if row.get("solve_time_s") else None,
                    "exported": True,
                }
    return done


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["D_front", "D_rear", "margin", "score_s", "full_lap_s",
            "status", "csv", "solve_time_s"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _print_table(rows: List[Dict[str, Any]], prefix: str) -> None:
    margins = sorted({r["margin"] for r in rows})
    combos = sorted({(r["D_front"], r["D_rear"]) for r in rows})
    by_key = {(r["D_front"], r["D_rear"], r["margin"]): r for r in rows}

    print("\n" + "=" * (24 + 12 * len(margins)))
    print(f"{prefix} FS score [s]  (avg of the two timed laps)")
    print("=" * (24 + 12 * len(margins)))
    hdr = f"{'D_front':>8}{'D_rear':>8}" + "".join(f"{'m='+format(m,'.2f'):>12}" for m in margins)
    print(hdr)
    print("-" * len(hdr))
    for f, r in combos:
        line = f"{f:>8.2f}{r:>8.2f}"
        for m in margins:
            res = by_key.get((f, r, m))
            if res and res.get("score_s") is not None:
                line += f"{res['score_s']:>12.4f}"
            else:
                line += f"{'fail':>12}"
        print(line)
    print("-" * len(hdr))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_margins(rows: List[Dict[str, Any]], out_dir: Path,
                  base: Baseline, show_combos: List[Tuple[float, float]],
                  prefix: str) -> Path:
    margins = sorted({r["margin"] for r in rows})
    by_key = {(r["D_front"], r["D_rear"], r["margin"]): r for r in rows}

    sk = base.track["skidpad"]
    c1 = np.array(sk["c_first"]); c2 = np.array(sk["c_second"])
    R_in, R_out = sk["R_in"], sk["R_out"]
    th = np.linspace(0, 2 * np.pi, 200)

    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(show_combos)))

    fig, axes = plt.subplots(1, len(margins), figsize=(6 * len(margins), 6.2),
                             sharex=True, sharey=True)
    if len(margins) == 1:
        axes = [axes]

    for ax, m in zip(axes, margins):
        for c in (c1, c2):
            for R, ls in ((R_in, "--"), (R_out, "-")):
                ax.plot(c[0] + R * np.cos(th), c[1] + R * np.sin(th),
                        color="0.6", lw=0.8, ls=ls, zorder=1)
        for (f, r), col in zip(show_combos, colors):
            res = by_key.get((f, r, m))
            if not res or not res.get("exported"):
                continue
            xy = np.loadtxt(out_dir / res["csv"], delimiter=",", skiprows=1,
                            usecols=(0, 1))
            lbl = f"F{f:.2f}/R{r:.2f} ({res['score_s']:.3f}s)"
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.4, label=lbl, zorder=3)
        ax.set_title(f"margin = {m:.2f} m")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("y [m]")
    fig.suptitle(f"{prefix} trajectories by corridor margin "
                 "(dashed = inner cone circle, solid = outer)", fontsize=12)
    fig.tight_layout()
    out_png = out_dir / f"{prefix}_margin_trajectories.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def run(
    config_path: Path,
    jobs: int,
    out_dir: Path,
    prefix: str = DEFAULT_PREFIX,
    margins: Tuple[float, ...] = DEFAULT_MARGINS,
    d_min: float = 1.20,
    d_max: float = DEFAULT_D_MAX,
    include_anchors: bool = True,
    initial_speed: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"Building skidpad baseline from {config_path} ...")
    base = _build_baseline(config_path)
    if initial_speed is not None:
        base.initial_speed = float(initial_speed)

    combos = d_sequence(d_min=d_min, d_max=d_max, include_anchors=include_anchors)
    table_csv = out_dir / f"{prefix}_times.csv"
    done = _load_table(table_csv, out_dir)
    results: List[Dict[str, Any]] = list(done.values())

    all_jobs: List[Dict[str, Any]] = []
    for (f, r) in combos:
        for m in margins:
            if (f, r, m) in done:
                continue
            all_jobs.append({
                "baseline": base, "D_front": f, "D_rear": r, "margin": m,
                "out_csv": str(out_dir / f"{_tag(prefix, f, r, m)}.csv"),
                "scratch_dir": str(scratch),
            })

    total = len(combos) * len(margins)
    print(f"{len(done)}/{total} already done; submitting {len(all_jobs)} solves "
          f"on {jobs} worker(s) ...")

    def _record(res: Dict[str, Any]) -> None:
        print(f"  [{_tag(prefix, res['D_front'], res['D_rear'], res['margin'])}] "
              f"score={res['score_s']} status={res['status']}")
        results.append(res)
        _write_table(table_csv, results)  # persist after every solve (resumable)

    if jobs <= 1:
        for j in all_jobs:
            _record(_solve_one(j))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for res in ex.map(_solve_one, all_jobs):
                _record(res)

    _write_table(table_csv, results)
    _print_table(results, prefix)

    # Show a representative spread of grip levels (the balanced F==R points of
    # the ramp) on the trajectory plot -- generalizes to any d_max/anchors.
    show = [c for c in combos if abs(c[0] - c[1]) < 1e-9]
    show = [c for c in show if c in set((r["D_front"], r["D_rear"]) for r in results)]
    png = _plot_margins(results, out_dir, base, show, prefix)

    print(f"\nWrote:\n  {table_csv}\n  {png}")
    print(f"  {len([r for r in results if r.get('exported')])} trajectory CSVs "
          f"in {out_dir}")

    try:
        for p in scratch.glob("*"):
            p.unlink()
        scratch.rmdir()
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path,
                    default=_repo_root() / "configs" / "skidpad.yaml")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel solve workers.")
    ap.add_argument("--out-dir", type=Path,
                    default=_repo_root() / "data" / "output_trajectories")
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX,
                    help="Output filename prefix.")
    ap.add_argument("--margins", type=str,
                    default=",".join(f"{m:.2f}" for m in DEFAULT_MARGINS),
                    help="Comma-separated boundary margins, e.g. 0.40,0.50")
    ap.add_argument("--d-min", type=float, default=1.20,
                    help="Lower D value for the front/rear ramp (default 1.20).")
    ap.add_argument("--d-max", type=float, default=DEFAULT_D_MAX,
                    help="Upper D value for the front/rear ramp.")
    ap.add_argument("--no-anchors", action="store_true",
                    help="Skip the uniform low-grip anchors (F=R=1.10, F=R=1.00).")
    ap.add_argument("--initial-speed", type=float, default=None,
                    help="Override the config's initial_speed (m/s). Also sets the "
                         "skidpad_lead_in_m constant-speed lead-in, since it must "
                         "match the OCP's fixed starting velocity.")
    args = ap.parse_args()
    margins = tuple(float(x) for x in args.margins.split(","))
    run(
        args.config, args.jobs, args.out_dir,
        prefix=args.prefix, margins=margins, d_min=args.d_min, d_max=args.d_max,
        include_anchors=not args.no_anchors,
        initial_speed=args.initial_speed,
    )


if __name__ == "__main__":
    main()
