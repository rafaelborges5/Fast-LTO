from __future__ import annotations

"""
Batch skidpad solves over a grid of tyre grip and corridor margin.

The YAML config is the baseline: track, mesh, integrator, launch speed,
regularisation, terminal speed and lead-in all come from it, and so do the
starting tyre-grip and margin values. Flags add sweep axes on top. With no
flags the sweep is a single point -- exactly what the config already says -- so
every difference between two outputs is a difference you asked for.

Two axes:

* tyre peak grip D, as a front/rear staircase between --d-min and --d-max, both
  defaulting to the config's own D values;
* corridor margin, from --margins, defaulting to the config's boundary_margin.

Each solve is exported as a controller-reference trajectory CSV named for its
point in the grid, and the score table is rewritten after every solve, so an
interrupted sweep resumes where it stopped.

Usage, from the repo root:
    python research/experiments/skidpad_batch.py
    python research/experiments/skidpad_batch.py --d-max 1.45 --margins 0.40,0.50 --jobs 4

Outputs under --out-dir, named after --prefix (default: the config's track_id):
    <prefix>_F*_R*_m*.csv             one trajectory per solve
    <prefix>_times.csv                score table
    <prefix>_margin_trajectories.png
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
from fast_lto.modes import get_mode
from fast_lto.optimization.global_ocp import solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator
from fast_lto.paths import default_data_root
from fast_lto.pipeline import PipelineConfig, _resolve_path, _splice_segment
from fast_lto.vehicle_models.four_wheel import FourWheelModel

#: Step of the front/rear grip staircase.
D_STEP = 0.05


def _repo_root() -> Path:
    return default_data_root()


def d_sequence(
    d_min: float, d_max: float, anchors: Tuple[float, ...] = ()
) -> List[Tuple[float, float]]:
    """(D_front, D_rear) pairs walking from d_min to d_max.

    The rear leads the front by one step. That is the realistic direction to
    explore -- a rear-limited car is stable, a front-limited one understeers
    off the circle -- so the staircase bumps the rear first and lets the front
    catch up. ``anchors`` appends uniform points (F == R) for a low-grip
    comparison; empty unless asked for.
    """
    seq: List[Tuple[float, float]] = []
    f = r = d_min
    seq.append((round(f, 2), round(r, 2)))
    while f < d_max - 1e-9 or r < d_max - 1e-9:
        if r <= f + 1e-9:  # equal -> bump the rear first
            r = round(r + D_STEP, 2)
        else:  # rear ahead -> the front catches up
            f = round(f + D_STEP, 2)
        seq.append((round(f, 2), round(r, 2)))
    for a in anchors:
        seq.append((round(a, 2), round(a, 2)))
    return seq


@dataclass
class Baseline:
    """What every point in the sweep shares, built once from the config.

    Carries the ``PipelineConfig`` itself rather than a copy of the values
    needed, so the mode can be asked for its own objective weights and
    prescribed segments instead of this file re-deriving them. Re-deriving is
    how it drifted out of step with ``modes.py``.
    """

    config: PipelineConfig
    track: Dict[str, Any]
    time_weights: np.ndarray
    model_params: Dict[str, Any]


def _build_baseline(config_path: Path) -> Baseline:
    from fast_lto.tracks.skidpad import build_skidpad_track

    rc = RunConfig.from_yaml(config_path)
    rc.validate_for_model()
    pc: PipelineConfig = rc.to_pipeline_config()
    pc.repo_root = _repo_root()
    pc.__post_init__()

    sk = pc.skidpad
    map_csv = _resolve_path(pc.repo_root, sk.map_csv)
    ref_csv = _resolve_path(pc.repo_root, sk.reference_csv)
    start_xy = (sk.start_x, sk.start_y) if sk.start_x is not None else None
    track = build_skidpad_track(
        map_csv=map_csv,
        ref_csv=ref_csv,
        ds_m=pc.ds_m,
        entry_exit_halfwidth=sk.entry_exit_halfwidth,
        kappa_blend_m=sk.kappa_blend_m,
        start_xy=start_xy,
    )

    # Asked of the mode rather than recomputed, so the objective this sweep
    # optimises is the one the pipeline optimises.
    time_weights = get_mode("skidpad").time_weights(track, pc)
    assert time_weights is not None  # skidpad always weights its objective

    return Baseline(
        config=pc,
        track=track,
        time_weights=time_weights,
        model_params=rc.vehicle.build_model_params("four_wheel"),
    )


def _config_d_range(model_params: Dict[str, Any]) -> Tuple[float, float]:
    """The config's own grip values, as the (min, max) of the four wheels.

    Used as the default sweep range, which makes a bare run a single point:
    the sweep starts from what the config says rather than from a grip range
    that was chosen for one particular event.
    """
    ds = [float(model_params[k]) for k in ("D_fl", "D_fr", "D_rr", "D_rl")]
    return round(min(ds), 2), round(max(ds), 2)


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
        pc = base.config
        mode = get_mode("skidpad")
        sol = solve_ocp_and_save(
            track=base.track,
            model=model,
            solution_path=sol_json,
            integrator=_make_integrator(pc.integrator_name),
            initial_speed=pc.launch_speed,
            reg_du=pc.reg_u,
            reg_u_l2=pc.reg_u_l2,
            run_config={
                "tag": tag,
                "model_name": "four_wheel",
                "mode": "skidpad",
                "boundary_margin": margin,
            },
            use_normalization=bool(pc.normalize_states_and_inputs),
            solver_verbose=False,
            boundary_margin=margin,
            mode="skidpad",
            time_weights=base.time_weights,
            # The rest of the solver arguments the mode owns, so a new skidpad
            # knob reaches this sweep without an edit here.
            **mode.solver_kwargs(pc),
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
                # Same prescribed segments the pipeline stitches on, asked of
                # the mode rather than reimplemented.
                for plan in mode.splices(pc, base.track, sol):
                    sol = _splice_segment(
                        sol, plan.segment, plan.speed, side=plan.side, timed=plan.timed
                    )
                with sol_json.open("w") as fh:
                    json.dump(sol, fh)
                export_reference_trajectory(sol_json, out_csv)
                result["exported"] = True
            except Exception as exc:  # solve is still valid; just note it
                result["status"] = f"{status}; EXPORT_ERROR: {exc}"
    except Exception as exc:  # keep the sweep alive on a single failure
        result.update(
            score_s=None,
            full_lap_s=None,
            status=f"ERROR: {type(exc).__name__}: {exc}",
            solve_time_s=None,
            exported=False,
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
                f_, r_, m_ = (
                    round(float(row["D_front"]), 2),
                    round(float(row["D_rear"]), 2),
                    round(float(row["margin"]), 2),
                )
            except (KeyError, TypeError, ValueError):
                continue
            # Only treat as done if the solve succeeded and its CSV is present.
            if (
                row.get("status") == "Solve_Succeeded"
                and row.get("csv")
                and (out_dir / row["csv"]).exists()
            ):
                done[(f_, r_, m_)] = {
                    "D_front": f_,
                    "D_rear": r_,
                    "margin": m_,
                    "score_s": float(row["score_s"]) if row.get("score_s") else None,
                    "full_lap_s": float(row["full_lap_s"]) if row.get("full_lap_s") else None,
                    "status": row["status"],
                    "csv": row["csv"],
                    "solve_time_s": float(row["solve_time_s"]) if row.get("solve_time_s") else None,
                    "exported": True,
                }
    return done


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["D_front", "D_rear", "margin", "score_s", "full_lap_s", "status", "csv", "solve_time_s"]
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


def _plot_margins(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    base: Baseline,
    show_combos: List[Tuple[float, float]],
    prefix: str,
) -> Path:
    margins = sorted({r["margin"] for r in rows})
    by_key = {(r["D_front"], r["D_rear"], r["margin"]): r for r in rows}

    sk = base.track["skidpad"]
    c1 = np.array(sk["c_first"])
    c2 = np.array(sk["c_second"])
    R_in, R_out = sk["R_in"], sk["R_out"]
    th = np.linspace(0, 2 * np.pi, 200)

    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(show_combos)))

    fig, axes = plt.subplots(
        1, len(margins), figsize=(6 * len(margins), 6.2), sharex=True, sharey=True
    )
    if len(margins) == 1:
        axes = [axes]

    for ax, m in zip(axes, margins):
        for c in (c1, c2):
            for R, ls in ((R_in, "--"), (R_out, "-")):
                ax.plot(
                    c[0] + R * np.cos(th),
                    c[1] + R * np.sin(th),
                    color="0.6",
                    lw=0.8,
                    ls=ls,
                    zorder=1,
                )
        for (f, r), col in zip(show_combos, colors):
            res = by_key.get((f, r, m))
            if not res or not res.get("exported"):
                continue
            xy = np.loadtxt(out_dir / res["csv"], delimiter=",", skiprows=1, usecols=(0, 1))
            lbl = f"F{f:.2f}/R{r:.2f} ({res['score_s']:.3f}s)"
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.4, label=lbl, zorder=3)
        ax.set_title(f"margin = {m:.2f} m")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("y [m]")
    fig.suptitle(
        f"{prefix} trajectories by corridor margin " "(dashed = inner cone circle, solid = outer)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = out_dir / f"{prefix}_margin_trajectories.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def run(
    config_path: Path,
    jobs: int,
    out_dir: Path,
    prefix: Optional[str] = None,
    margins: Optional[Tuple[float, ...]] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    anchors: Tuple[float, ...] = (),
    initial_speed: Optional[float] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"Building skidpad baseline from {config_path} ...")
    base = _build_baseline(config_path)
    pc = base.config
    if initial_speed is not None:
        pc.initial_speed = float(initial_speed)

    # Anything not being swept comes from the config, so a bare run reproduces
    # exactly what `fast-lto --config <this>` would solve.
    config_d = _config_d_range(base.model_params)
    if d_min is None:
        d_min = config_d[0]
    if d_max is None:
        d_max = config_d[1]
    if margins is None:
        margins = (float(pc.boundary_margin),)
    if prefix is None:
        prefix = str(pc.track_id)

    combos = d_sequence(d_min=d_min, d_max=d_max, anchors=anchors)
    print(
        f"  grip D {d_min:.2f} -> {d_max:.2f} in {D_STEP:.2f} steps "
        f"({len(combos)} combination(s)), margins {', '.join(f'{m:.2f}' for m in margins)}"
    )
    table_csv = out_dir / f"{prefix}_times.csv"
    done = _load_table(table_csv, out_dir)
    results: List[Dict[str, Any]] = list(done.values())

    all_jobs: List[Dict[str, Any]] = []
    for f, r in combos:
        for m in margins:
            if (f, r, m) in done:
                continue
            all_jobs.append(
                {
                    "baseline": base,
                    "D_front": f,
                    "D_rear": r,
                    "margin": m,
                    "out_csv": str(out_dir / f"{_tag(prefix, f, r, m)}.csv"),
                    "scratch_dir": str(scratch),
                }
            )

    total = len(combos) * len(margins)
    print(
        f"{len(done)}/{total} already done; submitting {len(all_jobs)} solves "
        f"on {jobs} worker(s) ..."
    )

    def _record(res: Dict[str, Any]) -> None:
        print(
            f"  [{_tag(prefix, res['D_front'], res['D_rear'], res['margin'])}] "
            f"score={res['score_s']} status={res['status']}"
        )
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
    print(f"  {len([r for r in results if r.get('exported')])} trajectory CSVs " f"in {out_dir}")

    try:
        for p in scratch.glob("*"):
            p.unlink()
        scratch.rmdir()
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=_repo_root() / "configs" / "skidpad.yaml")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel solve workers.")
    ap.add_argument("--out-dir", type=Path, default=_repo_root() / "data" / "output_trajectories")
    ap.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Default: the config's track_id.",
    )
    ap.add_argument(
        "--margins",
        type=str,
        default=None,
        help="Comma-separated boundary margins, e.g. 0.40,0.50. "
        "Default: the config's boundary_margin.",
    )
    ap.add_argument(
        "--d-min",
        type=float,
        default=None,
        help="Lower tyre-grip D for the front/rear staircase. "
        "Default: the config's own D values.",
    )
    ap.add_argument(
        "--d-max",
        type=float,
        default=None,
        help="Upper tyre-grip D. Default: the config's own D values, which "
        "makes a bare run a single point.",
    )
    ap.add_argument(
        "--anchors",
        type=str,
        default=None,
        help="Comma-separated uniform grip levels (F == R) to append, for a "
        "low-grip comparison, e.g. 1.10,1.00",
    )
    ap.add_argument(
        "--initial-speed",
        type=float,
        default=None,
        help="Override the config's initial_speed (m/s). Also sets the "
        "skidpad_lead_in_m constant-speed lead-in, since it must "
        "match the OCP's fixed starting velocity.",
    )
    args = ap.parse_args()

    def _floats(raw: Optional[str]) -> Optional[Tuple[float, ...]]:
        if raw is None:
            return None
        return tuple(float(x) for x in raw.split(",") if x.strip())

    run(
        args.config,
        args.jobs,
        args.out_dir,
        prefix=args.prefix,
        margins=_floats(args.margins),
        d_min=args.d_min,
        d_max=args.d_max,
        anchors=_floats(args.anchors) or (),
        initial_speed=args.initial_speed,
    )


if __name__ == "__main__":
    main()
