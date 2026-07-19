from __future__ import annotations

"""
Batch skidpad LTO generation for FSCZ.

Sweeps the tyre peak-grip D over a front/rear evolution and three corridor
margins, solving the four-wheel skidpad OCP for each combination.  Every run is
exported as a controller-reference trajectory CSV whose name encodes the D
parameters and the margin, e.g.::

    data/output_trajectories/fscz_skidpad_F1.30_R1.35_m0.50.csv

D evolution (rear leads each 0.05 step, 1.20 -> 1.50), plus two uniform low-grip
anchors at F=R=1.10 and F=R=1.00.  Margins: 0.40, 0.50, 0.60.

Usage (from repo root):
    PYTHONPATH=src python src/experiments/fscz_skidpad_batch.py --jobs 4

Outputs:
    data/output_trajectories/fscz_skidpad_F*_R*_m*.csv   one per solve
    data/output_trajectories/fscz_skidpad_times.csv       score table
    data/output_trajectories/fscz_skidpad_margin_*.png    trajectory plots
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RunConfig
from export.trajectory import export_reference_trajectory
from optimization.global_ocp import solve_ocp_and_save
from optimization.integrators import EulerIntegrator, RK4Integrator
from pipeline import _resolve_path, PipelineConfig
from vehicle_models.four_wheel import FourWheelModel

MARGINS = (0.40, 0.50, 0.60)
PREFIX = "fscz_skidpad"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def d_sequence() -> List[Tuple[float, float]]:
    """(D_front, D_rear) pairs: rear leads each 0.05 step from 1.20 to 1.50,
    then two uniform low-grip anchors."""
    seq: List[Tuple[float, float]] = []
    f = r = 1.20
    seq.append((f, r))
    while f < 1.50 - 1e-9 or r < 1.50 - 1e-9:
        if r <= f + 1e-9:  # equal -> bump rear first
            r = round(r + 0.05, 2)
        else:  # rear ahead -> front catches up
            f = round(f + 0.05, 2)
        seq.append((round(f, 2), round(r, 2)))
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


def _build_baseline(config_path: Path) -> Baseline:
    from tracks.skidpad import build_skidpad_track

    rc = RunConfig.from_yaml(config_path)
    rc.validate_for_model()
    pc: PipelineConfig = rc.to_pipeline_config()
    pc.repo_root = _repo_root()
    pc.__post_init__()

    map_csv = _resolve_path(pc.repo_root, pc.skidpad_map_csv)
    ref_csv = _resolve_path(pc.repo_root, pc.skidpad_reference_csv)
    track = build_skidpad_track(
        map_csv=map_csv,
        ref_csv=ref_csv,
        ds_m=pc.ds_m,
        entry_exit_halfwidth=pc.entry_exit_halfwidth,
        kappa_blend_m=pc.kappa_blend_m,
    )

    mask = np.asarray(track["timed_mask"], dtype=float)
    decel = np.asarray(track.get("decel_mask", np.zeros_like(mask)), dtype=float)
    time_weights = np.where(mask > 0.5, 1.0, float(pc.eps_time))
    time_weights = np.where(decel > 0.5, 0.0, time_weights)

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
    )


def _make_integrator(name: str):
    return RK4Integrator() if name == "rk4" else EulerIntegrator()


def _tag(f: float, r: float, m: float) -> str:
    return f"{PREFIX}_F{f:.2f}_R{r:.2f}_m{m:.2f}"


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


def _print_table(rows: List[Dict[str, Any]]) -> None:
    margins = sorted({r["margin"] for r in rows})
    combos = sorted({(r["D_front"], r["D_rear"]) for r in rows})
    by_key = {(r["D_front"], r["D_rear"], r["margin"]): r for r in rows}

    print("\n" + "=" * (24 + 12 * len(margins)))
    print("FSCZ skidpad FS score [s]  (avg of the two timed laps)")
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
                  base: Baseline, show_combos: List[Tuple[float, float]]) -> Path:
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
    fig.suptitle("FSCZ skidpad trajectories by corridor margin "
                 "(dashed = inner cone circle, solid = outer)", fontsize=12)
    fig.tight_layout()
    out_png = out_dir / f"{PREFIX}_margin_trajectories.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def run(config_path: Path, jobs: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"Building skidpad baseline from {config_path} ...")
    base = _build_baseline(config_path)

    combos = d_sequence()
    table_csv = out_dir / f"{PREFIX}_times.csv"
    done = _load_table(table_csv, out_dir)
    results: List[Dict[str, Any]] = list(done.values())

    all_jobs: List[Dict[str, Any]] = []
    for (f, r) in combos:
        for m in MARGINS:
            if (f, r, m) in done:
                continue
            all_jobs.append({
                "baseline": base, "D_front": f, "D_rear": r, "margin": m,
                "out_csv": str(out_dir / f"{_tag(f, r, m)}.csv"),
                "scratch_dir": str(scratch),
            })

    total = len(combos) * len(MARGINS)
    print(f"{len(done)}/{total} already done; submitting {len(all_jobs)} solves "
          f"on {jobs} worker(s) ...")

    def _record(res: Dict[str, Any]) -> None:
        print(f"  [{_tag(res['D_front'], res['D_rear'], res['margin'])}] "
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
    _print_table(results)

    # Show a representative spread of grip levels on the trajectory plot.
    show = [(1.00, 1.00), (1.20, 1.20), (1.35, 1.35), (1.50, 1.50)]
    show = [c for c in show if c in set((r["D_front"], r["D_rear"]) for r in results)]
    png = _plot_margins(results, out_dir, base, show)

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
    args = ap.parse_args()
    run(args.config, args.jobs, args.out_dir)


if __name__ == "__main__":
    main()
