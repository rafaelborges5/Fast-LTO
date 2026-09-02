"""
Does warm starting hold up across the knobs we actually turn?

For every (track, perturbation) pair the same OCP is solved twice — once cold
(the default centreline guess) and once seeded from the track's baseline
solution — and the two are compared on convergence, wall clock and objective.

The perturbations are the ones from day-to-day work: the boundary margin, the
top speed, the wheel-force rate limit and tyre grip. None of them changes the
track corridor, so seeding across them should be safe; this experiment is what
says whether that holds in practice, and on which tracks it does not.

    python src/experiments/warm_start_validation.py                 # full matrix
    python src/experiments/warm_start_validation.py --cases ipz     # one track
    python src/experiments/warm_start_validation.py --dry-run       # list the jobs

Outputs (under data/experiments/warm_start/):
    results.csv      one row per solve
    summary.md       cold-vs-warm table + acceptance criteria verdict
    warm_start.png   iterations / wall clock / objective, cold vs warm
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# One BLAS thread per worker. Must be set before numpy is imported. Without it
# each worker spawns its own thread pool, the pool oversubscribes the machine,
# and the wall-clock numbers this experiment exists to compare become a measure
# of core contention rather than of the solver.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np

from fast_lto.config import RunConfig  # noqa: E402
from fast_lto.optimization.global_ocp import (  # noqa: E402
    load_track_with_widths,
    solve_ocp_and_save,
)
from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator  # noqa: E402
from fast_lto.optimization.warm_start import resample_guess  # noqa: E402
from fast_lto.pipeline import (  # noqa: E402
    PipelineConfig,
    _autox_time_weights,
    _extend_track_for_autox,
    run_pipeline,
)
from fast_lto.vehicle_models.four_wheel import FourWheelModel  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "data" / "experiments" / "warm_start"


# --------------------------------------------------------------------------- #
#  Cases and perturbations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    key: str
    track_id: str
    config: str
    baseline_margin: float


CASES: Tuple[Case, ...] = (
    Case("ipz", "ipz_august_3", "configs/autox.yaml", 0.30),
    Case("maisach", "track_boundary_maisach", "configs/trackdrive.yaml", 0.30),
    Case("fscz25", "fscz25_track_boundary", "configs/trackdrive.yaml", 0.30),
)

# (axis, label, model overrides as multipliers, absolute margin)
VARIANTS: Tuple[Tuple[str, str, Dict[str, float], Optional[float]], ...] = (
    ("margin", "margin 0.35", {}, 0.35),
    ("margin", "margin 0.40", {}, 0.40),
    ("margin", "margin 0.45", {}, 0.45),
    ("v_max", "v_max x0.875", {"v_max": 0.875}, None),
    ("v_max", "v_max x1.125", {"v_max": 1.125}, None),
    ("v_max", "v_max x1.25", {"v_max": 1.25}, None),
    ("dFxmax", "dFxmax x0.6", {"dFxmax": 0.6}, None),
    ("dFxmax", "dFxmax x1.5", {"dFxmax": 1.5}, None),
    ("tyre_D", "D x0.9", {"D_fl": 0.9, "D_fr": 0.9, "D_rr": 0.9, "D_rl": 0.9}, None),
    ("tyre_D", "D x1.1", {"D_fl": 1.1, "D_fr": 1.1, "D_rr": 1.1, "D_rl": 1.1}, None),
)


@dataclass
class Baseline:
    """Everything a worker needs to reproduce one track's solve inputs."""

    key: str
    track: Dict[str, Any]
    time_weights: Optional[np.ndarray]
    mode: str
    model_params: Dict[str, Any]
    integrator_name: str
    initial_speed: float
    reg_du: Any
    reg_u_l2: Optional[float]
    terminal_speed: Optional[float]
    autox_timing_offset_m: Optional[float]
    boundary_margin: float
    normalize: bool


def build_baseline(case: Case) -> Baseline:
    rc = RunConfig.from_yaml(REPO / case.config)
    rc.track_id = case.track_id
    rc.validate_for_model()
    pc: PipelineConfig = rc.to_pipeline_config()
    pc.repo_root = REPO
    pc.track_id = case.track_id
    pc.boundary_margin = case.baseline_margin
    pc.__post_init__()

    run_pipeline(pc, start_from="track", end_at="bounds")
    track = load_track_with_widths(pc.track_with_widths_path)

    time_weights = None
    if pc.mode == "autox":
        track = _extend_track_for_autox(
            track,
            pc.autox_extension_m,
            pc.autox_lead_in_m,
            pc.autox_ocp_lead_m,
            timing_offset_m=pc.autox_timing_offset_m,
            start_x=pc.autox_start_x,
            start_y=pc.autox_start_y,
            start_node_offset=pc.autox_start_node_offset,
        )
        time_weights = _autox_time_weights(
            track["arc_lengths"],
            track["autox_base_length_m"],
            pc.autox_timing_offset_m,
            pc.eps_time,
            pc.decel_hold_m,
        )
        track["timed_mask"] = (time_weights >= 1.0 - 1e-9).astype(int).tolist()

    return Baseline(
        key=case.key,
        track=track,
        time_weights=time_weights,
        mode=pc.mode,
        model_params=rc.vehicle.build_model_params("four_wheel"),
        integrator_name=pc.integrator_name,
        initial_speed=float(pc.initial_speed),
        reg_du=pc.reg_u,
        reg_u_l2=pc.reg_u_l2,
        terminal_speed=pc.terminal_speed if pc.mode in ("autox", "skidpad") else None,
        autox_timing_offset_m=(pc.autox_timing_offset_m if pc.mode == "autox" else None),
        boundary_margin=float(case.baseline_margin),
        normalize=bool(pc.normalize_states_and_inputs),
    )


# --------------------------------------------------------------------------- #
#  Solving
# --------------------------------------------------------------------------- #
def _make_integrator(name: str):
    return RK4Integrator() if name == "rk4" else EulerIntegrator()


def _lap_time(sol: Dict[str, Any]) -> Optional[float]:
    prof = sol.get("profiling", {})
    for key in ("autox_lap_time_s", "lap_time_s"):
        if prof.get(key) is not None:
            return float(prof[key])
    return None


def _worst_corner_violation(sol: Dict[str, Any], margin: float, params: Dict) -> float:
    """Post-hoc feasibility check, so 'fast' can be told from 'sloppy'."""
    from fast_lto.vehicle_models.vehicle_base import CornerOffset

    corners = [
        c if isinstance(c, CornerOffset) else CornerOffset(*c) for c in params.get("corners", [])
    ]
    if not corners or "d" not in sol:
        return float("nan")

    d = np.asarray(sol["d"], float)
    psi = np.asarray(sol["psi_err"], float)
    kappa = np.asarray(sol["kappa"], float)
    w_left = np.asarray(sol["w_left"], float) - margin
    w_right = np.asarray(sol["w_right"], float) - margin

    sin_p, cos_p = np.sin(psi), np.cos(psi)
    d_kappa = 1.0 - kappa * d
    d_kappa[np.abs(d_kappa) < 1e-9] = 1e-9
    worst = 0.0
    for c in corners:
        long_proj = c.dx * cos_p - c.dy * sin_p
        d_corner = d + c.dx * sin_p + c.dy * cos_p - 0.5 * kappa / d_kappa * long_proj**2
        slack = (w_left - d_corner) if c.dy >= 0 else (d_corner + w_right)
        worst = max(worst, float(-slack.min()))
    return worst


def solve_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one (variant, start) solve. Must stay picklable: runs in a worker."""
    base: Baseline = job["baseline"]
    params = dict(base.model_params)
    for key, mult in job["model_multipliers"].items():
        params[key] = float(params[key]) * float(mult)
    margin = float(job["boundary_margin"])

    model = FourWheelModel(params=params)
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sol_path = out_dir / f"{base.key}_{job['tag']}_{job['start']}.json"

    row = {
        "case": base.key,
        "mode": base.mode,
        "axis": job["axis"],
        "variant": job["label"],
        "start": job["start"],
        "boundary_margin": margin,
    }

    # Resume: a solve that already produced a solution is not repeated. Each
    # job has a deterministic output path, so an interrupted sweep picks up
    # where it stopped instead of starting over.
    if sol_path.exists() and not job.get("redo"):
        try:
            sol = json.loads(sol_path.read_text())
            prof = sol.get("profiling", {})
            row.update(
                {
                    "status": prof.get("return_status"),
                    "iters": prof.get("iter_count"),
                    "solve_time_s": prof.get("solve_time_s"),
                    "objective": sol.get("obj_val"),
                    "lap_time_s": _lap_time(sol),
                    "corner_violation_m": _worst_corner_violation(sol, margin, params),
                    "solution": str(sol_path),
                    "resumed": True,
                }
            )
            return row
        except Exception:
            pass  # unreadable leftover: fall through and solve it again

    guess = None
    if job["start"] == "warm":
        seed = json.loads(Path(job["seed_path"]).read_text())
        guess = resample_guess(seed, base.track, model, base.normalize)
    t0 = time.perf_counter()
    try:
        sol = solve_ocp_and_save(
            track=base.track,
            model=model,
            solution_path=sol_path,
            integrator=_make_integrator(base.integrator_name),
            initial_speed=base.initial_speed,
            reg_du=base.reg_du,
            reg_u_l2=base.reg_u_l2,
            run_config={"tag": job["tag"], "start": job["start"]},
            use_normalization=base.normalize,
            solver_verbose=False,
            boundary_margin=margin,
            mode=base.mode,
            time_weights=base.time_weights,
            terminal_speed=base.terminal_speed,
            autox_timing_offset_m=base.autox_timing_offset_m,
            initial_guess=guess,
        )
        prof = sol.get("profiling", {})
        row.update(
            {
                "status": prof.get("return_status"),
                "iters": prof.get("iter_count"),
                "solve_time_s": prof.get("solve_time_s"),
                "objective": sol.get("obj_val"),
                "lap_time_s": _lap_time(sol),
                "corner_violation_m": _worst_corner_violation(sol, margin, params),
                "solution": str(sol_path),
                "resumed": False,
            }
        )
    except Exception as exc:  # a single failure must not kill the sweep
        row.update(
            {
                "status": f"FAILED: {type(exc).__name__}",
                "iters": None,
                "solve_time_s": time.perf_counter() - t0,
                "objective": None,
                "lap_time_s": None,
                "corner_violation_m": None,
                "solution": None,
                "resumed": False,
            }
        )
    return row


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: Dict[str, Any]) -> None:
    """Persist one result immediately, so an interrupted sweep loses nothing."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def summarise(rows: List[Dict[str, Any]]) -> str:
    """Pair cold with warm and check the acceptance criteria."""
    paired: Dict[Tuple[str, str], Dict[str, Dict]] = {}
    for row in rows:
        if row["variant"] == "baseline":
            continue
        paired.setdefault((row["case"], row["variant"]), {})[row["start"]] = row

    lines = [
        "| case | variant | cold status | warm status | iters c/w | "
        "time c/w [s] | objective c/w | warm - cold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    regressions, warm_worse, better, comparable = [], [], 0, 0
    for (case, variant), pair in sorted(paired.items()):
        cold, warm = pair.get("cold"), pair.get("warm")
        if cold is None or warm is None:
            continue
        cold_ok = cold["status"] == "Solve_Succeeded"
        warm_ok = warm["status"] == "Solve_Succeeded"
        if cold_ok and not warm_ok:
            regressions.append((case, variant))
        delta = ""
        if cold_ok and warm_ok:
            d = float(warm["objective"]) - float(cold["objective"])
            delta = f"{d:+.4f}"
            if d > 1e-3:
                warm_worse.append((case, variant, d))
            elif d < -1e-3:
                better += 1
            else:
                comparable += 1
        lines.append(
            f"| {case} | {variant} | {cold['status']} | {warm['status']} | "
            f"{cold['iters']}/{warm['iters']} | "
            f"{_fmt(cold['solve_time_s'])}/{_fmt(warm['solve_time_s'])} | "
            f"{_fmt(cold['objective'], 3)}/{_fmt(warm['objective'], 3)} | {delta} |"
        )

    ok = [r for r in rows if r["status"] == "Solve_Succeeded"]
    cold_iters = [r["iters"] for r in ok if r["start"] == "cold" and r["iters"]]
    warm_iters = [r["iters"] for r in ok if r["start"] == "warm" and r["iters"]]
    viol = [
        r["corner_violation_m"]
        for r in ok
        if r["corner_violation_m"] is not None and np.isfinite(r["corner_violation_m"])
    ]

    n_pairs = better + comparable + len(warm_worse)
    lines += [
        "",
        "## Acceptance criteria",
        "",
        f"1. warm never fails where cold succeeds: "
        f"{'PASS' if not regressions else 'FAIL ' + str(regressions)}",
        f"2. warm objective <= cold + 1e-3 in >= 90% of pairs: "
        f"{(better + comparable)}/{n_pairs} "
        f"({'PASS' if n_pairs and (better + comparable) / n_pairs >= 0.9 else 'FAIL'})"
        + (f"; warm worse at {warm_worse}" if warm_worse else ""),
        f"3. median iterations cold {np.median(cold_iters) if cold_iters else float('nan'):.0f} "
        f"vs warm {np.median(warm_iters) if warm_iters else float('nan'):.0f}: "
        f"{'PASS' if warm_iters and cold_iters and np.median(warm_iters) < np.median(cold_iters) else 'FAIL'}",
        f"4. worst post-hoc corner violation over all solves: "
        f"{max(viol) if viol else float('nan'):.2e} m",
    ]
    return "\n".join(lines)


def _fmt(value, digits: int = 1) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def plot(rows: List[Dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired: Dict[Tuple[str, str], Dict[str, Dict]] = {}
    for row in rows:
        if row["variant"] == "baseline" or row["status"] != "Solve_Succeeded":
            continue
        paired.setdefault((row["case"], row["variant"]), {})[row["start"]] = row
    pairs = [p for p in paired.values() if "cold" in p and "warm" in p]
    if not pairs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Warm start vs cold start — real tracks, one knob at a time")
    cases = sorted({p["cold"]["case"] for p in pairs})
    colors = dict(zip(cases, plt.cm.tab10.colors))

    for ax, key, label, log in zip(
        axes,
        ("iters", "solve_time_s", "objective"),
        ("IPOPT iterations", "wall clock [s]", "objective"),
        (True, True, False),
    ):
        for p in pairs:
            ax.scatter(
                p["cold"][key],
                p["warm"][key],
                s=34,
                color=colors[p["cold"]["case"]],
                alpha=0.85,
                label=p["cold"]["case"],
            )
        if log:
            # Cost spans two orders of magnitude across these tracks, and one
            # cold wall clock is a swap-thrashing artefact; log axes keep the
            # bulk readable without dropping the outlier.
            ax.set_xscale("log")
            ax.set_yscale("log")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"cold {label}")
        ax.set_ylabel(f"warm {label}")
        ax.set_title(f"{label} (below the line = warm wins)")
        ax.grid(alpha=0.3, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=[c.key for c in CASES])
    ap.add_argument("--axes", nargs="*", default=sorted({v[0] for v in VARIANTS}))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cases = [c for c in CASES if c.key in args.cases]
    if not cases:
        raise SystemExit(f"no cases matched {args.cases}")
    variants = [v for v in VARIANTS if v[0] in args.axes]

    if args.dry_run:
        for case in cases:
            print(f"{case.key}: baseline margin {case.baseline_margin}")
            for axis, label, _mults, margin in variants:
                print(
                    f"    {axis:8s} {label:16s} margin "
                    f"{margin if margin is not None else case.baseline_margin}"
                )
        print(
            f"\n{len(cases)} cases x ({len(variants)} variants x 2 + 1 baseline) "
            f"= {len(cases) * (len(variants) * 2 + 1)} solves"
        )
        return

    rows: List[Dict[str, Any]] = []
    baselines: Dict[str, Baseline] = {}
    seeds: Dict[str, Path] = {}

    # Phase 1 — one cold baseline per track, which every warm run seeds from.
    # Track preparation is cheap and touches shared files, so it stays in the
    # parent; only the solves go to the pool.
    print("=== phase 1: baselines ===", flush=True)
    baseline_jobs = []
    for case in cases:
        base = build_baseline(case)
        baselines[case.key] = base
        baseline_jobs.append(
            {
                "baseline": base,
                "axis": "baseline",
                "label": "baseline",
                "tag": "baseline",
                "start": "cold",
                "model_multipliers": {},
                "boundary_margin": case.baseline_margin,
                "out_dir": str(out_dir),
            }
        )

    progress_csv = out_dir / "progress.csv"
    with mp.Pool(processes=min(len(baseline_jobs), max(1, args.workers))) as pool:
        for row in pool.imap_unordered(solve_job, baseline_jobs):
            rows.append(row)
            append_row(progress_csv, row)
            print(
                f"  {row['case']}: {row['status']} in "
                f"{_fmt(row['solve_time_s'])} s"
                f"{' (resumed)' if row.get('resumed') else ''}",
                flush=True,
            )
            if row["solution"] is None:
                print(f"  {row['case']}: baseline failed, skipping this case")
                continue
            seeds[row["case"]] = Path(row["solution"])

    # Phase 2 — every variant, cold and warm, in parallel.
    jobs: List[Dict[str, Any]] = []
    for case in cases:
        if case.key not in seeds:
            continue
        base = baselines[case.key]
        for axis, label, mults, margin in variants:
            tag = label.replace(" ", "_").replace(".", "p")
            for start in ("cold", "warm"):
                jobs.append(
                    {
                        "baseline": base,
                        "axis": axis,
                        "label": label,
                        "tag": tag,
                        "start": start,
                        "model_multipliers": mults,
                        "boundary_margin": (margin if margin is not None else case.baseline_margin),
                        "seed_path": str(seeds[case.key]),
                        "out_dir": str(out_dir),
                    }
                )

    print(f"\n=== phase 2: {len(jobs)} solves on {args.workers} workers ===", flush=True)
    done = 0
    with mp.Pool(processes=max(1, args.workers)) as pool:
        for row in pool.imap_unordered(solve_job, jobs):
            rows.append(row)
            append_row(progress_csv, row)
            done += 1
            print(
                f"  [{done}/{len(jobs)}] {row['case']:8s} {row['variant']:16s} "
                f"{row['start']:4s} {row['status']} iters={row['iters']} "
                f"t={_fmt(row['solve_time_s'])}s obj={_fmt(row['objective'], 3)}"
                f"{' (resumed)' if row.get('resumed') else ''}",
                flush=True,
            )

    write_csv(out_dir / "results.csv", rows)
    summary = summarise(rows)
    (out_dir / "summary.md").write_text(summary)
    plot(rows, out_dir / "warm_start.png")
    print("\n" + summary)
    print(f"\nwrote {out_dir}/results.csv, summary.md, warm_start.png")


if __name__ == "__main__":
    main()
