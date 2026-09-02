from __future__ import annotations

"""
One-at-a-time (OAT) sensitivity study for the skidpad LTO.

The objective of interest is the FS *score*: the average of the two timed-lap
times (``profiling.skidpad_score_s``).  We perturb each parameter individually
by a fixed relative amount around the baseline config, re-solve the four-wheel
OCP, and measure how the score moves.  Cross-parameter ranking uses the
dimensionless *elasticity*

    E = (dscore / score_base) / (dp / p_base)

estimated from a centred low/high pair, so a value of e.g. ``-0.8`` means a
+1% increase in the parameter lowers the average lap time by 0.8%.

Parameters studied (four-wheel model only):
    m                mass
    Iz               yaw inertia
    reg_u_l2         L2 input-magnitude regularisation weight
    D_rear           rear tyre peak-grip (D_rr and D_rl scaled together)
    boundary_margin  corridor shrink margin on each side
    C_l              aero downforce coefficient (baseline 5.54)
    D_front          front tyre peak-grip (D_fl and D_fr) -- for front/rear contrast
    C_d              aero drag coefficient
    h                CG height (drives longitudinal/lateral load transfer)

Usage (from repo root):
    PYTHONPATH=src python src/experiments/skidpad_sensitivity.py \
        --config configs/skidpad.yaml --delta 0.15 --jobs 4

Outputs (under data/sensitivity/ by default):
    skidpad_sensitivity.csv     one row per solve (baseline + perturbations)
    skidpad_sensitivity.png     tornado plot ranked by |elasticity|
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
from fast_lto.optimization.global_ocp import solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator
from fast_lto.pipeline import PipelineConfig, _resolve_path
from fast_lto.vehicle_models.four_wheel import FourWheelModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------
#
# A "knob" describes how a single perturbation is applied.  ``kind`` selects
# where the value lives:
#   "model"  -> override one or more keys in the four-wheel params dict
#   "solve"  -> override a keyword argument to solve_ocp_and_save
# ``keys`` lists every param the knob scales together (e.g. both rear D's).


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str  # "model" | "solve"
    keys: Tuple[str, ...]
    label: str


KNOBS: List[Knob] = [
    Knob("m", "model", ("m",), "mass m"),
    Knob("Iz", "model", ("Iz",), "yaw inertia Iz"),
    Knob("reg_u_l2", "solve", ("reg_u_l2",), "reg_u_l2"),
    Knob("D_rear", "model", ("D_rr", "D_rl"), "rear grip D_rear"),
    Knob("boundary_margin", "solve", ("boundary_margin",), "margin"),
    Knob("C_l", "model", ("C_l",), "downforce C_l"),
    Knob("D_front", "model", ("D_fl", "D_fr"), "front grip D_front"),
    Knob("C_d", "model", ("C_d",), "drag C_d"),
    Knob("h", "model", ("h",), "CG height h"),
]


# ---------------------------------------------------------------------------
# Baseline / shared inputs
# ---------------------------------------------------------------------------


@dataclass
class Baseline:
    track: Dict[str, Any]
    time_weights: np.ndarray
    model_params: Dict[str, Any]
    integrator_name: str
    initial_speed: float
    reg_du: Any
    reg_u_l2: Optional[float]
    boundary_margin: float
    terminal_speed: Optional[float]
    normalize: bool


def _build_baseline(config_path: Path) -> Baseline:
    """Build the skidpad track once and gather all baseline solve inputs."""
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

    model_params = rc.vehicle.build_model_params("four_wheel")

    return Baseline(
        track=track,
        time_weights=time_weights,
        model_params=model_params,
        integrator_name=pc.integrator_name,
        initial_speed=float(pc.initial_speed),
        reg_du=pc.reg_u,
        reg_u_l2=pc.reg_u_l2,
        boundary_margin=float(pc.boundary_margin),
        terminal_speed=pc.terminal_speed,
        normalize=bool(pc.normalize_states_and_inputs),
    )


# ---------------------------------------------------------------------------
# Single solve
# ---------------------------------------------------------------------------


def _make_integrator(name: str):
    return RK4Integrator() if name == "rk4" else EulerIntegrator()


def _solve_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Solve one perturbed configuration and return the score + metadata.

    ``job`` carries the baseline payload plus the model/solve overrides for this
    run.  Runs as a worker process, so everything in it must be picklable.
    """
    base: Baseline = job["baseline"]
    model_overrides: Dict[str, Any] = job["model_overrides"]
    reg_u_l2 = job["reg_u_l2"]
    boundary_margin = job["boundary_margin"]
    tag = job["tag"]

    params = dict(base.model_params)
    params.update(model_overrides)
    model = FourWheelModel(params=params)
    integrator = _make_integrator(base.integrator_name)

    sol_path = Path(job["scratch_dir"]) / f"sol_{tag}.json"

    try:
        sol = solve_ocp_and_save(
            track=base.track,
            model=model,
            solution_path=sol_path,
            integrator=integrator,
            initial_speed=base.initial_speed,
            reg_du=base.reg_du,
            reg_u_l2=reg_u_l2,
            run_config={"tag": tag},
            use_normalization=base.normalize,
            solver_verbose=False,
            boundary_margin=boundary_margin,
            mode="skidpad",
            time_weights=base.time_weights,
            terminal_speed=base.terminal_speed,
        )
        prof = sol.get("profiling", {})
        result = {
            "score_s": prof.get("skidpad_score_s"),
            "timed_time_s": prof.get("pure_timed_time_s"),
            "full_time_s": prof.get("lap_time_s"),
            "status": prof.get("return_status"),
            "solve_time_s": prof.get("solve_time_s"),
            "iters": prof.get("iter_count"),
        }
    except Exception as exc:  # keep the sweep alive on a single failure
        result = {
            "score_s": None,
            "timed_time_s": None,
            "full_time_s": None,
            "status": f"ERROR: {type(exc).__name__}: {exc}",
            "solve_time_s": None,
            "iters": None,
        }
    finally:
        try:
            sol_path.unlink()
        except OSError:
            pass

    result.update(
        {
            "tag": tag,
            "knob": job["knob"],
            "param_value": job["param_value"],
            "multiplier": job["multiplier"],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------


def _baseline_value(base: Baseline, knob: Knob) -> float:
    if knob.kind == "model":
        return float(base.model_params[knob.keys[0]])
    if knob.name == "reg_u_l2":
        return float(base.reg_u_l2 if base.reg_u_l2 is not None else 0.0)
    if knob.name == "boundary_margin":
        return float(base.boundary_margin)
    raise ValueError(f"Unhandled knob {knob.name}")


def _make_job(
    base: Baseline,
    knob: Knob,
    multiplier: float,
    scratch_dir: Path,
) -> Dict[str, Any]:
    base_val = _baseline_value(base, knob)
    new_val = base_val * multiplier

    model_overrides: Dict[str, Any] = {}
    reg_u_l2 = base.reg_u_l2
    boundary_margin = base.boundary_margin

    if knob.kind == "model":
        for k in knob.keys:
            model_overrides[k] = float(base.model_params[k]) * multiplier
    elif knob.name == "reg_u_l2":
        reg_u_l2 = new_val
    elif knob.name == "boundary_margin":
        boundary_margin = new_val

    tag = f"{knob.name}_x{multiplier:.2f}".replace(".", "p")
    return {
        "baseline": base,
        "knob": knob.name,
        "param_value": new_val,
        "multiplier": multiplier,
        "model_overrides": model_overrides,
        "reg_u_l2": reg_u_l2,
        "boundary_margin": boundary_margin,
        "scratch_dir": str(scratch_dir),
        "tag": tag,
    }


def _baseline_job(base: Baseline, scratch_dir: Path) -> Dict[str, Any]:
    return {
        "baseline": base,
        "knob": "baseline",
        "param_value": float("nan"),
        "multiplier": 1.0,
        "model_overrides": {},
        "reg_u_l2": base.reg_u_l2,
        "boundary_margin": base.boundary_margin,
        "scratch_dir": str(scratch_dir),
        "tag": "baseline",
    }


# ---------------------------------------------------------------------------
# Analysis / output
# ---------------------------------------------------------------------------


def _elasticity(score_lo: float, score_hi: float, score_base: float, delta: float) -> float:
    """Centred elasticity over a +-delta multiplicative perturbation."""
    if score_base in (None, 0) or score_lo is None or score_hi is None:
        return float("nan")
    return ((score_hi - score_lo) / score_base) / (2.0 * delta)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "knob",
        "tag",
        "multiplier",
        "param_value",
        "score_s",
        "timed_time_s",
        "full_time_s",
        "status",
        "solve_time_s",
        "iters",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _tornado_plot(
    path: Path,
    summary: List[Dict[str, Any]],
    score_base: float,
    delta: float,
) -> None:
    # Rank by absolute lap-time swing (seconds), most influential at the top.
    summary = sorted(summary, key=lambda d: abs(d["swing_s"]), reverse=True)
    labels = [d["label"] for d in summary]
    lo = [d["score_lo"] - score_base for d in summary]
    hi = [d["score_hi"] - score_base for d in summary]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(labels) + 2))
    for yi, lo_i, hi_i in zip(y, lo, hi):
        ax.plot([lo_i, hi_i], [yi, yi], color="0.7", lw=2, zorder=1)
    ax.scatter(lo, y, color="tab:blue", zorder=3, label=f"-{delta*100:.0f}%")
    ax.scatter(hi, y, color="tab:red", zorder=3, label=f"+{delta*100:.0f}%")
    for yi, d in zip(y, summary):
        ax.annotate(
            f"E={d['elasticity']:+.2f}",
            xy=(0, yi),
            xytext=(0, yi + 0.18),
            ha="center",
            fontsize=8,
            color="0.25",
        )
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"change in average timed-lap score [s]  (baseline = {score_base:.3f} s)")
    ax.set_title(
        f"Skidpad score sensitivity (OAT, +-{delta*100:.0f}% per parameter)\n"
        "annotation E = elasticity (dscore%/dparam%); |swing| sets the ranking"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", ls="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config_path: Path, delta: float, jobs: int, out_dir: Path) -> None:
    scratch_dir = out_dir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building skidpad track and baseline from {config_path} ...")
    base = _build_baseline(config_path)

    multipliers = (1.0 - delta, 1.0 + delta)
    all_jobs: List[Dict[str, Any]] = [_baseline_job(base, scratch_dir)]
    for knob in KNOBS:
        for m in multipliers:
            all_jobs.append(_make_job(base, knob, m, scratch_dir))

    print(
        f"Submitting {len(all_jobs)} solves "
        f"({len(KNOBS)} knobs x2 + baseline) on {jobs} worker(s) ..."
    )

    results: List[Dict[str, Any]] = []
    if jobs <= 1:
        for j in all_jobs:
            r = _solve_one(j)
            print(f"  [{r['tag']:>22}] score={r['score_s']} status={r['status']}")
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(_solve_one, all_jobs):
                print(f"  [{r['tag']:>22}] score={r['score_s']} status={r['status']}")
                results.append(r)

    by_tag = {r["tag"]: r for r in results}
    score_base = by_tag["baseline"]["score_s"]
    if score_base is None:
        raise RuntimeError("Baseline solve failed; cannot compute sensitivities.")

    # Build per-knob summary.
    summary: List[Dict[str, Any]] = []
    for knob in KNOBS:
        lo_tag = f"{knob.name}_x{multipliers[0]:.2f}".replace(".", "p")
        hi_tag = f"{knob.name}_x{multipliers[1]:.2f}".replace(".", "p")
        s_lo = by_tag[lo_tag]["score_s"]
        s_hi = by_tag[hi_tag]["score_s"]
        base_val = _baseline_value(base, knob)
        elast = _elasticity(s_lo, s_hi, score_base, delta)
        swing = (s_hi - s_lo) if (s_lo is not None and s_hi is not None) else float("nan")
        summary.append(
            {
                "knob": knob.name,
                "label": knob.label,
                "base_value": base_val,
                "score_lo": s_lo,
                "score_hi": s_hi,
                "swing_s": swing,
                "elasticity": elast,
            }
        )

    # CSV with every raw solve.
    csv_path = out_dir / "skidpad_sensitivity.csv"
    _write_csv(csv_path, results)

    # Tornado plot.
    png_path = out_dir / "skidpad_sensitivity.png"
    _tornado_plot(png_path, summary, score_base, delta)

    # JSON summary for programmatic use.
    json_path = out_dir / "skidpad_sensitivity_summary.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": str(config_path),
                "delta": delta,
                "score_base_s": score_base,
                "knobs": summary,
            },
            f,
            indent=2,
        )

    # Console ranking.
    ranked = sorted(summary, key=lambda d: abs(d["swing_s"]), reverse=True)
    print("\n" + "=" * 78)
    print(f"Baseline average timed-lap score: {score_base:.4f} s   (+-{delta*100:.0f}% OAT)")
    print("=" * 78)
    hdr = f"{'parameter':<18}{'base':>10}{'score-':>10}{'score+':>10}{'swing[s]':>11}{'elasticity':>12}"
    print(hdr)
    print("-" * 78)
    for d in ranked:
        slo = f"{d['score_lo']:.3f}" if d["score_lo"] is not None else "  fail"
        shi = f"{d['score_hi']:.3f}" if d["score_hi"] is not None else "  fail"
        print(
            f"{d['label']:<18}{d['base_value']:>10.3f}{slo:>10}{shi:>10}"
            f"{d['swing_s']:>11.4f}{d['elasticity']:>12.3f}"
        )
    print("-" * 78)
    top = ranked[0]
    print(
        f"\nMost influential parameter: {top['label']} "
        f"(|swing| = {abs(top['swing_s']):.4f} s over +-{delta*100:.0f}%, "
        f"elasticity = {top['elasticity']:+.3f})."
    )
    print(f"\nWrote:\n  {csv_path}\n  {png_path}\n  {json_path}")

    # Tidy scratch.
    try:
        for p in scratch_dir.glob("*"):
            p.unlink()
        scratch_dir.rmdir()
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=_repo_root() / "configs" / "skidpad.yaml")
    ap.add_argument(
        "--delta",
        type=float,
        default=0.15,
        help="Relative one-at-a-time perturbation (fraction). Default 0.15 (+-15%%).",
    )
    ap.add_argument("--jobs", type=int, default=4, help="Parallel solve workers.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "data" / "sensitivity",
        help="Output directory for CSV/PNG/JSON.",
    )
    args = ap.parse_args()
    run(args.config, args.delta, args.jobs, args.out_dir)


if __name__ == "__main__":
    main()
