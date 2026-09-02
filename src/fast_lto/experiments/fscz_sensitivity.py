from __future__ import annotations

"""
One-at-a-time (OAT) lap-time sensitivity study for the trackdrive LTO.

Track-agnostic: defaults to the FSCZ boundary but ``--track-id`` selects any
track under ``data/tracks/{track_id}.csv``.  We perturb each parameter
individually by a fixed relative amount around the baseline config, re-solve the
four-wheel OCP (euler integrator, trackdrive mode), and measure how the lap time
(``profiling.lap_time_s``) moves.  Cross-parameter ranking uses the dimensionless
*elasticity*

    E = (dlap / lap_base) / (dp / p_base)

estimated from a centred low/high pair, so a value of e.g. ``-0.8`` means a +1%
increase in the parameter lowers the lap time by 0.8%.

The absolute lap-time calculation is known to be slightly off, but that is
irrelevant here: every solve uses the identical method, so relative
sensitivities are consistent.

Parameters studied (four-wheel model only):
    v_max            top-speed bound
    D_front          front tyre peak-grip (D_fl and D_fr scaled together)
    D_rear           rear tyre peak-grip (D_rr and D_rl scaled together)
    dFxmax           per-wheel longitudinal force-rate limit
    boundary_margin  corridor shrink margin on each side

Usage (from repo root):
    PYTHONPATH=src python src/experiments/fscz_sensitivity.py \
        --config configs/trackdrive.yaml --track-id fscz25_track_boundary \
        --delta 0.15 --jobs 4

Outputs (under data/sensitivity/ by default; {track_id} prefixes every file):
    {track_id}_sensitivity.csv           one row per solve (baseline + perturbations)
    {track_id}_sensitivity.png           tornado plot ranked by |swing|
    {track_id}_sensitivity_curves.png    per-knob lap-time response curves
    {track_id}_sensitivity_summary.json  machine-readable summary
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
from fast_lto.optimization.global_ocp import load_track_with_widths, solve_ocp_and_save
from fast_lto.optimization.integrators import EulerIntegrator, RK4Integrator
from fast_lto.pipeline import PipelineConfig, run_pipeline
from fast_lto.vehicle_models.four_wheel import FourWheelModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------
#
# A "knob" describes how a single perturbation is applied.  ``kind`` selects
# where the value lives:
#   "model"  -> override one or more keys in the four-wheel params dict
#   "solve"  -> override a keyword argument to solve_ocp_and_save
# ``keys`` lists every param the knob scales together (e.g. both front D's).


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str  # "model" | "solve"
    keys: Tuple[str, ...]
    label: str


KNOBS: List[Knob] = [
    Knob("v_max", "model", ("v_max",), "top speed v_max"),
    Knob("D_front", "model", ("D_fl", "D_fr"), "front grip D_front"),
    Knob("D_rear", "model", ("D_rr", "D_rl"), "rear grip D_rear"),
    Knob("dFxmax", "model", ("dFxmax",), "force rate dFxmax"),
    Knob("boundary_margin", "solve", ("boundary_margin",), "margin"),
]


# ---------------------------------------------------------------------------
# Baseline / shared inputs
# ---------------------------------------------------------------------------


@dataclass
class Baseline:
    track: Dict[str, Any]
    model_params: Dict[str, Any]
    integrator_name: str
    initial_speed: float
    reg_du: Any
    reg_u_l2: Optional[float]
    boundary_margin: float
    normalize: bool


def _build_baseline(config_path: Path, track_id: str) -> Baseline:
    """Build the track once and gather all baseline solve inputs."""
    rc = RunConfig.from_yaml(config_path)
    rc.validate_for_model()
    pc: PipelineConfig = rc.to_pipeline_config()
    pc.repo_root = _repo_root()
    pc.track_id = track_id
    pc.__post_init__()

    # Build the track-with-widths JSON once (track -> spline -> bounds).
    run_pipeline(pc, start_from="track", end_at="bounds")
    track = load_track_with_widths(pc.track_with_widths_path)

    model_params = rc.vehicle.build_model_params("four_wheel")

    return Baseline(
        track=track,
        model_params=model_params,
        integrator_name=pc.integrator_name,
        initial_speed=float(pc.initial_speed),
        reg_du=pc.reg_u,
        reg_u_l2=pc.reg_u_l2,
        boundary_margin=float(pc.boundary_margin),
        normalize=bool(pc.normalize_states_and_inputs),
    )


# ---------------------------------------------------------------------------
# Single solve
# ---------------------------------------------------------------------------


def _make_integrator(name: str):
    return RK4Integrator() if name == "rk4" else EulerIntegrator()


def _solve_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Solve one perturbed configuration and return the lap time + metadata.

    ``job`` carries the baseline payload plus the model/solve overrides for this
    run.  Runs as a worker process, so everything in it must be picklable.
    """
    base: Baseline = job["baseline"]
    model_overrides: Dict[str, Any] = job["model_overrides"]
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
            reg_u_l2=base.reg_u_l2,
            run_config={"tag": tag},
            use_normalization=base.normalize,
            solver_verbose=False,
            boundary_margin=boundary_margin,
            mode="trackdrive",
            time_weights=None,
            terminal_speed=None,
        )
        prof = sol.get("profiling", {})
        result = {
            "score_s": prof.get("lap_time_s"),
            "full_time_s": prof.get("lap_time_s"),
            "status": prof.get("return_status"),
            "solve_time_s": prof.get("solve_time_s"),
            "iters": prof.get("iter_count"),
        }
    except Exception as exc:  # keep the sweep alive on a single failure
        result = {
            "score_s": None,
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
    boundary_margin = base.boundary_margin

    if knob.kind == "model":
        for k in knob.keys:
            model_overrides[k] = float(base.model_params[k]) * multiplier
    elif knob.name == "boundary_margin":
        boundary_margin = new_val

    tag = f"{knob.name}_x{multiplier:.2f}".replace(".", "p")
    return {
        "baseline": base,
        "knob": knob.name,
        "param_value": new_val,
        "multiplier": multiplier,
        "model_overrides": model_overrides,
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
    track_id: str,
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
    ax.set_xlabel(f"change in lap time [s]  (baseline = {score_base:.3f} s)")
    ax.set_title(
        f"{track_id} trackdrive lap-time sensitivity (OAT, +-{delta*100:.0f}% per parameter)\n"
        "annotation E = elasticity (dlap%/dparam%); |swing| sets the ranking"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", ls="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _response_curves(
    path: Path,
    summary: List[Dict[str, Any]],
    score_base: float,
    delta: float,
    track_id: str,
) -> None:
    """Small-multiples: lap time vs each param at lo/base/hi (linearity check)."""
    n = len(summary)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for idx, d in enumerate(summary):
        ax = axes[idx // ncols][idx % ncols]
        base_val = d["base_value"]
        xs = [base_val * (1.0 - delta), base_val, base_val * (1.0 + delta)]
        ys = [d["score_lo"], score_base, d["score_hi"]]
        ax.plot(xs, ys, "-o", color="tab:blue", zorder=2)
        ax.scatter([base_val], [score_base], color="k", zorder=3, label="baseline")
        ax.set_title(f"{d['label']}  (E={d['elasticity']:+.2f})", fontsize=10)
        ax.set_xlabel("parameter value")
        ax.set_ylabel("lap time [s]")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    # Hide any unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        f"{track_id} lap-time response curves (OAT, +-{delta*100:.0f}%, baseline = {score_base:.3f} s)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config_path: Path, track_id: str, delta: float, jobs: int, out_dir: Path) -> None:
    scratch_dir = out_dir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building track '{track_id}' and baseline from {config_path} ...")
    base = _build_baseline(config_path, track_id)

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
            print(f"  [{r['tag']:>22}] lap={r['score_s']} status={r['status']}")
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(_solve_one, all_jobs):
                print(f"  [{r['tag']:>22}] lap={r['score_s']} status={r['status']}")
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

    prefix = f"{track_id}_sensitivity"

    # CSV with every raw solve.
    csv_path = out_dir / f"{prefix}.csv"
    _write_csv(csv_path, results)

    # Tornado plot.
    png_path = out_dir / f"{prefix}.png"
    _tornado_plot(png_path, summary, score_base, delta, track_id)

    # Per-knob response curves.
    curves_path = out_dir / f"{prefix}_curves.png"
    _response_curves(curves_path, summary, score_base, delta, track_id)

    # JSON summary for programmatic use.
    json_path = out_dir / f"{prefix}_summary.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": str(config_path),
                "track_id": track_id,
                "delta": delta,
                "lap_time_base_s": score_base,
                "knobs": summary,
            },
            f,
            indent=2,
        )

    # Console ranking.
    ranked = sorted(summary, key=lambda d: abs(d["swing_s"]), reverse=True)
    print("\n" + "=" * 78)
    print(f"Baseline lap time: {score_base:.4f} s   (+-{delta*100:.0f}% OAT, track={track_id})")
    print("=" * 78)
    hdr = f"{'parameter':<18}{'base':>10}{'lap-':>10}{'lap+':>10}{'swing[s]':>11}{'elasticity':>12}"
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
    print(f"\nWrote:\n  {csv_path}\n  {png_path}\n  {curves_path}\n  {json_path}")

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
    ap.add_argument("--config", type=Path, default=_repo_root() / "configs" / "trackdrive.yaml")
    ap.add_argument(
        "--track-id",
        type=str,
        default="fscz25_track_boundary",
        help="Track under data/tracks/{track_id}.csv. Default fscz25_track_boundary.",
    )
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
    run(args.config, args.track_id, args.delta, args.jobs, args.out_dir)


if __name__ == "__main__":
    main()
