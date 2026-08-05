"""
Solve the autox OCP over a ladder of boundary_margins, warm-starting each solve
from the previous (converged) one.

The default initial guess is d = 0, psi_err = 0, v = initial_speed, i.e. the car
sitting on the centreline. On a tight track that guess violates the corner
constraints as soon as the margin grows, and IPOPT has to find its way out of an
infeasible point through restoration. Continuation keeps every solve starting
from a feasible neighbour.

    python scripts/margin_continuation.py --track-id ipz_august_3 \
        --margins 0.30 0.35 0.40 0.45 --config configs/autox.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config import RunConfig  # noqa: E402
from optimization.global_ocp import load_track_with_widths, solve_ocp_and_save  # noqa: E402
from optimization.warm_start import resample_guess  # noqa: E402
from pipeline import (  # noqa: E402
    _autox_time_weights,
    _extend_track_for_autox,
    _make_integrator,
    _make_model,
)


def build_track(config, widths_path: Path):
    """Track data + time weights exactly as pipeline.step_solve_ocp builds them."""
    track = load_track_with_widths(widths_path)
    if config.mode != "autox":
        return track, None
    track = _extend_track_for_autox(
        track,
        config.autox_extension_m,
        config.autox_lead_in_m,
        config.autox_ocp_lead_m,
        timing_offset_m=config.autox_timing_offset_m,
    )
    tw = _autox_time_weights(
        track["arc_lengths"],
        track["autox_base_length_m"],
        config.autox_timing_offset_m,
        config.eps_time,
        config.decel_hold_m,
    )
    track["timed_mask"] = (tw >= 1.0 - 1e-9).astype(int).tolist()
    return track, tw


def guess_from_solution(sol: dict, model, track, use_norm: bool):
    """Resample a previous solution onto this OCP's node grid (normalised units).

    Thin wrapper over ``optimization.warm_start.resample_guess`` so this script
    and the pipeline share one implementation.
    """
    return resample_guess(sol, track, model, use_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/autox.yaml")
    ap.add_argument("--track-id", default="ipz_august_3")
    ap.add_argument("--margins", type=float, nargs="+",
                    default=[0.30, 0.35, 0.40, 0.45])
    ap.add_argument("--seed-solution", default=None,
                    help="JSON to warm-start the first margin from")
    ap.add_argument("--cold", action="store_true",
                    help="no continuation: every margin starts from the default guess")
    ap.add_argument("--out-dir", default="data/experiments/ipz_margin")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_config = RunConfig.from_yaml(REPO / args.config)
    run_config.track_id = args.track_id
    run_config.validate_for_model()
    config = run_config.to_pipeline_config()
    config.__post_init__()
    config.solver_verbose = args.verbose

    widths_path = config.track_with_widths_path
    if not widths_path.exists():
        raise SystemExit(f"missing {widths_path}; run the pipeline up to 'bounds' first")

    track, time_weights = build_track(config, widths_path)
    model = _make_model(config.model_name, vehicle_config=config.vehicle_config)
    integrator = _make_integrator(config.integrator_name)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    guess = None
    if args.seed_solution:
        seed = json.loads(Path(args.seed_solution).read_text())
        guess = guess_from_solution(seed, model, track,
                                    config.normalize_states_and_inputs)
        print(f"seeded from {args.seed_solution}")

    results = []
    for margin in args.margins:
        tag = f"m{int(round(margin * 100)):03d}"
        path = out_dir / f"{args.track_id}_{tag}.json"
        print(f"\n=== margin {margin:.2f} "
              f"({'cold' if (args.cold or guess is None) else 'warm'}) ===", flush=True)
        t0 = time.perf_counter()
        try:
            sol = solve_ocp_and_save(
                track=track,
                model=model,
                solution_path=path,
                integrator=integrator,
                initial_speed=config.initial_speed,
                reg_du=config.reg_u,
                reg_u_l2=config.reg_u_l2,
                run_config={"track_id": args.track_id, "boundary_margin": float(margin),
                            "mode": config.mode, "model_name": config.model_name,
                            "ds_m": float(track["ds_m"])},
                use_normalization=config.normalize_states_and_inputs,
                solver_verbose=args.verbose,
                boundary_margin=margin,
                mode=config.mode,
                time_weights=time_weights,
                terminal_speed=config.terminal_speed,
                autox_timing_offset_m=(config.autox_timing_offset_m
                                       if config.mode == "autox" else None),
                initial_guess=None if args.cold else guess,
            )
            dt = time.perf_counter() - t0
            prof = sol["profiling"]
            results.append(dict(margin=margin, status=prof["return_status"],
                                iters=prof["iter_count"], time_s=dt,
                                obj=sol["obj_val"],
                                lap=sol.get("autox_lap_time_s")))
            if not args.cold:
                guess = guess_from_solution(sol, model, track,
                                            config.normalize_states_and_inputs)
        except Exception as exc:  # noqa: BLE001 - report and carry on
            dt = time.perf_counter() - t0
            print(f"  FAILED after {dt:.1f} s: {type(exc).__name__}: {exc}")
            results.append(dict(margin=margin, status="FAILED", iters=None,
                                time_s=dt, obj=None, lap=None))

    print("\n=== summary ===")
    print(f"{'margin':>7} {'status':>28} {'iters':>7} {'time [s]':>9} "
          f"{'obj':>8} {'lap [s]':>8}")
    for r in results:
        lap = f"{r['lap']:.3f}" if r["lap"] is not None else "   -  "
        obj = f"{r['obj']:.3f}" if r["obj"] is not None else "   -  "
        it = r["iters"] if r["iters"] is not None else "-"
        print(f"{r['margin']:7.2f} {str(r['status']):>28} {str(it):>7} "
              f"{r['time_s']:9.1f} {obj:>8} {lap:>8}")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
