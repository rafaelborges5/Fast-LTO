"""Tests for the warm-start seed store and the corridor feasibility helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fast_lto.modes import get_mode
from fast_lto.optimization import warm_start as ws
from fast_lto.utils.corridor import corridor_at, corridor_widths, critical_margin
from fast_lto.vehicle_models import FourWheelModel


# --------------------------------------------------------------------------- #
#  Fixtures — a small synthetic track and a real four-wheel model
# --------------------------------------------------------------------------- #
@pytest.fixture
def model():
    return FourWheelModel(
        params={
            "corners": [
                ("FL", 1.73, 0.75),
                ("FR", 1.73, -0.75),
                ("RL", -0.85, 0.63),
                ("RR", -0.85, -0.63),
            ]
        }
    )


@pytest.fixture
def track():
    """Straight track with one tight corner in the middle."""
    n = 40
    s = np.arange(n) * 0.5
    kappa = np.zeros(n)
    kappa[20] = -0.5
    return {
        "arc_lengths": s.tolist(),
        "curvatures": kappa.tolist(),
        "curvatures_half": kappa.tolist(),
        "w_left": (np.full(n, 1.5)).tolist(),
        "w_right": (np.full(n, 1.5)).tolist(),
        "ds_m": 0.5,
        "num_points": n,
        "positions": np.column_stack([s, np.zeros(n)]).tolist(),
        "headings": np.zeros(n).tolist(),
    }


def make_signature(track, model, margin, **soft_overrides):
    params = dict(model.params)
    params.update(soft_overrides)
    return ws.seed_signature(
        track_id="unit",
        mode="autox",
        model_name="four_wheel",
        integrator_name="euler",
        continuity="C2",
        normalize_states_and_inputs=True,
        boundary_margin=margin,
        track=track,
        model=model,
        model_params=params,
    )


def make_solution(track, model, value=0.0):
    """A minimal solution JSON in the shape the store expects."""
    n = len(track["arc_lengths"])
    sol = {
        "arc_lengths": list(track["arc_lengths"]),
        "state_names": list(model.reduced_state_names()),
        "input_names": list(model.get_input_names()),
        "obj_val": 12.34,
    }
    for name in sol["state_names"]:
        sol[name] = (np.full(n, value) if name != "v_long" else np.full(n, 8.0)).tolist()
    for name in sol["input_names"]:
        sol[name] = np.zeros(n).tolist()
    return sol


# --------------------------------------------------------------------------- #
#  Corridor
# --------------------------------------------------------------------------- #
def test_corridor_shrinks_with_margin(track, model):
    corners = model.get_corner_offsets()
    wide = corridor_widths(track, corners, 0.0).min()
    tight = corridor_widths(track, corners, 0.3).min()
    assert tight == pytest.approx(wide - 0.6, abs=1e-6)


def test_critical_margin_matches_a_direct_scan(track, model):
    corners = model.get_corner_offsets()
    crit = critical_margin(track, corners)
    assert corridor_widths(track, corners, crit.margin).min() > 0.0
    assert corridor_widths(track, corners, crit.margin + 0.01).min() < 0.0
    # the tight station is the limiting one
    assert crit.index == 20


def test_corner_curvature_term_costs_the_inside_corner(model):
    corners = model.get_corner_offsets()
    straight = corridor_at(0.0, 1.5, 1.5, corners)
    curved = corridor_at(-0.5, 1.5, 1.5, corners)
    assert (curved[1] - curved[0]) < (straight[1] - straight[0])


def test_critical_margin_needs_corners(track):
    with pytest.raises(ValueError):
        critical_margin(track, [])


# --------------------------------------------------------------------------- #
#  Signatures
# --------------------------------------------------------------------------- #
def test_vehicle_params_are_soft_keys(track, model):
    base = make_signature(track, model, 0.3)
    other = make_signature(track, model, 0.3, v_max=18.0)
    assert ws.hard_keys_match(base, other)
    assert ws.vehicle_distance(base, other) > 0.0


@pytest.mark.parametrize("param,value", [("v_max", 18.0), ("D_fl", 1.3), ("dFxmax", 600.0)])
def test_seed_signature_tracks_each_tuning_knob(track, model, param, value):
    base = make_signature(track, model, 0.3, **{param: 1.0})
    other = make_signature(track, model, 0.3, **{param: value})
    assert base["soft"]["model_params"] != other["soft"]["model_params"]
    assert ws.vehicle_distance(base, other) > 0.0


def test_geometry_change_is_a_hard_key(track, model):
    base = make_signature(track, model, 0.3)
    moved = dict(track)
    moved["w_left"] = (np.asarray(track["w_left"]) + 0.1).tolist()
    other = make_signature(moved, model, 0.3)
    assert not ws.hard_keys_match(base, other)


def test_solver_verbose_is_not_part_of_the_signature(track, model):
    # solver_verbose never reaches seed_signature at all: two otherwise equal
    # runs must produce identical signatures regardless of logging.
    assert make_signature(track, model, 0.3) == make_signature(track, model, 0.3)


# --------------------------------------------------------------------------- #
#  Store
# --------------------------------------------------------------------------- #
def test_save_and_find_nearest_margin(tmp_path, track, model):
    root = tmp_path / "_seeds"
    ws.save_seed(root, make_signature(track, model, 0.30), make_solution(track, model))
    ws.save_seed(root, make_signature(track, model, 0.40), make_solution(track, model))

    match = ws.find_seed(root, make_signature(track, model, 0.42))
    assert match is not None
    assert match.margin == pytest.approx(0.40)
    assert match.margin_gap == pytest.approx(0.02)


def test_margin_gap_cutoff_rejects_distant_seeds(tmp_path, track, model):
    root = tmp_path / "_seeds"
    ws.save_seed(root, make_signature(track, model, 0.10), make_solution(track, model))
    assert ws.find_seed(root, make_signature(track, model, 0.45)) is None
    assert ws.find_seed(root, make_signature(track, model, 0.45), max_margin_gap=0.5)


def test_same_margin_prefers_the_closer_vehicle(tmp_path, track, model):
    root = tmp_path / "_seeds"
    ws.save_seed(root, make_signature(track, model, 0.3, v_max=25.0), make_solution(track, model))
    ws.save_seed(root, make_signature(track, model, 0.3, v_max=16.5), make_solution(track, model))
    match = ws.find_seed(root, make_signature(track, model, 0.3, v_max=16.0))
    assert match is not None
    assert (
        "16"
        in json.loads(match.path.read_text())["seed_signature"]["soft"]["model_params"][
            "v_max"
        ].__str__()
    )


def test_incompatible_geometry_is_never_returned(tmp_path, track, model):
    root = tmp_path / "_seeds"
    ws.save_seed(root, make_signature(track, model, 0.3), make_solution(track, model))
    other = dict(track)
    other["curvatures"] = (np.asarray(track["curvatures"]) + 0.05).tolist()
    assert ws.find_seed(root, make_signature(other, model, 0.3)) is None


def test_prune_keeps_the_most_recent(tmp_path, track, model):
    root = tmp_path / "_seeds"
    for margin in (0.10, 0.20, 0.30):
        ws.save_seed(
            root, make_signature(track, model, margin), make_solution(track, model), max_seeds=2
        )
    bucket = ws.bucket_dir(root, make_signature(track, model, 0.10))
    seeds = [p for p in bucket.glob("*.json") if p.name != ws.INDEX_FILENAME]
    assert len(seeds) == 2


def test_index_is_rebuilt_when_missing(tmp_path, track, model):
    root = tmp_path / "_seeds"
    sig = make_signature(track, model, 0.3)
    ws.save_seed(root, sig, make_solution(track, model))
    (ws.bucket_dir(root, sig) / ws.INDEX_FILENAME).unlink()
    assert ws.find_seed(root, make_signature(track, model, 0.31)) is not None


# --------------------------------------------------------------------------- #
#  Guess construction
# --------------------------------------------------------------------------- #
def test_resample_aligns_on_arc_length_not_index(track, model):
    """An autox solution carries a lead-in, so it is longer and starts earlier."""
    n = len(track["arc_lengths"])
    sol = make_solution(track, model)
    lead = [-1.0, -0.5]
    sol["arc_lengths"] = lead + list(track["arc_lengths"])
    for name in sol["state_names"] + sol["input_names"]:
        sol[name] = [sol[name][0]] * len(lead) + list(sol[name])

    guess = ws.resample_guess(sol, track, model, use_normalization=False)
    assert guess["X"].shape == (n, len(sol["state_names"]))
    assert guess["U"].shape == (n, len(sol["input_names"]))
    v_idx = sol["state_names"].index("v_long")
    assert np.allclose(guess["X"][:, v_idx], 8.0)


def test_resample_rejects_a_foreign_state_layout(track, model):
    sol = make_solution(track, model)
    sol["state_names"] = ["d", "psi_err"]
    with pytest.raises(ValueError):
        ws.resample_guess(sol, track, model, use_normalization=False)


def test_validate_guess_rejects_a_wildly_off_seed(track, model):
    n = len(track["arc_lengths"])
    nx = len(model.reduced_state_names())
    nu = len(model.get_input_names())
    good = {"X": np.zeros((n, nx)), "U": np.zeros((n, nu))}
    ok, _ = ws.validate_guess(good, track, model, 0.3)
    assert ok

    off = {"X": np.zeros((n, nx)), "U": np.zeros((n, nu))}
    off["X"][:, 0] = 5.0  # 5 m off the centreline of a 3 m wide track
    ok, why = ws.validate_guess(off, track, model, 0.3)
    assert not ok and "corridor" in why


def test_validate_guess_rejects_wrong_shapes_and_nans(track, model):
    n = len(track["arc_lengths"])
    nx = len(model.reduced_state_names())
    nu = len(model.get_input_names())
    assert not ws.validate_guess(
        {"X": np.zeros((n - 1, nx)), "U": np.zeros((n, nu))}, track, model, 0.3
    )[0]
    bad = {"X": np.zeros((n, nx)), "U": np.zeros((n, nu))}
    bad["X"][3, 0] = np.nan
    assert not ws.validate_guess(bad, track, model, 0.3)[0]


# --------------------------------------------------------------------------- #
#  Pipeline policy
# --------------------------------------------------------------------------- #
def test_policy_is_validated():
    from fast_lto.pipeline import PipelineConfig

    with pytest.raises(ValueError):
        PipelineConfig(mode="autox", warm_start="sometimes")


def test_ladder_planning(track, model):
    from fast_lto.pipeline import PipelineConfig

    corners = model.get_corner_offsets()
    crit = critical_margin(track, corners)

    easy = PipelineConfig(mode="autox", boundary_margin=crit.margin - 0.05)
    assert len(_ladder(easy, track, model)) == 1

    hard = PipelineConfig(mode="autox", boundary_margin=crit.margin + 0.10)
    auto = _ladder(hard, track, model)
    assert len(auto) == 2 and auto[0] < crit.margin < auto[-1]

    hard.warm_start = "ladder"
    rungs = _ladder(hard, track, model)
    assert len(rungs) >= 2 and rungs[-1] == pytest.approx(hard.boundary_margin)
    assert all(b > a for a, b in zip(rungs, rungs[1:]))


def _ladder(config, track, model):
    from fast_lto.pipeline import _plan_ladder

    return _plan_ladder(config, track, model)


def test_warm_start_off_does_no_store_io(tmp_path, track, model, monkeypatch):
    """`off` must not read or write the seed store."""
    from fast_lto import pipeline

    calls = {"find": 0, "save": 0}
    monkeypatch.setattr(ws, "find_seed", lambda *a, **k: calls.__setitem__("find", 1))
    monkeypatch.setattr(ws, "save_seed", lambda *a, **k: calls.__setitem__("save", 1))

    solved = {}

    def fake_solve(
        config,
        track_data,
        model_,
        integrator,
        time_weights,
        solution_path,
        run_config,
        boundary_margin,
        mode,
        initial_guess=None,
    ):
        solved["initial_guess"] = initial_guess
        solved["margin"] = boundary_margin
        return {"obj_val": 1.0}

    monkeypatch.setattr(pipeline, "_solve_once", fake_solve)

    config = pipeline.PipelineConfig(mode="autox", warm_start="off", boundary_margin=0.45)
    config.solutions_dir = tmp_path
    out = pipeline._solve_with_warm_start(
        config=config,
        track_data=track,
        model=model,
        integrator=None,
        mode=get_mode(config.mode),
        time_weights=None,
        solution_path=tmp_path / "sol.json",
        run_config={},
    )

    assert calls == {"find": 0, "save": 0}
    assert solved["initial_guess"] is None
    assert out["warm_start"]["policy"] == "off"
    assert not any(tmp_path.glob("_seeds/**/*.json"))


def test_run_pipeline_always_resolves(monkeypatch, tmp_path):
    """Solution reuse stays off: a repeat run with an unchanged config re-solves."""
    from fast_lto import pipeline

    calls = []
    monkeypatch.setattr(
        pipeline, "step_solve_ocp", lambda config, **kw: calls.append(1) or Path("sol.json")
    )
    monkeypatch.setattr(pipeline, "step_fit_spline", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "step_compute_bounds", lambda *a, **k: None)

    csv_path = tmp_path / "track.csv"
    csv_path.write_text("side,cone_id,x,y\n")
    widths = tmp_path / "track_with_widths.json"
    widths.write_text("{}")

    config = pipeline.PipelineConfig(mode="autox", track_id="track", track_csv_path=csv_path)
    config.discretized_dir = tmp_path
    (tmp_path / "track.json").write_text("{}")

    for _ in range(2):
        pipeline.run_pipeline(config=config, start_from="ocp", end_at="ocp")
    assert len(calls) == 2
