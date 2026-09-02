"""Tests for ``_splice_segment``, the one function that stitches prescribed
segments onto a solved trajectory.

It replaced three near-identical copies (autox lead-in, autox terminal pad,
skidpad lead-in), so the cases they each covered are pinned here together —
including the length invariant one of the copies quietly violated.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("casadi", reason="pipeline import chain requires casadi")

from fast_lto.pipeline import _splice_segment  # noqa: E402

STATE_NAMES = ["d", "psi_err", "v_long", "v_lat", "yaw_rate", "delta"]
INPUT_NAMES = ["Fx_fl", "delta_dot"]


def _solution() -> dict:
    return {
        "state_names": list(STATE_NAMES),
        "input_names": list(INPUT_NAMES),
        "path_xy": [[1.0, 2.0], [1.5, 2.5]],
        "arc_lengths": [0.0, 0.5],
        "w_left": [2.0, 2.1],
        "w_right": [2.2, 2.3],
        "kappa": [0.01, 0.02],
        "headings": [0.1, 0.2],
        "d": [0.0, 0.01],
        "psi_err": [0.0, 0.001],
        "v_long": [8.0, 8.2],
        "v_lat": [0.0, 0.01],
        "yaw_rate": [0.08, 0.16],
        "delta": [0.0, 0.02],
        "Fx_fl": [100.0, 110.0],
        "delta_dot": [0.0, 0.01],
        "timed_mask": [1, 1],
        "decel_mask": [0, 0],
    }


CURVED = {
    "positions": [[0.0, 0.0], [0.5, 0.1]],
    "headings": [0.0, 0.05],
    "curvatures": [0.26, 0.13],
    "arc_lengths": [-1.0, -0.5],
    "w_left": [1.9, 1.95],
    "w_right": [2.0, 2.05],
}

STRAIGHT = {
    "positions": [[-1.0, 0.0], [-0.5, 0.0]],
    "headings": [0.0, 0.0],
    "curvatures": [0.0, 0.0],
    "arc_lengths": [-1.0, -0.5],
    "w_left": [1.5, 1.5],
    "w_right": [1.5, 1.5],
}

ARRAY_KEYS = (
    ["path_xy", "arc_lengths", "w_left", "w_right", "kappa", "headings"]
    + STATE_NAMES
    + INPUT_NAMES
    + ["timed_mask", "decel_mask"]
)


@pytest.mark.parametrize(
    ("segment", "side", "timed"),
    [
        (CURVED, "before", 1),  # autox lead-in
        (CURVED, "after", 0),  # autox terminal pad
        (STRAIGHT, "before", 0),  # skidpad lead-in
    ],
    ids=["autox_lead_in", "autox_terminal_pad", "skidpad_lead_in"],
)
def test_every_array_grows_by_the_segment_length(segment, side, timed) -> None:
    """The invariant the old three-copy version broke for one case.

    A per-node array left un-extended is silently misaligned with every other
    array from that point on, which the exporter would then write out as rows
    that disagree with each other.
    """
    sol = _splice_segment(_solution(), copy.deepcopy(segment), 3.0, side=side, timed=timed)
    expected = 2 + len(segment["arc_lengths"])
    lengths = {key: len(sol[key]) for key in ARRAY_KEYS}
    assert set(lengths.values()) == {expected}, f"ragged arrays: {lengths}"


def test_prepend_puts_the_segment_first() -> None:
    sol = _splice_segment(_solution(), copy.deepcopy(CURVED), 3.0, side="before", timed=1)
    assert sol["arc_lengths"][:2] == CURVED["arc_lengths"]
    assert sol["path_xy"][:2] == CURVED["positions"]
    assert sol["v_long"][:2] == [3.0, 3.0]
    # The solved part is untouched at the tail.
    assert sol["v_long"][-2:] == [8.0, 8.2]


def test_append_puts_the_segment_last() -> None:
    sol = _splice_segment(_solution(), copy.deepcopy(CURVED), 1.3, side="after", timed=0)
    assert sol["arc_lengths"][-2:] == CURVED["arc_lengths"]
    assert sol["v_long"][-2:] == [1.3, 1.3]
    assert sol["v_long"][:2] == [8.0, 8.2]


def test_yaw_rate_is_curvature_times_speed() -> None:
    """The exporter recovers kappa as yaw_rate / v, so zero-filling would
    export a curving lead-in as a straight one."""
    speed = 3.0
    sol = _splice_segment(_solution(), copy.deepcopy(CURVED), speed, side="before", timed=1)
    assert sol["yaw_rate"][:2] == pytest.approx([k * speed for k in CURVED["curvatures"]])


def test_straight_segment_yields_zero_yaw_rate() -> None:
    """Why the skidpad lead-in needs no special case: the same rule gives 0."""
    sol = _splice_segment(_solution(), copy.deepcopy(STRAIGHT), 2.0, side="before", timed=0)
    assert sol["yaw_rate"][:2] == [0.0, 0.0]
    assert sol["kappa"][:2] == [0.0, 0.0]


@pytest.mark.parametrize("timed", [0, 1])
def test_timed_mask_uses_the_requested_value(timed: int) -> None:
    sol = _splice_segment(_solution(), copy.deepcopy(CURVED), 3.0, side="before", timed=timed)
    assert sol["timed_mask"][:2] == [timed, timed]


def test_decel_mask_is_never_marked_as_braking() -> None:
    sol = _splice_segment(_solution(), copy.deepcopy(CURVED), 1.3, side="after", timed=0)
    assert sol["decel_mask"][-2:] == [0, 0]


def test_absent_masks_are_left_absent() -> None:
    """Autox solutions carry no decel_mask; splicing must not invent one."""
    sol = _solution()
    sol["decel_mask"] = None
    spliced = _splice_segment(sol, copy.deepcopy(CURVED), 3.0, side="before", timed=1)
    assert spliced["decel_mask"] is None


def test_inputs_are_filled_with_zero() -> None:
    """A prescribed segment is not actuated: it is a reference, not a plan."""
    sol = _splice_segment(_solution(), copy.deepcopy(CURVED), 3.0, side="before", timed=1)
    for name in INPUT_NAMES:
        assert sol[name][:2] == [0.0, 0.0]
