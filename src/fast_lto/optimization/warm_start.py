"""
Warm-start seed store for the global OCP.

The default initial guess is ``X = 0, U = 0, v = initial_speed`` — the car on the
centreline with zero heading error. On tight tracks that point is outside the
feasible set (see ``utils.corridor.critical_margin``) and the solve is both slow
and prone to landing in a worse local minimum. Seeding from a previously solved,
nearby problem avoids both.

The store is a local cache under ``data/solutions/_seeds`` (gitignored): an empty
store simply means every solve is cold. It never decides *whether* to solve —
only what to start from.

Seed compatibility is split in two:

* **hard keys** must match exactly, because they define the node grid or the
  meaning of the variables (track geometry, ds, mode, model, integrator,
  normalisation, state/input names);
* **soft keys** may differ and only rank the candidates — the boundary margin
  and every vehicle parameter (``v_max``, ``dFxmax``, tyre ``D``s, …). Those
  knobs move the optimum *within* a basin without moving the track corridor that
  creates the basins, so a seed across them is still a good seed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEEDS_DIRNAME = "_seeds"
INDEX_FILENAME = "index.json"

# Keys of the solution/run description that must be identical for a seed to be
# usable at all. Everything else is a ranking criterion.
HARD_KEYS = (
    "track_id",
    "mode",
    "model_name",
    "integrator_name",
    "ds_m",
    "num_points",
    "continuity",
    "normalize_states_and_inputs",
    "state_names",
    "input_names",
    "geom_hash",
    "terminal_straight_m",
    "terminal_state_constraint",
    "terminal_window_nodes",
    "D_safe_braking",
)


@dataclass(frozen=True)
class SeedMatch:
    """A seed picked out of the store, with why it was picked."""

    path: Path
    margin: float
    margin_gap: float
    vehicle_distance: float
    objective: Optional[float]

    def as_provenance(self) -> Dict:
        return {
            "seed_file": str(self.path),
            "seed_margin": self.margin,
            "seed_margin_gap": self.margin_gap,
            "seed_vehicle_distance": self.vehicle_distance,
        }


# --------------------------------------------------------------------------- #
#  Signatures
# --------------------------------------------------------------------------- #
def _round_floats(obj, ndigits: int = 12):
    """Recursively round floats so float noise never changes a signature."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, np.generic):
        return _round_floats(obj.item(), ndigits)
    return obj


def geom_hash(track: Dict) -> str:
    """Fingerprint of the track the OCP will actually see.

    Covers the resampled centreline curvature and both half widths, so a
    re-fitted, re-smoothed or re-bounded track — and any change to the autox
    extension, which lengthens the node grid — produces a different hash.
    """
    hasher = hashlib.blake2b(digest_size=16)
    for key in ("curvatures", "w_left", "w_right", "arc_lengths"):
        arr = np.round(np.asarray(track[key], dtype=float), 9)
        hasher.update(key.encode())
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def seed_signature(
    *,
    track_id: str,
    mode: str,
    model_name: str,
    integrator_name: str,
    continuity: str,
    normalize_states_and_inputs: bool,
    boundary_margin: float,
    track: Dict,
    model,
    model_params: Optional[Dict] = None,
    reg_u=None,
    reg_u_l2=None,
    initial_speed: Optional[float] = None,
    terminal_speed: Optional[float] = None,
    eps_time: Optional[float] = None,
    decel_hold_m: Optional[float] = None,
    terminal_straight_m: Optional[float] = None,
    terminal_state_constraint: bool = False,
    terminal_window_nodes: Optional[int] = None,
    D_safe_braking: Optional[float] = None,
) -> Dict:
    """Build the ``{"hard": …, "soft": …}`` description of one solve."""
    params = model_params if model_params is not None else getattr(model, "params", {})
    hard = {
        "track_id": str(track_id),
        "mode": str(mode),
        "model_name": str(model_name),
        "integrator_name": str(integrator_name),
        "ds_m": round(float(track["ds_m"]), 9),
        "num_points": int(len(track["arc_lengths"])),
        "continuity": str(continuity),
        "normalize_states_and_inputs": bool(normalize_states_and_inputs),
        "state_names": list(model.reduced_state_names()),
        "input_names": list(model.get_input_names()),
        "geom_hash": geom_hash(track),
        # These add/remove hard constraints near the terminal region rather
        # than just retuning the objective (unlike boundary_margin, which
        # narrows the same corridor the ladder is built to climb), so a seed
        # solved under a different value isn't just lower quality -- it may
        # not even satisfy the target problem's constraints.
        "terminal_straight_m": _round_floats(terminal_straight_m),
        "terminal_state_constraint": bool(terminal_state_constraint),
        "terminal_window_nodes": (
            int(terminal_window_nodes) if terminal_window_nodes is not None else None
        ),
        # Also changes the feasible set (a different tyre D over the untimed
        # tail of the horizon), not just the objective -- same reasoning as
        # the terminal keys above.
        "D_safe_braking": _round_floats(D_safe_braking),
    }
    soft = {
        "boundary_margin": round(float(boundary_margin), 9),
        "model_params": _round_floats(dict(params)),
        "reg_u": _round_floats(reg_u),
        "reg_u_l2": _round_floats(reg_u_l2),
        "initial_speed": _round_floats(initial_speed),
        "terminal_speed": _round_floats(terminal_speed),
        "eps_time": _round_floats(eps_time),
        "decel_hold_m": _round_floats(decel_hold_m),
    }
    return {"hard": hard, "soft": soft}


def seed_bucket(signature: Dict) -> str:
    """Directory name for all seeds that could ever seed each other."""
    hard = signature["hard"]
    return (
        f"{hard['track_id']}_{hard['mode']}_{hard['model_name']}"
        f"_{hard['integrator_name']}_ds{hard['ds_m']:.3f}"
        f"_{hard['continuity']}_{hard['geom_hash'][:8]}"
    )


def hard_keys_match(a: Dict, b: Dict) -> bool:
    return all(a["hard"].get(k) == b["hard"].get(k) for k in HARD_KEYS)


def vehicle_distance(a: Dict, b: Dict) -> float:
    """Worst relative change across the vehicle parameters (0 = identical).

    Non-numeric parameters that differ (``load_transfer_mode``, ``corners``)
    count as a full unit of distance: still usable as a seed, heavily demoted.
    """
    pa = a["soft"].get("model_params") or {}
    pb = b["soft"].get("model_params") or {}
    worst = 0.0
    for key in set(pa) | set(pb):
        va, vb = pa.get(key), pb.get(key)
        if (
            isinstance(va, (int, float))
            and isinstance(vb, (int, float))
            and not isinstance(va, bool)
            and not isinstance(vb, bool)
        ):
            scale = max(abs(float(va)), abs(float(vb)), 1e-9)
            worst = max(worst, abs(float(va) - float(vb)) / scale)
        elif va != vb:
            worst = max(worst, 1.0)
    return worst


# --------------------------------------------------------------------------- #
#  Store
# --------------------------------------------------------------------------- #
def bucket_dir(seeds_root: Path, signature: Dict) -> Path:
    return Path(seeds_root) / seed_bucket(signature)


def _load_index(directory: Path) -> Dict[str, Dict]:
    index_path = directory / INDEX_FILENAME
    try:
        data = json.loads(index_path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_index(directory: Path, index: Dict[str, Dict]) -> None:
    (directory / INDEX_FILENAME).write_text(json.dumps(index, indent=2))


def _rebuild_index(directory: Path) -> Dict[str, Dict]:
    """Recover the index from the seed files themselves."""
    index: Dict[str, Dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == INDEX_FILENAME:
            continue
        try:
            data = json.loads(path.read_text())
            sig = data.get("seed_signature")
            if sig is None:
                continue
            index[path.name] = {
                "signature": sig,
                "objective": data.get("obj_val"),
                "saved_at": path.stat().st_mtime,
            }
        except Exception:
            continue
    return index


def save_seed(
    seeds_root: Path,
    signature: Dict,
    solution: Dict,
    max_seeds: int = 50,
) -> Path:
    """Store a solved solution so later runs can start from it."""
    directory = bucket_dir(seeds_root, signature)
    directory.mkdir(parents=True, exist_ok=True)

    soft_hash = hashlib.blake2b(
        json.dumps(signature["soft"], sort_keys=True, default=str).encode(),
        digest_size=3,
    ).hexdigest()
    # Margin in millimetres, so a ladder rung at 0.298 is not filed as "0.30".
    margin = float(signature["soft"]["boundary_margin"])
    path = directory / f"m{round(margin * 1000):04d}_{soft_hash}.json"

    payload = dict(solution)
    payload["seed_signature"] = signature
    path.write_text(json.dumps(payload))

    index = _load_index(directory) or _rebuild_index(directory)
    index[path.name] = {
        "signature": signature,
        "objective": solution.get("obj_val"),
        "saved_at": time.time(),
    }
    _write_index(directory, index)
    prune_seeds(directory, max_seeds)
    return path


def prune_seeds(directory: Path, max_seeds: int) -> List[Path]:
    """Keep only the ``max_seeds`` most recently saved seeds in a bucket."""
    directory = Path(directory)
    index = _load_index(directory) or _rebuild_index(directory)
    if max_seeds <= 0 or len(index) <= max_seeds:
        return []

    ordered = sorted(index.items(), key=lambda kv: kv[1].get("saved_at", 0.0))
    removed: List[Path] = []
    for name, _meta in ordered[: len(index) - max_seeds]:
        path = directory / name
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
        index.pop(name, None)
    _write_index(directory, index)
    return removed


def find_seed(
    seeds_root: Path,
    signature: Dict,
    max_margin_gap: float = 0.15,
) -> Optional[SeedMatch]:
    """Best compatible seed for this problem, or None.

    Candidates are ranked by ``(|margin gap|, vehicle distance, -recency)``:
    the margin moves the corridor and therefore the local minima, so it
    dominates; the vehicle knobs only move the optimum within a basin.
    """
    directory = bucket_dir(seeds_root, signature)
    if not directory.is_dir():
        return None

    index = _load_index(directory)
    if not index:
        index = _rebuild_index(directory)
        if index:
            _write_index(directory, index)

    target_margin = float(signature["soft"]["boundary_margin"])
    candidates: List[Tuple[Tuple[float, float, float], SeedMatch]] = []
    for name, meta in index.items():
        path = directory / name
        if not path.is_file():
            continue
        sig = meta.get("signature")
        if not isinstance(sig, dict) or not hard_keys_match(signature, sig):
            continue
        margin = float(sig["soft"]["boundary_margin"])
        gap = abs(margin - target_margin)
        if gap > max_margin_gap:
            continue
        dist = vehicle_distance(signature, sig)
        saved_at = float(meta.get("saved_at", 0.0))
        candidates.append(
            (
                (round(gap, 9), round(dist, 9), -saved_at),
                SeedMatch(path, margin, gap, dist, meta.get("objective")),
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


# --------------------------------------------------------------------------- #
#  Guess construction
# --------------------------------------------------------------------------- #
def resample_guess(
    solution: Dict,
    track: Dict,
    model,
    use_normalization: bool,
) -> Dict[str, np.ndarray]:
    """Map a previous solution onto this OCP's node grid, in solver units.

    Alignment is by arc length, not by index: an autox solution carries the
    prepended lead-in (so it starts before s = 0 and has more nodes than the
    OCP horizon), and a seed may come from a different node count entirely.
    """
    s_new = np.asarray(track["arc_lengths"], dtype=float)
    s_old = np.asarray(solution["arc_lengths"], dtype=float)

    state_names = list(solution["state_names"])
    input_names = list(solution["input_names"])
    expected_states = list(model.reduced_state_names())
    expected_inputs = list(model.get_input_names())
    if state_names != expected_states or input_names != expected_inputs:
        raise ValueError(
            "seed state/input names do not match the model: "
            f"{state_names} vs {expected_states}, {input_names} vs {expected_inputs}"
        )

    x_phys = np.column_stack(
        [np.interp(s_new, s_old, np.asarray(solution[n], dtype=float)) for n in state_names]
    )
    u_phys = np.column_stack(
        [np.interp(s_new, s_old, np.asarray(solution[n], dtype=float)) for n in input_names]
    )

    if not use_normalization:
        return {"X": x_phys, "U": u_phys}

    x_scale, x_shift = model.get_reduced_state_scaling()
    u_scale, u_shift = model.get_input_scaling()
    x_scale = np.asarray(x_scale, dtype=float).reshape(1, -1)
    x_shift = np.asarray(x_shift, dtype=float).reshape(1, -1)
    u_scale = np.asarray(u_scale, dtype=float).reshape(1, -1)
    u_shift = np.asarray(u_shift, dtype=float).reshape(1, -1)
    return {"X": (x_phys - x_shift) / x_scale, "U": (u_phys - u_shift) / u_scale}


def validate_guess(
    guess: Dict[str, np.ndarray],
    track: Dict,
    model,
    boundary_margin: float,
    corners: Optional[Sequence] = None,
    max_corner_violation_m: float = 0.5,
) -> Tuple[bool, str]:
    """Cheap sanity check on a resampled seed.

    Catches a corrupt or mismatched seed before it wastes a solve. A seed from a
    smaller margin legitimately pokes outside the new, tighter corridor, so the
    tolerance is generous — this is a smoke test, not a feasibility test.
    """
    n_nodes = len(track["arc_lengths"])
    for key in ("X", "U"):
        arr = np.asarray(guess.get(key))
        if arr.ndim != 2 or arr.shape[0] != n_nodes:
            return False, f"{key} has shape {arr.shape}, expected ({n_nodes}, n)"
        if not np.all(np.isfinite(arr)):
            return False, f"{key} contains non-finite values"

    if corners is None:
        corners = model.get_corner_offsets()
    if not corners:
        return True, "ok"

    # The corner check needs physical d / psi_err; every model in the repo puts
    # them first, but say so out loud rather than assuming it silently.
    names = list(model.reduced_state_names())
    if names[:2] != ["d", "psi_err"]:
        return True, "ok (no d/psi_err to check)"

    x = np.asarray(guess["X"], dtype=float)
    x_scale, x_shift = model.get_reduced_state_scaling()
    if x_scale is not None and x_shift is not None:
        x = x * np.asarray(x_scale, float).reshape(1, -1) + np.asarray(x_shift, float).reshape(
            1, -1
        )
    d = x[:, 0]
    psi = x[:, 1]

    kappa = np.asarray(track["curvatures"], dtype=float)
    w_left = np.asarray(track["w_left"], dtype=float) - boundary_margin
    w_right = np.asarray(track["w_right"], dtype=float) - boundary_margin

    sin_p, cos_p = np.sin(psi), np.cos(psi)
    d_kappa = 1.0 - kappa * d
    d_kappa[np.abs(d_kappa) < 1e-9] = 1e-9
    worst = 0.0
    for corner in corners:
        long_proj = corner.dx * cos_p - corner.dy * sin_p
        d_corner = d + corner.dx * sin_p + corner.dy * cos_p - 0.5 * kappa / d_kappa * long_proj**2
        slack = (w_left - d_corner) if corner.dy >= 0 else (d_corner + w_right)
        worst = max(worst, float(-slack.min()))

    if worst > max_corner_violation_m:
        return False, f"seed violates the corridor by {worst:.2f} m"
    return True, "ok"
