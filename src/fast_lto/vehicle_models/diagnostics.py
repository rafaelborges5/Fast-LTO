"""Evaluate ``VehicleModel.diagnostics`` over a solved trajectory.

Compiles the CasADi expressions into a ``ca.Function`` and maps it across
nodes, returning one NumPy array per diagnostic.
"""

from __future__ import annotations

from typing import Dict

import casadi as ca
import numpy as np

from fast_lto.vehicle_models.vehicle_base import VehicleModel


def evaluate_diagnostics(model: VehicleModel, solution: Dict) -> Dict[str, np.ndarray]:
    """Evaluate ``model.diagnostics`` at every node of ``solution``.

    Parameters
    ----------
    model:
        Built with the *solved* parameters — pass the solution's own
        ``model_params``, or the numbers will not be the ones that produced it.
    solution:
        A solution dict holding one list per reduced state name.

    Returns
    -------
    dict
        ``{name: array of length N}``. Empty if the model reports nothing.
    """
    state_names = model.reduced_state_names()
    input_names = model.get_input_names()
    x_sym = ca.MX.sym("x_red", len(state_names))
    u_sym = ca.MX.sym("u", len(input_names))

    expressions = model.diagnostics(x_sym, u_sym)
    if not expressions:
        return {}

    missing = [name for name in state_names + input_names if name not in solution]
    if missing:
        raise KeyError(
            f"solution is missing {missing} required to evaluate "
            f"{type(model).__name__} diagnostics"
        )

    keys = list(expressions)
    func = ca.Function("diagnostics", [x_sym, u_sym], [expressions[key] for key in keys])

    x_values = np.array([np.asarray(solution[name], dtype=float) for name in state_names])
    u_values = np.array([np.asarray(solution[name], dtype=float) for name in input_names])
    n_nodes = x_values.shape[1]

    outputs = func.map(n_nodes)(x_values, u_values)
    if len(keys) == 1:
        outputs = [outputs]

    return {key: np.asarray(outputs[i]).reshape(-1) for i, key in enumerate(keys)}
