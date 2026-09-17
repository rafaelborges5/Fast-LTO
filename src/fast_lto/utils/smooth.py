"""C1 stand-ins for non-smooth ops used inside the OCP."""

from __future__ import annotations

import casadi as ca


def smoothmax(a: ca.MX, b: ca.MX, eps: float) -> ca.MX:
    """C1 approximation of ``max(a, b)``.

    ``0.5 * (a + b + sqrt((a - b)^2 + eps^2))`` — ``|a - b|`` with a rounded
    corner. Bias is ``eps/2`` at ``a == b``, and falls off away from equality.
    """
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps**2))
