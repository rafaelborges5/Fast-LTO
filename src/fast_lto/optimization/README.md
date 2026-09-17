# optimization

The optimal control problem solved by IPOPT.

## The formulation

Arc length is the independent variable. With `N` nodes at spacing `ds`, the
decision variables are the reduced state `X` (everything but `s`, already
encoded by the grid) and the input `U` at every node.

- **Dynamics.** `integrator.step` advances the reduced state one spatial
  interval via `dx/ds = (dx/dt) / (ds/dt)`. `integrator.time_step` estimates
  `dt = ∫ ds / ṡ` over the same interval. Euler uses the left endpoint; RK4
  also needs curvature at the midpoint and right endpoint
  (`curvatures_half` on the discretised track).
- **Objective.** Sum of those `dt`, weighted per node by the event, plus a
  quadratic penalty on input rates (and optionally on inputs).
- **Constraints.** Model `g(x, u, kappa) <= 0` at every evaluation point,
  corridor box on `d`, one corner constraint per corner, and either the
  closed-loop equality or a pinned initial state.

States and inputs are mapped to roughly `[-1, 1]` from their physical bounds;
IPOPT's own scaling is off. Bounds must be finite.

## Smoothness

Hard `max` operations that appear in the physics — flooring `ṡ` in a
denominator, non-negative tyre load — are replaced by `smoothmax`
(`utils/smooth.py`), which is C1.

Inside the OCP, `D_kappa = 1 - kappa * d` is left unclamped: the models enforce
`D_kappa >= eps_D_kappa`. The NumPy corner geometry in `utils/corridor.py` and
`warm_start.py` has no such constraint, so it clamps `D_kappa` with sign
preservation (`D_kappa` changes sign across the singularity; an unsigned clamp
would map both sides onto the same one).

## The terminal region

- **Final-node constraints.** The node loop applies model constraints for
  `i < N-1` only. With `terminal_speed`, those constraints are reapplied at
  `X[N-1]` (needed when friction limits depend on persistent state, e.g.
  four-wheel tyre forces). Trackdrive omits the repeat: the closed-loop
  equality ties `X[N-1]` to `X[0]`, which the loop already constrains.
- **`terminal_speed`.** Inequality `v <= target`, not equality — an equality
  couples the discrete dynamics to a binding friction limit at a single point.
- **Terminal window.** Bounds lateral offset, heading, `yaw_rate`, and `v_lat`
  over the same nodes as the speed target. The window spans more than one node
  so rate-limited inputs can meet it.

Skidpad uses a metre-measured window before the finish. Autox bounds only the
last few nodes (the state handed to the terminal pad); the approach is free.

## Conservative-grip braking

`D_safe_braking` replaces the tyre `D` coefficients outright — an absolute
override, not a scale — on the autox nodes past the timing gate, so the car
brakes on a more conservative grip estimate. The boundary is
`track["timed_mask"]`, the same one the objective weights use, not a separately
specified distance.

## Warm starting

Default guess: centreline at `initial_speed`. On a tight corridor that point
can be infeasible; `utils/corridor.critical_margin` finds the margin where it
becomes feasible. The ladder solves at a safe margin first, then continues
from that solution.

Past solutions are stored under `data/solutions/_seeds` (gitignored; empty
means cold starts). Compatibility:

- **hard keys** — exact match (node grid / variable meaning): track geometry,
  `ds`, mode, model, integrator, normalisation, state and input names, and
  terminal-region constraint changes;
- **soft keys** — used only to rank candidates: boundary margin and vehicle
  parameters.

Anything that changes how a solution was produced belongs in the signature.
