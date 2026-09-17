# optimization

The optimal control problem, and what it took to make IPOPT solve it.

## The formulation

Arc length is the independent variable. With `N` nodes at spacing `ds`, the
decision variables are the reduced state `X` (everything but `s`, which the grid
already encodes) and the input `U` at every node.

- **Dynamics.** `integrator.step` advances the reduced state one spatial
  interval using `dx/ds = (dx/dt) / (ds/dt)`. `integrator.time_step` estimates
  `dt = ∫ ds / ṡ` over the same interval. Euler evaluates at the left endpoint;
  RK4 also wants curvature at the midpoint and right endpoint, which is why the
  discretised track carries `curvatures_half`.
- **Objective.** The sum of those `dt`, weighted per node where the event says
  so, plus a quadratic penalty on input rates (and optionally on inputs).
  Without the rate penalty the inputs chatter between nodes at no cost to lap
  time.
- **Constraints.** The model's own `g(x, u, kappa) <= 0` at every evaluation
  point, the corridor as a box on `d`, one corner constraint per corner of the
  car, and either the closed-loop equality or a pinned initial state.

Everything is solved normalised, states and inputs mapped to roughly `[-1, 1]`
from their physical bounds, with IPOPT's own scaling switched off. The bounds
therefore have to be finite: a model with an infinite bound cannot be normalised.

## Smoothness

IPOPT is a Newton-type method, so it builds a quadratic model from first and
second derivatives. An expression with a kink is a place it stalls or
oscillates. Where the physics wants a hard `max` — flooring `ṡ` in a
denominator, refusing a negative tyre load — the solver gets `smoothmax`
instead (`utils/smooth.py`), which is C1 everywhere.

The Frenet Jacobian `D_kappa = 1 - kappa * d` needs no clamp inside the OCP: the
models constrain `D_kappa >= eps_D_kappa`, so the expression stays smooth. The
NumPy corner geometry in `utils/corridor.py` and `warm_start.py` has no solver
enforcing that, so it clamps — keeping the sign, because `D_kappa` legitimately
changes sign past the singularity and an unsigned clamp puts both sides of it on
the same side.

## The terminal region

Most of the non-obvious code in `global_ocp.py` is here, and all of it exists
because a terminal condition turns a harmless gap into an exploitable one.

- **The final node's constraints are applied again, by hand.** The node loop
  evaluates the model's constraints at `x_i` for `i < N-1` only, so `X[N-1]` is
  never checked by it. Normally harmless, since nothing pins the final state to
  an extreme. But a `terminal_speed` does pin it, and a model whose friction
  limit depends on persistent state rather than on the input — four-wheel's
  per-wheel forces are states — can then land on a final state that violates its
  own physics, cheaply. Trackdrive needs no repeat: its closed-loop equality
  ties `X[N-1]` back to `X[0]`, which the loop does check at `i = 0`.
- **`terminal_speed` is an inequality**, `v <= target`. An equality forces the
  solver to land on one exact point through the discrete dynamics, right where
  the friction circle is also newly binding. Far more tightly coupled, and much
  slower, for no gain: nothing wants the car going *faster* than the target.
- **The terminal window bounds state shape, not just speed.** Holding the last
  stretch centred and heading-aligned is the obvious half. The other half is
  capping `yaw_rate` and `v_lat` over the same window: with a speed target
  active and nothing costing state *shape* in an untimed zone, the solver takes
  a violent, cost-free excursion on the last node or two to land exactly on the
  target. `v_long` decays smoothly throughout; `yaw_rate` spikes, and once it is
  capped alone, `v_lat` spikes instead. Capping both removes the cheat rather
  than one symptom of it.
- **The window is more than one node.** A single node is infeasible for
  rate-limited actuators, which cannot snap to the target in zero steps.

Skidpad bounds a window measured in metres before the finish; autox bounds the
last few nodes only, leaving the approach free, because that window is just the
state its prescribed terminal pad picks up from.

## Conservative-grip braking

`D_safe_braking` replaces the tyre `D` coefficients outright — an absolute
override, not a scale — on the autox nodes past the timing gate, so the car
brakes on a grip estimate it can trust while still attacking the timed lap. The
boundary is `track["timed_mask"]`, the same one the objective weights use, not a
separately specified distance.

This works because tyre coefficients are baked into each node's expressions as
plain floats rather than carried as CasADi parameters, so different nodes can be
built against different model instances. Off by default, and then the graph is
identical to one built without the feature.

## Warm starting

The default guess is the car on the centreline at `initial_speed`. On a tight
corridor that point is outside the feasible set, and a solve that starts
infeasible is slow and prone to a worse local minimum.
`utils/corridor.critical_margin` finds the margin where the default guess stops
being feasible, which is what the ladder climbs: solve an easier problem at a
safe margin first, then continue from it.

`warm_start.py` also keeps a store of past solutions under
`data/solutions/_seeds` (gitignored — an empty store just means every solve is
cold). Compatibility splits in two:

- **hard keys** must match exactly, because they define the node grid or the
  meaning of the variables: track geometry, `ds`, mode, model, integrator,
  normalisation, state and input names, and anything that adds or removes a
  constraint near the terminal region;
- **soft keys** may differ and only rank candidates: the boundary margin and the
  vehicle parameters. They move the optimum *within* a basin without moving the
  corridor that creates the basins, so a seed across them is still a good seed.

Anything that changes how a solution was produced has to appear in the
signature, or two different solves quietly share an entry.
