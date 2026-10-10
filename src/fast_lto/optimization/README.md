# optimization

The optimal control problem, and the solvers that run it.

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

## Building the NLP

One interval — integrator step, model and corridor constraints, and `dt` — is
written once as a CasADi function and expanded to SX. `build_ocp` applies it
to all nodes in a single `map` call, with a second map for brake-zone nodes.
The NLP graph therefore holds one call per vehicle model instead of `N` copies
of it, which keeps the build under a second and makes derivatives cheap to
evaluate. Model code must stay expandable: no MX-only operations such as a
symbolic `ca.solve`.

Solver settings: MUMPS with QAMD ordering and its own scaling off (about 30%
less time in the linear solves), and OpenBLAS limited to one thread, set in
`fast_lto/__init__.py` unless `OPENBLAS_NUM_THREADS` is already defined.

## Solvers and derivatives

Two settings choose how the OCP is solved. Neither changes the problem.

| `solver` | `derivatives` | What runs |
| --- | --- | --- |
| `ipopt` | `symbolic` | The Opti problem above; CasADi differentiates the whole NLP. The default. |
| `ipopt` | `compiled` | The same problem stage by stage, with per-stage derivatives compiled to C. |
| `fatrop` | `compiled` | The same stage-wise problem in FATROP, through its native C interface. |

Measured on a four-wheel lap at `ds = 0.5` m, single-threaded, library cached:

| | `ipopt` / `symbolic` | `ipopt` / `compiled` | `fatrop` / `compiled` |
| --- | --- | --- | --- |
| autox, 723 nodes | 8.2 s | 2.5 s | 0.9 s |
| trackdrive, 800 nodes | 9.0 s | 2.2 s | 1.1 s |

**Stage-wise problem.** `stage_problem.py` restates `build_ocp` stage by stage:
stage `k` owns `x_k` and `u_k`, and everything else it needs — curvatures,
widths, time weight, launch state — is a per-stage parameter vector. Stages
with the same maths share a type (first, ordinary, brake zone, terminal window,
last). A vehicle model needs nothing new for this.

**Compiled derivatives.** Each stage type's values, Jacobian and Lagrangian
Hessian are generated as C and compiled once (`codegen.py`). The library is
cached under `$FAST_LTO_CACHE` (default `~/.cache/fast_lto`) by a hash of its
source, so changing the model, its parameters, `reg_u_l2` or the event builds
a new one, and nothing is invalidated by hand. Track data and `ds` are runtime
inputs, so one build serves every track with the same settings. The first
solve of a new setup pays the compile, a few seconds to about twenty;
`fast-lto --prebuild` does it without solving.

**IPOPT.** The stage derivatives are mapped over the stages and permuted into
the NLP Jacobian and Hessian, which IPOPT takes in place of its own
(`stage_ipopt.py`).

**FATROP.** A structure-exploiting interior-point method whose linear algebra
is a Riccati recursion over the stages, so its cost grows linearly with the
number of nodes. CasADi's FATROP plugin cannot take custom derivatives, so
`stage_fatrop.py` generates each stage type in FATROP's own layout and compiles
it with a small adapter (`fatrop_glue.c`) against FATROP's C interface. FATROP
only couples neighbouring stages; the trackdrive wrap `x_{N-1} = x_0` is
carried as a copy of `x_0` through every stage.

FATROP is a different algorithm on a nonconvex problem. On some tracks it lands
on a better local optimum than IPOPT; it has never been worse in testing.

**Limits.** The compiled paths need a C compiler (`$CC`, else `cc`, `gcc` or
`clang`) and normalised states. The input-rate penalty `reg_u` couples
neighbouring stages and is rejected; every event config sets it to 0.

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

- **Final-node constraints.** The stage function applies model constraints for
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
