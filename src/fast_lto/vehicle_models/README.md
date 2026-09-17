# vehicle_models

What the car is. Three models, one interface, same physical car.

## The contract

A model is a `VehicleModel` (`vehicle_base.py`) that provides:

- `get_state_names` / `get_input_names`, with `state[0] = s` and `state[1] = d`
  always, the rest model-specific;
- `get_dynamics` — symbolic **time** derivatives `ẋ = f(x, u, kappa)`. The
  space-domain conversion happens once, in `optimization/global_ocp.py`, not
  per model;
- `get_constraints` — inequalities `g(x, u, kappa) <= 0`;
- `state_bounds` and `input_bounds`, **finite** for every optimised state and
  input, because the base class normalises from them.

Optionally:

- `diagnostics` — named per-node quantities (tyre loads, slip angles, friction
  usage) as CasADi expressions built from the same helpers as the dynamics. This
  exists so a plot never re-derives the physics in NumPy: a second
  implementation drifts, and then the figure you would use to catch the drift is
  itself what drifted.

`get_corner_offsets` reads the `corners` list from the vehicle config, and
`get_corner_constraints` turns it into one corridor constraint per corner, so it
is the car's extents that have to fit through a gap and not a point at its
centre of gravity.

## The three

| | State | Input |
| --- | --- | --- |
| `point_mass` | `s d psi_err v` | `a_long a_lat` |
| `dynamic_bicycle` | `+ v_lat yaw_rate` | `a_long delta` |
| `four_wheel` | `+ Fx_fl Fx_fr Fx_rr Fx_rl delta` | the rates of those five |

`point_mass` is a friction circle on a curve — seconds to solve, and the right
model for checking that a track, a corridor margin or a config change behaves
before spending real time on it.

`dynamic_bicycle` adds sideslip and yaw, with a simplified Magic Formula per
axle and an optional friction ellipse.

`four_wheel` is the one that gets used in anger: per-wheel Pacejka forces,
aerodynamic load, longitudinal and lateral load transfer, and torque vectoring.
Its actuators are **states**, driven by rate inputs, which is what makes
`dFxmax` and `ddeltamax` real limits rather than post-hoc filters — and also
what makes the terminal region delicate (see `optimization/README.md`).

## One car, three descriptions

`configs/vehicle.yaml` defines the car once; each model block holds only what
that model needs. Parameters a model does not use are still declared in it, so
the three can never describe different cars —
`test_models_agree_on_the_car_they_describe` pins this.

`lf` is the distance to the **front** axle, so the static front load share is
`lr / (lf + lr)`. Same convention in all three, and in the configs.

## Guards

Two quantities can go singular, and each model declares a floor for them:

- `eps_s_dot` — `ṡ` appears in a denominator everywhere, since the space-domain
  conversion divides by it;
- `eps_D_kappa` — the Frenet Jacobian `D_kappa = 1 - kappa * d` vanishes at the
  centre of the osculating circle, where `(s, d)` stops being unique.
  `examples/frenet_frame.py` draws it.

`dynamic_bicycle` and `four_wheel` enforce both as constraints, which is what
lets the symbolic expressions keep the raw quantity and stay smooth for the
solver.

**`point_mass` declares `eps_D_kappa` but does not enforce it**, while still
carrying four corners — so its corner constraints divide by an unconstrained
Jacobian. In practice `d_max` is far below `1 / kappa` on any real Formula
Student track, so it has never bitten; on a hairpin with a wide corridor it
could. Enforcing it would change `point_mass` lap times, so it is a deliberate
open question rather than an oversight.

## Adding one

1. Subclass `VehicleModel` here and implement the contract above.
2. Register it in `pipeline._make_model`.
3. Add a panel renderer and an entry in `visualization.PANEL_RENDERERS` —
   `tests/test_panel_registry.py` fails until you do, rather than letting the
   pipeline solve for minutes and then die at the last step.
4. Add its block to `configs/vehicle.yaml`.
5. Add a golden scenario in `tests/test_golden_solutions.py`.
