# vehicle_models

Three vehicle models, one `VehicleModel` interface, one physical car.

## Interface

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

- `diagnostics` — named per-node CasADi quantities (tyre loads, slip angles,
  friction usage); evaluate with `diagnostics.evaluate_diagnostics`.

`get_corner_offsets` reads `corners` from the vehicle config;
`get_corner_constraints` emits one corridor inequality per corner.

## Models

| | State | Input |
| --- | --- | --- |
| `point_mass` | `s d psi_err v` | `a_long a_lat` |
| `dynamic_bicycle` | `+ v_lat yaw_rate` | `a_long delta` |
| `four_wheel` | `+ Fx_fl Fx_fr Fx_rr Fx_rl delta` | the rates of those five |

- **`point_mass`** — friction circle on a curve; fastest; for track, corridor
  margin, and config checks.
- **`dynamic_bicycle`** — sideslip and yaw; simplified Magic Formula per axle;
  optional friction ellipse.
- **`four_wheel`** — per-wheel Pacejka, aero, load transfer, torque vectoring;
  wheel forces and steering are states; inputs are their rates (`dFxmax`,
  `ddeltamax`). Originally developed by
  [Tanmay Ganguli](https://github.com/TanmayGanguli09). This is the model AMZ
  used for most of the 2026 season; `point_mass` and `dynamic_bicycle` were
  mostly development aids.

## One car, three descriptions

`configs/vehicle.yaml` defines the car once; each model block holds only what
that model needs. Unused parameters are still declared so the three blocks
stay consistent — `test_models_agree_on_the_car_they_describe` enforces this.

`lf` is the distance to the **front** axle, so the static front load share is
`lr / (lf + lr)`. Same convention in all three, and in the configs.

## Guards

Two quantities can go singular; each model declares a floor:

- `eps_s_dot` — `ṡ` in the space-domain conversion (`dx/ds = (dx/dt) / ṡ`);
- `eps_D_kappa` — Frenet Jacobian `D_kappa = 1 - kappa * d`; see
  `examples/frenet_frame.py`.

`dynamic_bicycle` and `four_wheel` enforce both as constraints.

**`point_mass` declares `eps_D_kappa` but does not enforce it**; corner
constraints still divide by raw `D_kappa`. Not enforced because it would change
lap times.

## Adding one

1. Subclass `VehicleModel` here and implement the interface above.
2. Register it in `pipeline._make_model`.
3. Add a panel renderer and an entry in `visualization.PANEL_RENDERERS` —
   `tests/test_panel_registry.py` fails until both exist.
4. Add its block to `configs/vehicle.yaml`.
5. Add a golden scenario in `tests/test_golden_solutions.py`.
