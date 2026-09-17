# configs/

One file describes the car. One file per event describes what to do with it.

```
vehicle.yaml      the nominal car, and nothing else
trackdrive.yaml   extends: vehicle.yaml  + a trackdrive pipeline
autox.yaml        extends: vehicle.yaml  + autox deltas + an autox pipeline
skidpad.yaml      extends: vehicle.yaml  + skidpad deltas + a skidpad pipeline
```

Run one with `fast-lto --config configs/autox.yaml`. Any CLI flag overrides the
file; anything not passed comes from it.

## How a file resolves

`extends:` is resolved bottom-up and deep-merged, so an event file only has to
name what differs. `trackdrive.yaml` has no `vehicle:` block at all — it is the
nominal car. `autox.yaml` has eight vehicle values, because an autocross lap is
run on a different tyre and aero setup, and nothing else.

That means a tyre coefficient is edited in exactly one place, and the three
events cannot quietly end up describing three different cars. The path is
resolved relative to the file naming it, so the directory can be copied or moved
as a unit.

Every key is validated against a known field. A typo raises rather than being
silently ignored, and so does naming one event's settings block in another
event's config.

## Layout of `vehicle.yaml`

Grouped by meaning, not alphabetically:

1. **Mass and geometry** — `m`, `g`, `lf`, `lr`, and `corners`. `lf` is to the
   *front* axle. `corners` are the car's physical extents as `(name, dx, dy)`
   from the CoG, and they are defined here once: no event may redefine them,
   because they are not a tuning knob.
2. **Operating envelope** — the speed, lateral-offset and heading-error bounds
   the solver normalises against. They have to be finite.
3. **Numerical guards** — `v_eps`, `smoothmax_eps`, `eps_s_dot`, `eps_D_kappa`.
   See `src/fast_lto/vehicle_models/README.md`.
4. **One block per model** — `point_mass`, `dynamic_bicycle`, `four_wheel`, each
   ordered inertia → tyres → actuator limits. Only the block for the selected
   `model_name` is used, but all three stay in step.

Units are annotated inline where the field name does not already carry them, so
`m: 165.0  # kg` but `ds_m` and `lead_in_m` get nothing.

## The knobs that matter

Of the pipeline settings, four change the answer materially:

| | |
| --- | --- |
| `model_name` | `point_mass` solves in seconds and is how you check a setup; `four_wheel` is the real answer and takes minutes |
| `ds_m` | node spacing. 0.5 m is the working value; coarser is faster and less honest about actuator rates |
| `boundary_margin` | metres kept clear of each boundary. The single biggest lever on lap time, and on whether the solve starts feasible at all |
| `reg_u` / `reg_u_l2` | input-rate and input penalties. Too low and the inputs chatter; too high and the car drives conservatively for no physical reason |

Each event block holds that event's own settings —
`src/fast_lto/README.md` covers what they do.

## Your own car

Copy `vehicle.yaml`, change the numbers, and point an event config's `extends:`
at it. Nothing else needs to change.
