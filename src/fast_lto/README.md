# fast_lto

One minimum-time trajectory over one track, for one car, for one event.

The problem is posed in the **space domain**: arc length `s` along the track
centreline is the independent variable, not time. That makes the track corridor
a box constraint and lap time the integral being minimised. `optimization/`
covers the formulation; `examples/frenet_frame.py` covers the coordinate frame
it lives in and where that frame breaks down.

## The run

`pipeline.py` is six steps, each writing one artifact under the data root:

| Step | Writes |
| --- | --- |
| `track` | `data/tracks/{track_id}.csv` (generated tracks only) |
| `spline` | `data/discretized/{track_id}.json` |
| `bounds` | `data/discretized/{track_id}_with_widths.json` |
| `ocp` | `data/solutions/{track_id}_{model}_{integrator}_{mode}.json` |
| `export` | `data/output_trajectories/{track_id}_..._{stamp}.csv` |
| `plot` | `ocp_plots/{stamp}/panels.png` |

Every step runs every time. Spline fitting and bounds take hundredths of a
second in front of a solve measured in minutes, so nothing is cached; reuse is
something the caller asks for explicitly with `start_from`, which reads the
previous step's artifact off disk.

`PipelineConfig` is the whole request. Anything derived from another field is a
`@property`, never assigned in `__post_init__` — that method runs more than once
(`RunConfig.to_pipeline_config` re-runs it so a CLI flag can override the YAML),
so a field filled in behind an `is None` guard goes stale silently.

## Three axes of extension

**A vehicle model** (`vehicle_models/`) says what the car is: its state vector,
its dynamics, its constraints, its bounds. See that package's README.

**An event mode** (`modes.py`) says what the event asks of the solver. Each
`EventMode` answers four questions — how to reshape the track, how to weight the
time objective, which solver arguments to pass, which prescribed segments to
stitch on afterwards — so nothing else branches on `config.mode`.

**A track source** (`tracks/`) produces either a cone CSV that the generic
spline-and-bounds path consumes, or, where that path cannot work, the
discretised track directly. Skidpad is the second kind: the manoeuvre overlaps
itself in XY, so a single-pass spline fit cannot represent it.

## The three events

**Trackdrive** is a closed flying lap. `X[N-1] == X[0]`, and nothing at node 0
is pinned — the closed-loop constraint already keeps the state consistent across
the wrap, so pinning `d(0)` and `psi_err(0)` would force the lap through the
centreline for no reason. `TrackdriveMode` is empty; every default is right.

**Autox** is one timed lap from a standing start, with run-off to brake into.
Three things are worth knowing:

- `s = 0` is anchored at the car's real start position, not at the track CSV's
  first row. That row is whichever one the upstream boundary-estimation tool
  happened to emit first, and it has been seen to move several metres between
  mapping sessions on the same physical track. Everything measured from `s = 0`
  — the pinned launch node, the lead-in, the timing gate — moves together with
  the anchor, so anchoring on `(autox.start_x, autox.start_y)` takes that drift
  out of all of them at once.
- The **timed lap ends at the gate**, `timing_offset_m` downstream of the start
  and crossed again one lap later. The lap time is measured gate to gate, and
  the run-off is appended past the gate rather than past the nominal wrap point.
- The **lead-in and the terminal pad are not optimised.** They are prescribed
  constant-speed segments stitched on after the solve by `_splice_segment`.
  Solving for them jointly would ask a model with rate-limited actuators to hit
  an exact speed target while ramping those actuators from rest on a coarse
  mesh, which is frequently infeasible.

**Skidpad** is two timed circles each way, built from a cone map and a reference
line. Its track carries a `timed_mask` (the scored revolutions) and a
`decel_mask` (the exit), and the objective is weighted from them: full weight
where the run is scored, `eps_time` where it is not, zero on the exit so braking
after the finish costs nothing.

Autox writes the same `timed_mask` onto its own track, from the gate rather than
from a mask on disk. That is what lets one downstream convention serve both —
the solution JSON, the plot shading, and which nodes `D_safe_braking` applies
to.

## Configuration

`config.py` resolves YAML into a `VehicleConfig` (the car) plus a
`PipelineConfig` (what to do with it). Every key is validated against a known
field, so a typo raises instead of being ignored. Each event's settings live in
their own block, and naming the block of an event other than the one being run
is an error rather than a silent no-op. See `configs/README.md`.

`paths.py` decides where all of this reads and writes: `FAST_LTO_DATA` if set,
else the source checkout if there is one, else the working directory.
