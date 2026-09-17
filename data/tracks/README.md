# Track data

A track is one CSV of cone positions. The pipeline reads it as
`data/tracks/{track_id}.csv`, so `--track-id fscz_2025` reads
`data/tracks/fscz_2025.csv`.

## Format

Four columns, with a header row:

```
side,cone_id,x,y
L,0,2.057035,1.704656
L,1,5.058166,1.681698
M,0,3.512000,0.041000
R,0,2.143000,-1.620000
```

| Column | Meaning |
| --- | --- |
| `side` | `L` left boundary, `R` right boundary, `M` the midline between them. Case-insensitive; any other value is skipped. |
| `cone_id` | Order along that boundary. Rows are sorted by it, so they need not be in order in the file, but the numbering must follow the direction of travel. |
| `x`, `y` | Position in metres, in the map frame. |

Three things the pipeline assumes:

- **All three sides are present.** `M` is what the centreline spline is fitted
  to; `L` and `R` become the corridor the car has to stay inside.
- **The track is a closed loop.** The spline fit is periodic, and trackdrive's
  final node is constrained back to the first. Autox runs on the same closed
  loop and extends it past the timing gate.
- **The car starts near the origin** for autox, which anchors its launch node at
  the sample nearest `(autox.start_x, autox.start_y)` — both default to `0.0`.
  See `AutoxConfig` in `src/fast_lto/modes.py` if your mapping convention puts
  the start elsewhere.

`cone_id` restarts per side, and the counts per side need not match.

## What ships here

Four real event tracks:

| File | Event |
| --- | --- |
| `fscz_2025.csv` | Formula Student Czech Republic 2025 |
| `fscz_2026.csv` | Formula Student Czech Republic 2026 |
| `fsg_autox_2025.csv` | Formula Student Germany autocross 2025 |
| `fsg_autox_2026.csv` | Formula Student Germany autocross 2026 |

Plus `skidpad/`, the cone map and reference line that `configs/skidpad.yaml`
builds its track from.

`ellipse.csv`, `bean.csv` and `fsg_random.csv` are generated, not measured —
`--track-type ellipse|bean|fsg` rewrites them. They exist so a fresh clone can
run the whole pipeline without any real track data.

## Adding your own

Drop the CSV in this directory and pass its name as `--track-id`. Nothing else
needs to change.

Note that `.gitignore` here is an **allowlist**: anything not named above is
ignored by default, so a boundary CSV off a test day will not be committed by
accident. If you do want to publish one, add a `!data/tracks/<name>.csv` line.
