# research/

Studies and batch tools that sit on top of the library. Not part of the
package: nothing under `src/fast_lto` imports any of it, and it is excluded
from the wheel, so `pip install fast-lto` does not deliver it.

Run them from the repo root. They find the data directory the same way the CLI
does — see `fast_lto.paths` — so they work from a clone without configuration.

## Sweeps

Both take a YAML config as their baseline and add sweep axes on top of it. With
no flags, each solves exactly what `fast-lto --config <that file>` would, so
any difference between two outputs is a difference you asked for.

```bash
# Tyre grip x corridor margin, on the skidpad
python research/experiments/skidpad_batch.py --d-max 1.45 --margins 0.40,0.50 --jobs 4

# Force-rate limit x grip x margin, on an autox lap
python research/experiments/autox_dfx_grip_sweep.py --track-id fscz_2025 --dfx-values 1000,2000
```

Each writes one controller-reference CSV per solve plus a score table, and
rewrites the table after every solve, so an interrupted sweep resumes where it
stopped.

## Viewers

Both take a solution JSON and are worth a look at any solve:

```bash
# Animated lap: per-wheel forces, vertical loads, steering, torque-vectoring
python research/experiments/animate_lap.py data/solutions/<name>.json \
    --track-csv data/tracks/fscz_2025.csv

# Self-contained interactive HTML: slider, synced strip charts, hover states
python research/experiments/interactive_lap.py data/solutions/<name>.json \
    --track-csv data/tracks/fscz_2025.csv
```

The interactive viewer needs Plotly, which is not a library dependency:

```bash
pip install -e ".[research]"
```

## Adding your own

`.gitignore` here is an **allowlist** — a new file under `research/` is ignored
unless it is named there. That is deliberate: this is where one-off analysis
goes, and most of it is tied to data that never leaves the machine it was run
on. Add a `!research/...` line when something becomes general enough to share.
