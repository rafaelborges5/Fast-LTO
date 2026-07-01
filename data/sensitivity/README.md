# Skidpad score sensitivity study

One-at-a-time (OAT) local sensitivity of the **average timed-lap time** (the FS
*score*, `profiling.skidpad_score_s` = mean of the two timed laps) to the
four-wheel vehicle/solver parameters.

Reproduce with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python src/experiments/skidpad_sensitivity.py \
    --config configs/skidpad.yaml --delta 0.15 --jobs 4
```

## Method

- Skidpad track is built once from `configs/skidpad.yaml`; the four-wheel OCP is
  then re-solved for each perturbation (IPOPT, euler integrator, identical
  `time_weights`/`terminal_speed`).
- Each parameter is moved **±15 %** around the baseline, one at a time. Knobs
  that group several params (`D_rear` = `D_rr` & `D_rl`, `D_front` = `D_fl` &
  `D_fr`) scale all of them together.
- **Elasticity** `E = (Δscore/score) / (Δp/p)` is a dimensionless slope: `E=-0.45`
  means a +1 % parameter increase lowers the average lap by 0.45 %. The ranking
  uses the absolute lap-time **swing** (seconds) over the ±15 % window.
- Baseline average timed-lap score: **4.740 s**. All 19 solves returned
  `Solve_Succeeded`.

## Results (ranked by influence)

| Rank | Parameter | Base | −15 % | +15 % | swing [s] | elasticity |
|-----:|-----------|-----:|------:|------:|----------:|-----------:|
| 1 | rear grip `D_rear` | 1.20 | 5.095 | 4.457 | **0.638** | −0.449 |
| 2 | front grip `D_front` | 1.20 | 4.902 | 4.629 | 0.273 | −0.192 |
| 3 | mass `m` | 170 | 4.615 | 4.835 | 0.219 | +0.154 |
| 4 | downforce `C_l` | 5.54 | 4.844 | 4.634 | 0.210 | −0.148 |
| 5 | margin `boundary_margin` | 0.20 | 4.732 | 4.748 | 0.016 | +0.011 |
| 6 | drag `C_d` | 1.58 | 4.733 | 4.748 | 0.015 | +0.011 |
| 7 | `reg_u_l2` | 0.015 | 4.734 | 4.748 | 0.014 | +0.010 |
| 8 | CG height `h` | 0.246 | 4.737 | 4.744 | 0.007 | +0.005 |
| 9 | yaw inertia `Iz` | 250 | 4.738 | 4.742 | 0.004 | +0.003 |

See `skidpad_sensitivity.png` (tornado plot), `skidpad_sensitivity.csv` (every
raw solve), and `skidpad_sensitivity_summary.json` (machine-readable summary).

## Takeaways

- **Tyre peak grip dominates.** Rear grip `D_rear` is the single biggest lever
  (~3× the next parameter); front grip is second. Skidpad is a steady-state,
  grip-limited corner, so peak `D` maps almost directly into corner speed, and
  the rear-biased static load (`lf > lr`) makes the rear tyres the limiting
  pair. This is also the least free parameter physically — it reflects the tyre
  model, not something you tune.
- Of the *tunable / design* knobs, **mass** and **aero downforce `C_l`** are the
  meaningful ones (~0.21–0.22 s over ±15 %, elasticity ≈ 0.15). They trade off
  in opposite directions, as expected. Note `C_l` only enters through downforce,
  so its grip benefit scales with v² and is modest at skidpad speeds (~12 m/s).
- **Solver / corridor knobs are negligible**: `reg_u_l2`, `boundary_margin`,
  `C_d`, `h`, and `Iz` each move the score by <0.02 s (elasticity ≤ 0.01). The
  L2 regularisation is well below the level where it distorts the lap, and yaw
  inertia barely matters in this near-steady manoeuvre — both reassuring for the
  current setup.

> Caveat: these are *local* ±15 % OAT sensitivities about one operating point.
> Interactions between parameters and larger excursions (e.g. big margin or grip
> changes) can be non-linear; widen `--delta` or run a coupled sweep to probe
> those.
