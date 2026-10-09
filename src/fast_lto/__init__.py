"""
Fast-LTO: minimum-time trajectory optimisation for Formula Student Driverless.
"""

import os

# MUMPS calls OpenBLAS on small dense blocks; extra threads only add overhead.
# Must be set before CasADi loads its BLAS. An explicit user value wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from fast_lto.pipeline import (  # noqa: E402
    PipelineConfig,
    StepName,
    run_pipeline,
    step_compute_bounds,
    step_fit_spline,
    step_generate_track,
    step_solve_ocp,
    step_visualize,
)

__all__ = [
    "PipelineConfig",
    "StepName",
    "run_pipeline",
    "step_generate_track",
    "step_fit_spline",
    "step_compute_bounds",
    "step_solve_ocp",
    "step_visualize",
]
