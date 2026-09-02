"""
Fast-LTO: Lap Time Optimization Pipeline

Main entry points:
    from fast_lto.pipeline import run_pipeline, PipelineConfig
    from fast_lto.cli import main
"""

from fast_lto.pipeline import (
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
