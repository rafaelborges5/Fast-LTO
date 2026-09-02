"""
Experiment utilities for Fast-LTO.

Currently includes:
    - ds_scaling: study how OCP solve time and iterations scale with ds.
"""

from .ds_scaling import run_ds_scaling_experiment

__all__ = ["run_ds_scaling_experiment"]

