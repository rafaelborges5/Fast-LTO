"""Regression tests for track-width (corridor bound) extraction.

Guards the periodic-spline fix in ``_compute_lateral_bounds_kdtree``: on a
closed track the width spline must wrap the start/finish seam instead of
extrapolating linearly into the cone-free gap there (which used to leave a
~0.43 m width discontinuity at the seam).

The synthetic ellipse/bean generators build both boundaries as exact constant
offsets of the centreline (``half_width = 0.5 * track_width_m``), so the analytic
ground truth is ``w_left == w_right == half_width`` at every arc length,
including across the seam.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fast_lto.splines.spline_fitter import fit_and_discretize
from fast_lto.tracks.bean import BeanTrackConfig, generate_bean_track
from fast_lto.tracks.ellipse import EllipseTrackConfig, generate_ellipse_track
from fast_lto.utils.track_bounds import compute_lateral_bounds, load_boundaries


def _widths_for_track(gen, config, csv_path: Path):
    gen(csv_path, config=config)
    track = fit_and_discretize(csv_path, ds_m=0.5, continuity="C2")
    boundaries = load_boundaries(csv_path)
    result = compute_lateral_bounds(track, boundaries["left"], boundaries["right"])
    return track, result


def test_widths_uniform_on_ellipse(tmp_path):
    """Constant-offset ellipse boundaries -> near-uniform half-width everywhere."""
    config = EllipseTrackConfig()
    half_width = 0.5 * config.track_width_m

    _, result = _widths_for_track(generate_ellipse_track, config, tmp_path / "ellipse.csv")

    assert result.misses_left == 0
    assert result.misses_right == 0
    # Ellipse curvature is mild; the 2nd-order curvature correction keeps the
    # recovered width within a few cm of the analytic value everywhere.
    assert np.nanmax(np.abs(result.w_left - half_width)) < 0.05
    assert np.nanmax(np.abs(result.w_right - half_width)) < 0.05


def test_no_seam_discontinuity_on_bean(tmp_path):
    """The start/finish seam must not be an outlier vs. interior width steps.

    This is the direct regression for the periodicity bug: before the fix the
    bean track showed a large step in total width between the last and first
    sample (the linearly-extrapolated seam). After the fix that step is on the
    order of the interior sample-to-sample step.
    """
    config = BeanTrackConfig()
    half_width = 0.5 * config.track_width_m

    _, result = _widths_for_track(generate_bean_track, config, tmp_path / "bean.csv")

    assert result.misses_left == 0
    assert result.misses_right == 0

    width = result.w_left + result.w_right
    seam_step = abs(width[0] - width[-1])
    interior_steps = np.abs(np.diff(width))

    # Seam step is seam-local noise, not a 0.4 m cliff.
    assert seam_step < 0.05
    # And it is not an outlier relative to the interior discretization steps.
    assert seam_step < 5.0 * np.median(interior_steps) + 1e-3

    # The bean has stronger, varying curvature, so allow a looser but still
    # tight band around the analytic half-width.
    assert np.nanmax(np.abs(result.w_left - half_width)) < 0.2
    assert np.nanmax(np.abs(result.w_right - half_width)) < 0.2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
