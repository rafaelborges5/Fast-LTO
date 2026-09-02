"""The panel registry must stay in step with the models the pipeline can build.

Plots have no assertions on their pixels, so the failure this guards against is
quiet: add a fourth vehicle model, forget the renderer, and the pipeline solves
happily and then dies at the very last step — after the expensive part.
"""

from __future__ import annotations

import pytest

pytest.importorskip("casadi", reason="pipeline import chain requires casadi")

from fast_lto.pipeline import _make_model  # noqa: E402
from fast_lto.visualization.panels import PANEL_RENDERERS, render_panels  # noqa: E402

MODEL_NAMES = ["point_mass", "dynamic_bicycle", "four_wheel"]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_buildable_model_has_a_renderer(model_name: str) -> None:
    _make_model(model_name)  # raises if the pipeline cannot build it
    assert model_name in PANEL_RENDERERS


def test_registry_has_no_renderer_for_a_model_that_cannot_be_built() -> None:
    for model_name in PANEL_RENDERERS:
        _make_model(model_name)


def test_unknown_model_reports_what_is_supported() -> None:
    with pytest.raises(ValueError, match="No visualization available") as excinfo:
        render_panels("tricycle", {}, None, None, out_path=None, show=False)

    message = str(excinfo.value)
    for model_name in PANEL_RENDERERS:
        assert model_name in message, "the error should list the models that do work"
