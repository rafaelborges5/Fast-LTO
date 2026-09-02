"""Tests for RunConfig <-> PipelineConfig plumbing.

Value assertions run against fixtures in ``tests/data/configs``, never against
the shipped ``configs/*.yaml`` — those are tuning artifacts that change with the
car, and a test that pins their values breaks every time someone retunes.  The
shipped configs are still covered here, but only by the contract that actually
has to hold for them: they parse, and they validate against their own model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from fast_lto.config import RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
FIXTURE_CONFIGS = Path(__file__).resolve().parent / "data" / "configs"


def _shipped_configs() -> List[Path]:
    if not CONFIGS_DIR.is_dir():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_skidpad_start_defaults_to_none() -> None:
    rc = RunConfig()
    assert rc.skidpad_start_x is None
    assert rc.skidpad_start_y == 0.0

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x is None
    assert pc.skidpad_start_y == 0.0


def test_autox_ocp_lead_defaults_to_zero() -> None:
    rc = RunConfig()
    assert rc.autox_ocp_lead_m == 0.0

    pc = rc.to_pipeline_config()
    assert pc.autox_ocp_lead_m == 0.0


def test_autox_start_defaults_to_origin() -> None:
    rc = RunConfig()
    assert rc.autox_start_x == 0.0
    assert rc.autox_start_y == 0.0
    assert rc.autox_start_node_offset == 1

    pc = rc.to_pipeline_config()
    assert pc.autox_start_x == 0.0
    assert pc.autox_start_y == 0.0
    assert pc.autox_start_node_offset == 1


# ---------------------------------------------------------------------------
# YAML -> RunConfig -> PipelineConfig
# ---------------------------------------------------------------------------


def test_skidpad_yaml_loads_start_xy() -> None:
    rc = RunConfig.from_yaml(FIXTURE_CONFIGS / "skidpad_start_xy.yaml")
    assert rc.skidpad_start_x == pytest.approx(4.0)
    assert rc.skidpad_start_y == pytest.approx(1.5)

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x == pytest.approx(4.0)
    assert pc.skidpad_start_y == pytest.approx(1.5)


def test_autox_yaml_loads_anchor_fields() -> None:
    rc = RunConfig.from_yaml(FIXTURE_CONFIGS / "autox_anchor.yaml")
    assert rc.autox_lead_in_m == pytest.approx(8.0)
    assert rc.autox_ocp_lead_m == pytest.approx(2.5)
    assert rc.autox_start_x == pytest.approx(12.5)
    assert rc.autox_start_y == pytest.approx(-3.25)
    assert rc.autox_start_node_offset == 3

    pc = rc.to_pipeline_config()
    assert pc.autox_lead_in_m == pytest.approx(8.0)
    assert pc.autox_ocp_lead_m == pytest.approx(2.5)
    assert pc.autox_start_x == pytest.approx(12.5)
    assert pc.autox_start_y == pytest.approx(-3.25)
    assert pc.autox_start_node_offset == 3


def test_unset_mode_fields_keep_their_defaults() -> None:
    """A config that names no anchor fields must not invent values for them."""
    rc = RunConfig.from_yaml(FIXTURE_CONFIGS / "no_mode_overrides.yaml")
    assert rc.skidpad_start_x is None
    assert rc.autox_ocp_lead_m == 0.0
    assert rc.autox_start_node_offset == 1

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x is None
    assert pc.autox_ocp_lead_m == 0.0
    assert pc.autox_start_node_offset == 1


# ---------------------------------------------------------------------------
# Shipped configs: parse-only smoke test, no value assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path", _shipped_configs(), ids=lambda p: p.name
)
def test_shipped_config_parses_and_validates(config_path: Path) -> None:
    """Every config in configs/ must load and match its selected model.

    Deliberately asserts nothing about the values: this catches typo'd keys and
    parameters the chosen model cannot consume, and stays green through retuning.
    """
    rc = RunConfig.from_yaml(config_path)
    rc.validate_for_model()
    rc.to_pipeline_config()


def test_shipped_configs_are_present() -> None:
    """Guard against the parametrized test above silently collecting nothing."""
    assert _shipped_configs(), f"no *.yaml found in {CONFIGS_DIR}"
