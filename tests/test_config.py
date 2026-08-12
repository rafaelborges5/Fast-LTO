"""Tests for RunConfig <-> PipelineConfig plumbing, focused on skidpad_start_x/y."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_skidpad_start_defaults_to_none() -> None:
    rc = RunConfig()
    assert rc.skidpad_start_x is None
    assert rc.skidpad_start_y == 0.0

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x is None
    assert pc.skidpad_start_y == 0.0


def test_skidpad_yaml_loads_start_x() -> None:
    config_path = CONFIGS_DIR / "skidpad.yaml"
    if not config_path.exists():
        pytest.skip("configs/skidpad.yaml not present in this checkout")

    rc = RunConfig.from_yaml(config_path)
    assert rc.skidpad_start_x == pytest.approx(4.0)
    assert rc.skidpad_start_y == pytest.approx(0.0)

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x == pytest.approx(4.0)
    assert pc.skidpad_start_y == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["autox.yaml", "trackdrive.yaml"])
def test_non_skidpad_configs_unaffected_by_new_fields(name: str) -> None:
    """Non-skidpad configs must still load with the new fields left at defaults."""
    config_path = CONFIGS_DIR / name
    if not config_path.exists():
        pytest.skip(f"configs/{name} not present in this checkout")

    rc = RunConfig.from_yaml(config_path)
    assert rc.skidpad_start_x is None

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x is None


def test_autox_ocp_lead_defaults_to_zero() -> None:
    rc = RunConfig()
    assert rc.autox_ocp_lead_m == 0.0

    pc = rc.to_pipeline_config()
    assert pc.autox_ocp_lead_m == 0.0


def test_autox_yaml_loads_ocp_lead() -> None:
    config_path = CONFIGS_DIR / "autox.yaml"
    if not config_path.exists():
        pytest.skip("configs/autox.yaml not present in this checkout")

    rc = RunConfig.from_yaml(config_path)
    assert rc.autox_ocp_lead_m == pytest.approx(0.0)
    assert rc.autox_lead_in_m == pytest.approx(8.0)

    pc = rc.to_pipeline_config()
    assert pc.autox_ocp_lead_m == pytest.approx(0.0)


def test_autox_start_defaults_to_origin() -> None:
    rc = RunConfig()
    assert rc.autox_start_x == 0.0
    assert rc.autox_start_y == 0.0
    assert rc.autox_start_node_offset == 1

    pc = rc.to_pipeline_config()
    assert pc.autox_start_x == 0.0
    assert pc.autox_start_y == 0.0
    assert pc.autox_start_node_offset == 1


def test_autox_yaml_loads_start_fields() -> None:
    config_path = CONFIGS_DIR / "autox.yaml"
    if not config_path.exists():
        pytest.skip("configs/autox.yaml not present in this checkout")

    rc = RunConfig.from_yaml(config_path)
    assert rc.autox_start_x == pytest.approx(0.0)
    assert rc.autox_start_y == pytest.approx(0.0)
    assert rc.autox_start_node_offset == 1

    pc = rc.to_pipeline_config()
    assert pc.autox_start_node_offset == 1
