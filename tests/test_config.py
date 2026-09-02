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

from fast_lto.cli import build_parser, build_run_config
from fast_lto.config import RunConfig
from fast_lto.pipeline import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
FIXTURE_CONFIGS = Path(__file__).resolve().parent / "data" / "configs"


def _shipped_configs() -> List[Path]:
    if not CONFIGS_DIR.is_dir():
        return []
    return sorted(CONFIGS_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------


def test_default_repo_root_is_the_repository_not_the_package() -> None:
    """The repo root is derived by counting parents up from ``pipeline.py``.

    That count is silently wrong the moment the module moves to a different
    nesting depth, and every other test passes an explicit ``repo_root``, so
    nothing else would notice. Asserted against a marker that only the real
    repository root has.
    """
    root = PipelineConfig().repo_root

    assert (root / "src" / "fast_lto").is_dir(), (
        f"default repo_root {root} does not contain src/fast_lto; the parents[] "
        "depth in PipelineConfig.__post_init__ is out of step with the layout"
    )
    assert (root / "pyproject.toml").is_file()


def test_default_data_paths_hang_off_the_repo_root() -> None:
    config = PipelineConfig(track_id="some_track")
    root = config.repo_root

    assert config.track_csv_path == root / "data" / "tracks" / "some_track.csv"
    assert config.solutions_dir == root / "data" / "solutions"
    assert config.discretized_dir == root / "data" / "discretized"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_skidpad_start_defaults_to_none() -> None:
    rc = RunConfig()
    assert rc.pipeline.skidpad_start_x is None
    assert rc.pipeline.skidpad_start_y == 0.0

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x is None
    assert pc.skidpad_start_y == 0.0


def test_autox_ocp_lead_defaults_to_zero() -> None:
    rc = RunConfig()
    assert rc.pipeline.autox_ocp_lead_m == 0.0

    pc = rc.to_pipeline_config()
    assert pc.autox_ocp_lead_m == 0.0


def test_autox_start_defaults_to_origin() -> None:
    rc = RunConfig()
    assert rc.pipeline.autox_start_x == 0.0
    assert rc.pipeline.autox_start_y == 0.0
    assert rc.pipeline.autox_start_node_offset == 1

    pc = rc.to_pipeline_config()
    assert pc.autox_start_x == 0.0
    assert pc.autox_start_y == 0.0
    assert pc.autox_start_node_offset == 1


# ---------------------------------------------------------------------------
# YAML -> RunConfig -> PipelineConfig
# ---------------------------------------------------------------------------


def test_skidpad_yaml_loads_start_xy() -> None:
    rc = RunConfig.from_yaml(FIXTURE_CONFIGS / "skidpad_start_xy.yaml")
    assert rc.pipeline.skidpad_start_x == pytest.approx(4.0)
    assert rc.pipeline.skidpad_start_y == pytest.approx(1.5)

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x == pytest.approx(4.0)
    assert pc.skidpad_start_y == pytest.approx(1.5)


def test_autox_yaml_loads_anchor_fields() -> None:
    rc = RunConfig.from_yaml(FIXTURE_CONFIGS / "autox_anchor.yaml")
    assert rc.pipeline.autox_lead_in_m == pytest.approx(8.0)
    assert rc.pipeline.autox_ocp_lead_m == pytest.approx(2.5)
    assert rc.pipeline.autox_start_x == pytest.approx(12.5)
    assert rc.pipeline.autox_start_y == pytest.approx(-3.25)
    assert rc.pipeline.autox_start_node_offset == 3

    pc = rc.to_pipeline_config()
    assert pc.autox_lead_in_m == pytest.approx(8.0)
    assert pc.autox_ocp_lead_m == pytest.approx(2.5)
    assert pc.autox_start_x == pytest.approx(12.5)
    assert pc.autox_start_y == pytest.approx(-3.25)
    assert pc.autox_start_node_offset == 3


def test_unset_mode_fields_keep_their_defaults() -> None:
    """A config that names no anchor fields must not invent values for them."""
    rc = RunConfig.from_yaml(FIXTURE_CONFIGS / "no_mode_overrides.yaml")
    assert rc.pipeline.skidpad_start_x is None
    assert rc.pipeline.autox_ocp_lead_m == 0.0
    assert rc.pipeline.autox_start_node_offset == 1

    pc = rc.to_pipeline_config()
    assert pc.skidpad_start_x is None
    assert pc.autox_ocp_lead_m == 0.0
    assert pc.autox_start_node_offset == 1


# ---------------------------------------------------------------------------
# Shipped configs: parse-only smoke test, no value assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_path", _shipped_configs(), ids=lambda p: p.name)
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


# ---------------------------------------------------------------------------
# CLI overrides on top of YAML
# ---------------------------------------------------------------------------


def _config_from_argv(*argv: str) -> PipelineConfig:
    """Run the real CLI parser and override logic, without running a pipeline."""
    args = build_parser().parse_args(list(argv))
    return build_run_config(args).to_pipeline_config()


def test_cli_flag_overrides_the_yaml_value() -> None:
    config = _config_from_argv("--config", str(CONFIGS_DIR / "trackdrive.yaml"), "--ds", "2.0")
    assert config.ds_m == pytest.approx(2.0)


def test_unpassed_flag_does_not_clobber_the_yaml_value() -> None:
    """The old two-branch CLI applied some defaults unconditionally."""
    yaml_path = CONFIGS_DIR / "trackdrive.yaml"
    from_yaml = RunConfig.from_yaml(yaml_path).pipeline.ds_m
    config = _config_from_argv("--config", str(yaml_path))
    assert config.ds_m == pytest.approx(from_yaml)


def test_works_with_no_config_file() -> None:
    config = _config_from_argv("--track-id", "ellipse", "--model", "four_wheel")
    assert config.track_id == "ellipse"
    assert config.model_name == "four_wheel"
    assert config.mode == "trackdrive"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # Config omits initial_speed, so it follows the mode -- including a mode
        # that only arrives on the command line.
        (("--config", "FIXTURE/no_mode_overrides.yaml"), 5.0),
        (("--config", "FIXTURE/no_mode_overrides.yaml", "--mode", "autox"), 3.0),
        # ...unless the caller pinned it, which nothing may override.
        (
            (
                "--config",
                "FIXTURE/no_mode_overrides.yaml",
                "--mode",
                "autox",
                "--initial-speed",
                "7.5",
            ),
            7.5,
        ),
    ],
)
def test_initial_speed_follows_mode_unless_pinned(argv, expected: float) -> None:
    resolved = [a.replace("FIXTURE", str(FIXTURE_CONFIGS)) for a in argv]
    assert _config_from_argv(*resolved).initial_speed == pytest.approx(expected)


def test_yaml_initial_speed_survives_a_mode_override() -> None:
    """An explicit value in the file is the caller's too, not the mode's."""
    config = _config_from_argv("--config", str(CONFIGS_DIR / "trackdrive.yaml"), "--mode", "autox")
    assert config.mode == "autox"
    assert config.initial_speed == pytest.approx(5.0)
