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
    """Working from a clone puts outputs in the clone, whatever the cwd.

    This used to count parents up from ``pipeline.py``, which is right in a
    checkout and silently wrong once the module moves -- or once the package is
    installed, where it points inside site-packages.
    """
    root = PipelineConfig().repo_root

    assert (root / "src" / "fast_lto").is_dir(), (
        f"default repo_root {root} does not contain src/fast_lto; the checkout "
        "search in fast_lto.paths is out of step with the layout"
    )
    assert (root / "pyproject.toml").is_file()


def test_repo_root_ignores_the_working_directory_inside_a_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    """Running from /tmp must still write into the clone, as it always has."""
    monkeypatch.chdir(tmp_path)
    assert PipelineConfig().repo_root == REPO_ROOT


def test_data_root_env_var_wins(tmp_path: Path, monkeypatch) -> None:
    from fast_lto.paths import DATA_ROOT_ENV_VAR

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
    assert PipelineConfig().repo_root == tmp_path.resolve()


def test_an_installed_package_falls_back_to_the_working_directory() -> None:
    """Outside a checkout there is no repo to write into, so use the cwd.

    Checked through ``find_source_checkout`` rather than by installing a wheel:
    the CI ``install`` job covers the real thing. A path that looks like
    site-packages must not resolve to whichever project owns the virtualenv,
    which is why the search requires ``src/fast_lto`` and not merely a
    ``pyproject.toml``.
    """
    from fast_lto.paths import find_source_checkout

    assert find_source_checkout(Path("/usr/lib/python3/site-packages/fast_lto/pipeline.py")) is None
    assert find_source_checkout() == REPO_ROOT


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


# ---------------------------------------------------------------------------
# Derived paths follow the field they are derived from
# ---------------------------------------------------------------------------


def test_track_id_flag_moves_the_csv_it_reads() -> None:
    """``--track-id X`` must read X's CSV, not the previous id's.

    ``track_csv_path`` used to be a field resolved once in ``__post_init__``
    behind an ``is None`` guard. ``to_pipeline_config`` re-runs
    ``__post_init__`` so a flag can override the YAML, but the guard made that
    re-run a no-op for an already-resolved path -- so ``--track-id`` renamed
    every output while still reading the default track's CSV. Silently wrong
    output, and nothing failed.
    """
    config = _config_from_argv("--track-id", "some_other_track")

    assert config.track_id == "some_other_track"
    assert config.track_csv_path.name == "some_other_track.csv"


def test_track_id_moves_the_csv_when_set_after_construction() -> None:
    """The same guarantee for library callers, not just the CLI."""
    config = PipelineConfig(track_id="first")
    assert config.track_csv_path.name == "first.csv"

    config.track_id = "second"
    assert config.track_csv_path.name == "second.csv"


def test_every_output_path_follows_the_track_id() -> None:
    """Inputs and outputs must never disagree about which track this is."""
    config = PipelineConfig(track_id="renamed")

    assert config.track_csv_path.name == "renamed.csv"
    assert config.discretized_track_path.name == "renamed.json"
    assert config.track_with_widths_path.name == "renamed_with_widths.json"
    assert config.solution_path.name.startswith("renamed_")


def test_explicit_csv_override_wins_over_the_track_id() -> None:
    config = PipelineConfig(track_id="ignored", track_csv_override=Path("/elsewhere/real.csv"))
    assert config.track_csv_path == Path("/elsewhere/real.csv")


# ---------------------------------------------------------------------------
# Flags that used to contradict their own help text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ((), None),
        (("--savgol-bounds",), True),
        (("--no-savgol-bounds",), False),
    ],
)
def test_savgol_flag_says_what_it_does(argv, expected) -> None:
    """``--savgol-bounds`` enables smoothing; the negation disables it.

    It was declared ``store_false`` with help text saying "Disable", and the
    caller then set ``use_savgol_bounds = True`` when the flag was present --
    so passing it turned smoothing on, and nothing turned it off.
    """
    yaml_path = FIXTURE_CONFIGS / "no_mode_overrides.yaml"
    from_yaml = RunConfig.from_yaml(yaml_path).pipeline.use_savgol_bounds

    config = _config_from_argv("--config", str(yaml_path), *argv)

    assert config.use_savgol_bounds == (from_yaml if expected is None else expected)


# ---------------------------------------------------------------------------
# The CLI reaches everything the config supports
# ---------------------------------------------------------------------------


def test_cli_offers_every_event_mode() -> None:
    """``--mode`` hand-listed its choices and had gone stale, hiding skidpad."""
    from fast_lto.modes import MODE_NAMES

    choices = _parser_choices("--mode")
    assert set(choices) == set(MODE_NAMES)


def test_cli_offers_every_track_type() -> None:
    from fast_lto.pipeline import TRACK_TYPES

    assert set(_parser_choices("--track-type")) == set(TRACK_TYPES)


# ---------------------------------------------------------------------------
# Config inheritance
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_extends_pulls_in_the_base_config(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "vehicle:\n  m: 165.0\n  v_max: 20.0\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\npipeline:\n  ds_m: 0.25\n")

    config = RunConfig.from_yaml(child)

    assert config.vehicle.m == pytest.approx(165.0)
    assert config.vehicle.v_max == pytest.approx(20.0)
    assert config.pipeline.ds_m == pytest.approx(0.25)


def test_child_values_win_over_the_base(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "vehicle:\n  m: 165.0\n  v_max: 20.0\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\nvehicle:\n  v_max: 14.0\n")

    config = RunConfig.from_yaml(child)

    assert config.vehicle.v_max == pytest.approx(14.0), "the event's own value must win"
    assert config.vehicle.m == pytest.approx(165.0), "unmentioned base values survive"


def test_nested_blocks_merge_rather_than_replace(tmp_path: Path) -> None:
    """Overriding one tyre coefficient must not drop the other twenty."""
    _write(
        tmp_path / "base.yaml",
        "vehicle:\n  four_wheel:\n    B_fl: 9.0\n    C_fl: 1.3\n    D_fl: 1.1\n",
    )
    child = _write(
        tmp_path / "child.yaml", "extends: base.yaml\nvehicle:\n  four_wheel:\n    D_fl: 1.3\n"
    )

    four_wheel = RunConfig.from_yaml(child).vehicle.four_wheel or {}

    assert four_wheel["D_fl"] == pytest.approx(1.3)
    assert four_wheel["B_fl"] == pytest.approx(9.0)
    assert four_wheel["C_fl"] == pytest.approx(1.3)


def test_a_corner_list_replaces_rather_than_appends(tmp_path: Path) -> None:
    """The car has four corners, not eight."""
    _write(
        tmp_path / "base.yaml", "vehicle:\n  corners:\n  - [FL, 1.8, 0.75]\n  - [FR, 1.8, -0.75]\n"
    )
    child = _write(
        tmp_path / "child.yaml",
        "extends: base.yaml\nvehicle:\n  corners:\n  - [FL, 1.7, 0.75]\n  - [FR, 1.7, -0.75]\n",
    )

    corners = RunConfig.from_yaml(child).vehicle.corners or []

    assert len(corners) == 2
    assert corners[0][1] == pytest.approx(1.7)


def test_a_missing_base_says_which_file_wanted_it(tmp_path: Path) -> None:
    child = _write(tmp_path / "child.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        RunConfig.from_yaml(child)


def test_a_cycle_is_reported_not_recursed(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        RunConfig.from_yaml(tmp_path / "a.yaml")


def test_shipped_event_configs_inherit_the_shared_car() -> None:
    """Every event config extends vehicle.yaml and overrides only what it tunes.

    Before the split, 48 vehicle settings were copied into each of the three
    files, so a tyre coefficient had to be edited in three places -- and the
    three could silently come to describe different cars.
    """
    import yaml

    shared = yaml.safe_load((CONFIGS_DIR / "vehicle.yaml").read_text())
    assert "pipeline" not in shared, "vehicle.yaml describes the car, not a run"

    for path in _shipped_configs():
        if path.name == "vehicle.yaml":
            continue
        raw = yaml.safe_load(path.read_text())
        assert raw.get("extends") == "vehicle.yaml", f"{path.name} does not extend the shared car"
        assert "pipeline" in raw, f"{path.name} should carry its own pipeline settings"
        # Only the genuinely per-event knobs stay behind.
        overrides = set(raw.get("vehicle") or {})
        assert overrides <= {"v_max", "corners", "four_wheel"}, (
            f"{path.name} overrides {sorted(overrides)}; anything the events agree on "
            "belongs in vehicle.yaml"
        )


def _parser_choices(flag: str) -> List[str]:
    for action in build_parser()._actions:
        if flag in action.option_strings:
            assert action.choices is not None, f"{flag} declares no choices"
            return list(action.choices)
    raise AssertionError(f"{flag} is not a known flag")
