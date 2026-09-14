"""Tests for the centralized config loader (src/config.py)."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, cfg_get, load_config  # noqa: E402


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_loads_yaml_sections(tmp_path):
    path = _write(
        tmp_path,
        """
        stage4_synthesis:
          max_turns: 7
          guards:
            length_guard_gap: 9
        """,
    )
    cfg = load_config(path)
    assert cfg_get(cfg, "stage4_synthesis.max_turns") == 7
    assert cfg_get(cfg, "stage4_synthesis.guards.length_guard_gap") == 9


def test_cfg_get_returns_default_for_missing_path():
    assert cfg_get({}, "a.b.c", 5) == 5
    assert cfg_get({"a": {"b": 1}}, "a.b") == 1


def test_no_config_anywhere_returns_empty(tmp_path, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VILEX_CONFIG", raising=False)
    assert config.load_config() == {}


def test_env_override_beats_file(tmp_path, monkeypatch):
    path = _write(tmp_path, "llm:\n  temperature: 0.7\n")
    monkeypatch.setenv("VILEX_LLM__TEMPERATURE", "0.1")
    cfg = load_config(path)
    assert cfg_get(cfg, "llm.temperature") == pytest.approx(0.1)


def test_env_list_override(tmp_path, monkeypatch):
    path = _write(tmp_path, "run:\n  datasets: [interviewer]\n")
    monkeypatch.setenv("VILEX_RUN__DATASETS", "[multiwoz, soda]")
    cfg = load_config(path)
    assert cfg_get(cfg, "run.datasets") == ["multiwoz", "soda"]


def test_env_bool_override(tmp_path, monkeypatch):
    path = _write(tmp_path, "run:\n  dry_run: false\n")
    monkeypatch.setenv("VILEX_RUN__DRY_RUN", "true")
    cfg = load_config(path)
    assert cfg_get(cfg, "run.dry_run") is True


def test_repo_config_declares_every_stage_section():
    cfg = load_config(REPO_ROOT / "config_example.yaml")
    for key in (
        "run",
        "paths",
        "llm",
        "stage1_speechify",
        "stage1_5_cross_turn",
        "stage1_75_disfluency",
        "stage4_synthesis",
        "stage4b_backchannel",
        "stage5_tts",
    ):
        assert key in cfg, f"config_example.yaml is missing the {key!r} section"
