"""Tests for the centralized config loader (src/config.py)."""

import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    REPO_ROOT,
    apply_runtime_config,
    cfg_get,
    load_config,
    resolve_llm_model,
)


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


def test_resolve_llm_model_role_key_wins():
    llm = {"model": "shared", "writer_model": "writer-only"}
    assert resolve_llm_model(llm, "writer_model", "fallback") == "writer-only"


def test_resolve_llm_model_falls_back_to_shared_model():
    llm = {"model": "DeepSeek-V4-Flash", "writer_model": ""}
    assert resolve_llm_model(llm, "writer_model", "fallback") == "DeepSeek-V4-Flash"
    assert resolve_llm_model({"model": "x"}, "boundary_model", "fallback") == "x"


def test_resolve_llm_model_falls_back_to_default():
    assert resolve_llm_model({}, "tt_model", "in-code") == "in-code"
    assert resolve_llm_model(None, "tt_model", "in-code") == "in-code"
    assert resolve_llm_model({"writer_model": ""}, "writer_model", "in-code") == "in-code"


def test_apply_runtime_config_bridges_gemini_knobs(monkeypatch):
    monkeypatch.delenv("GEMINI_MIN_INTERVAL", raising=False)
    monkeypatch.delenv("GEMINI_LOCATION", raising=False)
    apply_runtime_config(
        {"llm": {"gemini_min_interval": 2.5, "gemini_location": "europe-west1"}}
    )
    assert os.environ["GEMINI_MIN_INTERVAL"] == "2.5"
    assert os.environ["GEMINI_LOCATION"] == "europe-west1"


def test_apply_runtime_config_env_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_MIN_INTERVAL", "9.0")
    monkeypatch.setenv("GEMINI_LOCATION", "us-central1")
    apply_runtime_config({"llm": {"gemini_min_interval": 0.5, "gemini_location": "asia"}})
    assert os.environ["GEMINI_MIN_INTERVAL"] == "9.0"
    assert os.environ["GEMINI_LOCATION"] == "us-central1"


def test_apply_runtime_config_empty_is_noop(monkeypatch):
    monkeypatch.delenv("GEMINI_MIN_INTERVAL", raising=False)
    apply_runtime_config({})
    assert "GEMINI_MIN_INTERVAL" not in os.environ
