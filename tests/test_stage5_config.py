"""Tests for the Stage 5 config resolver (per-sub-stage blocks + legacy wrap)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts_render"))

from stage5_config import (  # noqa: E402
    effective_language,
    monolithic_view,
    resolve_stage5,
    work_root,
)

BLOCKS = (
    "stage5",
    "stage5_1a_prep",
    "stage5_1b_render",
    "stage5_2a_align",
    "stage5_2b_assemble",
)


def _cfg() -> dict:
    return {
        "run": {"seed": 7, "target_language": "vi"},
        "stage5": {
            "mode": "split",
            "num_variants": 3,
            "language": "vi",
            "tags": {"render": True},
        },
        "stage5_1a_prep": {"work_root": "/tmp/w"},
        "stage5_1b_render": {
            "device": "cuda",
            "omnivoice": {"batch_size": 4},
        },
        "stage5_2a_align": {
            "vad_threshold": 0.4,
            "aligner": {"granularity": "sentence", "batch": True},
        },
        "stage5_2b_assemble": {
            "profile": True,
            "timing": {"gap": {"exp_scale": 0.5}},
            "audio": {"target_lufs": -20.0},
        },
    }


def test_new_blocks_resolve_and_defaults_fill():
    s5 = resolve_stage5(_cfg())
    for block in BLOCKS:
        assert block in s5, block
    assert s5["stage5"]["num_variants"] == 3
    assert s5["stage5"]["seed"] == 7  # filled from run.seed
    assert s5["stage5_1b_render"]["omnivoice"]["batch_size"] == 4
    assert s5["stage5_2a_align"]["aligner"]["granularity"] == "sentence"
    assert s5["stage5_2b_assemble"]["profile"] is True
    # defaults still present for untouched keys
    assert s5["stage5_2a_align"]["aligner"]["model"].startswith("Qwen/Qwen3-ForcedAligner")
    assert "supported" in s5["stage5"]["tags"]
    # single device per sub-stage: the aligner has no separate device key.
    assert s5["stage5_2a_align"]["device"] == "cuda"
    assert "device" not in s5["stage5_2a_align"]["aligner"]


def test_legacy_stage5_tts_is_wrapped():
    cfg = {
        "run": {"seed": 99, "target_language": "vi"},
        "stage5_tts": {
            "num_variants": 2,
            "seed": 5,
            "device": "cuda",
            "aligner": {"granularity": "sentence"},
            "omnivoice": {"batch_size": 4},
            "timing": {"gap": {"exp_scale": 0.5}},
            "audio": {"target_lufs": -20.0, "save_align_json": True},
        },
    }
    s5 = resolve_stage5(cfg)
    assert s5["stage5"]["num_variants"] == 2
    assert s5["stage5"]["seed"] == 5
    assert s5["stage5_1b_render"]["omnivoice"]["batch_size"] == 4
    assert s5["stage5_2a_align"]["aligner"]["granularity"] == "sentence"
    assert s5["stage5_2b_assemble"]["timing"]["gap"]["exp_scale"] == 0.5
    assert s5["stage5_2b_assemble"]["audio"]["target_lufs"] == -20.0
    assert s5["stage5_2b_assemble"]["audio"]["save_align_json"] is True


def test_work_root_prefers_prep_block():
    s5 = resolve_stage5(_cfg())
    assert work_root(s5, {"paths": {"stage5_work_root": "/p"}}) == "/tmp/w"

    s5b = resolve_stage5({"run": {}, "stage5_1a_prep": {"work_root": None}})
    assert work_root(s5b, {"paths": {"stage5_work_root": "/p"}}) == "/p"


def test_effective_language_override_and_shared():
    s5 = resolve_stage5(_cfg())
    assert effective_language(s5, "stage5_1b_render") == "vi"
    s5["stage5_1b_render"]["language"] = "en"
    assert effective_language(s5, "stage5_1b_render") == "en"


def test_monolithic_view_maps_flat_shape():
    view = monolithic_view(resolve_stage5(_cfg()))
    assert view["backend"] == "omnivoice"
    assert view["device"] == "cuda"
    assert view["num_variants"] == 3
    assert view["aligner"]["granularity"] == "sentence"
    assert view["aligner"]["device"] == "cuda"  # from stage5_2a_align.device
    assert view["omnivoice"]["batch_size"] == 4
    assert view["profile"] is True
    assert view["timing"]["gap"]["exp_scale"] == 0.5
    assert view["audio"]["target_lufs"] == -20.0
    assert view["audio"]["vad_threshold"] == 0.4
    assert view["audio"]["max_prompt_secs"] == 10


def test_legacy_stage5_env_translation(monkeypatch):
    from src.config import load_config

    monkeypatch.setenv("VILEX_STAGE5_TTS__DEVICE", "cpu")
    cfg = load_config(None)  # applies the legacy-env translation
    s5 = resolve_stage5(cfg)
    assert s5["stage5_1b_render"]["device"] == "cpu"
