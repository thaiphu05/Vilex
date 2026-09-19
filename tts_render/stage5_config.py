"""Resolve the Stage 5 config tree (shared + one block per sub-stage).

Canonical YAML keeps the shared parameters at the top level of ``stage5:`` and
gives every sub-stage its own block:

    stage5:               # shared: mode, num_variants, seed, language, tags, ...
    stage5_1a_prep:       # 5.1a CPU text prep
    stage5_1b_render:     # 5.1b GPU OmniVoice
    stage5_2a_align:      # 5.2a GPU VAD + Qwen3
    stage5_2b_assemble:   # 5.2b CPU placement + timing + LUFS

The monolithic ``convert_spoken.py`` keeps its flat view through
:func:`monolithic_view`.

Backward compatibility: a legacy flat ``stage5_tts:`` block is still accepted
and mapped onto the new tree (new-style blocks win key-by-key); the legacy env
``VILEX_STAGE5_TTS__*`` is translated in ``src/config.py``.
"""

import copy
from typing import Any, Dict, Optional

from src.config import cfg_get

_DEFAULT_TAGS = {
    "render": False,
    "supported": [
        "[laughter]",
        "[sigh]",
        "[confirmation-en]",
        "[question-en]",
        "[question-ah]",
        "[question-oh]",
        "[question-ei]",
        "[question-yi]",
        "[surprise-ah]",
        "[surprise-oh]",
        "[surprise-wa]",
        "[surprise-yo]",
        "[dissatisfaction-hnn]",
    ],
}

_DEFAULT_BC = {
    "candidates": {
        "vi": ["ưm", "à", "ừ", "vâng", "phải", "ồ", "mm-hm", "uh huh", "ok"],
        "en": ["yeah", "uh-huh", "mm-hmm", "right", "okay"],
    },
    "rising_tokens": ["yeah", "ưm", "vâng", "à", "ừ"],
}

_DEFAULT_VOICE = {
    "user_instruct": "male, northern accent",
    "assistant_instruct": "female, gentle",
}

_DEFAULT_ALIGNER = {
    "model": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
    "dtype": "auto",
    "granularity": "utterance",
    "max_secs": 240,
    "fallback": "proportional",
    "batch": False,
    "batch_size": 8,
}

_DEFAULT_OMNIVOICE = {
    # No `batch` flag: 5.1b always chunks the cross-dialogue queue by batch_size
    # (1 = one item per call); the monolithic path batches when batch_size > 1.
    "batch_size": 8,
    "max_unit_chars": 400,
    "max_retries": 2,
    "fallback_action": "silence",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        elif val is not None:
            base[key] = val
    return base


def _defaults() -> Dict[str, Any]:
    tree = {
        "stage5": {
            "mode": "split",
            "num_variants": 1,
            "max_dialogues": 0,
            "seed": None,
            "language": None,
            "target_sr": 24000,
            "prompt_sr": 16000,
            "tags": _DEFAULT_TAGS,
            "backchannels": _DEFAULT_BC,
            "voice": _DEFAULT_VOICE,
        },
        "stage5_1a_prep": {"work_root": None},
        "stage5_1b_render": {
            "backend": "omnivoice",
            "device": "cuda",
            "speed": 1.2,
            "language": None,
            "omnivoice": _DEFAULT_OMNIVOICE,
        },
        "stage5_2a_align": {
            "device": "cuda",
            "language": None,
            "vad_threshold": 0.3,
            "max_prompt_secs": 10,
            "aligner": _DEFAULT_ALIGNER,
        },
        "stage5_2b_assemble": {
            "profile": False,
            "bc_placement": "auto",
            "cleanup_intermediate": True,
            "timing": {},
            "audio": {},
        },
    }
    return copy.deepcopy(tree)


def _legacy_tree(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """Map a flat ``stage5_tts:`` block onto the new sibling blocks."""
    tree = _defaults()
    audio = _as_dict(legacy.get("audio"))
    tree["stage5"]["mode"] = legacy.get("mode", "split")
    for key in ("num_variants", "max_dialogues", "target_sr", "prompt_sr"):
        if key in legacy:
            tree["stage5"][key] = legacy[key]
    if legacy.get("seed") is not None:
        tree["stage5"]["seed"] = legacy["seed"]
    if legacy.get("language") is not None:
        tree["stage5"]["language"] = legacy["language"]
    if legacy.get("tags"):
        _deep_merge(tree["stage5"]["tags"], _as_dict(legacy["tags"]))
    if legacy.get("backchannels"):
        _deep_merge(tree["stage5"]["backchannels"], _as_dict(legacy["backchannels"]))
    if legacy.get("voice"):
        _deep_merge(tree["stage5"]["voice"], _as_dict(legacy["voice"]))

    _deep_merge(
        tree["stage5_1b_render"],
        {
            "backend": legacy.get("backend", "omnivoice"),
            "device": legacy.get("device", "cuda"),
            "omnivoice": _as_dict(legacy.get("omnivoice")),
        },
    )
    _deep_merge(
        tree["stage5_2a_align"],
        {
            "device": legacy.get("device", "cuda"),
            "vad_threshold": audio.get("vad_threshold", 0.3),
            "max_prompt_secs": audio.get("max_prompt_secs", 10),
            "aligner": _as_dict(legacy.get("aligner")),
        },
    )
    _deep_merge(
        tree["stage5_2b_assemble"],
        {
            "profile": legacy.get("profile", False),
            "bc_placement": legacy.get("bc_placement", "auto"),
            "timing": _as_dict(legacy.get("timing")),
            "audio": audio,
        },
    )
    return tree


def resolve_stage5(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the normalized Stage 5 config (new blocks, defaults filled).

    Reads the canonical sibling blocks when present, otherwise wraps a legacy
    ``stage5_tts:`` block. New-style blocks override legacy values key-by-key.
    """
    legacy = _as_dict(cfg.get("stage5_tts"))
    has_new = any(
        key in cfg
        for key in (
            "stage5",
            "stage5_1a_prep",
            "stage5_1b_render",
            "stage5_2a_align",
            "stage5_2b_assemble",
        )
    )
    if not has_new and legacy:
        tree = _legacy_tree(legacy)
    else:
        tree = _legacy_tree(legacy) if legacy else _defaults()
        for key in (
            "stage5",
            "stage5_1a_prep",
            "stage5_1b_render",
            "stage5_2a_align",
            "stage5_2b_assemble",
        ):
            if isinstance(cfg.get(key), dict):
                _deep_merge(tree[key], cfg[key])

    # run-level fallbacks for the shared fields.
    if tree["stage5"].get("seed") is None:
        tree["stage5"]["seed"] = cfg_get(cfg, "run.seed", 42)
    if tree["stage5"].get("language") is None:
        tree["stage5"]["language"] = cfg_get(cfg, "run.target_language", "vi")
    # One device per sub-stage: the aligner runs on stage5_2a_align.device.
    tree["stage5_2a_align"]["aligner"].pop("device", None)
    # `batch` was replaced by `batch_size` (1 = no batching).
    tree["stage5_1b_render"]["omnivoice"].pop("batch", None)
    return tree


def work_root(s5: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> str:
    """5.1a work root: stage5_1a_prep.work_root, else paths.stage5_work_root."""
    explicit = _as_dict(s5.get("stage5_1a_prep")).get("work_root")
    if explicit:
        return str(explicit)
    return cfg_get(cfg or {}, "paths.stage5_work_root", "data/vi_stage5_work")


def effective_language(s5: Dict[str, Any], block: str) -> str:
    """Per-stage ``language`` override, else the shared stage5.language."""
    override = _as_dict(s5.get(block)).get("language")
    if override:
        return str(override)
    return str(_as_dict(s5.get("stage5")).get("language") or "vi")


def monolithic_view(s5: Dict[str, Any]) -> Dict[str, Any]:
    """Flat ``stage5_tts``-shaped dict for the monolithic convert_spoken path."""
    shared = _as_dict(s5.get("stage5"))
    render = _as_dict(s5.get("stage5_1b_render"))
    align = _as_dict(s5.get("stage5_2a_align"))
    assemble = _as_dict(s5.get("stage5_2b_assemble"))
    audio = dict(_as_dict(assemble.get("audio")))
    audio["vad_threshold"] = align.get("vad_threshold", 0.3)
    audio["max_prompt_secs"] = align.get("max_prompt_secs", 10)
    # The monolithic path reads a single aligner dict; its device comes from the
    # 5.2a stage device (there is no separate aligner.device any more).
    aligner_view = dict(_as_dict(align.get("aligner")))
    aligner_view["device"] = align.get("device", "cuda")
    return {
        "backend": render.get("backend", "omnivoice"),
        "language": shared.get("language", "vi"),
        "device": render.get("device", "cuda"),
        "num_variants": shared.get("num_variants", 1),
        "max_dialogues": shared.get("max_dialogues", 0),
        "seed": shared.get("seed"),
        "target_sr": shared.get("target_sr", 24000),
        "prompt_sr": shared.get("prompt_sr", 16000),
        "tags": shared.get("tags", {}),
        "backchannels": shared.get("backchannels", {}),
        "voice": shared.get("voice", {}),
        "aligner": aligner_view,
        "omnivoice": render.get("omnivoice", {}),
        "profile": assemble.get("profile", False),
        "bc_placement": assemble.get("bc_placement", "auto"),
        "timing": assemble.get("timing", {}),
        "audio": audio,
    }
