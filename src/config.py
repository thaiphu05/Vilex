"""Central configuration loader for the Vilex pipeline.

Every stage reads its tunables from a single ``config.yaml`` (repo root by
default) so parameters live in one place instead of being scattered as module
constants and argparse defaults.

Resolution order (highest first):

1. Environment overrides -- any ``VILEX_<PATH>`` variable, where ``<PATH>`` is
   the dotted config path with ``__`` as the separator, e.g.
   ``VILEX_LLM__TEMPERATURE=0.3`` or ``VILEX_STAGE5__TAGS__RENDER=true``.
2. The YAML file selected by, in order: the explicit ``path`` argument, the
   ``VILEX_CONFIG`` env var, ``./config.yaml``, then ``<repo>/config.yaml``.
3. In-code defaults -- each module keeps its own constants as a fallback, used
   via :func:`cfg_get` when a key is absent.

Secrets (``GEMINI_CREDENTIALS`` / ``GEMINI_API_KEY`` / ``OPENAI_API_KEY``) are
deliberately *not* read from the YAML; keep them in the environment.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency in practice
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_PREFIX = "VILEX_"
_ENV_SEP = "__"


def resolve_config_path(path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the config file to load, or None when nothing exists."""
    candidates = []
    if path:
        candidates.append(Path(path))
    env_path = os.getenv(_ENV_PREFIX + "CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(REPO_ROOT / "config.yaml")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    parts = [p for p in dotted.split(_ENV_SEP) if p]
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - malformed env
            return
    if parts:
        node[parts[-1]] = value


def _parse_env_value(raw: str) -> Any:
    """Parse an env override into a bool/number/list/str.

    Prefers YAML, then JSON, then primitive coercion, then the raw string.
    """
    if yaml is not None:
        try:
            return yaml.safe_load(raw)
        except Exception:
            pass
    stripped = raw.strip()
    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    try:
        return json.loads(stripped)
    except Exception:
        pass
    for cast in (int, float):
        try:
            return cast(stripped)
        except Exception:
            pass
    return raw


# Legacy Stage 5 env names (flat `stage5_tts.*`) mapped onto the per-sub-stage
# keys. Kept for one migration cycle so existing runner/notebook exports keep
# working; the new names always win if both are set.
_LEGACY_STAGE5_ENV = {
    "STAGE5_TTS__BACKEND": "STAGE5_1B_RENDER__BACKEND",
    "STAGE5_TTS__DEVICE": "STAGE5_1B_RENDER__DEVICE",
    "STAGE5_TTS__OMNIVOICE__BATCH_SIZE": "STAGE5_1B_RENDER__OMNIVOICE__BATCH_SIZE",
    "STAGE5_TTS__OMNIVOICE__MAX_UNIT_CHARS": "STAGE5_1B_RENDER__OMNIVOICE__MAX_UNIT_CHARS",
    "STAGE5_TTS__OMNIVOICE__MAX_RETRIES": "STAGE5_1B_RENDER__OMNIVOICE__MAX_RETRIES",
    "STAGE5_TTS__OMNIVOICE__FALLBACK_ACTION": "STAGE5_1B_RENDER__OMNIVOICE__FALLBACK_ACTION",
    "STAGE5_TTS__ALIGNER__MODEL": "STAGE5_2A_ALIGN__ALIGNER__MODEL",
    "STAGE5_TTS__ALIGNER__DTYPE": "STAGE5_2A_ALIGN__ALIGNER__DTYPE",
    "STAGE5_TTS__ALIGNER__DEVICE": "STAGE5_2A_ALIGN__ALIGNER__DEVICE",
    "STAGE5_TTS__ALIGNER__GRANULARITY": "STAGE5_2A_ALIGN__ALIGNER__GRANULARITY",
    "STAGE5_TTS__ALIGNER__MAX_SECS": "STAGE5_2A_ALIGN__ALIGNER__MAX_SECS",
    "STAGE5_TTS__ALIGNER__FALLBACK": "STAGE5_2A_ALIGN__ALIGNER__FALLBACK",
    "STAGE5_TTS__ALIGNER__BATCH": "STAGE5_2A_ALIGN__ALIGNER__BATCH",
    "STAGE5_TTS__ALIGNER__BATCH_SIZE": "STAGE5_2A_ALIGN__ALIGNER__BATCH_SIZE",
    "STAGE5_TTS__AUDIO__VAD_THRESHOLD": "STAGE5_2A_ALIGN__VAD_THRESHOLD",
    "STAGE5_TTS__AUDIO__MAX_PROMPT_SECS": "STAGE5_2A_ALIGN__MAX_PROMPT_SECS",
    "STAGE5_TTS__PROFILE": "STAGE5_2B_ASSEMBLE__PROFILE",
    "STAGE5_TTS__BC_PLACEMENT": "STAGE5_2B_ASSEMBLE__BC_PLACEMENT",
    "STAGE5_TTS__TIMING__GAP__EXP_SCALE": "STAGE5_2B_ASSEMBLE__TIMING__GAP__EXP_SCALE",
    "STAGE5_TTS__TIMING__GAP__MIN_SEC": "STAGE5_2B_ASSEMBLE__TIMING__GAP__MIN_SEC",
    "STAGE5_TTS__TIMING__GAP__MAX_SEC": "STAGE5_2B_ASSEMBLE__TIMING__GAP__MAX_SEC",
    "STAGE5_TTS__TIMING__PAUSE__EXP_SCALE": "STAGE5_2B_ASSEMBLE__TIMING__PAUSE__EXP_SCALE",
    "STAGE5_TTS__TIMING__PAUSE__MIN_SEC": "STAGE5_2B_ASSEMBLE__TIMING__PAUSE__MIN_SEC",
    "STAGE5_TTS__TIMING__PAUSE__MAX_SEC": "STAGE5_2B_ASSEMBLE__TIMING__PAUSE__MAX_SEC",
    "STAGE5_TTS__TIMING__INTRA_PAUSE__EXP_SCALE": "STAGE5_2B_ASSEMBLE__TIMING__INTRA_PAUSE__EXP_SCALE",
    "STAGE5_TTS__TIMING__INTRA_PAUSE__MIN_SEC": "STAGE5_2B_ASSEMBLE__TIMING__INTRA_PAUSE__MIN_SEC",
    "STAGE5_TTS__TIMING__INTRA_PAUSE__MAX_SEC": "STAGE5_2B_ASSEMBLE__TIMING__INTRA_PAUSE__MAX_SEC",
    "STAGE5_TTS__TIMING__USER_INTERRUPT_OVERLAP_SEC": "STAGE5_2B_ASSEMBLE__TIMING__USER_INTERRUPT_OVERLAP_SEC",
    "STAGE5_TTS__TIMING__USER_INTERRUPT_PROB": "STAGE5_2B_ASSEMBLE__TIMING__USER_INTERRUPT_PROB",
    "STAGE5_TTS__AUDIO__TARGET_LUFS": "STAGE5_2B_ASSEMBLE__AUDIO__TARGET_LUFS",
    "STAGE5_TTS__AUDIO__NOISE_FLOOR_AMP": "STAGE5_2B_ASSEMBLE__AUDIO__NOISE_FLOOR_AMP",
    "STAGE5_TTS__AUDIO__BACKCHANNEL_ATTENUATION": "STAGE5_2B_ASSEMBLE__AUDIO__BACKCHANNEL_ATTENUATION",
    "STAGE5_TTS__AUDIO__SAVE_ALIGN_JSON": "STAGE5_2B_ASSEMBLE__AUDIO__SAVE_ALIGN_JSON",
    "STAGE5_TTS__NUM_VARIANTS": "STAGE5__NUM_VARIANTS",
    "STAGE5_TTS__MAX_DIALOGUES": "STAGE5__MAX_DIALOGUES",
    "STAGE5_TTS__SEED": "STAGE5__SEED",
    "STAGE5_TTS__LANGUAGE": "STAGE5__LANGUAGE",
    "STAGE5_TTS__TARGET_SR": "STAGE5__TARGET_SR",
    "STAGE5_TTS__PROMPT_SR": "STAGE5__PROMPT_SR",
    "STAGE5_TTS__TAGS__RENDER": "STAGE5__TAGS__RENDER",
    "STAGE5_TTS__TAGS__SUPPORTED": "STAGE5__TAGS__SUPPORTED",
}


def _translate_legacy_stage5_env() -> None:
    """Copy any ``VILEX_STAGE5_TTS__*`` var onto its new sub-stage name.

    Only fills names that are not already set explicitly, so the new keys always
    win during a partial migration.
    """
    translated = []
    for old_suffix, new_suffix in _LEGACY_STAGE5_ENV.items():
        old_key = _ENV_PREFIX + old_suffix
        new_key = _ENV_PREFIX + new_suffix
        if old_key in os.environ and new_key not in os.environ:
            os.environ[new_key] = os.environ[old_key]
            translated.append(old_key)
    if translated:
        print(
            "[config] deprecated VILEX_STAGE5_TTS__* env translated to the new "
            f"stage5_1b_render/stage5_2a_align/stage5_2b_assemble names: {sorted(translated)}",
            file=sys.stderr,
        )


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    for key, raw in os.environ.items():
        if not key.startswith(_ENV_PREFIX) or key == _ENV_PREFIX + "CONFIG":
            continue
        _set_path(cfg, key[len(_ENV_PREFIX) :].lower(), _parse_env_value(raw))


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load the merged configuration dict.

    Returns an empty dict when no YAML file is found, letting callers fall back
    to their in-code defaults.
    """
    _translate_legacy_stage5_env()
    cfg: Dict[str, Any] = {}
    resolved = resolve_config_path(path)
    if resolved is not None:
        if yaml is None:
            raise RuntimeError(
                f"Found config file {resolved} but PyYAML is not installed. "
                "Install it with `pip install pyyaml` (see requirements.txt)."
            )
        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            _deep_merge(cfg, data)
    _apply_env_overrides(cfg)
    return cfg


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Dotted accessor, e.g. ``cfg_get(cfg, "stage5.timing.gap.exp_scale", 0.7)``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def apply_runtime_config(cfg: Optional[Dict[str, Any]] = None) -> None:
    """Push non-secret runtime knobs from config onto the environment.

    The Gemini client reads two knobs from the environment rather than from the
    config dict: the request pacing interval (``GEMINI_MIN_INTERVAL``, read when
    a client is built) and the Vertex AI region (``GEMINI_LOCATION``). This
    bridges ``llm.gemini_min_interval`` / ``llm.gemini_location`` onto them so a
    single edit in config.yaml is enough.

    ``setdefault`` keeps the documented precedence: an explicit environment
    variable still wins over the config file.
    """
    cfg = cfg or {}
    llm = cfg_get(cfg, "llm", {}) or {}
    min_interval = llm.get("gemini_min_interval")
    if min_interval is not None:
        os.environ.setdefault("GEMINI_MIN_INTERVAL", str(min_interval))
    location = llm.get("gemini_location")
    if location:
        os.environ.setdefault("GEMINI_LOCATION", str(location))


def resolve_llm_model(llm_cfg: Optional[Dict[str, Any]], role_key: str, default: str) -> str:
    """Pick the model for one LLM role.

    Resolution order: the role-specific key (``writer_model`` ...), then the
    shared ``llm.model``, then the caller's in-code ``default``. This lets a user
    point every role at one served model by setting only ``llm.model`` (plus
    ``llm.base_url`` / ``llm.api_key``) and override individual roles only when
    they actually need to differ.
    """
    llm_cfg = llm_cfg or {}
    return llm_cfg.get(role_key) or llm_cfg.get("model") or default
