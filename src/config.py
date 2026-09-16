"""Central configuration loader for the Vilex pipeline.

Every stage reads its tunables from a single ``config.yaml`` (repo root by
default) so parameters live in one place instead of being scattered as module
constants and argparse defaults.

Resolution order (highest first):

1. Environment overrides -- any ``VILEX_<PATH>`` variable, where ``<PATH>`` is
   the dotted config path with ``__`` as the separator, e.g.
   ``VILEX_LLM__TEMPERATURE=0.3`` or ``VILEX_STAGE5_TTS__TAGS__RENDER=true``.
2. The YAML file selected by, in order: the explicit ``path`` argument, the
   ``VILEX_CONFIG`` env var, ``./config.yaml``, then ``<repo>/config.yaml``.
3. In-code defaults -- each module keeps its own constants as a fallback, used
   via :func:`cfg_get` when a key is absent.

Secrets (``GEMINI_CREDENTIALS`` / ``GEMINI_API_KEY`` / ``OPENAI_API_KEY``) are
deliberately *not* read from the YAML; keep them in the environment.
"""

import json
import os
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
