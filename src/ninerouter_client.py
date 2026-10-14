"""Dedicated LLM route through a local 9router gateway.

9router exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint and
forwards each request to the upstream named by the model id
(``ag/gemini-3.6-flash-medium``, ``cc/claude-...``, ...). This route is kept
separate from ``src/llm_client.make_client`` on purpose: that function picks a
backend from the model NAME, so a 9router id containing ``gemini`` / ``gpt`` /
``claude`` would be sent to Google / OpenAI / Anthropic directly instead of the
gateway. Here every request goes to ``nine_router.base_url``, whatever the name.

Enabled by a ``nine_router`` section with ``enabled: true`` in the config file
(see ``config_9router.example.yaml``). The retry policy mirrors
``gen_test/src/gendata_gemini.py``.
"""

import os
import re
import threading
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from src.config import cfg_get

DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
ROLES = ("writer", "boundary", "tt", "bc")

# Substrings of an exception message that mark a retryable upstream failure.
TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "Deadline",
    "overloaded",
    "timed out",
    "Connection error",
)
_RETRY_IN_RE = re.compile(r"retry in\s+([\d.]+)\s*s", re.IGNORECASE)
# 9router appends "(reset after 1m 4s)" / "(reset after 26h 3m 2s)" when it parks
# an upstream account. A short reset (cooldown 404/429) is worth waiting for; a
# long one (daily quota) is handled by max_wait in _create.
_RESET_AFTER_RE = re.compile(
    r"reset after\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?", re.IGNORECASE
)
# Auth / credit / permission failures never heal by waiting, whatever else the text says.
_PERMANENT_RE = re.compile(r"(?:\[|Error code: )(?:401|402|403)\b")


def _reset_after_seconds(msg: str) -> Optional[float]:
    m = _RESET_AFTER_RE.search(msg)
    if not m or not any(m.groups()):
        return None
    h, mins, sec = (int(g or 0) for g in m.groups())
    return float(h * 3600 + mins * 60 + sec)


def _is_transient(msg: str) -> bool:
    if _PERMANENT_RE.search(msg):
        return False
    return _reset_after_seconds(msg) is not None or any(m in msg for m in TRANSIENT_MARKERS)


def _wait_hint(msg: str) -> Optional[float]:
    """Seconds the upstream asks us to wait: "retry in Ns" or "reset after ..."."""
    hint = _RETRY_IN_RE.search(msg)
    if hint:
        return float(hint.group(1))
    return _reset_after_seconds(msg)


def is_9router_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when the config carries ``nine_router.enabled: true``."""
    return bool(cfg_get(cfg or {}, "nine_router.enabled", False))


def load_9router_settings(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve the ``nine_router`` section, falling back to ROUTER_* env vars."""
    section = cfg_get(cfg or {}, "nine_router", {}) or {}
    base_url = section.get("base_url") or os.getenv("ROUTER_BASE_URL") or DEFAULT_BASE_URL
    api_key = section.get("api_key") or os.getenv("ROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "9router route is enabled but no API key is set. "
            "Export ROUTER_API_KEY=... or fill nine_router.api_key."
        )
    if not section.get("model"):
        raise ValueError("9router route is enabled but nine_router.model is empty.")
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": section["model"],
        **{f"{role}_model": section.get(f"{role}_model") or "" for role in ROLES},
        "timeout": float(section.get("timeout", 300)),
        "retries": max(1, int(section.get("retries", 4))),
        "min_interval": float(section.get("min_interval", 2.0)),
        "max_wait": float(section.get("max_wait", 120)),
    }


def resolve_9router_model(settings: Dict[str, Any], role: str) -> str:
    """Model id for `role` (writer|boundary|tt|bc); empty role key -> shared model."""
    if role not in ROLES:
        raise ValueError(f"unknown 9router role {role!r}; expected one of {ROLES}")
    return settings.get(f"{role}_model") or settings["model"]


class _Completions:
    def __init__(self, owner: "NineRouterClient"):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class _Chat:
    def __init__(self, owner: "NineRouterClient"):
        self.completions = _Completions(owner)


class NineRouterClient:
    """``client.chat.completions.create(**kwargs)`` against 9router.

    - sends ``stream=False`` unless the caller says otherwise;
    - drops ``extra_body``: Stage 4 passes backend-specific thinking switches
      that 9router would forward upstream, where they can fail with 400;
    - spaces requests by ``min_interval`` seconds (shared across threads);
    - retries transient failures, honouring a "retry in Ns" hint, otherwise
      backing off from 8s and doubling, capped by ``max_wait``; a hint above
      ``max_wait`` raises at once instead of sleeping.
    """

    _is_9router = True

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 300,
        retries: int = 4,
        min_interval: float = 2.0,
        max_wait: float = 120,
    ):
        # max_retries=0: the loop in _create owns retrying.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.base_url = base_url
        self.retries = retries
        self.min_interval = min_interval
        self.max_wait = max_wait
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.chat = _Chat(self)

    def _wait_turn(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _create(self, **kwargs):
        kwargs.pop("extra_body", None)
        # 9router streams SSE when `stream` is absent; ask for one JSON body.
        kwargs.setdefault("stream", False)
        delay = 8.0
        for attempt in range(1, self.retries + 1):
            self._wait_turn()
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc)
                if attempt == self.retries or not _is_transient(msg):
                    raise
                hint = _wait_hint(msg)
                wait = hint if hint is not None else min(delay, self.max_wait)
                if wait > self.max_wait:
                    # A daily-quota hint ("reset after 26h", "retry in 39933s") is not
                    # worth sleeping on.
                    raise RuntimeError(
                        f"9router upstream asks to wait {wait:.0f}s (> max_wait "
                        f"{self.max_wait:.0f}s), quota likely exhausted: {msg[:200]}"
                    ) from exc
                first_line = msg.splitlines()[0][:90] if msg else type(exc).__name__
                print(
                    f"[9router] transient error, retry in {wait:.0f}s "
                    f"[{attempt}/{self.retries - 1}]: {first_line}"
                )
                time.sleep(wait)
                delay *= 2


def make_9router_client(settings: Dict[str, Any]) -> NineRouterClient:
    return NineRouterClient(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        timeout=settings["timeout"],
        retries=settings["retries"],
        min_interval=settings["min_interval"],
        max_wait=settings["max_wait"],
    )
