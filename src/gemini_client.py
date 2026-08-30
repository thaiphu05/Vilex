"""OpenAI-compatible shim over Google's Gen AI client (``google-genai``).

The Vilex pipeline talks to chat LLMs exclusively through the OpenAI Python
client's ``client.chat.completions.create(messages=...)`` interface. This module
wraps Google's supported ``google-genai`` SDK behind that same interface so no
call site has to change.

Google authenticates Gemini two ways, and the choice is forced by the credential
type:

* **API key** -> Generative Language API (``genai.Client(api_key=...)``).
* **Service account JSON** -> **Vertex AI only** (the Generative Language API
  rejects service-account auth with HTTP 403). So a service-account file is used
  as ``genai.Client(vertexai=True, project=<sa project>, location=<region>,
  credentials=<sa creds>)``.

The native SDK is imported lazily so environments that only use OpenAI / vLLM
models are not forced to install ``google-genai``.

Implemented surface (only what the pipeline uses):
* ``client.chat.completions.create(...)`` -> ``.choices[0].message.content``
* ``client.beta.chat.completions.parse(...)`` -> same (Gemini never takes this
  branch because its ``base_url`` does not contain ``openai.com``).
"""

import re
import time
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    types = None


# --------------------------------------------------------------------------
# OpenAI-shaped response objects
# --------------------------------------------------------------------------
class _Message:
    def __init__(self, content: str):
        self.content = content
        self.parsed = None
        self.refusal = None
        self.role = "assistant"


class _Choice:
    def __init__(self, content: str):
        self.index = 0
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Completion:
    def __init__(self, content: str):
        self.id = "chatcmpl-gemini"
        self.choices = [_Choice(content)]
        self.created = 0
        self.model = "gemini"
        self.object = "chat.completion"


# --------------------------------------------------------------------------
# Translation helpers
# --------------------------------------------------------------------------
def _messages_to_gemini(messages: List[Dict[str, str]]):
    """Split OpenAI ``messages`` into (system_instruction, gemini Contents)."""
    system_instruction = None
    contents: List[Any] = []
    for m in messages or []:
        role = (m.get("role") or "user").lower()
        text = m.get("content") or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if role == "system":
            system_instruction = text if system_instruction is None else system_instruction + "\n\n" + text
            continue
        if not text:
            continue
        grole = "user" if role == "user" else "model"
        contents.append(types.Content(role=grole, parts=[types.Part(text=text)]))
    # Gemini requires a user turn first; if the first content is from the model,
    # prepend a short user turn so generation is not rejected.
    if contents and contents[0].role != "user":
        contents.insert(0, types.Content(role="user", parts=[types.Part(text="continue")]))
    return system_instruction, contents


def _extract_text(resp) -> str:
    try:
        text = resp.text
    except Exception:
        text = ""
    if not text:
        try:
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            text = ""
    return text or ""


def _retry_delay_from_error(exc: Exception) -> float:
    """Best-effort parse of the ``RetryInfo`` delay (seconds) from a 429 error."""
    msg = getattr(exc, "message", None) or str(exc)
    m = re.search(r"retry in\s+([\d.]+)\s*s", msg)
    if m:
        return float(m.group(1))
    return 30.0


# --------------------------------------------------------------------------
# Global request pacing (free-tier per-minute caps, e.g. 5 req/min/model)
# --------------------------------------------------------------------------
import threading

_rate_lock = threading.Lock()
_last_call_ts = 0.0
_MIN_INTERVAL = 13.0  # keep comfortably under 5 requests/minute


def _rate_limit() -> None:
    """Block until at least ``_MIN_INTERVAL`` seconds have passed since the
    previous Gemini call, so the combined LLM/boundary/TT traffic stays within
    the free-tier per-minute quota."""
    global _last_call_ts
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    return s


# --------------------------------------------------------------------------
# Shim
# --------------------------------------------------------------------------
class GeminiClient:
    """Mimics ``openai.OpenAI`` for the chat-completions interface."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/",
    ):
        if genai is None or types is None:
            raise ImportError(
                "google-genai is required for Gemini models. "
                "Install it with: pip install google-genai"
            )
        self.model_name = model_name
        self._is_gemini = True
        # Sentinel only; the underlying SDK routes to its own endpoint. Kept as the
        # Generative Language host so is_gemini_client()/no_thinking_extra_body()
        # still recognise this client.
        self.base_url = base_url

        if credentials_path:
            self._client = self._client_from_service_account(credentials_path, location)
        elif api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            raise ValueError(
                "Gemini needs a service-account JSON (GEMINI_CREDENTIALS) or an API key "
                "(GEMINI_API_KEY / --api_key)."
            )

        self.chat = _ChatNamespace(self)
        self.beta = _BetaNamespace(self)

    @staticmethod
    def _client_from_service_account(path: str, location: Optional[str]):
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        import os

        region = (
            location
            or os.getenv("GEMINI_LOCATION")
            or "global"
        )
        project = creds.project_id
        if not project:
            raise ValueError("Service account JSON has no project_id.")
        return genai.Client(
            vertexai=True, project=project, location=region, credentials=creds
        )

    def _create(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> _Completion:
        model = model or self.model_name
        system_instruction, contents = _messages_to_gemini(messages or [])

        cfg_kwargs: Dict[str, Any] = {}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_tokens
        if response_format and response_format.get("type") == "json_object":
            cfg_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**cfg_kwargs)
        resp = self._generate_with_backoff(model, contents, config)
        text = _extract_text(resp)
        if response_format and response_format.get("type") == "json_object":
            text = _strip_json_fence(text)
        return _Completion(text)

    def _generate_with_backoff(self, model, contents, config):
        """Call Gemini, transparently backing off on HTTP 429 rate limits.

        The free tier caps requests per minute per model (e.g. 5/min for
        gemini-2.5-flash). We pace calls with a global minimum spacing so a
        single-dialogue run never trips the per-minute cap, and still honour
        any ``RetryInfo`` delay if a 429 does slip through.
        """
        _rate_limit()
        max_attempts = 8
        delay = 5.0
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as e:  # noqa: BLE001 - need status_code off any client error
                status = getattr(e, "status_code", None)
                if status != 429 or attempt >= max_attempts - 1:
                    raise
                delay = max(delay, _retry_delay_from_error(e))
                time.sleep(delay)
                delay = min(delay * 1.5, 60.0)
                last_exc = e
        if last_exc is not None:
            raise last_exc
        return _Completion(text)

    def with_options(self, **kwargs):
        return self


class _ChatNamespace:
    def __init__(self, client: "GeminiClient"):
        self.completions = _Completions(client)


class _BetaNamespace:
    def __init__(self, client: "GeminiClient"):
        self.chat = _ChatNamespace(client)


class _Completions:
    def __init__(self, client: "GeminiClient"):
        self._client = client

    def create(self, **kwargs) -> _Completion:
        return self._client._create(**kwargs)

    def parse(self, **kwargs) -> _Completion:
        return self._client._create(**kwargs)
