"""OpenAI-compatible shim for Anthropic-compatible API endpoints.

The pipeline talks to every backend through the OpenAI Python client
(``client.chat.completions.create(...)``). Anthropic's Messages API speaks a
different protocol, so this module wraps it behind that same interface: point
``llm.base_url`` at an Anthropic-compatible server, set a ``claude*`` /
``*anthropic*`` model name (or ``llm.anthropic_mode: true``), and the rest of
the pipeline needs no changes.

``requests`` is used directly rather than the ``anthropic`` SDK so we do not
depend on a specific SDK version -- some Anthropic-compatible gateways lag the
official feature set.
"""

import json
from typing import Any, Dict, List, Optional

import requests

ANTHROPIC_VERSION = "2023-06-01"


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str, model: str):
        self.choices = [_Choice(content)]
        self.model = model


class _Completions:
    def __init__(self, client: "AnthropicClient"):
        self._client = client

    def create(self, **kwargs) -> _Completion:
        return self._client._create(**kwargs)


class _Chat:
    def __init__(self, client: "AnthropicClient"):
        self.completions = _Completions(client)


class _ParsedMessage:
    def __init__(self, parsed: Any, content: str):
        self.parsed = parsed
        self.content = content


class _ParsedChoice:
    def __init__(self, parsed: Any, content: str):
        self.message = _ParsedMessage(parsed, content)


class _ParsedCompletion:
    def __init__(self, parsed: Any, content: str, model: str):
        self.choices = [_ParsedChoice(parsed, content)]
        self.model = model


class _BetaCompletions:
    def __init__(self, client: "AnthropicClient"):
        self._client = client

    def parse(self, response_format=None, **kwargs) -> _ParsedCompletion:
        """Mirror OpenAI's ``beta.chat.completions.parse``.

        The structured schema is enforced by asking for JSON and parsing the
        reply; the raw JSON dict is attached as ``parsed``.
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("response_format", {"type": "json_object"})
        completion = self._client._create(**kwargs)
        content = completion.choices[0].message.content
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        return _ParsedCompletion(parsed, content, completion.model)


class _Beta:
    def __init__(self, client: "AnthropicClient"):
        self.chat = _BetaChat(client)


class _BetaChat:
    def __init__(self, client: "AnthropicClient"):
        self.completions = _BetaCompletions(client)


def _split_system(messages: List[Dict[str, Any]]):
    """Split OpenAI messages into (system_text, anthropic_messages).

    Anthropic takes the system prompt as a top-level field, not a message.
    Multiple system messages are concatenated; other roles are passed through.
    """
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, list):  # OpenAI content parts -> text
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        if role == "system":
            if content:
                system_parts.append(str(content))
        else:
            out.append(
                {"role": "assistant" if role == "assistant" else "user", "content": str(content)}
            )
    return "\n\n".join(system_parts), out


def _extract_text(body: Dict[str, Any]) -> str:
    """Join the text blocks of an Anthropic Messages response."""
    parts = []
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict) and "text" in block:
            parts.append(block.get("text", ""))
    return "".join(parts)


class AnthropicClient:
    """Minimal stand-in for ``openai.OpenAI`` against an Anthropic endpoint."""

    _is_anthropic = True

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        if not api_key:
            raise ValueError(
                "Anthropic backend selected but no API key. Set ANTHROPIC_API_KEY "
                "(or llm.api_key) for an Anthropic-compatible endpoint."
            )
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = _Chat(self)
        self.beta = _Beta(self)

    def _endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def _create(self, **kwargs) -> _Completion:
        model = kwargs.get("model") or self.model_name
        system_text, messages = _split_system(kwargs.get("messages", []))

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens") or 1024,
        }
        if system_text:
            payload["system"] = system_text
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        resp = requests.post(self._endpoint(), headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        return _Completion(_extract_text(body), body.get("model", model))
