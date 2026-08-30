"""Single definition of the OpenAI-vs-self-hosted backend rule.

Every generation stage talks to a chat LLM through the OpenAI Python client,
and the backend is picked by the *model name*: proprietary OpenAI models read
``OPENAI_API_KEY`` and hit ``api.openai.com``; anything else (``Qwen/Qwen3-32B``
and friends) is posted to the OpenAI-compatible ``--base_url`` you serve
yourself.

The rule used to be spelled three different ways -- a hardcoded set in
``src/speechify_run.py`` and ``src/synthesis/run.py``, a name-prefix test in
``src/synthesis/run_add_bc.py``, and nothing at all in
``src/inference_turntaking_llm.py``, which sent ``gpt-4.1`` to localhost. Import
from here instead of re-deriving it.
"""

import os
from typing import Optional

from openai import OpenAI

# OpenAI-compatible endpoint for Google Gemini. Gemini speaks the OpenAI chat
# completions protocol, so we can keep using the `openai` client and just point
# it at this base URL with a Gemini API key.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Known-good proprietary model names, kept for documentation value. The prefix
# test below is what actually decides, so a newer OpenAI model absent from this
# set still routes correctly rather than silently going to localhost.
GPT_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
}

_OPENAI_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")


def is_openai_model(model_name: str) -> bool:
    """True when `model_name` should be served by api.openai.com."""
    if model_name in GPT_MODELS:
        return True
    return model_name.startswith(_OPENAI_PREFIXES)


def is_gemini_model(model_name: str) -> bool:
    """True when `model_name` is a Google Gemini model served via its
    OpenAI-compatible endpoint."""
    return "gemini" in model_name.lower()


def is_gemini_client(client: OpenAI) -> bool:
    """True when `client` points at the Gemini OpenAI-compatible endpoint."""
    if getattr(client, "_is_gemini", False):
        return True
    return "generativelanguage" in str(getattr(client, "base_url", "") or "")


def make_client(
    model_name: str,
    api_key: str = "EMPTY",
    base_url: Optional[str] = "http://localhost:8000/v1",
    credentials: Optional[str] = None,
) -> OpenAI:
    """Build the client for `model_name`.

    - OpenAI models (``gpt-*`` etc.) read ``OPENAI_API_KEY`` and hit
      ``api.openai.com``; ``api_key`` / ``base_url`` are ignored.
    - Gemini models (``gemini-*``) are served through Google's Gen AI client
      (``google-genai``), wrapped behind the OpenAI ``chat.completions.create``
      interface by ``GeminiClient``. A service-account JSON (``credentials`` / env
      ``GEMINI_CREDENTIALS`` / ``GOOGLE_APPLICATION_CREDENTIALS``) is used via
      **Vertex AI** (project + location from the SA / ``GEMINI_LOCATION``), or an
      API key (``--api_key`` / ``GEMINI_API_KEY``) hits the Generative Language API.
    - Everything else posts to ``base_url`` with ``api_key`` (e.g. a self-hosted
      vLLM server).
    """
    if is_openai_model(model_name):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                f"{model_name!r} is an OpenAI model but OPENAI_API_KEY is unset. "
                "Export it, or pass a self-hosted model name with --base_url."
            )
        return OpenAI(api_key=key)
    if is_gemini_model(model_name):
        from src.gemini_client import GeminiClient

        creds = credentials or os.getenv("GEMINI_CREDENTIALS") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        if creds:
            return GeminiClient(model_name, credentials_path=creds, base_url=GEMINI_BASE_URL)
        key = api_key if api_key not in ("EMPTY", None) else os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                f"{model_name!r} is a Gemini model but neither GEMINI_CREDENTIALS "
                "(service-account JSON) nor GEMINI_API_KEY / --api_key is set."
            )
        return GeminiClient(model_name, api_key=key, base_url=GEMINI_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def no_thinking_extra_body(client: OpenAI) -> Optional[dict]:
    """`extra_body` that turns a reasoning model's thinking block off, or None.

    `chat_template_kwargs` is a vLLM extension: api.openai.com *and* the Gemini
    OpenAI-compatible endpoint reject the unknown field outright, so it must
    only ever be sent to a self-hosted server (e.g. vLLM). We therefore return
    None for both OpenAI and Gemini clients and the JSON body only for others.
    """
    if getattr(client, "_is_gemini", False):
        return None
    base = str(getattr(client, "base_url", "") or "")
    if "openai.com" in base or "generativelanguage" in base:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}
