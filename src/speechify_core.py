# speechify_core.py
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from openai import OpenAI
from src.speechify_prompts import (
    SINGLE_STEP_CONVERSION_PROMPT,
    SINGLE_STEP_CONVERSION_PROMPT_CONCISE,
    build_single_step_prompt,
)
from pydantic import BaseModel


class DialogueTurn(BaseModel):
    role: str
    content: str


class DialogueResponse(BaseModel):
    turns: List[DialogueTurn]


CONFIG = {}

# When the target language is not English, we append this directive to the
# (English) system prompt. Prompts stay in English; only the *output* text is
# required to be the target language.
LANG_DIRECTIVE = {
    "vi": (
        "\n\nIMPORTANT: The source text may be in English. You MUST translate it and "
        "produce ALL spoken output text in Vietnamese (tiếng Việt). Keep the JSON "
        "`role`/`content` structure and any [TAKE_FLOOR] tokens unchanged."
    ),
}


def _generate_dialogue_structured(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    """Structured-output call that works for OpenAI, Gemini, and vLLM.

    OpenAI (api.openai.com) uses the beta `parse` API with a Pydantic schema.
    Gemini and self-hosted vLLM speak the OpenAI-compatible protocol but do not
    support `beta.parse`, so we request `response_format={"type":"json_object"}`
    and parse the JSON ourselves.
    """
    base = str(getattr(client, "base_url", "") or "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if "openai.com" in base:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format=DialogueResponse,
        )
        parsed_response = completion.choices[0].message.parsed
        if parsed_response:
            return [turn.model_dump() for turn in parsed_response.turns]
        return []

    # Gemini / vLLM: JSON object mode.
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except Exception:
        return []
    turns = data.get("turns", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out = []
    for t in turns:
        if isinstance(t, dict) and "role" in t and "content" in t:
            out.append({"role": t["role"], "content": t["content"]})
    return out


def speechify_full_dialogue(
    llm_model_name: str,
    client: OpenAI,
    source_turns: List[Tuple[str, str]],
    context: str,
    temperature: float = 0.7,
    concise: bool = False,
    target_language: str = "en",
) -> List[Dict[str, Any]]:
    if concise:
        # Halve the source before conversion; the concise prompt shortens what
        # remains, so feeding the full dialogue would fight against it.
        source_turns = source_turns[: len(source_turns) // 2]

    user_prompt = build_single_step_prompt(source_turns, context)

    system_prompt = (
        SINGLE_STEP_CONVERSION_PROMPT_CONCISE if concise else SINGLE_STEP_CONVERSION_PROMPT
    )
    directive = LANG_DIRECTIVE.get(target_language, "")
    if directive:
        system_prompt = system_prompt + directive

    turns = _generate_dialogue_structured(
        client, llm_model_name, system_prompt, user_prompt, temperature
    )
    # Some models (e.g. Gemini) may echo the instruction back as a `system`
    # turn inside the structured `turns` list. Only user/assistant turns are
    # valid dialogue; drop anything else so downstream stages don't choke.
    return [t for t in turns if isinstance(t, dict) and t.get("role") in ("user", "assistant")]
