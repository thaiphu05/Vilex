"""Translate existing English Stage-4 dialogues into Vietnamese.

This is an optional utility: if you already have English dialogues produced by
``src.synthesis.run`` (or the released corpus unpacked with
``tools/unpack_corpus.py``) and do not want to re-run Stages 1-4, this script
translates them to Vietnamese while preserving the turn-taking structure:

* The ``[TAKE_FLOOR]`` token is kept in place and its ``word_index`` in the slot
  metadata is recomputed from the translated text.
* ``backchannel`` slot ``content`` is translated to Vietnamese (ưm, à, ừ...).
* Plain ``full_content`` segments are derived from the translated turn text.

Usage (Gemini):

    python tools/translate_dialogues.py \
        --input_root outputs/generated_dialogues_with_tt \
        --output_root outputs/vi_tt \
        --model_name gemini-3.5-flash \
        --base_url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --api_key $GEMINI_API_KEY
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import make_client, no_thinking_extra_body  # noqa: E402

TOKEN_FT = "[TAKE_FLOOR]"

TRANSLATE_PROMPT = (
    "Translate the following spoken dialogue turn into natural Vietnamese (tiếng Việt). "
    "Keep the token [TAKE_FLOOR] exactly where it appears (do not move, rename, or remove it). "
    "Preserve the casual, spoken style. Output ONLY the translated text, with no quotes, "
    "labels, or explanation."
)

BC_TRANSLATE_PROMPT = (
    "Translate the following short listener backchannel into Vietnamese (tiếng Việt). "
    "Output ONLY the Vietnamese backchannel word(s) (e.g. ưm, à, ừ, vâng, phải, thật không), "
    "no quotes or explanation."
)


def _translate(client, model_name, text, instruction):
    if not text.strip():
        return text
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        extra_body=no_thinking_extra_body(client),
    )
    out = (resp.choices[0].message.content or "").strip()
    # strip stray quotes / markdown fences
    out = out.strip("`").strip()
    if out.startswith('"') and out.endswith('"'):
        out = out[1:-1]
    return out


def _take_floor_positions(text):
    """Word indices of each [TAKE_FLOOR] occurrence in `text`."""
    positions = []
    for m in re.finditer(re.escape(TOKEN_FT), text):
        prefix = text[: m.start()]
        idx = len(prefix.split()) - 1
        positions.append(max(0, idx))
    return positions


def translate_turn(turn, client, model_name):
    content = turn.get("content", "")
    if not isinstance(content, str):
        return turn

    vi_content = _translate(client, model_name, content, TRANSLATE_PROMPT)
    vi_full = re.sub(re.escape(TOKEN_FT), "", vi_content).strip()
    ft_positions = _take_floor_positions(vi_content)

    history = turn.get("history", [])
    ft_cursor = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        if "word_index" not in item:
            # plain full_content segment -> update with translated text
            if "full_content" in item:
                item["full_content"] = vi_full
            continue
        decision = item.get("decision")
        if decision == "floor_taking" or item.get("inserted_token") == TOKEN_FT:
            if ft_cursor < len(ft_positions):
                item["word_index"] = ft_positions[ft_cursor]
                ft_cursor += 1
        elif decision == "backchannel":
            bc = item.get("content")
            if isinstance(bc, str) and bc.strip():
                item["content"] = _translate(client, model_name, bc, BC_TRANSLATE_PROMPT)

    turn["content"] = vi_content
    return turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--model_name", default="gemini-3.5-flash")
    ap.add_argument("--base_url", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--max_dialogues", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    client = make_client(args.model_name, args.api_key, args.base_url)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    in_files = sorted(input_root.glob("text_dialogue_*/*/*.json"))
    if args.max_dialogues:
        in_files = in_files[: args.max_dialogues]

    print(f"Translating {len(in_files)} dialogues EN -> VI")
    for f in in_files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for turn in data.get("history", []):
            translate_turn(turn, client, args.model_name)
        data.setdefault("meta", {})
        data["meta"]["language"] = "vi"

        rel = f.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
