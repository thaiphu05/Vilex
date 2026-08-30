"""Verify the Gemini adapter can authenticate and call the model.

Usage:
    export GEMINI_CREDENTIALS=/abs/path/service-account.json
    python tools/verify_gemini.py --model_name gemini-3.5-flash

Or pass the path directly:
    python tools/verify_gemini.py --model_name gemini-3.5-flash \
        --credentials /abs/path/service-account.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import make_client  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gemini-3.5-flash")
    ap.add_argument("--credentials", default=None, help="Path to service-account JSON")
    ap.add_argument("--api_key", default="EMPTY")
    args = ap.parse_args()

    creds = args.credentials or os.getenv("GEMINI_CREDENTIALS") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if not creds:
        print("ERROR: set GEMINI_CREDENTIALS (or --credentials) to your service-account JSON")
        return 1

    print(f"[1] make_client({args.model_name!r}) with credentials={creds}")
    client = make_client(args.model_name, api_key=args.api_key, credentials=creds)
    print(f"    -> client type: {type(client).__name__}")

    print("[2] plain chat call")
    resp = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": "Reply in Vietnamese, briefly."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ],
        temperature=0.3,
    )
    print("    chat ->", repr((resp.choices[0].message.content or "")[:200]))

    print("[3] json_object call")
    resp = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return {"language":"vietnamese","ok":true}'},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    print("    json ->", repr(raw[:200]))
    try:
        obj = json.loads(raw)
        print("    parsed OK:", obj)
    except Exception as e:
        print("    JSON parse failed:", e)

    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
