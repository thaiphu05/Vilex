#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
PY="${PY:-python}"                 # Stages 1-4 interpreter
STAGE5_PY="${STAGE5_PY:-$PY}"      # Stage 5 interpreter (use a separate env)
# Gợi ý: conda activate vilex-omnivoice  (hoặc vilex)
# export PY="/path/to/envs/vilex/bin/python"
# export STAGE5_PY="/path/to/envs/vilex-omnivoice/bin/python"

CONFIG="${VILEX_CONFIG:-$REPO/config.yaml}"

# All stage parameters live in ./config.yaml (edit that file, not this script).
# Only demand Gemini credentials when the config actually uses a gemini model;
# a served OpenAI-compatible model (llm.base_url + llm.model) needs none.
if grep -qE '^[[:space:]]*(model|writer_model|boundary_model|tt_model|bc_model):.*gemini' \
        "$CONFIG" 2>/dev/null; then
  if [[ -z "${GEMINI_CREDENTIALS:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: chưa set GEMINI_CREDENTIALS hoặc GEMINI_API_KEY. Vui lòng export trước khi chạy." >&2
    exit 1
  fi
  export GEMINI_LOCATION="${GEMINI_LOCATION:-global}"
fi

# Stage 1 — Speechify (spoken-style conversion)
"$PY" -m src.speechify_run

# Stage 4 — Synthesis (turn-taking + boundary)
"$PY" -m src.synthesis.run

# Stage 4b — Backchannel content
"$PY" -m src.synthesis.run_add_bc

# Stage 5 — OmniVoice TTS (2-channel + voice-clone pool)
"$STAGE5_PY" tts_render/convert_spoken.py
