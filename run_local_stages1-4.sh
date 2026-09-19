#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

# Stages 1-4 interpreter; Stage 5 (optional) uses its own env.
PY="${PY:-python}"
STAGE5_PY="${STAGE5_PY:-}"                 # empty -> skip Stage 5
MODEL="${MODEL:-}"                         # empty -> use llm.model from config
DATASETS="${DATASETS:-}"                   # empty -> use run.datasets from config

CONFIG="${VILEX_CONFIG:-$REPO/config.yaml}"

# Only demand Gemini credentials when the config actually uses a gemini model.
# A served OpenAI-compatible model (llm.base_url + llm.model) needs none.
if grep -qE '^[[:space:]]*(model|writer_model|boundary_model|tt_model|bc_model):.*gemini' \
        "$CONFIG" 2>/dev/null; then
  if [[ -z "${GEMINI_CREDENTIALS:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: config uses a gemini model but GEMINI_CREDENTIALS/GEMINI_API_KEY is unset." >&2
    exit 1
  fi
  export GEMINI_LOCATION="${GEMINI_LOCATION:-global}"
else
  echo "NOTE: non-gemini models -> using llm.base_url/llm.api_key from config (no GEMINI_* needed)."
fi

# Explicit env overrides only; otherwise config.yaml wins.
if [[ -n "$DATASETS" ]]; then
  export VILEX_RUN__DATASETS="[${DATASETS// /, }]"
fi
if [[ -n "$MODEL" ]]; then
  export VILEX_LLM__WRITER_MODEL="$MODEL"
  export VILEX_LLM__BOUNDARY_MODEL="$MODEL"
  export VILEX_LLM__TT_MODEL="$MODEL"
  export VILEX_LLM__BC_MODEL="$MODEL"
fi

# Every stage logs straight to the terminal (no per-stage log files).
run() {
  local name="$1" cmd="$2"
  echo "=== ${name} start $(date -u +%FT%TZ) ==="
  if eval "$cmd"; then
    echo "[${name}] OK"
  else
    echo "[${name}] FAIL" >&2
    return 1
  fi
}

run stage1 "$PY -m src.speechify_run"
run stage4 "$PY -m src.synthesis.run"
run stage4b "$PY -m src.synthesis.run_add_bc"

# Stage 5 needs a separate environment (torch 2.8 + transformers + OmniVoice/aligner).
if [[ -n "$STAGE5_PY" ]]; then
  run stage5 "$STAGE5_PY tts_render/convert_spoken.py"
fi

echo "counts: $(find data/vi_tt_bc -type f 2>/dev/null | wc -l | tr -d ' ') files in data/vi_tt_bc"
