#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
PY="${PY:-python}"
MODEL="${MODEL:-gemini-3.6-flash}"
DATASETS="${DATASETS:-interviewer multiwoz negotiator socraticlm persuader}"
LOGDIR="${LOGDIR:-data/logs/run_$(date +%F_%H%M)}"
mkdir -p "$LOGDIR"

if [[ -z "${GEMINI_CREDENTIALS:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: export GEMINI_CREDENTIALS or GEMINI_API_KEY first." >&2
  exit 1
fi
export GEMINI_LOCATION="${GEMINI_LOCATION:-global}"

# All stage parameters live in ./config.yaml. This script just runs Stages 1-4
# in order; each stage loops over the datasets/splits declared there.
# `DATASETS`/`MODEL` here only seed the log header and can override the YAML
# via VILEX_* env vars when needed.
export VILEX_RUN__DATASETS="[${DATASETS// /, }]"
export VILEX_LLM__WRITER_MODEL="${MODEL}"
export VILEX_LLM__BOUNDARY_MODEL="${MODEL}"
export VILEX_LLM__TT_MODEL="${MODEL}"

STATUS="$LOGDIR/STATUS.txt"
: > "$STATUS"

run() {
  local name="$1" cmd="$2"
  echo "=== ${name} start $(date -u +%FT%TZ) ===" | tee -a "$STATUS"
  if eval "$cmd" >>"$LOGDIR/${name}.log" 2>&1; then
    echo "[${name}] OK" | tee -a "$STATUS"
  else
    echo "[${name}] FAIL (see $LOGDIR/${name}.log)" | tee -a "$STATUS"
    return 1
  fi
}

run stage1 "$PY -m src.speechify_run"
run stage1_5 "$PY -m src.cross_turn_slots"
run stage1_75 "$PY -m src.disfluency"
run stage4 "$PY -m src.synthesis.run"
run stage4b "$PY -m src.synthesis.run_add_bc"

echo "--- summary ---"
cat "$STATUS"
echo "counts: $(find data/vi_tt_bc -type f 2>/dev/null | wc -l | tr -d ' ') files in data/vi_tt_bc"
