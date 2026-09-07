#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
PY="${PY:-python}"
MODEL="${MODEL:-gemini-3.6-flash}"
SPLIT="train"
MAX_TRAIN="${MAX_TRAIN:-10}"
MAX_TEST="${MAX_TEST:-10}"
MAX_DIALOGUES="${MAX_DIALOGUES:-20}"
MAX_WORKERS="${MAX_WORKERS:-8}"
export GEMINI_MIN_INTERVAL="${GEMINI_MIN_INTERVAL:-0.5}"
DATA_ROOT="${DATA_ROOT:-}"
INPUT_PATH="${INPUT_PATH:-}"
LOGDIR="${LOGDIR:-logs/run_$(date +%F_%H%M)}"
mkdir -p "$LOGDIR"

if [ $# -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=(${DATASETS:-interviewer soda multiwoz negotiator socraticlm persuader})
fi

if [[ -z "${GEMINI_CREDENTIALS:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: export GEMINI_CREDENTIALS or GEMINI_API_KEY first." >&2
  exit 1
fi
export GEMINI_LOCATION="${GEMINI_LOCATION:-global}"

EXTRA1=()
if [ -n "$DATA_ROOT" ]; then
  EXTRA1+=(--data-root "$DATA_ROOT")
fi
if [ -n "$INPUT_PATH" ]; then
  EXTRA1+=(--input_path "$INPUT_PATH")
fi

STATUS="$LOGDIR/STATUS.txt"
: > "$STATUS"

for ds in "${DATASETS[@]}"; do
  log="$LOGDIR/${ds}.log"
  echo "=== [$ds] start $(date -u +%FT%TZ) caps train=${MAX_TRAIN} dialogues=${MAX_DIALOGUES} workers=${MAX_WORKERS} pacing=${GEMINI_MIN_INTERVAL}s ===" | tee -a "$log" "$STATUS"

  s1="$PY -m src.speechify_run -d $ds --split $SPLIT --save_dir results_vi --llm_model_name $MODEL --max_train_samples $MAX_TRAIN --max_test_samples $MAX_TEST ${EXTRA1[*]:-}"
  if "$PY" -m src.speechify_run -d "$ds" --split "$SPLIT" --save_dir results_vi --llm_model_name "$MODEL" --max_train_samples "$MAX_TRAIN" --max_test_samples "$MAX_TEST" ${EXTRA1[@]+"${EXTRA1[@]}"} >>"$log" 2>&1; then
    echo "[$ds] stage1 OK" | tee -a "$STATUS"
  else
    echo "[$ds] stage1 FAIL (code $?) cmd: $s1 (see $log)" | tee -a "$STATUS"
    continue
  fi

  if "$PY" -m src.synthesis.run -d "$ds" -s "$SPLIT" --input_root results_vi --save_root outputs/vi_tt --llm_model_name "$MODEL" --boundary_model_name "$MODEL" --tt_model_name "$MODEL" --max_dialogues "$MAX_DIALOGUES" --max_workers "$MAX_WORKERS" >>"$log" 2>&1; then
    echo "[$ds] synthesis OK" | tee -a "$STATUS"
  else
    echo "[$ds] synthesis FAIL (see $log)" | tee -a "$STATUS"
    continue
  fi

  if "$PY" -m src.synthesis.run_add_bc --dataset "$ds" --split "$SPLIT" --input_root outputs/vi_tt --output_root outputs/vi_tt_bc --model_name "$MODEL" >>"$log" 2>&1; then
    echo "[$ds] add_bc OK" | tee -a "$STATUS"
  else
    echo "[$ds] add_bc FAIL (see $log)" | tee -a "$STATUS"
    continue
  fi

  echo "=== [$ds] done $(date -u +%FT%TZ) ===" | tee -a "$log" "$STATUS"
done

echo "--- summary ---"
cat "$STATUS"
find results_vi outputs/vi_tt_bc -type f 2>/dev/null | head -20
echo "counts: $(find outputs/vi_tt_bc -type f 2>/dev/null | wc -l | tr -d ' ') files in outputs/vi_tt_bc (expect 180 when all 6×30 done)"
