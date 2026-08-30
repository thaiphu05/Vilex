#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
PY="${PY:-python}"
# Gợi ý: conda activate vilex-omnivoice  (hoặc vilex)
# export PY="/path/to/envs/vilex/bin/python"


MAX_TRAIN_SAMPLES=2      # Stage 1: số dialogue nguồn xử lý (vd 2). Trống = toàn bộ.
MAX_DIALOGUES=2          # Stage 4 & Stage 5: số dialogue xử lý. Trống = toàn bộ.
MAX_TURNS=10             # Stage 4: số lượt tối đa / dialogue (mặc định 20 nếu trống).
NUM_VARIANTS=3            # Stage 5: số bản audio / dialogue.

#   export GEMINI_CREDENTIALS="$REPO/NAME.json"   # Vertex AI
#   # hoặc:
#   export GEMINI_API_KEY="PASTE_YOUR_KEY"                                            # Generative Language API
if [[ -z "${GEMINI_CREDENTIALS:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: chưa set GEMINI_CREDENTIALS hoặc GEMINI_API_KEY. Vui lòng export trước khi chạy." >&2
  exit 1
fi
export GEMINI_LOCATION="${GEMINI_LOCATION:-global}"

# Stage 1 — Speechify (tạo dialogue gốc tiếng Việt, default vi)
"$PY" -m src.speechify_run -d interviewer --split train --save_dir results_vi \
  --llm_model_name gemini-3.6-flash --max_train_samples 1

# Stage 4 — Synthesis (turn-taking + boundary, default vi)
"$PY" -m src.synthesis.run -d interviewer -s train \
  --input_root results_vi --save_root outputs/vi_tt \
  --llm_model_name gemini-3.6-flash \
  --boundary_model_name gemini-3.6-flash \
  --tt_model_name gemini-3.6-flash \
  --max_dialogues 1 --max_turns 10

# Stage 4 (tiếp) — Backchannel (default vi)
"$PY" -m src.synthesis.run_add_bc --dataset interviewer --split train \
  --input_root outputs/vi_tt --output_root outputs/vi_tt_bc \
  --model_name gemini-3.6-flash

# Stage 5 — OmniVoice TTS (default vi + omnivoice, 2 kênh + clone từ voice pool)
# Mỗi dialogue pick ngẫu nhiên 2 giọng distinct từ thư mục voice_clone (user + assistant).
# Yêu cầu: mỗi file <tên>.wav phải có <tên>.txt cùng thư mục (transcript chính xác).
"$PY" tts_render/convert_spoken.py \
  --input_glob "outputs/vi_tt_bc/text_dialogue_interviewer/train/*.json" \
  --save_dir outputs/vi_audio \
  --omnivoice_voice_pool "voice_clone" \
  --num_variants 1 --device cpu
