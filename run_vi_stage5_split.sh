#!/usr/bin/env bash
# Stage 5 split pipeline (VI/OmniVoice) — runs the four stages in order.
#
#   5.1a prep   (CPU)          tts_render/stage5a_prep.py
#   5.1b render (GPU/OmniVoice) tts_render/stage5b_render.py
#   5.2a align  (GPU/Qwen3)    tts_render/stage5c_align.py
#   5.2b assemb (CPU)          tts_render/stage5d_assemble.py
#
# Each stage has its own interpreter so the GPU stages can run in the OmniVoice
# env and the CPU stages in the base env. Override per stage:
#   PREP_PY, RENDER_PY, ALIGN_PY, ASSEMBLE_PY, CONFIG
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

CONFIG="${CONFIG:-config.yaml}"
PREP_PY="${PREP_PY:-python}"
RENDER_PY="${RENDER_PY:-python}"
ALIGN_PY="${ALIGN_PY:-python}"
ASSEMBLE_PY="${ASSEMBLE_PY:-python}"

run() {
  local name="$1" py="$2" mod="$3"
  echo "=== ${name}: ${py} ${mod} --config ${CONFIG} ==="
  "$py" "$mod" --config "$CONFIG"
}

run "5.1a prep"    "$PREP_PY"     tts_render/stage5a_prep.py
run "5.1b render"  "$RENDER_PY"   tts_render/stage5b_render.py
run "5.2a align"   "$ALIGN_PY"    tts_render/stage5c_align.py
run "5.2b assemble" "$ASSEMBLE_PY" tts_render/stage5d_assemble.py

echo "Stage 5 split pipeline done."
