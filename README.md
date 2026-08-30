# Vilex: Vietnamese Turn-Taking Dialogue Synthesis

> **Vilex** là bản Việt hóa của **DuplexGen: Adaptive Synthesis of Human–AI Turn-Taking Dialogues** (EMNLP 2026) — pipeline 5 giai đoạn sinh hội thoại song song (duplex) với hành vi lượt nói (floor-taking / backchannel / silence) quyết định **từng từ** trong lượt nói.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.26178-b31b1b.svg)](https://arxiv.org/abs/2607.26178)

Mặc định Vilex sinh **tiếng Việt** (Gemini + OmniVoice). Tiếng Anh (Chatterbox) vẫn chạy như legacy với flag explicit.

## Pipeline overview

| Stage | What it does | Docs |
|---|---|---|
| 1. Spoken-style conversion | Rewrite clean text dialogues into spoken-style transcripts. | [stage1-speechify](docs/stage1-speechify.md) |
| 2. Slot identification | Detect candidate intra-utterance action points via heuristic + LLM boundary detection. | [stage2-slots](docs/stage2-slots.md) |
| 3. Turn-taking prediction | Train or run a turn-taking predictor over the identified slots. | [stage3-prediction](docs/stage3-prediction.md) |
| 4. Turn-taking dialogue generation | Query the predictor at each slot and insert the chosen behavior, then filter role-confused outputs. | [stage4-generation](docs/stage4-generation.md) |
| 5. TTS rendering | Render to two-channel audio — **OmniVoice** (VI, default) or **Chatterbox** (EN legacy). | [stage5-tts](docs/stage5-tts.md) |

Also: [troubleshooting](docs/TROUBLESHOOTING.md) · [data licenses](docs/DATA_LICENSES.md)

## Setup

This project needs **two separate environments**. Stage 5 pins `transformers==4.46.3` through the vendored Chatterbox package, while Stage 3 requires `transformers>=4.53`; they cannot share an interpreter.

```bash
# Stages 1-4: dialogue generation, slot identification, prediction, synthesis
python3 -m venv .venv                       # Python 3.10+
.venv/bin/pip install -r requirements.txt

# Stage 5 English (Chatterbox, legacy) — Python 3.11 ONLY
python3.11 -m venv .venv-tts
.venv-tts/bin/pip install -r requirements-stage5.txt
.venv-tts/bin/pip install -e vilex/tts/chatterbox

# Stage 5 Vietnamese (OmniVoice, default) — separate env
conda create -n vilex-omnivoice python=3.11 -y
conda activate vilex-omnivoice
pip install torch==2.8.* --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-stage5-vi.txt
# pip install git+https://github.com/k2-fsa/OmniVoice.git
```

**Run everything from the repository root.** Stages 1-4 are modules of the `src/` package, invoked as `.venv/bin/python -m src.<module>`. Stage 5 scripts are run as files with the *other* interpreter.

**System dependencies.** `nemo-text-processing` (English/Chatterbox only) depends on `pynini` (needs `OpenFst` headers on macOS/aarch64: `sudo apt-get install -y libfst-dev` or `conda-forge`). Vietnamese/OmniVoice does NOT need NeMo/pynini.

**Reproducing our exact environment.** `constraints.txt` carries the exact versions for Stages 1-4. Stage 5's versions come from the vendored `vilex/tts/chatterbox/pyproject.toml` plus the `numpy<2` bound.

```bash
.venv/bin/pip install -r requirements.txt -c constraints.txt
```

**LLM backend.** Every generation stage talks to a chat LLM through the OpenAI Python client, and the backend is chosen by the *model name*:

- **Gemini models** (`gemini-*`): Google GenAI OpenAI-compat, `GEMINI_API_KEY` or service-account JSON. **Default for Vilex.**
- **OpenAI models** (`gpt-*`, `o1`, `o3`, `o4`): `OPENAI_API_KEY`.
- **Open-weight models** (`Qwen/...`): self-hosted at `--base_url` (default `localhost:8000`).

Stage 4 has three independent LLM roles: `--llm_model_name` (writer), `--boundary_model_name` (slot detection), `--tt_model_name` (turn-taking predictor). Or use `--hf_model_name_or_path` for the Stage 3 LoRA predictor.

Defaults are now Vietnamese: `--target_language vi`, `--tts_backend omnivoice`, `--language vi`. For English use `--target_language en --tts_backend chatterbox --language en`.

## Quickstart (Vietnamese, default)

```bash
export GEMINI_API_KEY=...  # or GEMINI_CREDENTIALS=./gen-lang-client-*.json (Vertex AI)

# Stage 1 — convert source dialogues to spoken Vietnamese (default vi, no flag needed)
.venv/bin/python -m src.speechify_run \
  --dataset interviewer --save_dir results_vi/ --llm_model_name gemini-3.5-flash

# Stage 3 — train the turn-taking predictor (optional)
.venv/bin/python -m src.train_turntaking_hf \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_root data-annotations/ \
  --output_dir ./output/turntaking_qwen3_4b

# Stages 2+4 — detect slots and generate dialogues with turn-taking (default vi)
.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/vi_tt \
  --llm_model_name gemini-3.5-flash --boundary_model_name gemini-3.5-flash \
  --tt_model_name gemini-3.5-flash

# Add backchannel text (default vi)
.venv/bin/python -m src.synthesis.run_add_bc \
  --input_root outputs/vi_tt --output_root outputs/vi_tt_bc \
  --model_name gemini-3.5-flash

# Stage 5 — render to two-channel audio with OmniVoice (default vi+omnivoice)
python tts_render/convert_spoken.py \
  --input_glob 'outputs/vi_tt_bc/**/*.json' \
  --save_dir outputs/audios_omnivoice --num_variants 1 --device cpu
```

Full pipeline in one command: `./run_vi_pipeline.sh` (see file header for options).

## English legacy (Chatterbox)

```bash
.venv/bin/python -m src.speechify_run --dataset interviewer --save_dir results/ --llm_model_name gpt-4.1 --target_language en
.venv/bin/python -m src.synthesis.run --dataset interviewer --split train --input_root data-dialogues/ --save_root outputs/en_tt --llm_model_name gpt-4.1 --boundary_model_name gpt-4.1-mini --tt_model_name gpt-4.1 --target_language en
.venv/bin/python -m src.synthesis.run_add_bc --input_root outputs/en_tt --output_root outputs/en_tt_bc --model_name gpt-4.1 --target_language en
.venv-tts/bin/python tts_render/convert_spoken.py --tts_backend chatterbox --language en --input_glob 'outputs/en_tt_bc/**/*.json' --save_dir outputs/audios_chatterbox --num_variants 10
```

## Data

HF corpus is temporarily removed. Pipeline reads local `data-annotations/` and `data-dialogues/` (`text_dialogue_<dataset>/<split>/*.json`) produced by Stage 1 or `tools/unpack_corpus.py`. See `DATA_CARD.md` for schema.

## License

Code: **Apache License 2.0** (see `LICENSE`, Copyright 2026 The Vilex Authors (based on DuplexGen)). Data licenses per scenario in `docs/DATA_LICENSES.md`. Audio rendered with [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT, vendored at `vilex/tts/chatterbox/`) and [OmniVoice](https://github.com/k2-fsa/OmniVoice).

## Citation

Based on DuplexGen (EMNLP 2026). If you use this code, please cite the original paper and note Vilex as the Vietnamese adaptation:

```bibtex
@article{kim2026duplexgen,
  title   = {{DuplexGen: Adaptive Synthesis of Human--AI Turn-Taking Dialogues}},
  author  = {Kim, Takyoung and Kim, Kang-wook and Woo, Sang Hoon and
             Hirschberg, Julia and Kim, Gunhee and Hakkani-T{\"u}r, Dilek},
  journal = {arXiv preprint arXiv:2607.26178},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.26178}
}
% Vilex: Vietnamese adaptation of DuplexGen (defaults vi + OmniVoice).
```
