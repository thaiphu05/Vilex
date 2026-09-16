# Vilex: Vietnamese Turn-Taking Dialogue Synthesis

> **Vilex** is a Vietnamese adaptation of **DuplexGen: Adaptive Synthesis of Human–AI Turn-Taking Dialogues** (EMNLP 2026) — a 5-stage pipeline that synthesizes duplex (two-way) dialogues where the turn-taking behavior (floor-taking / backchannel / silence) is decided **word-by-word** within each utterance.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.26178-b31b1b.svg)](https://arxiv.org/abs/2607.26178)

By default, Vilex produces **Vietnamese** (Gemini + OmniVoice). English (Chatterbox) still runs as legacy with an explicit flag.

## Pipeline overview

| Stage | What it does | Docs |
|---|---|---|
| 1. Spoken-style conversion | Rewrite clean text dialogues into spoken-style transcripts. | [stage1-speechify](docs/stage1-speechify.md) |
| 2. Slot identification | Detect candidate intra-utterance action points via heuristic + LLM boundary detection. | [stage2-slots](docs/stage2-slots.md) |
| 3. Turn-taking prediction | Train or run a turn-taking predictor over the identified slots. | [stage3-prediction](docs/stage3-prediction.md) |
| 4. Turn-taking dialogue generation | Query the predictor at each slot and insert the chosen behavior, then filter role-confused outputs. | [stage4-generation](docs/stage4-generation.md) |
| 5. TTS rendering | Render to two-channel audio — **OmniVoice** (VI, default) or **Chatterbox** (EN legacy). | [stage5-tts](docs/stage5-tts.md) |

Also: [architecture](ARCHITECTURE.md) · [corpus & LLM calls](docs/CORPUS.md#5-llm-calls) · [troubleshooting](docs/TROUBLESHOOTING.md) · [data licenses](docs/DATA_LICENSES.md)

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
#   (OmniVoice is installed from source, see requirements-stage5.txt Block C)
conda create -n vilex-omnivoice python=3.11 -y
conda activate vilex-omnivoice
pip install torch==2.8.* --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-stage5.txt
# pip install git+https://github.com/k2-fsa/OmniVoice.git
```

**Run everything from the repository root.** Stages 1-4 are modules of the `src/` package, invoked as `.venv/bin/python -m src.<module>`. Stage 5 scripts are run as files with the *other* interpreter.

**System dependencies.** `nemo-text-processing` (English/Chatterbox only) depends on `pynini` (needs `OpenFst` headers on macOS/aarch64: `sudo apt-get install -y libfst-dev` or `conda-forge`). Vietnamese/OmniVoice does NOT need NeMo/pynini.

**Reproducing our exact environment.** `requirements.lock` carries the exact versions for Stages 1-4. Stage 5's versions come from the vendored `vilex/tts/chatterbox/pyproject.toml` plus the `numpy<2` bound.

```bash
.venv/bin/pip install -r requirements.txt -c requirements.lock
```

**Development tooling** (pytest, ruff, black — the checks CI gates on). Install on top of `requirements.txt` via the `dev` extra:

```bash
.venv/bin/pip install -e '.[dev]'
```

**LLM backend.** Every generation stage talks to a chat LLM through the OpenAI Python client, and the backend is chosen by the *model name*:

- **Gemini models** (`gemini-*`): Google GenAI OpenAI-compat, `GEMINI_API_KEY` or service-account JSON. **Default for Vilex.**
- **OpenAI models** (`gpt-*`, `o1`, `o3`, `o4`): `OPENAI_API_KEY`.
- **Everything else** (self-hosted `Qwen/...`, `DeepSeek-...`, any served model): posted to the OpenAI-compatible `llm.base_url` with `llm.api_key` (sent as `Authorization: Bearer`). Because routing keys off the name, a served model name must not contain `gemini` or start with `gpt-`/`o1|o3|o4`.

Set **one** served model for every role with three keys; per-role keys override it:
```yaml
llm:
  base_url: http://host:8000/v1
  api_key: <key>
  model: DeepSeek-V4-Flash   # used by writer / boundary / tt / bc
```

Stage 4 has three independent LLM roles: `llm.writer_model` (writer), `llm.boundary_model` (slot detection), `llm.tt_model` (turn-taking predictor); each falls back to `llm.model`. Or set `stage4_synthesis.hf_model_name_or_path` for the Stage 3 LoRA predictor.

Defaults are now Vietnamese: `run.target_language: vi`, `stage5_tts.backend: omnivoice`, `stage5_tts.language: vi`. For English set `VILEX_RUN__TARGET_LANGUAGE=en VILEX_STAGE5_TTS__BACKEND=chatterbox VILEX_STAGE5_TTS__LANGUAGE=en`.

## Quickstart (Vietnamese, default)

All parameters live in **`config.yaml`** at the repo root (models, paths, datasets,
timing, tags, ...). A committed sample is `config_example.yaml`; `config.yaml`
itself is gitignored, so copy it once:

```bash
cp config_example.yaml config.yaml
```

Each stage reads it and takes only an optional `--config PATH`. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

```bash
export GEMINI_API_KEY=...  # or GEMINI_CREDENTIALS=./project-name-*.json (Vertex AI)

# Stage 1 — convert source dialogues to spoken Vietnamese (default vi)
.venv/bin/python -m src.speechify_run

# Stage 2+4 — detect slots AND generate dialogues with turn-taking in one pass
# (Stage 2 slot detection runs inside the same synthesis entry point)
.venv/bin/python -m src.synthesis.run

# Stage 4 (cont.) — add backchannel text
.venv/bin/python -m src.synthesis.run_add_bc

# Stage 5 — render to two-channel audio with OmniVoice, picking 2 distinct
# voices from the voice pool per dialogue
python tts_render/convert_spoken.py

# Override any key without editing the file, e.g. run only the interviewer set:
VILEX_RUN__DATASETS='[interviewer]' .venv/bin/python -m src.synthesis.run
```

**Stage 3 — train the turn-taking predictor** (optional: the pipeline above falls back to the LLM predictor configured via `llm.tt_model`). Run this separately to train the HF LoRA predictor:

```bash
.venv/bin/python -m src.train_turntaking_hf \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_root data-annotations/ \
  --output_dir ./output/turntaking_qwen3_4b
```

Full pipeline in one command: `./run_vi_pipeline.sh` (see file header for options).

## English legacy (Chatterbox)

Edit `config.yaml` (or override via `VILEX_*`) to switch models/language, e.g.
`VILEX_RUN__TARGET_LANGUAGE=en VILEX_STAGE5_TTS__BACKEND=chatterbox`:

```bash
.venv/bin/python -m src.speechify_run
.venv/bin/python -m src.synthesis.run
.venv/bin/python -m src.synthesis.run_add_bc
.venv-tts/bin/python tts_render/convert_spoken.py
```

## Data

6 upstream source datasets downloaded locally in `data/raw/`. Pipeline reads `data-annotations/` and `data-dialogues/` (`text_dialogue_<dataset>/<split>/*.json`) produced by Stage 1 or `tools/unpack_corpus.py`. See `docs/CORPUS.md` for data flow, schemas, and counts.

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
