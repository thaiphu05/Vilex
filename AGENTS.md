# AGENTS.md — Vilex

5-stage pipeline: Vietnamese turn-taking dialogue synthesis (DuplexGen adaptation).
Default path: Vietnamese (`run.target_language: vi`, Gemini, OmniVoice TTS).
All stage parameters live in `config.yaml` (sample: `config_example.yaml`); each
entry point takes only an optional `--config PATH` (env override: `VILEX_*`).


## Two environments (never merge)

- **Stages 1–4:** `.venv` (Python 3.10+, `transformers>=4.53`)
- **Stage 5:** `.venv-tts` or conda `vilex-omnivoice` (Python 3.11, `transformers==4.46.3` from vendored Chatterbox)
- `requirements.txt` = Stages 1–4. `requirements-stage5.txt` = Stage 5. Dependency conflict prevents sharing.

## Stage order (data pipeline)

```
Stage 1  →  Stage 4  →  Stage 4b  →  Stage 5
speechify    synthesis    run_add_bc    TTS
                           (backchannel)
```

- `results_root (data/results_vi)` → `synthesis_root (data/vi_tt)` → `bc_root (data/vi_tt_bc)` → audio
- Stage 4 reads `paths.results_root` (Stage 1 output).

Stage 5 has a split variant (VI/OmniVoice, batch-capable, 4 tiers) beside the
monolithic `tts_render/convert_spoken.py`:

```
5.1a prep (CPU) → 5.1b render (GPU, OmniVoice) → 5.2a align (GPU, whisperx) → 5.2b assemble (CPU)
stage5a_prep.py    stage5b_render.py              stage5c_align.py             stage5d_assemble.py
```

Intermediates live under `paths.stage5_work_root`; final output matches the
monolithic schema. Glue: `./run_vi_stage5_split.sh`.

## Running stages

Stages 1–4: `python -m src.<module>` from repo root, `.venv/bin/python`.
Stage 5: `python tts_render/convert_spoken.py` (file path, **different interpreter**).

```bash
# Stage 1 — spoken-style conversion
python -m src.speechify_run

# Stage 4 — dialogue generation + backchannel
python -m src.synthesis.run
python -m src.synthesis.run_add_bc

# Stage 5 — TTS (different interpreter)
python tts_render/convert_spoken.py
```

Full batch scripts: `./run_vi_pipeline.sh` (1 dialogue), `./run_local_stages1-4.sh` (5 datasets),
`./run_vi_stage5_split.sh` (Stage 5 split).

## LLM routing (`src/llm_client.py`)

Model name prefix determines backend. Stage 4 has 3 independent LLM roles:
- `llm.writer_model` (writer), `llm.boundary_model` (slot detection), `llm.tt_model` (turn-taking predictor); each falls back to `llm.model`
- `gemini-*` → `GEMINI_API_KEY` or `GEMINI_CREDENTIALS` (Vertex AI). **Default.**
- `gpt-*`, `o1`, `o3`, `o4` → `OPENAI_API_KEY`
- `claude*`/`*anthropic*` (or `llm.anthropic_mode`) → Anthropic Messages shim (`src/anthropic_client.py`)
- anything else → self-hosted OpenAI-compatible `llm.base_url` (default `localhost:8000`)

## Testing

```bash
pytest -q          # 190 tests expected, lives in tests/
ruff --select=E9,F # lint (CI gate)
black --check      # format (CI gate)
py_compile         # on entry points for smoke check
```

Tests in `tests/test_*.py`.

## Gotchas

- **English = legacy.** Set `run.target_language: en`, `stage5_1b_render.backend: chatterbox`, `stage5.language: en`.
- **Stage 5 picks voices deterministically** via `seed + crc32(filename)`. Changing seed → different voices.
- **`voice_clone/`** needs ≥2 `.wav`+`.txt` pairs or Stage 5 `SystemExit`.
- **Silero VAD** trims silence per sentence. Short tokens (`"Ừm,"`) may produce empty audio → silent 0.2s guard.
- **Gemini rate limit:** `llm.gemini_min_interval` (env `GEMINI_MIN_INTERVAL`), 429 backoff in `src/gemini_client.py`.
- **Never hardcode `flash_attention_2`.** Use `src/hf_attn.py` (default `auto`: FA2 if importable, else SDPA).
- **Stage 2 has no standalone command.** Slot detection runs inside Stage 4 (`src/synthesis/core.detect_turn_boundaries`).
- **Scenario codes are case-sensitive.** `SOC` = soda, `soc` = socraticlm. Always route via `CODE2DIR` in `tools/unpack_corpus.py`.
- **Stage 5 forced alignment uses whisperx** (`_load_forced_aligner` / `_align_words_once` in `tts_render/convert_spoken.py`); `stage5c_align.py` reuses those helpers.

## Key files

| Area | Files |
|---|---|
| Entry points | `src/speechify_run.py`, `src/synthesis/run.py`, `src/synthesis/run_add_bc.py`, `tts_render/convert_spoken.py` |
| Core logic | `src/synthesis/core.py` (Stage 2+4), `src/llm_client.py`, `src/gemini_client.py` |
| Training | `src/train_turntaking_hf.py`, `src/inference_turntaking_hf.py` |
| Vendored | `vilex/tts/chatterbox/` (MIT, own `pyproject.toml`) |
| Config | `src/config.py` + `config_example.yaml`; `pyproject.toml` (line-length 100, black+ruff exclude `vilex/tts/chatterbox`) |
