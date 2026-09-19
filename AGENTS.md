# AGENTS.md — Vilex

5-stage pipeline: Vietnamese turn-taking dialogue synthesis (DuplexGen adaptation).
Default path: Vietnamese (`--target_language vi`, Gemini, OmniVoice TTS).


## Two environments (never merge)

- **Stages 1–4:** `.venv` (Python 3.10+, `transformers>=4.53`)
- **Stage 5:** `.venv-tts` or conda `vilex-omnivoice` (Python 3.11, `transformers==4.46.3` from vendored Chatterbox)
- `requirements.txt` = Stages 1–4. `requirements-stage5.txt` = Stage 5. Dependency conflict prevents sharing.

## Stage order (data pipeline)

```
Stage 1  →  Stage 1.5  →  Stage 1.75  →  Stage 4  →  Stage 4b  →  Stage 5
speechify    cross_turn     disfluency     synthesis     run_add_bc    TTS
               _slots                                     (backchannel)
```

- `results_vi/` → `results_vi_xt/` → `results_vi_dis/` → `outputs/vi_tt/` → `outputs/vi_tt_bc/` → audio
- Stage 4 reads `results_vi_dis` (not `results_vi`). Feed wrong dir → silent failure.

Stage 5 has a split variant (VI/OmniVoice, batch-capable, 4 tiers) beside the
monolithic `tts_render/convert_spoken.py`:

```
5.1a prep (CPU) → 5.1b render (GPU, OmniVoice) → 5.2a align (GPU, Qwen3) → 5.2b assemble (CPU)
stage5a_prep.py    stage5b_render.py              stage5c_align.py          stage5d_assemble.py
```

Intermediates live under `paths.stage5_work_root`; final output matches the
monolithic schema. Glue: `./run_vi_stage5_split.sh`.

## Running stages

Stages 1–4: `python -m src.<module>` from repo root, `.venv/bin/python`.
Stage 5: `python tts_render/convert_spoken.py` (file path, **different interpreter**).

```bash
# Stage 1 — spoken-style conversion
python -m src.speechify_run -d interviewer --save_dir results_vi --llm_model_name gemini-3.6-flash

# Stage 1.5 — cross-turn slots (rule-based, 0 API)
python -m src.cross_turn_slots --input_root results_vi --output_root results_vi_xt \
  --split train --dataset interviewer --perror 0.20 --seed 42 --target_language vi --roles both

# Stage 1.75 — disfluency injection (rule-based, 0 API)
python -m src.disfluency --input_root results_vi_xt --output_root results_vi_dis \
  --split train --dataset interviewer --seed 42 --target_language vi

# Stage 4 — dialogue generation + backchannel
python -m src.synthesis.run -d interviewer -s train --input_root results_vi_dis \
  --save_root outputs/vi_tt --llm_model_name gemini-3.6-flash \
  --boundary_model_name gemini-3.6-flash --tt_model_name gemini-3.6-flash
python -m src.synthesis.run_add_bc --input_root outputs/vi_tt \
  --output_root outputs/vi_tt_bc --model_name gemini-3.6-flash

# Stage 5 — TTS (different interpreter)
python tts_render/convert_spoken.py \
  --input_glob 'outputs/vi_tt_bc/text_dialogue_interviewer/train/*.json' \
  --save_dir outputs/audios --omnivoice_voice_pool voice_clone --num_variants 1 --device cpu
```

Full batch scripts: `./run_vi_pipeline.sh` (1 dialogue), `./run_local_stages1-4.sh` (5 datasets).

## LLM routing (`src/llm_client.py`)

Model name prefix determines backend. Stage 4 has 3 independent LLM roles:
- `--llm_model_name` (writer), `--boundary_model_name` (slot detection), `--tt_model_name` (turn-taking predictor)
- `gemini-*` → `GEMINI_API_KEY` or `GEMINI_CREDENTIALS` (Vertex AI). **Default.**
- `gpt-*`, `o1`, `o3`, `o4` → `OPENAI_API_KEY`
- `Qwen/...` → self-hosted at `--base_url` (default `localhost:8000`)

## Testing

```bash
pytest -q          # 80 tests expected, lives in tools/ and src/
ruff --select=E9,F # lint (CI gate)
black --check      # format (CI gate)
py_compile         # on entry points for smoke check
```

Tests in `tools/test_*.py` and `src/test_*.py`. Stage 1.5/1.75 tests run 0 API calls.

## Gotchas

- **English = legacy.** Always pass `--target_language en --tts_backend chatterbox --language en` explicitly.
- **Stage 5 picks voices deterministically** via `seed + crc32(filename)`. Changing seed → different voices.
- **`voice_clone/`** needs ≥2 `.wav`+`.txt` pairs or Stage 5 `SystemExit`.
- **Silero VAD** trims silence per sentence. Short tokens (`"Ừm,"`) may produce empty audio → silent 0.2s guard.
- **Gemini rate limit:** `_MIN_INTERVAL=13s`, 429 backoff in `src/gemini_client.py`. Free-tier ≈20 req/day/model.
- **Never hardcode `flash_attention_2`.** Use `src/hf_attn.py` (default `auto`: FA2 if importable, else SDPA).
- **Stage 2 has no standalone command.** Slot detection runs inside Stage 4 (`src/synthesis/core.detect_turn_boundaries`).
- **Scenario codes are case-sensitive.** `SOC` = soda, `soc` = socraticlm. Always route via `CODE2DIR` in `tools/unpack_corpus.py`.

## Key files

| Area | Files |
|---|---|
| Entry points | `src/speechify_run.py`, `src/cross_turn_slots.py`, `src/disfluency.py`, `src/synthesis/run.py`, `src/synthesis/run_add_bc.py`, `tts_render/convert_spoken.py` |
| Core logic | `src/synthesis/core.py` (Stage 2+4), `src/llm_client.py`, `src/gemini_client.py` |
| Training | `src/train_turntaking_hf.py`, `src/inference_turntaking_hf.py` |
| Vendored | `vilex/tts/chatterbox/` (MIT, own `pyproject.toml`) |
| Config | `pyproject.toml` (line-length 100, black+ruff exclude `vilex/tts/chatterbox`) |
