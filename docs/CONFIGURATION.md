# Configuration (`config.yaml`)

Every tunable parameter for Stages 1-5 lives in a single **`config.yaml`** at the
repo root. A committed sample lives at **`config_example.yaml`**; `config.yaml` is
gitignored so your local settings stay out of the repo:

```bash
cp config_example.yaml config.yaml
```

The stage scripts no longer take flags; each entry point reads the config and
exposes only an optional `--config PATH` to point at a different file. If
`config.yaml` is missing, the loader falls back to in-code defaults:

```bash
python -m src.speechify_run                 # Stage 1
python -m src.cross_turn_slots              # Stage 1.5
python -m src.disfluency                    # Stage 1.75
python -m src.synthesis.run                 # Stage 4
python -m src.synthesis.run_add_bc          # Stage 4b
python tts_render/convert_spoken.py         # Stage 5

# or a custom file:
python -m src.synthesis.run --config /path/to/other.yaml
```

## Loader & precedence

`src/config.py` loads the config and merges, highest priority first:

1. **Environment overrides** — `VILEX_<PATH>` where `<PATH>` is the dotted key
   with `__` between levels. Values are parsed as YAML/JSON/bool/number.
   ```bash
   VILEX_LLM__TEMPERATURE=0.3 \
   VILEX_RUN__DATASETS='[interviewer, soda]' \
   VILEX_STAGE5_TTS__TAGS__RENDER=true \
     python -m src.synthesis.run
   ```
2. **`config.yaml`** — chosen by: `--config PATH` → `VILEX_CONFIG` env →
   `./config.yaml` → `<repo>/config.yaml`.
3. **In-code defaults** — each module keeps its constants as a fallback, so a
   missing key never crashes a run.

Secrets stay in the environment and are **never** read from the YAML:
`GEMINI_CREDENTIALS`, `GEMINI_API_KEY`, `OPENAI_API_KEY`.

## Sections

| Section | Controls |
|---|---|
| `run` | `seed`, `datasets`, `splits`, `target_language`, `dry_run`, `test_parse` |
| `paths` | all input/output roots (`results_root`, `results_xt_root`, `results_dis_root`, `synthesis_root`, `bc_root`, `audio_root`, `data_root`, `input_path`, `voice_clone_pool`, `logdir`) |
| `llm` | model names per role (writer / boundary / tt / bc), `base_url`s, `api_key`, `temperature`, `gemini_min_interval`, `gemini_location` |
| `stage1_speechify` | sample budgets, `max_variants`, `interviewer_subset`, `temperature`, `concise` |
| `stage1_5_cross_turn` | `perror`, `seed`, `roles`, `min_digits`, `min_code_len` |
| `stage1_75_disfluency` | `scales`, `shriberg_b`, `types`, `rep_span`, `inventories` (FP/DM/EDIT per language) |
| `stage4_synthesis` | `max_turns`, `max_dialogues`, `max_workers`, temperatures, `guards`, `ft_terminal_punct`, `hesitations`, `judge`, `hf` |
| `stage4b_backchannel` | `max_tokens`, `temperature`, `max_retries`, `valid_max_words`, fallback pools |
| `stage5_tts` | backend/language/device, `target_sr`/`prompt_sr`, `timing` (gap/pause/intra-pause/interrupt), `audio` (LUFS, VAD, noise floor), `voice` (instructs, pool), `tags` (13 supported + `render`), `backchannels` (candidates, rising tokens) |

Each stage loops over `run.datasets` × `run.splits` internally, so a single
invocation processes every dataset/split declared there.

## Notes

- **Prompts** stay in `src/synthesis/prompts.py` / `src/speechify_prompts.py`
  (large text blocks, not config).
- **Regexes and structural tokens** (`[TAKE_FLOOR]`, `[BACKCHANNEL]`,
  `[PAUSE]`) stay in code.
- `tools/test_config.py` covers the merge/precedence behaviour; see
  `config.yaml` for every key with its default value.
