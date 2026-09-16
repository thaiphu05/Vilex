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

### Gemini runtime knobs

Two LLM settings are read by the Gemini client from the **environment**, not the
config dict, so `apply_runtime_config()` bridges them from the YAML before any
client is built (same precedence: an explicit env var still wins):

| YAML key | Environment variable | Default |
|---|---|---|
| `llm.gemini_min_interval` | `GEMINI_MIN_INTERVAL` | `1.0` (seconds between calls) |
| `llm.gemini_location` | `GEMINI_LOCATION` | `global` (Vertex AI region) |

Pacing is read when a client is constructed, so a `config.yaml` edit is enough;
you only need the env vars to override a machine-specific value at runtime.

## Sections

| Section | Controls |
|---|---|
| `run` | `seed`, `datasets`, `splits`, `target_language`, `dry_run`, `test_parse` |
| `paths` | all input/output roots (`results_root`, `results_xt_root`, `results_dis_root`, `synthesis_root`, `bc_root`, `audio_root`, `source` + `parsed_source_root`, `data_root`, `input_path` + `input_paths`, `voice_clone_pool`, `logdir`). `logdir` is consumed by the `run_*.sh` wrappers only, not by any stage. |
| `llm` | shared `model` + per-role overrides (writer / boundary / tt / bc), `base_url` + `bc_base_url` / `boundary_base_url`, `api_key` + `boundary_api_key`, `temperature`, `gemini_min_interval`, `gemini_location` (both bridged to env, see above) |
| `stage1_speechify` | sample budgets, `max_variants`, `interviewer_subset`, `temperature`, `concise` |
| `stage1_5_cross_turn` | `perror`, `seed`, `roles`, `min_digits`, `min_code_len` |
| `stage1_75_disfluency` | `scales`, `shriberg_b`, `types`, `rep_span`, `inventories` (FP/DM/EDIT per language) |
| `stage4_synthesis` | `max_turns`, `max_dialogues`, `max_workers`, temperatures, `guards`, `ft_terminal_punct`, `hesitations`, `judge`, `hf` |
| `stage4b_backchannel` | `max_tokens`, `temperature`, `max_retries`, `valid_max_words`, fallback pools |
| `stage5_tts` | backend/language/device, `target_sr`/`prompt_sr`, `timing` (gap/pause/intra-pause/interrupt), `audio` (LUFS, VAD, noise floor, `save_align_json`), `aligner` (`model`/`dtype`/`device`, Qwen3 forced aligner), `voice` (instructs, pool), `tags` (13 supported + `render`), `backchannels` (candidates, rising tokens) |

### Offline Stage 1 source (`paths.source` / `paths.parsed_source_root`)

`paths.source: parsed_source` makes Stage 1 read the materialized dump at
`<paths.parsed_source_root>/<dataset>/<split>/*.json` for **every** dataset —
including `interviewer`/`soda`, which otherwise hit the HF Hub. No HF access,
no `paths.data_root`, no `paths.input_paths`. Each split is sampled at even
spacing with `stage1_speechify.max_train_samples` / `max_test_samples`, so a
small budget never parses a whole split (SODA is ~1.19M files). Default is
`auto` (raw corpora / Hub).

### Single-file corpora (`paths.input_path` / `paths.input_paths`)

Priority: **`input_paths[dataset]` > `input_path` (global fallback) > `data_root`**.

- **Run all 5**: `input_path: ""` + `input_paths.persuader: <file>` — socraticlm
  falls back to `data_root/SocraticLM/...`, multiwoz/negotiator to `data_root`.
- **Only persuader**: `input_paths.persuader: <file>` **or** `input_path: <file>`.
- **Only socraticlm**: `data_root` (`SocraticLM/...`) **or** `input_paths.socraticlm: <file>`.

Each stage loops over `run.datasets` × `run.splits` internally, so a single
invocation processes every dataset/split declared there.

## Notes

- **Prompts** stay in `src/synthesis/prompts.py` / `src/speechify_prompts.py`
  (large text blocks, not config).
- **Regexes and structural tokens** (`[TAKE_FLOOR]`, `[BACKCHANNEL]`,
  `[PAUSE]`) stay in code.
- `tests/test_config.py` covers the merge/precedence behaviour; see
  `config_example.yaml` for every key with its default value.
