# Stage 4: Turn-taking dialogue generation

Queries the predictor at each slot and inserts the chosen behavior, then filters
role-confused outputs.

This stage regenerates each dialogue turn by turn and, at every slot Stage 2
detects inside a user turn, samples a turn-taking behavior from the predictor's
distribution. It drives **three separate LLM roles**, each with its own config
key, so decide what serves each one before running (`cp config_example.yaml
config.yaml`; see [CONFIGURATION.md](CONFIGURATION.md)):

| Role | Config key | Default | What it does |
|---|---|---|---|
| Writer | `llm.writer_model` (with `llm.api_key`, `llm.base_url`) | `gemini-3.6-flash` | Generates each turn's text and judges when the dialogue is done |
| Boundary detector (Stage 2) | `llm.boundary_model` | `gemini-3.6-flash` | Marks clause boundaries, which become the candidate slots |
| Turn-taking predictor (Stage 3) | `llm.tt_model` | `gemini-3.6-flash` | Scores `floor_taking` / `backchannel` / `silence` at each slot |

Each follows the model-name rule from [
  
](../README.md#setup) independently, so a bare run needs whatever backend those
names imply (`GEMINI_API_KEY` / `GEMINI_CREDENTIALS`, `OPENAI_API_KEY`, or a
self-hosted `llm.base_url`). A minimal trial run:

```yaml
run:
  datasets: [interviewer]
  splits: [train]
llm:
  writer_model: gemini-3.6-flash
  boundary_model: gemini-3.6-flash
  tt_model: gemini-3.6-flash
stage4_synthesis:
  max_dialogues: 15
  max_turns: 0        # 0 = auto = source length
paths:
  results_root: data/results_vi           # input (Stage 1 output)
  synthesis_root: data/vi_tt              # output
```

```bash
export GEMINI_API_KEY=...   # or GEMINI_CREDENTIALS / OPENAI_API_KEY
.venv/bin/python -m src.synthesis.run
```

## Using the Stage 3 predictor instead

`llm.tt_model` asks a chat model for verbalized probabilities. To use the LoRA
token classifier Stage 3 trains — the configuration behind the released corpus,
whose dialogues carry `"tt_mode": "hf_classification"` — set
`stage4_synthesis.hf_model_name_or_path` to the adapter directory. It takes
priority over `llm.tt_model`, needs no `llm.base_url`, and loads in process, so
this run needs a GPU:

```yaml
stage4_synthesis:
  hf_model_name_or_path: ./output/turntaking_qwen3_4b
  hf:
    load_in_4bit: true
    max_seq_length: 1024
    use_last_n_history: 4
    batch_size: 8
```

Point it at the `--output_dir` of `src.train_turntaking_hf`; the base model is
read from the adapter's `adapter_config.json`, so there is no separate
base-model key here.

## Inputs and outputs

`run.datasets` accepts `socraticlm | multiwoz | interviewer | negotiator |
persuader | soda` (plus `all`); `run.splits` is `train | test`. Stage 4 reads
`paths.results_root` in the `text_dialogue_<dataset>/<split>/*.json` layout
produced by the Stage 1 step — **not** the published HF layout; convert with
`tools/unpack_corpus.py --kind dialogues` first (see [CORPUS.md](CORPUS.md)).
That half is train-only, so `test` needs dialogues of your own from Stage 1.
Results are written to
`<paths.synthesis_root>/text_dialogue_<dataset>/<split>/*.json`, and an existing
output file is skipped, so a re-run resumes. `stage4_synthesis.max_dialogues`
(15) and `max_turns` (0 = auto = source length) bound the cost of a trial run.

Backchannel decisions are recorded per slot in each turn's `history`, not
inserted into `content`: only `[TAKE_FLOOR]` appears in the transcript text.
This matches the released corpus, and Stage 5 renders backchannels from the
slot metadata.

## Backchannel content

Optional, and comes last, filling in what each backchannel slot actually says.
It is a fourth endpoint, configured by `llm.bc_model` at `llm.bc_base_url`
(default `http://localhost:8008/v1`) — port **8008**, not the 8000 the other
stages default to, so that it can be a second server:

```yaml
llm:
  bc_model: gemini-3.6-flash
  bc_base_url: http://localhost:8008/v1
stage4b_backchannel:
  prompt_kind: qwen
  max_tokens: 32
  valid_max_words: 3
paths:
  synthesis_root: data/vi_tt      # input (Stage 4 output)
  bc_root: data/vi_tt_bc          # output
```

```bash
.venv/bin/python -m src.synthesis.run_add_bc
```

Both roots come from `paths.synthesis_root` (input) and `paths.bc_root`
(output). The run writes the generated text into each backchannel slot's
`content` field, leaving the transcript text unchanged.

Source: `src/synthesis/*.py`.

Next: [Stage 5](stage5-tts.md).
