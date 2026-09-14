# Stage 1: Spoken-style conversion

Rewrites clean text dialogues from six scenario datasets into spoken-style
transcripts. See [Setup](../README.md#setup) for the environment and the
model-name rule that picks the LLM backend.

`interviewer` and `soda` pull their source dialogues from the Hugging Face Hub,
so they run with nothing downloaded ahead of time. The stage is driven entirely
by `config.yaml` (`cp config_example.yaml config.yaml`; see
[CONFIGURATION.md](CONFIGURATION.md)) — every stage takes only an optional
`--config PATH`, no dataset/flag CLI:

```yaml
run:
  datasets: [interviewer, soda]
  splits: [train, test]
llm:
  writer_model: gemini-3.6-flash
```

```bash
export OPENAI_API_KEY=sk-...   # or GEMINI_API_KEY / GEMINI_CREDENTIALS
.venv/bin/python -m src.speechify_run            # reads ./config.yaml
.venv/bin/python -m src.speechify_run --config my.yaml
```

The other scenarios read a local copy of their upstream corpus: point
`paths.data_root` at it (and, for the two single-file corpora, `paths.input_path`
or the per-dataset `paths.input_paths`):

```yaml
paths:
  data_root: /path/to/raw/datasets
  input_paths:
    persuader: /path/to/DailyPersuasion_full_version.json
```

Output lands in `<paths.results_root>/text_dialogue_<dataset>/{train,test}/*.json`
— the same layout Stages 2-4 read. Existing files are skipped, so a re-run resumes.

**Sample budget.** Every dialogue costs one LLM call, so each split is capped
by `stage1_speechify.max_train_samples` / `max_test_samples` (25 each by
default; `0` means no cap). Both count *dialogues*, and both sample at even
spacing across the whole source rather than taking a prefix, so a small budget
still spans the corpus. `stage1_speechify.max_variants` controls how many
retellings of one scenario to keep for the corpora that ship several
(`socraticlm`, `persuader`); raising it does not inflate the per-split counts.

To smoke-test the stage without spending anything, set `run.test_parse: true`:
it dumps the parsed source dialogues and never calls the LLM.

The dump lands in
`<paths.results_root>/parsed_source/<dataset>/<split>/*.json`, apart from the
converted output, so a later real run into the same `paths.results_root` still
converts every dialogue.

## Source datasets

`run.datasets` selects scenarios: two download automatically through
`datasets.load_dataset`; the other four must be obtained yourself and placed
under `paths.data_root` at the exact paths in `LOCAL_CORPUS_PATHS`
(`src/speechify_run.py:42`):

| `run.datasets` entry | Source | Obtain from | Expected path |
|---|---|---|---|
| `interviewer` | Anthropic Interviewer | auto (`Anthropic/AnthropicInterviewer`) | — |
| `soda` | SODA | auto (`allenai/soda`) | — |
| `multiwoz` | MultiWOZ 2.2 | <https://github.com/budzianowski/multiwoz> | `<paths.data_root>/multiwoz/data/MultiWOZ_2.2/{train,test}/` |
| `negotiator` | CraigslistBargain | <https://stanfordnlp.github.io/cocoa/> | `<paths.data_root>/CraigslistBargain/train.json`, `<paths.data_root>/CraigslistBargain/test.json` |
| `socraticlm` | SocraticLM (SocraTeach) | <https://github.com/Ljyustc/SocraticLM> | `<paths.data_root>/SocraticLM/data/SocraTeach_multi.json` |
| `persuader` | DailyPersuasion (PersuGPT) | see the PersuGPT release | `paths.input_paths.persuader` (required) |

`paths.data_root` is required only for the three scenarios with a local source
(`multiwoz`, `negotiator`, `socraticlm`); omitting it for one of those raises a
message naming the key. `persuader` ignores `paths.data_root` and **requires**
`paths.input_paths.persuader` (or the global `paths.input_path`); it raises
immediately without one. `socraticlm` also accepts either override to replace
its default location.

`run.datasets` can list all six at once; that needs `paths.data_root` plus a
`paths.input_paths.persuader` entry. A per-dataset override always wins over the
global `paths.input_path`.

Other knobs: `llm.writer_model` and `stage1_speechify.temperature` (0.7) control
the conversion call, `stage1_speechify.interviewer_subset` (`workforce`) picks
which Anthropic Interviewer subset to read, and `stage1_speechify.concise`
switches to a prompt that shortens the dialogue as it converts.

Source: `src/speechify_run.py`, `src/speechify_core.py`,
`src/speechify_datasets.py`, `src/speechify_prompts.py`.

Next: [Stage 2](stage2-slots.md).
