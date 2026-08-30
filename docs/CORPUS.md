# Using local corpus with Vilex

> HF corpus is temporarily removed. Pipeline reads local `data-annotations/` and `data-dialogues/` produced by Stage 1 or `tools/unpack_corpus.py`.

Stages 1 and 2 need no corpus — Stage 1 generates from third-party sources, Stage 2 runs inside Stage 4.

| | `dialogues/` | `annotations/` |
|---|---|---|
| Contains | Vilex-generated dialogue text + per-word turn-taking decisions | same dialogues + human rater votes at boundaries |
| Splits | `train` only | `train` and `test` |
| Needed by | Stage 4 input, Stage 5 rendering | Stage 3 training/evaluation |

Layout every stage reads: `text_dialogue_<dataset>/<split>/*.json` (one file per dialogue).

Use `tools/unpack_corpus.py` to convert between layouts if you have the packed JSONL. See `tools/` for `unpack_corpus.py` and `pack_corpus.py` arguments.

> Scenario codes are NOT case-insensitive shorthands. Release code `SOC` is **soda**, while `soc` is **socraticlm**. Always use `CODE2DIR` in `tools/unpack_corpus.py`.
