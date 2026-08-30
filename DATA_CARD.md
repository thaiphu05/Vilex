# Data Card: Vilex datasets

> HF corpus temporarily removed. Pipeline reads local `data-annotations/` and `data-dialogues/` produced by Stage 1 or `tools/unpack_corpus.py`.

Two parts, laid out per scenario code (`TEA`, `PLN`, `INT`, `NEG`, `PER`, `SOC`):

```
dialogues/<CODE>/train.jsonl        # Vilex-generated dialogues (train-only)
annotations/<CODE>/train.jsonl      # human slot-level preference annotations
annotations/<CODE>/test.jsonl
```

## `dialogues/<CODE>/train.jsonl`

Each line is one generated dialogue:

```
{
  "example_id": str,
  "scenario": str,          # scenario code, e.g. "TEA"
  "license": str,           # inherited upstream license, see docs/DATA_LICENSES.md
  "context": str,
  "style": str,             # e.g. "spoken"
  "disfluency_target": str, # which speaker role disfluency was targeted at
  "speakers": [str, ...],
  "history": [
    {
      "role": str,
      "content": str,
      "segments": [
        # a segment is EITHER a plain content span:
        {"full_content": str},
        # OR a per-word turn-taking decision slot:
        {
          "word_index": int,
          "probs": {
            "floor_taking": float,
            "backchannel": float,
            "silence": float
          },
          "decision": str,          # the sampled/selected action at this slot
          "inserted_token": str | null
        }
      ]
    },
    ...
  ]
}
```

`segments` interleaves plain-text spans with per-word turn-taking decision slots in document order.

## `annotations/<CODE>/{train,test}.jsonl`

Human slot-level preference annotations. Note terminology: annotations use `"silent"`/`"take_floor"` where dialogues use `"silence"`/`"floor_taking"` (`"backchannel"` shared).

```
{
  "example_id": str,
  "scenario": str,
  "license": str,
  "history": [
    {
      "role": str,
      "content": str,
      "boundaries": [
        {
          "word_index": int,
          "total_count": int,
          "counts": {"silent": int, "backchannel": int, "take_floor": int},
          "probabilities": {"silent": float, "backchannel": float, "take_floor": float}
        },
        ...
      ]
    },
    ...
  ]
}
```

These are the labels Stage 3 is trained/evaluated against.

## Provenance and licensing

| Scenario | Code | Source dataset | License |
|---|---|---|---|
| Socratic teaching | TEA | SocraticLM | Apache-2.0 |
| Planning | PLN | MultiWOZ | MIT |
| Interview | INT | Anthropic Interviewer | CC-BY-4.0 |
| Negotiation | NEG | CraigslistBargain | MIT |
| Persuasion | PER | DailyPersuasion | Apache-2.0 |
| Social chat | SOC | SODA | CC-BY-4.0 |

See `docs/DATA_LICENSES.md` for full license text.

## Generator credit

Dialogue **text** is generated via LLM (default Gemini for Vilex VI). **Audio** is rendered with **OmniVoice** (VI, default) or **Chatterbox** (EN legacy). No third-party raw text is redistributed.
