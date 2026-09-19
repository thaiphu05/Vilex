# Corpus, Data Flow & Schema — Vilex

Pipeline reads 6 upstream text-dialogue corpora, converts them through 6 stages to spoken Vietnamese with turn-taking behavior. Verified counts: 2026-09-11.

---

## 1. Upstream Source Data

### 1.1 Corpus Summary

| # | Corpus | Code | Source | License | Local path | Download |
|---|---|---|---|---|---|---|
| 1 | Interviewer | INT | HF `Anthropic/AnthropicInterviewer` | CC-BY-4.0 | `data/raw/interviewer/` | `datasets.load_dataset()` |
| 2 | MultiWOZ 2.2 | PLN | `budzianowski/multiwoz` | MIT | `data/raw/multiwoz/` | git clone |
| 3 | CraigslistBargain | NEG | `aladar/craigslist_bargains` (adapted) | MIT | `data/raw/CraigslistBargain/` | adapter script |
| 4 | SocraticLM (SocraTeach) | TEA | `SocraticLM/SocraTeach` | Apache-2.0 | `data/raw/SocraticLM/` | git clone |
| 5 | DailyPersuasion | PER | DailyPersuasion (PersuGPT) | Apache-2.0 | `data/raw/persuader/` | zip→JSON |
| 6 | SODA | SOC | HF `allenai/soda` | CC-BY-4.0 | HF streaming (no local cache) | auto-streamed |

### 1.2 Dialogue Counts

**Raw data** (what's in `data/raw/`):

| Corpus | Split | Files | Dialogue count | Notes |
|---|---|---|---|---|
| **INT** | workforce | `workforce.json` | 1,000 | HF split |
| | creatives | `creatives.json` | 125 | HF split |
| | scientists | `scientists.json` | 125 | HF split |
| | **subtotal** | | **1,250** | 3 JSON files, 12 MB total |
| **PLN** | train | `MultiWOZ_2.2/train/*.json` (17 files) | 8,437 | MultiWOZ_2.2 folder |
| | test | `MultiWOZ_2.2/test/*.json` (2 files) | 1,000 | |
| | **subtotal** | | **9,437** | ~288 MB |
| **NEG** | train | `train.json` | 5,147 | cocoa format (adapted from HF) |
| | test | `test.json` | 826 | |
| | **subtotal** | | **5,973** | ~50 MB |
| **TEA** | multi | `SocraTeach_multi.json` | 35,151 sessions (10,273 problems) | 5 sessions/problem avg |
| | single | `SocraTeach_single.json` | 20,845 sessions (20,845 problems) | 1 session/problem |
| | **subtotal** | | **55,996** | ~119 MB |
| **PER** | full | `DailyPersuasion_full_version.json` | 77,999 sessions (13,000 scenarios) | 6 sessions/scenario avg |
| | **subtotal** | | **77,999** | ~360 MB |
| **SOC** | train | HF streaming | 1,191,582 | not cached locally |
| | test | HF streaming | 148,968 | |
| | **subtotal** | | **1,340,550** | streamed at runtime |
| | | | | |
| **LOCAL TOTAL** | | | **150,416** | (5 corpora, excl. SODA) |

### 1.3 Domain & Sub-labels

| Corpus | Domain | Sub-labels | Count |
|---|---|---|---|
| **INT** | Job interviews | `workforce` — general job interviews | 1,000 |
| | | `creatives` — creative industry interviews | 125 |
| | | `scientists` — academic/scientific interviews | 125 |
| **PLN** | Task-oriented dialogue | `restaurant` — restaurant booking | 3,836 |
| | | `hotel` — hotel reservation | 3,369 |
| | | `train` — train ticket booking | 2,963 |
| | | `attraction` — attraction info | 2,681 |
| | | `taxi` — taxi booking | 1,463 |
| | | `hospital` — hospital booking | 107 |
| | | `bus` — bus info | 6 |
| **NEG** | Price negotiation | `furniture` — furniture bargaining | 1,279 |
| | | `housing` — rental negotiation | 1,051 |
| | | `bike` — bicycle bargaining | 924 |
| | | `car` — car negotiation | 676 |
| | | `electronics` — electronics bargaining | 673 |
| | | `phone` — phone bargaining | 544 |
| **TEA** | Math teaching | `GSM8K` — grade school math (8,973 problems) | 7,973 problems |
| | | `MAWPS` — math word problems (2,300 problems) | 2,300 problems |
| **PER** | Persuasion dialogue | 35 domains, top 10: | |
| | | `Education` — school/university topics | 1,979 |
| | | `Lifestyle` — daily life, home, hobbies | 1,713 |
| | | `Career` — job, workplace, growth | 1,168 |
| | | `Business` — enterprise, startup, market | 1,093 |
| | | `Technology` — AI, software, digital | 1,054 |
| | | `Health` — fitness, diet, wellness | 1,026 |
| | | `Psychology` — mental health, relationships | 876 |
| | | `Finance` — budgeting, investing, saving | 744 |
| | | `Ecology` — environment, sustainability | 715 |
| | | `Travel` — destinations, planning | 616 |
| | | (25 more domains: Ethics, Family, Art, Literature, ...) | |
| **SOC** | Social dialogue | Open-domain social conversations | 1,340,550 |

> **Persuader tags:** Each scenario has a `tag` field (specific topic, e.g. "Cultural Relics and Historic Sites") and a `domain` field (broad category). There are **5,097 unique tags** across 35 domains. Tags are too fine-grained for per-tag stats; domain-level distribution shown above. Each scenario also has 50 persuasion strategy items in `strategy` field.

### 1.4 Pipeline Input (Stage 1 reads)

Stage 1 uses `stage1_speechify.max_variants: 1` (default) for SocraticLM and Persuader — picks 1 dialogue per problem/scenario:

| Corpus | Code | Dialogue count | Source |
|---|---|---|---|
| Interviewer | INT | 1,250 | all 3 splits |
| MultiWOZ | PLN | 9,437 | train + test |
| CraigslistBargain | NEG | 5,973 | train + test |
| SocraticLM | TEA | 10,273 | `stage1_speechify.max_variants: 1` → 1/problem |
| Persuasion | PER | 13,000 | 13,000 scenarios |
| **5 tractable total** | | **39,933** | |
| SODA | SOC | 1,340,550 | competitive-filtered |

> **SODA not in `data/raw/`:** SODA is auto-streamed from HF at Stage-1 runtime (no local download). The pipeline calls `load_dataset('allenai/soda', streaming=True)` and processes rows on-the-fly. This avoids downloading ~1.3M dialogues (~2 GB parquet). SODA dialogues go through `is_competitive_scenario` LLM filter before conversion.

### 1.5 Turn Statistics

Turns = utterances (each speaker turn counts as 1). Measured from raw data.

| Corpus | Split | Dialogues | Turns | Avg turns/dialogue |
|---|---|---|---|---|
| **INT** | workforce | 1,000 | 25,140 | 25.1 |
| | creatives | 125 | 2,639 | 21.1 |
| | scientists | 125 | 2,444 | 19.6 |
| | **INT total** | **1,250** | **30,223** | **24.2** |
| **PLN** | train | 8,437 | 113,552 | 13.5 |
| | test | 1,000 | 14,744 | 14.7 |
| | **PLN total** | **9,437** | **128,296** | **13.6** |
| **NEG** | train | 5,147 | 38,688 | 7.5 |
| | test | 826 | 6,258 | 7.6 |
| | **NEG total** | **5,973** | **44,946** | **7.5** |
| **TEA** | multi | 35,151 | 185,602 | 5.3 |
| | single | 20,845 | 118,040 | 5.7 |
| | **TEA total** | **55,996** | **303,642** | **5.4** |
| **PER** | full | 77,999 | 792,343 | 10.2 |
| | **PER total** | **77,999** | **792,343** | **10.2** |
| **LOCAL TOTAL** | | **150,416** | **1,299,450** | |

SODA excluded — 1.3M dialogues, streaming-only, turn stats impractical to compute locally.

### 1.6 Disk Sizes

| Item | Size | Location |
|---|---|---|
| `data/raw/` (all 6 source corpora) | **932 MB** | gitignored |
| Interviewer (3 JSONs) | ~12 MB | `data/raw/interviewer/` |
| MultiWOZ 2.2 | ~288 MB | `data/raw/multiwoz/data/MultiWOZ_2.2/` |
| CraigslistBargain | ~50 MB | `data/raw/CraigslistBargain/` |
| SocraticLM (multi + single) | ~119 MB | `data/raw/SocraticLM/data/` |
| Persuader | ~360 MB | `data/raw/persuader/` |
| SODA (parquet, if materialized) | ~231 MB | HF cache |

### 1.7 Leftover Zip Files

All under `data/raw/` (gitignored). Safe to delete:

| Zip | Size | Notes |
|---|---|---|
| `multiwoz/data/MultiWOZ_1.0.zip` | 13 MB | legacy, Vilex uses 2.2 folder |
| `multiwoz/data/MultiWOZ_2.0.zip` | 14 MB | legacy |
| `multiwoz/data/MultiWOZ_2.1.zip` | 20 MB | legacy |
| **Total** | **~47 MB** | |

Note: `DailyPersuasion.zip` was previously present (69 MB) but has been deleted after extraction.

---

## 2. Data Flow Through Pipeline

Source data flows through 6 stages. Each stage reads from the previous stage's output directory.

```
data/raw/                    (upstream source corpora)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: speechify_run.py                          │
│  Input:  data/raw/<corpus>/                          │
│  Output: data/results_vi/text_dialogue_<dataset>/{split}/ │
│  LLM:    1 call/dialogue (spoken-style conversion)   │
│  What:   Text dialogue → spoken Vietnamese dialogue  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: synthesis/run.py                          │
│  Input:  data/results_vi/                            │
│  Output: data/vi_tt/                             │
│  LLM:    ~100 calls/dialogue                        │
│          Writer (~20) + Boundary (~10) + TT (~40)   │
│          + Done check (~20) + retries                │
│  What:   Generate turn-taking dialogue with         │
│          floor_taking / backchannel / silence        │
│          decisions at word-level boundaries          │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4b: synthesis/run_add_bc.py                  │
│  Input:  data/vi_tt/                             │
│  Output: data/vi_tt_bc/                          │
│  LLM:    ~10 calls/dialogue                         │
│  What:   Generate backchannel text for [INSERT]     │
│          points (1-3 word listener responses)        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5: tts_render/convert_spoken.py              │
│  Input:  data/vi_tt_bc/                          │
│  Output: data/vi_audio/ (2-channel WAV)             │
│  LLM:    0 calls (OmniVoice/Chatterbox TTS)        │
│  What:   Render dialogue to audio with voice-clone  │
│          pool, inter-turn gaps, intra-turn pauses   │
└─────────────────────────────────────────────────────┘
```

### Stage I/O Summary

| Stage | Input dir | Output dir | LLM calls | What changes |
|---|---|---|---|---|
| 1 | `data/raw/` | `data/results_vi/` | 1/dialogue | Text → spoken Vietnamese |
| 4 | `data/results_vi/` | `data/vi_tt/` | ~100/dialogue | Turn-taking generation |
| 4b | `data/vi_tt/` | `data/vi_tt_bc/` | ~10/dialogue | Backchannel text |
| 5 | `data/vi_tt_bc/` | `data/vi_audio/` | 0 | TTS audio rendering |

> **Critical:** Stage 4 reads from `paths.results_root` (`data/results_vi/`). Feeding the wrong directory → silent failure.

---

## 3. Output Schemas

### 3.1 Stage 1 Output (`data/results_vi/text_dialogue_<dataset>/{split}/*.json`)

One JSON file per dialogue:

```json
{
  "example_id": "interviewer__workforce__00042",
  "scenario": "INT",
  "license": "CC-BY-4.0",
  "context": "Job interview for software engineer position...",
  "style": "spoken",
  "disfluency_target": "none",
  "speakers": ["interviewer", "candidate"],
  "history": [
    {
      "role": "interviewer",
      "content": "Chào bạn, hãy giới thiệu về bản thân.",
      "segments": [
        {"full_content": "Chào bạn, hãy giới thiệu về bản thân."}
      ]
    },
    {
      "role": "candidate",
      "content": "Em tên là Minh, sinh viên năm cuối...",
      "segments": [
        {"full_content": "Em tên là Minh, sinh viên năm cuối..."}
      ]
    }
  ]
}
```

### 3.2 Stage 4 Output (`data/vi_tt/text_dialogue_<dataset>/{split}/*.json`)

Turn-taking dialogue with per-word boundary decisions:

```json
{
  "example_id": "interviewer__workforce__00042",
  "scenario": "INT",
  "history": [
    {
      "role": "interviewer",
      "content": "Chào bạn, hãy giới thiệu về bản thân.",
      "segments": [
        {"full_content": "Chào bạn, hãy giới thiệu về bản thân."}
      ]
    },
    {
      "role": "candidate",
      "content": "Em tên là Minh|sinh viên năm cuối|...",
      "segments": [
        {"full_content": "Em tên là Minh"},
        {
          "word_index": 5,
          "probs": {
            "floor_taking": 0.1,
            "backchannel": 0.7,
            "silence": 0.2
          },
          "decision": "backchannel",
          "inserted_token": "[BACKCHANNEL]"
        },
        {"full_content": "sinh viên năm cuối"},
        {
          "word_index": 10,
          "probs": {
            "floor_taking": 0.6,
            "backchannel": 0.2,
            "silence": 0.2
          },
          "decision": "floor_taking",
          "inserted_token": "[FLOOR_TAKING]"
        },
        {"full_content": "..."}
      ]
    }
  ]
}
```

`segments` interleaves plain-text spans with per-word turn-taking decision slots. `probs` = LLM-predicted probabilities. `decision` = sampled action.

### 3.3 Stage 4b Output (`data/vi_tt_bc/`)

Same as Stage 4, but `[BACKCHANNEL]` tokens replaced with generated text:

```json
{
  "inserted_token": "ừm",
  "decision": "backchannel"
}
```

### 3.4 Annotation Schema (for Stage 3 training)

Human slot-level preference labels. Stored in `annotations/<CODE>/{train,test}.jsonl`:

```json
{
  "example_id": "interviewer__workforce__00042",
  "scenario": "INT",
  "license": "CC-BY-4.0",
  "history": [
    {
      "role": "candidate",
      "content": "Em tên là Minh...",
      "boundaries": [
        {
          "word_index": 5,
          "total_count": 10,
          "counts": {"silent": 2, "backchannel": 7, "take_floor": 1},
          "probabilities": {"silent": 0.2, "backchannel": 0.7, "take_floor": 0.1}
        }
      ]
    }
  ]
}
```

> **Terminology note:** Annotations use `"silent"`/`"take_floor"` where dialogues use `"silence"`/`"floor_taking"` (`"backchannel"` shared).

---

## 4. Scenario Codes

| Code | Corpus | Directory name |
|---|---|---|
| INT | Interviewer | `interviewer` |
| PLN | MultiWOZ | `multiwoz` |
| NEG | CraigslistBargain | `negotiator` |
| TEA | SocraticLM | `socraticlm` |
| PER | Persuasion | `persuader` |
| SOC | SODA | `soda` |

> **Case-sensitive.** `SOC` = soda, `soc` = socraticlm. Always use `CODE2DIR` in `tools/unpack_corpus.py`.

---

## 5. LLM Calls

All roles use `gemini-3.6-flash`. Per-dialogue cost ≈ **101 calls** (1 Stage 1 + ~100 Stage 4/4b). `stop_check_every=1`.

### 5.1 Call sites (7 locations)

Only Stages 1, 4, and 4b make API calls; Stage 5 is LLM-free (§5.5).

| # | Stage | Function | File:Line | Model config | Purpose |
|---|---|---|---|---|---|
| 1 | **1** | `_generate_dialogue_structured` | `src/speechify_core.py:41` | `llm.writer_model` | Convert source dialogue → spoken-style Vietnamese |
| 1b | **1** (SODA only) | `is_competitive_scenario` | `src/speechify_datasets.py:275` | `llm.writer_model` | Filter SODA dialogues for competitive scenarios (temp=0, max_tokens=5) |
| 2 | **4** Writer | `generate_raw_content_turn` | `src/synthesis/core.py:517` | `llm.writer_model` | Generate spoken text for 1 dialogue turn (user or assistant) |
| 3 | **4** Boundary | `detect_turn_boundaries` | `src/synthesis/core.py:175` | `llm.boundary_model` | Insert `\|` markers at clause boundaries in user turns (temp=0) |
| 4 | **4** TT Predictor | `predict_turn_taking_probabilities` | `src/synthesis/core.py:244` | `llm.tt_model` | Score each boundary index: floor_taking / backchannel / silence (temp=0) |
| 5 | **4** Done check | `judge_done` | `src/synthesis/core.py:603` | `llm.writer_model` | Coverage + stall check: compare generated vs source to decide whether to stop (temp=0) |
| 6 | **4b** Backchannel | `generate_backchannel` | `src/synthesis/run_add_bc.py:121` | `llm.bc_model` | Generate 1-3 word listener backchannel text for each `[INSERT]` point |

### 5.2 Per-dialogue estimates

For a dialogue with ~20 turns (~10 user turns, ~4 boundaries/user turn):

| Call site | Count per dialogue | Notes |
|---|---|---|
| Stage 1 — convert | **1** | 1 call/dialogue total |
| Stage 4 — Writer | **~20** | 1 call per generated turn (± retries up to 3) |
| Stage 4 — Boundary | **~10** | 1 call per user turn (temp=0, short response) |
| Stage 4 — TT Predictor | **~40** | 1 call per boundary index, per user turn (temp=0) |
| Stage 4 — Done check | **~5-20** | Every `stop_check_every` turns (default=1 → ~20) |
| Stage 4b — Backchannel | **~8-15** | 1 call per `[INSERT]` backchannel point |
| **Total / dialogue** | **~84-96** | |

### 5.3 Full-run projections

| Corpus | Dialogues | Stage 1 | Stage 4+4b | Total |
|---|---|---|---|---|
| Interviewer | 1,250 | 1,250 | ~125,000 | ~126,250 |
| MultiWOZ | 9,437 | 9,437 | ~943,700 | ~953,137 |
| CraigslistBargain | 5,973 | 5,973 | ~597,300 | ~603,273 |
| SocraticLM | 10,273 | 10,273 | ~1,027,300 | ~1,037,573 |
| Persuasion | 13,000 | 13,000 | ~1,300,000 | ~1,313,000 |
| **5 tractable** | **39,933** | **39,933** | **~3,993,300** | **~4,033,233** |
| SODA (incl. filter) | 1,340,550 | 1,340,550¹ | ~134,055,000 | **~135,395,550** |

¹ SODA adds 1 LLM competitive-filter call per row inside `iter_soda`.

**Bottom line:** 5 tractable corpora ≈ **4.03M LLM calls**. All 6 (incl. SODA) ≈ **135M calls** — SODA alone is ~33× the rest.

Gemini free-tier ≈ 20 req/day. Vertex AI (`GEMINI_CREDENTIALS`) has higher quotas; check GCP console.

### 5.4 Call-site details

**Stage 1 — Spoken-style conversion** (`src/speechify_core.py:41`). Receives the full source dialogue + context, returns a structured JSON list of `{role, content}` turns. Pydantic `parse` API for OpenAI, `response_format={"type":"json_object"}` for Gemini/vLLM. System prompt: `SINGLE_STEP_CONVERSION_PROMPT` (or `_CONCISE` variant, `src/speechify_prompts.py:1`). Temperature = 0.7.

**Stage 4 — Writer** (`src/synthesis/core.py:517`). Generates **one utterance** at a time. Two system prompts in `src/synthesis/prompts.py`: `ROLE_USER_TURN_PROMPT` (human) and `ROLE_AI_TURN_PROMPT` (assistant). Receives: original full dialogue (reference), partial spoken history, current source turn. Handles `{TOKEN_FT}` (floor-taking/interruption) logic. Retries up to 3 times on format contamination (role prefixes, indices, quotes). Temperature = 0.7 for user, 0.2 for assistant.

**Stage 4 — Boundary detection** (`src/synthesis/core.py:175`). Asks the LLM to insert `|` at clause boundaries (`PROMPT_BOUNDARY_DETECTION`, `src/synthesis/prompts.py`), then maps markers to word indices. Also applies heuristic rules: punctuation endings `[.,?!;]` and hesitation words (`ưm, à, ừ, ơ, hử, chà`). Drops the final boundary to avoid end-of-turn slot. Temperature = 0.

**Stage 4 — TT predictor** (`src/synthesis/core.py:244`). Loops over each boundary index in a user turn. For each boundary, sends dialogue context + partial utterance up to that point (`PROMPT_VERBALIZED_SCORING`, `src/synthesis/prompts.py`). Expects JSON with `{floor_taking, backchannel, silence}` probabilities. Temperature = 0. **Most expensive call: N boundaries × N user turns per dialogue.** HF LoRA alternative (`predict_turn_taking_probabilities_hf`, `src/synthesis/core.py:309`) makes 0 API calls.

**Stage 4 — Done check** (`src/synthesis/core.py:603`). Compares original source turns vs generated spoken turns (`DONE_JUDGE_PROMPT`, `src/synthesis/prompts.py`). Returns JSON: `{done, missing, stop, stop_reason, notes}`. Called every `stop_check_every` turns. Two criteria: (1) coverage — all source content accounted for? (2) stall — recent turns add no new information? Temperature = 0.

**Stage 4b — Backchannel** (`src/synthesis/run_add_bc.py:121`). Produces 1-3 word listener responses for each `[INSERT]` point in Stage 4 output. System prompt: `SYSTEM_PROMPT_QWEN_VI` (`src/synthesis/run_add_bc.py`, or English variants). Temperature = 0.7, max_tokens = 12.

### 5.5 Stages with 0 LLM calls

| Stage | Script | Mechanism |
|---|---|---|
| **5** TTS | `tts_render/convert_spoken.py` | OmniVoice/Chatterbox TTS model; no LLM calls (uses voice-clone pool for inference) |

---

## 6. How to Download

```bash
# Create venv (python 3.11 required)
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# MultiWOZ
git clone https://github.com/budzianowski/multiwoz.git data/raw/multiwoz

# SocraticLM
git clone https://github.com/SocraticLM/SocraTeach.git data/raw/SocraticLM

# CraigslistBargain (adapt from HF)
.venv/bin/pip install datasets
# Use adapter script to convert HF → cocoa format

# DailyPersuasion
# Download DailyPersuasion_full_version.zip → unzip to data/raw/persuader/

# Interviewer + SODA
# Auto-streamed from HF at Stage-1 runtime (no manual download needed)
```

---

## 7. Gitignore

All pipeline outputs and source data are gitignored:

- `data/raw/` — upstream source corpora
- `results/`, `data/results_vi/` — Stage 1
- `outputs/`, `output/` — Stages 4→5
- `data/results_vi/parsed_source/` — test-parse dumps (under `paths.results_root`)
- `text_dialogue_*/**/*.json` — generated dialogue JSON
- `Sample/` — Stage 5 audio output

---

## 8. Provenance & Licensing

| Scenario | Code | Source dataset | License |
|---|---|---|---|
| Socratic teaching | TEA | SocraticLM | Apache-2.0 |
| Planning | PLN | MultiWOZ | MIT |
| Interview | INT | Anthropic Interviewer | CC-BY-4.0 |
| Negotiation | NEG | CraigslistBargain | MIT |
| Persuasion | PER | DailyPersuasion | Apache-2.0 |
| Social chat | SOC | SODA | CC-BY-4.0 |

See `docs/DATA_LICENSES.md` for full license text. No third-party raw text is redistributed — only Vilex-generated dialogues and audio.
