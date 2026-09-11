# ARCHITECTURE — Vilex Pipeline

> Vilex = Vietnamese adaptation of DuplexGen (EMNLP 2026). 5-stage pipeline synthesizing duplex dialogues with word-by-word turn-taking.
> Vilex = DuplexGen Việt hóa. Pipeline 5 stage sinh hội thoại duplex, quyết định turn-taking từng từ.

## 1. Overview

Each dialogue flows through 5 stages. Turn-taking decision uses 3 labels: `floor_taking` / `backchannel` / `silence`.
Mỗi hội thoại đi qua 5 stage. Nhãn turn-taking gồm 3 loại trên.

Default path: Vietnamese (`--target_language vi`, Gemini writer, OmniVoice TTS). English (Chatterbox) is legacy, requires explicit flags.
Đường mặc định: tiếng Việt. English là legacy, cần flag explicit.

## 2. Data flow

```
6 source corpora
  → [Stage 1] spoken-style JSON (text_dialogue_<dataset>/<split>/*.json)
  → [Stage 2+4] slots detected + behavior sampled per slot
  → [Stage 4b] backchannel text filled (run_add_bc)
  → [Stage 5] two-channel audio (dialogue.wav + meta.json)
```

| Stage | Entry point | Env | Input | Output |
|---|---|---|---|---|
| 1. Spoken-style conversion | `src.speechify_run` | `.venv` | 6 source corpora (HF Hub or `--data-root`) | `text_dialogue_<dataset>/{train,test}/*.json` |
| 2. Slot identification | `src/synthesis/core.detect_turn_boundaries` (no separate command) | `.venv` | Stage 1 JSON `history` | `boundaries` list per user turn (`word_index`) |
| 3. Turn-taking prediction | `src.train_turntaking_hf`, `src.inference_turntaking_hf`, `src.inference_turntaking_llm` | `.venv` | `data-annotations/` | LoRA adapter or per-slot `probs` |
| 4. Dialogue generation | `src.synthesis.run` + `src.synthesis.run_add_bc` | `.venv` | Stage 1 JSON | JSON with `history` metadata + BC `content` |
| 5. TTS rendering | `tts_render/convert_spoken.py` (file path, not `-m`) | `.venv-tts` / omnivoice env | Stage 4 JSON | `var00/dialogue.wav`, `user.wav`, `assistant.wav`, `utterances/`, `backchannels/`, `meta.json` |

One-command run: `./run_vi_pipeline.sh`.
Chạy full pipeline một lệnh trên.

## 3. Environments (two interpreters)

Stages 1-4 run in `.venv` (Python 3.10+, `transformers>=4.53`). Stage 5 runs in separate interpreter (Python 3.11, `transformers==4.46.3` via vendored `vilex/tts/chatterbox`). Never merge: dependency conflict.
Stage 1-4 chạy `.venv`. Stage 5 chạy interpreter riêng. Không gộp env.

## 4. LLM backend routing (`src/llm_client.py`)

Backend chosen by model name. Backend chọn theo tên model.

- `gemini-*` → Google GenAI OpenAI-compat (`GEMINI_API_KEY` or `GEMINI_CREDENTIALS` service-account JSON). Default for Vilex.
- `gpt-*`, `o1`, `o3`, `o4` → `api.openai.com` (`OPENAI_API_KEY`).
- `Qwen/...` → self-hosted `--base_url` (default `localhost:8000`).

Stage 4 uses 3 independent roles, each own flags: writer `--llm_model_name`, boundary `--boundary_model_name`, predictor `--tt_model_name` (or `--hf_model_name_or_path` for Stage 3 LoRA).
Stage 4 dùng 3 vai LLM độc lập, mỗi vai flag riêng.

## 5. Stage 1 — Spoken-style conversion (`src.speechify_run`)

Purpose: rewrite clean text dialogues into spoken-style transcripts with fillers and natural breaks. Output feeds Stages 2-4.
Mục đích: viết lại hội thoại văn bản thành bản thoại nói tự nhiên.

Steps:

1. Load source per `--dataset`: `interviewer` + `soda` auto-download from HF Hub; `multiwoz` / `negotiator` / `socraticlm` need `--data-root`; `persuader` requires `--input_path`.
2. Split by one of 3 strategies in `src/speechify_run.py`: `split_by_file` (multiwoz/negotiator), `split_by_label` (socraticlm), `split_in_halves` (interviewer/persuader/soda).
3. Sample even spacing via `--max_train_samples` / `--max_test_samples` (default 25 each; `0` = all). Covers whole corpus, not prefix.
4. One LLM call per dialogue (`temperature 0.7`): prompt `SINGLE_STEP_CONVERSION_PROMPT` (`src/speechify_prompts.py`), structured JSON output `{"role","content"}`.
5. If `--target_language vi` (now default): append `LANG_DIRECTIVE[vi]`; output must be Vietnamese; keep `[TAKE_FLOOR]` unchanged.
6. Filter turns with role != `user`/`assistant` (drops Gemini `system` echo).
7. Write `<save_dir>/text_dialogue_<dataset>/{train,test}/*.json`. Existing files skipped (resume).

Knobs: `--test-parse` (dump source, no LLM call, smoke test miễn phí), `--concise` (shorten while converting), `--max_variants` (socraticlm/persuader).
Chi tiết sâu: `docs/stage1-speechify.md`.

## 6. Stage 2 — Slot identification (`src/synthesis/core.detect_turn_boundaries`)

Purpose: detect candidate intra-utterance action points (slots) inside user turns. No separate command; runs inside Stage 4.
Mục đích: tìm slot trong lượt user, nơi có thể chen hành vi. Không lệnh riêng.

Steps:

1. LLM boundary detector (`--boundary_model_name`, temperature `0.0`) marks clauses: prompt `PROMPT_BOUNDARY_DETECTION` asks model to insert `|` after each clause, punctuation kept.
2. Parse `|` markers → `predicted_indices` (word positions). LLM failure → empty list, heuristic only.
3. Apply 3 heuristic rules per word `i`: (A) `i` in LLM indices, (B) word ends with `[.,?!;]`, (C) word in `HESITATIONS` set (EN `um, uh, hmm` + VI `ưm, à, ừ, ơ, hử, chà`).
4. Drop final-turn boundary (`sorted(set(...))[:-1]`) to avoid end-of-turn slot.
5. Return `boundaries: List[int]` per turn; Stage 3 trains on these slots, Stage 4 queries predictor at each.

Chi tiết sâu: `docs/stage2-slots.md`.

## 7. Stage 3 — Turn-taking prediction

Purpose: score each Stage 2 slot with distribution over 3 labels. Metric: KL divergence vs human annotation.
Mục đích: chấm mỗi slot 3 xác suất. Điểm bằng KL divergence.

Two interchangeable predictors. Hai predictor thay thế nhau.

Steps (HF path, recommended, `tt_mode: hf_classification`):

1. Unpack annotations with `tools/unpack_corpus.py` → `--input_root data-annotations/`.
2. Train: `src.train_turntaking_hf --model_name_or_path Qwen/Qwen3-4B --style soft_classification` (default). QLoRA 4-bit (`bitsandbytes` + `accelerate`), LoRA `r=8`, KL loss on full human distribution.
3. Inference: `src.inference_turntaking_hf --peft_path <adapter>` — context 4 recent turns + partial turn up to slot → single forward → `softmax` → `floor_taking`/`backchannel`/`silence` probs. Failure → fallback `silence 1.0`.

Steps (LLM path, zero-shot alternative):

1. Run `src.inference_turntaking_llm --model <name>`: prompt `PROMPT_VERBALIZED_SCORING` asks chat model for 3 probs summing to 1.
2. Parse JSON, normalize. Unparsable/failed request → uniform fallback, counted in `n_fallback`.

Note label vocab mapping: train labels `backchannel/interruption/none` ↔ human labels `backchannel/take_floor/silent` (see `docs/CORPUS.md`).
Chú ý map vocab nhãn train vs human khác nhau.
Chi tiết sâu: `docs/stage3-prediction.md`.

## 8. Stage 4 — Turn-taking dialogue generation (`src.synthesis.run` + `run_add_bc`)

Purpose: regenerate each dialogue turn by turn, sampling behavior at every slot, filtering role-confused outputs.
Mục đích: sinh lại hội thoại từng lượt, sample hành vi tại slot.

Steps (`speechify_turn_by_turn` in `src/synthesis/core.py`):

1. Read Stage 1 JSON from `--input_root` (`text_dialogue_<dataset>/<split>/*.json`). `--max_dialogues` (1000), `--max_turns` (20) bound cost. Existing output skipped (resume).
2. Writer LLM (`--llm_model_name`) generates raw turn: user prompt (brief, disfluency allowed) vs assistant prompt (concise, no disfluency). VI appends `LANG_DIRECTIVE[vi]`.
3. Sanitize (`sanitize_utterance`): strip `<think>` tags, code fences, `role:`/`index)` prefixes, quotes; em-dash → comma. Contamination → retry up to 3 with corrective reminder, temp `+0.1` per attempt.
4. If turn is user: run Stage 2 slot detection → query predictor (HF adapter via `--hf_model_name_or_path` takes priority, else `--tt_model_name` chat model).
5. Insert action tokens with guards (`length_guard_start 0`, `interruption_guard_start 3`, `length_guard_gap 4`): `floor_taking` only from word 3+, max once per turn; backchannels spaced ≥4 words apart; turn start forced `silence`.
6. Only `[TAKE_FLOOR]` enters transcript text (truncates turn). Backchannel decisions stored in turn `history` metadata (`word_index`, `probs`, `decision`), stripped from LLM context of next turns.
7. Every `stop_check_every` turns after minimum 2: judge LLM (`DONE_JUDGE_PROMPT`) decides early stop (coverage done vs stalled).
8. Backchannel content (`run_add_bc`, 4th endpoint default port `8008`): fills each BC slot `content` (1-3 lowercase VI words, e.g. `ưm`, `à`, `vâng`), transcript unchanged.

Vilex VI uses `gemini-3.6-flash` for all 3-4 roles (no self-host). Vilex VI dùng Gemini cho mọi vai.
Chi tiết sâu: `docs/stage4-generation.md`.

## 9. Stage 5 — TTS rendering (`tts_render/convert_spoken.py`)

Purpose: render each dialogue JSON into `--num_variants` two-channel audio variants (ch0 assistant, ch1 user, 24 kHz 16-bit stereo).
Mục đích: render JSON thành audio 2 kênh.

Two backends. Hai backend:

- **OmniVoice (VI, default):** voice-design `instruct` string or voice-clone pool (`--omnivoice_voice_pool voice_clone/`, wav + sidecar txt, clamp 10s). No NeMo/pynini.
- **Chatterbox (EN, legacy):** fixed assistant prompt + LibriSpeech user voices. Needs NeMo + `pynini`.

Steps (`main_process`):

1. Parse Stage 4 JSON: walk `history`, reconstruct text with tokens, sort BC decisions by `word_index`; missing BC text → `DEFAULT_BC_CANDIDATES_VI` fallback; `[TAKE_FLOOR]` → cut turn plus one word.
2. Cast 2 distinct voices per dialogue, seeded `seed + crc32(filename)` (deterministic, resume-safe).
3. Synthesize sentence by sentence (`generate_audio`), cache backchannels (LRU). Empty audio → 0.2s silence guard.
4. Silero VAD trims leading/trailing silence per sentence for tight joins.
5. Timing layer 1 (`aggregate_speech`): same speaker joins directly; turn change → `0.16s` white-noise gap (`-44dBFS`); interrupt → cross-fade overlap `0.45s`–`0.64s`.
6. Timing layer 2 (backchannels only): `whisperx.align` (VI: `language_code="vi"`, `nguyenvulebinh/wav2vec2-base-vi-vlsp2020`) anchors BC to word-end timestamps.
7. Mix stereo, normalize full track to LUFS `-23` (`pyloudnorm`, peak fallback).
8. Write per variant `var00/`: `dialogues/dialogue.wav`, `user.wav`/`assistant.wav`, `utterances/`, `backchannels/`, `meta.json`. Variant with `dialogue.wav` + `meta.json` exists → skip (resume).

Cost knobs: `--max_dialogues`, `--num_variants` (linear cost), `--num_shards`/`--shard_id` for multi-GPU.
Chi tiết sâu: `docs/stage5-tts.md`.

## 10. Vietnamese path (Vilex default)

- Stages 1/4/add_bc: `--target_language vi` (default) + `gemini-3.6-flash` (or `-lite` when quota exhausted). Prompts stay English; output forced Vietnamese.
- Stage 5: `--tts_backend omnivoice --language vi` (default). No prompt wavs, no NeMo.
- Voice pool `voice_clone/`: each `<name>.wav` needs matching `<name>.txt` transcript; resampled 24k mono, cached once.
- Credentials: Vertex AI (`GEMINI_CREDENTIALS` + `GEMINI_LOCATION=global`) recommended; fallback `GEMINI_API_KEY` (free-tier ~20 req/day). `run_vi_pipeline.sh` errors clearly if both missing.

## 11. Data schema & scenario codes

6 scenarios: `socraticlm` (TEA), `multiwoz` (PLN), `interviewer` (INT), `negotiator` (NEG), `persuader` (PER), `soda` (SOC).
Case quirk: `SOC` = soda, `soc` = socraticlm — never casefold, always via `CODE2DIR` (`tools/unpack_corpus.py`).
Pipeline reads local `data-annotations/` and `data-dialogues/` (`text_dialogue_<dataset>/<split>/*.json`); HF corpus temporarily removed. Schema: `history[].segments` interleave `full_content` with slot `word_index/probs/decision/inserted_token`. Annotations use `silent/take_floor`; dialogues use `silence/floor_taking`.
Schema chi tiết: `docs/CORPUS.md`. Licenses: `docs/DATA_LICENSES.md`.

## 12. Verification

- `python -m pytest -q` (44 passed expected).
- `python -m py_compile` on 3 entry points: `src/speechify_run.py`, `src/synthesis/run.py`, `tts_render/convert_spoken.py`.
- Smoke: 1 dialogue end-to-end (e.g. `interviewer/work_0000`) Stages 1→5 on CPU, check `var00/dialogue.wav` + LUFS `-23`.
- Troubleshooting: `docs/TROUBLESHOOTING.md`.

## 13. Further reading

- [docs/stage1-speechify.md](docs/stage1-speechify.md), [docs/stage2-slots.md](docs/stage2-slots.md), [docs/stage3-prediction.md](docs/stage3-prediction.md), [docs/stage4-generation.md](docs/stage4-generation.md), [docs/stage5-tts.md](docs/stage5-tts.md)
- [docs/CORPUS.md](docs/CORPUS.md) (schema + data flow), [KNOWLEDGE.md](KNOWLEDGE.md) (code-level handover, Vietnamese)
- [docs/CORPUS.md](CORPUS.md#5-llm-calls) — every LLM call: call sites, per-dialogue estimates, full-run projections
