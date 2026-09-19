# KNOWLEDGE — Vilex Handover for AI

> Chắt lọc từ `Overview.md` (433 dòng, doc pipeline VI) + `Session.md` (8844 dòng, log session chuyển đổi DuplexGen→Vilex). Bỏ log thô, hardcode máy cá nhân. Giữ đủ sửa/chạy pipeline, khỏi đọc lại 2 file gốc.

---

## 1. TL;DR

* **Vilex** = DuplexGen Việt hóa, pipeline 5 stages, hội thoại duplex, turn-taking **từng từ** (`floor_taking`/`backchannel`/`silence`). Default **tiếng Việt** (`vi` + Gemini + OmniVoice), English (Chatterbox) legacy (`run.target_language: en`).
* **Hai env tách** `src/pyproject.toml:requires-python` vs `vilex/tts/chatterbox/pyproject.toml:requires-python`: Stage 1-4 (`transformers>=4.53`) vs Stage 5 (`transformers==4.46.3` + `Python >=3.11,<3.12`) khác interpreter (`AGENTS.md:7`, `tools/test_env_consistency.py:47`).
* Entry: `python -m src.prepare_corpus` (Stage0, optional), `python -m src.speechify_run` (Stage1), `python -m src.cross_turn_slots` (Stage1.5), `python -m src.disfluency` (Stage1.75), `python -m src.synthesis.run` + `run_add_bc` (Stage2/4), `python tts_render/convert_spoken.py` (Stage5 file, env khác). Quickstart 1 dialogue: `./run_vi_pipeline.sh`; batch 5 datasets: `./run_local_stages1-4.sh`.
* **Stage 1.5/1.75 rule-based, 0 API call** — dictation slot đa lượt + misspeak-repair (`stage1_5_cross_turn.perror 0.20`) và disfluency Switchboard/Shriberg. Chạy theo thứ tự `data/results_vi → data/results_vi_xt → data/results_vi_dis`; Stage 4 đọc `data/results_vi_dis`.
* **Voice pool:** `voice_clone/` (85 pairs) build từ Kaggle vi-common-voice + Whisper ASR server (`tools/build_voice_clone_pool.py`, spec `ExternalAsvFileWhisper.openapi.yaml:9670`).

---

## 2. Repo map (đụng file nào khi sửa stage nào)

```
vilex/                          # đổi từ duplexgen/ (Phase 2)
├── src/
│   ├── speechify_run.py:357    # run.target_language default vi (Phase 3)
│   ├── speechify_core.py       # speechify_full_dialogue lọc role != user/assistant (fix 2026-08-26)
│   ├── speechify_datasets.py   # 6 datasets, get_sampled_indices even spacing
│   ├── speechify_prompts.py    # LANG_DIRECTIVE[vi]
│   ├── cross_turn_slots.py     # Stage 1.5: slot dictation đa lượt + misspeak-repair (rule-based, 0 API)
│   ├── disfluency.py           # Stage 1.75: inject FP/DM/EDIT/REP + "...">[PAUSE] (Switchboard/Shriberg)
│   ├── prepare_corpus.py       # Stage 0: parse 6 corpus gốc → JSON cho src/synthesis/run.py
│   ├── swbd_parse.py, swbd_convert.py  # Switchboard dialog-act → streaming dialogue JSON
│   ├── hf_attn.py              # resolve_attn_implementation: FA2 nếu có else SDPA (Stage 3)
│   ├── (tests moved to repo-root tests/: test_label_taxonomy.py, test_pipeline_contracts.py, ...)
│   ├── synthesis/
│   │   ├── run.py:191          # run.target_language default vi
│   │   ├── core.py             # detect_turn_boundaries + speechify_turn_by_turn (Stage2+4)
│   │   ├── prompts.py          # ROLE_*_TURN_PROMPT, đã gỡ OMNI_PARALINGUIST_TAGS 2026-08-27
│   │   └── run_add_bc.py:17,256 # TARGET_LANGUAGE default vi
│   ├── gemini_client.py:3      # shim OpenAI→google-genai, rate limit _MIN_INTERVAL 13s + 429 backoff
│   ├── boundary_annotator.py:187 # help string đã bỏ HF, trỏ data-dialogues/
│   ├── inference_turntaking_*.py, train_turntaking_hf.py:745
│   └── llm_client.py           # routing theo tên model (gpt-* / gemini / Qwen)
├── tts_render/
│   ├── convert_spoken.py:42,1589 # stage5_1b_render.backend default omnivoice, language vi, device cpu/cuda
│   │                              # aggregate_speech (gap 0.16s, overlap 0.45-0.64s), Qwen3 aligner vi
│   ├── bc_cache.py, prompt_wavs/, librispeech_samples/
│   └── requirements? không, dùng vilex/tts/chatterbox
├── vilex/tts/chatterbox/       # vendored MIT, pyproject.toml: name chatterbox-tts
├── tools/ unpack_corpus.py, pack_corpus.py, build_voice_clone_pool.py, test_*.py
├── run_vi_pipeline.sh:4        # relocatable REPO="$(cd $(dirname $0)&&pwd)", PY=${PY:-python}, default vi+omnivoice
├── run_local_stages1-4.sh      # batch 5 datasets: Stage1→1.5→1.75→4→add_bc, log thẳng ra terminal
├── ExternalAsvFileWhisper.openapi.yaml  # spec Whisper ASR server (port 9670) cho build voice pool
├── kaggle/ vilex_kaggle.ipynb (2-phase), vilex_stage5_kaggle.ipynb (GPU T4)
├── data/vi-common-voice/       # cache Kaggle clip+TSV nguồn build voice_clone
├── data/results_vi/ data/results_vi_xt/ data/results_vi_dis/   # Stage 1 → 1.5 → 1.75
├── data/vi_tt/ data/vi_tt_bc/             # Stage 4 → add_bc
├── pyproject.toml:1            # name=vilex, requires-python>=3.10, [project.urls] vilex
├── REUSE.toml:1                # vilex/tts/chatterbox/** MIT
├── requirements*.txt, constraints.txt, docs/CORPUS.md, docs/
└── voice_clone/                # pool wav+txt cho OmniVoice clone (85 pairs cvvi_*, build bởi tools/build_voice_clone_pool.py)
```

**Test gate:** `tools/test_docs.py:214` cần `pyproject.toml`, `tools/test_docs.py:235` cần `REUSE.toml` với `vilex/tts/chatterbox/**`. `tools/test_env_consistency.py:58` đọc `vilex/tts/chatterbox/pyproject.toml`.

---

## 3. Pipeline 5 stages — chi tiết vận hành (tổng hợp Overview.md:56-172)

| Stage | Tên | Entry point | Docs |
|---|---|---|---|
| 0 | Corpus prep (optional) | `src.prepare_corpus` | `docs/CORPUS.md` |
| 1 | Spoken-style conversion | `src.speechify_run` | `docs/stage1-speechify.md` |
| 1.5 | Cross-turn slot dictation | `src.cross_turn_slots` (rule-based, 0 API) | — (xem §3.1) |
| 1.75 | Disfluency injection | `src.disfluency` (rule-based, 0 API) | — (xem §3.2) |
| 2 | Slot identification | `src/synthesis/core.detect_turn_boundaries` (không lệnh riêng) | `docs/stage2-slots.md` |
| 3 | Turn-taking prediction | `src.train_turntaking_hf`, `src.inference_turntaking_hf/llm` | `docs/stage3-prediction.md` |
| 4 | Dialogue generation | `src.synthesis.run` + `run_add_bc` | `docs/stage4-generation.md` |
| 5 | TTS rendering | `tts_render/convert_spoken.py` | `docs/stage5-tts.md` |

### Stage 0 — Corpus prep (`src/prepare_corpus.py`, optional)

* **Mục đích:** Parse 6 corpus gốc thành JSON Stage 1/4 đọc được, khỏi phụ thuộc loader rải rác. Output schema: `{example_id, speakers:["user","assistant"], config, history:[{role,content}], context}` (`src/prepare_corpus.py:1-20`).
* **Loader per dataset:** `iter_interviewer:86`, `iter_multiwoz:127`, `iter_negotiator:152`, `iter_socraticlm:195`, `iter_persuader:232`, `iter_soda:272` — `DATASET_CHOICES:400` (`interviewer multiwoz negotiator persuader socraticlm soda` + `all`).
* **Knobs:** `-d/--dataset`, `--save_dir` (default `results/annotated_dialogues`), `--max_train`, `--max_test` (default 50), `--test_ref_dir`, `--start_index`, `--no_soda_filter` (bỏ LLM competitive filter SODA). `EXPLICIT_SPLIT_DATASETS:51 = {multiwoz, negotiator, soda}` — 3 này có train/test file riêng, 3 còn lại split bằng label/halves.
* **Đóng gói:** `tools/pack_corpus.py` gom dialogues + slot annotations thành JSONL (`DIR2CODE` `socraticlm→TEA, multiwoz→PLN, interviewer→INT, neg/negotiator→NEG, persuader→PER, soda→SOC`), chỉ đọc output Vilex sinh, khỏi đụng text gốc upstream.

### Stage 1 — Spoken-style conversion (`src.speechify_run` — Overview.md:88-102)

* **Mục đích:** Viết lại hội thoại "sạch" (6 nguồn) thành bản nói — filler, ngắt tự nhiên, giọng đời thường. Bản Stage 2-4 đọc.
* **Cơ chế:** Mỗi hội thoại gốc = **1 gọi LLM** (`llm.writer_model`, `stage1_speechify.temperature 0.7`, `src/speechify_core.speechify_full_dialogue`). 6 kịch bản: `interviewer` (INT) + `soda` (SOC) tự tải từ HF Hub (`datasets.load_dataset` trong `src/speechify_run.py:203,210`) khỏi `paths.data_root`; `multiwoz` (PLN) / `negotiator` (NEG) / `socraticlm` (TEA) / `persuader` (PER) cần `paths.data_root` (riêng `persuader` bắt buộc `paths.input_paths.persuader` trỏ file `DailyPersuasion`).
* **3 strategy split trong `src/speechify_run.py:95-127`:** (1) `split_by_file` cho multiwoz/negotiator (train/test file riêng), (2) `split_by_label` cho socraticlm (1 file có label split), (3) `split_in_halves` cho interviewer/persuader/soda (cắt đôi stream). SODA: budget đẩy vào `iter_soda(max_samples)` dừng sớm, khỏi materialize list.
* **Knobs:** `stage1_speechify.max_train_samples/max_test_samples` default 25, **even spacing** (`get_sampled_indices` — `src/speechify_run.py:90` + `src/speechify_datasets.py`) phủ toàn corpus, khỏi lấy prefix; `0` = toàn bộ. `stage1_speechify.max_variants` cho socraticlm/persuader. `run.test_parse` dump source khỏi gọi LLM (smoke test miễn phí, output vào `<paths.results_root>/parsed_source/` tránh resume nhầm `is_converted` check `meta.style=="spoken"`).
* **Output:** `<save_dir>/text_dialogue_<dataset>/{train,test}/*.json` (`split_output_dir` — `src/speechify_run.py:147`), file tồn tại skip (resume). `soda` thêm `_train`/`_test` suffix.
* **Vilex VI:** `run.target_language: vi` (default) + Gemini → LLM sinh bản thoại VI, prompt giữ tiếng Anh (`src/speechify_prompts.py: LANG_DIRECTIVE[vi]`). `pyproject.toml` đảo default, khỏi override trừ khi muốn `en`.

#### Vận hành nội bộ Stage 1 (code-level `src/speechify_core.py`, `speechify_prompts.py`)

* **Prompt:** `SINGLE_STEP_CONVERSION_PROMPT` (27 dòng, `src/speechify_prompts.py:1`) — role "Expert in refining static text", instructions: user dùng disfluency (`um, uh, you know, I, I`), AI tránh disfluency nhưng dùng contractions, reactive khỏi monologue, giữ facts/persona, xóa tên riêng, cấm em-dash `—`. **Rule `...`:** chỉ dùng `...` cho ngập ngừng dài/đúng cảnh (long, noticeable pause), tiết chế, không rải — để Stage 1.75 map thành `[PAUSE]`. Output **ONLY JSON array** `{"role","content"}`. `SINGLE_STEP_CONVERSION_PROMPT_CONCISE` tương tự nhưng rút ngắn. `build_single_step_prompt` ghép `Context:` + `Original Text Dialogue:\n i) role: text`.
* **VI directive:** `LANG_DIRECTIVE["vi"] = "\n\nIMPORTANT: ... MUST translate ... in Vietnamese ... Keep [TAKE_FLOOR] unchanged."` (`src/speechify_core.py:32`) — append system prompt, prompts tiếng Anh, output bắt buộc VI.
* **Structured output dual path** `_generate_dialogue_structured` (`src/speechify_core.py:41`): check `client.base_url` chứa `openai.com` → `client.beta.chat.completions.parse(response_format=DialogueResponse)` (Pydantic `DialogueTurn(role,content)`, `DialogueResponse(turns: List)`); ngược lại Gemini/vLLM → `client.chat.completions.create(response_format={"type":"json_object"})` + `json.loads(raw)` + filter `role in ("user","assistant")` (bỏ system echo Gemini). `concise=True` thì cắt `source_turns[:len//2]` trước convert.
* **Lỗi hay gặp:** Gemini echo `system` turn trong `turns` → fix bằng filter `t.get("role") in ("user","assistant")` (`src/speechify_core.py:122`).

### Stage 1.5 — Cross-turn slot dictation (`src/cross_turn_slots.py`)

* **Mục đích:** Biến slot giá trị (email / SĐT / mã số / code) trong 1 lượt thành **chuỗi đọc từng phần qua nhiều lượt** (dictation), kèm **misspeak-repair** (đọc sai → tự sửa → listener xác nhận). Rule-based, **0 API call**, deterministic theo `stage1_5_cross_turn.seed`.
* **Vị trí:** `data/results_vi` (Stage 1) → `data/results_vi_xt` → `data/results_vi_dis` (Stage 1.75) → Stage 4. **Thứ tự bắt buộc 1.5 trước 1.75.**
* **Phát hiện slot** `_PATTERNS:30` ưu tiên `email > phone > numeric > alnum`: `_RE_EMAIL`, `_RE_PHONE_VN`, `_RE_NUMERIC \d{6,}`, `_RE_ALNUM [A-Z0-9]{5,}`. Lọc: `numeric` cần `len >= stage1_5_cross_turn.min_digits` (6), `alnum` phải có cả chữ và số (`_inject_slots_in_turn:312`). Chỉ xử lý turn `role == "user"` (`_process_dialogue:370`).
* **Chia nhỏ + đọc số** `_segment:149` (`_segment_numeric:90`, `_segment_email:114`, `_segment_alnum:131`) → `_vocalize_segment:160` đánh vần từng chữ số/chữ cái (`_DIGITS_VI/EN`, `_LETTERS_VI/EN`).
* **Dựng lượt** `_build_dictation_turns:254`: `segs < 2` → 1 lượt đọc nguyên giá trị; ngược lại mỗi chunk = 1 lượt dictator + `ack_short` của listener, cuối `ack_final`. Nếu `rng.random() < perror` → chọn `error_idx`, chèn 3 lượt: `wrong` (corrupt) → `self_correct` ("Khoan, ý tôi là {correct}.") → `ack_correct` (`_TEMPLATES:234` VI/EN). Dictator = `rng.choice(["user","assistant"])` khi `stage1_5_cross_turn.roles: both`.
* **Corruption** `_corrupt_segment:210` → `_corrupt_numeric_chunk:175` (đổi chữ số), `_corrupt_letter_group:183`, `_corrupt_alnum_group:191`.
* **Meta & idempotent:** turn nguồn + mỗi turn sinh ra gắn `meta.cross_turn_slot=True`; `data["meta"]["cross_turn_slots_applied"]=True`; skip nếu `meta.cross_turn_slot` đã có hoặc file đích tồn tại.
* **Config:** `paths.results_root → paths.results_xt_root`, `run.datasets/splits/target_language`; `stage1_5_cross_turn.{perror 0.20, seed 42, roles, min_digits 6, min_code_len 5}`; `run.dry_run` (`main:455`).

### Stage 1.75 — Disfluency injection (`src/disfluency.py`)

* **Mục đích:** Bơm nhiễu tự nhiên vào lượt nói theo chuẩn **Switchboard (Meteer et al. 1995)** + mô hình **Shriberg 1996** `p_disfluent(L) = 1 - 0.9453^L` (`_B:42`). Rule-based, **0 API call**.
* **Đơn vị áp dụng:** **per-unit** (`_inject_disfluency_turn`) — tách đoạn `\n\n` → tách unit theo `(?<=[.?!,])\s+` (cả dấu `,`) → mỗi unit Bernoulli riêng. Turn dài nhiều unit ⇒ nhiều injection; unit dài dễ vấp hơn (đúng nghĩa "utterance" của Shriberg, không phải cả turn 80 từ).
* **Scale theo role:** `p = min(1, (1 - 0.9453^L) * scale)`, default `user=0.4`, `assistant=0.25`, **clip [0,1]**. Unit đã có `[PAUSE]` (từ `...`) bị **bỏ hẳn** (không thêm disfluency — tránh pause + filler chồng nhau). Override theo dialogue: `data["meta"]["disfluency_scale"] = {"user":..,"assistant":..}` (dict, ưu tiên hơn config) — `_resolve_scale`.
* **4 loại active:** `fp` (filled pause: `ừm, à, ờ, ừ`), `dm` (discourse marker **clause-initial**: `à mà, thì, mà`), `edit` (`ý tôi là, tức là, hay là, không phải, à mà`), `rep` (lặp span). Inventories **clause-initial only** — đã bỏ `nhé/nhỉ` (tiểu từ cuối câu) và `này/đó` (đại từ chỉ định) vì chèn đầu câu sai ngữ pháp. `FP_INVENTORY`, `DM_INVENTORY`, `EDIT_INVENTORY`.
* **`cor`/`rst` — tạm tắt:** nhánh xử lý + `_find_slot_positions` đã comment, note **implement bằng LLM sau** (COR = thay slot value; RST = viết lại phần tiếp).
* **Chèn:** chọn loại ngẫu nhiên từ `{fp,dm,edit,rep}`; vị trí = **ranh giới clause** `{0} ∪ {sau `, . ? ! ;`}` (`_safe_positions:106`, bỏ fallback từ cuối). Vì tách cả `,`, mỗi unit thường chỉ còn `_safe_positions={0}`. `edit` **A1**: được chèn ở `pos=0` **nếu không phải unit đầu của turn** (`allow_edit_start`), vẫn `pos>0` cho unit đầu; rỗng → fallback fp/dm/rep. FP/DM/EDIT chèn **trước target**; REP lặp span trong clause.
* **Dấu phẩy:** sau mỗi token chèn (FP/DM/EDIT/REP) thêm `,` nếu chưa kết thúc `,` (`_with_comma:83`) để TTS ngắt nhịp.
* **`...` → `[PAUSE]`:** LLM Stage 1 tự viết `...` cho ngắt quãng (source gốc không có); TTS không đọc được `...` nên Stage 1.75 đổi `...`/`…` → `[PAUSE]` (`_normalize_ellipsis`, space-padded) cho **mọi turn cả 2 role**, kể cả turn không inject. Stage 5 map `[PAUSE]` → khoảng lặng ngắn.
* **Meta & idempotent:** turn sửa gắn `meta.disfluency_applied=True`, `meta.disfluency_injections[{type, position, scale, sentence_index, original, result}]`, `meta.pauses` (số `...`→`[PAUSE]`); file-level `meta.disfluency_applied`, `meta.pauses`, `meta.disfluency_scale`. Skip turn đã có flag (`_process_dialogue`), skip file đích tồn tại.
* **Newline:** `\n\n` (ngắt đoạn Stage 1) được **giữ nguyên** khi ghép lại (không còn gotcha mất newline).
* **Config:** `paths.results_xt_root → paths.results_dis_root`, `run.datasets/splits/target_language`; `stage1_75_disfluency.{seed 42, scales.user 0.4, scales.assistant 0.25}`; `run.dry_run`.

### Stage 2 — Slot identification (`src/synthesis/core.detect_turn_boundaries` — Overview.md:104-109)

* Phát hiện **slot** (điểm hành động) *bên trong* lượt user — nơi `floor_taking`/`backchannel`/`silence` xảy ra. Đơn vị quyết định chi tiết hơn biên giữa 2 lượt.
* **Khỏi lệnh riêng:** detector biên câu (heuristic + LLM boundary `src/boundary_annotator.py`) chạy trong Stage 4 (`synthesis/core.py`). Kết quả `boundaries` list per turn, mỗi slot có `word_index`. Stage 3 train trên slot này.
* **Input/Output:** đọc `history` từ Stage 1 JSON, ghi `boundaries` vào turn cho Stage 3/4 dùng.

#### Vận hành nội bộ Stage 2 (`src/synthesis/core.py:175-238`)

* **LLM marker:** `PROMPT_BOUNDARY_DETECTION` (`src/synthesis/prompts.py:305`) dặn LLM chèn `|` sau mỗi clause/sentence (giữ punctuation). `detect_turn_boundaries` (`core.py:175`) gọi `client_boundary.chat.completions.create(temperature 0.0, extra_body=no_thinking)` với `system="You are a linguistic expert..."`. Split `marked_text.split()`, đếm `|` map về `predicted_indices` (đếm token hiện tại). LLM fail → `predicted_indices` rỗng, chỉ dùng heuristic.
* **Heuristic 3 rules per word `i`:**
  * A: `i in predicted_indices` (LLM)
  * B: `re.search(STOP_PUNCTUATION r"[.,?!;]+$", word)` (dấu câu)
  * C: `clean_word(word) in HESITATIONS` — set gồm EN `um, uh, hmm, huh` + VI `ưm, um, à, ừ, ơ, ơi, hử, chà, á, ú` (`core.py:106`)
* **Cuối:** `sorted(set(final_indices))[:-1]` — bỏ boundary cuối lượt (tránh end-of-turn). Trả list `int` word indices.

### Stage 3 — Turn-taking prediction (Overview.md:111-122)

* **Nhiệm vụ:** Mỗi slot Stage 2, dự đoán **phân bố 3 nhãn** `floor_taking`/`backchannel`/`silence` (box `probs` trong `docs/CORPUS.md:44-48`). Điểm bằng **KL divergence** vs phân bố human (`src/inference_turntaking_hf.py:427` `mean_kl_human_pred`).
* **2 predictor hoán đổi:**
  * **HF (khuyên, corpus dùng `tt_mode: hf_classification`):** LoRA token-classification, soft classification trên toàn phân bố nhãn, QLoRA 4-bit (`bitsandbytes` + `accelerate`), base `Qwen/Qwen3-4B`, `src/train_turntaking_hf.py:766` default `--style soft_classification`. Train trên `data-annotations/`, eval cả train/test. Inference cần GPU (`src/inference_turntaking_hf.py`).
  * **LLM:** hỏi chat model xác suất verbalized zero-shot (`src/inference_turntaking_llm.py`), fallback uniform nếu request lỗi. Stage 4 thay predictor này bằng `llm.tt_model` Gemini/Qwen.
* **Flag chính:** `--input_root` (bắt buộc), `--model_name_or_path`, `--output_dir`, `--max_seq_length 1024`, `--load_in_4bit`, `--attn_implementation auto`, `--lora_rank 8`.

#### Vận hành nội bộ Stage 3 (`src/train_turntaking_hf.py`, `src/models/gemma3_token_classification.py`, `src/inference_turntaking_hf.py:25`)

* **Labels 2 bộ từ vựng dễ nhầm:** train `LABELS=["backchannel","interruption","none"]` (`train_turntaking_hf.py:44` + `inference_turntaking_hf.py:25`) ↔ human `backchannel/take_floor/silent` (`docs/CORPUS.md:67`). Map `HUMAN_TO_TRAIN_LABEL` và `train_dist_to_human_dict` (`inference_turntaking_hf.py:31`).
* **Model head:** `GenericForTokenClassification` (Qwen) hoặc `Gemma3ForTokenClassification` (`src/models/gemma3_token_classification.py:15`) — `Gemma3TextModel` + `dropout` + `Linear(hidden_size, 3)`, hook `_merge_multimodal_keys_hook` strip `text_model.` prefix khi load multimodal checkpoint.
* **3 styles training** (`--style`): `soft_classification` (default, KL trên toàn phân bố human, `SoftLabelTokenClassificationTrainer`), `classification` (hard label `label_map backchannel 0/interruption 1/none 2`), `chat` (generation). `LABELS` vs human mapping giữ trong `human_probs_to_train_dist` (`src/turntaking_common.py`).
* **QLoRA:** `BitsAndBytesConfig(load_in_4bit True, nf4, double_quant, dtype bfloat16/float16)` + `LoraConfig r 8 alpha 16 dropout 0 target_modules [q/k/v/o/gate/up/down_proj]` (`train_turntaking_hf.py:836,890`). `prepare_model_for_kbit_training` + `gradient_checkpointing True`.
* **Tokenization HF inference:** `predict_turn_taking_probabilities_hf` (`core.py:309`) — truncate `max_seq_length 1024`, `apply_chat_template` với `context` 4 turns gần nhất (strip `TOKEN_FT/BC`), tìm `word_end_offsets` trong utterance, map `word_idx` → `token_idx` qua `offset_mapping`, single forward `logits[tok_indices]`, `softmax` → probs `floor_taking=prob[1], backchannel=prob[0], silence=prob[2]`. Fail → fallback uniform `silence 1.0`.
* **LLM predictor:** `predict_turn_taking_probabilities` (`core.py:244`) — mỗi boundary `b_idx` tạo `partial_turn = words[:b_idx+1]` + 4 turns context, gọi `PROMPT_VERBALIZED_SCORING` (`prompts.py:331` yêu cầu JSON 3 probs tổng 1), parse `safe_json_parse`, normalize nếu tổng >0, lỗi → fallback `silence`.
* **Đánh giá:** `safe_kl(human,pred)` (`inference_turntaking_hf.py:49`) — `human>0` mask, clip `pred` `1e-12`. `mean_kl_human_pred` log + confusion matrix heatmap TensorBoard (`train_turntaking_hf.py:56`).

### Stage 4 — Dialogue generation (`src.synthesis.run` + `run_add_bc` — Overview.md:124-139)

* **Cơ chế:** Tái sinh hội thoại **từng lượt** (`speechify_turn_by_turn` trong `src/synthesis/core.py`); mỗi slot Stage 2, **sample hành vi** từ predictor rồi chèn, sau đó lọc output nhầm vai (role-confused).
* **3 vai LLM độc lập** (mỗi vai 1 config key + 1 backend `src/llm_client.py`):
  1. **Writer** `llm.writer_model` (Vilex dùng `gemini-3.6-flash`): sinh content từng lượt + quyết định kết thúc.
  2. **Boundary detector** `llm.boundary_model`: đánh dấu biên câu → candidate slot.
  3. **Turn-taking predictor** `llm.tt_model` (chat) hoặc `stage4_synthesis.hf_model_name_or_path` (LoRA): chấm slot. Vilex VI dùng `gemini-3.6-flash` cho cả 3 vai (khỏi host model).
* **Backchannel:** Quyết định `backchannel` ghi vào `turn["history"]` metadata mỗi slot (`decision`, `word_index`, `probs`), chỉ `[TAKE_FLOOR]` xuất hiện trong transcript text; nội dung backchannel điền **sau** bởi `run_add_bc` (endpoint thứ 4, `llm.bc_base_url` default port `8008`), ghi vào `item["content"]` khỏi đổi transcript. `_normalize_ws` sau sinh.
* **Input/Output:** `paths.results_dis_root` (layout `text_dialogue_<dataset>/<split>/*.json` từ Stage 1.75) → `paths.synthesis_root`; resume nếu tồn tại. `stage4_synthesis.max_dialogues` (config 15), `stage4_synthesis.max_turns` (0 = auto) giới hạn chi phí. `ensure_source_turns` fix bỏ qua role != `user`/`assistant`.
* **Vilex VI:** `run.target_language: vi` (default) + Gemini cho cả 3 vai; `run_vi_pipeline.sh` khỏi override explicit.

#### Vận hành nội bộ Stage 4 (`src/synthesis/core.py:517-743`, `prompts.py:47-416`)

* **Vòng lặp chính** `speechify_turn_by_turn` (`core.py:626`): `spoken_turns=[]`, `current_role=source_turns[0][0]`, `GLOBAL_RNG=Random(42)`, `pbar(max_turns)`. Mỗi iteration: (1) `generate_raw_content_turn` → `raw_content`, (2) nếu `user` thì `detect_turn_boundaries` → `predict_*_hf/llm` → `insert_action_tokens...` → `final_content` + `history_meta`, (3) append `(role, final_content, history_meta)`, đổi `current_role`, `judge_done` mỗi `stop_check_every` sau `min_turns_before_stop 2` dừng sớm khi coverage đủ hoặc stalled.
* **Sinh raw turn** `generate_raw_content_turn` (`core.py:517`): chọn `sys_prompt = ROLE_USER_TURN_PROMPT` (user: brief, disfluency `um/uh`, self-repair) hoặc `ROLE_AI_TURN_PROMPT` (assistant: concise, no disfluency) (`prompts.py:47,90`). Append `LANG_DIRECTIVE[vi]` nếu `target_language=="vi"`. Build user prompt qua `build_rewrite_user_prompt` (`prompts.py:375`): nếu `history_turns` có thì dùng `REWRITE_WITH_HISTORY_TEMPLATE` (`full_source_block` + `history_block` via `dialog_to_block`), nếu không thì `REWRITE_FIRST_TURN_TEMPLATE`. Gọi `client.chat.completions.create(temperature user 0.7 / ai 0.2, extra_body=no_thinking)`, lấy `_strip_think_tags`, `sanitize_utterance`, check `has_prefix_contamination`. Contamination → retry tối đa `max_format_retries 3` với corrective reminder ("Regenerate SAME utterance as raw text ONLY...") và tăng temp `+0.1*attempt`. Cuối trả `normalize_ws(clean)`.
* **Sanitizer** `sanitize_utterance` (`core.py:53`): strip `<think>`, code fence ```, peel 4 lần loop: `_PREFIX_RX` (`index + role:`), `_ROLE_ONLY_RX` (`user:`), `_INDEX_ONLY_RX` (`10)`), quotes `"...'`, sau đó `em-dash→,`, `re.sub \s+`.
* **Chèn token** `insert_action_tokens_from_llm_annotations` (`core.py`): sort `paired = zip(boundary_indices, b_dists)`, duyệt `words`/`boundary_ptr`. Guards `CONFIG = length_guard_start 0, interruption_guard_start 3, length_guard_gap 4`.
  * **Floor_taking — 1 quyết định/turn:** `_ft_candidate` chọn boundary có `p_ft` **max** với `idx >= interruption_guard_start` (boundary kết thúc `. ? !` **được tính**, không loại nữa); sample **1 Bernoulli** với raw `p_ft`; trúng thì chèn `TOKEN_FT="[TAKE_FLOOR]"` tại boundary đó (tối đa 1/turn). Vì chỉ 1 lần/turn, `floor_taking_triggered` chỉ để chặn trùng.
  * **Backchannel:** sample mỗi boundary qua `_sample_action` (cumulative rng), chỉ chèn `TOKEN_BC="[BACKCHANNEL]"` khi `last_bc_pos is None or i-last_bc_pos >= length_guard_gap`.
  * Ghi `action_history` `{word_index, probs, decision, inserted_token}`. Cuối: nếu có `TOKEN_FT` thì **giữ `[BACKCHANNEL]` trước FT**, bỏ BC sau FT; else `out.replace(TOKEN_BC,"")`. Trả `(out, action_history)`.
* **Prompt TT** `PROMPT_VERBALIZED_SCORING` (`prompts.py`): dặn "user còn đang nói, đừng coi hết câu là hết lượt", **nhưng** assistant **MAY** take floor khi có lý do (nêu xong ý / hỏi trực tiếp / bất đồng cần đính chính / user lan man). Thứ tự JSON `silence, floor_taking, backchannel`.
* **History strip:** `process_history` (`core.py:157`) xóa `TOKEN_BC`, cắt tại `TOKEN_FT` trước khi đưa vào LLM tiếp theo (tránh leak token).
* **Judge dừng:** `judge_done` (`core.py:603`) gọi `DONE_JUDGE_PROMPT` (`prompts.py:265` — rules coverage vs stalled, output JSON `done/stop/missing`), parse `safe_json_parse`, trả `done is True`.
* **HF ưu tiên:** Trong `speechify_turn_by_turn`, nếu `hf_model+tokenizer` có thì dùng HF, else cần `client_tt` (raise nếu thiếu). `src/synthesis/run.py:191` đảo default vi, `run_add_bc.py:86` dùng prompt `SYSTEM_PROMPT_QWEN_VI` riêng (1-3 từ VI lowercase).

### Stage 5 — TTS rendering (`tts_render/convert_spoken.py` — Overview.md:141-171 + 332-431)

* **Mục tiêu:** Mỗi dialogue JSON → `stage5.num_variants` bản audio **2 kênh** (ch0 assistant, ch1 user, 24kHz 16-bit stereo, swap để assistant ở 0 cho `personaplex-finetune`).
* **2 backends:**
  * **Chatterbox (EN legacy):** assistant `assistant_en.wav` cố định (LibriSpeech `train-clean-100`), user sample từ `tts_render/librispeech_samples/` (12 speaker) hoặc full LibriSpeech (code `librispeech_root`, không expose config), backchannel render kém (paper dùng ElevenLabs). Cần NeMo `nemo-text-processing` normalize + `pynini`/`OpenFst`, Python `>=3.11,<3.12`.
  * **OmniVoice (VI default):** voice-design qua `instruct` string (`male, british accent` / `female, american accent` — `tts_render/convert_spoken.py:45`), khỏi audio prompt, khỏi NeMo; **Qwen3-ForcedAligner** (`stage5_2a_align.aligner`, `Qwen/Qwen3-ForcedAligner-0.6B-hf`) neo backchannel.
* **Cơ chế `main_process`:**
  * Mỗi câu TTS → `generate_audio()` (OmniVoice: có `ref_audio/ref_text` thì clone, ngược lại `instruct`; Chatterbox: `s3tokenizer`).
  * **Silero VAD** cắt lặng đầu/cuối mỗi câu/utterance (`get_speech_timestamps`) để nối sát.
  * **Qwen3 forced aligner** (`_align_words_once`, transformers native) forced-align host audio với transcript; `stage5_2a_align.device` (mặc định `cuda`).
  * Variant có `dialogue.wav`+`meta.json` (và `alignment_*.json` khi `stage5_2b_assemble.audio.save_align_json`) thì skip (resume). `stage5.seed` + `crc32(fpath)` cho deterministic voice pick.
* **Output per variant `var00/`:** `dialogues/dialogue.wav` (stereo), `user.wav`/`assistant.wav` mono (giữ timestamps, map qua `total_speech_meta["speakers"]`), `utterances/`, `backchannels/`, `meta.json` (+`user_wav`/`assistant_wav`).
* **Voice-clone pool** `paths.voice_clone_pool` (Overview.md:161-166): pool wav+txt sidecar, clamp 10s `stage5_2a_align.max_prompt_secs`, resample 24k mono, cache 1 lần, pick 2 giọng distinct per dialogue.
* **2 tầng timing chi tiết (tts_render/convert_spoken.py:582,922):**
  * **Tầng 1 `aggregate_speech` (refactored):** 3 nhánh, **GAP/PAUSE sampled** thay vì hằng 0.16s.
    * Đổi speaker + `uttr_type==interrupt` ([TAKE_FLOOR]) → cross-fade overlap `N(0.45, 0.05)` clip `[0, USER_INTERRUPT_OVERLAP_SEC=0.64]`, label `timing="overlap"`.
    * Đổi speaker, no interrupt, có leading BC → GAP sampled, label `timing="gap"`.
    * Đổi speaker, no interrupt, no leading BC → `USER_INTERRUPT_PROB=0.5`; BC lead → gap, else overlap.
    * Cùng speaker + `boundary=="turn"` (`is_last_text`) → **PAUSE** sampled, label `timing="pause"`.
    * Cùng speaker + câu giữa lượt (`boundary=="sentence"`) → concat 0, label `timing="none"`.
    * Sample helpers `_sample_gap(turn_dur_sec)` (Exponential scale `0.7067`, clip `[0.003, 2.50]`, scale factor `(dur/3)^0.25 ∈ [0.7, 1.5]`), `_sample_pause()` (scale `0.6780`, clip `[0.001, 4.00]`). Mỗi `speech_meta[i]` ghi `{timing, duration_sec}`; `meta.json` ghi `timing_config.{gap,pause,overlap_max_sec,user_interrupt_prob}` (thay `turn_gap_sec` cũ).
  * **Tầng 2 neo backchannel:** Qwen3 aligner lấy `all_words[word_idx]["end"]*TARGET_SR`, `place_backchannel` + cursor + `generate_delay(mode="backchannel")` delay 0. Chỉ phục vụ backchannel, khỏi can thiệp gap tầng 1.
  * **`[PAUSE]` token (Stage 1.75):** tách `tts_text` theo `[PAUSE]` → synth từng đoạn → chèn `_noise_floor_segment(_sample_intra_pause()*TARGET_SR)` giữa các đoạn (`timing="pause_token"`). `_sample_intra_pause()` Exponential scale `0.30`, clip `[0.10, 1.00]`. Fallback: `sanitize_text` cũng đổi `...`/`…` sót → `[PAUSE]`; token nằm trong `_INTENTIONAL_BRACKET_TOKENS`.
* **Hậu xử lý âm thanh:** bỏ normalize per-câu, gap white-noise, LUFS `-23` toàn track (xem §7).

#### Vận hành nội bộ Stage 5 (`tts_render/convert_spoken.py:502-1130`)

* **Parse input** `convert_original_to_expected` (`convert_spoken.py:502`): duyệt `history` Stage 4, bỏ role != user/assistant, `reconstruct_text_with_tokens` + `sanitize_text`, đọc `msg["history"]` decisions sort `word_index`, lấy `bc_contents_list` (missing thì `random.choice DEFAULT_BC_CANDIDATES_VI` cho VI), detect `TAKE_FLOOR_TOKEN in text` → `_cut_at_take_floor_plus_one_word`, `_split_backchannel_segments` (tách `texts` thành `segments` per backchannel + `triggers` cho interrupt), `split_sentences(texts, TTS_LANGUAGE)` (`nltk sent_tokenize` theo lang), đánh `uttr_type backchannel/interrupt/None`, `triggers` map `target_spk → start_idx` để đánh `interrupt` cho lượt sau.
* **Vòng lặp TTS `main_process` (`convert_spoken.py:784`):** `spk_offset` theo init speaker, `cumulative_prompt_paths` (OmniVoice: instruct strings; Chatterbox: wav copy), `speaker_ref[2]` (OmniVoice clone), `bc_cache` LRU, `bc_queue[2]`, `tts_texts[2]` buffer, `accumulated_flag[2]` (đợi backchannel xong mới flush), `interrupt_flag`. Duyệt `dialogue["utterances_with_bc"]` + `texts_styled`: nếu `cleaned_tts_text` rỗng/`-`/`...` → `isUttered False`; `prev_uttr_type==backchannel` thì append; `next_uttr_type==backchannel` thì `accumulated_flag True` và `continue`; else flush `tts_texts[curr_idx]` qua `_strip_paralinguistic` + `split_sentences` → mỗi `sentence_` gọi `generate_audio` (OmniVoice: `ref_audio/ref_text` nếu `speaker_ref` có else `instruct`; Chatterbox: `model.generate(text, audio_prompt_path)`), cache backchannel `bc_cache.get/put`, xử lý `VAD` cắt lặng, xử lý `empty audio` → silence 0.2s.
* **Voice-clone load** `_load_voice_pool` (`convert_spoken.py:711`): `glob pool_dir/*.wav` + sidecar `<name>.txt`, `_load_omnivoice_ref_audio` mean mono + resample 24k + clamp 10s, cần >=2 items else `SystemExit`. `generate_audio` trả `[1,T]` float tensor, OmniVoice `wav.dim()==1 → unsqueeze(0)`.
* **Delay & place:** `generate_delay` (`convert_spoken.py:590`) normal `loc 0.38 scale 0.2` (backchannel `0`, bc_mhm `0.13±0.02`), clip `0..pad_size`, trả `int(delay*TARGET_SR)`. `place_backchannel` (`convert_spoken.py:613`) grow cả 2 tracks nếu `start + len(bc)` vượt quá, `F.pad` white-noise gap trước đó đã tính.
* **Resume & determinism:** `dialogue_rng = random.Random(stage5.seed + zlib.crc32(fpath.name))` pick 2 voices, skip nếu `dialogue.wav+meta.json` tồn tại. `stage5_2a_align.device` (mặc định `cuda`) truyền cho Qwen3 forced aligner.

---

## 4. Đường dẫn tiếng Việt (Vilex default)

* **Stages 1/1.5/1.75/4/add_bc:** `run.target_language: vi` (default, `en` explicit) + `llm.*_model: gemini-3.6-flash` (hoặc `lite` khi hết quota). Khỏi host model. Stage 1.5/1.75 rule-based, khỏi tốn quota — chạy sau Stage 1, trước Stage 4 (`data/results_vi → data/results_vi_xt → data/results_vi_dis`).
* **Stage 5:** `stage5_1b_render.backend: omnivoice`, `stage5.language: vi` (default). Khỏi `prompt_wavs`, khỏi NeMo, khỏi pynini.
* **Voice-clone pool** `paths.voice_clone_pool`: mỗi `<name>.wav` + `<name>.txt` transcript chính xác, clamp 10s `stage5_2a_align.max_prompt_secs`, resample 24k mono, cache 1 lần. Mỗi dialogue pick 2 giọng distinct qua `dialogue_rng = stage5.seed + crc32(fpath.name)` deterministic resume-safe (`Overview.md:396`).
* **Instruct fallback** (khi khỏi dùng pool): `stage5.voice.user_instruct` / `assistant_instruct` (`male, northern accent` / `female, gentle`).
* **Backchannel VI:** `src/synthesis/run_add_bc.py:86` prompt `SYSTEM_PROMPT_QWEN_VI` (ưm, à, ừ, vâng...).

**Credential (Overview.md:237-242):**
* Khuyên Vertex AI: `export GEMINI_CREDENTIALS="$(pwd)/gen-lang-client....json"` + `GEMINI_LOCATION=global` (quota theo project, khỏi giới hạn 20 req/ngày).
* Fallback API key: `export GEMINI_API_KEY=...` (Generative Language API, free-tier 20 req/ngày/model, cạn khi test → dùng `gemini-3.6-flash-lite`).
* `run_vi_pipeline.sh` báo lỗi rõ nếu thiếu cả 2.

### 4.1 Build voice-clone pool (`tools/build_voice_clone_pool.py`)

* **Luồng:** Kaggle `tuannguyenvananh/vi-common-voice` (`ensure_kaggle_cache:89`, cache `data/vi-common-voice`) → gom clip theo `client_id` từ TSV (`load_client_map:134`) → **1 clip/speaker** (clip dài nhất với `--min-sec < dur`, default 4s) → clamp `--max-sec` (default 10s = `MAX_PROMPT_SECS`) → convert WAV 16kHz/16-bit/mono ≤25MB (`convert_clip:173`) → **transcript** qua Whisper ASR (`transcribe:69`) → ghi `voice_clone/<stem>.wav` + `<stem>.txt` (`make_stem:224`, prefix `cvvi_`).
* **ASR service:** OpenAI-compat Whisper (Qualcomm NPU), spec `ExternalAsvFileWhisper.openapi.yaml`, server mặc định port **9670**, endpoint `POST /v1/asr/transcribe/file`, `GET /health` (status/server/port). WAV 16k 16-bit mono ≤25MB hỗ trợ EN+VI.
* **Chất lượng:** `--require-lang vi` (skip nếu `detected_text_language` khác), `--min-conf 0.85`, `--timeout 120`. Transcript sai = clone giọng sai → đừng hạ 2 ngưỡng này.
* **Determinism + resume:** `--seed 7` shuffle, manifest `.build_voice_clone_pool.json` (`load_manifest:212`), `existing_pairs:200` — speaker đã lấy khỏi bị pick lại; đủ `--count` (default 100) thì thoát sớm. Pool hiện có **85 pairs** `cvvi_*`.
* **Yêu cầu:** `ffmpeg` trên PATH + `soundfile`. Có thể `--clips-dir` (repeatable) + `--tsv` để khỏi tải lại.
* **Ngưỡng Stage 5:** `_load_voice_pool` cần **≥2 pairs** wav+txt,否则 `SystemExit`.

---

## 5. Môi trường & LLM routing

* **2 env tách:** Stages1-4 `.venv` (Python 3.10+, `transformers>=4.53`) vs Stage5 `.venv-tts` (Python 3.11, `transformers==4.46.3` từ `vilex/tts/chatterbox/pyproject.toml:17`). `constraints.txt:5` chỉ cho Stages1-4.
* **LLM routing** `src/llm_client.py`: tên model quyết backend — `gpt-/o1/o3/o4` → `api.openai.com` (`OPENAI_API_KEY`), `gemini` → `Google GenAI OpenAI-compat`, còn lại → `base_url` localhost:8000 (`EMPTY`). Stage4 dùng `llm.base_url`; Stage 4b có `llm.bc_base_url` riêng.
* **Chạy:** từ repo root, `python -m src.<module>` cho Stages1-4, `python tts_render/convert_spoken.py` cho Stage5 (interpreter khác).
* **Attention backend** `src/hf_attn.py`: `resolve_attn_implementation:26` — `CHOICES = auto/flash_attention_2/sdpa/eager`. Stage 3 scripts từng hardcode `flash_attention_2` → clean install crash `ImportError: FlashAttention2 has been toggled on...` vì `flash-attn` không có universal manylinux wheel (compile từ source theo CUDA toolkit) nên không thể là hard dep. Giờ default `auto`: FA2 nếu importable,否则 SDPA.
* **CI** `.github/workflows/ci.yml` + `requirements-dev.txt:1` → `pytest/ruff/black`. Gate: `ruff --select=E9,F`, `black --check`, `pytest -q` (**170 passed, 1 skipped** — xem §9; con số 44/80 cũ đã stale).

---

## 6. Dữ liệu & scenario codes

* 6 scenarios: `socraticlm` (TEA), `multiwoz` (PLN), `interviewer` (INT), `negotiator` (NEG), `persuader` (PER), `soda` (SOC) (`AGENTS.md:81`).
* **Quirk case:** `SOC` = soda, `soc` = socraticlm — khỏi casefold, luôn qua `CODE2DIR` (`tools/unpack_corpus.py`).
* **Data local:** `docs/CORPUS.md:1` — HF corpus tạm gỡ, pipeline đọc `data-annotations/` / `data-dialogues/` (`text_dialogue_<dataset>/<split>/*.json`). Schema `history[].segments` đan `full_content` và slot `word_index/probs/decision/inserted_token`; annotations dùng `silent/take_floor` vs dialogues `silence/floor_taking`.
* **Counts (Overview.md:188):** dialogues 5999 (125,137 slots), annotations 420 (120 train +300 test). Generator Qwen3.5-122B-A10B, audio Chatterbox/OmniVoice.
* **Giấy phép:** Code Apache-2.0 (`LICENSE:189` Vilex Authors based on DuplexGen), data kế thừa upstream (`docs/DATA_LICENSES.md`), audio Chatterbox MIT vendored `vilex/tts/chatterbox/`.
* **Switchboard path (real-corpus branch):** `src/swbd_parse.py` parse `swb1_dialogact_annot` (`utt_re` bắt `da spkturn utt#: text`), `parse_file:49`, `clean_repairs_keep_both:26`, `clean_markup:32`. `extract_typed_boundaries_from_raw:100` + `merge_boundary_candidates_and_types:134` + `compute_candidates_with_gpt_and_heuristics:198` (LLM + heuristic). `convert_to_streaming_dialogue_json:299`: backchannel (`da` bắt đầu `b` hoặc `da == "%"` và `end != "-/"`) gắn `[BACKCHANNEL]` vào lượt cùng speaker, `end == "-/"` → append `[TAKE_FLOOR]`, gộp lượt liên tiếp cùng speaker, speaker `A→assistant`, `B→user`. `src/swbd_convert.py:41` chạy đa tiến trình `ProcessPoolExecutor` qua OpenAI client, in/out `swb1_dialogact_annot/` → `results/swb1_dialogact_annot`.
* **Common Voice VI (local):** `data/vi-common-voice/` = cache Kaggle clip + TSV, nguồn build `voice_clone/` (§4.1).
* **Kaggle:** `kaggle/vilex_kaggle.ipynb` full pipeline **2 phase bắt buộc** (1 interpreter: Stages1-4 `transformers>=4.53` xung đột Stage5 OmniVoice → restart kernel giữa phase); `kaggle/vilex_stage5_kaggle.ipynb` chỉ Stage 5 (Internet ON, GPU T4 x2, input = zip `data/vi_tt_bc` upload làm Dataset, `data/`+`outputs/` trong `.gitignore`). Notebook Stage-5 cài **lean** (bỏ `nemo-text-processing`, chỉ giữ nhánh OmniVoice/VI), tự dò `bc_root`/`voice_clone_pool`, `cp config_example.yaml config.yaml`, mặc định `save_align_json: true`; resume skip variant đã có `dialogue.wav`+`meta.json`(+`alignment_*.json`).

---

## 7. Kinh nghiệm xương máu — các fix đã verify (Overview.md:284-431 + Session.md extract)

**Verify end-to-end 1 sample `interviewer/work_0000` Stage1→5 trên CPU env `vilex` ~10ph (Overview.md:287).**

1. **`src/speechify_core.py` + `src/synthesis/run.py:68`:** lọc bỏ turn role != `user`/`assistant` (system echo) — `continue` thay raise.
2. **`src/gemini_client.py`:** fix `_create` thiếu `return`, thêm backoff 429 + global rate limiter `_MIN_INTERVAL=13.0` (5 req/phút) + parse `RetryInfo delay`.
3. **`tts_render/convert_spoken.py` OmniVoice 0.2.1:** `OmniVoice.from_pretrained("k2-fsa/OmniVoice")` khỏi nhận `device=`; thêm `stage5_1b_render.device: cpu` default; Qwen3 aligner truyền `stage5_2a_align.device` (trước hardcode `cuda` crash `Torch not compiled with CUDA enabled`).
4. **VAD trim rỗng:** câu ngắn `"Ừm,"`/`"À,"` VAD `end≈0` → `speech_[:, :0]` rỗng → `normalize_audio` crash `max()`. Guard chỉ cắt khi `0 < end < len`, `normalize_audio` return nếu rỗng, `generate_audio` trả silence 0.2s + warning.
5. **Paralinguistic tag:** `[laughter]/[question-en]/...` bị checkpoint 0.2.1 đọc thành tiếng nên mặc định bị strip trước `generate()` (`_strip_paralinguistic`). **`stage5.tags.render` (default false):** khi bật, giữ 13 tag hợp lệ (`RENDERABLE_TAGS`) cho OmniVoice render thành âm thanh; strip tag lạ (ngoài whitelist, trừ `[PAUSE]`). Align/word-count + `ref_text` luôn strip. Tag có gạch nối (`[question-ah]`) được bảo vệ khỏi `replace("-"," ")` bởi `_replace_dashes_outside_brackets`. Text chỉ còn tag (tag-only) → phát 0.2s silence thay vì `torch.cat([])` crash.
6. **Log ồn:** `TQDM_DISABLE=1/TRANSFORMERS_SILENT/HF_HUB_VERBOSITY=error` trước import + `logging.Filter` drop `HTTP Request`/`unauthenticated` + `httpx/huggingface_hub→ERROR` + `tqdm.disable` + `warnings.filterwarnings`.
7. **Voice-consistency:** lưu `ref_audio/ref_text` câu đầu mỗi speaker (10s) truyền lại OmniVoice; tách `user.wav`/`assistant.wav` từ 2 kênh `dialogue.wav` giữ timestamps (map qua `total_speech_meta["speakers"]` để đúng người khi `--swap`).
8. **Hậu xử lý âm thanh 2026-08-27:** bỏ normalize-per-câu (tránh clip overlap), gap = white-noise `0.006`, normalize toàn track LUFS `-23` via `pyloudnorm` (fallback peak nếu thiếu), thêm `pyloudnorm` vào `requirements-stage5-vi.txt:35`.
9. **`\n\n` trong JSON Stage 1:** Stage 1 sinh assistant turn nhiều đoạn → LF thật, `json.dump` escape thành `\n\n`. Stage 1.75 **giữ nguyên** `\n\n` khi tái tạo (tách đoạn → ghép `\n\n`); `meta.disfluency_injections[].original` là snapshot pre-inject. Stage 4 (`split()`) + Stage 5 (`sanitize_text`, `normalize_ws`) gom whitespace tiếp → TTS an toàn.
10. **Prompt leak Stage 1:** có file (vd `data/results_vi/.../work_0266.json`) turn `user` chứa nguyên system prompt + checklist ("Translate formal text dialogue to..."). Bộ lọc role ở `speechify_core.py` không chặn được vì turn mang role hợp lệ. Cần lọc/heuristic riêng nếu sinh corpus lớn.
11. **flash-attn:** khỏi thêm vào `requirements.txt`. Dùng `src/hf_attn.py` default `auto` (FA2 nếu có,否则 SDPA) — hardcode `flash_attention_2` làm clean install crash ngay lúc load model (§5).

---

## 8. Lệnh chạy tham khảo nhanh

```bash
# VI default (Vilex) — mọi stage đọc config.yaml, chỉ còn --config PATH
conda activate vilex-omnivoice  # hoặc vilex cho Stages1-4
export GEMINI_CREDENTIALS="$(pwd)/gen-lang-client....json"  # Vertex khuyên
cp config_example.yaml config.yaml   # nếu chưa có; sửa run.datasets/splits + paths.*
# Stage1 (llm.writer_model / stage1_speechify.* trong config)
python -m src.speechify_run
# Stage1.5 (rule-based, 0 API; stage1_5_cross_turn.*)
python -m src.cross_turn_slots
# Stage1.75 (rule-based, 0 API; stage1_75_disfluency.*)
python -m src.disfluency
# Stage4 (đọc paths.results_dis_root, ghi paths.synthesis_root)
python -m src.synthesis.run
python -m src.synthesis.run_add_bc
# Stage5 (stage5_1b_render.backend: omnivoice, language: vi)
python tts_render/convert_spoken.py
# Full 1 dialogue
./run_vi_pipeline.sh  # sửa stage5_1b_render.device nếu có GPU, MAX_* để trống = full 43 dialogues

# Batch 5 datasets (Stage1→1.5→1.75→4→add_bc), log trực tiếp ra terminal
MAX_TRAIN=30 MAX_DIALOGUES=30 MAX_WORKERS=16 ./run_local_stages1-4.sh
#   env (đều optional, trống = lấy từ config.yaml): PY, STAGE5_PY, MODEL, DATASETS
#   pacing/config: llm.gemini_min_interval (default 1.0) + llm.gemini_location trong config.yaml; env GEMINI_MIN_INTERVAL/GEMINI_LOCATION vẫn thắng nếu set
#   output: mỗi stage in header/OK-FAIL ra terminal; không còn file log — expect 150 files data/vi_tt_bc khi đủ 5×30
# Stage0 corpus prep (optional; vẫn argparse)
python -m src.prepare_corpus -d all --save_dir results/annotated_dialogues --max_train 1000 --max_test 50
# Build voice pool (argparse tool; cần ASR server + ffmpeg + soundfile)
python tools/build_voice_clone_pool.py --base-url http://HOST:9670 --out-dir voice_clone --count 100 --min-sec 4 --max-sec 10

# EN legacy (đổi config qua env VILEX_*, không flag)
export VILEX_RUN__TARGET_LANGUAGE=en VILEX_STAGE5_1B_RENDER__BACKEND=chatterbox VILEX_STAGE5__LANGUAGE=en
python -m src.speechify_run
python tts_render/convert_spoken.py
```

**Output Stage5 per variant `var00/`:** `dialogue.wav` (stereo ch0 assistant), `user.wav`/`assistant.wav` mono, `utterances/`, `backchannels/`, `meta.json` (+speakers, user_wav...), và `alignment_user.json`/`alignment_assistant.json` khi `stage5_2b_assemble.audio.save_align_json: true`.

---

## 9. Kiểm thử & verify

* `python -m pytest -q` — **170 passed, 1 skipped** (Stage 5 timing test cần torch; con số 44/80 cũ đã stale; loại trừ `vilex/tts/chatterbox` cho ruff/black).
* **Python <3.11:** `tools/test_docs.py` 2 test fail `ModuleNotFoundError: No module named 'tomllib'` (đọc `pyproject.toml`/`REUSE.toml`). Khỏi phải bug — chạy pytest bằng interpreter ≥3.11.
* **Test files (đã dời vào `tests/`):** `tests/test_docs.py`, `tests/test_env_consistency.py`, `tests/test_pack_corpus.py`, `tests/test_cross_turn_slots.py` (segmentation/vocalization/corruption/templates/dictation/injection/perror), `tests/test_disfluency.py` (Shriberg model, slot positions, insertion, inventory), `tests/test_label_taxonomy.py`, `tests/test_pipeline_contracts.py`, `tests/test_swbd_importable.py`, `tests/test_turntaking_common.py`. Stage 1.5/1.75 tests chạy **0 API call**.
* `grep -rn duplexgen` chỉ còn `README.md:122` bibkey + `Overview.md:211` upstream `duplexgen/personaplex-finetune` — sạch filesystem.
* Smoke: `python -m py_compile tts_render/convert_spoken.py && grep -c _fade_in_out` + chạy 1 dialogue vi với `GEMINI_API_KEY`/`CREDENTIALS` và check `data/vi_audio/.../var00/dialogue.wav` + LUFS `-23`.

---

## 10. Handover checklist cho AI tiếp theo

* [ ] Đọc `KNOWLEDGE.md` trước, khỏi đọc `Session.md` thô (8844 dòng) trừ debug sâu.
* [ ] Sửa code: tôn trọng default `vi+omnivoice` (config `run.target_language`, `stage5_1b_render.backend/language`, `speechify_run.py:360`, `run.py:191`, `convert_spoken.py:42`), EN qua override `VILEX_RUN__TARGET_LANGUAGE=en VILEX_STAGE5_1B_RENDER__BACKEND=chatterbox VILEX_STAGE5__LANGUAGE=en`.
* [ ] Khỏi gộp env: giữ `requirements.txt` vs `requirements-stage5*.txt` tách (transformers conflict).
* [ ] Trước khi đụng Stage5: check `vilex/tts/chatterbox` path (đã rename), `pyloudnorm` installed, `voice_clone/*.wav+txt` đủ >=2 file.
* [ ] Stage 1.5 trước 1.75, Stage 4 đọc `data/results_vi_dis` (khỏi đưa `data/results_vi` thẳng vào Stage 4).
* [ ] Idempotent: khỏi xóa `meta.cross_turn_slot` / `meta.disfluency_applied` — 2 flag này chặn inject lặp; file đích tồn tại thì skip.
* [ ] Build voice pool: `curl $ASR/health` trước, cần `ffmpeg` + `soundfile`, giữ `--require-lang vi --min-conf 0.85` (transcript sai = clone sai).
* [ ] Khỏi hardcode `flash_attention_2`; dùng `src/hf_attn.py` default `auto`.
* [ ] Khỏi commit secret `gen-lang-client*.json` / `*sa.json` (gitignore `.gitignore:36`), khỏi git ops nếu RULE cấm.
* [ ] Verify sau sửa: `pytest -q` (170 passed, 1 skipped), `grep duplexgen` sạch, `py_compile` entry points (`src.speechify_run`, `src.cross_turn_slots`, `src.disfluency`, `src.synthesis.run`, `tts_render/convert_spoken.py`).

*Appendix nguồn:* Overview.md:1-8, 19-54, 56-65, 88-172, 284-431 + Session.md fix 2026-08-26/27 (VAD, aligner device, paralinguistic, LUFS, voice-clone deterministic) + code 2026-09 (Stage 1.5/1.75, build_voice_clone_pool + Whisper ASR OpenAPI, prepare_corpus, hf_attn, swbd_parse/convert, Kaggle 2-phase).
