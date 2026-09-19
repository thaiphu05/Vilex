# Stage 5 internals — models, constants & per-turn pipeline

Technical inventory of the TTS stage. User-facing usage lives in
[stage5-tts](stage5-tts.md); this file maps **which model does what** and how a
single turn becomes audio. Source: `tts_render/convert_spoken.py`
(entry `main:1257`, per-dialogue `main_process:847`).

Stage 5 makes **0 LLM calls**.

## Input → Output

```
Stage 4 JSON  data/vi_tt_bc/text_dialogue_<ds>/<split>/*.json
   history[] + [BACKCHANNEL]/[TAKE_FLOOR] + word_index decisions
      │  should_drop_dialogue()  (filter)
      │  convert_original_to_expected()  (schema)
      ▼
   main_process()  (per turn TTS + backchannel placement)
      │  aggregate_speech()  (timing: gap/pause/overlap)
      │  flip channels (assistant → ch0)
      ▼
<save_dir>/text_dialogue_<ds>/<split>/<id>/var<NN>/
  dialogues/dialogue.wav    stereo 24 kHz, ch0=assistant ch1=user
  assistant.wav / user.wav  mono split, timing preserved
  utterances/00.wav …       one 2-channel file per utterance, pre-mix
  backchannels/05_0.wav …   <utterance>_<slot>, isolated backchannel
  meta.json                 speech_meta + timing_config + voice ids
```

## Model inventory

| Component | Load / call site | Role |
|---|---|---|
| **OmniVoice** `k2-fsa/OmniVoice` | `from_pretrained("k2-fsa/OmniVoice")` `:1394`; `generate(...)` `:805-822` | TTS main path (VI default). `speed=1.3`, `language="Vietnamese"`, `normalize_text=True`. Turn 1 uses voice-design `instruct`; later turns voice-clone `ref_audio/ref_text`. Paralinguistic tags stripped before `generate()` by default; keep the 13 supported tags with `stage5.tags.render: true` (unknown tags always stripped). |
| **Chatterbox** `vilex/tts/chatterbox` | `ChatterboxTurboTTS.from_pretrained(device="cuda")` `:1406`; `generate(text, audio_prompt_path)` `:831` | TTS legacy EN path. Cumulative prompt `cumulative_*.wav` (max 10 s) grown per non-BC utterance. |
| **Silero VAD** `silero_vad.load_silero_vad` | `get_speech_timestamps(threshold=0.3)` `:24, :993, :1043` | Silence trim per sentence + per utterance; guards against 0-length. |
| **whisperx aligner** (wav2vec2, per language) | `_load_forced_aligner()` → `whisperx.load_align_model`; `_align_words_once()` → `whisperx.align` | Word-level timestamps for the **host utterance** (BC anchoring) and the alignment JSON. Not used for transcription. |
| **NeMo normalizer** | `Normalizer(input_case="cased", lang="en")` `:1415` (lazy) | EN path only: normalize text before word count / alignment. VI sets `normalizer=None`. |
| **pyloudnorm** | `_loudness_normalize(target=-23.0 LUFS)` `:135-155` | Final EBU R128 normalization of the 2-ch master; peak-norm fallback. |
| **LibriSpeech sample** | `list_librispeech_speakers` `:171`, `sample_n_librispeech_prompts` `:238` | EN voice source: 12 speakers under `tts_render/librispeech_samples/train-clean-100/`, assistant speaker `458` excluded. |
| **Voice-clone pool** | `_load_voice_pool` `:772`, `_load_omnivoice_ref_audio` `:759` | VI voice source: `voice_clone/` = **85 `.wav` + 85 `.txt` pairs**; 2 distinct drawn per dialogue. Needs ≥2 or `SystemExit`. |
| **BackchannelCache** | `bc_cache.BackchannelCache` `:29, :888` | Caches generated BC audio per speaker to avoid re-synth of short tokens. |

## Environments

Two interpreters, never merged (dependency conflict):

| | Stages 1–4 | Stage 5 |
|---|---|---|
| env | `.venv` | `.venv-tts` or conda `vilex-omnivoice` |
| python | ≥3.10 | 3.11 |
| transformers | ≥4.53 | Chatterbox EN path pins `==4.46.3` via `vilex/tts/chatterbox`; OmniVoice brings its own |
| key deps | — | `torchaudio>=2.6,<3`, `silero-vad>=6.2.1`, `whisperx>=3.3.1`, `numpy<2`, `pyloudnorm` (VI), `nemo-text-processing` (EN) |

Source: `requirements-stage5.txt`. Note `numpy<2` is deliberate (NEP 50
breaks Chatterbox `norm_loudness`); OmniVoice/VI needs no NeMo. Forced alignment
uses whisperx (wav2vec2 per language).

## Constants

| Name | Value | Where |
|---|---|---|
| `TARGET_SR` | 24000 | `:72` (set from model) |
| `PROMPT_SR` | 16000 | `:73` |
| gap (inter-speaker) | Exp scale `0.7067`, clip `[0.003, 2.50]` s, ×`(dur/3)**0.25` | `:77-79, :121-126` |
| pause (same speaker) | Exp scale `0.6780`, clip `[0.001, 4.00]` s | `:82-84, :129-132` |
| interrupt overlap | `USER_INTERRUPT_OVERLAP_SEC=0.64`, prob `0.5` | `:86-87` |
| explicit `[TAKE_FLOOR]` overlap | Normal `0.45±0.05` s | `:679` |
| `MAX_PROMPT_SECS` | 10 | `:88` |
| `TARGET_LUFS` | −23.0 | `:91` |
| `NOISE_FLOOR_AMP` | `0.000` (room tone disabled in working tree) | `:92` |
| `ASSISTANT_SPEAKER_ID` | `"458"` | `:97` |
| VAD threshold | `0.3` | `:994, :1044` |

## Per-turn pipeline

Loop: `for turn, utterance in enumerate(dialogue["utterances_with_bc"])` `:899`.
State: `tts_texts[2]`, `bc_queue[2]`, `speaker_ref[2]`, `interrupt_flag[2]`,
`accumulated_flag[2]`, `prev_uttr_type`.

1. **Clean text** `:908-952` — strip `[interrupted]/[MASK1]/[MASK2]`. Empty / `-` /
   `...` → mark `isUttered=False`; if previous was a backchannel, drop it from the
   queue. If `next == backchannel`, accumulate and `continue` (defer TTS until the
   BC chain ends). If `prev == backchannel`, append text to the running buffer.
2. **Per-sentence TTS** `:957-1037`
   - Pick voice ref: OmniVoice clones from `speaker_ref[curr]` if seeded, else
     `instruct`.
   - Backchannel: append `"?"` for rising tokens (`_BC_RISING_TOKENS`: `yeah, ưm,
     vâng, à, ừ` `:107`), consult `bc_cache`, else generate + cache.
   - VAD-trim trailing silence between sentences (not the last) `:989-1007`.
   - BC attenuated `×0.8` `:1012-1013`.
   - Chatterbox appends non-BC audio to `cumulative_audio`; OmniVoice seeds
     `speaker_ref` from the speaker's first non-BC sentence `:1016-1035`.
   - Concatenate sentences; VAD-trim the full utterance (`start/end` per
     `is_last_text` / next interrupt / last) `:1040-1077`.
3. **Dispatch** `:1096-1245`
   - `uttr_type == backchannel` → `word_count = count_words_for_align(host_text)`
     and enqueue into `bc_queue[curr]` (no track yet) `:1105`.
   - host utterance → allocate `listener_speech=zeros`; if the other queue has BC,
     align the host and place each BC after `all_words[clamp(wc-1,0,len-2)].end`,
     with a `cursor` preventing BC overlap and `place_backchannel` padding on
     overrun `:1161-1204`. Stack `[tts, listener]` (order by speaker) → stereo,
     append to `total_speech` + `total_speech_meta`.
4. **Assembly** `aggregate_speech:662` walks utterances in time order:
   - explicit `interrupt` + speaker change → cross-fade overlap `~0.45 s`.
   - speaker change, no interrupt → 50% try interrupt when `assistant→user`
     (leading-BC check decides gap vs overlap), else a sampled **gap**.
   - same speaker → sampled **pause** if `boundary=="turn"`, else butt-join.
   - final `_loudness_normalize` to −23 LUFS.

## Voice casting (deterministic)

- Per dialogue: `dialogue_seed = stage5.seed + crc32(filename)` `:1448`
  (crc32, not `hash()`, so a resumed run reproduces the same cast).
- OmniVoice pool: `dialogue_rng.sample(range(len(pool)), 2)` → user + assistant
  distinct `:1454`. No pool → all variants use the two `stage5.voice.*`
  design strings.
- Chatterbox: `sample_n_librispeech_prompts(..., num_variants, ...)` distinct
  speakers, `458` excluded.
- Per variant: `random/np/torch` reseeded with `seed + variant_idx` `:1496-1501`
  so TTS/timing randomness is reproducible across dialogues.
- Resume: a variant with both `dialogue.wav` and `meta.json` is skipped `:1475`.

## Guards & drop rules

- `should_drop_dialogue` `:447` — drop on `</think>`, multi-turn concat
  (`>=2` `"N) role:"`), `**bold**`, `*next turn*` markers, template placeholders
  (`[Email protected]`, `[City]`), >1500 chars, outer/inline role mismatch.
- `sanitize_text` `:503` — soft-strip `[User Turn]`, `[TAKING_FLOOR]→[TAKE_FLOOR]`,
  role prefixes, `*stage directions*`.
- `_cut_at_take_floor_plus_one_word` `:348` — keep 1 word after `[TAKE_FLOOR]`
  (or 3 when the left side is <5 words) to avoid a truncated fragment.
- TTS empty output → 0.2 s silence `:828/:835`.
- Alignment returns no words → all queued backchannels dropped, `isUttered=False`
  `:1151-1159`.
- Backchannel with `pad_size<=0` on the last slot before an interrupt → dropped
  `:1187-1193`.
- Run with zero rendered/skipped → `SystemExit` `:1595`.

## Timing meta written per utterance

`total_speech_meta["speech_meta"][i]` carries `speaker`, `tts_text`,
`timing ∈ {none, gap, pause, overlap}`, `duration_sec`, `start_sample` (offset
of this utterance in the merged timeline, set by `aggregate_speech`), and
optional `backchannels[{idx, tts_text}]`. The run-level `timing_config` records
the gap / pause distributions, `overlap_max_sec`, and `user_interrupt_prob`.

## Word alignment JSON (`stage5_2b_assemble.audio.save_align_json: true`)

`main_process` forced-aligns each host utterance (and BCs with >1 word) with the
whisperx aligner (`_align_words_once`); the variant loop writes `alignment_user.json`
and `alignment_assistant.json` next to `meta.json` (schema: `turns[]` with `turn`,
`kind ∈ {utterance, backchannel}`, `text`, absolute `start_sec`/`end_sec`,
`words[{word,start,end}]`). Utterance/BC word times are absolute in the merged
track (`start_sample/24000 + local`); a BC whose alignment is empty gets an even
split of its clip span. Enabling the flag adds both files to the resume skip
check.
