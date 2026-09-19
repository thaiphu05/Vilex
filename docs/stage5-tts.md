# Stage 5: TTS rendering

Renders the Stage 4 result to two-channel audio with OmniVoice (Vietnamese,
default) or Chatterbox (English, legacy).

> For the model inventory, constants, and per-turn pipeline, see
> [stage5-internals](stage5-internals.md).

Like every stage it is config-driven — no dataset/flag CLI beyond an optional
`--config PATH` (see [CONFIGURATION.md](CONFIGURATION.md)):

```yaml
run:
  target_language: vi
stage5:
  language: vi
  num_variants: 1
  max_dialogues: 0        # 0 = all
stage5_1b_render:
  backend: omnivoice
  device: cpu
paths:
  bc_root: data/vi_tt_bc  # input (Stage 4b output)
  audio_root: data/vi_audio
```

```bash
.venv/bin/python tts_render/convert_spoken.py            # reads ./config.yaml
.venv/bin/python tts_render/convert_spoken.py --config my.yaml
```

Note the **other interpreter** (`.venv-tts` / the OmniVoice env, see
[Setup](../README.md#setup)) and that this is run as a *file path*, not as a
`-m` module like Stages 1-4.

Each dialogue is rendered into `stage5.num_variants` two-channel variants,
seeded per dialogue from `stage5.seed`. Under Chatterbox the assistant uses
a fixed cloning prompt (`tts_render/prompt_wavs/assistant_en.wav`, a held-out
LibriSpeech speaker) and each variant's user voice is sampled from a distinct
LibriSpeech speaker (bundled 12-speaker sample). Under OmniVoice both voices come
from the per-speaker `instruct` strings (`stage5.voice.*`). Backchannels are
synthesized through the same voice as the speaker uttering them, so they match
that variant.

> [!IMPORTANT]
> **Chatterbox renders backchannels poorly.** Short interjections — `mhm`,
> `mm-hmm` — come out flat and mispronounced; a zero-shot TTS has almost no
> context to work from on a two-phoneme utterance. **The paper's dialogues used
> the ElevenLabs API for these instead.** That dependency was dropped for 1.0 so
> the pipeline runs without a paid key, and what ships is the self-contained
> fallback. To use another backend, replace the `uttr_type == "backchannel"`
> branch in `main_process` (`tts_render/convert_spoken.py`) — placement, caching
> and the `backchannels/*.wav` outputs take whatever waveform it returns. Keep
> the backchannel voice matched to the speaker uttering it.

## Output layout

Every `paths.bc_root/**/*.json` is matched, and the trailing
`text_dialogue_<dataset>/<split>/` of each input path is mirrored under
`paths.audio_root`:

```
data/vi_audio/text_dialogue_interviewer/train/work_0000/var00/
├── dialogues/dialogue.wav     # the full two-channel mix
├── utterances/00.wav ...      # one two-channel file per utterance, pre-mix
├── backchannels/05_0.wav ...  # <utterance>_<slot>, the isolated backchannel
├── meta.json                  # per-utterance text, timings, and the voice used
└── alignment_user.json        # only when save_align_json: true
    alignment_assistant.json   #   (word-level timestamps per speaker)
```

`dialogue.wav` is 24 kHz 16-bit stereo with **channel 0 = assistant, channel 1 =
user** — the channels are swapped on the way out so the assistant lands first,
which is the order `personaplex-finetune` expects. `meta.json` records that as
`"speakers": ["assistant", "user"]`, along with the voice used for the variant.

A variant whose `dialogue.wav` and `meta.json` both exist (and the alignment
files, when `save_align_json` is on) is skipped, so a re-run resumes rather than
re-synthesizing. Voice casting is seeded per dialogue from `stage5.seed` and
the file name, so a resumed run fills the missing variants from the same cast as
the ones already on disk.

## Split pipeline (VI/OmniVoice, GPU-friendly)

Beside the monolithic `convert_spoken.py` there is a 4-tier split path that lets
each resource (CPU prep, GPU TTS, GPU align, CPU assemble) run on its own box:

```bash
./run_vi_stage5_split.sh          # 5.1a -> 5.1b -> 5.2a -> 5.2b
```

- **5.1a** `tts_render/stage5a_prep.py` (CPU) freezes all text decisions into
  `paths.stage5_work_root/<rel>/varNN/manifest.json` + a global
  `_batch/omnivoice_units.jsonl`.
- **5.1b** `stage5b_render.py` (GPU, OmniVoice env) renders raw unit wavs; it
  batches across dialogues grouped by reference voice, records `failed_units`,
  `pause_samples` and `unit_audio`, and refuses if the voice pool changed.
- **5.2a** `stage5c_align.py` (GPU, Qwen3 env) applies the same VAD trims, ×0.8
  backchannel attenuation and forced alignment, writing `alignment.json`.
- **5.2b** `stage5d_assemble.py` (CPU) places backchannels, runs the timing
  layer, normalises LUFS and writes the final `varNN/` tree with the monolithic
  schema (plus `alignment_*.json` when `save_align_json: true`).

All four read the shared config via `tts_render/stage5_config.resolve_stage5`
(`stage5:` block, or the legacy `stage5_tts:` block).

## Cost knobs

Rendering runs at roughly real time, so cap a trial run: `stage5.max_dialogues
N` takes only the first N dialogues, and `stage5.num_variants` scales the
per-dialogue cost linearly. (Sharding is not exposed in the config; run separate
`paths.audio_root` trees if you parallelise.)

### Batching, alignment granularity, fallbacks

Everything here defaults to the sequential behaviour and can be turned on for a
GPU box:

- `stage5_1b_render.omnivoice.batch_size` — 5.1b flattens **every** manifest into
  one cross-dialogue queue and chunks it by `batch_size` (`1` = one item per
  call). A chunk may mix speakers/dialogues because each item carries its own
  reference voice (OmniVoice takes per-item `ref_audio`/`ref_text` lists).
  **Needs a voice pool** (always present on the split path). Units longer than
  `max_unit_chars` are generated one per call so a long sentence cannot inflate a
  shared chunk. Each chunk is also written to `_batch/batch_<NNNN>.jsonl`
  (`batches.json` indexes them). A chunk failure falls back to per-item calls; an
  item that still fails after `max_retries` becomes 0.2 s silence
  (`fallback_action`), is recorded in `failed_units`, and is dropped from
  alignment. Manifests are rewritten once at the end of the run, and resume is
  filesystem-based (an existing unit wav is skipped).
- `stage5_2a_align.aligner.granularity: utterance | sentence` — `sentence` aligns each
  generated unit separately (then offsets into the turn timeline), which keeps
  each forward short; `utterance` is the legacy whole-turn alignment.
- `stage5_2a_align.aligner.batch: true` — batch the aligner forward across a dialogue.
  It implies `bc_placement: defer` (via `auto`), i.e. the whole dialogue is
  rendered before aligning and placing; gap/pause timing is unchanged because
  placement still happens before the merged timeline is assembled.
- `stage5_2a_align.aligner.max_secs` — audio above this length skips the model (its
  ceiling is ~300 s) and goes straight to the fallback.
- `stage5_2a_align.aligner.fallback: drop | proportional` — what to do when no words
  came back (aligner error, empty result, or over `max_secs`). `proportional`
  splits the turn span evenly across its words so backchannels are still placed
  (approximate); `drop` discards that turn's queued backchannels, leaves
  `isUttered=false`, and writes no `backchannels/*.wav` for them.
- `stage5_2b_assemble.profile: true` — log per-dialogue generate/align timings.

## Dropped inputs

Dialogues carrying LLM artifacts the sanitizer cannot repair — leaked think
blocks, concatenated turns, template placeholders — are skipped with a
`DROP-DIALOGUE` warning naming the reason. Individual backchannels are dropped
with a warning when forced alignment returns no words to anchor them to; the
dialogue still renders, and `utterances_with_bc[i].texts_styled[j].isUttered` in
`meta.json` records which backchannels actually made it into the audio. The run
ends with a summary line counting rendered, skipped, dropped, and failed
variants, and exits non-zero if it rendered nothing at all.

## Voices

The Chatterbox path uses the 12-speaker LibriSpeech sample bundled in
`tts_render/librispeech_samples/`, so it runs out of the box; that sample bounds
`num_variants` at 12. For paper-scale English rendering, download the full
[LibriSpeech](https://www.openslr.org/12) corpus and point the code's
`librispeech_root` at it (not currently a config key; this path is legacy).

## Vietnamese rendering with OmniVoice

Stage 5 renders **Vietnamese** dialogues with
[OmniVoice](https://github.com/k2-fsa/OmniVoice) — the default and the path used
for this project (Stages 1-4 emit Vietnamese via Gemini; `run.target_language:
vi`). OmniVoice needs **no voice-prompt audio** — it uses a *voice-design*
`instruct` string per speaker — and it runs in its own environment (see
`requirements-stage5.txt`, Block C); NeMo is **not** required.

```bash
conda create -n vilex-omnivoice python=3.11 -y
conda activate vilex-omnivoice
pip install torch==2.8.* --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-stage5.txt

python tts_render/convert_spoken.py   # config: backend omnivoice, language vi
```

A smoking-test `config.yaml` for one Vietnamese sample:

```yaml
stage5:
  language: vi
  num_variants: 1
  max_dialogues: 1        # one sample before a full render
  voice:
    user_instruct: "male, northern accent"
    assistant_instruct: "female, gentle"
stage5_1b_render:
  backend: omnivoice
  device: cuda
paths:
  bc_root: data/vi_tt_bc
  audio_root: data/vi_audio
```

What changes under `backend: omnivoice` / `language: vi`:

* **TTS backend.** `generate_audio()` calls `OmniVoice.generate(text, instruct,
  language="Vietnamese", normalize_text=True)`; `instruct` is the per-speaker
  voice-design string (no LibriSpeech prompt, no cumulative voice audio).
* **Forced alignment.** Word timestamps (backchannel anchoring + the alignment
  JSON) come from the **Qwen3 forced aligner** (`stage5_2a_align.aligner`, default
  `Qwen/Qwen3-ForcedAligner-0.6B-hf`) run on each OmniVoice clip via
  `transformers`. Note its checkpoints only officially cover 11 languages and
  **not Vietnamese**, so VI timestamps are best-effort.
* **No NeMo.** `normalizer=None`; word counting falls back to a Vietnamese regex
  (`_VI_WORD_RE`) instead of `Normalizer.normalize`.
* **Sentence splitting.** `split_sentences()` avoids the English-only `nltk`
  punkt model and splits on `.<>?`/`!` for Vietnamese.
* **Backchannels.** Vietnamese fallbacks (`ưm`, `à`, `ừ`, `vâng`, `phải`,
  `thật không`, `ồ`) and a rising-intonation set (`_BC_RISING_TOKENS`) replace the
  English `yeah/uh-huh` defaults (`stage5.backchannels.*`).
* **Sample rate.** OmniVoice runs at 24 kHz (`stage5.target_sr = 24000`); the
  output layout is identical to the Chatterbox path.
* **Paralinguistic tags.** By default tags (`[laughter]`, `[sigh]`,
  `[question-*]`, `[surprise-*]`, `[confirmation-en]`, `[dissatisfaction-hnn]`)
  are stripped before `generate()` because the checkpoint otherwise reads them as
  literal text. Set `stage5.tags.render: true` to keep the 13 supported tags
  (`stage5.tags.supported`) so OmniVoice renders them as audio; unknown tags
  are still stripped.
* **Word alignments for timestamps.** Set `stage5_2b_assemble.audio.save_align_json: true`
  to also write `alignment_user.json` and `alignment_assistant.json` next to
  `meta.json` in each `varNN/`. Each entry lists `turn`, `kind`
  (`utterance`/`backchannel`), `text`, absolute `start_sec`/`end_sec`, and
  word-level `words[{word,start,end}]` (times absolute in the merged
  dialogue). Utterance words come from the Qwen3 forced aligner; backchannel
  words come from alignment when the BC has more than one word, otherwise an even
  split of its clip span. Enabling it makes those files part of the resume check
  (existing variants are re-rendered to fill them in).

> [!NOTE]
> OmniVoice's voice-design was tuned mostly on Chinese/English; Vietnamese voice
> quality and stability can vary. Verify one `stage5.max_dialogues: 1` sample
> before a full render, and tune `stage5.voice.*` to taste. If you already have
> **English** Stage 4 JSONs, translate them to Vietnamese first with
> `tools/translate_dialogues.py` (preserves `[TAKE_FLOOR]` and backchannel slots).
