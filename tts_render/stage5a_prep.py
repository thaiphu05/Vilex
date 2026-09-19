"""Stage 5.1a — Normalized text prep for the split Stage 5 pipeline.

Reads Stage 4b JSONs (paths.bc_root) and, per dialogue x variant, freezes every
text decision into a manifest.json plus one global batch file for the OmniVoice
CLI. CPU-only, no model, no torch: everything downstream (5.1b render, 5.2a
align, 5.2b assemble) consumes these files and never re-derives text.

Usage:
    python tts_render/stage5a_prep.py [--config PATH]
"""

import argparse
import json
import random
import sys
import zlib
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

from src.config import cfg_get, load_config  # noqa: E402

import stage5_lib as lib  # noqa: E402
from stage5_config import resolve_stage5, work_root as stage5_work_root  # noqa: E402
from stage5_schema import (  # noqa: E402
    Backchannel,
    Manifest,
    Unit,
    Utterance,
    VoicePick,
    read_manifest,
    write_manifest,
)

TARGET_SR = 24000  # informational; 5.1a writes no audio

# Salt for the seeded backchannel fallback RNG (decision S1): keeps BC choice
# deterministic per dialogue, independent of other RNG streams.
_BC_SEED_SALT = 0xBC01


def _voice_pool_index(pool_dir: Path) -> List[Dict[str, str]]:
    """Index the voice-clone pool: sorted [{wav, txt, name}], needs >= 2 pairs.

    Pure metadata (paths + sidecar text), no audio is read or copied.
    """
    items = []
    for wav in sorted(pool_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            print(f"WARNING: skipping {wav.name}, no sidecar .txt")
            continue
        st = wav.stat()
        items.append(
            {
                "wav": str(wav.resolve()),
                "txt": str(txt.resolve()),
                "ref_text": txt.read_text(encoding="utf-8").strip(),
                "wav_size": int(st.st_size),
                "wav_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            }
        )
    if len(items) < 2:
        raise SystemExit(
            f"ERROR: voice pool needs >=2 wavs with sidecar .txt, found {len(items)} in {pool_dir}"
        )
    return items


def _variant_dir(work_root: Path, rel_path: Path, variant_idx: int) -> Path:
    return work_root / rel_path / f"var{variant_idx:02d}"


def _jsonl_id(rel_path: Path, variant_idx: int, unit_key: str) -> str:
    flat = "__".join(rel_path.parts)
    return f"{flat}__var{variant_idx:02d}__{unit_key}"


def _flush_host_units(tts_text: str, host_idx: int, render_tags: bool) -> List[Unit]:
    """Split one accumulated host text into text/pause units (convert_spoken
    :1608-1634): strip tags, split sentences (VI), split on [PAUSE]."""
    keep = lib.RENDERABLE_TAGS if render_tags else None
    tts_text = lib.strip_paralinguistic(tts_text, keep=keep)
    tts_text = (
        tts_text.replace("[interrupted]", "").replace("[MASK1]", "").replace("[MASK2]", "").strip()
    )
    units: List[Unit] = []
    for sentence in lib.split_sentences_vi(tts_text):
        parts = sentence.split(lib.PAUSE_TOKEN)
        for pi, part in enumerate(parts):
            part = part.strip()
            if part:
                units.append(Unit(kind="text", unit_id=f"u_{host_idx}_{len(units)}", text=part))
            if pi != len(parts) - 1:
                units.append(Unit(kind="pause", unit_id=f"u_{host_idx}_{len(units)}"))
    return units


def build_manifest(
    original: Dict[str, Any],
    dialogue_id: str,
    rel_path: Path,
    variant_idx: int,
    seed: int,
    pool: List[Dict[str, str]],
    render_tags: bool,
    language: str,
) -> Manifest:
    """Freeze one Stage-4b dialogue into a Manifest (all text decisions)."""
    # Voice cast: deterministic per (filename, seed) — convert_spoken:2368-2387.
    dialogue_seed = seed + zlib.crc32(rel_path.name.encode("utf-8"))
    rng_voice = random.Random(dialogue_seed)
    ui, ai = rng_voice.sample(range(len(pool)), 2)
    picks = {
        "user": VoicePick(
            wav=pool[ui]["wav"],
            txt=pool[ui]["txt"],
            wav_size=int(pool[ui].get("wav_size", 0)),
            wav_mtime_ns=int(pool[ui].get("wav_mtime_ns", 0)),
        ),
        "assistant": VoicePick(
            wav=pool[ai]["wav"],
            txt=pool[ai]["txt"],
            wav_size=int(pool[ai].get("wav_size", 0)),
            wav_mtime_ns=int(pool[ai].get("wav_mtime_ns", 0)),
        ),
    }

    # Seeded BC fallback (S1).
    rng_bc = random.Random(dialogue_seed + _BC_SEED_SALT)
    candidates = (
        lib.DEFAULT_BC_CANDIDATES_VI if language.startswith("vi") else lib.DEFAULT_BC_CANDIDATES
    )

    def pick_bc() -> str:
        return rng_bc.choice(candidates)

    raw_utts = lib.convert_dialogue_to_utterances(original, pick_bc=pick_bc, language=language)

    utterances: List[Utterance] = []
    backchannels: List[Backchannel] = []
    tts_texts = ["", ""]
    prev_uttr_type = "first"
    accumulated_flag = [False, False]
    interrupt_flag = [False, False]
    host_idx_of_bc: List[int] = []  # pending BC utterance idx -> host word_count anchor

    n = len(raw_utts)
    if not n:
        return Manifest(
            dialogue_id=dialogue_id,
            rel_path=str(rel_path),
            variant_idx=variant_idx,
            seed=seed,
            voice_picks=picks,
        )
    # Channel is POSITIONAL (monolithic main_process :1564-1566):
    # curr_idx = (turn + spk_offset) % 2. The BC turn's speaker label names
    # the host's message, but the BC itself plays on the listener channel.
    init_spk = raw_utts[0]["speaker"]
    spk_offset = 1 if init_spk == lib.DEFAULT_SPEAKERS[1] else 0
    for turn, utt in enumerate(raw_utts):
        speaker_idx = (turn + spk_offset) % 2
        other_idx = 1 - speaker_idx
        next_uttr_type = "last" if turn == n - 1 else raw_utts[turn + 1]["uttr_type"]
        raw = utt["text"]
        cleaned = (
            raw.replace("[interrupted]", "").replace("[MASK1]", "").replace("[MASK2]", "").strip()
        )
        is_last_text = True  # convert_original_to_expected makes 1 text per utterance

        # convert_spoken:1584-1585
        if utt["uttr_type"] == "interrupt":
            interrupt_flag[speaker_idx] = True

        do_flush = False
        if not cleaned or cleaned in ["-", "...", ".", ","]:
            # :1587-1597 — not uttered; drop a dangling BC on the other speaker.
            if prev_uttr_type == "backchannel" and host_idx_of_bc:
                dropped = host_idx_of_bc.pop()
                backchannels = [b for b in backchannels if b.bc_idx != dropped]
            if next_uttr_type == "backchannel" or not is_last_text:
                prev_uttr_type = utt["uttr_type"]
                continue
            if accumulated_flag[speaker_idx]:
                do_flush = True  # flush leftover accumulated text
            # else: skip entirely
        elif utt["uttr_type"] == "backchannel":
            pass  # handled below
        else:
            # Host: accumulate text across BC interruptions (:1599-1602).
            if prev_uttr_type == "backchannel":
                tts_texts[speaker_idx] += " " + raw
            else:
                tts_texts[speaker_idx] = raw
            if next_uttr_type == "backchannel" and is_last_text:
                accumulated_flag[speaker_idx] = True  # :1604-1606
            else:
                do_flush = True

        if utt["uttr_type"] == "backchannel" and cleaned and cleaned not in ["-", "...", ".", ","]:
            # BC utterance: no units; anchor word_count into the CURRENT host
            # text of the other speaker (convert_spoken:1828-1847).
            bc_idx = len(backchannels)
            wc = lib.count_words_for_align_vi(tts_texts[other_idx])
            text = raw
            if lib.strip_paralinguistic(text).strip().lower() in lib.BC_RISING_TOKENS:
                text = text.rstrip() + "?"
            backchannels.append(
                Backchannel(
                    bc_idx=bc_idx,
                    speaker_idx=speaker_idx,
                    text=text,
                    word_count=wc,
                    host_idx=len(utterances),  # next host to be appended
                    next_uttr_type_of_host=None,
                    unit_id=f"bc_{bc_idx}",
                )
            )
            host_idx_of_bc.append(bc_idx)
            prev_uttr_type = "backchannel"
            continue

        if not do_flush:
            prev_uttr_type = utt["uttr_type"]
            continue

        accumulated_flag[speaker_idx] = False
        tts_text = tts_texts[speaker_idx]
        host_idx = len(utterances)
        units = _flush_host_units(tts_text, host_idx, render_tags)
        utterances.append(
            Utterance(
                turn=turn,
                text_idx=0,
                speaker_idx=speaker_idx,
                uttr_type="interrupt" if interrupt_flag[speaker_idx] else None,
                next_uttr_type=next_uttr_type,
                is_last_text=is_last_text,
                isUttered=True,
                tts_text=tts_text.strip(),
                units=units,
            )
        )
        interrupt_flag[speaker_idx] = False
        tts_texts[speaker_idx] = ""
        prev_uttr_type = utt["uttr_type"]

    # Resolve each BC's host context now that hosts are appended.
    for bc in backchannels:
        hi = min(bc.host_idx, len(utterances) - 1)
        bc.host_idx = hi
        bc.next_uttr_type_of_host = utterances[hi].next_uttr_type if utterances else None

    return Manifest(
        dialogue_id=dialogue_id,
        rel_path=str(rel_path),
        variant_idx=variant_idx,
        seed=seed,
        voice_picks=picks,
        utterances=utterances,
        backchannels=backchannels,
    )


def manifest_jsonl_rows(
    m: Manifest, work_root: Path, language: str = "vi", pool_dir: str = "voice_clone"
) -> List[Dict[str, Any]]:
    """One JSONL row per text unit + per backchannel, for omnivoice-infer-batch.

    id encodes the variant + unit so 5.1b can map the rendered wav back. Pause
    units are excluded (OmniVoice cannot read [PAUSE]; 5.1b synthesizes noise).

    ``ref_audio`` is written the way the pool is configured (``pool_dir`` +
    basename) rather than as an absolute path, e.g. ``voice_clone/id00963.wav``.
    """
    rows: List[Dict[str, Any]] = []
    rel = Path(m.rel_path)
    ref_text_by_role = {}
    for role in ("user", "assistant"):
        txt_path = Path(m.voice_picks[role].txt)
        ref_text_by_role[role] = txt_path.read_text(encoding="utf-8").strip()

    def row(unit_key: str, text: str, speaker_idx: int) -> Dict[str, Any]:
        role = "user" if speaker_idx == 0 else "assistant"
        return {
            "id": _jsonl_id(rel, m.variant_idx, unit_key),
            "text": text,
            "ref_audio": str(Path(pool_dir) / Path(m.voice_picks[role].wav).name),
            "ref_text": ref_text_by_role[role],
            "language_id": "vi" if str(language).lower().startswith("vi") else "en",
            "speed": 1.2,
        }

    for u in m.utterances:
        for un in u.units:
            if un.kind == "text":
                rows.append(row(un.unit_id, un.text, u.speaker_idx))
    for bc in m.backchannels:
        rows.append(row(bc.unit_id, bc.text, bc.speaker_idx))
    return rows


def _existing_unit_audio(mpath: Path) -> Dict[str, str]:
    """Resume: read a prior manifest's unit_audio so re-runs only emit the
    units that still lack audio (5.1b skips ids already rendered)."""
    if not mpath.is_file():
        return {}
    try:
        return read_manifest(mpath).unit_audio
    except Exception:
        return {}


def main(config_path=None) -> None:
    cfg = load_config(config_path)
    paths = cfg_get(cfg, "paths", {})
    s5 = resolve_stage5(cfg)
    shared = s5.get("stage5", {})

    bc_root = Path(paths.get("bc_root", "data/vi_tt_bc"))
    work_root = Path(stage5_work_root(s5, cfg))
    pool_dir_str = str(paths.get("voice_clone_pool", "voice_clone"))
    pool_dir = Path(pool_dir_str)
    seed = int(shared.get("seed") or cfg_get(cfg, "run.seed", 42))
    language = str(shared.get("language") or cfg_get(cfg, "run.target_language", "vi"))
    num_variants = int(shared.get("num_variants", 1))
    tags = shared.get("tags", {}) if isinstance(shared.get("tags"), dict) else {}
    render_tags = bool(tags.get("render", False))

    pool = _voice_pool_index(pool_dir)
    json_files = sorted(glob(str(bc_root / "**" / "*.json"), recursive=True))
    if not json_files:
        raise SystemExit(
            f"No Stage 4b JSONs under {bc_root} (need text_dialogue_<ds>/<split>/*.json)."
        )

    batch_dir = work_root / "_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = batch_dir / "omnivoice_units.jsonl"

    n_dropped = n_variants = n_rows = n_skipped = 0
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for fpath_str in json_files:
            fpath = Path(fpath_str)
            rel_dialogue = Path(*fpath.parts[-3:-1])  # text_dialogue_<ds>/<split>
            rel_path = rel_dialogue / fpath.stem
            dialogue_id = fpath.stem

            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            drop, reason = lib.should_drop_dialogue(data)
            if drop:
                n_dropped += 1
                print(f"DROP-DIALOGUE: {fpath.name} ({reason})")
                continue

            for variant_idx in range(num_variants):
                vdir = _variant_dir(work_root, rel_path, variant_idx)
                mpath = vdir / "manifest.json"
                prior_audio = _existing_unit_audio(mpath)
                m = build_manifest(
                    data, dialogue_id, rel_path, variant_idx, seed, pool, render_tags, language
                )
                # Resume: keep prior unit_audio; only emit rows still missing audio.
                m.unit_audio = prior_audio
                write_manifest(mpath, m)
                n_variants += 1

                for r in manifest_jsonl_rows(m, work_root, language, pool_dir_str):
                    unit_key = r["id"].split("__")[-1]
                    if unit_key in prior_audio and Path(prior_audio[unit_key]).is_file():
                        n_skipped += 1
                        continue
                    jf.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += 1

    print(
        f"Stage 5.1a done: {n_variants} variants prepped, {n_rows} units queued, "
        f"{n_skipped} units resumed, {n_dropped} dialogues dropped. JSONL -> {jsonl_path}"
    )


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Stage 5.1a: normalized text prep.")
    _ap.add_argument("--config", default=None, help="Path to config.yaml")
    main(_ap.parse_args().config)
