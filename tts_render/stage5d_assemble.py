"""Stage 5.2b — Backchannel placement, timing, LUFS and final output (CPU).

Consumes the trimmed host audio + ``alignment.json`` from 5.2a, places each
backchannel at its aligned word-end, runs the timing layer (gaps/pauses/overlap),
normalizes to the target LUFS, flips the channels and writes the final variant
tree that matches the monolithic output schema.

Writes to ``<paths.audio_root>/<rel_path>/varNN/``:
    dialogues/dialogue.wav, user.wav, assistant.wav,
    utterances/, backchannels/, meta.json, alignment_*.json

Reuses convert_spoken's helpers so the audio math stays identical to monolithic.

Usage:
    python tts_render/stage5d_assemble.py [--config PATH] [--limit N]
"""

import argparse
import json
import random
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

from src.config import cfg_get, load_config  # noqa: E402

from stage5_config import resolve_stage5, work_root  # noqa: E402
from stage5_schema import read_manifest  # noqa: E402

TIMING_SALT = 0x5F15


def _load_alignment(vdir: Path) -> Dict[str, Any]:
    with open(vdir / "alignment.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _timing_config(cs) -> Dict[str, Any]:
    return {
        "gap": {
            "distribution": "exponential",
            "scale_sec": cs.GAP_EXP_SCALE,
            "min_sec": cs.GAP_MIN_SEC,
            "max_sec": cs.GAP_MAX_SEC,
            "percentiles_fitted": {
                "min": 0.003,
                "P25": 0.203,
                "P50": 0.49,
                "P75": 0.98,
            },
        },
        "pause": {
            "distribution": "exponential",
            "scale_sec": cs.PAUSE_EXP_SCALE,
            "min_sec": cs.PAUSE_MIN_SEC,
            "max_sec": cs.PAUSE_MAX_SEC,
            "percentiles_fitted": {
                "min": 0.001,
                "P1": 0.007,
                "P25": 0.195,
                "P50": 0.47,
                "P75": 0.94,
                "max": 4.0,
            },
        },
        "overlap_max_sec": cs.USER_INTERRUPT_OVERLAP_SEC,
        "user_interrupt_prob": cs.USER_INTERRUPT_PROB,
        "intra_pause": {
            "scale_sec": cs.PAUSE_INTRA_EXP_SCALE,
            "min_sec": cs.PAUSE_INTRA_MIN_SEC,
            "max_sec": cs.PAUSE_INTRA_MAX_SEC,
        },
    }


def assemble_variant(
    cs,
    m,
    align: Dict[str, Any],
    vdir: Path,
    out_dir: Path,
    audio_cfg: Dict[str, Any],
) -> None:
    import torch

    variant_seed = int(m.seed) + m.variant_idx + TIMING_SALT
    random.seed(variant_seed)
    cs.np.random.seed(variant_seed)
    torch.manual_seed(variant_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(variant_seed)

    hosts = {h["host_idx"]: h for h in align["hosts"]}
    total_speech: List[Any] = []
    total_speech_meta: List[Dict[str, Any]] = []
    align_utt: List[Dict[str, Any]] = []
    align_bc: List[Dict[str, Any]] = []
    backchannel_list: List[Any] = []

    bcs_by_host: Dict[int, List[Dict[str, Any]]] = {}
    for bc in align["bcs"]:
        bcs_by_host.setdefault(bc["host_idx"], []).append(bc)

    for host_idx, u in enumerate(m.utterances):
        h = hosts.get(host_idx)
        if h is None:
            continue
        host_wav = cs.torchaudio.load(h["audio"])[0]
        if host_wav.shape[0] > 1:
            host_wav = host_wav.mean(dim=0, keepdim=True)
        listener = torch.zeros_like(host_wav)

        words = h["words"]
        if not words:
            words = cs._words_with_fallback([], h["text"], host_wav.size(1) / cs.TARGET_SR)
        if cs.SAVE_ALIGN_JSON:
            align_utt.append(
                {
                    "host_idx": host_idx,
                    "speaker": u.speaker_idx,
                    "text": h["text"],
                    "words": words,
                }
            )

        bcs = bcs_by_host.get(host_idx, [])
        uttered: List[Dict[str, Any]] = []
        if bcs and words:
            # Inline placement (port of convert_spoken._place_queued_bcs) so the
            # BC words aligned in 5.2a are carried straight into alignment_*.json.
            bc_pos = []
            for bc in bcs:
                word_idx = max(0, min(bc["word_count"] - 1, len(words) - 2))
                bc_pos.append(round(words[word_idx]["end"] * cs.TARGET_SR))
            bc_pos.append(host_wav.size(1))
            cursor = 0
            for i, bc in enumerate(bcs):
                bc_wav = cs.torchaudio.load(bc["audio"])[0]
                if bc_wav.shape[0] > 1:
                    bc_wav = bc_wav.mean(dim=0, keepdim=True)
                start = max(bc_pos[i], cursor)
                pad_size = bc_pos[i + 1] - start - bc_wav.size(1)
                start += cs.generate_delay(pad_size, mode="backchannel")
                if pad_size <= 0 and i == len(bcs) - 1 and u.next_uttr_type == "interrupt":
                    continue
                listener, host_wav = cs.place_backchannel(listener, host_wav, bc_wav, start)
                cursor = start + bc_wav.size(1)
                uttered.append({"speech": bc_wav, "text": bc["text"]})
                if cs.SAVE_ALIGN_JSON:
                    align_bc.append(
                        {
                            "host_idx": host_idx,
                            "speaker": u.speaker_idx,
                            "text": bc["text"],
                            "local_start_sec": start / cs.TARGET_SR,
                            "length_sec": bc_wav.size(1) / cs.TARGET_SR,
                            "words": bc.get("words", []),
                        }
                    )
        total_speech_meta_pending = (
            [{"idx": i, "tts_text": b["text"]} for i, b in enumerate(uttered)] if uttered else None
        )
        for b in uttered:
            backchannel_list.append(b["speech"].clone())

        channels = [None, None]
        channels[u.speaker_idx] = host_wav
        channels[1 - u.speaker_idx] = listener
        total_speech.append(cs.torch.cat(channels, dim=0))
        entry = {
            "uttr_type": "interrupt" if u.uttr_type == "interrupt" else None,
            "speaker": u.speaker_idx,
            "tts_text": (u.tts_text or "").strip(),
            "boundary": "turn",
        }
        if total_speech_meta_pending:
            entry["backchannels"] = total_speech_meta_pending
        total_speech_meta.append(entry)

    merged = cs.aggregate_speech(total_speech, total_speech_meta)
    merged = merged.flip(dims=[0])
    total_speech = [s.flip(dims=[0]) for s in total_speech]
    total_speech_meta_speakers = ["assistant", "user"]
    for sm in total_speech_meta:
        spk = sm.get("speaker")
        if spk in (0, 1):
            sm["speaker"] = 1 - spk

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dialogues").mkdir(parents=True, exist_ok=True)
    cs.torchaudio.save(str(out_dir / "dialogues" / "dialogue.wav"), merged, cs.TARGET_SR)
    for ch, name in enumerate(total_speech_meta_speakers):
        if merged.size(0) > ch:
            cs.torchaudio.save(str(out_dir / f"{name}.wav"), merged[ch : ch + 1], cs.TARGET_SR)

    (out_dir / "utterances").mkdir(parents=True, exist_ok=True)
    (out_dir / "backchannels").mkdir(parents=True, exist_ok=True)
    for idx, speech in enumerate(total_speech):
        cs.torchaudio.save(str(out_dir / "utterances" / f"{idx:02d}.wav"), speech, cs.TARGET_SR)
    for i, bc_speech in enumerate(backchannel_list):
        cs.torchaudio.save(str(out_dir / "backchannels" / f"{i:03d}.wav"), bc_speech, cs.TARGET_SR)

    meta = {
        "dialogue_id": m.dialogue_id,
        "variant_idx": m.variant_idx,
        "speakers": total_speech_meta_speakers,
        "speech_meta": total_speech_meta,
        "timing_config": _timing_config(cs),
        "user_prompt_wav": m.voice_picks["user"].wav,
        "assistant_prompt_wav": m.voice_picks["assistant"].wav,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    if cs.SAVE_ALIGN_JSON:
        payloads = cs._build_alignment_payloads(
            m.dialogue_id,
            m.variant_idx,
            total_speech_meta,
            {"utterances": align_utt, "backchannels": align_bc},
            total_speech,
        )
        with open(out_dir / "alignment_user.json", "w", encoding="utf-8") as fh:
            json.dump(payloads["user"], fh, indent=4, ensure_ascii=False)
        with open(out_dir / "alignment_assistant.json", "w", encoding="utf-8") as fh:
            json.dump(payloads["assistant"], fh, indent=4, ensure_ascii=False)


def _cleanup(vdir: Path) -> None:
    import shutil

    for name in ("units", "bcs", "hosts_trimmed", "bcs_trimmed"):
        shutil.rmtree(vdir / name, ignore_errors=True)


def iter_manifests(root: Path) -> List[Path]:
    return sorted(Path(p) for p in glob(str(root / "**" / "manifest.json"), recursive=True))


def main(config_path=None, limit: int = 0) -> None:
    cfg = load_config(config_path)
    s5 = resolve_stage5(cfg)
    root = Path(work_root(cfg))
    audio_root = Path(cfg_get(cfg, "paths.audio_root", "data/vi_audio"))
    cleanup = bool(s5.get("assemble", {}).get("cleanup_intermediate", True))
    audio_cfg = s5.get("assemble", {}).get("audio", {})

    import convert_spoken as cs

    cs._apply_tts_config(cfg)

    manifests = iter_manifests(root)
    if not manifests:
        raise SystemExit(f"No manifests under {root}. Run 5.1a/5.1b/5.2a first.")
    if limit:
        manifests = manifests[:limit]

    n = 0
    for mpath in manifests:
        m = read_manifest(mpath)
        vdir = mpath.parent
        if not (vdir / "alignment.json").is_file():
            print(f"[skip] {m.rel_path}/var{m.variant_idx:02d}: no alignment.json (run 5.2a)")
            continue
        align = _load_alignment(vdir)
        out_dir = audio_root / m.rel_path / f"var{m.variant_idx:02d}"
        assemble_variant(cs, m, align, vdir, out_dir, audio_cfg)
        if cleanup:
            _cleanup(vdir)
        n += 1
        print(f"[5.2b] {m.rel_path}/var{m.variant_idx:02d} -> {out_dir}")

    print(f"Stage 5.2b done: {n} variants assembled. Output root -> {audio_root}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Stage 5.2b: assemble final audio (CPU).")
    _ap.add_argument("--config", default=None, help="Path to config.yaml")
    _ap.add_argument("--limit", type=int, default=0, help="cap variants (0 = all)")
    _args = _ap.parse_args()
    main(_args.config, _args.limit)
