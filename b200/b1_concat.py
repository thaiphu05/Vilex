#!/usr/bin/env python3
"""b200 stage 1 — unit wavs + align txt -> per-turn stereo wavs + turn_map.json.

Standalone (no repo imports): numpy + soundfile required, scipy optional (it is
only used for a better resampler; a linear-interpolation fallback runs without).

Inputs (flat folders produced by the OmniVoice render + the align dump):

    <wav-dir>/<conv>__<unit_id>.wav
    <align-dir>/<conv>__<unit_id>.txt      # "<start> <end> <word>" per line

with ``<conv> = <rel_path with '/' -> '__'>__var<NN>`` (the exact naming the
user's align dump uses). The manifest is the only source of speaker/channel:
``speaker_idx`` 0 = user = channel 0 = left, 1 = assistant = agent = channel 1
= right. Nothing is flipped anywhere.

Per host turn the units are concatenated (pause units become the sampled pause
silence), backchannels are mixed into the listener channel at the host's
``word_count`` anchor (same placement rule as the old split assembler) and the
turn is stored as a stereo ``[user, assistant]`` wav. No gap/pause/overlap
between turns is applied here — that is stage b2.

Writes to ``<STATE_DIR>/<conv>/``:
    turn_000.wav ...    stereo float32, [user, assistant], bcs placed
    turn_map.json       turn order, speakers, per-turn word records (turn-local)

No CLI: set the paths in the PATHS block below, then run ``python b1_concat.py``.
"""

import json
import random
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

# ---- PATHS (insert B200 paths) ---------------------------------------------
MANIFEST_ROOT = ""  # dir walked for **/manifest.json
MANIFEST_FILE = ""  # single manifest.json; wins over MANIFEST_ROOT
WAV_DIR = ""
ALIGN_DIR = ""
OUT_DIR = ""
STATE_DIR = ""  # "" -> OUT_DIR/_stage
NAME_PREFIX = ""  # prepended to conversation names
LIMIT = 0  # 0 = all
# ----------------------------------------------------------------------------

SAMPLE_RATE = 24000

# ---- TUNING (edit here, no CLI flags) --------------------------------------
# Intra-utterance [PAUSE] units: exponential, clipped [min, max] seconds.
# Same values as config_example.yaml stage5_2b_assemble.timing.intra_pause.
MISSING_UNIT_SEC = 0.2
PAUSE_EXP_SCALE = 0.30
PAUSE_MIN_SEC = 0.10
PAUSE_MAX_SEC = 1.00
BC_ATTENUATION = 0.8
SEED_SALT = 0x5F15

SPEAKER_NAMES = {0: "user", 1: "agent"}


# --------------------------------------------------------------- audio helpers
def resample_audio(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float32 array; scipy if available, else linear interp."""
    if sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target_sr))
        return resample_poly(x, int(target_sr) // g, int(sr) // g).astype(np.float32)
    except Exception:
        n_out = int(round(len(x) * target_sr / sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(xq, xp, x).astype(np.float32)


def load_mono(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return resample_audio(mono, sr, target_sr)


def read_words_txt(path: Path) -> List[Dict[str, Any]]:
    """Parse ``<start> <end> <word>`` lines (user's align txt format)."""
    words: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        words.append({"word": " ".join(parts[2:]), "start": start, "end": end, "src": "align"})
    return words


def even_words(text: str, start_sec: float, end_sec: float) -> List[Dict[str, Any]]:
    tokens = (text or "").split()
    if not tokens:
        return []
    width = (end_sec - start_sec) / len(tokens)
    return [
        {
            "word": tok,
            "start": start_sec + i * width,
            "end": start_sec + (i + 1) * width,
            "src": "fallback",
        }
        for i, tok in enumerate(tokens)
    ]


def place_backchannel(own: np.ndarray, other: np.ndarray, bc: np.ndarray, start: int):
    """Write ``bc`` into the listener track, growing both tracks to fit."""
    start = max(0, int(start))
    overrun = start + bc.size - other.size
    if overrun > 0:
        other = np.pad(other, (0, overrun))
        own = np.pad(own, (0, overrun))
    other[start : start + bc.size] = bc
    return own, other


# ------------------------------------------------------------------ b200 plumbing
def conv_name(rel_path: str, variant_idx: int, prefix: str = "") -> str:
    base = rel_path.replace("/", "__")
    return f"{prefix}{base}__var{int(variant_idx):02d}"


def iter_manifests(manifest_root: Optional[str], manifest_file: Optional[str]) -> List[Path]:
    if manifest_file:
        return [Path(manifest_file)]
    if not manifest_root:
        raise SystemExit("set MANIFEST_ROOT (or MANIFEST_FILE) at the top of this file")
    return sorted(
        Path(p) for p in glob(str(Path(manifest_root) / "**" / "manifest.json"), recursive=True)
    )


class AudioIndex:
    """name -> path for a flat folder; missing names resolve to None."""

    def __init__(self, folder: Optional[str], suffix: str) -> None:
        self.by_name: Dict[str, Path] = {}
        if folder:
            for p in Path(folder).iterdir():
                if p.is_file() and p.suffix == suffix:
                    self.by_name[p.stem] = p

    def get(self, conv: str, unit_id: str) -> Optional[Path]:
        return self.by_name.get(f"{conv}__{unit_id}")


def merge_words(records: List[Dict[str, Any]], speaker: str) -> List[Dict[str, Any]]:
    out = []
    for w in sorted(records, key=lambda x: float(x["start"])):
        out.append(
            {
                "speaker": speaker,
                "word": w["word"],
                "start": round(float(w["start"]), 6),
                "end": round(float(w["end"]), 6),
                "src": w.get("src", "align"),
            }
        )
    return out


def build_turn(
    m: Dict[str, Any],
    u: Dict[str, Any],
    idx: int,
    conv: str,
    wavs: AudioIndex,
    aligns: AudioIndex,
    rng: random.Random,
    sr: int,
    pause_scale: float,
    pause_min: float,
    pause_max: float,
) -> Dict[str, Any]:
    pieces: List[np.ndarray] = []
    unit_meta: List[Dict[str, Any]] = []
    offset = 0
    host_words: List[Dict[str, Any]] = []

    for un in u.get("units", []):
        unit_id = un["unit_id"]
        if un.get("kind") == "pause":
            n = int(m.get("pause_samples", {}).get(unit_id, 0))
            if n <= 0:
                sec = float(
                    np.clip(rng.expovariate(1.0 / max(pause_scale, 1e-6)), pause_min, pause_max)
                )
                n = int(sec * sr)
            seg = np.zeros(n, dtype=np.float32)
            kind = "pause"
        else:
            path = wavs.get(conv, unit_id)
            if path is None:
                print(f"[b1] WARN {conv}: missing wav for {unit_id}, {MISSING_UNIT_SEC}s silence")
                seg = np.zeros(int(MISSING_UNIT_SEC * sr), dtype=np.float32)
                words = []
            else:
                seg = load_mono(path, sr)
                if seg.size == 0:
                    print(f"[b1] WARN {conv}: empty wav for {unit_id}, {MISSING_UNIT_SEC}s silence")
                    seg = np.zeros(int(MISSING_UNIT_SEC * sr), dtype=np.float32)
                    words = []
                else:
                    align_path = aligns.get(conv, unit_id)
                    words = read_words_txt(align_path) if align_path else []
                    if not words:
                        words = even_words(un.get("text", ""), 0.0, seg.size / sr)
            kind = "text"
            host_words.extend(
                {
                    "word": w["word"],
                    "start": offset / sr + w["start"],
                    "end": offset / sr + w["end"],
                    "src": w.get("src", "fallback"),
                }
                for w in words
            )
        unit_meta.append(
            {
                "unit_id": unit_id,
                "kind": kind,
                "start_sec": round(offset / sr, 6),
                "end_sec": round((offset + seg.size) / sr, 6),
            }
        )
        offset += seg.size
        pieces.append(seg)

    host = (
        np.concatenate(pieces) if pieces else np.zeros(int(MISSING_UNIT_SEC * sr), dtype=np.float32)
    )
    own = host.copy()
    other = np.zeros_like(own)

    host_speaker = SPEAKER_NAMES.get(int(u["speaker_idx"]), "user")
    bc_records: List[Dict[str, Any]] = []
    bcs = [bc for bc in m.get("backchannels", []) if int(bc.get("host_idx", -1)) == idx]
    if bcs and host_words:
        positions: List[int] = []
        for bc in bcs:
            word_idx = max(0, min(int(bc.get("word_count", 1)) - 1, len(host_words) - 2))
            positions.append(round(host_words[word_idx]["end"] * sr))
        positions.append(host.size)
        cursor = 0
        for i, bc in enumerate(bcs):
            bc_path = wavs.get(conv, bc["unit_id"])
            if bc_path is None:
                print(f"[b1] WARN {conv}: missing bc wav {bc['unit_id']}, skipped")
                continue
            bc_wav = load_mono(bc_path, sr) * BC_ATTENUATION
            start = max(positions[i], cursor)
            pad_size = positions[i + 1] - start - bc_wav.size
            if pad_size <= 0 and i == len(bcs) - 1 and u.get("next_uttr_type") == "interrupt":
                continue
            own, other = place_backchannel(own, other, bc_wav, start)
            cursor = start + bc_wav.size
            bc_speaker = SPEAKER_NAMES.get(int(bc.get("speaker_idx", 0)), "user")
            bc_txt = aligns.get(conv, bc["unit_id"])
            bc_words = read_words_txt(bc_txt) if bc_txt else []
            if not bc_words:
                bc_words = even_words(bc.get("text", ""), 0.0, bc_wav.size / sr)
            bc_records.extend(
                {
                    "speaker": bc_speaker,
                    "word": w["word"],
                    "start": round(start / sr + w["start"], 6),
                    "end": round(start / sr + w["end"], 6),
                    "src": w.get("src", "fallback"),
                }
                for w in bc_words
            )
    elif bcs:
        print(f"[b1] WARN {conv}: no host words to anchor {len(bcs)} bc(s), skipped")

    if int(u["speaker_idx"]) == 0:
        channels = np.stack([own, other])
    else:
        channels = np.stack([other, own])

    words = merge_words(host_words, host_speaker) + bc_records
    words.sort(key=lambda w: float(w["start"]))
    return {
        "turn_idx": idx,
        "speaker_idx": int(u["speaker_idx"]),
        "speaker": host_speaker,
        "uttr_type": u.get("uttr_type"),
        "next_uttr_type": u.get("next_uttr_type"),
        "dur_sec": round(host.size / sr, 6),
        "units": unit_meta,
        "words": words,
        "backchannels": [b["unit_id"] for b in bcs],
        "audio": channels,
    }


def process_manifest(
    mpath: Path,
    wavs: AudioIndex,
    aligns: AudioIndex,
    state: Path,
    prefix: str,
    sr: int,
    pause_scale: float,
    pause_min: float,
    pause_max: float,
) -> None:
    m = json.loads(mpath.read_text(encoding="utf-8"))
    rel = m["rel_path"]
    conv = conv_name(rel, m["variant_idx"], prefix)
    seed = int(m.get("seed", 42)) + int(m["variant_idx"]) + SEED_SALT
    rng = random.Random(seed)

    turn_dir = state / conv
    turn_dir.mkdir(parents=True, exist_ok=True)
    turns: List[Dict[str, Any]] = []

    for idx, u in enumerate(m.get("utterances", [])):
        rec = build_turn(m, u, idx, conv, wavs, aligns, rng, sr, pause_scale, pause_min, pause_max)
        channels = rec.pop("audio")
        sf.write(str(turn_dir / f"turn_{idx:03d}.wav"), channels.T, sr, subtype="FLOAT")
        turns.append(rec)

    payload = {
        "conversation": conv,
        "dialogue_id": m.get("dialogue_id", ""),
        "variant_idx": int(m["variant_idx"]),
        "seed": int(m.get("seed", 42)),
        "sample_rate": sr,
        "turns": turns,
    }
    with open(turn_dir / "turn_map.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[b1] {conv}: {len(turns)} turns -> {turn_dir}")


def main() -> None:
    out = Path(OUT_DIR)
    state = Path(STATE_DIR) if STATE_DIR else out / "_stage"
    manifests = iter_manifests(MANIFEST_ROOT, MANIFEST_FILE)
    if LIMIT:
        manifests = manifests[:LIMIT]
    if not manifests:
        raise SystemExit("no manifest.json found (check MANIFEST_ROOT / MANIFEST_FILE)")
    wavs = AudioIndex(WAV_DIR, ".wav")
    aligns = AudioIndex(ALIGN_DIR, ".txt")

    for mpath in manifests:
        try:
            process_manifest(
                mpath,
                wavs,
                aligns,
                state,
                NAME_PREFIX,
                SAMPLE_RATE,
                PAUSE_EXP_SCALE,
                PAUSE_MIN_SEC,
                PAUSE_MAX_SEC,
            )
        except Exception as exc:  # keep the batch alive on one bad dialogue
            print(f"[b1] ERROR {mpath}: {exc}", file=sys.stderr)
    print(f"[b1] done. state -> {state}")


if __name__ == "__main__":
    main()
