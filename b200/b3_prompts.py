#!/usr/bin/env python3
"""b200 stage 3 — voice prompts per channel (left = user, right = agent).

Standalone (numpy + soundfile). For each conversation and each channel:

  1. collect that speaker's turn tracks (own channel of the stereo turn wav,
     so no cross-speaker bleed) in dialogue order, dropping silent ones;
  2. if a single turn is longer than ``MIN_SEC`` (default 6 s) it is written
     as-is (never trimmed);
  3. otherwise merge consecutive same-channel turns separated by
     ``JOIN_SILENCE_SEC`` (default 0.25 s) until the merged clip is longer
     than ``MIN_SEC``;
  4. still shorter than ``MIN_SEC`` -> skip the file, keep the batch going.

All thresholds are constants at the top of this file (see TUNING), no CLI flags.

Channel comes from the manifest: speaker_idx 0 = user = channel 0 = LEFT,
speaker_idx 1 = agent/assistant = channel 1 = RIGHT. Never borrows the other
channel's voice.

Writes:
    <OUT_DIR>/<conv>/voice_prompt_left.wav     (user,  mono, float32)
    <OUT_DIR>/<conv>/voice_prompt_right.wav    (agent, mono, float32)

No CLI: set the paths in the PATHS block below, then run ``python b3_prompts.py``.
"""

import json
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# ---- PATHS (insert B200 paths) ---------------------------------------------
MANIFEST_ROOT = ""  # dir walked for **/manifest.json
MANIFEST_FILE = ""  # single manifest.json; wins over MANIFEST_ROOT
OUT_DIR = ""
STATE_DIR = ""  # "" -> OUT_DIR/_stage
NAME_PREFIX = ""  # prepended to conversation names
LIMIT = 0  # 0 = all
# ----------------------------------------------------------------------------

SAMPLE_RATE = 24000
# ---- TUNING (edit here, no CLI flags) --------------------------------------
MIN_SEC = 6.0
JOIN_SILENCE_SEC = 0.25
MIN_RMS = 1e-4
SINGLE_FIRST = True  # prefer a single turn longer than MIN_SEC before merging
# ----------------------------------------------------------------------------

CHANNELS = [(0, "left"), (1, "right")]


def iter_manifests(manifest_root: Optional[str], manifest_file: Optional[str]) -> List[Path]:
    if manifest_file:
        return [Path(manifest_file)]
    if not manifest_root:
        raise SystemExit("set MANIFEST_ROOT (or MANIFEST_FILE) at the top of this file")
    return sorted(
        Path(p) for p in glob(str(Path(manifest_root) / "**" / "manifest.json"), recursive=True)
    )


def conv_name(rel_path: str, variant_idx: int, prefix: str = "") -> str:
    return f"{prefix}{rel_path.replace('/', '__')}__var{int(variant_idx):02d}"


def merge_segments(
    segments: List[np.ndarray], sr: int, min_sec: float, join_sec: float
) -> Optional[np.ndarray]:
    """Concat same-channel segments (+join silence) until longer than ``min_sec``.

    Returns None when the available audio never crosses the threshold.
    """
    out: Optional[np.ndarray] = None
    silence = np.zeros(int(join_sec * sr), dtype=np.float32)
    for seg in segments:
        if out is None:
            out = seg.copy()
        else:
            if silence.size:
                out = np.concatenate((out, silence), axis=0)
            out = np.concatenate((out, seg), axis=0)
        if out.size / sr > min_sec:
            return out
    return None


def pick_prompt(
    segments: List[np.ndarray],
    sr: int,
    min_sec: float,
    join_sec: float,
    single_first: bool,
) -> Tuple[Optional[np.ndarray], str]:
    if single_first:
        for seg in segments:
            if seg.size / sr > min_sec:
                return seg, f"single {seg.size / sr:.2f}s"
    merged = merge_segments(segments, sr, min_sec, join_sec)
    if merged is not None:
        return merged, f"merged {merged.size / sr:.2f}s"
    if not single_first:
        for seg in segments:
            if seg.size / sr > min_sec:
                return seg, f"single {seg.size / sr:.2f}s"
    return None, "shorter than min_sec"


def process_conversation(conv: str, state: Path, out_dir: Path) -> None:
    turn_dir = state / conv
    with open(turn_dir / "turn_map.json", "r", encoding="utf-8") as fh:
        tmap = json.load(fh)

    per_channel: Dict[int, List[np.ndarray]] = {0: [], 1: []}
    for rec in tmap["turns"]:
        ch = int(rec["speaker_idx"])
        path = turn_dir / f"turn_{int(rec['turn_idx']):03d}.wav"
        if not path.is_file():
            print(f"[b3] WARN {conv}: missing {path.name}")
            continue
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        if file_sr != SAMPLE_RATE or wav.shape[1] <= ch:
            continue
        track = wav[:, ch].astype(np.float32)
        rms = float(np.sqrt(np.mean(track**2))) if track.size else 0.0
        if rms < MIN_RMS:
            continue
        per_channel[ch].append(track)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ch, side in CHANNELS:
        segments = per_channel[ch]
        if not segments:
            print(f"[b3] WARN {conv}: voice_prompt_{side} skipped (no {side} audio)")
            continue
        prompt, how = pick_prompt(segments, SAMPLE_RATE, MIN_SEC, JOIN_SILENCE_SEC, SINGLE_FIRST)
        if prompt is None:
            total = sum(s.size for s in segments) / SAMPLE_RATE
            print(
                f"[b3] WARN {conv}: voice_prompt_{side} skipped "
                f"({how}, {len(segments)} turns, {total:.2f}s total)"
            )
            continue
        if prompt.ndim > 1:
            prompt = prompt.mean(axis=1)
        sf.write(str(out_dir / f"voice_prompt_{side}.wav"), prompt, SAMPLE_RATE, subtype="FLOAT")
        print(f"[b3] {conv}: voice_prompt_{side}.wav {prompt.size / SAMPLE_RATE:.2f}s ({how})")


def main() -> None:
    out = Path(OUT_DIR)
    state = Path(STATE_DIR) if STATE_DIR else out / "_stage"
    manifests = iter_manifests(MANIFEST_ROOT, MANIFEST_FILE)
    if LIMIT:
        manifests = manifests[:LIMIT]
    if not manifests:
        raise SystemExit("no manifest.json found (check MANIFEST_ROOT / MANIFEST_FILE)")

    for mpath in manifests:
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            conv = conv_name(m["rel_path"], m["variant_idx"], NAME_PREFIX)
            process_conversation(conv, state, out / conv)
        except Exception as exc:
            print(f"[b3] ERROR {mpath}: {exc}", file=sys.stderr)
    print(f"[b3] done. output -> {out}")


if __name__ == "__main__":
    main()
