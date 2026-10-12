#!/usr/bin/env python3
"""b200 stage 2 — per-turn wavs -> final stereo conversation.wav (+ timing.json).

Standalone (numpy + soundfile; pyloudnorm optional). Ports the old split
assembler's timing math, with the exponential gap/pause distributions from
``config_example.yaml`` (the old code's ``_sample_gap``/``_sample_pause`` were
stubs returning constants; the config values live here now):

  * speaker change -> gap ~ Exp(scale) clipped [min, max] (defaults below)
  * same speaker   -> pause ~ Exp(scale) clipped [min, max] (defaults below)
  * speaker change, next turn is an interrupt -> overlap ~N(0.45, 0.05) s, mix-add
  * speaker change (assistant -> user) with prob ``user_interrupt_prob`` and no
    leading backchannel in the current turn -> overlap (<= cap), else gap

Channels are never flipped: channel 0 = user = left, channel 1 = agent = right.
Loudness is normalized (LUFS target, peak fallback when pyloudnorm is missing).

A conversation whose merged track is silent (peak <= ``SILENCE_PEAK_EPS``) is
reported and NOT written, so a silent file never lands in the dataset; partial
silence (some turns with no audio) is reported as a warning per conversation.

Writes:
    <OUT_DIR>/<conv>/conversation.wav    stereo float32
    <STATE_DIR>/<conv>/timing.json       per-turn start_sample/timing/duration

No CLI: set the paths in the PATHS block below, then run ``python b2_assemble.py``.
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
OUT_DIR = ""
STATE_DIR = ""  # "" -> OUT_DIR/_stage
NAME_PREFIX = ""  # prepended to conversation names
LIMIT = 0  # 0 = all
# ----------------------------------------------------------------------------

# ---- TUNING (edit here, no CLI flags) --------------------------------------
# Same values as config_example.yaml stage5_2b_assemble.timing.
GAP_EXP_SCALE = 0.7067
GAP_MIN_SEC = 0.003
GAP_MAX_SEC = 2.50
PAUSE_EXP_SCALE = 0.6780
PAUSE_MIN_SEC = 0.001
PAUSE_MAX_SEC = 4.00
USER_INTERRUPT_OVERLAP_SEC = 0.64
USER_INTERRUPT_PROB = 0.5
NOISE_FLOOR_AMP = 0.0
TARGET_LUFS = -23.0
SAMPLE_RATE = 24000
# A track whose peak is at/below this counts as silence (float32 samples).
SILENCE_PEAK_EPS = 1e-6
# ----------------------------------------------------------------------------
LEAD_BC_CHECK_SEC = 2.0
LEAD_BC_EPS = 1e-6
SEED_SALT = 0x5F15


def peak_of(wav: np.ndarray) -> float:
    return float(np.abs(wav).max()) if wav.size else 0.0


def conv_name(rel_path: str, variant_idx: int, prefix: str = "") -> str:
    return f"{prefix}{rel_path.replace('/', '__')}__var{int(variant_idx):02d}"


def iter_manifests(manifest_root: Optional[str], manifest_file: Optional[str]) -> List[Path]:
    if manifest_file:
        return [Path(manifest_file)]
    if not manifest_root:
        raise SystemExit("set MANIFEST_ROOT (or MANIFEST_FILE) at the top of this file")
    return sorted(
        Path(p) for p in glob(str(Path(manifest_root) / "**" / "manifest.json"), recursive=True)
    )


def noise_fill(n: int, amp: float, channels: int = 2) -> np.ndarray:
    if n <= 0:
        return np.zeros((channels, 0), dtype=np.float32)
    if amp <= 0:
        return np.zeros((channels, n), dtype=np.float32)
    return (np.random.randn(channels, n) * amp).astype(np.float32)


def loudness_normalize(wav: np.ndarray, sr: int, target: float) -> np.ndarray:
    """Normalize a (2, N) float32 track to ``target`` LUFS; peak fallback."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(wav.T)
        if not np.isfinite(loud) or loud < -80:
            return wav
        gain = 10 ** ((target - loud) / 20.0)
        return (wav * gain).astype(np.float32)
    except Exception:
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        return (wav / peak).astype(np.float32) if peak > 0 else wav


def sample_gap() -> float:
    """Turn-change gap: Exp(scale) clipped [min, max] (uses the seeded RNG)."""
    return float(
        np.clip(random.expovariate(1.0 / max(GAP_EXP_SCALE, 1e-6)), GAP_MIN_SEC, GAP_MAX_SEC)
    )


def sample_pause() -> float:
    """Same-speaker pause: Exp(scale) clipped [min, max] (uses the seeded RNG)."""
    return float(
        np.clip(random.expovariate(1.0 / max(PAUSE_EXP_SCALE, 1e-6)), PAUSE_MIN_SEC, PAUSE_MAX_SEC)
    )


def merge_turns(
    turns: List[np.ndarray],
    metas: List[Dict[str, Any]],
    sr: int,
) -> np.ndarray:
    merged = turns[0].copy()
    metas[0].update({"timing": "none", "duration_sec": 0.0, "start_sample": 0})

    for i in range(1, len(turns)):
        cur = turns[i]
        prev_meta, meta = metas[i - 1], metas[i]
        prev_len = merged.shape[1]
        overlap_max = int(USER_INTERRUPT_OVERLAP_SEC * sr)

        if meta["speaker_idx"] != prev_meta["speaker_idx"] and meta.get("uttr_type") == "interrupt":
            overlap = int(
                np.clip(np.random.normal(loc=0.45, scale=0.05) * sr, 0, min(prev_len, cur.shape[1]))
            )
            if overlap <= 0:
                merged = np.concatenate((merged, cur), axis=1)
                meta.update({"timing": "none", "duration_sec": 0.0, "start_sample": prev_len})
                continue
            merged = np.concatenate(
                (merged[:, :-overlap], merged[:, -overlap:] + cur[:, :overlap], cur[:, overlap:]),
                axis=1,
            )
            meta.update(
                {
                    "timing": "overlap",
                    "duration_sec": round(overlap / sr, 4),
                    "start_sample": max(0, prev_len - overlap),
                }
            )
            continue

        if meta["speaker_idx"] != prev_meta["speaker_idx"]:
            prev_assistant = prev_meta["speaker_idx"] == 1
            curr_user = meta["speaker_idx"] == 0
            if prev_assistant and curr_user and random.random() < USER_INTERRUPT_PROB:
                lead = cur[1, : min(int(LEAD_BC_CHECK_SEC * sr), cur.shape[1])]
                has_leading_bc = bool(lead.size and float(np.abs(lead).max()) > LEAD_BC_EPS)
                overlap = 0 if has_leading_bc else min(overlap_max, prev_len, cur.shape[1])
                if overlap > 0:
                    merged = np.concatenate(
                        (
                            merged[:, :-overlap],
                            merged[:, -overlap:] + cur[:, :overlap],
                            cur[:, overlap:],
                        ),
                        axis=1,
                    )
                    meta.update(
                        {
                            "timing": "overlap",
                            "duration_sec": round(overlap / sr, 4),
                            "start_sample": max(0, prev_len - overlap),
                        }
                    )
                    continue
            gap = sample_gap()
            gap_samples = int(gap * sr)
            merged = np.concatenate((merged, noise_fill(gap_samples, NOISE_FLOOR_AMP), cur), axis=1)
            meta.update(
                {
                    "timing": "gap",
                    "duration_sec": round(gap, 4),
                    "start_sample": prev_len + gap_samples,
                }
            )
            continue

        if meta.get("boundary", "turn") == "turn":
            pause = sample_pause()
            pause_samples = int(pause * sr)
            merged = np.concatenate(
                (merged, noise_fill(pause_samples, NOISE_FLOOR_AMP), cur), axis=1
            )
            meta.update(
                {
                    "timing": "pause",
                    "duration_sec": round(pause, 4),
                    "start_sample": prev_len + pause_samples,
                }
            )
        else:
            merged = np.concatenate((merged, cur), axis=1)
            meta.update({"timing": "none", "duration_sec": 0.0, "start_sample": prev_len})

    return merged


def process_conversation(conv: str, state: Path, out_dir: Path) -> bool:
    """Assemble one conversation. Returns True when conversation.wav was written."""
    turn_dir = state / conv
    with open(turn_dir / "turn_map.json", "r", encoding="utf-8") as fh:
        tmap = json.load(fh)
    turns_meta = tmap["turns"]
    seed = int(tmap.get("seed", 42)) + int(tmap.get("variant_idx", 0)) + SEED_SALT
    random.seed(seed)
    np.random.seed(seed)

    turns: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    silent_turns = 0
    for rec in turns_meta:
        path = turn_dir / f"turn_{int(rec['turn_idx']):03d}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; rerun b1")
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        if file_sr != SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate {file_sr} != {SAMPLE_RATE}")
        if wav.shape[1] == 1:
            wav = np.repeat(wav, 2, axis=1)
        if peak_of(wav.T) <= SILENCE_PEAK_EPS:
            silent_turns += 1
        turns.append(wav.T)
        metas.append(
            {
                "turn_idx": int(rec["turn_idx"]),
                "speaker_idx": int(rec["speaker_idx"]),
                "speaker": rec["speaker"],
                "uttr_type": rec.get("uttr_type"),
                "boundary": "turn",
            }
        )

    if not turns:
        print(f"[b2] WARN {conv}: no turns, skipped")
        return False

    merged = merge_turns(turns, metas, SAMPLE_RATE)
    merged = loudness_normalize(merged, SAMPLE_RATE, TARGET_LUFS)

    if peak_of(merged) <= SILENCE_PEAK_EPS:
        print(
            f"[b2] WARN {conv}: conversation is silent "
            f"({silent_turns}/{len(turns)} silent turns), conversation.wav not written"
        )
        return False
    if silent_turns:
        print(f"[b2] WARN {conv}: {silent_turns}/{len(turns)} turns are silent")

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "conversation.wav"), merged.T, SAMPLE_RATE, subtype="FLOAT")

    timing = {
        "conversation": conv,
        "sample_rate": SAMPLE_RATE,
        "turns": [
            {
                **meta,
                "start_sec": round(meta["start_sample"] / SAMPLE_RATE, 6),
                "end_sec": round(
                    (meta["start_sample"] + turns[meta["turn_idx"]].shape[1]) / SAMPLE_RATE, 6
                ),
            }
            for meta in metas
        ],
    }
    with open(turn_dir / "timing.json", "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=2, ensure_ascii=False)
    print(f"[b2] {conv}: {merged.shape[1] / SAMPLE_RATE:.2f}s -> {out_dir / 'conversation.wav'}")
    return True


def main() -> None:
    out = Path(OUT_DIR)
    state = Path(STATE_DIR) if STATE_DIR else out / "_stage"
    manifests = iter_manifests(MANIFEST_ROOT, MANIFEST_FILE)
    if LIMIT:
        manifests = manifests[:LIMIT]
    if not manifests:
        raise SystemExit("no manifest.json found (check MANIFEST_ROOT / MANIFEST_FILE)")

    written = skipped = failed = 0
    for mpath in manifests:
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            conv = conv_name(m["rel_path"], m["variant_idx"], NAME_PREFIX)
            if process_conversation(conv, state, out / conv):
                written += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[b2] ERROR {mpath}: {exc}", file=sys.stderr)
    print(f"[b2] done. output -> {out} (written {written}, skipped {skipped}, failed {failed})")


if __name__ == "__main__":
    main()
