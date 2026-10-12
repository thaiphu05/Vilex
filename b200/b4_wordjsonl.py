#!/usr/bin/env python3
"""b200 stage 4 — word.jsonl (absolute word timestamps over conversation.wav).

Standalone and stdlib-only (json + pathlib): reads the JSON state written by b1
(``turn_map.json``, word times relative to each turn start) and b2
(``timing.json``, each turn's absolute start in the merged timeline) and emits
one JSON object per word:

    {"speaker": "user"|"agent", "word": "...", "start": 1.2345, "end": 1.4567}

Times are absolute in ``conversation.wav`` (gap/pause/overlap already included
via the turn start offsets), rounded to ``PRECISION`` decimals, sorted by
start. ``alignment_status`` is not carried over.

Writes:
    <OUT_DIR>/<conv>/word.jsonl

No CLI: set the paths in the PATHS block below, then run ``python b4_wordjsonl.py``.
"""

import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- PATHS (insert B200 paths) ---------------------------------------------
MANIFEST_ROOT = ""  # dir walked for **/manifest.json
MANIFEST_FILE = ""  # single manifest.json; wins over MANIFEST_ROOT
OUT_DIR = ""
STATE_DIR = ""  # "" -> OUT_DIR/_stage
NAME_PREFIX = ""  # prepended to conversation names
LIMIT = 0  # 0 = all
# ----------------------------------------------------------------------------

# ---- TUNING (edit here, no CLI flags) --------------------------------------
PRECISION = 4  # decimals for start/end seconds
# ----------------------------------------------------------------------------


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


def process_conversation(conv: str, state: Path, out_dir: Path) -> None:
    turn_dir = state / conv
    with open(turn_dir / "turn_map.json", "r", encoding="utf-8") as fh:
        tmap = json.load(fh)
    timing_path = turn_dir / "timing.json"
    if not timing_path.is_file():
        print(f"[b4] WARN {conv}: no timing.json (run b2 first), skipped")
        return
    with open(timing_path, "r", encoding="utf-8") as fh:
        timing = json.load(fh)

    starts: Dict[int, float] = {int(t["turn_idx"]): float(t["start_sec"]) for t in timing["turns"]}
    rows: List[Dict[str, Any]] = []
    for rec in tmap["turns"]:
        offset = starts.get(int(rec["turn_idx"]))
        if offset is None:
            print(f"[b4] WARN {conv}: turn {rec['turn_idx']} missing in timing.json")
            continue
        for w in rec.get("words", []):
            start = offset + float(w["start"])
            end = offset + float(w["end"])
            rows.append(
                {
                    "speaker": w.get("speaker", rec["speaker"]),
                    "word": w["word"],
                    "start": round(start, PRECISION),
                    "end": round(end, PRECISION),
                }
            )

    rows.sort(key=lambda r: (r["start"], r["end"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "word.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[b4] {conv}: {len(rows)} words -> {out_dir / 'word.jsonl'}")


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
            print(f"[b4] ERROR {mpath}: {exc}", file=sys.stderr)
    print(f"[b4] done. output -> {out}")


if __name__ == "__main__":
    main()
