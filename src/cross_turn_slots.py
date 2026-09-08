"""Cross-turn slot dictation: split slot values into multi-turn chunks with
optional misspeak-repair. Rule-based, 0 API calls.

Pipeline: Stage1 -> cross_turn_slots -> disfluency -> Stage4
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_RE_PHONE_VN = re.compile(
    r"(?:\+84|84)[\s.-]?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}"   # +84 xxx xxx xxx
    r"|0\d{9}"                                             # 0xxxxxxxxx
    r"|\(\d{3,4}\)[\s.-]?\d{3}[\s.-]?\d{3,4}"             # (xxx) xxx-xxxx
)
_RE_NUMERIC = re.compile(r"\d{6,}")
_RE_ALNUM = re.compile(r"[A-Z0-9]{5,}")

# Priority: email > phone > numeric > alnum
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("email", _RE_EMAIL),
    ("phone", _RE_PHONE_VN),
    ("numeric", _RE_NUMERIC),
    ("alnum", _RE_ALNUM),
]

# ---------------------------------------------------------------------------
# Vocalization (digit-by-digit)
# ---------------------------------------------------------------------------

_DIGITS_VI = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_DIGITS_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

_LETTERS_VI = {
    "A": "a", "B": "bờ", "C": "sờ", "D": "dờ", "E": "e", "F": "ép", "G": "giê",
    "H": "hát", "I": "i", "J": "gì", "K": "ka", "L": "e-lờ", "M": "mờ", "N": "nờ",
    "O": "ô", "P": "pê", "Q": "cu", "R": "e-rờ", "S": "e-xờ", "T": "tê",
    "U": "u", "V": "vê", "W": "đúp", "X": "ích", "Y": "y", "Z": "dét",
}
_LETTERS_EN = {
    "A": "a", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "ef", "G": "gee",
    "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "el", "M": "em", "N": "en",
    "O": "oh", "P": "pee", "Q": "cue", "R": "are", "S": "ess", "T": "tee",
    "U": "you", "V": "vee", "W": "double-u", "X": "ex", "Y": "why", "Z": "zee",
}


def _vocalize_digits(text: str, lang: str) -> str:
    """Convert digit string to spoken words, digit-by-digit."""
    digits = _DIGITS_VI if lang == "vi" else _DIGITS_EN
    return " ".join(digits[int(d)] for d in text if d.isdigit())


def _vocalize_letters(text: str, lang: str) -> str:
    """Convert letter string to spoken letter names."""
    table = _LETTERS_VI if lang == "vi" else _LETTERS_EN
    return " ".join(table.get(c, c) for c in text.upper() if c.isalpha())


def _vocalize_alnum_group(text: str, lang: str) -> str:
    """Vocalize a group of letters or digits."""
    if text.isdigit():
        return _vocalize_digits(text, lang)
    if text.isalpha():
        return _vocalize_letters(text, lang)
    parts = []
    for c in text:
        if c.isdigit():
            parts.append(_vocalize_digits(c, lang))
        elif c.isalpha():
            parts.append(_vocalize_letters(c, lang))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _segment_numeric(value: str) -> List[str]:
    """Split numeric string into chunks of 3-4 from left (0901 234 567)."""
    n = len(value)
    if n <= 4:
        return [value]
    # Calculate chunk sizes: prefer 3-4 per chunk, no chunk < 3
    chunks = []
    remaining = n
    while remaining > 0:
        if remaining <= 4:
            chunks.append(value[n - remaining:])
            break
        elif remaining == 5:
            chunks.append(value[n - remaining:n - remaining + 3])
            remaining -= 3
        elif remaining == 6:
            chunks.append(value[n - remaining:n - remaining + 3])
            remaining -= 3
        else:
            chunks.append(value[n - remaining:n - remaining + 4])
            remaining -= 4
    return chunks


def _segment_email(value: str, lang: str) -> List[str]:
    """Split email: local + at + domain parts with dot."""
    parts = value.split("@")
    if len(parts) != 2:
        return [value]
    local, domain = parts
    domain_parts = domain.split(".")
    segs = [local]
    at_word = "a-còng" if lang == "vi" else "at"
    dot_word = "chấm" if lang == "vi" else "dot"
    segs.append(at_word)
    for p in domain_parts:
        segs.append(p)
        segs.append(dot_word)
    return segs[:-1]


def _segment_alnum(value: str) -> List[str]:
    """Split alphanumeric code into letter/digit groups (AB12CD -> AB / 12 / CD)."""
    groups = []
    cur = ""
    for c in value:
        if c.isalpha() and cur and cur[-1].isdigit():
            groups.append(cur)
            cur = c
        elif c.isdigit() and cur and cur[-1].isalpha():
            groups.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        groups.append(cur)
    return groups


def _segment(value: str, slot_type: str, lang: str) -> List[str]:
    """Segment a slot value by type."""
    if slot_type in ("numeric", "phone"):
        return _segment_numeric(value)
    if slot_type == "email":
        return _segment_email(value, lang)
    if slot_type == "alnum":
        return _segment_alnum(value)
    return [value]


def _vocalize_segment(seg: str, slot_type: str, lang: str) -> str:
    """Convert a segment to spoken form."""
    if slot_type in ("numeric", "phone"):
        return _vocalize_digits(seg, lang)
    if slot_type == "email":
        return seg
    if slot_type == "alnum":
        return _vocalize_alnum_group(seg, lang)
    return seg


# ---------------------------------------------------------------------------
# Corruption rules
# ---------------------------------------------------------------------------

def _corrupt_numeric_chunk(chunk: str, rng: random.Random) -> str:
    """Alter one digit: (d+1) % 10."""
    idx = rng.randint(0, len(chunk) - 1)
    d = int(chunk[idx])
    new_d = (d + rng.randint(1, 9)) % 10
    return chunk[:idx] + str(new_d) + chunk[idx + 1:]


def _corrupt_letter_group(group: str, rng: random.Random) -> str:
    """Swap two adjacent letters."""
    if len(group) < 2:
        return group
    idx = rng.randint(0, len(group) - 2)
    return group[:idx] + group[idx + 1] + group[idx] + group[idx + 2:]


def _corrupt_alnum_group(group: str, rng: random.Random) -> str:
    """Corrupt one char in the group."""
    if len(group) < 2:
        return group
    if group.isdigit():
        return _corrupt_numeric_chunk(group, rng)
    if group.isalpha():
        return _corrupt_letter_group(group, rng)
    idx = rng.randint(0, len(group) - 1)
    c = group[idx]
    if c.isdigit():
        new_c = str((int(c) + rng.randint(1, 9)) % 10)
    else:
        new_c = chr(ord(c) + rng.randint(1, 2))
        if not new_c.isalpha():
            new_c = "Z"
    return group[:idx] + new_c + group[idx + 1:]


def _corrupt_segment(seg: str, slot_type: str, rng: random.Random) -> str:
    """Corrupt a segment by type-specific rule."""
    if slot_type in ("numeric", "phone"):
        return _corrupt_numeric_chunk(seg, rng)
    if slot_type == "email":
        if "@" in seg or "." in seg:
            return seg
        idx = rng.randint(0, len(seg) - 1)
        c = seg[idx]
        if c.isalpha():
            new_c = chr(ord(c) + rng.randint(1, 2))
            if not new_c.isalpha():
                new_c = "z"
            return seg[:idx] + new_c + seg[idx + 1:]
        return seg
    if slot_type == "alnum":
        return _corrupt_alnum_group(seg, rng)
    return seg


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "vi": {
        "ack_short": ["Ừ, đọc tiếp đi.", "Ừ, tiếp đi.", "Nghe rồi, tiếp đi.", "Được, tiếp."],
        "ack_final": ["Đã ghi nhận.", "Xong rồi.", "Cảm ơn."],
        "self_correct": "Khoan, ý tôi là {correct}.",
        "ack_correct": "Ừ, {correct} rồi, tiếp đi.",
    },
    "en": {
        "ack_short": ["Go on.", "Continue.", "Yes, go ahead."],
        "ack_final": ["Got it.", "Done.", "Thank you."],
        "self_correct": "Wait, I meant {correct}.",
        "ack_correct": "Got it, {correct}, go on.",
    },
}


# ---------------------------------------------------------------------------
# Core injection
# ---------------------------------------------------------------------------

def _build_dictation_turns(
    value: str,
    slot_type: str,
    rng: random.Random,
    lang: str,
    roles: str,
    perror: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Build dictation turn list for a slot value."""
    segs = _segment(value, slot_type, lang)
    if len(segs) < 2:
        vocal = _vocalize_segment(value, slot_type, lang)
        return [{"role": "user", "content": vocal}], []

    has_error = rng.random() < perror
    error_idx = rng.randint(0, len(segs) - 1) if has_error else -1

    turns: List[Dict[str, str]] = []
    meta_list: List[Dict[str, Any]] = []
    tpl = _TEMPLATES[lang]

    if roles == "both":
        dictator = rng.choice(["user", "assistant"])
    else:
        dictator = roles
    listener = "assistant" if dictator == "user" else "user"

    for i, seg in enumerate(segs):
        vocal = _vocalize_segment(seg, slot_type, lang)

        if i == error_idx:
            wrong = _corrupt_segment(seg, slot_type, rng)
            wrong_vocal = _vocalize_segment(wrong, slot_type, lang)
            correct_vocal = vocal
            turns.append({"role": dictator, "content": wrong_vocal})
            turns.append({"role": dictator, "content": tpl["self_correct"].format(correct=correct_vocal)})
            turns.append({"role": listener, "content": tpl["ack_correct"].format(correct=correct_vocal)})
            meta_list.append({
                "type": slot_type, "original": value, "chunk_idx": i,
                "chunk_original": seg, "chunk_corrupted": wrong,
                "chunks": segs, "error_idx": i,
            })
        else:
            turns.append({"role": dictator, "content": vocal})

        if i < len(segs) - 1:
            turns.append({"role": listener, "content": rng.choice(tpl["ack_short"])})

    turns.append({"role": listener, "content": rng.choice(tpl["ack_final"])})

    if not meta_list:
        meta_list.append({
            "type": slot_type, "original": value, "chunks": segs, "error_idx": -1,
        })

    return turns, meta_list


def _inject_slots_in_turn(
    content: str,
    rng: random.Random,
    lang: str,
    roles: str,
    min_digits: int,
    min_code_len: int,
    perror: float,
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, Any]]]:
    """Inject cross-turn dictation for slot values in a single turn."""
    matches: List[Tuple[int, int, str, str]] = []
    for slot_type, pattern in _PATTERNS:
        for m in pattern.finditer(content):
            val = m.group()
            if slot_type == "numeric" and len(val) < min_digits:
                continue
            if slot_type == "alnum":
                if not any(c.isalpha() for c in val) or not any(c.isdigit() for c in val):
                    continue
                if len(val) < min_code_len:
                    continue
            matches.append((m.start(), m.end(), val, slot_type))

    if not matches:
        return content, [], []

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: List[Tuple[int, int, str, str]] = []
    last_end = 0
    for start, end, val, slot_type in matches:
        if start >= last_end:
            selected.append((start, end, val, slot_type))
            last_end = end

    if not selected:
        return content, [], []

    new_turns: List[Dict[str, str]] = []
    meta_list: List[Dict[str, Any]] = []

    last_end = 0
    for start, end, val, slot_type in selected:
        before = content[last_end:start].strip()
        if before:
            new_turns.append({"role": "user", "content": before})

        dict_turns, dict_meta = _build_dictation_turns(val, slot_type, rng, lang, roles, perror)
        new_turns.extend(dict_turns)
        meta_list.extend(dict_meta)
        last_end = end

    after = content[last_end:].strip()
    if after:
        new_turns.append({"role": "user", "content": after})

    return content, new_turns, meta_list


def _process_dialogue(
    history: List[Dict[str, Any]],
    rng: random.Random,
    lang: str,
    roles: str,
    min_digits: int,
    min_code_len: int,
    perror: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Process one dialogue. Returns (new_history, modified)."""
    new_history: List[Dict[str, Any]] = []
    modified = False

    for turn in history:
        content = turn.get("content", "")
        role = turn.get("role", "")

        if turn.get("meta", {}).get("cross_turn_slot"):
            new_history.append(turn)
            continue

        if role != "user":
            new_history.append(turn)
            continue

        new_content, new_turns, meta_list = _inject_slots_in_turn(
            content, rng, lang, roles, min_digits, min_code_len, perror
        )

        if not new_turns:
            new_history.append(turn)
            continue

        modified = True
        orig_turn = dict(turn)
        orig_turn["meta"] = dict(orig_turn.get("meta", {}))
        orig_turn["meta"]["cross_turn_slot"] = True
        new_history.append(orig_turn)

        for dt in new_turns:
            dt["meta"] = {"cross_turn_slot": True}
            new_history.append(dt)

    return new_history, modified


def process_file(
    src: Path,
    dst: Path,
    rng: random.Random,
    lang: str,
    roles: str,
    min_digits: int,
    min_code_len: int,
    perror: float,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Process one JSON file. Returns stats."""
    with open(src) as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        return {"file": str(src), "modified": False, "slots_found": 0}

    new_history, modified = _process_dialogue(history, rng, lang, roles, min_digits, min_code_len, perror)

    if dry_run:
        return {"file": str(src), "modified": modified, "slots_found": len([t for t in new_history if t.get("meta", {}).get("cross_turn_slot")])}

    if modified:
        data["history"] = new_history
        data.setdefault("meta", {})["cross_turn_slots_applied"] = True

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": str(src),
        "modified": modified,
        "slots_found": len([t for t in new_history if t.get("meta", {}).get("cross_turn_slot")]),
    }


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    p = argparse.ArgumentParser(description="Cross-turn slot dictation injection.")
    p.add_argument("--input_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--perror", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_language", default="vi", choices=["vi", "en"])
    p.add_argument("--roles", default="both", choices=["user", "assistant", "both"])
    p.add_argument("--min-digits", type=int, default=6)
    p.add_argument("--min-code-len", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    input_dir = Path(args.input_root) / f"text_dialogue_{args.dataset}" / args.split
    output_dir = Path(args.output_root) / f"text_dialogue_{args.dataset}" / args.split

    if not input_dir.exists():
        logger.error("Input dir not found: %s", input_dir)
        sys.exit(1)

    files = sorted(input_dir.glob("*.json"))
    logger.info("Found %d files in %s", len(files), input_dir)

    rng = random.Random(args.seed)
    total_slots = 0
    total_modified = 0

    for f in files:
        rel = f.relative_to(input_dir)
        dst = output_dir / rel

        if dst.exists() and not args.dry_run:
            logger.info("Skip (exists): %s", dst)
            continue

        stats = process_file(f, dst, rng, args.target_language, args.roles, args.min_digits, args.min_code_len, args.perror, args.dry_run)
        total_slots += stats["slots_found"]
        if stats["modified"]:
            total_modified += 1

        if args.dry_run:
            logger.info("[dry-run] %s: %d slots, modified=%s", f.name, stats["slots_found"], stats["modified"])

    logger.info("Done. Modified %d/%d files, %d total slots.", total_modified, len(files), total_slots)


if __name__ == "__main__":
    main()
