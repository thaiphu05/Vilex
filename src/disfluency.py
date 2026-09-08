"""Disfluency injection following Switchboard Corpus guidelines (Meteer et al., 1995)
and Shriberg (1996) exponential length-dependent model.

6 types: [FP], [DM], [EDIT], [REP], [COR], [RST]
- FP/DM/EDIT/REP: rule-based, 0 API calls
- COR/RST: LLM-based (Gemini), require prompt templates (user-provided)
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root


# ---------------------------------------------------------------------------
# Inventories
# ---------------------------------------------------------------------------

FP_INVENTORY = {
    "vi": ["ừm", "à", "ờ", "ừ", "hừm", "ơ", "ưm"],
    "en": ["um", "uh", "er", "hmm", "ah"],
}

DM_INVENTORY = {
    "vi": ["này", "à mà", "thì", "mà", "đó", "nhé", "nhỉ"],
    "en": ["like", "you know", "well", "so", "I mean", "actually"],
}

EDIT_INVENTORY = {
    "vi": ["ý tôi là", "tức là", "hay là", "không phải", "à mà"],
    "en": ["I mean", "that is", "or rather", "no wait", "wait"],
}

# ---------------------------------------------------------------------------
# Shriberg (1996) exponential length-dependent model
# ---------------------------------------------------------------------------

_B = 0.9453  # word-level fluency rate (AMEX corpus)


def p_disfluent(length_words: int) -> float:
    """Probability that an utterance of length L words is disfluent."""
    return 1.0 - _B ** length_words


# ---------------------------------------------------------------------------
# Slot value detection (for placement)
# ---------------------------------------------------------------------------

_RE_SLOT = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"  # email
    r"|\+?\d[\d\s.-]{5,}\d"      # phone/numeric
    r"|[A-Z0-9]{5,}"             # alnum code
)


def _find_slot_positions(text: str) -> List[int]:
    """Find word indices that are slot values or within 2 words of them."""
    words = text.split()
    positions = set()
    for m in _RE_SLOT.finditer(text):
        prefix = text[:m.start()]
        word_idx = len(prefix.split())
        slot_words = m.group().split()
        for i in range(max(0, word_idx - 2), min(len(words), word_idx + len(slot_words) + 2)):
            positions.add(i)
    return sorted(positions)


# ---------------------------------------------------------------------------
# Rule-based disfluency insertion
# ---------------------------------------------------------------------------

def _insert_fp(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    fp = rng.choice(FP_INVENTORY[lang])
    return words[:idx] + [fp] + words[idx:]


def _insert_dm(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    dm = rng.choice(DM_INVENTORY[lang])
    return words[:idx] + [dm] + words[idx:]


def _insert_edit(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    edit = rng.choice(EDIT_INVENTORY[lang])
    return words[:idx] + [edit] + words[idx:]


def _insert_rep(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    if idx >= len(words):
        return words
    span = rng.choice([1, 1, 2])
    span = min(span, len(words) - idx)
    repeat = words[idx:idx + span]
    return words[:idx] + repeat + words[idx:]


# ---------------------------------------------------------------------------
# Core injection
# ---------------------------------------------------------------------------

def _inject_disfluency(
    text: str,
    rng: random.Random,
    lang: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Inject disfluencies into a single utterance."""
    words = text.split()
    if len(words) < 2:
        return text, []

    p = p_disfluent(len(words))
    if rng.random() >= p:
        return text, []

    dtype = rng.choice(["fp", "dm", "edit", "rep", "cor", "rst"])

    slot_positions = _find_slot_positions(text)

    if dtype == "cor":
        if slot_positions:
            idx = rng.choice(slot_positions)
        else:
            dtype = rng.choice(["fp", "dm", "edit", "rep"])
            idx = rng.randint(0, len(words) - 1)
    else:
        if slot_positions and rng.random() < 0.5:
            idx = rng.choice(slot_positions)
        else:
            idx = rng.randint(0, len(words) - 1)

    if dtype == "fp":
        words = _insert_fp(words, idx, lang, rng)
    elif dtype == "dm":
        words = _insert_dm(words, idx, lang, rng)
    elif dtype == "edit":
        words = _insert_edit(words, idx, lang, rng)
    elif dtype == "rep":
        words = _insert_rep(words, idx, lang, rng)
    elif dtype == "cor":
        # Placeholder: use EDIT until LLM template provided
        words = _insert_edit(words, idx, lang, rng)
    elif dtype == "rst":
        # Placeholder: use FP until LLM template provided
        words = _insert_fp(words, idx, lang, rng)

    new_text = " ".join(words)
    return new_text, [{"type": dtype, "position": idx, "original": text, "result": new_text}]


def _process_dialogue(
    history: List[Dict[str, Any]],
    rng: random.Random,
    lang: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Process one dialogue. Returns (new_history, modified)."""
    new_history: List[Dict[str, Any]] = []
    modified = False

    for turn in history:
        content = turn.get("content", "")
        role = turn.get("role", "")

        if turn.get("meta", {}).get("disfluency_applied"):
            new_history.append(turn)
            continue

        if role not in ("user", "assistant"):
            new_history.append(turn)
            continue

        new_content, meta_list = _inject_disfluency(content, rng, lang)

        if not meta_list:
            new_history.append(turn)
            continue

        modified = True
        new_turn = dict(turn)
        new_turn["content"] = new_content
        new_turn["meta"] = dict(new_turn.get("meta", {}))
        new_turn["meta"]["disfluency_applied"] = True
        new_turn["meta"]["disfluency_injections"] = meta_list
        new_history.append(new_turn)

    return new_history, modified


def process_file(
    src: Path,
    dst: Path,
    rng: random.Random,
    lang: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Process one JSON file. Returns stats."""
    with open(src) as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        return {"file": str(src), "modified": False, "injections": 0}

    new_history, modified = _process_dialogue(history, rng, lang)

    if dry_run:
        return {"file": str(src), "modified": modified, "injections": sum(len(t.get("meta", {}).get("disfluency_injections", [])) for t in new_history)}

    if modified:
        data["history"] = new_history
        data.setdefault("meta", {})["disfluency_applied"] = True

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": str(src),
        "modified": modified,
        "injections": sum(len(t.get("meta", {}).get("disfluency_injections", [])) for t in new_history),
    }


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    p = argparse.ArgumentParser(description="Disfluency injection (Switchboard/Shriberg).")
    p.add_argument("--input_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_language", default="vi", choices=["vi", "en"])
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
    total_injections = 0
    total_modified = 0

    for f in files:
        rel = f.relative_to(input_dir)
        dst = output_dir / rel

        if dst.exists() and not args.dry_run:
            logger.info("Skip (exists): %s", dst)
            continue

        stats = process_file(f, dst, rng, args.target_language, args.dry_run)
        total_injections += stats["injections"]
        if stats["modified"]:
            total_modified += 1

        if args.dry_run:
            logger.info("[dry-run] %s: %d injections, modified=%s", f.name, stats["injections"], stats["modified"])

    logger.info("Done. Modified %d/%d files, %d total injections.", total_modified, len(files), total_injections)


if __name__ == "__main__":
    main()
