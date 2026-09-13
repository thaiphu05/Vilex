"""Disfluency injection following Switchboard Corpus guidelines (Meteer et al., 1995)
and Shriberg (1996) exponential length-dependent model.

Active types: FP, DM, EDIT, REP — all rule-based, 0 API calls.
The model is applied per *sentence* (Shriberg's "utterance"), not per whole
turn: P(disfluent | L) = 1 - 0.9453**L with L = words in that sentence, so
longer sentences are more likely to carry a disfluency.

COR (replace slot value) and RST (rephrase continuation) are temporarily
disabled and kept commented below; they require LLM templates (TODO).

Rate is dampened per role: P = min(1, (1 - 0.9453**L) * scale_role), default
``user=0.4`` / ``assistant=0.25``. A unit that already carries a hesitation
(``[PAUSE]`` from "...") is skipped entirely. Overridable per dialogue via
``meta["disfluency_scale"] = {"user": .., "assistant": ..}``.

Stage 1's LLM writes "..." for hesitation; those are rewritten to ``[PAUSE]``
so Stage 5 can render them as a short silence (the TTS cannot read "...").
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

# The Stage 1 LLM emits "..." for hesitation; the TTS cannot read it, so we
# rewrite it to this token which Stage 5 maps to a short silence.
PAUSE_TOKEN = "[PAUSE]"
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")

# Per-role damping on the Shriberg probability. A unit that already carries a
# hesitation ([PAUSE] from "...") is skipped entirely, so a pause and a
# disfluency never pile up.
DEFAULT_SCALES = {
    "user": 0.4,
    "assistant": 0.25,
}


# ---------------------------------------------------------------------------
# Inventories
# ---------------------------------------------------------------------------

# Inventories are clause-INITIAL only (insertion happens at the start of an
# utterance unit). Position-sensitive tokens were removed: the clause-final
# particles "nhé"/"nhỉ" and the demonstratives "này"/"đó".
FP_INVENTORY = {
    "vi": ["ừm", "à", "ờ", "ừ", "hừm", "ơ", "ưm"],
    "en": ["um", "uh", "er", "hmm", "ah"],
}

DM_INVENTORY = {
    "vi": ["à mà", "thì", "mà"],
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
    return 1.0 - _B**length_words


# ---------------------------------------------------------------------------
# Slot value detection (for placement)
# ---------------------------------------------------------------------------

_RE_SLOT = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"  # email
    r"|\+?\d[\d\s.-]{5,}\d"  # phone/numeric
    r"|[A-Z0-9]{5,}"  # alnum code
)


def _find_slot_positions(text: str) -> List[int]:
    """Find word indices that are slot values or within 2 words of them."""
    words = text.split()
    positions = set()
    for m in _RE_SLOT.finditer(text):
        prefix = text[: m.start()]
        word_idx = len(prefix.split())
        slot_words = m.group().split()
        for i in range(max(0, word_idx - 2), min(len(words), word_idx + len(slot_words) + 2)):
            positions.add(i)
    return sorted(positions)


# ---------------------------------------------------------------------------
# Rule-based disfluency insertion
# ---------------------------------------------------------------------------


def _with_comma(tok: str) -> str:
    """Append a comma to a disfluency token unless it already ends with one."""
    return tok if tok.endswith(",") else tok + ","


def _insert_fp(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    fp = rng.choice(FP_INVENTORY[lang])
    return words[:idx] + [_with_comma(fp)] + words[idx:]


def _insert_dm(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    dm = rng.choice(DM_INVENTORY[lang])
    return words[:idx] + [_with_comma(dm)] + words[idx:]


def _insert_edit(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    edit = rng.choice(EDIT_INVENTORY[lang])
    return words[:idx] + [_with_comma(edit)] + words[idx:]


_CLAUSE_PUNCT = (",", ".", "?", "!", ";")


def _safe_positions(words: List[str]) -> List[int]:
    """Return clause-boundary word indices safe for disfluency insertion.

    Allowed positions are the start of the sentence and immediately after a
    clause-ending punctuation mark. This prevents splitting multi-syllable
    Vietnamese compounds (e.g. 'chiến lược') or breaking phrases mid-clause.
    The last word is intentionally NOT a candidate: inserting before it would
    split the final phrase (e.g. 'công việc ừ nào?').
    """
    safe = {0}
    for i, w in enumerate(words):
        if w.endswith(_CLAUSE_PUNCT) and i + 1 < len(words):
            safe.add(i + 1)
    return sorted(safe)


def _next_punct_index(words: List[str], start: int) -> int:
    """First word index >= start that ends with clause punctuation, else len(words)."""
    for i in range(start, len(words)):
        if words[i].endswith(_CLAUSE_PUNCT):
            return i
    return len(words)


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank-line paragraph breaks, preserving the paragraph units."""
    parts = re.split(r"\n\s*\n", text)
    out = [p for p in parts if p.strip()]
    return out or ([text] if text.strip() else [])


def _split_sentences(text: str) -> List[str]:
    """Split a paragraph into utterance units on terminal punctuation or comma."""
    parts = re.split(r"(?<=[.?!,])\s+", text.strip())
    out = [p for p in parts if p.strip()]
    return out or ([text.strip()] if text.strip() else [])


def _insert_rep(words: List[str], idx: int, lang: str, rng: random.Random) -> List[str]:
    if idx >= len(words):
        return words
    span = rng.choice([1, 1, 2])
    # Keep the repeated span inside the current clause.
    max_end = _next_punct_index(words, idx)
    span = min(span, max_end - idx)
    span = max(1, span)
    span = min(span, len(words) - idx)
    repeat = words[idx : idx + span]
    # Repeat the target span (Switchboard REP); comma after the retrace.
    repeat = repeat[:-1] + [_with_comma(repeat[-1])]
    return words[:idx] + repeat + words[idx:]


# ---------------------------------------------------------------------------
# Core injection
# ---------------------------------------------------------------------------


def _inject_disfluency(
    text: str,
    rng: random.Random,
    lang: str,
    scale: float = 1.0,
    allow_edit_start: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Inject at most one disfluency into a single utterance (sentence).

    The length-dependent probability is applied here, so ``text`` must be one
    sentence/utterance — see ``_inject_disfluency_turn``. ``scale`` (>0) dampens
    the rate per role; the final probability is clipped to [0, 1].

    ``allow_edit_start`` permits an EDIT term at position 0 (only valid when the
    unit follows an earlier clause in the same turn, so there is something to
    edit; the very first unit of a turn keeps it False).
    """
    words = text.split()
    if len(words) < 2:
        return text, []

    p = min(1.0, p_disfluent(len(words)) * scale)
    if rng.random() >= p:
        return text, []

    dtype = rng.choice(["fp", "dm", "edit", "rep"])
    safe_positions = _safe_positions(words)

    # EDIT terms are inserted before the word being edited, which implies a
    # preceding clause. They are allowed at position 0 only when a prior clause
    # exists in this turn (see ``allow_edit_start``).
    if dtype == "edit":
        edit_positions = (
            list(safe_positions) if allow_edit_start else [pos for pos in safe_positions if pos > 0]
        )
        if not edit_positions:
            dtype = rng.choice(["fp", "dm", "rep"])
            idx = rng.choice(safe_positions)
        else:
            idx = rng.choice(edit_positions)
    else:
        idx = rng.choice(safe_positions)

    # --- COR / RST (disabled; implement via LLM later) -----------------------
    # COR: replace the original slot value with an alternative.
    # RST: abandon the current utterance and begin a rephrased continuation.
    # Both need LLM templates; slot detection is kept for that future path:
    # slot_positions = _find_slot_positions(text)
    # if dtype == "cor":
    #     idx = rng.choice(slot_positions) if slot_positions else rng.choice(safe_positions)

    if dtype == "fp":
        words = _insert_fp(words, idx, lang, rng)
    elif dtype == "dm":
        words = _insert_dm(words, idx, lang, rng)
    elif dtype == "edit":
        words = _insert_edit(words, idx, lang, rng)
    elif dtype == "rep":
        words = _insert_rep(words, idx, lang, rng)
    # elif dtype == "cor":  # TODO: LLM slot-value replacement
    #     words = _insert_edit(words, idx, lang, rng)
    # elif dtype == "rst":  # TODO: LLM rephrased continuation
    #     words = _insert_fp(words, idx, lang, rng)

    new_text = " ".join(words)
    return new_text, [
        {"type": dtype, "position": idx, "scale": scale, "original": text, "result": new_text}
    ]


def _inject_disfluency_turn(
    text: str,
    rng: random.Random,
    lang: str,
    scale: float = 1.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply the per-unit model across a whole turn.

    Splits the turn into paragraphs (blank-line breaks preserved as ``\\n\\n``)
    and then into utterance units (on ``. ? ! ,``); each unit gets its own
    Bernoulli trial. A unit that already carries a ``[PAUSE]`` (from "...") is
    skipped outright. Returns the new text plus a flat list of injection meta
    records.
    """
    new_paragraphs: List[str] = []
    meta_all: List[Dict[str, Any]] = []
    unit_index = 0

    for para in _split_paragraphs(text):
        new_sentences: List[str] = []
        for sent in _split_sentences(para):
            if PAUSE_TOKEN in sent:
                # Already a hesitation here: do not add a disfluency on top.
                new_sentences.append(sent)
                unit_index += 1
                continue
            new_sent, meta_list = _inject_disfluency(
                sent, rng, lang, scale=scale, allow_edit_start=(unit_index > 0)
            )
            for m in meta_list:
                m["sentence_index"] = unit_index
            meta_all.extend(meta_list)
            new_sentences.append(new_sent)
            unit_index += 1
        if new_sentences:
            new_paragraphs.append(" ".join(new_sentences))

    new_text = "\n\n".join(new_paragraphs)
    return new_text, meta_all


def _normalize_ellipsis(text: str) -> Tuple[str, int]:
    """Replace "..." / "…" with the ``[PAUSE]`` token; return (text, count).

    The token is space-padded so Stage 5 can split on it cleanly (the source
    writes ``"là..."`` with no separating space). Only spaces/tabs are collapsed
    so paragraph breaks (``\\n\\n``) survive.
    """
    count = len(_ELLIPSIS_RE.findall(text))
    if not count:
        return text, 0
    text = _ELLIPSIS_RE.sub(" " + PAUSE_TOKEN + " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text, count


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve_scale(
    role: str,
    dialogue_meta: Optional[Dict[str, Any]],
    scales: Dict[str, float],
) -> float:
    """Role scale: dialogue meta override (dict) > CLI default. Clipped [0,1]."""
    key = "assistant" if role == "assistant" else "user"
    override = dialogue_meta.get("disfluency_scale") if isinstance(dialogue_meta, dict) else None
    if isinstance(override, dict) and key in override:
        return _clip01(override[key])
    return _clip01(scales.get(key, 1.0))


def _process_dialogue(
    history: List[Dict[str, Any]],
    rng: random.Random,
    lang: str,
    scales: Optional[Dict[str, float]] = None,
    dialogue_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Process one dialogue. Returns (new_history, modified)."""
    scales = dict(DEFAULT_SCALES) if scales is None else scales
    dialogue_meta = dialogue_meta or {}
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

        content_paused, n_pauses = _normalize_ellipsis(content)
        scale = _resolve_scale(role, dialogue_meta, scales)
        new_content, meta_list = _inject_disfluency_turn(content_paused, rng, lang, scale=scale)

        if not meta_list and n_pauses == 0:
            new_history.append(turn)
            continue

        modified = True
        new_turn = dict(turn)
        new_turn["content"] = new_content
        new_turn["meta"] = dict(new_turn.get("meta", {}))
        new_turn["meta"]["disfluency_applied"] = True
        new_turn["meta"]["disfluency_injections"] = meta_list
        new_turn["meta"]["pauses"] = n_pauses
        new_history.append(new_turn)

    return new_history, modified


def _sum_meta(new_history: List[Dict[str, Any]], key: str) -> int:
    total = 0
    for t in new_history:
        if key == "injections":
            total += len(t.get("meta", {}).get("disfluency_injections", []))
        else:
            total += int(t.get("meta", {}).get(key, 0) or 0)
    return total


def process_file(
    src: Path,
    dst: Path,
    rng: random.Random,
    lang: str,
    dry_run: bool = False,
    scales: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Process one JSON file. Returns stats."""
    scales = dict(DEFAULT_SCALES) if scales is None else scales
    with open(src) as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        return {"file": str(src), "modified": False, "injections": 0, "pauses": 0}

    dialogue_meta = data.get("meta", {})
    new_history, modified = _process_dialogue(history, rng, lang, scales, dialogue_meta)
    injections = _sum_meta(new_history, "injections")
    pauses = _sum_meta(new_history, "pauses")

    if dry_run:
        return {
            "file": str(src),
            "modified": modified,
            "injections": injections,
            "pauses": pauses,
        }

    if modified:
        data["history"] = new_history
        meta = data.setdefault("meta", {})
        meta["disfluency_applied"] = True
        meta["pauses"] = pauses
        meta["disfluency_scale"] = {
            "user": _resolve_scale("user", dialogue_meta, scales),
            "assistant": _resolve_scale("assistant", dialogue_meta, scales),
        }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "file": str(src),
        "modified": modified,
        "injections": injections,
        "pauses": pauses,
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
    p.add_argument(
        "--scale_user",
        type=float,
        default=DEFAULT_SCALES["user"],
        help="Multiplier on the Shriberg probability for user units (clipped [0,1]).",
    )
    p.add_argument(
        "--scale_assistant",
        type=float,
        default=DEFAULT_SCALES["assistant"],
        help="Multiplier on the Shriberg probability for assistant units (clipped [0,1]).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    scales = {
        "user": _clip01(args.scale_user),
        "assistant": _clip01(args.scale_assistant),
    }

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

        stats = process_file(
            f,
            dst,
            rng,
            args.target_language,
            args.dry_run,
            scales=scales,
        )
        total_injections += stats["injections"]
        if stats["modified"]:
            total_modified += 1

        if args.dry_run:
            logger.info(
                "[dry-run] %s: %d injections, %d pauses, modified=%s",
                f.name,
                stats["injections"],
                stats.get("pauses", 0),
                stats["modified"],
            )

    logger.info(
        "Done. Modified %d/%d files, %d total injections.",
        total_modified,
        len(files),
        total_injections,
    )


if __name__ == "__main__":
    main()
