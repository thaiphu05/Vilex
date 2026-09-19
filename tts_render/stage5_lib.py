"""Pure-Python text helpers for Stage 5.1a (text prep), ported verbatim from
tts_render/convert_spoken.py so the prep stage imports them WITHOUT pulling in
torch / torchaudio / silero / OmniVoice (which convert_spoken needs at import).

VI/OmniVoice path only: no nltk, no NeMo. Every function here is a byte-for-byte
logic copy of its convert_spoken counterpart; if you change one, change both.
"""

import re
from typing import List, Optional, Tuple

# --- tokens (convert_spoken.py:215-217, 190-206) ------------------------------
TAKE_FLOOR_TOKEN = "[TAKE_FLOOR]"
PAUSE_TOKEN = "[PAUSE]"
DEFAULT_SPEAKERS = ["user", "assistant"]
DEFAULT_STYLE = "A speaker with normal speaking rate"

DEFAULT_BC_CANDIDATES = ["yeah", "uh-huh", "mm-hmm", "right", "okay"]
DEFAULT_BC_CANDIDATES_VI = [
    "ưm",
    "à",
    "ừ",
    "vâng",
    "phải",
    "ồ",
    "mm-hm",
    "uh huh",
    "ok",
    "ừ [confirmation-en]",
    "ồ [surprise-oh]",
    "[laughter]",
    "[sigh]",
]
# Backchannel tokens that read better with a rising (question) intonation.
BC_RISING_TOKENS = {"yeah", "ưm", "vâng", "à", "ừ"}

# OmniVoice paralinguistic tags (convert_spoken.py:99-117).
OMNI_PARALINGUIST_TAGS = [
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
]
RENDERABLE_TAGS = frozenset(OMNI_PARALINGUIST_TAGS)
_PARALINGUIST_RE = re.compile("|".join(re.escape(t) for t in OMNI_PARALINGUIST_TAGS))
_BRACKET_TAG_RE = re.compile(r"\[[A-Za-z0-9_\-]+\]")

# Vietnamese token regex: sequences of letters (incl. diacritics) / digits.
_VI_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ]+(?:['’\-][A-Za-zÀ-ỹ]+)*|\d+(?:\.\d+)?")


def strip_paralinguistic(text: str, keep: Optional[frozenset] = None) -> str:
    """Strip paralinguistic tags (convert_spoken._strip_paralinguistic)."""
    if keep is None:
        return _PARALINGUIST_RE.sub("", text)

    def _repl(m):
        tok = m.group(0)
        if tok in keep or tok == PAUSE_TOKEN:
            return tok
        return ""

    return _BRACKET_TAG_RE.sub(_repl, text)


def replace_dashes_outside_brackets(text: str) -> str:
    """'-' -> ' ' but keep hyphens inside [bracket] tags (convert_spoken:139)."""
    parts = re.split(r"(\[[A-Za-z0-9_\-]+\])", text)
    return "".join(
        p if (p.startswith("[") and p.endswith("]")) else p.replace("-", " ") for p in parts
    )


def split_sentences_vi(text: str) -> List[str]:
    """VI branch of split_sentences (convert_spoken:444-448): split on .?!"""
    chunks = re.split(r"(?<=[\.\?!])\s+", text)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks or [text.strip()]


def count_words_for_align_vi(text: str) -> int:
    """VI branch of count_words_for_align (convert_spoken:464-474)."""
    t = text.replace("[BACKCHANNEL]", "").replace(TAKE_FLOOR_TOKEN, "").replace(PAUSE_TOKEN, "")
    t = strip_paralinguistic(
        t.replace("[interrupted]", "").replace("[MASK1]", "").replace("[MASK2]", "")
    )
    t = t.strip()
    return len(_VI_WORD_RE.findall(t))


def cut_at_take_floor_plus_one_word(text: str) -> str:
    """convert_spoken:489-506 (keep 1 word, or 3 when left side < 5 words)."""
    if TAKE_FLOOR_TOKEN not in text:
        return text
    left, right = text.split(TAKE_FLOOR_TOKEN, 1)
    left = left.strip()
    right = right.strip()
    words = list(_VI_WORD_RE.finditer(right))
    if not words:
        return left
    left_word_count = len(_VI_WORD_RE.findall(left))
    keep = 3 if left_word_count < 5 else 1
    kept = " ".join(m.group(0) for m in words[:keep])
    if left:
        return f"{left} {kept}".strip()
    return kept.strip()


def clean_text(s: str) -> str:
    """convert_spoken:403."""
    return s.replace("[BACKCHANNEL]", "").replace(TAKE_FLOOR_TOKEN, "").strip()


# --- LLM-artifact sanitizer (convert_spoken.py:509-654) ------------------------
_INTENTIONAL_BRACKET_TOKENS = {
    "[BACKCHANNEL]",
    "[TAKE_FLOOR]",
    "[PAUSE]",
    "[MASK1]",
    "[MASK2]",
    "[interrupted]",
    "[sigh]",
}

_STAGE_DIRECTION_WORDS = {
    "sigh",
    "sighs",
    "sighing",
    "yawn",
    "yawns",
    "yawning",
    "laugh",
    "laughs",
    "laughing",
    "chuckle",
    "chuckles",
    "cough",
    "coughs",
    "clears throat",
    "clearing throat",
    "whistle",
    "whistles",
    "whistling",
    "wink",
    "winks",
    "wink wink",
    "smile",
    "smiles",
    "smiling",
    "grin",
    "grins",
    "pause",
    "pauses",
    "hands receipt",
    "three times",
}

_NUM_ROLE_RX = re.compile(r"^\s*\d+\)\s*(user|assistant|system)\s*:\s*", re.IGNORECASE)
_NUM_ROLE_ANY_RX = re.compile(r"\d+\)\s*(user|assistant|system)\s*:", re.IGNORECASE)
_BARE_ROLE_RX = re.compile(r"^\s*(user|assistant|system)\s*:\s*", re.IGNORECASE)
_MD_BOLD_RX = re.compile(r"\*\*[^*\n]{1,200}\*\*")
_AST_TURN_MARKER_RX = re.compile(
    r"\*[^*\n]*?(?:next\s+turn|user\s*:|assistant\s*:|user\s*'?s?\s+next|assistant\s*'?s?\s+next)[^*\n]*\*",
    re.IGNORECASE,
)
_STAR_PHRASE_RX = re.compile(r"\*([^*\n]{1,60})\*")
_BRACKET_RX = re.compile(r"\[([^\[\]\n]{1,60})\]")


def _is_template_placeholder(token: str) -> bool:
    """convert_spoken:566-586."""
    if token in _INTENTIONAL_BRACKET_TOKENS:
        return False
    inner = token[1:-1]
    if inner.upper() == "TAKING_FLOOR":
        return False
    low = inner.lower()
    if low in {"user turn", "user's turn", "assistant turn", "assistant's turn"}:
        return False
    if low in {"email protected", "email-protected"}:
        return True
    if re.match(
        r"^(?:your|some|city|state|country|old|new|insert|enter|placeholder)\b", inner, re.I
    ):
        return True
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", inner):
        return True
    return False


def should_drop_dialogue(d: dict) -> Tuple[bool, str]:
    """convert_spoken:589-622 (verbatim)."""
    for hi, msg in enumerate(d.get("history", [])):
        role = (msg.get("role") or "").lower()
        candidates = []
        outer = msg.get("content")
        if isinstance(outer, str):
            candidates.append(outer)
        for sub in msg.get("history", []):
            fc = sub.get("full_content") if isinstance(sub, dict) else None
            if isinstance(fc, str):
                candidates.append(fc)
        for s in candidates:
            if len(_NUM_ROLE_ANY_RX.findall(s)) >= 2:
                return True, f"msg{hi}: multi-turn concat"
            if re.search(r"</?think\b", s, re.IGNORECASE):
                return True, f"msg{hi}: think-block leak"
            if len(s) > 1500:
                return True, f"msg{hi}: very-long ({len(s)} chars)"
            if _MD_BOLD_RX.search(s):
                return True, f"msg{hi}: markdown bold leak"
            if _AST_TURN_MARKER_RX.search(s):
                return True, f"msg{hi}: asterisk turn marker"
            for m in _BRACKET_RX.finditer(s):
                tok = m.group(0)
                if _is_template_placeholder(tok):
                    return True, f"msg{hi}: template placeholder {tok}"
            m = _NUM_ROLE_RX.match(s) or _BARE_ROLE_RX.match(s)
            if m and role:
                embedded = m.group(1).lower()
                if embedded != role:
                    return True, f"msg{hi}: role-mismatch outer={role} inline={embedded}"
    return False, ""


def _strip_stage_directions(s: str) -> str:
    def _repl(m):
        inner = m.group(1).strip(" ,").lower()
        if inner in _STAGE_DIRECTION_WORDS:
            return ""
        return m.group(1)

    return _STAR_PHRASE_RX.sub(_repl, s)


def _peel_role_prefix(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _NUM_ROLE_RX.sub("", s)
        s = _BARE_ROLE_RX.sub("", s)
    return s


def sanitize_text(s: str) -> str:
    """convert_spoken:645-654 (verbatim)."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"\[TAKING_FLOOR\]", "[TAKE_FLOOR]", s, flags=re.IGNORECASE)
    s = re.sub(r"\.\.\.|…", PAUSE_TOKEN, s)
    s = re.sub(r"\[User Turn\]|\[User turn\]|\[Assistant Turn\]|\[Assistant turn\]", "", s)
    s = _peel_role_prefix(s)
    s = _strip_stage_directions(s)
    return re.sub(r"\s{2,}", " ", s).strip()


# --- schema conversion (convert_spoken.py:657-762) -----------------------------
def reconstruct_text_with_tokens(msg_data: dict) -> str:
    """convert_spoken:657-672 (verbatim)."""
    if "history" not in msg_data or not msg_data["history"]:
        return msg_data.get("content", "")
    full_content = msg_data["history"][0].get("full_content", "")
    decisions = [item for item in msg_data["history"] if "word_index" in item]
    words = full_content.split()
    for d in sorted(decisions, key=lambda x: x["word_index"], reverse=True):
        token = d.get("inserted_token")
        if token:
            idx = min(d["word_index"] + 1, len(words))
            words.insert(idx, token)
    return " ".join(words)


def _split_backchannel_segments(text, specific_contents, role2spk, bc_spk, pick_bc):
    """convert_spoken:407-439, with `pick_bc(candidates) -> str` instead of
    module-global random.choice so 5.1a can seed it deterministically."""
    parts = re.split(r"\[BACKCHANNEL\]", text)
    segments = []
    bc_counter = 0
    for i, seg in enumerate(parts):
        seg = seg.strip(" ,")
        if seg:
            segments.append((seg, None, None))
        if i < len(parts) - 1:
            if (
                specific_contents
                and bc_counter < len(specific_contents)
                and specific_contents[bc_counter]
            ):
                bc_text = specific_contents[bc_counter]
                bc_counter += 1
            else:
                bc_text = pick_bc()
            bc_spk_override = (
                bc_spk if bc_spk is not None else (role2spk.get("assistant") if role2spk else None)
            )
            segments.append((bc_text, "backchannel", bc_spk_override))
    return segments


def convert_dialogue_to_utterances(original: dict, pick_bc, language: str = "vi"):
    """convert_spoken.convert_original_to_expected (675-762), parameterized on
    `pick_bc` (seeded) instead of random.choice. Returns (utterances_with_bc, ...).

    Each entry: {"speaker": "user"|"assistant", "uttr_type": None|"backchannel"|"interrupt",
                 "text": seg_text}
    """
    history = original.get("history", [])
    speakers = DEFAULT_SPEAKERS[:]
    role2spk = {"user": speakers[0], "assistant": speakers[1]}
    utterances = []
    triggers = []

    for msg in history:
        role = msg.get("role")
        if role not in role2spk:
            continue
        msg_role_spk = role2spk[role]
        text = reconstruct_text_with_tokens(msg)
        text = sanitize_text(text)
        if not text.strip():
            continue

        bc_contents_list = []
        if "history" in msg:
            decisions = [h for h in msg["history"] if "word_index" in h]
            decisions.sort(key=lambda x: x["word_index"])
            for d in decisions:
                if d.get("decision") == "backchannel" or d.get("inserted_token") == "[BACKCHANNEL]":
                    bc_contents_list.append(d.get("content") or pick_bc())

        take_floor = TAKE_FLOOR_TOKEN in text
        if take_floor:
            text = cut_at_take_floor_plus_one_word(text)

        segments = _split_backchannel_segments(
            text,
            specific_contents=bc_contents_list,
            role2spk=role2spk,
            bc_spk=msg_role_spk,
            pick_bc=pick_bc,
        )
        for seg_idx, (seg_text_raw, bc_type, spk_override) in enumerate(segments):
            seg_text = clean_text(seg_text_raw)
            if not seg_text and bc_type is None:
                continue
            uttr_type = "backchannel" if bc_type == "backchannel" else None
            current_spk = spk_override if spk_override else msg_role_spk
            item = {"speaker": current_spk, "uttr_type": uttr_type, "text": seg_text}
            if take_floor and uttr_type is None:
                is_last_non_bc = all(s[1] is not None for s in segments[seg_idx + 1 :])
                if is_last_non_bc and seg_text:
                    other_spk = speakers[1] if msg_role_spk == speakers[0] else speakers[0]
                    triggers.append((other_spk, len(utterances) + 1))
            utterances.append(item)

    for target_spk, start_idx in triggers:
        if start_idx < len(utterances):
            if utterances[start_idx]["speaker"] == target_spk:
                utterances[start_idx]["uttr_type"] = "interrupt"
    return utterances
