"""Tests for disfluency module. 0 API calls for rule-based tests."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.disfluency import (
    p_disfluent,
    _find_slot_positions,
    _inject_disfluency,
    _inject_disfluency_turn,
    _insert_fp,
    _insert_dm,
    _insert_edit,
    _insert_rep,
    _normalize_ellipsis,
    _process_dialogue,
    _resolve_scale,
    _safe_positions,
    _split_paragraphs,
    _split_sentences,
    _with_comma,
    PAUSE_TOKEN,
    DEFAULT_SCALES,
    FP_INVENTORY,
    DM_INVENTORY,
    EDIT_INVENTORY,
)

ZERO_SCALES = {"user": 0.0, "assistant": 0.0}


class TestShribergModel:
    def test_p_disfluent_short(self):
        p = p_disfluent(3)
        assert p < 0.2

    def test_p_disfluent_long(self):
        p = p_disfluent(50)
        assert p > 0.9

    def test_p_disfluent_monotonic(self):
        assert p_disfluent(5) < p_disfluent(10) < p_disfluent(20)


class TestSlotPositions:
    def test_phone_slot(self):
        text = "SĐT tôi là 0901234567 nhé."
        positions = _find_slot_positions(text)
        assert len(positions) > 0

    def test_email_slot(self):
        text = "Email là john@gmail.com nhé."
        positions = _find_slot_positions(text)
        assert len(positions) > 0

    def test_no_slot(self):
        text = "Xin chào bạn."
        positions = _find_slot_positions(text)
        assert len(positions) == 0


class TestInsertion:
    def test_fp(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_fp(words, 1, "vi", rng)
        assert len(result) == len(words) + 1
        assert result[1].rstrip(",") in FP_INVENTORY["vi"]
        assert result[1].endswith(",")

    def test_dm(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_dm(words, 1, "vi", rng)
        assert len(result) == len(words) + 1
        assert result[1].rstrip(",") in DM_INVENTORY["vi"]
        assert result[1].endswith(",")

    def test_edit(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_edit(words, 1, "vi", rng)
        assert len(result) == len(words) + 1
        assert result[1].rstrip(",") in EDIT_INVENTORY["vi"]
        assert result[1].endswith(",")

    def test_rep(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_rep(words, 1, "vi", rng)
        assert len(result) > len(words)
        assert any(t.endswith(",") for t in result)


class TestInjection:
    def test_no_injection_short(self):
        rng = random.Random(42)
        text = "OK."
        new_text, meta = _inject_disfluency(text, rng, "vi")
        assert new_text == text
        assert meta == []

    def test_deterministic(self):
        text = "SĐT tôi là 0901234567."
        r1 = _inject_disfluency(text, random.Random(42), "vi")
        r2 = _inject_disfluency(text, random.Random(42), "vi")
        assert r1[0] == r2[0]

    def test_long_utterance_more_disfluent(self):
        rng = random.Random(42)
        short_count = 0
        long_count = 0
        for _ in range(100):
            _, m1 = _inject_disfluency("a b c", rng, "vi")
            _, m2 = _inject_disfluency(
                "a b c d e f g h i j k l m n o p q r s t u v w x y z", rng, "vi"
            )
            short_count += len(m1)
            long_count += len(m2)
        assert long_count > short_count


class TestSafePositions:
    def test_safe_positions_start_and_after_punct(self):
        words = "Tôi sẽ dùng chiến lược này, để bán hàng.".split()
        assert _safe_positions(words) == [0, 6]

    def test_safe_positions_no_punct(self):
        words = "Tôi sẽ dùng chiến lược này để bán hàng".split()
        assert _safe_positions(words) == [0]

    def test_last_word_is_not_a_candidate(self):
        words = "Tôi sẽ dùng chiến lược này để bán hàng".split()
        assert len(words) - 1 not in _safe_positions(words)


class TestComma:
    def test_with_comma(self):
        assert _with_comma("ừm") == "ừm,"
        assert _with_comma("ừm,") == "ừm,"

    def test_inserted_tokens_have_comma(self):
        rng = random.Random(0)
        words = "Xin chào bạn".split()
        for fn in (_insert_fp, _insert_dm, _insert_edit):
            result = fn(words, 1, "vi", rng)
            assert result[1].endswith(","), (fn.__name__, result)


class TestPerSentence:
    def test_multiple_sentences_can_get_multiple_injections(self):
        text = (
            "Tôi sẽ dùng chiến lược này để bán hàng. "
            "Sau đó tôi kiểm tra lại kết quả và chỉnh sửa. "
            "Cuối cùng tôi gửi cho khách hàng."
        )
        multi = sum(
            1
            for seed in range(300)
            if len(_inject_disfluency_turn(text, random.Random(seed), "vi")[1]) >= 2
        )
        assert multi > 0, "expected some turns with >=2 injections"

    def test_meta_has_sentence_index(self):
        text = "Tôi sẽ dùng chiến lược này để bán hàng. Sau đó tôi kiểm tra lại."
        for seed in range(50):
            _, meta = _inject_disfluency_turn(text, random.Random(seed), "vi")
            for m in meta:
                assert "sentence_index" in m


class TestSplitting:
    def test_split_paragraphs(self):
        assert _split_paragraphs("A. B.\n\nC. D.") == ["A. B.", "C. D."]

    def test_split_sentences(self):
        assert _split_sentences("A. B? C!") == ["A.", "B?", "C!"]

    def test_split_sentences_on_comma(self):
        assert _split_sentences("A, B. C") == ["A,", "B.", "C"]

    def test_comma_makes_more_units(self):
        assert len(_split_sentences("Tôi nghĩ là, tôi sẽ dùng, chiến lược này.")) == 3


class TestNewlinePreserved:
    def test_blank_line_preserved(self):
        text = (
            "Tuyệt vời! Chúng ta bắt đầu nhé.\n\n" "Bạn có thể chia sẻ một chút về công việc không?"
        )
        new_text, _ = _inject_disfluency_turn(text, random.Random(42), "vi")
        assert "\n\n" in new_text


class TestTypeAndPlacement:
    def test_dtype_never_cor_or_rst(self):
        allowed = {"fp", "dm", "edit", "rep"}
        text = "Tôi sẽ dùng chiến lược này, để bán hàng cho khách."
        for seed in range(300):
            _, meta = _inject_disfluency(text, random.Random(seed), "vi")
            for m in meta:
                assert m["type"] in allowed

    def test_edit_never_at_sentence_start(self):
        text = "Tôi sẽ dùng chiến lược này, để bán hàng cho khách."
        for seed in range(300):
            _, meta = _inject_disfluency(text, random.Random(seed), "vi")
            for m in meta:
                if m["type"] == "edit":
                    assert m["position"] > 0


class TestCompoundSafeInjection:
    def test_no_split_vietnamese_compound(self):
        """Disfluency insertion must not insert tokens between 'chiến' and 'lược'."""
        for seed in range(200):
            text = "Tôi sẽ dùng chiến lược này để bán hàng"
            new_text, meta = _inject_disfluency(text, random.Random(seed), "vi")
            if meta:
                toks = new_text.split()
                for i in range(len(toks) - 1):
                    assert not (
                        toks[i] == "chiến" and toks[i + 1] != "lược"
                    ), f"compound split at seed {seed}: {new_text}"


class TestScale:
    def test_scale_zero_never_injects(self):
        text = "Tôi sẽ dùng chiến lược này để bán hàng cho khách"
        for seed in range(50):
            new_text, meta = _inject_disfluency(text, random.Random(seed), "vi", scale=0.0)
            assert meta == []
            assert new_text == text

    def test_scale_reduces_injections(self):
        text = "Tôi sẽ dùng chiến lược này để bán hàng cho khách hàng mới"
        full = sum(
            len(_inject_disfluency(text, random.Random(s), "vi", scale=1.0)[1]) for s in range(200)
        )
        half = sum(
            len(_inject_disfluency(text, random.Random(s), "vi", scale=0.3)[1]) for s in range(200)
        )
        assert half < full

    def test_meta_records_scale(self):
        text = "Tôi sẽ dùng chiến lược này để bán hàng cho khách"
        for seed in range(50):
            _, meta = _inject_disfluency(text, random.Random(seed), "vi", scale=0.5)
            for m in meta:
                assert m["scale"] == 0.5


class TestResolveScale:
    def test_default_by_role(self):
        assert _resolve_scale("user", {}, DEFAULT_SCALES) == 0.4
        assert _resolve_scale("assistant", {}, DEFAULT_SCALES) == 0.25

    def test_dict_override(self):
        meta = {"disfluency_scale": {"user": 0.2, "assistant": 0.9}}
        assert _resolve_scale("user", meta, DEFAULT_SCALES) == 0.2
        assert _resolve_scale("assistant", meta, DEFAULT_SCALES) == 0.9

    def test_clip_to_unit_interval(self):
        assert _resolve_scale("user", {"disfluency_scale": {"user": 5}}, DEFAULT_SCALES) == 1.0
        assert _resolve_scale("user", {"disfluency_scale": {"user": -1}}, DEFAULT_SCALES) == 0.0


class TestEditStart:
    def test_edit_disallowed_at_start_by_default(self):
        text = "Tôi sẽ dùng chiến lược này để bán hàng cho khách"
        for seed in range(300):
            _, meta = _inject_disfluency(text, random.Random(seed), "vi")
            for m in meta:
                if m["type"] == "edit":
                    assert m["position"] > 0

    def test_edit_allowed_at_start_when_flag(self):
        text = "Tôi sẽ dùng chiến lược này để bán hàng cho khách"
        seen = any(
            m["type"] == "edit" and m["position"] == 0
            for seed in range(500)
            for m in _inject_disfluency(text, random.Random(seed), "vi", allow_edit_start=True)[1]
        )
        assert seen


class TestPauseSkipsInjection:
    def test_unit_with_pause_gets_no_injection(self):
        # A single unit containing "...": after normalization it carries
        # [PAUSE], so no disfluency may be added on top.
        history = [
            {
                "role": "user",
                "content": "Tôi nghĩ là... vậy đó và điều này rất quan trọng với tôi.",
            }
        ]
        for seed in range(300):
            new_hist, _ = _process_dialogue(history, random.Random(seed), "vi")
            assert new_hist[0].get("meta", {}).get("disfluency_injections", []) == []
            assert "[PAUSE]" in new_hist[0]["content"]

    def test_other_units_still_injected(self):
        # Unit 0 has a pause (skipped); later units can still be injected.
        history = [
            {
                "role": "user",
                "content": "Tôi nghĩ là... vậy đó. Sau đó tôi kiểm tra lại và chỉnh sửa mọi thứ.",
            }
        ]
        seen = any(
            new_hist[0].get("meta", {}).get("disfluency_injections")
            for new_hist, _ in (
                _process_dialogue(history, random.Random(seed), "vi") for seed in range(300)
            )
        )
        assert seen


class TestEllipsisPause:
    def test_pause_token_value(self):
        assert PAUSE_TOKEN == "[PAUSE]"

    def test_normalize_ellipsis(self):
        out, n = _normalize_ellipsis("tôi nghĩ là... ừm")
        assert out == "tôi nghĩ là [PAUSE] ừm"
        assert n == 1

    def test_ellipsis_to_pause_both_roles(self):
        history = [
            {"role": "user", "content": "Tôi nghĩ là... vậy đó."},
            {"role": "assistant", "content": "Vâng... tôi hiểu."},
        ]
        new_hist, modified = _process_dialogue(history, random.Random(42), "vi", ZERO_SCALES)
        assert modified
        for turn in new_hist:
            assert "[PAUSE]" in turn["content"]
            assert "..." not in turn["content"]
            assert turn["meta"]["pauses"] == 1

    def test_no_ellipsis_no_modification(self):
        history = [{"role": "user", "content": "Dạ vâng ạ."}]
        new_hist, modified = _process_dialogue(history, random.Random(42), "vi", ZERO_SCALES)
        assert not modified
        assert new_hist == history


class TestInventory:
    def test_fp_not_empty(self):
        assert len(FP_INVENTORY["vi"]) > 0
        assert len(FP_INVENTORY["en"]) > 0

    def test_dm_not_empty(self):
        assert len(DM_INVENTORY["vi"]) > 0
        assert len(DM_INVENTORY["en"]) > 0

    def test_dm_is_clause_initial_only(self):
        # Position-sensitive tokens must not be usable at clause start.
        removed = {"nhé", "nhỉ", "này", "đó"}
        assert removed.isdisjoint(set(DM_INVENTORY["vi"]))

    def test_edit_not_empty(self):
        assert len(EDIT_INVENTORY["vi"]) > 0
        assert len(EDIT_INVENTORY["en"]) > 0
