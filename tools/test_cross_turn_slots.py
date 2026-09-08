"""Tests for cross_turn_slots module. 0 API calls."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.cross_turn_slots import (
    _segment, _vocalize_segment, _corrupt_segment, _inject_slots_in_turn,
    _build_dictation_turns, _process_dialogue,
)


class TestSegmentation:
    def test_numeric_3_4_chunks(self):
        assert _segment("0901234567", "numeric", "vi") == ["0901", "234", "567"]
        assert _segment("123456", "numeric", "vi") == ["123", "456"]
        assert _segment("12345", "numeric", "vi") == ["123", "45"]
        assert _segment("1234", "numeric", "vi") == ["1234"]

    def test_email(self):
        segs = _segment("john.doe@gmail.com", "email", "vi")
        assert segs == ["john.doe", "a-còng", "gmail", "chấm", "com"]

    def test_alnum(self):
        assert _segment("AB12CD", "alnum", "vi") == ["AB", "12", "CD"]
        assert _segment("ABC123", "alnum", "vi") == ["ABC", "123"]
        assert _segment("12AB", "alnum", "vi") == ["12", "AB"]


class TestVocalization:
    def test_digits_vi(self):
        assert _vocalize_segment("0901", "numeric", "vi") == "không chín không một"

    def test_digits_en(self):
        assert _vocalize_segment("0901", "numeric", "en") == "zero nine zero one"

    def test_alnum_vi(self):
        assert _vocalize_segment("AB", "alnum", "vi") == "a bờ"


class TestCorruption:
    def test_corrupt_numeric(self):
        rng = random.Random(42)
        for _ in range(100):
            wrong = _corrupt_segment("0901", "numeric", rng)
            assert wrong != "0901"
            assert len(wrong) == 4

    def test_corrupt_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert _corrupt_segment("0901", "numeric", rng1) == _corrupt_segment("0901", "numeric", rng2)


class TestTemplates:
    def test_vi_templates(self):
        from src.cross_turn_slots import _TEMPLATES
        assert len(_TEMPLATES["vi"]["ack_short"]) > 0
        assert len(_TEMPLATES["vi"]["ack_final"]) > 0
        assert "Khoan" in _TEMPLATES["vi"]["self_correct"]
        assert "Ừ" in _TEMPLATES["vi"]["ack_correct"]

    def test_en_templates(self):
        from src.cross_turn_slots import _TEMPLATES
        assert len(_TEMPLATES["en"]["ack_short"]) > 0
        assert "Wait" in _TEMPLATES["en"]["self_correct"]


class TestDictation:
    def test_no_error_single_chunk(self):
        rng = random.Random(42)
        turns, meta = _build_dictation_turns("123", "numeric", rng, "vi", "user", 0.20)
        assert len(turns) == 1

    def test_multi_chunk_no_error(self):
        rng = random.Random(42)
        turns, meta = _build_dictation_turns("0901234567", "numeric", rng, "vi", "user", 0.20)
        assert len(turns) >= 3
        assert any("đọc tiếp" in t["content"] or "tiếp đi" in t["content"] for t in turns if t["role"] == "assistant")

    def test_error_repair_structure(self):
        rng = random.Random(42)
        turns, meta = _build_dictation_turns("0901234567", "numeric", rng, "vi", "user", 1.0)  # force error
        contents = [t["content"] for t in turns]
        has_self_correct = any("Khoan" in c or "Wait" in c for c in contents)
        has_ack_correct = any("rồi" in c or "Got it" in c for c in contents)
        assert has_self_correct
        assert has_ack_correct


class TestInjection:
    def test_no_slots(self):
        rng = random.Random(42)
        content = "Xin chào, tôi cần đặt phòng."
        new_content, new_turns, meta = _inject_slots_in_turn(content, rng, "vi", "user", 6, 5, 0.20)
        assert new_content == content
        assert new_turns == []

    def test_phone_slot(self):
        rng = random.Random(42)
        content = "SĐT tôi là 0901234567 nhé."
        new_content, new_turns, meta = _inject_slots_in_turn(content, rng, "vi", "user", 6, 5, 0.20)
        assert len(new_turns) > 0
        assert any("không chín" in t["content"] for t in new_turns)

    def test_email_slot(self):
        rng = random.Random(42)
        content = "Email là john@gmail.com."
        new_content, new_turns, meta = _inject_slots_in_turn(content, rng, "vi", "user", 6, 5, 0.20)
        assert len(new_turns) > 0
        assert any("a-còng" in t["content"] for t in new_turns)

    def test_deterministic(self):
        content = "SĐT: 0901234567."
        r1 = _inject_slots_in_turn(content, random.Random(42), "vi", "user", 6, 5, 0.20)
        r2 = _inject_slots_in_turn(content, random.Random(42), "vi", "user", 6, 5, 0.20)
        assert r1[1] == r2[1]


class TestProcessDialogue:
    def test_idempotence(self):
        rng = random.Random(42)
        history = [
            {"role": "user", "content": "SĐT: 0901234567."},
            {"role": "assistant", "content": "Đã ghi nhận."},
        ]
        h1, m1 = _process_dialogue(history, rng, "vi", "user", 6, 5, 0.20)
        h2, m2 = _process_dialogue(h1, rng, "vi", "user", 6, 5, 0.20)
        assert h1 == h2

    def test_schema_compat(self):
        rng = random.Random(42)
        history = [
            {"role": "user", "content": "SĐT: 0901234567."},
            {"role": "assistant", "content": "OK."},
        ]
        new_history, _ = _process_dialogue(history, rng, "vi", "user", 6, 5, 0.20)
        for turn in new_history:
            assert "role" in turn
            assert "content" in turn
            assert isinstance(turn["content"], str)


class TestPerror:
    def test_perror_distribution(self):
        rng = random.Random(42)
        errors = 0
        for _ in range(500):
            turns, meta = _build_dictation_turns("0901234567", "numeric", rng, "vi", "user", 0.20)
            has_error = any(m.get("error_idx", -1) >= 0 for m in meta)
            if has_error:
                errors += 1
        rate = errors / 500
        assert 0.15 <= rate <= 0.25, f"Perror rate {rate} out of range"
