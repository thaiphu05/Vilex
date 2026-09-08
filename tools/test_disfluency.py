"""Tests for disfluency module. 0 API calls for rule-based tests."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.disfluency import (
    p_disfluent, _find_slot_positions, _inject_disfluency,
    _insert_fp, _insert_dm, _insert_edit, _insert_rep,
    FP_INVENTORY, DM_INVENTORY, EDIT_INVENTORY,
)


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
        assert result[1] in FP_INVENTORY["vi"]

    def test_dm(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_dm(words, 1, "vi", rng)
        assert len(result) == len(words) + 1
        assert result[1] in DM_INVENTORY["vi"]

    def test_edit(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_edit(words, 1, "vi", rng)
        assert len(result) == len(words) + 1
        assert result[1] in EDIT_INVENTORY["vi"]

    def test_rep(self):
        rng = random.Random(42)
        words = "Xin chào bạn".split()
        result = _insert_rep(words, 1, "vi", rng)
        assert len(result) > len(words)
        assert result[1] == result[2] or result[1] == result[3]


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
            _, m2 = _inject_disfluency("a b c d e f g h i j k l m n o p q r s t u v w x y z", rng, "vi")
            short_count += len(m1)
            long_count += len(m2)
        assert long_count > short_count


class TestInventory:
    def test_fp_not_empty(self):
        assert len(FP_INVENTORY["vi"]) > 0
        assert len(FP_INVENTORY["en"]) > 0

    def test_dm_not_empty(self):
        assert len(DM_INVENTORY["vi"]) > 0
        assert len(DM_INVENTORY["en"]) > 0

    def test_edit_not_empty(self):
        assert len(EDIT_INVENTORY["vi"]) > 0
        assert len(EDIT_INVENTORY["en"]) > 0
