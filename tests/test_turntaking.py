"""Tests for turn-taking floor_taking insertion."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthesis.core import insert_action_tokens_from_llm_annotations


class TestTurnLevelFt:
    def test_max_one_ft_per_turn(self):
        text = "tôi sẽ dùng chiến lược này để bán hàng nhiều"
        b_idx = [2, 4, 6]
        b_dists = [{"floor_taking": 0.99, "backchannel": 0.0, "silence": 0.01}] * 3
        out, _ = insert_action_tokens_from_llm_annotations(
            text,
            b_idx,
            b_dists,
            length_guard_start=0,
            length_guard_gap=4,
            interruption_guard_start=0,
            rng=random.Random(42),
        )
        assert out.count("[TAKE_FLOOR]") == 1

    def test_ft_can_be_inserted_at_terminal(self):
        text = "tôi sẽ dùng chiến lược. để bán hàng"
        words = text.split()
        dot_idx = next(i for i, w in enumerate(words) if w.endswith("."))
        b_idx = [dot_idx]
        b_dists = [{"floor_taking": 0.99, "backchannel": 0.0, "silence": 0.01}]
        out, _ = insert_action_tokens_from_llm_annotations(
            text,
            b_idx,
            b_dists,
            length_guard_start=0,
            length_guard_gap=4,
            interruption_guard_start=0,
            rng=random.Random(42),
        )
        assert "[TAKE_FLOOR]" in out
