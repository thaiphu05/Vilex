"""Tests for turn-taking floor_taking filtering and turn-level decision."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthesis.core import _ft_candidate, insert_action_tokens_from_llm_annotations


class TestFtCandidate:
    def test_excludes_terminal_punct(self):
        words = ["tôi", "sẽ", "dùng", "chiến", "lược.", "này", "để", "bán", "hàng"]
        paired = [(4, {"floor_taking": 0.85, "backchannel": 0.1, "silence": 0.05})]
        assert _ft_candidate(paired, words, guard=3) is None

    def test_picks_max_non_terminal(self):
        words = ["tôi", "sẽ", "dùng", "chiến", "lược", "này", "để"]
        paired = [
            (4, {"floor_taking": 0.3, "backchannel": 0.5, "silence": 0.2}),
            (5, {"floor_taking": 0.85, "backchannel": 0.1, "silence": 0.05}),
        ]
        idx, _ = _ft_candidate(paired, words, guard=3)
        assert idx == 5

    def test_respects_guard(self):
        words = ["tôi", "sẽ", "dùng"]
        paired = [(1, {"floor_taking": 0.9, "backchannel": 0.05, "silence": 0.05})]
        assert _ft_candidate(paired, words, guard=3) is None

    def test_returns_max_even_if_zero(self):
        words = ["tôi", "sẽ", "dùng", "nó"]
        paired = [(3, {"floor_taking": 0.0, "backchannel": 0.5, "silence": 0.5})]
        idx, probs = _ft_candidate(paired, words, guard=0)
        assert idx == 3
        assert probs["floor_taking"] == 0.0


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

    def test_no_ft_at_terminal(self):
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
        assert "[TAKE_FLOOR]" not in out
