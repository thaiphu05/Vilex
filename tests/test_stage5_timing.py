"""Tests for Stage 5 sampled GAP / PAUSE timing distributions.

Covers `_sample_gap`, `_sample_pause`, and the timing labels that
`aggregate_speech` writes onto `speech_meta` entries. The TTS model and
WhisperX alignment are not exercised -- only the timing math.
"""

import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

from tts_render.convert_spoken import (  # noqa: E402
    GAP_EXP_SCALE,
    GAP_MAX_SEC,
    GAP_MIN_SEC,
    PAUSE_EXP_SCALE,
    PAUSE_INTRA_EXP_SCALE,
    PAUSE_INTRA_MAX_SEC,
    PAUSE_INTRA_MIN_SEC,
    PAUSE_MAX_SEC,
    PAUSE_MIN_SEC,
    PAUSE_TOKEN,
    USER_INTERRUPT_OVERLAP_SEC,
    USER_INTERRUPT_PROB,
    _INTENTIONAL_BRACKET_TOKENS,
    _sample_gap,
    _sample_intra_pause,
    _sample_pause,
    aggregate_speech,
)

# --- Distribution sampling ---------------------------------------------------


class TestSampleGap:
    def test_within_bounds(self):
        np.random.seed(0)
        for _ in range(2000):
            g = _sample_gap(3.0)
            assert GAP_MIN_SEC <= g <= GAP_MAX_SEC

    def test_median_around_half_second(self):
        np.random.seed(123)
        samples = [_sample_gap(3.0) for _ in range(2000)]
        median = float(np.median(samples))
        # Fitted scale 0.7067s -> median ~0.49s; allow wide slack for sample noise.
        assert 0.25 <= median <= 0.85, median

    def test_turn_duration_scaling(self):
        # Very short turns should compress the gap; very long turns should stretch it.
        np.random.seed(42)
        short = np.mean([_sample_gap(0.5) for _ in range(2000)])
        np.random.seed(42)
        long = np.mean([_sample_gap(12.0) for _ in range(2000)])
        assert long > short


class TestSamplePause:
    def test_within_bounds(self):
        np.random.seed(0)
        for _ in range(2000):
            p = _sample_pause()
            assert PAUSE_MIN_SEC <= p <= PAUSE_MAX_SEC

    def test_median_around_half_second(self):
        np.random.seed(7)
        samples = [_sample_pause() for _ in range(2000)]
        median = float(np.median(samples))
        assert 0.25 <= median <= 0.85, median


class TestSampleIntraPause:
    """Short pause used for an explicit [PAUSE] token between utterances."""

    def test_bounds_config(self):
        assert PAUSE_INTRA_MIN_SEC == 0.1
        assert PAUSE_INTRA_MAX_SEC == 1.0
        assert 0.0 < PAUSE_INTRA_EXP_SCALE < PAUSE_INTRA_MAX_SEC

    def test_within_bounds(self):
        np.random.seed(0)
        for _ in range(2000):
            p = _sample_intra_pause()
            assert PAUSE_INTRA_MIN_SEC <= p <= PAUSE_INTRA_MAX_SEC


class TestPauseTokenHandling:
    def test_pause_token_in_intentional_tokens(self):
        assert PAUSE_TOKEN == "[PAUSE]"
        assert "[PAUSE]" in _INTENTIONAL_BRACKET_TOKENS


# --- aggregate_speech labelling ----------------------------------------------


def _two_ch_silence(samples):
    import torch

    return torch.zeros(2, samples, dtype=torch.float32)


class TestAggregateTimingLabels:
    def _build_speech(self, chunk_samples=2400):
        return [_two_ch_silence(chunk_samples) for _ in range(3)]

    def _build_meta(self, n=3):
        return [
            {"uttr_type": None, "speaker": 0, "tts_text": "a", "boundary": "turn"},
            {"uttr_type": None, "speaker": 1, "tts_text": "b", "boundary": "turn"},
            {"uttr_type": None, "speaker": 0, "tts_text": "c", "boundary": "turn"},
        ][:n]

    def test_first_chunk_is_none(self):
        random.seed(0)
        np.random.seed(0)
        meta = self._build_meta()
        merged = aggregate_speech(self._build_speech(), meta)
        assert merged.size(1) > 0
        assert meta[0]["timing"] == "none"
        assert meta[0]["duration_sec"] == 0.0

    def test_speaker_change_with_bc_takes_gap_branch(self):
        random.seed(0)
        np.random.seed(0)
        speech = [
            _two_ch_silence(2400),
            _two_ch_silence(2400),
        ]
        # Pre-populate the assistant channel on the second chunk with a backchannel
        # so the BC-leader branch fires and a GAP is sampled.
        speech[1][1, :100] = 0.05
        meta = [
            {"uttr_type": None, "speaker": 0, "tts_text": "a", "boundary": "turn"},
            {"uttr_type": None, "speaker": 1, "tts_text": "b", "boundary": "turn"},
        ]
        aggregate_speech(speech, meta)
        assert meta[1]["timing"] == "gap"
        assert GAP_MIN_SEC <= meta[1]["duration_sec"] <= GAP_MAX_SEC

    def test_same_speaker_turn_boundary_takes_pause(self):
        random.seed(0)
        np.random.seed(0)
        speech = [
            _two_ch_silence(2400),
            _two_ch_silence(2400),
        ]
        meta = [
            {"uttr_type": None, "speaker": 0, "tts_text": "a", "boundary": "turn"},
            {"uttr_type": None, "speaker": 0, "tts_text": "b", "boundary": "turn"},
        ]
        aggregate_speech(speech, meta)
        assert meta[1]["timing"] == "pause"
        assert PAUSE_MIN_SEC <= meta[1]["duration_sec"] <= PAUSE_MAX_SEC

    def test_same_speaker_mid_utterance_is_none(self):
        random.seed(0)
        np.random.seed(0)
        speech = [
            _two_ch_silence(2400),
            _two_ch_silence(2400),
        ]
        meta = [
            {"uttr_type": None, "speaker": 0, "tts_text": "a", "boundary": "turn"},
            {"uttr_type": None, "speaker": 0, "tts_text": "b", "boundary": "sentence"},
        ]
        aggregate_speech(speech, meta)
        assert meta[1]["timing"] == "none"
        assert meta[1]["duration_sec"] == 0.0

    def test_interrupt_label_is_overlap(self):
        random.seed(0)
        np.random.seed(0)
        speech = [
            _two_ch_silence(4800),  # assistant already speaking
            _two_ch_silence(2400),  # user barges in
        ]
        meta = [
            {"uttr_type": None, "speaker": 1, "tts_text": "a", "boundary": "turn"},
            {"uttr_type": "interrupt", "speaker": 0, "tts_text": "b", "boundary": "turn"},
        ]
        aggregate_speech(speech, meta)
        assert meta[1]["timing"] == "overlap"
        assert 0.0 <= meta[1]["duration_sec"] <= USER_INTERRUPT_OVERLAP_SEC


# --- Configuration sanity -----------------------------------------------------


def test_user_interrupt_prob_below_one():
    assert 0.0 < USER_INTERRUPT_PROB < 1.0


def test_overlap_cap_positive():
    assert USER_INTERRUPT_OVERLAP_SEC > 0


def test_gap_bounds_consistent():
    assert GAP_MIN_SEC < GAP_EXP_SCALE < GAP_MAX_SEC
    assert PAUSE_MIN_SEC < PAUSE_EXP_SCALE < PAUSE_MAX_SEC
