"""Tests for the Stage 5 batching/fallback helpers in convert_spoken.

Only the pure helpers are exercised (no OmniVoice / aligner weights): batched
unit generation with retry+silence, aligner fallback policy, sentence-granularity
offsetting, and backchannel placement.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

torch = pytest.importorskip("torch")

import tts_render.convert_spoken as cs  # noqa: E402


def test_audio_to_tensor_wraps_mono_and_guards_empty():
    wav = cs._audio_to_tensor(np.zeros(240, dtype=np.float32))
    assert tuple(wav.shape) == (1, 240)
    empty = cs._audio_to_tensor(np.zeros(0, dtype=np.float32))
    assert empty.size(1) == int(0.2 * cs.TARGET_SR)


def test_words_with_fallback_drop_and_proportional(monkeypatch):
    monkeypatch.setattr(cs, "ALIGN_FALLBACK", "drop")
    assert cs._words_with_fallback([], "a b c", 3.0) == []

    monkeypatch.setattr(cs, "ALIGN_FALLBACK", "proportional")
    words = cs._words_with_fallback([], "a b", 2.0)
    assert [w["word"] for w in words] == ["a", "b"]
    assert words[0]["start"] == 0.0 and words[-1]["end"] == 2.0

    kept = [{"word": "x", "start": 0.0, "end": 0.1}]
    assert cs._words_with_fallback(kept, "a", 1.0) is kept


class _StubModel:
    """Records each generate() call; returns one short wav per input text."""

    def __init__(self, fail_calls=0, short_return=False):
        self.calls = []
        self.fail_calls = fail_calls
        self.short_return = short_return

    def generate(self, **kwargs):
        self.calls.append(list(kwargs.get("text") or []))
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise RuntimeError("boom")
        texts = kwargs.get("text") or []
        n = len(texts) - 1 if self.short_return else len(texts)
        return [np.zeros(1600, dtype=np.float32) for _ in range(max(0, n))]


def _units(*texts):
    units = []
    for i, t in enumerate(texts):
        if i:
            units.append(("pause", ""))
        units.append(("text", t))
    return units


def test_batch_generate_units_groups_and_splits_long(monkeypatch):
    monkeypatch.setattr(cs, "OMNI_BATCH_SIZE", 2)
    monkeypatch.setattr(cs, "OMNI_MAX_UNIT_CHARS", 10)
    monkeypatch.setattr(cs, "OMNI_MAX_RETRIES", 0)
    stub = _StubModel()
    units = _units("short a", "short b", "short c", "x" * 20)

    pregen, failed = cs._batch_generate_units(
        stub, units, 0, [None, None], ["instruct", "instruct"]
    )

    assert failed == set()
    assert set(pregen) == {0, 2, 4, 6}
    # 2-item chunk + 1-item chunk + the long unit on its own
    assert [len(c) for c in stub.calls] == [2, 1, 1]
    assert stub.calls[-1] == ["x" * 20]


def test_batch_generate_units_retries_then_silences(monkeypatch):
    monkeypatch.setattr(cs, "OMNI_BATCH_SIZE", 8)
    monkeypatch.setattr(cs, "OMNI_MAX_UNIT_CHARS", 400)
    monkeypatch.setattr(cs, "OMNI_MAX_RETRIES", 2)
    stub = _StubModel(fail_calls=1)  # fails once, then succeeds

    pregen, failed = cs._batch_generate_units(
        stub, _units("hello"), 0, [None, None], ["instruct", "instruct"]
    )
    assert failed == set() and 0 in pregen
    assert len(stub.calls) == 2  # first attempt + one retry

    always_fail = _StubModel(fail_calls=99)
    pregen, failed = cs._batch_generate_units(
        always_fail, _units("hello"), 0, [None, None], ["instruct", "instruct"]
    )
    assert failed == {0}
    assert pregen[0].size(1) == int(0.2 * cs.TARGET_SR)
    assert len(always_fail.calls) == 3  # 1 + OMNI_MAX_RETRIES


def test_align_hosts_sentence_offsets(monkeypatch):
    monkeypatch.setattr(cs, "ALIGN_GRANULARITY", "sentence")
    monkeypatch.setattr(cs, "ALIGN_BATCH", True)
    monkeypatch.setattr(cs, "ALIGN_FALLBACK", "drop")

    def fake_align_items(align, items):
        # one word per unit, relative to the unit's own start
        return [[{"word": f"w{i}", "start": 0.0, "end": 0.1}] for i, _ in enumerate(items)]

    monkeypatch.setattr(cs, "_align_items", fake_align_items)

    audio = torch.zeros(1, 24000)  # 1 s at 24 kHz
    spec = {
        "audio": audio,
        "text": "one two",
        "units": [("one", 0, 12000), ("two", 12000, 24000)],
    }
    words = cs._align_hosts(None, [spec])[0]
    assert [w["word"] for w in words] == ["w0", "w1"]
    assert words[0]["start"] == 0.0
    assert words[1]["start"] == pytest.approx(12000 / cs.TARGET_SR)


def test_place_queued_bcs_drops_without_words():
    bcs = [{"speech": torch.ones(1, 100), "word_count": 1, "turn": 0, "text_idx": 0, "text": "ừ"}]
    listener = torch.zeros(1, 1000)
    host = torch.ones(1, 1000)
    utterance = {"texts_styled": [{"isUttered": False}]}
    modified = [utterance]

    listener, host, uttered, entries = cs._place_queued_bcs(
        bcs, [], listener, host, 1, 0, utterance, "last", modified, None, 0, False
    )
    assert uttered == [] and entries == []
    assert utterance["texts_styled"][0]["isUttered"] is False


def test_place_queued_bcs_writes_when_words_present():
    bcs = [{"speech": torch.ones(1, 100), "word_count": 1, "turn": 0, "text_idx": 0, "text": "ừ"}]
    listener = torch.zeros(1, 1000)
    host = torch.ones(1, 1000)
    utterance = {"texts_styled": [{"isUttered": False}]}
    words = [{"word": "a", "start": 0.0, "end": 0.004}, {"word": "b", "start": 0.004, "end": 0.02}]

    listener, host, uttered, entries = cs._place_queued_bcs(
        bcs, words, listener, host, 1, 0, utterance, "last", [utterance], None, 0, False
    )
    assert len(uttered) == 1
    assert utterance["texts_styled"][0]["isUttered"] is True
    assert listener.abs().sum() > 0
