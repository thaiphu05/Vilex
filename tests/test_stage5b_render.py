"""Tests for Stage 5.1b render (cross-dialogue queue; no GPU: a stub model)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

import stage5b_render as render  # noqa: E402
from stage5_schema import Manifest, Unit, Utterance, VoicePick, write_manifest  # noqa: E402


class StubModel:
    """Batched generate: returns one short wav per input text.

    ``fail_batch`` raises on the list call (forcing the per-item fallback);
    ``fail_all`` raises on both list and single calls (forcing silence).
    """

    def __init__(self, fail_batch=False, fail_all=False):
        self.fail_batch = fail_batch
        self.fail_all = fail_all
        self.batch_calls = 0
        self.item_calls = 0

    def generate(self, **kwargs):
        text = kwargs.get("text")
        is_batch = isinstance(text, list)
        if is_batch:
            self.batch_calls += 1
            if self.fail_batch or self.fail_all:
                raise RuntimeError("batch boom")
            return [np.zeros(1600, dtype=np.float32) for _ in text]
        self.item_calls += 1
        if self.fail_all:
            raise RuntimeError("item boom")
        return np.zeros(1600, dtype=np.float32)


def _pool(tmp_path):
    picks = {}
    for role, freq in (("user", 220.0), ("assistant", 440.0)):
        wav = tmp_path / f"{role}.wav"
        t = torch.arange(12000) / 24000.0
        torchaudio.save(str(wav), (0.1 * torch.sin(2 * np.pi * freq * t)).unsqueeze(0), 24000)
        (tmp_path / f"{role}.txt").write_text(f"ref {role}", encoding="utf-8")
        picks[role] = VoicePick(wav=str(wav), txt=str(tmp_path / f"{role}.txt"))
    return picks


def _manifest(tmp_path, picks, name, units):
    vdir = tmp_path / name / "var00"
    vdir.mkdir(parents=True)
    m = Manifest(
        dialogue_id=name,
        rel_path=f"text_dialogue_interviewer/train/{name}",
        variant_idx=0,
        seed=42,
        voice_picks=picks,
        utterances=[
            Utterance(
                turn=0,
                text_idx=0,
                speaker_idx=0,
                uttr_type=None,
                next_uttr_type="last",
                is_last_text=True,
                isUttered=True,
                tts_text="hello",
                units=units,
            )
        ],
    )
    mpath = vdir / "manifest.json"
    write_manifest(mpath, m)
    return mpath, m


def test_build_jobs_resume_and_pause(tmp_path):
    picks = _pool(tmp_path)
    units = [
        Unit(kind="text", unit_id="u_0_0", text="hi"),
        Unit(kind="pause", unit_id="u_0_1"),
        Unit(kind="text", unit_id="u_0_2", text="bye"),
    ]
    mpath, m = _manifest(tmp_path, picks, "a", units)
    # pretend u_0_0 already rendered
    out = mpath.parent / "units"
    out.mkdir()
    torchaudio.save(str(out / "u_0_0.wav"), torch.zeros(1, 1600), 24000)

    jobs = render.build_jobs([(mpath, m)], {})
    assert [j["unit_id"] for j in jobs] == ["u_0_2"]
    assert "u_0_1" in m.pause_samples and m.pause_samples["u_0_1"] > 0
    assert jobs[0]["out_path"] == out / "u_0_2.wav"
    # resume backfills unit_audio for wavs that exist but were never recorded
    assert m.unit_audio["u_0_0"] == str(out / "u_0_0.wav")


def test_chunk_jobs_by_batch_size_and_long_units():
    jobs = [
        {"unit_id": f"u{i}", "text": "short", "mi": 0, "ref_wav": "x", "is_bc": False}
        for i in range(5)
    ]
    jobs.append({"unit_id": "long", "text": "x" * 50, "mi": 0, "ref_wav": "x", "is_bc": False})

    chunks = render.chunk_jobs(jobs, batch_size=2, max_unit_chars=10)
    sizes = [len(c) for c in chunks]
    assert sizes == [2, 2, 1, 1]  # 5 short -> 2/2/1, then the long unit alone
    assert chunks[-1][0]["unit_id"] == "long"


def test_render_chunks_batches_cross_dialogue_and_writes(tmp_path):
    picks = _pool(tmp_path)
    mpath_a, m_a = _manifest(tmp_path, picks, "a", [Unit(kind="text", unit_id="u_0_0", text="hi")])
    mpath_b, m_b = _manifest(tmp_path, picks, "b", [Unit(kind="text", unit_id="u_0_0", text="yo")])
    items = [(mpath_a, m_a), (mpath_b, m_b)]
    ref_prep = render.RefPrep(tmp_path / "_batch", 10.0)

    jobs = render.build_jobs(items, {})
    chunks = render.chunk_jobs(jobs, batch_size=8, max_unit_chars=400)
    assert len(chunks) == 1 and len(chunks[0]) == 2  # one batch across both dialogues

    rendered, failed = render.render_chunks(StubModel(), chunks, [m_a, m_b], ref_prep, 1.2, "vi", 2)
    assert (rendered, failed) == (2, 0)
    assert (mpath_a.parent / "units" / "u_0_0.wav").is_file()
    assert (mpath_b.parent / "units" / "u_0_0.wav").is_file()
    assert m_a.unit_audio["u_0_0"].endswith("a/var00/units/u_0_0.wav")


def test_batch_failure_falls_back_to_per_item(tmp_path):
    picks = _pool(tmp_path)
    mpath, m = _manifest(tmp_path, picks, "a", [Unit(kind="text", unit_id="u_0_0", text="hi")])
    ref_prep = render.RefPrep(tmp_path / "_batch", 10.0)
    chunks = render.chunk_jobs(render.build_jobs([(mpath, m)], {}), 8, 400)

    model = StubModel(fail_batch=True)
    rendered, failed = render.render_chunks(model, chunks, [m], ref_prep, 1.2, "vi", 1)
    assert (rendered, failed) == (1, 0)
    assert model.batch_calls == 2 and model.item_calls == 1  # list retried twice, then per item
    assert m.failed_units == []


def test_all_failures_become_silence(tmp_path):
    picks = _pool(tmp_path)
    mpath, m = _manifest(tmp_path, picks, "a", [Unit(kind="text", unit_id="u_0_0", text="hi")])
    ref_prep = render.RefPrep(tmp_path / "_batch", 10.0)
    chunks = render.chunk_jobs(render.build_jobs([(mpath, m)], {}), 8, 400)

    rendered, failed = render.render_chunks(
        StubModel(fail_all=True), chunks, [m], ref_prep, 1.2, "vi", 1
    )
    assert (rendered, failed) == (1, 1)
    assert m.failed_units == ["u_0_0"]
    wav, _sr = torchaudio.load(str(mpath.parent / "units" / "u_0_0.wav"))
    assert wav.numel() == int(0.2 * render.TARGET_SR)


def test_refuses_mutated_pool(tmp_path):
    picks = _pool(tmp_path)
    picks["user"].wav_size = int(picks["user"].wav_size) + 1
    _mpath, m = _manifest(tmp_path, picks, "a", [Unit(kind="text", unit_id="u_0_0", text="hi")])
    with pytest.raises(SystemExit, match="voice pool changed"):
        render.check_fingerprint(m)
