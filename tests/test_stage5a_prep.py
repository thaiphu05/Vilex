"""Tests for Stage 5.1a (normalized text prep).

Covers: drop filter passthrough, manifest structure, positional channel
assignment, BC word_count/host_idx anchors, rising-token rewrite, seeded BC
fallback determinism, JSONL emission (text units + BCs, no pauses), pool
sidecars on disk under a voice_clone-like path, and the resume path that skips
ids already rendered by 5.1b.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

import stage5_lib as lib  # noqa: E402
import stage5a_prep as prep  # noqa: E402
import stage5_schema as sch  # noqa: E402

REL = Path("text_dialogue_interviewer/train/work_0000")


@pytest.fixture
def pool(tmp_path):
    """Real wav+txt pairs under voice_clone/ so JSONL ref_audio resolves on disk."""
    pool_dir = tmp_path / "voice_clone"
    pool_dir.mkdir()
    items = []
    for name, text in (("a", "ref a"), ("b", "ref b"), ("c", "ref c")):
        wav = pool_dir / f"{name}.wav"
        txt = pool_dir / f"{name}.txt"
        wav.write_bytes(b"RIFF....fake")
        txt.write_text(text, encoding="utf-8")
        st = wav.stat()
        items.append(
            {
                "wav": str(wav.resolve()),
                "txt": str(txt.resolve()),
                "ref_text": text,
                "wav_size": int(st.st_size),
                "wav_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            }
        )
    return items


def _dialogue():
    return {
        "history": [
            {
                "role": "user",
                "content": "",
                "history": [
                    {"full_content": "Chao ban, toi muon dat lich hen vao sang mai"},
                    {
                        "word_index": 3,
                        "decision": "backchannel",
                        "inserted_token": "[BACKCHANNEL]",
                        "content": "um",
                    },
                    {
                        "word_index": 5,
                        "decision": "backchannel",
                        "inserted_token": "[BACKCHANNEL]",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "history": [
                    {"full_content": "Vang, toi co the sap xep lich cho ban"},
                ],
            },
        ]
    }


def test_turns_split_and_channels_are_positional(pool):
    utts = lib.convert_dialogue_to_utterances(_dialogue(), pick_bc=lambda: "fb")
    assert [u["uttr_type"] for u in utts] == [None, "backchannel", None, "backchannel", None, None]
    m = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 42, pool, False, "vi")
    # 1 merged user host + 1 assistant host; BCs on the listener channel
    assert [u.speaker_idx for u in m.utterances] == [0, 1]
    assert all(u.is_last_text for u in m.utterances)
    assert [b.speaker_idx for b in m.backchannels] == [1, 1]
    assert all(b.host_idx == 0 for b in m.backchannels)
    assert all(b.word_count > 0 for b in m.backchannels)


def test_bc_text_and_rising_rewrite(pool):
    m = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 42, pool, False, "vi")
    texts = [b.text for b in m.backchannels]
    assert texts[0] == "um"
    assert texts[1] == "v\u00e2ng?"  # seeded fallback (VI pool) + rising "?"


def test_manifest_validates_and_jsonl_is_complete(pool):
    m = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 42, pool, False, "vi")
    sch.validate_manifest(m)
    assert m.voice_picks["user"].wav_size > 0
    assert m.voice_picks["assistant"].wav_mtime_ns > 0
    rows = prep.manifest_jsonl_rows(m, Path("/nonexistent"))
    # 2 host text units + 2 BC lines; pause units never enter the JSONL
    assert len(rows) == 4
    assert all(
        set(r) == {"id", "text", "ref_audio", "ref_text", "language_id", "speed"} for r in rows
    )
    assert all(r["language_id"] == "vi" and r["speed"] == 1.2 for r in rows)
    for r in rows:
        assert "voice_clone" in r["ref_audio"]
        assert Path(r["ref_audio"]).is_file()
        assert r["ref_text"] in ("ref a", "ref b", "ref c")


def test_seeded_fallback_deterministic_across_runs(pool):
    a = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 42, pool, False, "vi")
    b = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 42, pool, False, "vi")
    assert sch.manifest_to_dict(a) == sch.manifest_to_dict(b)
    ra = prep.manifest_jsonl_rows(a, Path("/x"))
    rb = prep.manifest_jsonl_rows(b, Path("/x"))
    assert [r["id"] for r in ra] == [r["id"] for r in rb]
    assert [r["text"] for r in ra] == [r["text"] for r in rb]


def test_voice_cast_distinct_and_seed_dependent(pool):
    m = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 42, pool, False, "vi")
    assert m.voice_picks["user"].wav != m.voice_picks["assistant"].wav
    other = prep.build_manifest(_dialogue(), "work_0000", REL, 0, 99, pool, False, "vi")
    # Different seed must change at least one pick for this 3-voice pool.
    assert (
        m.voice_picks["user"].wav != other.voice_picks["user"].wav
        or m.voice_picks["assistant"].wav != other.voice_picks["assistant"].wav
    )


def test_drop_filter_marks_bad_dialogues():
    # Multi numbered-role leak trips should_drop (convert_spoken:589+).
    bad = {"history": [{"role": "user", "content": "1) user: hello 2) assistant: hi"}]}
    drop, reason = lib.should_drop_dialogue(bad)
    assert drop and "multi-turn concat" in reason
    ok = _dialogue()
    assert lib.should_drop_dialogue(ok) == (False, "")


def test_pause_unit_never_in_jsonl(pool):
    d = _dialogue()
    d["history"][0]["history"][0]["full_content"] += " ... nghi mot chut"
    m = prep.build_manifest(d, "work_0000", REL, 0, 42, pool, False, "vi")
    sch.validate_manifest(m)
    pause_ids = {u.unit_id for u in m.utterances[0].units if u.kind == "pause"}
    assert pause_ids, "expected a [PAUSE] unit from the ellipsis"
    rows = prep.manifest_jsonl_rows(m, Path("/x"))
    ids = {r["id"].split("__")[-1] for r in rows}
    assert not (pause_ids & ids)


def test_resume_skips_units_with_audio_on_disk(pool, tmp_path):
    d = _dialogue()
    m = prep.build_manifest(d, "work_0000", REL, 0, 42, pool, False, "vi")
    rows = prep.manifest_jsonl_rows(m, Path("/x"))
    first = rows[0]["id"].split("__")[-1]
    wav = tmp_path / f"{first}.wav"
    wav.write_bytes(b"RIFF")
    m.unit_audio[first] = str(wav)
    sch.write_manifest(tmp_path / "manifest.json", m)
    prior = prep._existing_unit_audio(tmp_path / "manifest.json")
    assert prior[first] == str(wav)
    kept = [
        r
        for r in prep.manifest_jsonl_rows(sch.read_manifest(tmp_path / "manifest.json"), tmp_path)
        if not (r["id"].split("__")[-1] in prior and Path(prior[r["id"].split("__")[-1]]).is_file())
    ]
    assert len(kept) == len(rows) - 1


def test_voice_pool_index_records_fingerprint(tmp_path):
    pool_dir = tmp_path / "voice_clone"
    pool_dir.mkdir()
    for name in ("u", "a"):
        (pool_dir / f"{name}.wav").write_bytes(b"RIFF1234")
        (pool_dir / f"{name}.txt").write_text(f"transcript {name}", encoding="utf-8")
    items = prep._voice_pool_index(pool_dir)
    assert len(items) == 2
    assert all(i["wav_size"] > 0 and i["wav_mtime_ns"] > 0 for i in items)
    assert all(Path(i["wav"]).is_file() and "voice_clone" in i["wav"] for i in items)
