"""Tests for the Stage 5 split-pipeline manifest schema (5.1a output contract).

Pure stdlib: no torch / OmniVoice needed. The manifest is the hand-off between
5.1a -> 5.1b -> 5.2a -> 5.2b, so a silent schema drift is the failure mode
these pin down.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

import stage5_schema as sch  # noqa: E402


def _manifest(**over):
    base = dict(
        dialogue_id="work_0000",
        rel_path="text_dialogue_interviewer/train/work_0000",
        variant_idx=0,
        seed=42,
        voice_picks={
            "user": sch.VoicePick(
                wav="/pool/u.wav", txt="/pool/u.txt", wav_size=10, wav_mtime_ns=100
            ),
            "assistant": sch.VoicePick(
                wav="/pool/a.wav", txt="/pool/a.txt", wav_size=20, wav_mtime_ns=200
            ),
        },
        utterances=[
            sch.Utterance(
                turn=0,
                text_idx=0,
                speaker_idx=0,
                uttr_type=None,
                next_uttr_type="backchannel",
                is_last_text=True,
                isUttered=True,
                tts_text="xin chao",
                units=[sch.Unit(kind="text", unit_id="u_0_0", text="xin chao")],
            )
        ],
        backchannels=[
            sch.Backchannel(
                bc_idx=0,
                speaker_idx=1,
                text="u?",
                word_count=2,
                host_idx=0,
                next_uttr_type_of_host="backchannel",
                unit_id="bc_0",
            )
        ],
    )
    base.update(over)
    return sch.Manifest(**base)


def test_valid_manifest_passes():
    assert sch.validate_manifest(_manifest()) is not None


def test_roundtrip_through_json(tmp_path):
    m = _manifest()
    p = tmp_path / "var00" / "manifest.json"
    sch.write_manifest(p, m)
    back = sch.read_manifest(p)
    assert sch.manifest_to_dict(back) == sch.manifest_to_dict(m)
    assert back.backchannels[0].unit_id == "bc_0"
    assert back.utterances[0].units[0].kind == "text"


def test_schema_version_mismatch_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        sch.validate_manifest(_manifest(schema_version=999))


def test_wrong_stage_is_rejected():
    with pytest.raises(ValueError, match="stage"):
        sch.validate_manifest(_manifest(stage="5.1b"))


def test_missing_voice_role_is_rejected():
    m = _manifest()
    m.voice_picks.pop("assistant")
    with pytest.raises(ValueError, match="voice_picks keys"):
        sch.validate_manifest(m)


def test_empty_text_unit_is_rejected():
    m = _manifest()
    m.utterances[0].units = [sch.Unit(kind="text", unit_id="u_0_0", text="")]
    with pytest.raises(ValueError, match="empty text"):
        sch.validate_manifest(m)


def test_duplicate_unit_id_is_rejected():
    m = _manifest()
    m.utterances[0].units.append(sch.Unit(kind="text", unit_id="u_0_0", text="dup"))
    with pytest.raises(ValueError, match="duplicate unit_id"):
        sch.validate_manifest(m)


def test_backchannel_host_idx_out_of_range_is_rejected():
    m = _manifest()
    m.backchannels[0].host_idx = 7
    with pytest.raises(ValueError, match="host_idx"):
        sch.validate_manifest(m)


def test_backchannel_unit_id_collides_with_host_unit():
    m = _manifest()
    m.backchannels[0].unit_id = "u_0_0"
    with pytest.raises(ValueError, match="collides"):
        sch.validate_manifest(m)


def test_bad_uttr_type_is_rejected():
    m = _manifest()
    m.utterances[0].uttr_type = "floor_taking"  # label taxonomy, not utterance type
    with pytest.raises(ValueError, match="uttr_type"):
        sch.validate_manifest(m)


def test_from_dict_missing_field_raises():
    d = sch.manifest_to_dict(_manifest())
    del d["dialogue_id"]
    with pytest.raises(KeyError):
        sch.manifest_from_dict(d)


def test_5_1b_owned_fields_default_empty_and_survive_roundtrip(tmp_path):
    m = _manifest(
        failed_units=["u_0_1"],
        pause_samples={"u_0_1": 4800},
        unit_audio={"u_0_0": "/batch/audio/u_0_0.wav"},
    )
    sch.write_manifest(tmp_path / "m.json", m)
    back = sch.read_manifest(tmp_path / "m.json")
    assert back.failed_units == ["u_0_1"]
    assert back.pause_samples == {"u_0_1": 4800}
    assert back.unit_audio == {"u_0_0": "/batch/audio/u_0_0.wav"}
    assert back.voice_picks["user"].wav_size == 10
    assert back.voice_picks["assistant"].wav_mtime_ns == 200


def test_negative_fingerprint_is_rejected():
    m = _manifest()
    m.voice_picks["user"].wav_size = -1
    with pytest.raises(ValueError, match="fingerprint"):
        sch.validate_manifest(m)
