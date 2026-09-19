"""Integration test for Stage 5.2b assemble (synthetic manifest + alignment)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

import convert_spoken as cs  # noqa: E402
from src.config import load_config  # noqa: E402

import stage5d_assemble as assemble  # noqa: E402
from stage5_schema import Manifest, Unit, Utterance, VoicePick  # noqa: E402


def _wav(path: Path, seconds=1.0, freq=220.0):
    t = torch.arange(int(seconds * 24000)) / 24000.0
    torchaudio.save(str(path), (0.1 * torch.sin(2 * 3.14159 * freq * t)).unsqueeze(0), 24000)
    return path


def _fixture(tmp_path):
    cs._apply_tts_config(load_config(REPO / "config_example.yaml"))
    vdir = tmp_path / "var00"
    vdir.mkdir()
    host_wav = _wav(vdir / "h_000.wav", 1.0, 220.0)
    bc_wav = _wav(vdir / "bc_0.wav", 0.4, 440.0)

    m = Manifest(
        dialogue_id="work_0000",
        rel_path="text_dialogue_interviewer/train/work_0000",
        variant_idx=0,
        seed=42,
        voice_picks={
            "user": VoicePick(wav=str(host_wav), txt=str(host_wav)),
            "assistant": VoicePick(wav=str(bc_wav), txt=str(bc_wav)),
        },
        utterances=[
            Utterance(
                turn=0,
                text_idx=0,
                speaker_idx=0,
                uttr_type=None,
                next_uttr_type="last",
                is_last_text=True,
                isUttered=True,
                tts_text="xin chào bạn",
                units=[Unit(kind="text", unit_id="u_0_0", text="xin chào bạn")],
            )
        ],
    )
    words = [
        {"word": "xin", "start": 0.0, "end": 0.2},
        {"word": "chào", "start": 0.2, "end": 0.5},
        {"word": "bạn", "start": 0.5, "end": 0.9},
    ]
    align = {
        "dialogue_id": "work_0000",
        "variant_idx": 0,
        "hosts": [
            {
                "host_idx": 0,
                "speaker_idx": 0,
                "text": "xin chào bạn",
                "words": words,
                "unit_spans": [["xin chào bạn", 0, 24000]],
                "dur_sec": 1.0,
                "audio": str(host_wav),
            }
        ],
        "bcs": [
            {
                "unit_id": "bc_0",
                "bc_idx": 0,
                "host_idx": 0,
                "speaker_idx": 1,
                "text": "ừ",
                "word_count": 1,
                "words": [],
                "audio": str(bc_wav),
            }
        ],
        "fallback": [],
    }
    return m, align, vdir


def test_assemble_writes_final_schema(tmp_path):
    m, align, vdir = _fixture(tmp_path)
    out_dir = tmp_path / "out"
    assemble.assemble_variant(cs, m, align, vdir, out_dir, {})

    assert (out_dir / "dialogues" / "dialogue.wav").is_file()
    assert (out_dir / "user.wav").is_file()
    assert (out_dir / "assistant.wav").is_file()
    assert (out_dir / "utterances" / "00.wav").is_file()
    assert list((out_dir / "backchannels").glob("*.wav")), "backchannel wav missing"

    meta = json.loads((out_dir / "meta.json").read_text())
    for key in ("speakers", "speech_meta", "timing_config", "variant_idx"):
        assert key in meta, key
    assert meta["speakers"] == ["assistant", "user"]
    assert "start_sample" in meta["speech_meta"][0]
