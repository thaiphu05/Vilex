import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import convert_format as cf  # noqa: E402


def _touch(path: Path, data: bytes = b"RIFF"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_is_audio_filename_matches_common_extensions():
    assert cf.is_audio_filename("a.wav")
    assert cf.is_audio_filename("b.WAV")
    assert cf.is_audio_filename("dir/c.flac")
    assert cf.is_audio_filename("d.mp3")
    assert not cf.is_audio_filename("transcript")
    assert not cf.is_audio_filename("hello.wav.txt")


def test_parse_manifest_alternating_with_trailing_spaces():
    text = "ten_file.wav\ntranscript \nten_file_2.wav \ntranscript \n"
    pairs, warnings = cf.parse_manifest(text)
    assert pairs == [("ten_file.wav", "transcript"), ("ten_file_2.wav", "transcript")]
    assert warnings == []


def test_parse_manifest_multiline_transcript_is_joined():
    text = "a.wav\nline one\nline two\n\nline three\nb.wav\nsecond\n"
    pairs, _ = cf.parse_manifest(text)
    assert pairs == [("a.wav", "line one line two line three"), ("b.wav", "second")]


def test_parse_manifest_reports_orphan_text_and_empty_transcript():
    text = "stray text\na.wav\nb.wav\nreal transcript\n"
    pairs, warnings = cf.parse_manifest(text)
    assert pairs == [("b.wav", "real transcript")]
    assert any("before any filename" in w for w in warnings)
    assert any("no transcript for 'a.wav'" in w for w in warnings)


def test_parse_manifest_dedupes_filenames():
    text = "a.wav\nfirst\na.wav\nsecond\n"
    pairs, warnings = cf.parse_manifest(text)
    assert pairs == [("a.wav", "first")]
    assert any("duplicate entry" in w for w in warnings)


def test_resolve_wav_is_case_insensitive(tmp_path):
    _touch(tmp_path / "Clip.WAV")
    index = cf._audio_index(tmp_path)
    resolved = cf.resolve_wav("clip.wav", tmp_path, index)
    assert resolved is not None and resolved.is_file()
    assert resolved.name.lower() == "clip.wav"
    assert cf.resolve_wav("missing.wav", tmp_path, index) is None


def test_safe_stem_sanitizes_and_falls_back():
    assert cf.safe_stem("my clip (1).wav") == "my_clip_1"
    assert cf.safe_stem("sub/dir/clip.wav") == "clip"
    assert cf.safe_stem("!!!.wav") == "clip"


def test_build_writes_pairs_and_reports_missing(tmp_path):
    wav_dir = tmp_path / "wavs"
    _touch(wav_dir / "a.wav", b"AAAA")
    _touch(wav_dir / "b.WAV", b"BBBB")
    manifest = tmp_path / "transcripts.txt"
    manifest.write_text(
        "a.wav\nxin chào\nb.WAV\ncảm ơn\nghost.wav\nkhông có\n", encoding="utf-8"
    )

    out_dir = tmp_path / "voice_clone"
    stats, warnings = cf.build(wav_dir, manifest, out_dir)

    assert stats == {"written": 2, "skipped": 0, "missing": 1}
    assert any("ghost.wav" in w for w in warnings)
    assert (out_dir / "a.wav").read_bytes() == b"AAAA"
    assert (out_dir / "a.txt").read_text(encoding="utf-8").strip() == "xin chào"
    assert (out_dir / "b.txt").read_text(encoding="utf-8").strip() == "cảm ơn"


def test_build_resumes_and_overwrite(tmp_path):
    wav_dir = tmp_path / "wavs"
    _touch(wav_dir / "a.wav")
    _touch(wav_dir / "b.wav")
    manifest = tmp_path / "m.txt"
    manifest.write_text("a.wav\none\nb.wav\ntwo\n", encoding="utf-8")
    out_dir = tmp_path / "pool"

    stats, _ = cf.build(wav_dir, manifest, out_dir)
    assert stats["written"] == 2

    stats, _ = cf.build(wav_dir, manifest, out_dir)
    assert stats == {"written": 0, "skipped": 2, "missing": 0}

    stats, _ = cf.build(wav_dir, manifest, out_dir, overwrite=True)
    assert stats["written"] == 2


def test_main_returns_nonzero_when_nothing_written(tmp_path, capsys):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    manifest = tmp_path / "m.txt"
    manifest.write_text("ghost.wav\nhi\n", encoding="utf-8")
    rc = cf.main(
        [
            "--wav-dir",
            str(wav_dir),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "pool"),
        ]
    )
    assert rc == 1


def test_main_auto_detects_single_manifest(tmp_path):
    wav_dir = tmp_path / "wavs"
    _touch(wav_dir / "a.wav")
    (wav_dir / "transcripts.txt").write_text("a.wav\nchào\n", encoding="utf-8")
    rc = cf.main(["--wav-dir", str(wav_dir), "--out-dir", str(tmp_path / "pool")])
    assert rc == 0
    assert (tmp_path / "pool" / "a.txt").exists()
