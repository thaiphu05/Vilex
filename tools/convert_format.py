"""Build an OmniVoice voice-clone pool from a folder of wavs + a transcript manifest.

Input
-----
1. A folder holding the source ``.wav`` files (``--wav-dir``).
2. A single text manifest listing, for each clip, its filename followed by its
   transcript. The transcript may wrap across several lines and is accumulated
   until the next filename appears::

       ten_file.wav
       transcript
       ten_file_2.wav
       transcript

Output
------
``--out-dir`` (default ``voice_clone/``) filled with ``<stem>.wav`` + ``<stem>.txt``
pairs, where the sidecar holds the exact transcript of its clip. This is the
layout ``tts_render/convert_spoken.py`` expects: every ``*.wav`` needs a matching
``*.txt`` and the pool needs at least two pairs.

Source wavs are copied byte-for-byte; Stage 5 downmixes to mono, resamples to
24 kHz and clamps to ``MAX_PROMPT_SECS`` when loading the pool.

Usage
-----
    python tools/convert_format.py \
        --wav-dir /path/to/wavs \
        --manifest /path/to/transcripts.txt \
        --out-dir voice_clone
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a")
_AUDIO_RE = re.compile(r"\.(" + "|".join(e.lstrip(".") for e in AUDIO_EXTS) + r")$", re.IGNORECASE)
_ILLEGAL_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_audio_filename(line: str) -> bool:
    """True if a manifest line names an audio file."""
    return bool(_AUDIO_RE.search(line.strip()))


def parse_manifest(text: str):
    """Parse a filename/transcript manifest into (filename, transcript) pairs.

    A line naming an audio file starts a new entry; every other non-empty line is
    accumulated into that entry's transcript (so transcripts may wrap). Lines
    before the first filename, and entries whose transcript is empty, are
    reported as warnings and dropped.

    Returns (pairs, warnings) where pairs is a list of (filename, transcript).
    """
    pairs = []
    warnings = []
    filename = None
    buffer = []

    def flush():
        nonlocal filename, buffer
        if filename is None:
            return
        transcript = " ".join(" ".join(buffer).split())
        if transcript:
            pairs.append((filename, transcript))
        else:
            warnings.append(f"no transcript for {filename!r}; skipped")
        filename = None
        buffer = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if is_audio_filename(line):
            flush()
            filename = line
        elif filename is None:
            warnings.append(f"text before any filename ignored: {line[:60]!r}")
        else:
            buffer.append(line)

    flush()

    # De-duplicate filenames, keeping the first occurrence.
    seen = set()
    deduped = []
    for name, transcript in pairs:
        key = name.lower()
        if key in seen:
            warnings.append(f"duplicate entry for {name!r}; kept first")
            continue
        seen.add(key)
        deduped.append((name, transcript))
    return deduped, warnings


def _audio_index(wav_dir: Path):
    """Map lowercased relative path and lowercased basename -> actual Path."""
    index = {}
    if not wav_dir.is_dir():
        return index
    for path in sorted(wav_dir.rglob("*")):
        if path.is_file() and is_audio_filename(path.name):
            rel = str(path.relative_to(wav_dir)).lower()
            index.setdefault(rel, path)
            index.setdefault(path.name.lower(), path)
    return index


def resolve_wav(name: str, wav_dir: Path, index):
    """Resolve a manifest filename to an existing wav under wav_dir, or None."""
    direct = wav_dir / name
    if direct.is_file():
        return direct
    return index.get(name.lower())


def safe_stem(name: str) -> str:
    """Sanitize a manifest filename into a filesystem-safe stem."""
    stem = Path(name).stem
    stem = _ILLEGAL_STEM_RE.sub("_", stem).strip("._-")
    return stem or "clip"


def build(wav_dir, manifest, out_dir, overwrite=False, limit=None, dry_run=False):
    """Copy resolved wavs into out_dir with sidecar transcripts. Returns stats."""
    text = manifest.read_text(encoding="utf-8-sig")
    pairs, warnings = parse_manifest(text)

    if limit is not None:
        pairs = pairs[:limit]

    index = _audio_index(wav_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped": 0, "missing": 0}
    used_stems = set()

    for name, transcript in pairs:
        wav = resolve_wav(name, wav_dir, index)
        if wav is None:
            warnings.append(f"wav not found: {name!r}")
            stats["missing"] += 1
            continue

        stem = safe_stem(name)
        candidate = stem
        n = 2
        while candidate.lower() in used_stems:
            candidate = f"{stem}_{n}"
            n += 1
        used_stems.add(candidate.lower())

        wav_out = out_dir / f"{candidate}.wav"
        txt_out = out_dir / f"{candidate}.txt"
        if not overwrite and wav_out.exists() and txt_out.exists():
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"[dry-run] {name} -> {wav_out.name} + {txt_out.name}")
            stats["written"] += 1
            continue

        shutil.copy2(wav, wav_out)
        txt_out.write_text(transcript + "\n", encoding="utf-8")
        stats["written"] += 1

    return stats, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--wav-dir", type=Path, default=None, help="folder holding the source wavs")
    parser.add_argument("--manifest", type=Path, default=None, help="filename/transcript text file")
    parser.add_argument("--out-dir", type=Path, default=Path("voice_clone"), help="output pool dir")
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing pairs")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N entries")
    parser.add_argument("--dry-run", action="store_true", help="print planned pairs, write nothing")
    args = parser.parse_args(argv)

    manifest = args.manifest
    wav_dir = args.wav_dir
    if manifest is None:
        search_dir = wav_dir or Path(".")
        txts = sorted(p for p in search_dir.glob("*.txt") if p.is_file())
        if len(txts) != 1:
            parser.error(f"--manifest is required (found {len(txts)} .txt files in {search_dir})")
        manifest = txts[0]
        print(f"Using manifest: {manifest}")
    if wav_dir is None:
        wav_dir = manifest.parent

    if not manifest.is_file():
        parser.error(f"manifest not found: {manifest}")
    if not wav_dir.is_dir():
        parser.error(f"wav dir not found: {wav_dir}")

    stats, warnings = build(
        wav_dir,
        manifest,
        args.out_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    print(
        f"written={stats['written']} skipped={stats['skipped']} "
        f"missing={stats['missing']} -> {args.out_dir}"
    )
    if stats["written"] == 0:
        print("ERROR: no pairs written.", file=sys.stderr)
        return 1
    if stats["written"] < 2:
        print(
            "WARNING: voice pool has fewer than 2 pairs; Stage 5 _load_voice_pool needs >=2.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
