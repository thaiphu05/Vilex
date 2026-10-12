"""Read-only statistics for Stage 5.1a artifacts (Python 3.10+, standard library).

Run from the repo, with no TTS environment, models, voice pool or API keys:
    python3 tools/stage5a_stats.py --input /path/to/vi_stage5_work --output-dir /path/to/stats

Input: a work directory, one manifest.json, or omnivoice_units.jsonl. A directory
uses manifests as the corpus source of truth; its _batch JSONL is summarized
separately because resume may omit already rendered units. With only JSONL,
counts describe the queue ONLY, never the complete corpus.

Outputs: summary.json, records.csv (one row per variant), summary.md (Vietnamese).
Output directory must be new to prevent accidental overwrite. Invalid records
are excluded and listed in summary.json; exit 1 means incomplete statistics,
exit 2 means an input/output error, exit 0 means success. No input is modified.

Host utterances are the merged hosts in the manifest, not original dialogue
turns. Text lengths count whitespace-separated tokens, not Vietnamese lexical
words or model tokens. All volume totals include variants; distinct dialogues
are counted separately by rel_path. Audio duration/quality and drop rates cannot
be inferred from prep artifacts. Voice paths are counted as stored, not opened.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts_render.stage5_schema import read_manifest  # noqa: E402

METRICS = (
    "host_utterances",
    "interrupt_hosts",
    "backchannels",
    "pause_units",
    "text_units",
    "tts_units",
    "host_tokens",
    "bc_tokens",
    "text_characters",
    "user_tts_units",
    "assistant_tts_units",
)


def distribution(values):
    """Linear-interpolated percentiles; empty population has no mean/percentiles."""
    values = sorted(values)
    if not values:
        return dict.fromkeys(("min", "mean", "p50", "p95", "max"))

    def percentile(q):
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return {
        "min": values[0],
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": values[-1],
    }


def dialogue_parts(rel_path):
    parts = PurePosixPath(rel_path).parts
    if (
        len(parts) != 3
        or not parts[0].startswith("text_dialogue_")
        or not parts[0][len("text_dialogue_") :]
        or any(part in (".", "..", "/") for part in parts)
    ):
        raise ValueError("rel_path must be text_dialogue_<dataset>/<split>/<dialogue>")
    return parts[0].removeprefix("text_dialogue_"), parts[1]


def manifest_record(path):
    m = read_manifest(path)
    dataset, split = dialogue_parts(m.rel_path)
    if m.variant_idx < 0:
        raise ValueError("negative variant_idx")
    units = [(un, u.speaker_idx) for u in m.utterances for un in u.units]
    ids = [un.unit_id for un, _ in units] + [bc.unit_id for bc in m.backchannels]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate unit_id (including backchannels)")
    texts = [(un.text, speaker) for un, speaker in units if un.kind == "text"]
    bc_texts = [(bc.text, bc.speaker_idx) for bc in m.backchannels]
    for content, _ in texts + bc_texts:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty or non-string TTS text")
    for bc in m.backchannels:
        if bc.speaker_idx == m.utterances[bc.host_idx].speaker_idx:
            raise ValueError("backchannel speaker must be the listener of its host")
    record = {
        "source": str(path),
        "rel_path": m.rel_path,
        "dataset": dataset,
        "split": split,
        "variant_idx": m.variant_idx,
        "host_utterances": len(m.utterances),
        "interrupt_hosts": sum(u.uttr_type == "interrupt" for u in m.utterances),
        "backchannels": len(bc_texts),
        "pause_units": sum(un.kind == "pause" for un, _ in units),
        "text_units": len(texts),
        "tts_units": len(texts) + len(bc_texts),
        "host_tokens": sum(len(t.split()) for t, _ in texts),
        "bc_tokens": sum(len(t.split()) for t, _ in bc_texts),
        "text_characters": sum(len(t) for t, _ in texts + bc_texts),
        "user_tts_units": sum(s == 0 for _, s in texts + bc_texts),
        "assistant_tts_units": sum(s == 1 for _, s in texts + bc_texts),
    }
    voices = [(role, pick.wav) for role, pick in m.voice_picks.items()]
    return record, voices, [len(t.split()) for t, _ in texts + bc_texts]


def aggregate(records, metrics):
    totals = {key: sum(r[key] for r in records) for key in metrics}
    return {
        "dialogues": len({r["rel_path"] for r in records}),
        "variants": len(records),
        "totals": totals,
        "per_variant": {key: distribution([r[key] for r in records]) for key in metrics},
    }


def grouped(records, metrics):
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["split"])].append(record)
    return [
        {"dataset": ds, "split": split, **aggregate(rows, metrics)}
        for (ds, split), rows in sorted(groups.items())
    ]


def scan_queue(path):
    """Stream JSONL; hold identifiers and compact counts, not all source text."""
    records, seen, lengths = {}, set(), []
    voices, languages = Counter(), Counter()
    errors = []
    scanned = valid = 0
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            scanned += 1
            try:
                row = json.loads(line)
                row_id, content = row["id"], row["text"]
                if (
                    not isinstance(row_id, str)
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    raise ValueError("id/text must be nonempty strings")
                flat, variant, unit = row_id.rsplit("__", 2)
                if not re.fullmatch(r"var\d+", variant):
                    raise ValueError("invalid variant in JSONL id")
                if not re.fullmatch(r"(?:u_\d+_\d+|bc_\d+)", unit):
                    raise ValueError("invalid unit in JSONL id")
                # The emitter joins rel_path components with '__'. Split twice
                # so dialogue names containing '__' remain intact.
                rel_path = "/".join(flat.split("__", 2))
                dataset, split = dialogue_parts(rel_path)
                for field in ("ref_audio", "language_id"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"missing/invalid {field}")
                if row_id in seen:
                    raise ValueError("duplicate JSONL id")
                seen.add(row_id)
                key = (rel_path, int(variant[3:]))
                record = records.setdefault(
                    key,
                    {
                        "source": str(path),
                        "rel_path": rel_path,
                        "dataset": dataset,
                        "split": split,
                        "variant_idx": key[1],
                        "text_units": 0,
                        "backchannels": 0,
                        "tts_units": 0,
                        "host_tokens": 0,
                        "bc_tokens": 0,
                        "text_characters": 0,
                    },
                )
                is_bc = unit.startswith("bc_")
                count = len(content.split())
                record["backchannels" if is_bc else "text_units"] += 1
                record["bc_tokens" if is_bc else "host_tokens"] += count
                record["tts_units"] += 1
                record["text_characters"] += len(content)
                lengths.append(count)
                voices[row["ref_audio"]] += 1
                languages[row["language_id"]] += 1
                valid += 1
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                errors.append({"source": str(path), "line": lineno, "error": str(exc)})
    return (
        {
            "scope": "queued_units_only",
            "scanned_rows": scanned,
            "valid_rows": valid,
            "invalid_rows": len(errors),
            "languages": dict(sorted(languages.items())),
            "voice_unit_counts": dict(sorted(voices.items())),
            "tokens_per_tts_unit": distribution(lengths),
        },
        list(records.values()),
        errors,
    )


def analyze(source):
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    manifests = sorted(source.rglob("manifest.json")) if source.is_dir() else []
    if source.is_file() and source.name == "manifest.json":
        manifests = [source]
    queue = source / "_batch/omnivoice_units.jsonl" if source.is_dir() else source
    if not (queue.is_file() and queue.suffix == ".jsonl"):
        queue = None
    if not manifests and queue is None:
        raise ValueError("No manifest.json or _batch/omnivoice_units.jsonl found")

    records, errors, seen, lengths = [], [], set(), []
    voices = {role: Counter() for role in ("user", "assistant")}
    for path in manifests:
        try:
            record, picks, unit_lengths = manifest_record(path)
            key = (record["rel_path"], record["variant_idx"])
            if key in seen:
                raise ValueError("duplicate (rel_path, variant_idx); record excluded")
            seen.add(key)
            records.append(record)
            lengths.extend(unit_lengths)
            for role, voice in picks:
                voices[role][voice] += 1
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            errors.append({"source": str(path), "error": str(exc)})

    queue_stats = None
    if queue is not None:
        queue_stats, queue_records, queue_errors = scan_queue(queue)
        errors.extend(queue_errors)
        queue_metrics = (
            "text_units",
            "backchannels",
            "tts_units",
            "host_tokens",
            "bc_tokens",
            "text_characters",
        )
        queue_stats["overall"] = aggregate(queue_records, queue_metrics)
        queue_stats["by_dataset_split"] = grouped(queue_records, queue_metrics)
        if not manifests:
            records = queue_records
    metrics = METRICS if manifests else queue_metrics
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "scope": "manifest_corpus" if manifests else "queued_units_only",
        "complete": not errors,
        "manifest_files_found": len(manifests),
        "valid_manifests": len(records) if manifests else 0,
        "overall": aggregate(records, metrics),
        "by_dataset_split": grouped(records, metrics),
        "queue": queue_stats,
        "errors": errors,
        "definitions": {
            "dialogues": "Distinct rel_path among valid records, NOT dialogue_id alone.",
            "totals": "All valid variants; not deduplicated text content.",
            "host_utterances": "Merged manifest hosts, NOT source dialogue turns.",
            "interrupt_hosts": "Hosts with uttr_type=interrupt, NOT slot decisions.",
            "tokens": "Whitespace-separated tokens of TTS text, including any retained tags; "
            "NOT Vietnamese lexical words or model tokens. Reference transcripts excluded.",
            "tts_units": "Host text units + backchannels; pauses excluded.",
            "queue": "Only queued rows; resume can omit rendered units. Not corpus size.",
            "complete": "No rejected records; does NOT prove all corpus files were supplied.",
            "unavailable": "Audio hours, audio quality, overlap timing, source drop rate, "
            "and original slot-label distribution require other stages/evaluation.",
        },
    }
    if manifests:
        summary["voice_variant_counts"] = {
            role: dict(sorted(counts.items())) for role, counts in voices.items()
        }
        summary["tokens_per_tts_unit"] = distribution(lengths)
    return summary, sorted(records, key=lambda r: (r["rel_path"], r["variant_idx"]))


def markdown(summary):
    overall = summary["overall"]
    totals = overall["totals"]
    scope = (
        "Corpus theo manifest"
        if summary["scope"] == "manifest_corpus"
        else "Chỉ hàng đợi TTS trong JSONL, không đại diện toàn corpus"
    )
    lines = [
        "# Thống kê dữ liệu Stage 5.1a",
        "",
        f"Thời điểm UTC: {summary['generated_at_utc']}",
        f"Phạm vi: **{scope}**.",
        f"Số bản ghi bị loại do lỗi: **{len(summary['errors'])}** (xem summary.json).",
        "",
        f"- Hội thoại phân biệt: **{overall['dialogues']:,}**",
        f"- Biến thể: **{overall['variants']:,}**",
        f"- Đơn vị cần TTS: **{totals['tts_units']:,}**",
        f"- Backchannel: **{totals['backchannels']:,}**",
        f"- Token cách trắng của lời thoại chính: **{totals['host_tokens']:,}**",
        f"- Token cách trắng của backchannel: **{totals['bc_tokens']:,}**",
        "",
        "## Theo dataset và split",
        "",
        "| Dataset | Split | Hội thoại | Biến thể | Đơn vị TTS | Backchannel |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in summary["by_dataset_split"]:
        ds = group["dataset"].replace("|", "\\|").replace("\n", " ")
        split = group["split"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {ds} | {split} | {group['dialogues']} | {group['variants']} "
            f"| {group['totals']['tts_units']} | {group['totals']['backchannels']} |"
        )
    lines.extend(
        [
            "",
            "## Cách đọc số liệu",
            "",
            "Tổng khối lượng tính trên tất cả biến thể. Hội thoại phân biệt được đếm bằng rel_path. "
            "Lượt host trong manifest đã gộp quanh backchannel, không phải lượt thoại nguồn.",
            "",
            "Token được đếm bằng khoảng trắng trên văn bản TTS, không phải từ vựng tiếng Việt "
            "hoặc token của mô hình. Không tính ref_text. Pause không phải đơn vị TTS.",
            "",
            "Stage 5.1a chưa chứng minh số giờ audio, chất lượng giọng, thời gian chồng lấn, "
            "tỷ lệ loại dữ liệu đầu vào hoặc phân bố nhãn tại các slot gốc. "
            "Không có lỗi đọc cũng không chứng minh bộ dữ liệu đã đầy đủ.",
        ]
    )
    if summary["queue"] is not None:
        lines.extend(
            [
                "",
                f"JSONL có {summary['queue']['valid_rows']:,} dòng hợp lệ. "
                "Đây là hàng đợi tại thời điểm xuất; resume có thể bỏ qua đơn vị đã render. "
                "Không lấy số dòng này làm kích thước corpus khi có manifest.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New directory for statistics"
    )
    args = parser.parse_args(argv)
    try:
        summary, records = analyze(args.input)
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
        fields = (
            list(records[0])
            if records
            else ["source", "rel_path", "dataset", "split", "variant_idx"]
        )
        with (output / "records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f"Statistics: {output} ({summary['scope']})")
        print(f"Variants: {len(records)}; errors: {len(summary['errors'])}")
        return 0 if summary["complete"] else 1
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
