"""Read-only end-to-end statistics for the Vilex pipeline (standard library only).

Run it where the data lives (no models, GPU or API keys needed):

    python3 tools/pipeline_stats.py --data-root data --output-dir stats_out

It reads, when present, the directories written by each stage and writes three
files into a NEW output directory (an existing one is refused, nothing is
overwritten, no input is modified):

    summary.json         every metric with raw values
    report_numbers.json  {KEY: formatted string} for tools/fill_report.py
    report_numbers.md    the same keys as a table, to fill the report by hand

Stage directories default to the config.yaml names under --data-root and can be
overridden one by one (--tt-bc, --audio, ...). A missing directory is skipped and
its keys are simply absent. Counts are exact; LLM-call totals are ESTIMATES
derived from stage 4/4b outputs (1 call per generated turn, per user turn
boundary pass, per slot, per backchannel, plus 1 for stage 1) and ignore
retries. Text lengths count whitespace-separated tokens, not Vietnamese
lexical words. Audio duration is read from WAV headers only.

Exit codes: 0 ok, 1 ok but some files were unreadable (listed in summary.json),
2 input/output error.
"""

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.stage5a_stats import analyze, distribution  # noqa: E402

DATASETS = (
    "interviewer",
    "multiwoz",
    "negotiator",
    "socraticlm",
    "persuader",
    "soda",
    "deepdialogue",
)
STAGES = (
    ("S1", "results_vi"),
    ("S15", "results_vi_xt"),
    ("S175", "results_vi_dis"),
    ("S4", "vi_tt"),
    ("S4B", "vi_tt_bc"),
)


# ----------------------------------------------------------------- formatting
def fmt_int(n):
    return f"{int(n):,}".replace(",", ".")


def fmt_num(x, digits=1):
    s = f"{x:,.{digits}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


# ------------------------------------------------------------------- loading
def dialogue_files(root):
    """Yield (dataset, split, path) for <root>/text_dialogue_<ds>/<split>/*.json."""
    for p in sorted(Path(root).glob("text_dialogue_*/*/*.json")):
        yield p.parent.parent.name[len("text_dialogue_") :], p.parent.name, p


def load_stage(root, errors):
    """Return list of (dataset, split, data) for readable dialogue JSONs."""
    out = []
    for ds, split, p in dialogue_files(root):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data.get("history"), list):
                raise ValueError("no history list")
            out.append((ds, split, data))
        except (OSError, ValueError, AttributeError) as exc:
            errors.append({"source": str(p), "error": str(exc)})
    return out


def tokens(text):
    return len(str(text).split())


def basic(items):
    turns = [len(d["history"]) for _, _, d in items]
    words = sum(tokens(t.get("content", "")) for _, _, d in items for t in d["history"])
    return {
        "dialogues": len(items),
        "turns": sum(turns),
        "words": words,
        "turns_per_dialogue": distribution(turns),
    }


# ------------------------------------------------------------ stage analyses
def parsed_source_counts(root):
    counts = {}
    for ds_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        counts[ds_dir.name] = sum(1 for _ in ds_dir.glob("*/*.json"))
    return counts


def cross_turn_stats(items):
    turns = sum(
        1
        for _, _, d in items
        for t in d["history"]
        if isinstance(t.get("meta"), dict) and t["meta"].get("cross_turn_slot")
    )
    dialogues = sum(1 for _, _, d in items if (d.get("meta") or {}).get("cross_turn_slots_applied"))
    return {"turns": turns, "dialogues": dialogues}


def disfluency_stats(items):
    types, pauses, turns = Counter(), 0, 0
    for _, _, d in items:
        pauses += int((d.get("meta") or {}).get("pauses") or 0)
        for t in d["history"]:
            meta = t.get("meta") if isinstance(t.get("meta"), dict) else {}
            inj = meta.get("disfluency_injections") or []
            turns += 1 if inj else 0
            types.update(str(i.get("type", "?")) for i in inj if isinstance(i, dict))
    n = max(len(items), 1)
    return {
        "injections": sum(types.values()),
        "by_type": dict(types),
        "turns_with_injection": turns,
        "pauses": pauses,
        "injections_per_dialogue": sum(types.values()) / n,
    }


def slot_stats(items):
    decisions, by_role = Counter(), Counter()
    bc, user_turns, ft_tokens, modes, models = Counter(), 0, 0, Counter(), Counter()
    bc_total = bc_filled = bc_words = bc_tagged = 0
    for _, _, d in items:
        meta = d.get("meta") or {}
        modes[str(meta.get("tt_mode", "unknown"))] += 1
        models[str(meta.get("tt_model", "unknown"))] += 1
        for t in d["history"]:
            user_turns += 1 if t.get("role") == "user" else 0
            ft_tokens += str(t.get("content", "")).count("[TAKE_FLOOR]")
            for seg in t.get("history") or []:
                if not isinstance(seg, dict) or "decision" not in seg:
                    continue
                decisions[seg["decision"]] += 1
                by_role[(t.get("role"), seg["decision"])] += 1
                if seg["decision"] == "backchannel":
                    bc_total += 1
                    text = str(seg.get("content") or "").strip().lower()
                    if text:
                        bc_filled += 1
                        bc[text] += 1
                        bc_words += tokens(re.sub(r"\[[^\]]*\]", "", text))
                        bc_tagged += 1 if "[" in text else 0
    slots = sum(decisions.values())
    top = bc.most_common(10)
    return {
        "slots": slots,
        "decisions": dict(decisions),
        "by_role": {f"{r}:{k}": v for (r, k), v in sorted(by_role.items(), key=str)},
        "user_turns": user_turns,
        "take_floor_tokens": ft_tokens,
        "tt_modes": dict(modes),
        "tt_models": dict(models),
        "backchannel": {
            "slots": bc_total,
            "filled": bc_filled,
            "distinct": len(bc),
            "top": top,
            "top1_share": pct(top[0][1], bc_filled) if top else 0.0,
            "mean_words": bc_words / bc_filled if bc_filled else 0.0,
            "with_tag": bc_tagged,
        },
    }


def llm_calls(items, slots):
    """Estimated calls: stage 1 + writer + boundary + predictor + backchannel."""
    n = len(items)
    turns = sum(len(d["history"]) for _, _, d in items)
    parts = {
        "stage1": n,
        "writer": turns,
        "boundary": slots["user_turns"],
        "predictor": slots["slots"],
        "backchannel": slots["backchannel"]["slots"],
    }
    total = sum(parts.values())
    return {"parts": parts, "total": total, "per_dialogue": total / n if n else 0.0}


def stage5_manifest_extras(work_root):
    failed = rendered = units_expected = variants = 0
    for p in sorted(Path(work_root).rglob("manifest.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            variants += 1
            failed += len(m.get("failed_units", []))
            rendered += len(m.get("unit_audio", {}))
            units_expected += sum(
                1
                for u in m.get("utterances", [])
                for x in u.get("units", [])
                if x.get("kind") == "text"
            ) + len(m.get("backchannels", []))
        except (OSError, ValueError, AttributeError):
            continue
    return {
        "variants": variants,
        "failed_units": failed,
        "units_with_audio": rendered,
        "units_expected": units_expected,
    }


def wav_info(path):
    """(seconds, sample_rate, channels) from a RIFF/WAVE header, any PCM/float format."""
    with open(path, "rb") as fh:
        head = fh.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        fmt = None
        while True:
            chunk = fh.read(8)
            if len(chunk) < 8:
                raise ValueError("no data chunk")
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                body = fh.read(size + (size & 1))
                channels, rate, _, block = struct.unpack("<HIIH", body[2:14])
                fmt = (channels, rate, block)
            elif cid == b"data":
                if fmt is None or fmt[2] == 0 or fmt[1] == 0:
                    raise ValueError("data before/without valid fmt")
                return size / fmt[2] / fmt[1], fmt[1], fmt[0]
            else:
                fh.seek(size + (size & 1), 1)


def audio_stats(audio_root, errors):
    seconds, per_ds = [], defaultdict(float)
    rates, chans = Counter(), Counter()
    for p in sorted(Path(audio_root).rglob("dialogue.wav")):
        try:
            sec, rate, ch = wav_info(p)
        except (OSError, ValueError, struct.error) as exc:
            errors.append({"source": str(p), "error": str(exc)})
            continue
        ds = next(
            (x[len("text_dialogue_") :] for x in p.parts if x.startswith("text_dialogue_")),
            "unknown",
        )
        seconds.append(sec)
        per_ds[ds] += sec
        rates[rate] += 1
        chans[ch] += 1
    return {
        "files": len(seconds),
        "hours": sum(seconds) / 3600,
        "minutes_per_dialogue": distribution([s / 60 for s in seconds]),
        "hours_by_dataset": {k: v / 3600 for k, v in sorted(per_ds.items())},
        "sample_rates": dict(rates),
        "channels": dict(chans),
    }


# ----------------------------------------------------------------- assembling
def collect(args):
    root = args.data_root
    dirs = {
        "parsed": args.parsed_source or root / "parsed_source",
        "work": args.stage5_work or root / "vi_stage5_work",
        "audio": args.audio or root / "vi_audio",
    }
    for key, default in STAGES:
        dirs[key] = getattr(args, key.lower()) or root / default
    errors, stages, numbers = [], {}, {}
    summary = {"dirs": {k: str(v) for k, v in dirs.items()}, "stages": stages}

    if dirs["parsed"].is_dir():
        counts = parsed_source_counts(dirs["parsed"])
        summary["source_counts"] = counts
        numbers["SRC_TOTAL"] = fmt_int(sum(counts.values()))
        numbers["SRC_NONSODA"] = fmt_int(sum(v for k, v in counts.items() if k != "soda"))
        for ds in DATASETS:
            if ds in counts:
                numbers[f"DS_{ds}_SRC"] = fmt_int(counts[ds])

    loaded = {}
    for key, _ in STAGES:
        if not dirs[key].is_dir():
            continue
        loaded[key] = load_stage(dirs[key], errors)
        stages[key] = basic(loaded[key])
        numbers[f"N_{key}"] = fmt_int(stages[key]["dialogues"])

    if "S15" in loaded:
        stages["S15"]["cross_turn"] = cross_turn_stats(loaded["S15"])
        numbers["XT_TURNS"] = fmt_int(stages["S15"]["cross_turn"]["turns"])
        numbers["XT_DIALOGUES"] = fmt_int(stages["S15"]["cross_turn"]["dialogues"])
    if "S175" in loaded:
        dis = disfluency_stats(loaded["S175"])
        stages["S175"]["disfluency"] = dis
        numbers["DIS_INJECTIONS"] = fmt_int(dis["injections"])
        numbers["DIS_PER_DIALOGUE"] = fmt_num(dis["injections_per_dialogue"])
        numbers["PAUSES"] = fmt_int(dis["pauses"])
        for t in ("fp", "dm", "edit", "rep"):
            numbers[f"DIS_{t.upper()}"] = fmt_int(dis["by_type"].get(t, 0))
    if "S4B" in loaded:
        b = stages["S4B"]
        numbers["TURNS_S4B"] = fmt_int(b["turns"])
        numbers["WORDS_S4B"] = fmt_int(b["words"])
        numbers["TURNS_MEAN"] = fmt_num(b["turns_per_dialogue"]["mean"] or 0)
        numbers["TURNS_P95"] = fmt_num(b["turns_per_dialogue"]["p95"] or 0)
        slots = slot_stats(loaded["S4B"])
        b["slots"] = slots
        total = slots["slots"]
        numbers["SLOTS"] = fmt_int(total)
        for name, key in (("silence", "SILENCE"), ("backchannel", "BC"), ("floor_taking", "FT")):
            n = slots["decisions"].get(name, 0)
            numbers[f"SLOT_{key}"] = fmt_int(n)
            numbers[f"SLOT_{key}_PCT"] = fmt_num(pct(n, total))
        numbers["TT_MODE"] = ", ".join(sorted(slots["tt_modes"]))
        numbers["TT_MODEL"] = ", ".join(sorted(slots["tt_models"]))
        bc = slots["backchannel"]
        numbers["BC_FILLED_PCT"] = fmt_num(pct(bc["filled"], bc["slots"]))
        numbers["BC_DISTINCT"] = fmt_int(bc["distinct"])
        numbers["BC_TOP1"] = bc["top"][0][0] if bc["top"] else "-"
        numbers["BC_TOP1_SHARE"] = fmt_num(bc["top1_share"])
        numbers["BC_TOP5"] = ", ".join(f"{t} ({fmt_int(c)})" for t, c in bc["top"][:5]) or "-"
        numbers["BC_MEAN_WORDS"] = fmt_num(bc["mean_words"])
        calls = llm_calls(loaded["S4B"], slots)
        summary["llm_calls_estimate"] = calls
        numbers["CALLS_PER_DIALOGUE"] = fmt_int(round(calls["per_dialogue"]))
        for name, value in calls["parts"].items():
            numbers[f"CALLS_{name.upper()}"] = fmt_int(round(value / max(b["dialogues"], 1)))
        summary["projection"] = {}
        for n in args.project:
            total_calls = calls["per_dialogue"] * n
            days = total_calls * args.interval / 86400
            summary["projection"][str(n)] = {"calls": total_calls, "days_sequential": days}
            numbers[f"PROJ_{n}_CALLS"] = fmt_int(round(total_calls))
            numbers[f"PROJ_{n}_DAYS"] = fmt_num(days)
        for ds in DATASETS:
            n = sum(1 for d, _, _ in loaded["S4B"] if d == ds)
            if n:
                numbers[f"DS_{ds}_N4B"] = fmt_int(n)
    if "S1" in loaded and "S4B" in loaded:
        numbers["YIELD_PCT"] = fmt_num(pct(len(loaded["S4B"]), len(loaded["S1"])))

    if dirs["work"].is_dir():
        try:
            s5, _ = analyze(dirs["work"])
        except ValueError as exc:
            errors.append({"source": str(dirs["work"]), "error": str(exc)})
        else:
            totals = s5["overall"]["totals"]
            summary["stage5a"] = s5
            errors.extend(s5["errors"])
            numbers["N_S5A"] = fmt_int(s5["overall"]["dialogues"])
            numbers["S5A_VARIANTS"] = fmt_int(s5["overall"]["variants"])
            numbers["S5A_ERRORS"] = fmt_int(len(s5["errors"]))
            numbers["TTS_UNITS"] = fmt_int(totals.get("tts_units", 0))
            numbers["TTS_BC"] = fmt_int(totals.get("backchannels", 0))
            numbers["PAUSE_UNITS"] = fmt_int(totals.get("pause_units", 0))
            numbers["INTERRUPT_HOSTS"] = fmt_int(totals.get("interrupt_hosts", 0))
            numbers["HOST_UTTERANCES"] = fmt_int(totals.get("host_utterances", 0))
            numbers["INTERRUPT_PCT"] = fmt_num(
                pct(totals.get("interrupt_hosts", 0), totals.get("host_utterances", 0))
            )
            numbers["HOST_TOKENS"] = fmt_int(totals.get("host_tokens", 0))
            per_ds = Counter()
            for g in s5["by_dataset_split"]:
                per_ds[g["dataset"]] += g["totals"]["tts_units"]
            for ds, n in per_ds.items():
                numbers[f"DS_{ds}_TTS"] = fmt_int(n)
        extra = stage5_manifest_extras(dirs["work"])
        summary["stage5_manifest_extras"] = extra
        numbers["TTS_FAILED"] = fmt_int(extra["failed_units"])
        numbers["TTS_RENDERED_PCT"] = fmt_num(
            pct(extra["units_with_audio"], extra["units_expected"])
        )

    if dirs["audio"].is_dir():
        audio = audio_stats(dirs["audio"], errors)
        summary["audio"] = audio
        numbers["AUDIO_FILES"] = fmt_int(audio["files"])
        numbers["AUDIO_HOURS"] = fmt_num(audio["hours"])
        mpd = audio["minutes_per_dialogue"]
        numbers["AUDIO_MEAN_MIN"] = fmt_num(mpd["mean"] or 0)
        numbers["AUDIO_P95_MIN"] = fmt_num(mpd["p95"] or 0)
        numbers["AUDIO_SR"] = ", ".join(fmt_int(r) for r in sorted(audio["sample_rates"]))
        numbers["AUDIO_CH"] = ", ".join(str(c) for c in sorted(audio["channels"]))
        for ds, hours in audio["hours_by_dataset"].items():
            numbers[f"DS_{ds}_HOURS"] = fmt_num(hours)
        variants = summary.get("stage5_manifest_extras", {}).get("variants")
        if variants:
            numbers["AUDIO_COVERAGE_PCT"] = fmt_num(pct(audio["files"], variants))

    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["errors"] = errors
    summary["definitions"] = {
        "calls": "Estimate from stage 4/4b outputs; ignores retries and the stop judge.",
        "projection": "calls = per-dialogue estimate x N; days assume sequential calls at --interval seconds.",
        "tokens": "Whitespace-separated tokens, not Vietnamese lexical words.",
        "audio": "Duration from WAV header of every dialogue.wav under the audio root.",
    }
    return summary, numbers


def write_outputs(out, summary, numbers):
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out / "report_numbers.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Số liệu cho báo cáo", "", "| Mã | Giá trị |", "|---|---|"]
    lines += [f"| `[[{k}]]` | {str(v).replace('|', '/')} |" for k, v in sorted(numbers.items())]
    (out / "report_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, required=True, help="New directory for the outputs")
    ap.add_argument("--parsed-source", type=Path)
    ap.add_argument("--s1", type=Path, help="Stage 1 dir (results_vi)")
    ap.add_argument("--s15", type=Path, help="Stage 1.5 dir (results_vi_xt)")
    ap.add_argument("--s175", type=Path, help="Stage 1.75 dir (results_vi_dis)")
    ap.add_argument("--s4", type=Path, help="Stage 4 dir (vi_tt)")
    ap.add_argument("--s4b", type=Path, help="Stage 4b dir (vi_tt_bc)")
    ap.add_argument("--stage5-work", type=Path, help="Stage 5.1a work root (vi_stage5_work)")
    ap.add_argument("--audio", type=Path, help="Stage 5 audio root (vi_audio)")
    ap.add_argument(
        "--project", type=int, nargs="*", default=[1000, 5000], help="Dialogue counts to project"
    )
    ap.add_argument(
        "--interval", type=float, default=2.0, help="Seconds between LLM calls (projection)"
    )
    args = ap.parse_args(argv)
    try:
        summary, numbers = collect(args)
        if not numbers:
            raise ValueError(f"No known stage directory found under {args.data_root}")
        write_outputs(args.output_dir.expanduser().resolve(), summary, numbers)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Statistics: {args.output_dir} ({len(numbers)} keys, {len(summary['errors'])} unreadable files)"
    )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
