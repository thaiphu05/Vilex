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
from tools.pack_corpus import DIR2CODE  # noqa: E402
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

# Scenario order of the DuplexGen paper (cooperative | competitive).
SCENARIOS = ("TEA", "PLN", "INT", "NEG", "PER", "SOC")
# Single-token filled pauses: FP_INVENTORY (src/disfluency.py) + HESITATIONS
# (src/synthesis/core.py), copied here to keep this tool standard-library only.
FILLERS = {
    "ừm", "à", "ờ", "ừ", "hừm", "ơ", "ưm", "um", "ơi", "hử", "chà", "á", "ú",
    "uh", "umm", "uhh", "hm", "hmm", "huh", "er", "ah",
}  # fmt: skip
PUNCT = ".,?!;:…\"'“”‘’()-"
# Reference numbers from DuplexGen (arXiv 2607.26178v2).
# Fig. 3: human-annotated (backchannel, take_floor) proportion per slot.
PAPER_HUMAN_RATES = {
    "TEA": (0.32, 0.08),
    "PLN": (0.27, 0.09),
    "INT": (0.25, 0.05),
    "NEG": (0.23, 0.11),
    "PER": (0.24, 0.12),
    "SOC": (0.23, 0.12),
}
# Table 5 slots (calibration + evaluation) over 140 annotated utterances per task.
PAPER_SLOTS_PER_UTT = {
    "TEA": (160 + 383) / 140,
    "PLN": (148 + 341) / 140,
    "INT": (415 + 1016) / 140,
    "NEG": (242 + 439) / 140,
    "PER": (232 + 577) / 140,
    "SOC": (198 + 497) / 140,
}
# Table 3: (label, words, filler %, repetition %); Table 4: words and slots per utterance.
PAPER_SPOKEN = (
    ("Switchboard (người thật)", 1518024, 3.88, 1.39),
    ("DuplexGen user", 127786, 3.50, 0.43),
    ("DuplexGen assistant", 148430, 0.38, 0.01),
)
PAPER_WORDS_PER_UTT = 24.57
PAPER_AVG_SLOTS_PER_UTT = 5.57


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


def realized_slots(turn):
    """(slots, ignored): decisions up to the first floor_taking of the turn.

    Text after [TAKE_FLOOR] is cut at render time, so decisions recorded past it
    (older Stage 4 outputs still sampled backchannels there) are never heard.
    """
    segs = [s for s in turn.get("history") or [] if isinstance(s, dict) and "decision" in s]
    ft = min(
        (s["word_index"] for s in segs if s["decision"] == "floor_taking" and "word_index" in s),
        default=None,
    )
    if ft is None:
        return segs, 0
    kept = [s for s in segs if s.get("word_index", ft) <= ft]
    return kept, len(segs) - len(kept)


def spoken_tokens(text):
    """Normalized tokens actually spoken: text before [TAKE_FLOOR], tags removed."""
    text = str(text).split("[TAKE_FLOOR]")[0]
    words = re.sub(r"\[[^\]]*\]", " ", text).split()
    return [w for w in (w.strip(PUNCT).lower() for w in words) if w]


def slot_stats(items):
    decisions, by_role = Counter(), Counter()
    bc, user_turns, ft_tokens, modes, models = Counter(), 0, 0, Counter(), Counter()
    bc_total = bc_filled = bc_words = bc_tagged = ignored = 0
    for _, _, d in items:
        meta = d.get("meta") or {}
        modes[str(meta.get("tt_mode", "unknown"))] += 1
        models[str(meta.get("tt_model", "unknown"))] += 1
        for t in d["history"]:
            user_turns += 1 if t.get("role") == "user" else 0
            ft_tokens += str(t.get("content", "")).count("[TAKE_FLOOR]")
            segs, dropped = realized_slots(t)
            ignored += dropped
            for seg in segs:
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
        "post_ft_ignored": ignored,
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


def scenario_of(dataset):
    return DIR2CODE.get(dataset, dataset.upper())


def duplexgen_stats(items):
    """Per-scenario corpus / turn-taking / spoken-style numbers, as DuplexGen reports them."""
    rows = defaultdict(
        lambda: {
            "dialogues": 0,
            "turns": Counter(),
            "words": Counter(),
            "slots": 0,
            "decisions": Counter(),
            "probs": Counter(),
            "user_turns_with_ft": 0,
        }
    )
    spoken = {r: Counter() for r in ("user", "assistant")}
    for ds, _, d in items:
        row = rows[scenario_of(ds)]
        row["dialogues"] += 1
        for t in d["history"]:
            role = t.get("role")
            if role not in spoken:
                continue
            toks = spoken_tokens(t.get("content", ""))
            row["turns"][role] += 1
            row["words"][role] += len(toks)
            spoken[role]["words"] += len(toks)
            spoken[role]["fillers"] += sum(w in FILLERS for w in toks)
            spoken[role]["repetitions"] += sum(a == b for a, b in zip(toks, toks[1:]))
            if role != "user":
                continue
            segs, _ = realized_slots(t)
            row["slots"] += len(segs)
            row["user_turns_with_ft"] += any(s["decision"] == "floor_taking" for s in segs)
            for s in segs:
                row["decisions"][s["decision"]] += 1
                for label, p in (s.get("probs") or {}).items():
                    if isinstance(p, (int, float)):
                        row["probs"][label] += p

    def derive(row):
        users, slots = row["turns"]["user"], row["slots"]

        def per(n, d):
            return n / d if d else 0.0

        return {
            "dialogues": row["dialogues"],
            "user_utts": users,
            "assistant_utts": row["turns"]["assistant"],
            "words_per_utt_user": per(row["words"]["user"], users),
            "words_per_utt_assistant": per(row["words"]["assistant"], row["turns"]["assistant"]),
            "slots": slots,
            "slots_per_utt": per(slots, users),
            "words_per_slot": per(row["words"]["user"], slots),
            "bc_per_slot": per(row["decisions"]["backchannel"], slots),
            "ft_per_slot": per(row["decisions"]["floor_taking"], slots),
            "silence_per_slot": per(row["decisions"]["silence"], slots),
            "p_bc": per(row["probs"]["backchannel"], slots),
            "p_ft": per(row["probs"]["floor_taking"], slots),
            "ft_turn_pct": pct(row["user_turns_with_ft"], users),
            "bc_per_utt": per(row["decisions"]["backchannel"], users),
        }

    total = rows["ALL"]
    for code, row in list(rows.items()):
        if code == "ALL":
            continue
        total["dialogues"] += row["dialogues"]
        total["slots"] += row["slots"]
        total["user_turns_with_ft"] += row["user_turns_with_ft"]
        for key in ("turns", "words", "decisions", "probs"):
            total[key].update(row[key])
    order = [c for c in SCENARIOS if c in rows] + sorted(set(rows) - set(SCENARIOS) - {"ALL"})
    return {
        "scenarios": {c: derive(rows[c]) for c in order + ["ALL"]},
        "spoken": {
            role: {
                "words": c["words"],
                "filler_pct": pct(c["fillers"], c["words"]),
                "repetition_pct": pct(c["repetitions"], c["words"]),
            }
            for role, c in spoken.items()
        },
    }


def tts_by_scenario(s5):
    """Clips (variants), distinct dialogues, BCs and interrupt hosts per scenario."""
    rows = defaultdict(Counter)
    for g in s5["by_dataset_split"]:
        for code in (scenario_of(g["dataset"]), "ALL"):
            row = rows[code]
            row["clips"] += g["variants"]
            row["dialogues"] += g["dialogues"]
            row["backchannels"] += g["totals"].get("backchannels", 0)
            row["interrupt_hosts"] += g["totals"].get("interrupt_hosts", 0)
            row["host_utterances"] += g["totals"].get("host_utterances", 0)
    order = [c for c in SCENARIOS if c in rows] + sorted(set(rows) - set(SCENARIOS) - {"ALL"})
    return {c: dict(rows[c]) for c in order + ["ALL"]}


def duplexgen_numbers(dg, tts):
    numbers = {}
    for code, r in dg["scenarios"].items():
        k = f"DG_{code}_"
        numbers[k + "DIALOGUES"] = fmt_int(r["dialogues"])
        numbers[k + "USER_UTTS"] = fmt_int(r["user_utts"])
        numbers[k + "ASSISTANT_UTTS"] = fmt_int(r["assistant_utts"])
        numbers[k + "WORDS_PER_UTT_USER"] = fmt_num(r["words_per_utt_user"], 2)
        numbers[k + "WORDS_PER_UTT_ASSISTANT"] = fmt_num(r["words_per_utt_assistant"], 2)
        numbers[k + "SLOTS"] = fmt_int(r["slots"])
        numbers[k + "SLOTS_PER_UTT"] = fmt_num(r["slots_per_utt"], 2)
        numbers[k + "WORDS_PER_SLOT"] = fmt_num(r["words_per_slot"], 2)
        for name in ("bc_per_slot", "ft_per_slot", "silence_per_slot", "p_bc", "p_ft"):
            numbers[k + name.upper()] = fmt_num(r[name], 3)
        numbers[k + "FT_TURN_PCT"] = fmt_num(r["ft_turn_pct"])
        numbers[k + "BC_PER_UTT"] = fmt_num(r["bc_per_utt"], 2)
    for role, s in dg["spoken"].items():
        k = f"DG_{role.upper()}_"
        numbers[k + "WORDS"] = fmt_int(s["words"])
        numbers[k + "FILLER_PCT"] = fmt_num(s["filler_pct"], 2)
        numbers[k + "REP_PCT"] = fmt_num(s["repetition_pct"], 2)
    for code, r in (tts or {}).items():
        k = f"DG_{code}_"
        numbers[k + "CLIPS"] = fmt_int(r["clips"])
        numbers[k + "BC_PER_CLIP"] = fmt_num(r["backchannels"] / r["clips"] if r["clips"] else 0, 2)
        numbers[k + "INT_PER_CLIP"] = fmt_num(
            r["interrupt_hosts"] / r["clips"] if r["clips"] else 0, 2
        )
    return numbers


def duplexgen_markdown(dg, tts):
    sc = dg["scenarios"]
    codes = [c for c in sc if c != "ALL"]
    head = "| | " + " | ".join(codes) + " | Tổng |"
    sep = "|---|" + "---:|" * (len(codes) + 1)

    def row(label, key, digits=2, fn=fmt_num):
        return f"| {label} | " + " | ".join(fn(sc[c][key], digits) for c in codes + ["ALL"]) + " |"

    def paper(label, values, digits=2):
        cells = [fmt_num(values[c], digits) if c in values else "-" for c in codes]
        return f"| *{label}* | " + " | ".join(cells) + " | |"

    def ints(x, _):
        return fmt_int(x)

    lines = [
        "# Thống kê kiểu DuplexGen",
        "",
        "Số liệu Vilex tính trên vi_tt_bc (Stage 4b) và manifest Stage 5.1a. Dòng in nghiêng là số",
        "của paper DuplexGen (arXiv 2607.26178v2) để so sánh.",
        "",
        "## 1. Corpus (paper Table 4, Table 5, dataset card)",
        "",
        head,
        sep,
        row("Hội thoại", "dialogues", fn=ints),
        row("Lượt user", "user_utts", fn=ints),
        row("Lượt assistant", "assistant_utts", fn=ints),
        row("Từ / lượt user", "words_per_utt_user"),
        row("Từ / lượt assistant", "words_per_utt_assistant"),
        row("Slot", "slots", fn=ints),
        row("Slot / lượt user", "slots_per_utt"),
        paper("Paper: slot / lượt (Table 5)", PAPER_SLOTS_PER_UTT),
        row("Từ / slot", "words_per_slot"),
        "",
        f"Paper Table 4 (gộp 6 scenario): {fmt_num(PAPER_WORDS_PER_UTT, 2)} từ/lượt, "
        f"{fmt_num(PAPER_AVG_SLOTS_PER_UTT, 2)} slot/lượt. Dataset card: 5.999 hội thoại, "
        "125.137 slot.",
        "",
        "## 2. Hành vi turn-taking mỗi slot (paper Fig. 3)",
        "",
        head,
        sep,
        row("BC / slot", "bc_per_slot", 3),
        paper("Paper: BC / slot (người)", {c: v[0] for c, v in PAPER_HUMAN_RATES.items()}),
        row("FT / slot", "ft_per_slot", 3),
        paper("Paper: FT / slot (người)", {c: v[1] for c, v in PAPER_HUMAN_RATES.items()}),
        row("Silence / slot", "silence_per_slot", 3),
        row("Xác suất BC trung bình", "p_bc", 3),
        row("Xác suất FT trung bình", "p_ft", 3),
        row("% lượt user có FT", "ft_turn_pct", 1),
        row("BC / lượt user", "bc_per_utt"),
        "",
        "## 3. Văn nói theo role (paper Table 3)",
        "",
        "| Nguồn | Số từ | Filler % | Lặp từ % |",
        "|---|---:|---:|---:|",
    ]
    for role, s in dg["spoken"].items():
        lines.append(
            f"| Vilex {role} | {fmt_int(s['words'])} | {fmt_num(s['filler_pct'], 2)} "
            f"| {fmt_num(s['repetition_pct'], 2)} |"
        )
    for label, words, filler, rep in PAPER_SPOKEN:
        lines.append(f"| *{label}* | {fmt_int(words)} | {fmt_num(filler, 2)} | {fmt_num(rep, 2)} |")
    if tts:
        tcodes = [c for c in tts if c != "ALL"] + ["ALL"]
        lines += [
            "",
            "## 4. Audio Stage 5.1a (dataset card duplexgen-spoken)",
            "",
            "| | " + " | ".join(c if c != "ALL" else "Tổng" for c in tcodes) + " |",
            "|---|" + "---:|" * len(tcodes),
            "| Clip (variant) | " + " | ".join(fmt_int(tts[c]["clips"]) for c in tcodes) + " |",
            "| Hội thoại | " + " | ".join(fmt_int(tts[c]["dialogues"]) for c in tcodes) + " |",
            "| BC / clip | "
            + " | ".join(
                fmt_num(tts[c]["backchannels"] / tts[c]["clips"] if tts[c]["clips"] else 0, 2)
                for c in tcodes
            )
            + " |",
            "| Interrupt / clip | "
            + " | ".join(
                fmt_num(tts[c]["interrupt_hosts"] / tts[c]["clips"] if tts[c]["clips"] else 0, 2)
                for c in tcodes
            )
            + " |",
            "",
            "Paper (duplexgen-spoken): 35.746 clip (TEA 5.942, PLN 6.000, INT 5.875, NEG 5.988, "
            "PER 5.987, SOC 5.954), 10 variant/hội thoại, 24 kHz.",
        ]
    lines += [
        "",
        "## Cách đọc",
        "",
        "- Từ = token cách trắng. Tiếng Việt tách theo âm tiết nên số từ cao hơn tiếng Anh cho cùng nội dung.",
        "- Lượt có [TAKE_FLOOR] chỉ tính phần trước điểm ngắt; quyết định sau điểm ngắt bị loại "
        "(không được phát ra).",
        "- Tỉ lệ paper ở mục 2 là phân bố nhãn của người gán nhãn; tỉ lệ Vilex là quyết định "
        "đã bốc từ predictor, xác suất trung bình là đầu ra predictor.",
        '- Filler = token đơn trong tập filled pause tiếng Việt/Anh; "à", "á", "ơi" còn là '
        "tiểu từ nên filler % có thể bị đếm dư. Lặp từ = token trùng token liền trước.",
        "- Mục 4 cộng trên mọi variant; interrupt = host có uttr_type=interrupt.",
    ]
    return "\n".join(lines) + "\n"


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
        numbers["SLOT_POST_FT_IGNORED"] = fmt_int(slots["post_ft_ignored"])
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
        summary["duplexgen"] = {"text": duplexgen_stats(loaded["S4B"]), "tts": None}
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
            if "duplexgen" in summary:
                summary["duplexgen"]["tts"] = tts_by_scenario(s5)
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

    if "duplexgen" in summary:
        dg = summary["duplexgen"]
        numbers.update(duplexgen_numbers(dg["text"], dg["tts"]))

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
    if "duplexgen" in summary:
        dg = summary["duplexgen"]
        (out / "duplexgen_tables.md").write_text(
            duplexgen_markdown(dg["text"], dg["tts"]), encoding="utf-8"
        )


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
