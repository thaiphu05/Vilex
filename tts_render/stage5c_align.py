"""Stage 5.2a — VAD trim + whisperx forced alignment (GPU) for the split pipeline.

Consumes the raw unit wavs written by 5.1b, rebuilds each host utterance with
its pause units, applies the same Silero VAD trimming as the monolithic path,
attenuates backchannels (x0.8) and runs the whisperx aligner on hosts (and
multi-word backchannels). Writes a trimmed host wav plus ``alignment.json``.

Writes (per variant dir):
    hosts_trimmed/h_<idx>.wav, bcs_trimmed/<bc_unit_id>.wav
    alignment.json {hosts:[{host_idx, speaker_idx, text, words, unit_spans,
                            dur_sec, audio}], bcs:[...], fallback:[...]}

Reuses convert_spoken's helpers (same math as monolithic); imports it lazily so
CPU-only tooling can import this module's constants without loading torch.

Usage:
    python tts_render/stage5c_align.py [--config PATH] [--limit N]
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

from src.config import load_config  # noqa: E402

from stage5_config import effective_language, resolve_stage5, work_root  # noqa: E402
from stage5_schema import Manifest, read_manifest  # noqa: E402


def _assemble_utterance(cs, m: Manifest, u) -> Tuple[Any, List[Tuple[str, int, int]]]:
    """Concat a host's units into one mono tensor, tracking (text, start, end) spans.

    Pause units become faint noise of the sampled length 5.1b recorded.
    """
    pieces = []
    spans: List[Tuple[str, int, int]] = []
    pos = 0
    for un in u.units:
        if un.kind == "pause":
            n = int(m.pause_samples.get(un.unit_id, 0))
            if n > 0:
                pieces.append(cs._noise_floor_segment(n, cs.TARGET_SR, channels=1))
                pos += n
            continue
        path = m.unit_audio.get(un.unit_id)
        if not path or not Path(path).is_file():
            continue
        wav = cs.torchaudio.load(path)[0]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        pieces.append(wav)
        spans.append((un.text, pos, pos + wav.size(1)))
        pos += wav.size(1)
    if not pieces:
        empty = cs._noise_floor_segment(int(0.2 * cs.TARGET_SR), cs.TARGET_SR, channels=1)
        return empty, []
    return cs.torch.cat(pieces, dim=1), spans


def _trim_utterance(cs, raw, next_uttr_type, is_first_overall: bool, vad_model):
    """Port of convert_spoken's per-utterance VAD trim (:1381-1386 semantics).

    Returns (trimmed_audio, start_sample, end_sample) with values relative to
    ``raw`` so callers can shift unit spans.
    """
    ts = cs.get_speech_timestamps(
        cs._resample(raw, orig_freq=cs.TARGET_SR, new_freq=cs.PROMPT_SR),
        vad_model,
        threshold=0.3,
        sampling_rate=cs.PROMPT_SR,
    )
    if not ts:
        return raw, 0, raw.size(1)
    if is_first_overall:
        start = 0
    else:
        start = int(ts[0]["start"] * cs.TARGET_SR / cs.PROMPT_SR)
    if next_uttr_type == "interrupt" or next_uttr_type not in ("last",):
        end = int(ts[-1]["end"] * cs.TARGET_SR / cs.PROMPT_SR)
    else:
        end = raw.size(1)
    if not (0 <= start < end <= raw.size(1)):
        return raw, 0, raw.size(1)
    return raw[:, start:end], start, end


def process_manifest(
    cs, align_model, vad_model, m: Manifest, vdir: Path, language: str
) -> Dict[str, Any]:
    (vdir / "hosts_trimmed").mkdir(parents=True, exist_ok=True)
    (vdir / "bcs_trimmed").mkdir(parents=True, exist_ok=True)

    host_specs: List[Dict[str, Any]] = []
    seen_any_host = False

    for host_idx, u in enumerate(m.utterances):
        raw, spans = _assemble_utterance(cs, m, u)
        if u.uttr_type == "backchannel":
            trimmed = raw
        else:
            trimmed, start, _end = _trim_utterance(
                cs, raw, u.next_uttr_type, not seen_any_host, vad_model
            )
            spans = [(t, s - start, e - start) for t, s, e in spans]
        if u.uttr_type != "backchannel":
            seen_any_host = True
        out = vdir / "hosts_trimmed" / f"h_{host_idx:03d}.wav"
        cs.torchaudio.save(str(out), trimmed, cs.TARGET_SR)

        text = cs._align_text(u.tts_text, None)
        # A failed unit's text must not be aligned against its silence.
        if any(un.unit_id in m.failed_units and un.kind == "text" for un in u.units):
            text = " ".join(t for t, _s, _e in spans)
        host_specs.append(
            {"audio": trimmed, "text": text, "units": spans, "u": u, "path": str(out)}
        )

    # Host alignment (granularity/batch policy lives in convert_spoken globals).
    specs = [{"audio": h["audio"], "text": h["text"], "units": h["units"]} for h in host_specs]
    words_list = cs._align_hosts(align_model, specs) if specs else []

    hosts: List[Dict[str, Any]] = []
    fallback: List[int] = []
    for host_idx, (h, words) in enumerate(zip(host_specs, words_list)):
        if not words:
            fallback.append(host_idx)
        hosts.append(
            {
                "host_idx": host_idx,
                "speaker_idx": h["u"].speaker_idx,
                "text": h["text"],
                "words": words,
                "unit_spans": [[t, int(s), int(e)] for t, s, e in h["units"]],
                "dur_sec": h["audio"].size(1) / cs.TARGET_SR,
                "audio": h["path"],
            }
        )

    # Backchannels: raw -> x0.8 -> align if multi-word.
    bcs: List[Dict[str, Any]] = []
    for bc in m.backchannels:
        path = m.unit_audio.get(bc.unit_id)
        if not path or not Path(path).is_file():
            continue
        wav = cs.torchaudio.load(path)[0]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav * 0.8
        out = vdir / "bcs_trimmed" / f"{bc.unit_id}.wav"
        cs.torchaudio.save(str(out), wav, cs.TARGET_SR)
        words = cs._align_words_once(align_model, wav, bc.text) if len(bc.text.split()) > 1 else []
        bcs.append(
            {
                "unit_id": bc.unit_id,
                "bc_idx": bc.bc_idx,
                "host_idx": bc.host_idx,
                "speaker_idx": bc.speaker_idx,
                "text": bc.text,
                "word_count": bc.word_count,
                "words": words,
                "audio": str(out),
            }
        )

    payload = {
        "dialogue_id": m.dialogue_id,
        "variant_idx": m.variant_idx,
        "language": language,
        "hosts": hosts,
        "bcs": bcs,
        "fallback": fallback,
    }
    with open(vdir / "alignment.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return payload


def iter_manifests(root: Path) -> List[Path]:
    return sorted(Path(p) for p in glob(str(root / "**" / "manifest.json"), recursive=True))


def main(config_path=None, limit: int = 0) -> None:
    cfg = load_config(config_path)
    s5 = resolve_stage5(cfg)
    root = Path(work_root(s5, cfg))
    language = effective_language(s5, "stage5_2a_align")
    language = "vi" if language.lower().startswith("vi") else "en"

    import convert_spoken as cs  # heavy: torch + VAD + aligner

    cs._apply_tts_config(cfg)
    if (
        str(s5["stage5_2a_align"].get("device", "cuda")) == "cuda"
        and not cs.torch.cuda.is_available()
    ):
        raise SystemExit("align.device=cuda but no CUDA device is available.")
    align_model = cs._load_forced_aligner()
    vad_model = cs.load_silero_vad()

    manifests = iter_manifests(root)
    if limit:
        manifests = manifests[:limit]
    if not manifests:
        raise SystemExit(
            f"No manifests under {root}. Run stage5a_prep.py / stage5b_render.py first."
        )

    for mpath in manifests:
        m = read_manifest(mpath)
        payload = process_manifest(cs, align_model, vad_model, m, mpath.parent, language)
        n_fb = len(payload["fallback"])
        print(
            f"[5.2a] {m.rel_path}/var{m.variant_idx:02d}: "
            f"{len(payload['hosts'])} hosts, {len(payload['bcs'])} bcs, {n_fb} fallback"
        )
    print(f"Stage 5.2a done. Work root -> {root}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Stage 5.2a: VAD + whisperx alignment (GPU).")
    _ap.add_argument("--config", default=None, help="Path to config.yaml")
    _ap.add_argument("--limit", type=int, default=0, help="cap variants (0 = all)")
    _args = _ap.parse_args()
    main(_args.config, _args.limit)
