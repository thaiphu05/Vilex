"""Stage 5.1b — OmniVoice rendering (GPU) for the split Stage 5 pipeline.

Reads every manifest written by 5.1a into one global work queue and renders the
text units + backchannels to raw 24 kHz mono wavs (no VAD, no BC attenuation, no
timing). Batching is **cross-dialogue**: jobs are chunked purely by
``stage5_1b_render.omnivoice.batch_size`` (``1`` = one item per call) and a chunk
may mix speakers/dialogues because each item carries its own reference voice
(OmniVoice accepts per-item ``ref_audio``/``ref_text`` lists). Every chunk is
also written to ``<work_root>/_batch/batch_<NNNN>.jsonl`` for inspection.

Writes (per variant dir):
    units/<unit_id>.wav, bcs/<bc_unit_id>.wav
    manifest.json updated with: unit_audio, failed_units, pause_samples
and under ``<work_root>/_batch/``: ``batch_<NNNN>.jsonl`` + ``batches.json``.

Resume is filesystem-based: a job whose output wav already exists is skipped, so
manifests are only rewritten once at the end of the run.

Never imports convert_spoken (which pulls the aligner + VAD). VI/OmniVoice only.

Usage:
    python tts_render/stage5b_render.py [--config PATH] [--limit N]
"""

import argparse
import json
import random
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tts_render"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from src.config import load_config  # noqa: E402

from stage5_config import effective_language, resolve_stage5, work_root  # noqa: E402
from stage5_schema import Manifest, read_manifest, write_manifest  # noqa: E402

MODEL_ID = "k2-fsa/OmniVoice"
TARGET_SR = 24000


# --------------------------------------------------------------------------- audio
def _audio_to_tensor(item) -> torch.Tensor:
    wav = torch.from_numpy(np.asarray(item)).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.numel() == 0:
        wav = torch.zeros(1, int(0.2 * TARGET_SR))
    return wav


def _audio_duration(path: str) -> float:
    """Duration in seconds, preferring soundfile (torchaudio.info is deprecated)."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / max(info.samplerate, 1)
    except Exception:
        pass
    try:
        info = torchaudio.info(path)
        return info.num_frames / max(info.sample_rate, 1)
    except Exception:
        return 0.0


def _resample_mono(path: str, max_prompt_secs: float) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    max_samples = int(max_prompt_secs * TARGET_SR)
    if max_samples > 0 and wav.size(1) > max_samples:
        wav = wav[:, -max_samples:]
    return wav


def check_fingerprint(m: Manifest) -> None:
    """Refuse if the pool wavs changed since 5.1a recorded them."""
    for role, vp in m.voice_picks.items():
        path = Path(vp.wav)
        if not path.is_file():
            raise SystemExit(f"voice pool file missing for {role}: {vp.wav}")
        st = path.stat()
        mtime = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        if vp.wav_size and (st.st_size != vp.wav_size or mtime != vp.wav_mtime_ns):
            raise SystemExit(
                f"voice pool changed since prep for {role}: {vp.wav} "
                f"(recorded {vp.wav_size}/{vp.wav_mtime_ns}, now {st.st_size}/{mtime}). "
                "Re-run stage5a_prep.py."
            )


def _pause_samples(rng: random.Random, timing: Dict[str, Any]) -> int:
    intra = timing.get("intra_pause", {}) if isinstance(timing.get("intra_pause"), dict) else {}
    scale = float(intra.get("exp_scale", 0.30))
    lo = float(intra.get("min_sec", 0.10))
    hi = float(intra.get("max_sec", 1.00))
    sec = float(np.clip(rng.expovariate(1.0 / max(scale, 1e-6)), lo, hi))
    return int(sec * TARGET_SR)


class RefPrep:
    """Resolve each pool wav to a path safe to hand to OmniVoice.

    A reference longer than ``max_prompt_secs`` is trimmed once into
    ``<batch_dir>/refs/`` so the per-item list stays all-clone and still respects
    the clamp (the previous per-dialogue path did this on the tensor).
    """

    def __init__(self, batch_dir: Path, max_prompt_secs: float):
        self.refs_dir = batch_dir / "refs"
        self.max_prompt_secs = max_prompt_secs
        self._texts: Dict[str, str] = {}
        self._paths: Dict[str, str] = {}

    def text(self, wav_path: str) -> str:
        if wav_path not in self._texts:
            sidecar = Path(wav_path).with_suffix(".txt")
            self._texts[wav_path] = sidecar.read_text(encoding="utf-8").strip()
        return self._texts[wav_path]

    def path_from(self, wav_path: str) -> str:
        if wav_path in self._paths:
            return self._paths[wav_path]
        duration = _audio_duration(wav_path)
        if self.max_prompt_secs <= 0 or duration <= self.max_prompt_secs:
            resolved = wav_path
        else:
            self.refs_dir.mkdir(parents=True, exist_ok=True)
            out = self.refs_dir / f"{Path(wav_path).stem}_{int(self.max_prompt_secs)}s.wav"
            if not out.is_file():
                torchaudio.save(str(out), _resample_mono(wav_path, self.max_prompt_secs), TARGET_SR)
            resolved = str(out)
        self._paths[wav_path] = resolved
        return resolved

    def tensor_from(self, wav_path: str) -> torch.Tensor:
        return _resample_mono(self.path_from(wav_path), self.max_prompt_secs)


# --------------------------------------------------------------------------- jobs
def build_jobs(
    items: List[Tuple[Path, Manifest]], timing_cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Flatten every manifest into one queue, skipping units whose wav exists.

    Mutates each manifest's ``pause_samples`` (per-dialogue RNG); text units are
    returned with their reference path/text so the queue can be batched
    cross-dialogue. Pause units never enter the queue (rendered in 5.2a).
    """
    jobs: List[Dict[str, Any]] = []
    for mi, (mpath, m) in enumerate(items):
        rng = random.Random(int(m.seed) + m.variant_idx)
        vdir = mpath.parent
        for u in m.utterances:
            role = "user" if u.speaker_idx == 0 else "assistant"
            vp = m.voice_picks[role]
            for un in u.units:
                if un.kind == "pause":
                    if un.unit_id not in m.pause_samples:
                        m.pause_samples[un.unit_id] = _pause_samples(rng, timing_cfg)
                    continue
                out = vdir / "units" / f"{un.unit_id}.wav"
                if out.is_file():
                    # Resume: also backfill unit_audio when the manifest was not
                    # written yet (e.g. the previous run crashed before the end).
                    m.unit_audio.setdefault(un.unit_id, str(out))
                    continue
                jobs.append(_job(mi, un.unit_id, un.text, vp, False, out))
        for bc in m.backchannels:
            role = "user" if bc.speaker_idx == 0 else "assistant"
            vp = m.voice_picks[role]
            out = vdir / "bcs" / f"{bc.unit_id}.wav"
            if out.is_file():
                m.unit_audio.setdefault(bc.unit_id, str(out))
                continue
            jobs.append(_job(mi, bc.unit_id, bc.text, vp, True, out))
    return jobs


def _job(mi, unit_id, text, vp, is_bc: bool, out: Path) -> Dict[str, Any]:
    return {
        "mi": mi,
        "unit_id": unit_id,
        "text": text,
        "ref_wav": vp.wav,
        "is_bc": is_bc,
        "out_path": out,
    }


def chunk_jobs(
    jobs: List[Dict[str, Any]], batch_size: int, max_unit_chars: int
) -> List[List[Dict[str, Any]]]:
    """Chunk the queue by ``batch_size``; a unit over ``max_unit_chars`` goes alone."""
    normalized = [j for j in jobs if len(j["text"]) <= max_unit_chars]
    long_jobs = [j for j in jobs if len(j["text"]) > max_unit_chars]
    size = max(1, int(batch_size))
    chunks = [normalized[i : i + size] for i in range(0, len(normalized), size)]
    chunks += [[j] for j in long_jobs]
    return chunks


def write_batch_files(
    batch_dir: Path,
    chunks: List[List[Dict[str, Any]]],
    ref_prep: RefPrep,
    language: str,
    batch_size: int,
) -> None:
    """Materialize every chunk as _batch/batch_<NNNN>.jsonl + a batches.json index."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for bi, chunk in enumerate(chunks, start=1):
        name = f"batch_{bi:04d}.jsonl"
        with open(batch_dir / name, "w", encoding="utf-8") as fh:
            for j in chunk:
                fh.write(
                    json.dumps(
                        {
                            "id": j["unit_id"],
                            "text": j["text"],
                            "ref_audio": ref_prep.path_from(j["ref_wav"]),
                            "ref_text": ref_prep.text(j["ref_wav"]),
                            "language_id": language,
                            "speed": 1.2,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        index.append({"file": name, "n": len(chunk), "ids": [j["unit_id"] for j in chunk]})
    with open(batch_dir / "batches.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"batch_size": batch_size, "n_batches": len(chunks), "batches": index},
            fh,
            indent=2,
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------- generate
def _unwrap(out) -> List[torch.Tensor]:
    """Normalize an OmniVoice return into a list of [1, T] tensors."""
    if isinstance(out, (list, tuple)):
        items = list(out)
    else:
        items = [out]
    return [_audio_to_tensor(a) for a in items]


def _gen_batch(model, texts, ref_paths, ref_texts, speed, language):
    with torch.inference_mode():
        audios = model.generate(
            text=list(texts),
            speed=speed,
            ref_audio=list(ref_paths),
            ref_text=list(ref_texts),
            language=language,
            normalize_text=True,
        )
    return _unwrap(audios)


def _gen_one(model, text, ref_audio, ref_text, speed, language) -> torch.Tensor:
    with torch.inference_mode():
        out = model.generate(
            text=text,
            speed=speed,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            normalize_text=True,
        )
    return _unwrap(out)[0]


def render_chunks(
    model,
    chunks: List[List[Dict[str, Any]]],
    manifests: List[Manifest],
    ref_prep: RefPrep,
    speed: float,
    language: str,
    max_retries: int,
) -> Tuple[int, int]:
    """Render each chunk (batched), falling back to per-item, then silence."""
    rendered = failed = 0
    for chunk in chunks:
        texts = [j["text"] for j in chunk]
        paths = [ref_prep.path_from(j["ref_wav"]) for j in chunk]
        rtexts = [ref_prep.text(j["ref_wav"]) for j in chunk]

        audios = None
        for attempt in range(max_retries + 1):
            try:
                audios = _gen_batch(model, texts, paths, rtexts, speed, language)
                break
            except Exception as exc:  # pragma: no cover - model dependent
                print(f"WARNING: OmniVoice batch failed (attempt {attempt + 1}): {exc}")

        if audios is not None and len(audios) == len(chunk):
            for job, audio in zip(chunk, audios):
                _write_unit(job, audio, manifests)
                rendered += 1
            continue

        # Per-item fallback (each item keeps its own reference voice).
        for job in chunk:
            audio = None
            for attempt in range(max_retries + 1):
                try:
                    audio = _gen_one(
                        model,
                        job["text"],
                        (ref_prep.tensor_from(job["ref_wav"]), TARGET_SR),
                        ref_prep.text(job["ref_wav"]),
                        speed,
                        language,
                    )
                    break
                except Exception as exc:  # pragma: no cover - model dependent
                    print(f"WARNING: OmniVoice item {job['unit_id']} failed: {exc}")
            if audio is None:
                audio = torch.zeros(1, int(0.2 * TARGET_SR))
                manifests[job["mi"]].failed_units.append(job["unit_id"])
                failed += 1
            _write_unit(job, audio, manifests)
            rendered += 1
    return rendered, failed


def _write_unit(job: Dict[str, Any], audio: torch.Tensor, manifests: List[Manifest]) -> None:
    job["out_path"].parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(job["out_path"]), audio, TARGET_SR)
    manifests[job["mi"]].unit_audio[job["unit_id"]] = str(job["out_path"])


def iter_manifests(root: Path) -> List[Path]:
    return sorted(Path(p) for p in glob(str(root / "**" / "manifest.json"), recursive=True))


def main(config_path=None, limit: int = 0) -> None:
    cfg = load_config(config_path)
    s5 = resolve_stage5(cfg)
    root = Path(work_root(s5, cfg))
    language = effective_language(s5, "stage5_1b_render")
    language = "vi" if language.lower().startswith("vi") else "en"

    render_cfg = s5["stage5_1b_render"]
    if str(render_cfg.get("backend", "omnivoice")) != "omnivoice":
        raise SystemExit("Stage 5.1b only supports backend: omnivoice (VI path).")
    device = str(render_cfg.get("device", "cuda"))
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("render.device=cuda but no CUDA device is available.")
    omni = render_cfg.get("omnivoice", {}) if isinstance(render_cfg.get("omnivoice"), dict) else {}
    batch_size = int(omni.get("batch_size", 8))
    max_unit_chars = int(omni.get("max_unit_chars", 400))
    max_retries = int(omni.get("max_retries", 2))
    speed = float(render_cfg.get("speed", 1.2))
    max_prompt_secs = float(s5["stage5_2a_align"].get("max_prompt_secs", 10))
    timing_cfg = s5["stage5_2b_assemble"].get("timing", {})

    mpaths = iter_manifests(root)
    if not mpaths:
        raise SystemExit(f"No manifests under {root}. Run stage5a_prep.py first.")
    if limit:
        mpaths = mpaths[:limit]

    items = [(p, read_manifest(p)) for p in mpaths]
    for _p, m in items:
        check_fingerprint(m)

    batch_dir = root / "_batch"
    ref_prep = RefPrep(batch_dir, max_prompt_secs)

    jobs = build_jobs(items, timing_cfg)
    chunks = chunk_jobs(jobs, batch_size, max_unit_chars)
    write_batch_files(batch_dir, chunks, ref_prep, language, batch_size)
    print(
        f"Stage 5.1b: {len(items)} variants, {len(jobs)} pending units, "
        f"{len(chunks)} batches (batch_size={batch_size})."
    )
    if not jobs:
        # Persist the backfilled unit_audio/pause_samples from build_jobs.
        for mpath, m in items:
            write_manifest(mpath, m)
        print("Stage 5.1b done: nothing to render (all unit wavs present).")
        return

    from omnivoice import OmniVoice  # lazy: keeps CPU imports light

    model = OmniVoice.from_pretrained(MODEL_ID)
    if device == "cuda":
        model = model.cuda()
    if hasattr(model, "eval"):
        model.eval()

    manifests = [m for _p, m in items]
    rendered, failed = render_chunks(
        model, chunks, manifests, ref_prep, speed, language, max_retries
    )

    # Manifests are written once at the end; resume next run relies on the wav files.
    for mpath, m in items:
        write_manifest(mpath, m)
        if m.failed_units:
            print(
                f"[5.1b] failed: {m.rel_path}/var{m.variant_idx:02d}: "
                f"{', '.join(m.failed_units)} ({len(m.failed_units)})"
            )

    print(
        f"Stage 5.1b done: {rendered} rendered, {failed} failed -> "
        f"{batch_dir}/batch_*.jsonl. Work root -> {root}"
    )


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Stage 5.1b: OmniVoice render (GPU).")
    _ap.add_argument("--config", default=None, help="Path to config.yaml")
    _ap.add_argument("--limit", type=int, default=0, help="cap variants (0 = all)")
    _args = _ap.parse_args()
    main(_args.config, _args.limit)
