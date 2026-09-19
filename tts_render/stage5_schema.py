"""Schema for the Stage 5 split-pipeline intermediate files (5.1a/5.1b/5.2a/5.2b).

Pure stdlib (dataclasses + json) so Stage 5.1a/5.2b run on CPU-only machines
without torch, pydantic, or the OmniVoice/Chatterbox environments. Do NOT add
third-party imports here.

The manifest written by 5.1a is the single source of truth for every later
stage: 5.1b only *fills* ``failed_units`` / ``pause_samples`` / ``unit_audio``
and must never rewrite the fields 5.1a owns.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
STAGE_PREP = "5.1a"

_UTTR_TYPES = {None, "backchannel", "interrupt"}
_NEXT_TYPES = {None, "backchannel", "interrupt", "last"}
_UNIT_KINDS = {"text", "pause"}


@dataclass
class Unit:
    kind: str  # "text" | "pause"
    unit_id: str  # "u_<host_idx>_<unit_idx>"
    text: str = ""  # empty for "pause"


@dataclass
class Utterance:
    turn: int
    text_idx: int
    speaker_idx: int  # 0 = user, 1 = assistant (pre-flip convention)
    uttr_type: Optional[str]
    next_uttr_type: Optional[str]
    is_last_text: bool
    isUttered: bool
    tts_text: str  # full accumulated text for this flush (hosts only)
    units: List[Unit] = field(default_factory=list)


@dataclass
class Backchannel:
    bc_idx: int
    speaker_idx: int  # speaker that utters the BC (the listener)
    text: str  # BC text as synthesized (rising tokens already carry "?")
    word_count: int  # words of the host text up to the BC slot (alignment anchor)
    host_idx: int  # index of the host utterance the BC anchors into
    next_uttr_type_of_host: Optional[str]
    unit_id: str  # "bc_<bc_idx>"


@dataclass
class VoicePick:
    wav: str  # absolute path into the voice-clone pool
    txt: str  # sidecar transcript path
    # Recorded by 5.1a so later stages can detect pool mutation mid-run.
    wav_size: int = 0
    wav_mtime_ns: int = 0


@dataclass
class Manifest:
    dialogue_id: str
    rel_path: str  # "text_dialogue_<ds>/<split>/<id>"
    variant_idx: int
    seed: int
    voice_picks: Dict[str, VoicePick]  # {"user": VoicePick, "assistant": VoicePick}
    utterances: List[Utterance] = field(default_factory=list)
    backchannels: List[Backchannel] = field(default_factory=list)
    # Filled by 5.1b (never by 5.1a):
    failed_units: List[str] = field(default_factory=list)  # unit_ids -> silence
    pause_samples: Dict[str, int] = field(default_factory=dict)  # unit_id -> samples
    unit_audio: Dict[str, str] = field(default_factory=dict)  # unit_id -> wav path
    schema_version: int = SCHEMA_VERSION
    stage: str = STAGE_PREP


def _err(msg: str) -> None:
    raise ValueError(f"manifest validation: {msg}")


def validate_manifest(m: Manifest) -> Manifest:
    """Raise ValueError on any contract violation. Returns m for chaining."""
    if m.schema_version != SCHEMA_VERSION:
        _err(f"schema_version {m.schema_version} != {SCHEMA_VERSION}")
    if m.stage != STAGE_PREP:
        _err(f"stage {m.stage!r} != {STAGE_PREP!r}")
    if not m.dialogue_id:
        _err("empty dialogue_id")
    if set(m.voice_picks) != {"user", "assistant"}:
        _err(f"voice_picks keys {sorted(m.voice_picks)} != ['assistant', 'user']")
    for role, vp in m.voice_picks.items():
        if not vp.wav or not vp.txt:
            _err(f"voice_picks[{role}] missing wav/txt")
        if vp.wav_size < 0 or vp.wav_mtime_ns < 0:
            _err(f"voice_picks[{role}] negative fingerprint")
    unit_ids = []
    for u in m.utterances:
        if u.uttr_type not in _UTTR_TYPES:
            _err(f"turn {u.turn}: bad uttr_type {u.uttr_type!r}")
        if u.next_uttr_type not in _NEXT_TYPES:
            _err(f"turn {u.turn}: bad next_uttr_type {u.next_uttr_type!r}")
        if u.speaker_idx not in (0, 1):
            _err(f"turn {u.turn}: bad speaker_idx {u.speaker_idx}")
        for un in u.units:
            if un.kind not in _UNIT_KINDS:
                _err(f"turn {u.turn}: bad unit kind {un.kind!r}")
            if un.kind == "text" and not un.text:
                _err(f"turn {u.turn}: text unit {un.unit_id} has empty text")
            unit_ids.append(un.unit_id)
    if len(unit_ids) != len(set(unit_ids)):
        _err("duplicate unit_id in utterances")
    seen = set(unit_ids)
    n_hosts = len(m.utterances)
    for bc in m.backchannels:
        if bc.speaker_idx not in (0, 1):
            _err(f"bc {bc.bc_idx}: bad speaker_idx {bc.speaker_idx}")
        if not bc.text:
            _err(f"bc {bc.bc_idx}: empty text")
        if not (0 <= bc.host_idx < n_hosts):
            _err(f"bc {bc.bc_idx}: host_idx {bc.host_idx} out of range {n_hosts}")
        if bc.word_count < 0:
            _err(f"bc {bc.bc_idx}: negative word_count")
        if bc.next_uttr_type_of_host not in _NEXT_TYPES:
            _err(f"bc {bc.bc_idx}: bad next_uttr_type_of_host")
        if bc.unit_id in seen:
            _err(f"bc {bc.bc_idx}: unit_id {bc.unit_id} collides with a host unit")
    return m


def _unit_from(d: Dict[str, Any]) -> Unit:
    return Unit(kind=d["kind"], unit_id=d["unit_id"], text=d.get("text", ""))


def manifest_from_dict(d: Dict[str, Any]) -> Manifest:
    m = Manifest(
        dialogue_id=d["dialogue_id"],
        rel_path=d["rel_path"],
        variant_idx=int(d["variant_idx"]),
        seed=int(d["seed"]),
        voice_picks={
            k: VoicePick(
                wav=v["wav"],
                txt=v["txt"],
                wav_size=int(v.get("wav_size", 0)),
                wav_mtime_ns=int(v.get("wav_mtime_ns", 0)),
            )
            for k, v in d["voice_picks"].items()
        },
        utterances=[
            Utterance(
                turn=u["turn"],
                text_idx=u["text_idx"],
                speaker_idx=u["speaker_idx"],
                uttr_type=u["uttr_type"],
                next_uttr_type=u["next_uttr_type"],
                is_last_text=bool(u["is_last_text"]),
                isUttered=bool(u["isUttered"]),
                tts_text=u.get("tts_text", ""),
                units=[_unit_from(x) for x in u.get("units", [])],
            )
            for u in d.get("utterances", [])
        ],
        backchannels=[
            Backchannel(
                bc_idx=b["bc_idx"],
                speaker_idx=b["speaker_idx"],
                text=b["text"],
                word_count=int(b["word_count"]),
                host_idx=int(b["host_idx"]),
                next_uttr_type_of_host=b["next_uttr_type_of_host"],
                unit_id=b["unit_id"],
            )
            for b in d.get("backchannels", [])
        ],
        failed_units=list(d.get("failed_units", [])),
        pause_samples=dict(d.get("pause_samples", {})),
        unit_audio=dict(d.get("unit_audio", {})),
        schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        stage=d.get("stage", STAGE_PREP),
    )
    return validate_manifest(m)


def manifest_to_dict(m: Manifest) -> Dict[str, Any]:
    return asdict(m)


def write_manifest(path: Path, m: Manifest) -> None:
    validate_manifest(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest_to_dict(m), fh, indent=2, ensure_ascii=False)


def read_manifest(path: Path) -> Manifest:
    with open(path, "r", encoding="utf-8") as fh:
        return manifest_from_dict(json.load(fh))
