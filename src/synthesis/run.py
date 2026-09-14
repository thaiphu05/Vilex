import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

# Note: build_boundary_annotation_queue is removed as the new logic
# handles detection dynamically within speechify_turn_by_turn
from src.config import cfg_get, load_config
from src.llm_client import make_client
from src.synthesis.core import speechify_turn_by_turn
from src.synthesis.prompts import _normalize_ws


# HF classification model support (lazy import to keep synthesis env lightweight)
def _load_hf_tt_model(model_name_or_path: str, peft_path: Optional[str], load_in_4bit: bool):
    from src.inference_turntaking_hf import get_token_classification_model
    from transformers import AutoTokenizer

    # For PEFT adapter checkpoints, load the tokenizer from the base model to
    # avoid a transformers version mismatch where extra_special_tokens is saved
    # as a list instead of the expected dict.
    tokenizer_path = model_name_or_path
    adapter_cfg_path = os.path.join(model_name_or_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as _f:
            _adapter_cfg = json.load(_f)
        tokenizer_path = _adapter_cfg.get("base_model_name_or_path", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_token_classification_model(
        model_name_or_path=model_name_or_path,
        load_in_4bit=load_in_4bit,
        peft_path=peft_path,
    )
    return model, tokenizer


DATASET_CHOICES = ["socraticlm", "multiwoz", "interviewer", "negotiator", "persuader", "soda"]
SPLIT_CHOICES = ["train", "test"]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as rf:
        return json.load(rf)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        json.dump(obj, wf, indent=2, ensure_ascii=False)


def iter_input_jsons(root: Path, dataset: str, split: str) -> List[Path]:
    """
    results/llm_qwen/text_dialogue_{DATASET}/{SPLIT}/*.json
    """
    in_dir = root / f"text_dialogue_{dataset}" / split
    return sorted(in_dir.glob("*.json"))


def ensure_source_turns(example: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Converts example["history"] (list of {"role","content",...}) into (role, content) tuples.
    """
    hist = example.get("history", [])
    turns: List[Tuple[str, str]] = []
    for h in hist:
        role = h.get("role")
        content = h.get("content", "")
        if role not in ("user", "assistant"):
            # Skip non-dialogue entries (e.g. a `system` turn some models echo
            # back as part of the structured output) instead of aborting.
            continue
        turns.append((role, content))
    return turns


def extract_scenario(example: Dict[str, Any]) -> str:
    """
    Attempts to find a scenario description from common JSON keys.
    """
    if example.get("context"):
        return str(example["context"])
    return "A generic conversation."


def _build_args(cfg):
    """Adapt the config into the lightweight Namespace the loop below reads."""
    from types import SimpleNamespace

    s4 = cfg_get(cfg, "stage4_synthesis", {})
    llm = cfg_get(cfg, "llm", {})
    paths = cfg_get(cfg, "paths", {})
    hf = s4.get("hf", {}) if isinstance(s4.get("hf"), dict) else {}
    return SimpleNamespace(
        input_root=paths.get("results_dis_root", "data/results_vi_dis"),
        save_root=paths.get("synthesis_root", "data/vi_tt"),
        max_dialogues=s4.get("max_dialogues", 1000),
        llm_model_name=llm.get("writer_model", "gpt-4.1"),
        api_key=llm.get("api_key", "EMPTY"),
        base_url=llm.get("base_url", "http://localhost:8000/v1"),
        boundary_model_name=llm.get("boundary_model", "gpt-4.1-mini"),
        boundary_api_key=None,
        boundary_base_url=None,
        tt_model_name=llm.get("tt_model", "Qwen/Qwen3-14B"),
        tt_api_key=llm.get("api_key", "EMPTY"),
        tt_base_url=llm.get("base_url", "http://localhost:8000/v1"),
        hf_model_name_or_path=s4.get("hf_model_name_or_path"),
        hf_peft_path=s4.get("hf_peft_path"),
        hf_load_in_4bit=hf.get("load_in_4bit", False),
        hf_max_seq_length=hf.get("max_seq_length", 1024),
        hf_use_last_n_history=hf.get("use_last_n_history", 4),
        hf_batch_size=hf.get("batch_size", 8),
        max_turns=s4.get("max_turns", 0),
        temperature_user=s4.get("temperature_user", 0.2),
        temperature_ai=s4.get("temperature_ai", 0.2),
        target_language=cfg_get(cfg, "run.target_language", "vi"),
        max_workers=s4.get("max_workers", 1),
    )


def _apply_stage4_config(cfg):
    """Push guards / seed / FT punctuation onto core's module-level state."""
    from src.synthesis import core

    guards = cfg_get(cfg, "stage4_synthesis.guards", {})
    for key in ("length_guard_start", "interruption_guard_start", "length_guard_gap"):
        if isinstance(guards, dict) and key in guards:
            core.CONFIG[key] = guards[key]
    core.CONFIG["sampling_seed"] = cfg_get(cfg, "run.seed", core.CONFIG.get("sampling_seed", 42))
    ft_punct = cfg_get(cfg, "stage4_synthesis.ft_terminal_punct")
    if isinstance(ft_punct, list) and ft_punct:
        core._FT_TERMINAL_PUNCT = tuple(ft_punct)


def main(config_path=None):
    cfg = load_config(config_path)
    args = _build_args(cfg)
    _apply_stage4_config(cfg)

    input_root = Path(args.input_root)
    save_root = Path(args.save_root)

    # 1. Initialize Clients
    # Main generation client
    client = make_client(args.llm_model_name, args.api_key, args.base_url)

    # Boundary detection client (usually GPT-4-mini)
    b_key = args.boundary_api_key or os.getenv("OPENAI_API_KEY")
    b_url = args.boundary_base_url  # If None, OpenAI default
    client_boundary = make_client(args.boundary_model_name, b_key, b_url)

    # Turn-taking scoring: HF classification model OR LLM-based client
    hf_model = None
    hf_tokenizer = None
    if args.hf_model_name_or_path:
        print(f"Loading HF TT classification model: {args.hf_model_name_or_path}")
        hf_model, hf_tokenizer = _load_hf_tt_model(
            model_name_or_path=args.hf_model_name_or_path,
            peft_path=args.hf_peft_path,
            load_in_4bit=args.hf_load_in_4bit,
        )
        client_tt = None  # not needed when using HF model
    else:
        client_tt = make_client(args.tt_model_name, args.tt_api_key, args.tt_base_url)

    datasets = cfg_get(cfg, "run.datasets", [])
    splits = cfg_get(cfg, "run.splits", ["train"])
    pairs = [(d, s) for d in datasets for s in splits]

    n_inputs_found = 0
    for dataset, split in pairs:
        all_paths = iter_input_jsons(input_root, dataset, split)
        if not all_paths:
            print(f"[WARN] No input JSONs for {dataset}/{split}, skipping.")
            continue
        n_inputs_found += len(all_paths)

        # Filter pending before capping so reruns don't waste quota on already-done files
        out_dir = save_root / f"text_dialogue_{dataset}" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = [p for p in all_paths if not (out_dir / p.name).exists()]
        in_paths = pending[: args.max_dialogues] if args.max_dialogues else pending
        if not in_paths:
            print(f"[{dataset}/{split}] all {len(all_paths)} done, skipping.")
            continue

        def _process_one(in_path: Path):
            out_path = out_dir / in_path.name
            # atomic skip: second claim loses
            if out_path.exists():
                return ("skip", in_path.name, None)
            try:
                ex = read_json(in_path)
                source_turns = ensure_source_turns(ex)
                scenario_desc = extract_scenario(ex)
                utterances = speechify_turn_by_turn(
                    llm_model_name=args.llm_model_name,
                    client=client,
                    client_tt=client_tt,
                    client_boundary=client_boundary,
                    source_turns=source_turns,
                    scenario_description=scenario_desc,
                    max_turns=args.max_turns,
                    stop_check_every=0,
                    temperature_user=args.temperature_user,
                    temperature_ai=args.temperature_ai,
                    dataset=dataset,
                    tt_model_name=args.tt_model_name,
                    boundary_model_name=args.boundary_model_name,
                    target_language=args.target_language,
                    hf_model=hf_model,
                    hf_tokenizer=hf_tokenizer,
                    hf_max_seq_length=args.hf_max_seq_length,
                    hf_use_last_n_history=args.hf_use_last_n_history,
                    hf_batch_size=args.hf_batch_size,
                )
                for u in utterances:
                    u["content"] = _normalize_ws(u["content"])
                result = dict(ex)
                result["history"] = utterances
                result["meta"] = dict(ex.get("meta", {}))
                result["meta"]["turn_taking_applied"] = True
                result["meta"]["tt_model"] = args.hf_model_name_or_path or args.tt_model_name
                result["meta"]["tt_mode"] = "hf_classification" if hf_model else "llm_verbalized"
                result["meta"]["boundary_model"] = args.boundary_model_name
                result["meta"]["input_path"] = str(in_path)
                write_json(out_path, result)
                return ("ok", in_path.name, None)
            except Exception as e:
                return ("fail", in_path.name, str(e))

        if args.max_workers and args.max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_ok = n_fail = n_skip = 0
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {pool.submit(_process_one, p): p for p in in_paths}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(in_paths),
                    desc=f"Applying TT [{dataset}/{split}] x{args.max_workers}",
                ):
                    status, name, err = fut.result()
                    if status == "ok":
                        n_ok += 1
                    elif status == "skip":
                        n_skip += 1
                    else:
                        n_fail += 1
                        print(f"[FAIL] {name}: {err}")
            print(
                f"[{dataset}/{split}] workers={args.max_workers} ok={n_ok} skip={n_skip} fail={n_fail}"
            )
        else:
            for in_path in tqdm(in_paths, desc=f"Applying TT predictor [{dataset}/{split}]"):
                status, name, err = _process_one(in_path)
                if status == "fail":
                    print(f"[FAIL] {name}: {err}")

    # Every dataset warned and was skipped: the root is wrong (the published HF
    # layout rather than text_dialogue_<dataset>/<split>/, say). Exiting 0 here
    # reads as "nothing left to do" when nothing was ever read.
    if n_inputs_found == 0:
        raise SystemExit(
            f"No input dialogues under {input_root} for datasets={datasets} splits={splits}. "
            f"The root must contain text_dialogue_<dataset>/<split>/*.json; "
            f"convert the released corpus with "
            f"`tools/unpack_corpus.py --kind dialogues` first."
        )


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(
        description="Apply turn-taking predictor (TOKEN_BC/TOKEN_FT insertion)."
    )
    _ap.add_argument("--config", default=None, help="Path to config.yaml")
    main(_ap.parse_args().config)
