"""Tests for Stage 1's offline `parsed_source` source mode.

`paths.source: parsed_source` makes Stage 1 read materialized per-dialogue JSON
instead of the raw corpora / HF Hub, so these cover the loader, the turn
coercion, and that `get_iterators` short-circuits before the Hub branches.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, load_config  # noqa: E402
from src.speechify_run import (  # noqa: E402
    _history_to_turns,
    _iter_parsed_source,
    get_iterators,
    split_by_parsed_source,
)


def _write_dialogue(path: Path, history, context="ctx", example_id=None) -> None:
    path.write_text(
        json.dumps(
            {
                "example_id": example_id or path.stem,
                "speakers": ["user", "assistant"],
                "context": context,
                "history": history,
                "meta": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )


def _args(root: Path, train=0, test=0) -> SimpleNamespace:
    return SimpleNamespace(
        source="parsed_source",
        parsed_source_root=str(root),
        max_train_samples=train,
        max_test_samples=test,
    )


def test_history_to_turns_accepts_pairs_and_dicts():
    assert _history_to_turns([["user", "hi"], ["assistant", "hello"]]) == [
        ("user", "hi"),
        ("assistant", "hello"),
    ]
    assert _history_to_turns([{"role": "user", "content": " hi "}]) == [("user", "hi")]
    assert _history_to_turns([["user", ""], ["user"], "bad", {}]) == []


def test_iter_parsed_source_reads_splits(tmp_path):
    train = tmp_path / "interviewer" / "train"
    train.mkdir(parents=True)
    _write_dialogue(train / "a.json", [["user", "one"]])
    _write_dialogue(train / "b.json", [["assistant", "two"]])

    rows = list(_iter_parsed_source(str(tmp_path), "interviewer", "train"))
    assert [r[0] for r in rows] == ["a", "b"]
    assert rows[0][1] == [("user", "one")]
    assert rows[0][2] == "ctx"


def test_iter_parsed_source_missing_split_is_empty(tmp_path):
    assert list(_iter_parsed_source(str(tmp_path), "nope", "train")) == []


def test_iter_parsed_source_respects_limit(tmp_path):
    train = tmp_path / "soda" / "train"
    train.mkdir(parents=True)
    for i in range(10):
        _write_dialogue(train / f"{i:02d}.json", [["user", f"u{i}"]])
    rows = list(_iter_parsed_source(str(tmp_path), "soda", "train", limit=4))
    assert len(rows) == 4


def test_split_by_parsed_source_returns_both_splits(tmp_path):
    for split in ("train", "test"):
        d = tmp_path / "multiwoz" / split
        d.mkdir(parents=True)
        _write_dialogue(d / f"{split}.json", [["user", split]])

    splits = dict(split_by_parsed_source("multiwoz", _args(tmp_path), _budget(tmp_path)))
    assert set(splits) == {"train", "test"}
    assert [r[1] for r in splits["train"]] == [[("user", "train")]]


def _budget(root):
    from src.speechify_run import SampleBudget

    return SampleBudget(train=None, test=None)


def test_get_iterators_short_circuits_to_parsed_source(tmp_path):
    """The parsed_source branch must run before the HF/raw dataset branches."""
    d = tmp_path / "interviewer" / "train"
    d.mkdir(parents=True)
    _write_dialogue(d / "a.json", [["user", "hello"]])

    # No HF/network access here; a fall-through would try load_dataset() and fail.
    splits = get_iterators("interviewer", _args(tmp_path), None)
    assert dict(splits)["train"][0][1] == [("user", "hello")]


def test_config_example_declares_offline_source_keys():
    cfg = load_config(REPO_ROOT / "config_example.yaml")
    assert cfg["paths"]["source"] in ("auto", "parsed_source")
    assert "parsed_source_root" in cfg["paths"]
