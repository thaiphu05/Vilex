"""Unpack parquet dumps back into per-dialogue JSON (parsed_source layout).

Nested layout
    <src>/<dataset>/<dataset>_<split>.parquet
Flat layout (--src holds the files directly, or pass --flat)
    <src>/<dataset>_<split>.parquet
Writes
    <out>/<dataset>/<split>/<example_id>.json

Handles the parsed_source schema (example_id / speakers / context / history /
meta), the raw SODA schema (dialogue / narrative -> alternating roles) and the
deepdialogue schema (transcript_id / split / turns).

Datasets that do not ship exactly {train, test} are reshaped: every row across
their parquet files is concatenated in order (train, validation, test, ...) and
cut into train/test at --ratio. This turns SODA's train/validation/test and
deepdialogue's lone `creatives` split into two splits each. Pass --no-resplit to
keep the source splits verbatim.

Usage:
    python tools/parquet_source.py --src data/parsed_source --out data/parsed_source
    python tools/parquet_source.py --src /path/flat --out data/parsed_source --dry-run
    python tools/parquet_source.py --src /path/flat --out data/parsed_source --keep-parquet
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

_UNSAFE = re.compile(r"[^\w\-.]")

# Concatenation order when reshaping a dataset's splits; unknown labels last.
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def safe_id(example_id) -> str:
    """Filename sanitization, identical to tools/unpack_corpus.py:43."""
    return _UNSAFE.sub("_", str(example_id))


def infer_split(parquet_path: Path, dataset: str) -> str:
    """Split label from `<dataset>_<split>.parquet`; fall back to `train`."""
    stem = parquet_path.stem
    prefix = f"{dataset}_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return "train"


def infer_dataset_split(
    parquet_path: Path, src: Path, forced_dataset: str = "", flat: bool = False
) -> tuple:
    """Resolve (dataset, split) from a per-dataset subfolder or a flat filename.

    Flat layout is `{dataset}_{split}.parquet` directly under `--src`; nested
    layout is `{src}/{dataset}/{dataset}_{split}.parquet`.
    """
    if flat or parquet_path.parent == src:
        parts = parquet_path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
        if forced_dataset:
            return forced_dataset, parquet_path.stem
        raise ValueError(
            f"cannot infer dataset/split from {parquet_path.name}; pass --dataset"
        )
    dataset = parquet_path.parent.name
    return dataset, infer_split(parquet_path, dataset)


def normalize_turns(raw) -> list:
    """Coerce a turn list into [[role, content], ...]."""
    turns = []
    for t in raw or []:
        if isinstance(t, dict):
            role, content = t.get("role"), t.get("content", "")
        elif isinstance(t, (list, tuple)):
            role = t[0] if len(t) > 0 else None
            content = t[1] if len(t) > 1 else ""
        else:
            continue
        if role is None:
            continue
        turns.append([role, content])
    return turns


def _is_raw_soda(row: dict) -> bool:
    """Raw SODA rows carry `dialogue`/`narrative`; parsed rows carry history/turns."""
    return (
        row.get("history") is None
        and row.get("turns") is None
        and row.get("dialogue") is not None
    )


def adapt_soda_row(row: dict, split: str, index: int) -> dict:
    """Normalize a raw SODA row the way src/speechify_datasets.iter_soda does."""
    narrative = (row.get("narrative") or "").strip()
    speakers = row.get("speakers") or []
    dialogue = row.get("dialogue") or []

    history = [
        [("user" if idx % 2 == 0 else "assistant"), str(utt).strip()]
        for idx, utt in enumerate(dialogue)
    ]
    ex_id = f"soda_{row.get('original_index', index)}"

    context = {
        "description": f"Social interaction scenario: {narrative}",
        "speakers": speakers,
        "original_narrative": narrative,
    }
    return {
        "example_id": ex_id,
        "speakers": ["user", "assistant"],
        "context": context,
        "history": history,
        "meta": {"split": split},
    }


def adapt_row(row: dict, dataset: str, split: str, index: int) -> dict:
    """Map one parquet row (any supported schema) to the parsed_source JSON record."""
    if _is_raw_soda(row):
        return adapt_soda_row(row, split, index)

    ex_id = (
        row.get("example_id")
        or row.get("transcript_id")
        or row.get("id")
        or f"{split}_{index}"
    )

    history = row.get("history")
    if history is None:
        history = row.get("turns")

    context = row.get("context")
    if context is None:
        context = row.get("narrative", "")

    meta = dict(row.get("meta") or {})
    meta["split"] = split
    if row.get("example_id") is None and row.get("transcript_id") is not None:
        meta.setdefault("dataset", dataset)

    return {
        "example_id": ex_id,
        "speakers": row.get("speakers") or ["user", "assistant"],
        "context": context if context is not None else "",
        "history": normalize_turns(history),
        "meta": meta,
    }


def write_split(dataset: str, split: str, rows: list, out: Path, args) -> tuple:
    """Write one split's rows to `<out>/<dataset>/<split>/*.json`; returns (written, skipped)."""
    split_dir = out / dataset / split
    if not args.dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for i, row in enumerate(tqdm(rows, desc=f"{dataset}/{split}")):
        rec = adapt_row(row, dataset, split, i)
        path = split_dir / f"{safe_id(rec['example_id'])}.json"
        if path.exists():
            skipped += 1
            continue
        if not args.dry_run:
            with path.open("w", encoding="utf-8") as w:
                json.dump(rec, w, indent=2, ensure_ascii=False)
        written += 1

    print(f"    {split}: {written} written, {skipped} skipped")
    return written, skipped


def clean_old_splits(dataset_dir: Path):
    """Drop stale split dirs (validation/, creatives/, ...) so only train/test remain."""
    if not dataset_dir.is_dir():
        return
    removed = []
    for child in dataset_dir.iterdir():
        if child.is_dir() and child.name not in ("train", "test"):
            shutil.rmtree(child)
            removed.append(child.name)
    if removed:
        print(f"    removed old split dir(s): {', '.join(sorted(removed))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/parsed_source", help="root holding <dataset>/*.parquet")
    ap.add_argument("--out", default="data/parsed_source", help="root to write <dataset>/<split>/*.json")
    ap.add_argument("--datasets", default="", help="comma list to restrict (default: all found)")
    ap.add_argument(
        "--flat",
        action="store_true",
        help="treat --src as a flat folder of {dataset}_{split}.parquet (auto-detected when parquet sits directly under --src)",
    )
    ap.add_argument(
        "--dataset",
        default="",
        help="fallback dataset name for filenames with no {dataset}_ prefix",
    )
    ap.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="train fraction when reshaping a dataset (default: 0.5)",
    )
    ap.add_argument(
        "--no-resplit",
        action="store_true",
        help="keep source splits verbatim instead of reshaping non-{train,test} datasets",
    )
    ap.add_argument(
        "--keep-old-splits",
        action="store_true",
        help="do not remove stale split dirs after reshaping",
    )
    ap.add_argument("--keep-parquet", action="store_true", help="do not delete source parquet")
    ap.add_argument("--max-rows", type=int, default=0, help="cap rows per parquet (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="scan only, write/delete nothing")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}

    parquets = sorted(src.rglob("*.parquet"))
    if not parquets:
        print(f"no parquet found under {src}")
        return

    by_dataset = {}
    for pf in parquets:
        dataset, split = infer_dataset_split(pf, src, args.dataset, args.flat)
        by_dataset.setdefault(dataset, []).append((pf, split))

    total_written = total_skipped = 0
    consumed = []

    for dataset in sorted(by_dataset):
        if wanted and dataset not in wanted:
            continue

        entries = by_dataset[dataset]
        splits = {s for _, s in entries}
        reshape = not args.no_resplit and splits != {"train", "test"}

        if reshape:
            ordered = sorted(entries, key=lambda e: (SPLIT_ORDER.get(e[1], 9), e[1]))
            ordered_rows = []
            for pf, _ in ordered:
                rows = pq.read_table(pf).to_pylist()
                if args.max_rows > 0:
                    rows = rows[: args.max_rows]
                ordered_rows.extend(rows)

            mid = int(len(ordered_rows) * args.ratio)
            pools = {"train": ordered_rows[:mid], "test": ordered_rows[mid:]}
            print(
                f"  [resplit] {dataset}: {len(ordered_rows)} rows "
                f"({' + '.join(s for _, s in ordered)}) -> "
                f"train {len(pools['train'])} / test {len(pools['test'])}"
            )
            for split in ("train", "test"):
                if args.dry_run:
                    print(f"    {split}: would write {len(pools[split])}")
                    total_written += len(pools[split])
                    continue
                w, s = write_split(dataset, split, pools[split], out, args)
                total_written += w
                total_skipped += s

            if args.max_rows <= 0:
                consumed.extend(pf for pf, _ in ordered)
            if not args.keep_old_splits and not args.dry_run:
                clean_old_splits(out / dataset)
            continue

        for pf, split in entries:
            rows = pq.read_table(pf).to_pylist()
            if args.max_rows > 0:
                rows = rows[: args.max_rows]
            print(f"  {pf.name} -> {dataset}/{split} ({len(rows)} rows)")
            if args.dry_run:
                print(f"    {split}: would write {len(rows)}")
                total_written += len(rows)
            else:
                w, s = write_split(dataset, split, rows, out, args)
                total_written += w
                total_skipped += s
            if args.max_rows <= 0:
                consumed.append(pf)

    if args.dry_run:
        print(f"dry-run: nothing written; {len(consumed)} parquet would be consumed")
    elif args.keep_parquet:
        print(f"kept {len(consumed)} parquet file(s)")
    else:
        for pf in consumed:
            pf.unlink()
        print(f"removed {len(consumed)} parquet file(s)")

    verb = "would write" if args.dry_run else "written"
    print(f"done: {total_written} {verb}, {total_skipped} skipped")


if __name__ == "__main__":
    main()
