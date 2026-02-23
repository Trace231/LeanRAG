from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


def _pick(record: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    meta = record.get("metadata")
    if isinstance(meta, dict):
        for k in keys:
            if k in meta and meta[k] is not None:
                return meta[k]
    return None


def _parse_pos(value: Any) -> Optional[List[int]]:
    """Parse position into [line, col] if possible."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])]
        except Exception:
            return None
    if isinstance(value, dict):
        line = value.get("line") or value.get("row") or value.get("l")
        col = value.get("column") or value.get("col") or value.get("c")
        if line is not None and col is not None:
            try:
                return [int(line), int(col)]
            except Exception:
                return None
    if isinstance(value, str):
        m = re.match(r"Pos\((\d+),\s*(\d+)\)", value.strip())
        if m:
            return [int(m.group(1)), int(m.group(2))]
        m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", value.strip())
        if m:
            return [int(m.group(1)), int(m.group(2))]
    return None


def _to_proving_row(
    record: Dict[str, Any],
    default_url: str,
    default_commit: str,
) -> Optional[Dict[str, Any]]:
    url = str(_pick(record, "url", "repo_url", "github_url") or default_url).strip()
    commit = str(_pick(record, "commit", "sha", "hash") or default_commit).strip()
    full_name = str(_pick(record, "full_name", "theorem_full_name", "name") or "").strip()
    file_path = str(_pick(record, "file_path", "path", "theorem_path") or "").strip()

    start = _parse_pos(_pick(record, "start", "theorem_start", "start_pos", "pos"))
    end = _parse_pos(_pick(record, "end", "theorem_end", "end_pos"))
    if end is None:
        end = start

    theorem_statement = _pick(record, "theorem_statement", "statement")
    theorem_statement = str(theorem_statement) if theorem_statement is not None else ""

    if not (url and commit and full_name and file_path and start and end):
        return None

    return {
        "url": url,
        "commit": commit,
        "full_name": full_name,
        "file_path": file_path,
        "start": start,
        "end": end,
        "theorem_statement": theorem_statement,
    }


def _dedup_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, Tuple[int, int], Tuple[int, int]]:
    return (
        row["url"],
        row["commit"],
        row["file_path"],
        row["full_name"],
        tuple(row["start"]),
        tuple(row["end"]),
    )


def _iter_records(dataset_obj: Any) -> Iterable[Dict[str, Any]]:
    for r in dataset_obj:
        if isinstance(r, dict):
            yield r


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an HF dataset and convert to proving-theorem JSONL with dedup."
    )
    parser.add_argument(
        "--dataset",
        default="cat-searcher/leandojo-benchmark-4-random-sft",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--output",
        default="outputs/datasets/leandojo_benchmark4_proving_dedup.jsonl",
        help="Output JSONL path for proving dataset.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap after dedup (0 = no cap).")
    parser.add_argument("--default-url", default="", help="Fallback URL when row lacks url.")
    parser.add_argument("--default-commit", default="", help="Fallback commit when row lacks commit.")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept: List[Dict[str, Any]] = []
    skipped = 0

    for rec in _iter_records(ds):
        row = _to_proving_row(rec, args.default_url, args.default_commit)
        if row is None:
            skipped += 1
            continue
        key = _dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
        if args.max_rows > 0 and len(kept) >= args.max_rows:
            break

    with out_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "output": str(out_path),
                "num_rows": len(kept),
                "num_skipped_missing_fields": skipped,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

