#!/usr/bin/env python3
"""Atomically snapshot input UIDs from valid completed subtask records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audit_vla_subtask_output import validate_record


def complete_lines(path: Path) -> tuple[list[bytes], int]:
    payload = path.read_bytes()
    cutoff = payload.rfind(b"\n") + 1
    return payload[:cutoff].splitlines(), len(payload) - cutoff


def snapshot_uids(
    root: Path, source_keys: set[str] | None,
) -> tuple[list[str], dict]:
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
    files = [path for path in files if "_quarantine" not in path.parts]
    seen: set[str] = set()
    selected: list[str] = []
    invalid = []
    partial_bytes = 0
    records = 0
    for path in files:
        lines, trailing = complete_lines(path)
        partial_bytes += trailing
        for line_number, raw in enumerate(lines, 1):
            if not raw.strip():
                continue
            records += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                invalid.append({
                    "file": str(path), "line": line_number,
                    "errors": [f"invalid_json:{error}"],
                })
                continue
            failures = validate_record(record, seen)
            if failures:
                invalid.append({
                    "file": str(path), "line": line_number, "errors": failures,
                })
                continue
            if source_keys is None or str(record.get("source_key")) in source_keys:
                selected.append(str(record["provenance"]["input_record_uid"]))
    summary = {
        "schema_version": "ready-subtask-uid-snapshot/v1",
        "root": str(root.resolve()),
        "source_keys": sorted(source_keys) if source_keys is not None else None,
        "files": len(files),
        "records": records,
        "selected_uids": len(selected),
        "ignored_partial_tail_bytes": partial_bytes,
        "valid": not invalid,
        "error_examples": invalid[:20],
        "error_count": len(invalid),
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subtask_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-key", action="append", dest="source_keys",
        help="Source key to include; repeat as needed. Omit to include all records under the input root.",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if not args.subtask_root.exists():
        parser.error(f"subtask root does not exist: {args.subtask_root}")

    uids, summary = snapshot_uids(
        args.subtask_root, set(args.source_keys) if args.source_keys else None,
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        temporary_summary = args.summary.with_name(
            f".{args.summary.name}.tmp.{os.getpid()}"
        )
        temporary_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_summary, args.summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if not summary["valid"]:
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text("".join(f"{uid}\n" for uid in sorted(uids)), encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
