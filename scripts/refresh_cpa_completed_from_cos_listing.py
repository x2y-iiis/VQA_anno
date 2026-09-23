#!/usr/bin/env python3
"""Merge new immutable CPA object names into an audited completed UID list."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


HASH_RECORD = re.compile(r"/([0-9a-f]{64})\.jsonl$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--uid-path", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hashes: set[str] = set()
    zero_byte = 0
    exact_records = 0
    for line in args.listing.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 5:
            continue
        match = HASH_RECORD.search(fields[0])
        if not match:
            continue
        exact_records += 1
        if fields[4] == "0.00 B":
            zero_byte += 1
            continue
        hashes.add(match.group(1))

    uids: list[str] = []
    for path in args.uid_path:
        uids.extend(line for line in path.read_text().splitlines() if line)
    baseline = set(args.baseline.read_text().splitlines()) if args.baseline.is_file() else set()
    name_completed = {
        uid for uid in uids if hashlib.sha256(uid.encode()).hexdigest() in hashes
    }
    completed = sorted(baseline | name_completed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text("".join(f"{uid}\n" for uid in completed))
    os.replace(temporary, args.output)
    result = {
        "schema": "cpa-completed-content-baseline-plus-immutable-names/v1",
        "listing_lines": sum(1 for _ in args.listing.open()),
        "exact_jsonl_objects": exact_records,
        "zero_byte_exact_jsonl": zero_byte,
        "nonzero_name_hashes": len(hashes),
        "baseline_completed": len(baseline),
        "name_completed": len(name_completed),
        "merged_completed": len(completed),
        "output": str(args.output),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
