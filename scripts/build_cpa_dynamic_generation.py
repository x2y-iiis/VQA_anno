#!/usr/bin/env python3
"""Build deterministic CPA worker slots from STA-ready and completed UID sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def read_uids(paths: list[Path]) -> list[str]:
    values: list[str] = []
    for path in paths:
        values.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    return values


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-uids", type=Path, action="append", required=True)
    parser.add_argument("--available-sta-uids", type=Path, required=True)
    parser.add_argument("--completed-cpa-uids", type=Path, required=True)
    parser.add_argument("--excluded-cpa-uids", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--max-uids-per-slot", type=int, default=64)
    args = parser.parse_args()
    if args.slots <= 0:
        parser.error("--slots must be positive")
    if args.max_uids_per_slot <= 0:
        parser.error("--max-uids-per-slot must be positive")

    master = list(dict.fromkeys(read_uids(args.master_uids)))
    available = set(read_uids([args.available_sta_uids]))
    completed = set(read_uids([args.completed_cpa_uids]))
    excluded = set(read_uids(args.excluded_cpa_uids))
    eligible = [
        uid for uid in master
        if uid not in completed
        and uid not in excluded
        and uid.split(":view=", 1)[0] in available
    ]

    # Stable hash placement prevents a refresh from moving every still-pending
    # UID between hosts.  Workers consume finite snapshots; a later generation
    # only schedules records absent from the refreshed completed set.
    partitions: list[list[str]] = [[] for _ in range(args.slots)]
    for uid in eligible:
        index = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "big") % args.slots
        partitions[index].append(uid)
    partitions = [values[:args.max_uids_per_slot] for values in partitions]

    for index, values in enumerate(partitions):
        atomic_write(
            args.output_dir / f"slot-{index:02d}.txt",
            "".join(f"{uid}\n" for uid in values),
        )
    manifest = {
        "schema": "cpa-dynamic-generation/v1",
        "master_uids": len(master),
        "available_sta_uids": len(available),
        "completed_cpa_uids": len(completed),
        "excluded_cpa_uids": len(excluded),
        "eligible_pending_uids": len(eligible),
        "scheduled_uids": sum(map(len, partitions)),
        "max_uids_per_slot": args.max_uids_per_slot,
        "slots": args.slots,
        "slot_counts": [len(values) for values in partitions],
    }
    atomic_write(args.output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
