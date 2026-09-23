#!/usr/bin/env python3
"""Copy reusable GRD unit checkpoints for one UID ownership partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def copy_missing_file(source: Path, target: Path) -> tuple[int, int]:
    if target.exists():
        return 0, 0
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return 1, source.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uids", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()

    uids = [line.strip() for line in args.uids.read_text().splitlines() if line.strip()]
    uid_hits = files_copied = bytes_copied = 0
    for uid in uids:
        digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()
        source_parent = args.source_root / digest[:2]
        target_parent = args.target_root / digest[:2]
        hit = False

        source_jsonl = source_parent / f"{digest}.jsonl"
        if source_jsonl.is_file():
            hit = True
            count, size = copy_missing_file(
                source_jsonl, target_parent / source_jsonl.name
            )
            files_copied += count
            bytes_copied += size

        source_units = source_parent / f"{digest}.units"
        if source_units.is_dir():
            hit = True
            for source in source_units.rglob("*"):
                if not source.is_file():
                    continue
                target = target_parent / source_units.name / source.relative_to(source_units)
                count, size = copy_missing_file(source, target)
                files_copied += count
                bytes_copied += size
        uid_hits += int(hit)

    print(json.dumps({
        "uids": len(uids),
        "uid_checkpoint_hits": uid_hits,
        "files_copied": files_copied,
        "bytes_copied": bytes_copied,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
