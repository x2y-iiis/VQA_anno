#!/usr/bin/env python3
"""Stream reusable GRD checkpoints for a UID partition as a tar archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uids", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    uids = [line.strip() for line in args.uids.read_text().splitlines() if line.strip()]
    hits = files = bytes_total = 0
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for uid in uids:
            digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()
            parent = args.source_root / digest[:2]
            candidates: list[Path] = []
            main_file = parent / f"{digest}.jsonl"
            if main_file.is_file():
                candidates.append(main_file)
            unit_dir = parent / f"{digest}.units"
            if unit_dir.is_dir():
                candidates.extend(path for path in unit_dir.rglob("*") if path.is_file())
            if candidates:
                hits += 1
            for source in candidates:
                relative = source.relative_to(args.source_root)
                archive.add(source, arcname=str(relative), recursive=False)
                files += 1
                bytes_total += source.stat().st_size

    print(json.dumps({
        "uids": len(uids),
        "uid_checkpoint_hits": hits,
        "files_streamed": files,
        "bytes_streamed": bytes_total,
    }, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
