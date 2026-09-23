"""Split one existing fleet UID partition into duration-balanced child owners."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--input-uids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--owners", nargs="+", required=True)
    args = parser.parse_args()

    if len(args.owners) < 2 or len(set(args.owners)) != len(args.owners):
        parser.error("owners must contain at least two unique labels")
    source = [line.strip() for line in args.input_uids.read_text().splitlines() if line.strip()]
    if not source or len(source) != len(set(source)):
        raise SystemExit("input_uids_must_be_nonempty_and_unique")

    table = pq.read_table(args.catalog, columns=["record_uid", "target_duration_ns"])
    durations = {
        str(uid): max(0, int(duration or 0))
        for uid, duration in zip(
            table.column("record_uid").to_pylist(),
            table.column("target_duration_ns").to_pylist(),
        )
    }
    missing = [uid for uid in source if uid not in durations]
    if missing:
        raise SystemExit(f"catalog_missing_input_uids:{len(missing)}")

    assigned: dict[str, list[str]] = {owner: [] for owner in args.owners}
    heap = [(0, index, owner) for index, owner in enumerate(args.owners)]
    heapq.heapify(heap)
    for uid in sorted(source, key=lambda value: (-durations[value], value)):
        total, index, owner = heapq.heappop(heap)
        assigned[owner].append(uid)
        heapq.heappush(heap, (total + durations[uid], index, owner))

    manifest = {
        "schema_version": "fleet-donor-split/v1",
        "catalog": str(args.catalog.resolve()),
        "input_uids": str(args.input_uids.resolve()),
        "input_count": len(source),
        "input_sha256": hashlib.sha256(
            "".join(f"{uid}\n" for uid in source).encode()
        ).hexdigest(),
        "owners": {},
    }
    union: set[str] = set()
    for owner in args.owners:
        selected = set(assigned[owner])
        ordered = [uid for uid in source if uid in selected]
        payload = "".join(f"{uid}\n" for uid in ordered)
        path = args.output_dir / f"{owner}.txt"
        atomic_write(path, payload)
        manifest["owners"][owner] = {
            "uid_path": str(path.resolve()),
            "episodes": len(ordered),
            "video_hours": sum(durations[uid] for uid in ordered) / 3_600_000_000_000,
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
        if union & selected:
            raise SystemExit("partition_overlap")
        union |= selected
    if union != set(source):
        raise SystemExit("partition_union_mismatch")
    atomic_write(args.output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
