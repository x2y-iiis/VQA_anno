#!/usr/bin/env python3
"""Emit newly durable CPA UIDs from the currently active finite workers.

The fleet controller calls this program every cycle.  Keeping a byte cursor per
active attempt log avoids repeatedly grepping every historical CPA log.  A UID
is admitted only while the producer is alive and its final-publication outbox
is fully drained, so a local ``stage_written`` line cannot outrun cloud
durability.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


SESSION_PREFIX = "vqa-cpa-dynamic-v3-slot-"
STAGE_PATTERN = re.compile(r"stage_written uid=([^ ]+) task=cpa(?: |$)")
COMPLETED_LEDGER = Path("/run/ti/cpa-completed-uids.log")
LAUNCH_PATTERN = re.compile(
    # Production panes invoke the launcher through an absolute path.  Accept
    # that form as well as a bare script name so live attempt logs are visible
    # to the durability-gated controller collector.
    r"(?:^|\s)(?:\S*/)?run_cpa_fleet_shard\.sh\s+"
    r"(?P<node>\S+)\s+(?P<slot>\d+)\s+\d+\s+(?P<root>\S+)"
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def outbox_drained(status: Path, max_age_seconds: float) -> bool:
    try:
        value = json.loads(status.read_text(encoding="utf-8"))
        pid = int(value["pid"])
        age = time.time() - float(value["updated_at_unix"])
        outbox = value["final_registration_outbox"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    counters = (
        "pending_files", "inflight_syncs", "inflight_units",
        "active_cloud_writers", "local_queued_files", "ready_files",
        "queued_cloud_acknowledgements",
    )
    return (
        age <= max_age_seconds
        and Path(f"/proc/{pid}").exists()
        and all(int(outbox.get(key) or 0) == 0 for key in counters)
        and not outbox.get("worker_failure_type")
    )


def active_attempt_logs(max_age_seconds: float) -> list[Path]:
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{session_name}|#{pane_start_command}"],
            text=True, capture_output=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        # The terminal ledger remains authoritative even when an overloaded
        # tmux server cannot enumerate live panes promptly.
        return []
    if result.returncode != 0:
        return []
    logs: list[Path] = []
    for row in result.stdout.splitlines():
        session, separator, command = row.partition("|")
        if not separator or not session.startswith(SESSION_PREFIX):
            continue
        match = LAUNCH_PATTERN.search(command)
        if not match:
            continue
        root = Path(match.group("root").strip("'\""))
        runtime = root / match.group("node") / f"shard-{match.group('slot')}"
        status = runtime / "state/request-parallel-status.json"
        if not outbox_drained(status, max_age_seconds):
            continue
        attempts = sorted(runtime.glob("attempt-*.log"), key=lambda path: path.stat().st_mtime)
        if attempts:
            logs.append(attempts[-1])
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("/run/ti/cpa-ready-index.json"))
    parser.add_argument("--max-age-seconds", type=float, default=180)
    args = parser.parse_args()
    try:
        state = json.loads(args.state.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        state = {"schema": "active-cpa-ready-index/v1", "logs": {}, "uids": []}
    logs = state.setdefault("logs", {})
    uids = set(state.get("uids") or [])
    # Finite workers append here only after annotate_videos has returned with
    # its final-publication outbox drained.  Unlike arbitrary historical logs,
    # the ledger is therefore safe to consume from byte zero.
    if COMPLETED_LEDGER.is_file():
        stat = COMPLETED_LEDGER.stat()
        saved = state.get("completed_ledger")
        if (
            not isinstance(saved, dict)
            or saved.get("inode") != stat.st_ino
            or stat.st_size < int(saved.get("offset", 0))
        ):
            offset = 0
        else:
            offset = int(saved["offset"])
        with COMPLETED_LEDGER.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read()
        newline = payload.rfind(b"\n")
        complete = payload[: newline + 1] if newline >= 0 else b""
        for raw in complete.splitlines():
            uid = raw.decode("utf-8", errors="replace").strip()
            if uid:
                uids.add(uid)
        state["completed_ledger"] = {
            "inode": stat.st_ino, "offset": offset + len(complete),
        }
    for log in active_attempt_logs(args.max_age_seconds):
        stat = log.stat()
        saved = logs.get(str(log))
        # The verified controller seed contains historical completions.  Start
        # at EOF for a newly seen log and collect only future durable records.
        if (
            not isinstance(saved, dict)
            or saved.get("inode") != stat.st_ino
            or stat.st_size < int(saved.get("offset", 0))
        ):
            logs[str(log)] = {"inode": stat.st_ino, "offset": stat.st_size}
            continue
        offset = int(saved["offset"])
        with log.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read()
        newline = payload.rfind(b"\n")
        complete = payload[: newline + 1] if newline >= 0 else b""
        for raw in complete.splitlines():
            match = STAGE_PATTERN.search(raw.decode("utf-8", errors="replace"))
            if match:
                uids.add(match.group(1))
        logs[str(log)] = {"inode": stat.st_ino, "offset": offset + len(complete)}
    state.update(uids=sorted(uids), updated_at_unix=time.time())
    atomic_json(args.state, state)
    print("".join(f"{uid}\n" for uid in sorted(uids)), end="")


if __name__ == "__main__":
    main()
