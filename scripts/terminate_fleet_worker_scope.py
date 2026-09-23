#!/usr/bin/env python3
"""Terminate processes that still own one exact fleet UID partition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


MARKERS = (
    "annotate_videos.py",
    "run_cosmos3_full_seed_1024.sh",
    "run_grd_sta_fleet_worker.sh",
)


def command_line(pid: int) -> list[str]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode("utf-8", "replace") for part in payload.split(b"\0") if part]


def matching_pids(uid_path: str) -> list[int]:
    own_ancestors = {os.getpid(), os.getppid()}
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in own_ancestors:
            continue
        args = command_line(pid)
        if uid_path not in args:
            continue
        # A detached tmux server keeps the complete argv used to create its
        # first session.  That argv may contain both the worker marker and UID
        # path even after the pane has gone away.  Killing this process tears
        # down every sibling session on the node, so it must never be treated
        # as an owned worker process.
        if args and Path(args[0]).name == "tmux":
            continue
        joined = " ".join(args)
        if any(marker in joined for marker in MARKERS):
            matches.append(pid)
    return sorted(matches, reverse=True)


def signal_all(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uid_path")
    parser.add_argument("--grace-seconds", type=float, default=5.0)
    args = parser.parse_args()

    targets = matching_pids(args.uid_path)
    signal_all(targets, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, args.grace_seconds)
    survivors = targets
    while survivors and time.monotonic() < deadline:
        time.sleep(0.2)
        survivors = [pid for pid in survivors if Path(f"/proc/{pid}").exists()]
    signal_all(survivors, signal.SIGKILL)
    time.sleep(0.1)
    survivors = [pid for pid in survivors if Path(f"/proc/{pid}").exists()]
    print(json.dumps({
        "uid_path": args.uid_path,
        "matched_pids": targets,
        "sigkill_pids": survivors,
        "remaining_matching_pids": matching_pids(args.uid_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
