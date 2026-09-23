#!/usr/bin/env python3
"""Restart one tmux fleet session with its exact original pane command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, check=check, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("uid_path", nargs="?")
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    args = parser.parse_args()
    target = f"={args.session}"
    command = run(
        "tmux", "list-panes", "-t", target, "-F", "#{pane_start_command}"
    ).stdout.splitlines()[0]
    if not command:
        raise SystemExit(f"empty_pane_start_command:{args.session}")
    from parse_fleet_worker_command import parse_command

    worker = parse_command(command)
    if args.uid_path is None:
        args.uid_path = worker["uid_path"]

    run("tmux", "kill-session", "-t", target, check=False)
    helper = Path(__file__).with_name("terminate_fleet_worker_scope.py")
    termination = run("python3", str(helper), args.uid_path)
    project = Path(
        os.environ.get("VQA_PROJECT", Path(__file__).resolve().parents[1])
    )
    scripts = project / "scripts"
    if worker["task"] == "grd":
        run(
            "bash",
            str(scripts / "launch_grd_production_worker.sh"),
            worker["worker_id"],
            worker["uid_path"],
            worker["output_root"],
            worker["resume_root"],
            "large",
            args.session,
        )
    elif worker["task"] == "sta":
        run(
            "bash",
            str(scripts / "launch_sta_production_worker.sh"),
            worker["worker_id"],
            worker["uid_path"],
            worker["output_root"],
            worker["resume_root"],
            args.session,
        )
    else:
        raise SystemExit(f"unsupported_task:{worker['task']}")
    time.sleep(max(0.0, args.startup_seconds))
    health = run(
        "tmux", "list-panes", "-t", target, "-F", "#{pane_dead}\t#{pane_pid}"
    ).stdout.strip()
    if not health or health.split("\t", 1)[0] != "0":
        raise SystemExit(f"session_restart_failed:{args.session}:{health}")
    print(
        json.dumps(
            {
                "session": args.session,
                "uid_path": args.uid_path,
                "health": health,
                "termination": json.loads(termination.stdout),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
