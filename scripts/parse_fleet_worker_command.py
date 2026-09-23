#!/usr/bin/env python3
"""Parse one tmux pane command for a GRD/STA production worker."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys


def tokens_from_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    # tmux preserves an outer quoted `bash -c` payload as one field for older
    # sessions. Parse that payload once more so all launcher generations share
    # the same representation.
    if len(tokens) == 1 and "run_grd_sta_fleet_worker.sh" in tokens[0]:
        tokens = shlex.split(tokens[0])
    return tokens


def parse_command(command: str) -> dict[str, str]:
    tokens = tokens_from_command(command)
    positions = [
        index for index, token in enumerate(tokens)
        if os.path.basename(token) == "run_grd_sta_fleet_worker.sh"
    ]
    if not positions:
        raise ValueError("fleet_worker_launcher_not_found")
    index = positions[-1]
    arguments = tokens[index + 1:index + 8]
    if len(arguments) != 7 or arguments[0] not in {"grd", "sta"}:
        raise ValueError("fleet_worker_arguments_invalid")
    environment = {}
    for token in tokens[:index]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.replace("_", "").isalnum():
            environment[key] = value
    task, worker_id, uid_path, output_root, resume_root, model, batch_size = arguments
    scratch = environment.get(
        "VQA_WORKER_SCRATCH",
        f"/run/ti/vqa-grd-sta-fleet/production-{worker_id}",
    )
    return {
        "task": task,
        "worker_id": worker_id,
        "uid_path": uid_path,
        "output_root": output_root,
        "resume_root": resume_root,
        "model": model,
        "batch_size": batch_size,
        "scratch": scratch,
        "grd_model": environment.get("GRD_MODEL", ""),
        "tos_upload_via_cos_fetch": environment.get(
            "TOS_UPLOAD_VIA_COS_FETCH", ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command")
    args = parser.parse_args()
    command = args.command if args.command is not None else sys.stdin.read().strip()
    if not command:
        parser.error("empty command")
    print(json.dumps(parse_command(command), sort_keys=True))


if __name__ == "__main__":
    main()
