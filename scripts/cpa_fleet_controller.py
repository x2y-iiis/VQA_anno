#!/usr/bin/env python3
"""Continuously feed newly completed STA records into the finite CPA GPU fleet."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time


PROJECT = Path(os.environ.get("VQA_PROJECT", Path(__file__).resolve().parents[1]))
SEED_ROOT = Path("/run/ti/cpa-dynamic-v3")
MASTER_GLOB = PROJECT / "_runtime/cpa-fleet-20260919/partitions/uids-shard-*.txt"
HISTORICAL_STA_NODES = (
    "x2y-5", "x2y-6_4", "x2y-6_5", "x2y-6_7",
    "x2y-6_9", "x2y-6_10", "x2y-6_19", "x2y-6_20",
)
STA_NODES = tuple(os.environ.get("STA_NODES", " ".join(HISTORICAL_STA_NODES)).split())
if not STA_NODES or len(STA_NODES) != len(set(STA_NODES)):
    raise ValueError("STA_NODES must contain unique node names")
if STA_NODES[:len(HISTORICAL_STA_NODES)] != HISTORICAL_STA_NODES:
    raise ValueError(
        "STA_NODES must keep the historical node prefix unchanged; append new nodes only"
    )
HISTORICAL_CPA_NODES = (
    "x2y-6_12", "x2y-6_13", "x2y-6_14", "x2y-6_15",
    "x2y-6_16", "x2y-6_17", "x2y-6_18", "x2y-5_1", "x2y-5_2", "local",
)
CPA_NODES = tuple(os.environ.get(
    "CPA_NODES",
    " ".join(HISTORICAL_CPA_NODES),
).split())
if not CPA_NODES or len(CPA_NODES) != len(set(CPA_NODES)):
    raise ValueError("CPA_NODES must contain unique node names")
if CPA_NODES[:len(HISTORICAL_CPA_NODES)] != HISTORICAL_CPA_NODES:
    raise ValueError(
        "CPA_NODES must keep the historical node prefix unchanged; append new nodes only"
    )
CPA_GPUS_PER_NODE = 4
# Four isolated processes per 72 GiB GPU overlap LAS waits, frame decoding,
# SAM3 and CoTracker.  Two processes per GPU used only 15-18 GiB of 72 GiB,
# left most hosts 92-95% CPU-idle, and averaged roughly one-third GPU use.
# Keep the environment override so operators can step down without editing.
CPA_SLOTS_PER_NODE = int(os.environ.get("CPA_SLOTS_PER_NODE", "16"))
if CPA_SLOTS_PER_NODE < CPA_GPUS_PER_NODE:
    raise ValueError("CPA_SLOTS_PER_NODE must cover every GPU")
CPA_LOCAL_SLOTS = int(os.environ.get("CPA_LOCAL_SLOTS", "4"))
if CPA_LOCAL_SLOTS < 1:
    raise ValueError("CPA_LOCAL_SLOTS must be positive")
CPA_EXTRA_SLOTS_PER_NODE = int(os.environ.get("CPA_EXTRA_SLOTS_PER_NODE", "8"))
CPA_LOCAL_EXTRA_SLOTS = int(os.environ.get("CPA_LOCAL_EXTRA_SLOTS", "4"))
CPA_NEW_SLOTS_PER_NODE = int(os.environ.get("CPA_NEW_SLOTS_PER_NODE", "24"))
CPA_NEW_TAIL_EXTRA_NODES = tuple(
    os.environ.get("CPA_NEW_TAIL_EXTRA_NODES", "").split()
)
CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE = int(
    os.environ.get("CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE", "0")
)
CPA_NEW_TAIL2_EXTRA_NODES = tuple(
    os.environ.get("CPA_NEW_TAIL2_EXTRA_NODES", "").split()
)
CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE = int(
    os.environ.get("CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE", "0")
)
if CPA_EXTRA_SLOTS_PER_NODE < 0 or CPA_LOCAL_EXTRA_SLOTS < 0:
    raise ValueError("CPA extra slots must be non-negative")
if CPA_NEW_SLOTS_PER_NODE < CPA_GPUS_PER_NODE:
    raise ValueError("CPA_NEW_SLOTS_PER_NODE must cover every GPU")
if CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE < 0 or CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE < 0:
    raise ValueError("CPA new tail extra slot counts must be non-negative")
if (
    CPA_SLOTS_PER_NODE != 16 or CPA_LOCAL_SLOTS != 4
    or CPA_EXTRA_SLOTS_PER_NODE != 8 or CPA_LOCAL_EXTRA_SLOTS != 4
):
    raise ValueError(
        "historical CPA layout is immutable: remote=16+8 slots and local=4+4 slots"
    )

# Slot IDs are durable controller state.  Allocate the historical base layout
# first, then append expansion slots after it.  Increasing capacity therefore
# never remaps a live slot to another node/GPU and never requires a destructive
# fleet restart.
_cpa_node_slots: dict[str, list[int]] = {node: [] for node in CPA_NODES}
_cpa_slot_to_node_gpu: dict[int, tuple[str, int]] = {}
_cpa_next_slot = 0
for _cpa_node in HISTORICAL_CPA_NODES:
    _gpu_count = 1 if _cpa_node == "local" else CPA_GPUS_PER_NODE
    _base_count = CPA_LOCAL_SLOTS if _cpa_node == "local" else CPA_SLOTS_PER_NODE
    for _index in range(_base_count):
        _cpa_node_slots[_cpa_node].append(_cpa_next_slot)
        _cpa_slot_to_node_gpu[_cpa_next_slot] = (_cpa_node, _index % _gpu_count)
        _cpa_next_slot += 1
for _cpa_node in HISTORICAL_CPA_NODES:
    _gpu_count = 1 if _cpa_node == "local" else CPA_GPUS_PER_NODE
    _extra_count = (
        CPA_LOCAL_EXTRA_SLOTS if _cpa_node == "local" else CPA_EXTRA_SLOTS_PER_NODE
    )
    for _index in range(_extra_count):
        _cpa_node_slots[_cpa_node].append(_cpa_next_slot)
        _cpa_slot_to_node_gpu[_cpa_next_slot] = (_cpa_node, _index % _gpu_count)
        _cpa_next_slot += 1
for _cpa_node in CPA_NODES[len(HISTORICAL_CPA_NODES):]:
    for _index in range(CPA_NEW_SLOTS_PER_NODE):
        _cpa_node_slots[_cpa_node].append(_cpa_next_slot)
        _cpa_slot_to_node_gpu[_cpa_next_slot] = (
            _cpa_node, _index % CPA_GPUS_PER_NODE,
        )
        _cpa_next_slot += 1
_new_cpa_nodes = CPA_NODES[len(HISTORICAL_CPA_NODES):]
if len(CPA_NEW_TAIL_EXTRA_NODES) != len(set(CPA_NEW_TAIL_EXTRA_NODES)):
    raise ValueError("CPA_NEW_TAIL_EXTRA_NODES must contain unique node names")
if any(node not in _new_cpa_nodes for node in CPA_NEW_TAIL_EXTRA_NODES):
    raise ValueError("CPA_NEW_TAIL_EXTRA_NODES must contain expansion nodes only")
# Append optional capacity after the complete immutable base layout. Increasing
# the original per-node count here would remap every later node's durable slot
# IDs, whereas these tail slots preserve all existing assignments.
for _cpa_node in CPA_NEW_TAIL_EXTRA_NODES:
    for _index in range(CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE):
        _cpa_node_slots[_cpa_node].append(_cpa_next_slot)
        _cpa_slot_to_node_gpu[_cpa_next_slot] = (
            _cpa_node, _index % CPA_GPUS_PER_NODE,
        )
        _cpa_next_slot += 1
if len(CPA_NEW_TAIL2_EXTRA_NODES) != len(set(CPA_NEW_TAIL2_EXTRA_NODES)):
    raise ValueError("CPA_NEW_TAIL2_EXTRA_NODES must contain unique node names")
if any(node not in _new_cpa_nodes for node in CPA_NEW_TAIL2_EXTRA_NODES):
    raise ValueError("CPA_NEW_TAIL2_EXTRA_NODES must contain expansion nodes only")
for _cpa_node in CPA_NEW_TAIL2_EXTRA_NODES:
    for _index in range(CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE):
        _cpa_node_slots[_cpa_node].append(_cpa_next_slot)
        _cpa_slot_to_node_gpu[_cpa_next_slot] = (
            _cpa_node, _index % CPA_GPUS_PER_NODE,
        )
        _cpa_next_slot += 1
CPA_NODE_SLOTS = {node: tuple(slots) for node, slots in _cpa_node_slots.items()}
CPA_SLOT_TO_NODE_GPU = dict(_cpa_slot_to_node_gpu)
CPA_TOTAL_SLOTS = _cpa_next_slot


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "time": time.time(), **fields}, sort_keys=True), flush=True)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def read_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_lines(path: Path, values: set[str]) -> None:
    atomic_text(path, "".join(f"{value}\n" for value in sorted(values)))


def ssh(node: str, command: str, *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    if node == "local":
        return subprocess.run(
            ["bash", "-lc", command], text=True, capture_output=True, timeout=timeout,
        )
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"root@{node}", command],
        text=True, capture_output=True, timeout=timeout,
    )


def collect_stage_uids(
    node: str, task: str, locations: dict[str, str] | None = None,
) -> set[str]:
    if task in {"sta", "cpa"}:
        collector = f"/run/ti/collect_active_{task}_ready.py"
        source = PROJECT / "scripts" / f"collect_active_{task}_ready.py"
        try:
            emit_locations = " --emit-locations" if task == "sta" and locations is not None else ""
            result = ssh(
                node,
                f"nice -n -10 /usr/bin/python3 {collector}{emit_locations}",
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            emit("collection_timed_out", node=node, task=task)
            return set()
        # /run/ti is intentionally ephemeral and disappears after a node
        # reboot. Repair only the missing helper, then retry once; treating the
        # node as empty forever would starve CPA while STA is producing valid
        # durable records.
        if result.returncode != 0 and "can't open file" in result.stderr and source.is_file():
            if node == "local":
                shutil.copy2(source, collector)
                os.chmod(collector, 0o755)
                deployed = subprocess.CompletedProcess([], 0, "", "")
            else:
                deployed = subprocess.run(
                    ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                     str(source), f"root@{node}:{collector}"],
                    text=True, capture_output=True, timeout=30,
                )
                if deployed.returncode == 0:
                    ssh(node, f"chmod 755 {collector}", timeout=15)
            if deployed.returncode == 0:
                emit("collector_redeployed", node=node, task=task, path=collector)
                try:
                    result = ssh(
                        node,
                        f"nice -n -10 /usr/bin/python3 {collector}{emit_locations}",
                        timeout=60,
                    )
                except subprocess.TimeoutExpired:
                    emit("collection_timed_out", node=node, task=task)
                    return set()
        if result.returncode != 0:
            emit("collection_failed", node=node, task=task, returncode=result.returncode,
                 stderr=result.stderr[-500:])
            return set()
        uids = set()
        for line in result.stdout.splitlines():
            if not line:
                continue
            if line.startswith("@location "):
                if locations is None:
                    continue
                try:
                    uid, path = json.loads(line[len("@location "):])
                except (ValueError, TypeError, json.JSONDecodeError):
                    emit("location_parse_failed", node=node, line=line[-500:])
                    continue
                if isinstance(uid, str) and isinstance(path, str):
                    locations[uid] = path
                continue
            uids.add(line)
        return uids
    raise ValueError(f"unsupported_stage_collection_task:{task}")


def active_cpa_slots_on_node(node: str) -> tuple[str, list[int]]:
    """Return verified active slots for one node without serializing peers."""
    # A tmux session alone is not proof of a live worker. Under cgroup OOM the
    # old `python | tee` launcher could lose Python while `tee` survived.
    command = r'''for session in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^vqa-cpa-dynamic-v3-slot-[0-9]+$'); do
  pane_pid=$(tmux list-panes -t "$session" -F '#{pane_pid}' 2>/dev/null | head -1)
  slot=${session##*-}
  pane_args=$([ -z "$pane_pid" ] || ps -p "$pane_pid" -o args= 2>/dev/null)
  if [ -n "$pane_pid" ] && { 
       ps --ppid "$pane_pid" -o args= 2>/dev/null | grep -q '[r]un_cosmos3_cpa.py' \
       || printf '%s\n' "$pane_args" | grep -q '[r]un_cpa_fleet_shard.sh';
     }; then
    echo "$slot"
  else
    [ -z "$pane_pid" ] || kill -TERM -- "-$pane_pid" 2>/dev/null || true
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
done'''
    allowed = set(CPA_NODE_SLOTS[node])
    try:
        result = ssh(node, command, timeout=30)
    except subprocess.TimeoutExpired:
        # An unreachable node is conservatively treated as fully active.
        return node, sorted(allowed)
    if result.returncode != 0:
        return node, sorted(allowed)
    return node, [
        int(value) for value in result.stdout.splitlines()
        if value.isdigit() and int(value) in allowed
    ]


def active_cpa_slots() -> tuple[set[int], dict[str, list[int]]]:
    active: set[int] = set()
    slots_by_node: dict[str, list[int]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(CPA_NODES), thread_name_prefix="cpa-active-probe",
    ) as executor:
        futures = [executor.submit(active_cpa_slots_on_node, node) for node in CPA_NODES]
        for future in concurrent.futures.as_completed(futures):
            node, node_slots = future.result()
            slots_by_node[node] = node_slots
            active.update(node_slots)
    return active, slots_by_node


def load_attempts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {key: int(value) for key, value in json.loads(path.read_text()).items()}


def save_attempts(path: Path, attempts: dict[str, int]) -> None:
    atomic_text(path, json.dumps(attempts, sort_keys=True, separators=(",", ":")) + "\n")


def load_active_slots(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return set()
    return {int(slot) for slot in value if str(slot).isdigit()}


def save_active_slots(path: Path, slots: set[int]) -> None:
    atomic_text(path, json.dumps(sorted(slots), separators=(",", ":")) + "\n")


def record_finished_attempts(
    attempts: dict[str, int],
    assignments: dict[int, str],
    finished_slots: set[int],
    available: set[str],
    completed: set[str],
) -> dict[str, int]:
    """Count only unfinished UIDs from leases whose worker has exited."""
    result = dict(attempts)
    for slot in finished_slots:
        assigned_value = assignments.get(slot)
        if not assigned_value:
            continue
        for uid in read_lines(Path(assigned_value)):
            if uid not in completed and uid.split(":view=", 1)[0] in available:
                result[uid] = result.get(uid, 0) + 1
    return result


def discover_sta_reuse_outputs(node: str) -> str:
    script = r'''python3 - <<'PY'
import glob, json, os

cache = '/run/ti/cpa-sta-reuse-roots.txt'
try:
    cached = open(cache).read().strip()
except OSError:
    cached = ''
# This cache is written only after a successful discovery.  Do not re-stat all
# FUSE-backed roots every controller cycle: one slow mount previously made the
# "fast" path exceed the 30-second SSH deadline.  The worker launcher still
# validates every selected root before starting annotation.
if cached:
    print(cached)
    raise SystemExit(0)

# A launch manifest is only a snapshot of the roots known when that worker
# started.  Reusing it verbatim omitted later/fallback STA publishers (for
# example sta-local and an early rebalance shard), even though the durable
# coverage audit had admitted their records into the CPA seed.  Preserve the
# manifest roots and union every locally mounted STA publisher before making a
# new finite CPA generation.
paths = glob.glob('/run/ti/cpa-dynamic-v3/*/shard-*/launch.json')
paths += glob.glob('/run/ti/cpa-dynamic-controller/generations/*/*/shard-*/launch.json')
roots = []
if paths:
    path = max(paths, key=os.path.getmtime)
    roots.extend(json.load(open(path)).get('reuse_sta_output') or [])
roots.extend([
    '/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-seed-sta-full-20260909',
    '/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-turbo-default-20260912/sta-output',
])
for base in ('/mnt/human_data/video_cleaning', '/mnt/human_data/video-cleaning'):
    roots.extend(glob.glob(
        base + '/cosmos3-video-generation-general-v1.5-vqa-grd-sta-fleet-*/sta-*'
    ))
usable = []
seen = set()
for root in roots:
    root = root.rstrip('/')
    logical = root
    for marker in ('/video_cleaning/', '/video-cleaning/'):
        if marker in root:
            logical = root.split(marker, 1)[1]
            break
    if logical in seen or not os.path.isdir(os.path.join(root, 'sta', 'shards')):
        continue
    seen.add(logical)
    usable.append(root)
value = ':'.join(usable)
if value:
    temporary = cache + '.tmp.%s' % os.getpid()
    with open(temporary, 'w') as stream:
        stream.write(value + '\n')
    os.replace(temporary, cache)
print(value)
PY'''
    result = ssh(node, script, timeout=30)
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise RuntimeError(f"sta_reuse_outputs_unavailable:{node}:{result.stderr[-300:]}")
    return value


def deploy_generation(local_root: Path, remote_root: Path, node: str) -> None:
    remote = ssh(node, f"mkdir -p {shlex.quote(str(remote_root.parent))}", timeout=30)
    if remote.returncode != 0:
        raise RuntimeError(f"remote_generation_mkdir_failed:{node}:{remote.stderr[-300:]}")
    # Do not stream ``tar | ssh`` here.  When an overloaded node times out,
    # subprocess.run terminates only ssh and leaves the tar producer blocked
    # forever on its pipe.  A small temporary archive makes each phase bounded
    # and leaves no producer child behind after a timeout.
    with tempfile.TemporaryDirectory(prefix="cpa-generation-", dir="/run/ti") as temporary:
        archive = Path(temporary) / "generation.tar"
        packed = subprocess.run(
            ["tar", "-C", str(local_root), "-cf", str(archive), "."],
            text=True, capture_output=True, timeout=30,
        )
        if packed.returncode != 0:
            raise RuntimeError(f"generation_pack_failed:{node}:{packed.stderr[-300:]}")
        remote_archive = remote_root.parent / f".{remote_root.name}.{os.getpid()}.tar"
        cos_archive = (
            "cos://datasets-1409717487/video-cleaning/_runtime/"
            f"cpa-controller-generations/{remote_root.name}/{node}.tar"
        )
        copied = subprocess.run(
            ["/root/coscli", "cp", str(archive), cos_archive,
             "--thread-num", "16", "--disable-log"],
            text=True, capture_output=True, timeout=60,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"generation_cos_upload_failed:{node}:{copied.stderr[-300:]}")
        unpacked = ssh(
            node,
            f"mkdir -p {shlex.quote(str(remote_root))} && "
            f"/root/coscli cp {shlex.quote(cos_archive)} {shlex.quote(str(remote_archive))} "
            f"--thread-num 16 --disable-log && "
            f"tar -C {shlex.quote(str(remote_root))} -xf {shlex.quote(str(remote_archive))} && "
            f"rm -f {shlex.quote(str(remote_archive))}",
            timeout=120,
        )
        if unpacked.returncode != 0:
            raise RuntimeError(f"generation_cos_download_or_unpack_failed:{node}:{unpacked.stderr[-300:]}")


def deploy_and_launch_node(
    generation_root: Path,
    remote_root: Path,
    node: str,
    slot_gpu: dict[int, int],
) -> tuple[str, list[int], bool]:
    """Deploy and launch one node without blocking unrelated fleet nodes."""
    try:
        # Mount visibility is node-local.  Build the same logical 34-root STA
        # union on each destination so an underscore/hyphen mount alias that
        # exists on one GPU host cannot make every pane on another host exit.
        reuse_outputs = discover_sta_reuse_outputs(node)
        if node == "local":
            launch_root = generation_root / "mapped"
        else:
            deploy_generation(generation_root / "mapped", remote_root, node)
            launch_root = remote_root
    except Exception as error:
        emit("generation_deploy_failed", node=node, generation=generation_root.name,
             error_type=type(error).__name__, error=str(error))
        return node, [], False
    node_ok = True
    for slot, gpu_index in sorted(slot_gpu.items()):
        command = (
            f"CPA_STA_REUSE_OUTPUTS={shlex.quote(reuse_outputs)} "
            f"bash {PROJECT}/scripts/launch_cpa_dynamic_generation.sh "
            f"{shlex.quote(node)} {slot} 1 {shlex.quote(str(launch_root))} {gpu_index}"
        )
        try:
            result = ssh(node, command, timeout=120)
        except subprocess.TimeoutExpired:
            # A saturated GPU node can accept the tmux launch and then delay
            # the SSH channel teardown.  Isolate that timeout to this slot;
            # the live-session probe below is authoritative and may still
            # recover the successfully created worker.
            node_ok = False
            emit("generation_launch_timed_out", node=node, slot=slot,
                 generation=generation_root.name)
            continue
        if result.returncode != 0:
            node_ok = False
            emit("generation_launch_failed", node=node, slot=slot,
                 generation=generation_root.name, returncode=result.returncode,
                 stderr=result.stderr[-1000:])
    # tmux creation only proves that the wrapper was accepted.  Validate all
    # launched panes after their argument and model initialization checks.
    time.sleep(3)
    live_result = ssh(
        node,
        "tmux list-sessions -F '#{session_name}' 2>/dev/null | "
        "sed -nE 's/^vqa-cpa-dynamic-v3-slot-([0-9]+)$/\\1/p'",
        timeout=30,
    )
    live_slots = {int(value) for value in live_result.stdout.splitlines() if value.isdigit()}
    survived = []
    for slot in slot_gpu:
        if slot not in live_slots:
            node_ok = False
            emit("generation_slot_died_at_startup", node=node, slot=slot,
                 generation=generation_root.name)
        else:
            survived.append(slot)
    return node, survived, node_ok


def build_generation(
    root: Path, available: Path, completed: Path, excluded: Path, slots: int,
) -> tuple[Path, dict]:
    generation = time.strftime("g%Y%m%d-%H%M%S")
    generation_root = root / "generations" / generation
    partitions = generation_root / "partitions"
    command = [
        "/usr/bin/python3", str(PROJECT / "scripts/build_cpa_dynamic_generation.py"),
        "--available-sta-uids", str(available),
        "--completed-cpa-uids", str(completed),
        "--excluded-cpa-uids", str(excluded),
        "--output-dir", str(partitions), "--slots", str(slots),
        # Small 64-UID generations made every GPU process repeatedly rescan
        # the 361k-view catalog and reload SAM3/CoTracker.  A longer finite
        # lease amortizes startup while the controller still refills any slot
        # independently when it finishes.
        "--max-uids-per-slot", os.environ.get("CPA_GENERATION_UIDS_PER_SLOT", "512"),
    ]
    master_paths = sorted(glob.glob(str(MASTER_GLOB)))
    for path in master_paths:
        command.extend(("--master-uids", path))
    if not master_paths:
        raise RuntimeError(f"cpa_master_uid_files_missing:{MASTER_GLOB}")
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"generation_build_failed:{result.stderr[-1000:]}")
    manifest = json.loads((partitions / "manifest.json").read_text())
    return generation_root, manifest


def load_assignments(path: Path) -> dict[int, str]:
    return {int(key): value for key, value in json.loads(path.read_text()).items()}


def save_assignments(path: Path, assignments: dict[int, str]) -> None:
    atomic_text(path, json.dumps(assignments, sort_keys=True, indent=2) + "\n")


def choose_tail_donor(
    assignments: dict[int, str],
    active: set[int],
    available: set[str],
    completed: set[str],
    minimum_remaining: int,
) -> tuple[int, int] | None:
    """Choose one live finite lease whose unfinished tail is worth splitting."""
    candidates: list[tuple[int, int]] = []
    for slot in active:
        path_value = assignments.get(slot)
        if not path_value:
            continue
        assigned_path = Path(path_value)
        if not assigned_path.is_file():
            continue
        remaining = len((read_lines(assigned_path) & available) - completed)
        if remaining >= minimum_remaining:
            candidates.append((remaining, slot))
    if not candidates:
        return None
    remaining, slot = max(candidates)
    return slot, remaining


def node_for_slot(slot: int) -> str:
    try:
        return CPA_SLOT_TO_NODE_GPU[slot][0]
    except KeyError as error:
        raise ValueError(f"cpa_slot_outside_layout:{slot}") from error


def choose_generation_slots(idle: list[int], pending: int) -> list[int]:
    """Spread a bounded number of non-trivial leases across the idle fleet."""
    if pending <= 0 or not idle:
        return []
    target = max(1, int(os.environ.get("CPA_TARGET_UIDS_PER_SLOT", "8")))
    count = min(len(idle), max(1, math.ceil(pending / target)))
    # ``idle`` is grouped by node in the static slot layout.  Even spacing
    # prevents a small generation from repeatedly landing on only the first
    # GPU host while retaining deterministic assignment.
    return [idle[index * len(idle) // count] for index in range(count)]


def maybe_rebalance_tail(
    root: Path,
    assignments: dict[int, str],
    active: set[int],
    idle: list[int],
    available: set[str],
    completed: set[str],
) -> bool:
    """Release one long finite lease so its remainder can fill idle GPUs."""
    if os.environ.get("CPA_REBALANCE_ACTIVE_TAIL", "1") != "1":
        return False
    # Two idle GPU slots already represent enough lost tail capacity to repay
    # one donor restart.  Waiting for four left long finite leases stranded on
    # 146/148 slots for hours near the end of a generation.
    minimum_idle = int(os.environ.get("CPA_REBALANCE_MIN_IDLE_SLOTS", "2"))
    minimum_remaining = int(os.environ.get("CPA_REBALANCE_MIN_REMAINING_UIDS", "256"))
    cooldown = float(os.environ.get("CPA_REBALANCE_COOLDOWN_SECONDS", "300"))
    if len(idle) < minimum_idle:
        return False
    marker = root / "last-tail-rebalance-unix.txt"
    try:
        last_rebalance = float(marker.read_text().strip())
    except (OSError, ValueError):
        last_rebalance = 0.0
    if time.time() - last_rebalance < cooldown:
        return False
    donor = choose_tail_donor(
        assignments, active, available, completed, minimum_remaining,
    )
    if donor is None:
        return False
    slot, remaining = donor
    node = node_for_slot(slot)
    session = f"vqa-cpa-dynamic-v3-slot-{slot:02d}"
    result = ssh(node, f"tmux kill-session -t {shlex.quote(session)}", timeout=30)
    if result.returncode != 0:
        emit(
            "tail_rebalance_stop_failed", node=node, slot=slot,
            remaining_uids=remaining, stderr=result.stderr[-500:],
        )
        return False
    atomic_text(marker, f"{time.time()}\n")
    emit(
        "tail_rebalance_released", node=node, slot=slot,
        remaining_uids=remaining, idle_slots_before=idle,
    )
    return True


def controller_cycle(root: Path, max_attempts: int) -> None:
    available_path = root / "available-sta-uids.txt"
    completed_path = root / "completed-cpa-uids.txt"
    attempts_path = root / "attempts.json"
    assignments_path = root / "slot-assignments.json"
    active_slots_path = root / "active-slots.json"
    excluded_path = root / "excluded-cpa-uids.txt"
    locations_path = root / "sta-location-index.json"

    # Rebuild from the last verified durable seed on every cycle.  The prior
    # controller scanned historical logs, including stage-written records that
    # were acknowledged to a local WAL but later left an empty COS placeholder.
    # Carrying that cache forward made CPA retry invalid upstream records.
    # Every UID admitted by this controller comes from the collector's
    # cloud-acknowledged `published` ledger.  Preserve that monotonic
    # high-water mark across cycles: rebuilding only from the old seed made a
    # transient collector timeout remove newly published STA records from the
    # eligible set until that same overloaded node answered again.
    available = (
        read_lines(SEED_ROOT / "available-sta-uids.txt")
        | read_lines(available_path)
    )
    # The immutable audit seed is the floor.  Preserve the controller's
    # monotonic high-water mark too: future additions come only from the
    # durability-gated incremental collector, so resetting to the older seed
    # would reschedule already published CPA records after every restart.
    completed = (
        read_lines(SEED_ROOT / "completed-cpa-uids.txt")
        | read_lines(completed_path)
    )
    # The active collector reads only the spool's `published` ledger.  A UID is
    # inserted there in the same transaction that removes its successfully
    # cloud-acknowledged outbox job, so local WAL acceptance alone is never
    # enough to make STA eligible for CPA.  Keep the historical environment
    # flag name for deployment compatibility.
    if os.environ.get("CPA_INCLUDE_UNAUDITED_LIVE_STA") == "1":
        try:
            locations = json.loads(locations_path.read_text(encoding="utf-8"))
            if not isinstance(locations, dict):
                locations = {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            locations = {}
        def collect_live_sta(node: str) -> tuple[set[str], dict[str, str]]:
            node_locations: dict[str, str] = {}
            return collect_stage_uids(node, "sta", node_locations), node_locations

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(STA_NODES), thread_name_prefix="sta-ready-collect",
        ) as executor:
            future_to_node = {
                executor.submit(collect_live_sta, node): node for node in STA_NODES
            }
            for future in concurrent.futures.as_completed(future_to_node):
                try:
                    node_uids, node_locations = future.result()
                    available.update(node_uids)
                    locations.update(node_locations)
                except Exception as error:
                    emit(
                        "collection_failed", node=future_to_node[future], task="sta",
                        error_type=type(error).__name__, error=str(error),
                    )
        atomic_text(
            locations_path,
            json.dumps(locations, sort_keys=True, separators=(",", ":")) + "\n",
        )
    elif not locations_path.exists():
        atomic_text(locations_path, "{}\n")
    # Node-local ledgers are independent.  A slow tmux/FUSE namespace on one
    # GPU host must not serialize or abort completion collection for the rest.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(CPA_NODES), thread_name_prefix="cpa-ready-collect",
    ) as executor:
        future_to_node = {
            executor.submit(collect_stage_uids, node, "cpa"): node
            for node in CPA_NODES
        }
        for future in concurrent.futures.as_completed(future_to_node):
            try:
                completed.update(future.result())
            except Exception as error:
                emit(
                    "collection_failed", node=future_to_node[future], task="cpa",
                    error_type=type(error).__name__, error=str(error),
                )
    write_lines(available_path, available)
    write_lines(completed_path, completed)

    assignments = load_assignments(assignments_path)
    active, per_node = active_cpa_slots()
    previous_active = load_active_slots(active_slots_path)
    attempts = load_attempts(attempts_path)
    attempts = {
        uid: count for uid, count in attempts.items()
        if uid.split(":view=", 1)[0] in available
    }
    # An assignment is not a failed attempt.  Count it only after the worker
    # that owned the finite lease has exited and the durability-gated
    # completion collector still cannot see the UID.  The old launch-time
    # counter quarantined healthy records after controller restarts or a
    # deliberate capacity re-shard.
    attempts = record_finished_attempts(
        attempts, assignments, previous_active - active, available, completed,
    )
    save_attempts(attempts_path, attempts)
    quarantined = {uid for uid, count in attempts.items() if count >= max_attempts and uid not in completed}
    active_uids: set[str] = set()
    for slot in active:
        assigned_path = Path(assignments[slot])
        if not assigned_path.exists():
            raise RuntimeError(f"active_slot_assignment_missing:slot={slot}:path={assigned_path}")
        active_uids.update(read_lines(assigned_path))
    excluded = quarantined | active_uids
    write_lines(excluded_path, excluded)
    idle = sorted(set(range(CPA_TOTAL_SLOTS)) - active)
    emit("fleet_snapshot", available_sta=len(available), completed_cpa=len(completed),
         quarantined_cpa=len(quarantined), active_assigned_uids=len(active_uids),
         active_slots=len(active), idle_slots=idle, slots_by_node=per_node)
    if not idle:
        save_active_slots(active_slots_path, active)
        return

    master_paths = sorted(glob.glob(str(MASTER_GLOB)))
    if not master_paths:
        raise RuntimeError(f"cpa_master_uid_files_missing:{MASTER_GLOB}")
    master = set()
    for path in master_paths:
        master.update(read_lines(Path(path)))
    pending = sum(
        uid not in completed
        and uid not in excluded
        and uid.split(":view=", 1)[0] in available
        for uid in master
    )
    if pending == 0:
        save_active_slots(active_slots_path, active)
        if maybe_rebalance_tail(
            root, assignments, active, idle, available, completed,
        ):
            return
        emit("generation_empty", generation="none")
        return

    generation_slots = choose_generation_slots(idle, pending)
    generation_root, manifest = build_generation(
        root, available_path, completed_path, excluded_path,
        len(generation_slots),
    )

    # The builder emits dense slots 0..N-1. Map those files onto the actual
    # idle GPU slot numbers without disturbing active assignments.
    dense = generation_root / "partitions"
    mapped = generation_root / "mapped" / "partitions"
    mapped.mkdir(parents=True)
    slot_counts: dict[int, int] = {}
    for dense_slot, actual_slot in enumerate(generation_slots):
        source = dense / f"slot-{dense_slot:02d}.txt"
        target = mapped / f"slot-{actual_slot:02d}.txt"
        shutil.copyfile(source, target)
        slot_counts[actual_slot] = len(read_lines(target))
    mapped_manifest = {
        **manifest, "actual_slots": generation_slots,
        "actual_slot_counts": slot_counts,
    }
    atomic_text(mapped / "manifest.json", json.dumps(mapped_manifest, indent=2) + "\n")
    shutil.copyfile(SEED_ROOT / "partitions" / "row-groups.txt", mapped / "row-groups.txt")
    shutil.copyfile(completed_path, generation_root / "mapped" / "completed-cpa-uids.txt")
    shutil.copyfile(locations_path, generation_root / "mapped" / "sta-location-index.json")

    remote_root = Path("/run/ti/cpa-dynamic-controller/generations") / generation_root.name
    launched: list[str] = []
    launched_slots: set[int] = set()
    node_jobs = []
    for node in CPA_NODES:
        slot_gpu = {
            slot: CPA_SLOT_TO_NODE_GPU[slot][1]
            for slot in generation_slots
            if slot in CPA_NODE_SLOTS[node] and slot_counts[slot] > 0
        }
        if not slot_gpu:
            continue
        node_jobs.append((node, slot_gpu))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(node_jobs)), thread_name_prefix="cpa-node-launch",
    ) as executor:
        futures = [
            executor.submit(
                deploy_and_launch_node, generation_root, remote_root,
                node, slot_gpu,
            )
            for node, slot_gpu in node_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                node, live_slots, node_ok = future.result()
            except Exception as error:
                # One node must never abort persistence of launch results from
                # every other node in the same generation.
                emit("generation_node_job_failed", error_type=type(error).__name__,
                     error=str(error))
                continue
            for slot in live_slots:
                launched_slots.add(slot)
                assigned = generation_root / "mapped" / "partitions" / f"slot-{slot:02d}.txt"
                assignments[slot] = str(assigned)
            if node_ok:
                launched.append(node)
    save_assignments(assignments_path, assignments)
    save_active_slots(active_slots_path, active | launched_slots)
    emit("generation_launched", generation=generation_root.name, pending=pending,
         launched_nodes=launched, manifest=manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/run/ti/cpa-dynamic-controller"))
    parser.add_argument("--interval-seconds", type=int, default=300)
    # A finite lease can end while an upstream STA pack is still being
    # materialized on the GPU node.  Four rapid leases were enough to
    # quarantine otherwise valid records during cache propagation.  Keep the
    # safety valve, but give transient upstream publication time to converge.
    parser.add_argument(
        "--max-attempts", type=int,
        default=int(os.environ.get("CPA_MAX_ATTEMPTS", "20")),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    available_path = args.root / "available-sta-uids.txt"
    completed_path = args.root / "completed-cpa-uids.txt"
    attempts_path = args.root / "attempts.json"
    assignments_path = args.root / "slot-assignments.json"
    if not available_path.exists():
        shutil.copyfile(SEED_ROOT / "available-sta-uids.txt", available_path)
    if not completed_path.exists():
        shutil.copyfile(SEED_ROOT / "completed-cpa-uids.txt", completed_path)
    if not attempts_path.exists():
        # Attempts belong to this controller generation.  Importing old static
        # partition membership as an attempt prematurely quarantines clean
        # records after a single transient failure.
        initial: dict[str, int] = {}
        save_attempts(attempts_path, initial)
        emit("attempt_ledger_initialized", records=len(initial))
    if not assignments_path.exists():
        save_assignments(
            assignments_path,
            {
                slot: str(SEED_ROOT / "partitions" / f"slot-{slot:02d}.txt")
                for slot in range(CPA_TOTAL_SLOTS)
            },
        )

    while True:
        started = time.monotonic()
        try:
            controller_cycle(args.root, args.max_attempts)
        except Exception as exc:  # keep the controller alive across node outages
            emit("controller_cycle_failed", error_type=type(exc).__name__, error=str(exc))
        if args.once:
            return
        time.sleep(max(1, args.interval_seconds - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
