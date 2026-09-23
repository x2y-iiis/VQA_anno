#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 TASK HOST SESSION COPIES" >&2
  exit 2
fi
task=$1
host=$2
session=$3
copies=$4
[[ ${task} == grd || ${task} == sta ]] || { echo "invalid_task:${task}" >&2; exit 2; }
[[ ${copies} =~ ^[2-9]$ ]] || { echo "copies_must_be_2_to_9:${copies}" >&2; exit 2; }

project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
catalog=/run/ti/cosmos3-physical-catalog/catalog/samples.parquet
timestamp="$(date +%Y%m%d-%H%M%S)-$$"
run_root="/run/ti/fleet-fanout-${task}-${host//_/-}-${timestamp}"
mkdir -p "${run_root}"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

scp_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if timeout 45 scp -q "${ssh_options[@]}" "$@"; then
      return 0
    fi
    echo "scp_retry:attempt=${attempt}:args=$*" >&2
    sleep $((attempt * 2))
  done
  echo "scp_failed_after_retries:args=$*" >&2
  return 1
}

ssh_put_retry() {
  local source=$1
  local destination=$2
  local attempt payload size digest object_uri
  size=$(stat -c %s "${source}")
  if (( size > 8192 )); then
    digest=$(sha256sum "${source}" | awk '{print $1}')
    object_uri="cos://datasets-1409717487/video-cleaning/_runtime/vqa-fleet-control/fanout/${host}/${timestamp}/${digest}-$(basename "${destination}")"
    /root/coscli cp "${source}" "${object_uri}" --thread-num 4 --disable-log
    for attempt in 1 2 3; do
      if timeout 60 ssh "${ssh_options[@]}" "root@${host}" \
        "/root/coscli cp '${object_uri}' '${destination}.partial' --thread-num 4 --disable-log && mv -f '${destination}.partial' '${destination}'"; then
        return 0
      fi
      echo "cos_put_retry:attempt=${attempt}:source=${source}:destination=${destination}" >&2
      sleep $((attempt * 2))
    done
    echo "cos_put_failed_after_retries:source=${source}:destination=${destination}" >&2
    return 1
  fi
  payload=$(gzip -c "${source}" | base64 -w0)
  for attempt in 1 2 3 4 5; do
    if timeout 45 ssh "${ssh_options[@]}" "root@${host}" \
      "umask 022; printf '%s' '${payload}' | base64 -d | gzip -d > '${destination}.partial'; mv -f '${destination}.partial' '${destination}'"; then
      return 0
    fi
    echo "ssh_put_retry:attempt=${attempt}:source=${source}:destination=${destination}" >&2
    sleep $((attempt * 2))
  done
  echo "ssh_put_failed_after_retries:source=${source}:destination=${destination}" >&2
  return 1
}

sync_file_if_needed() {
  local source=$1
  local destination=$2
  local expected actual staged
  expected=$(sha256sum "${source}" | awk '{print $1}')
  actual=$(ssh "${ssh_options[@]}" "root@${host}" \
    "sha256sum '$destination' 2>/dev/null | cut -d ' ' -f 1" 2>/dev/null || true)
  if [[ ${actual} == "${expected}" ]]; then
    return 0
  fi
  staged="/run/ti/.fanout-sync-$(basename "${destination}")-${timestamp}"
  ssh_put_retry "${source}" "${staged}"
  ssh "${ssh_options[@]}" "root@${host}" \
    "install -m 0755 '$staged' '$destination' && rm -f '$staged'"
}

pane_command=$(ssh "${ssh_options[@]}" "root@${host}" \
  "tmux list-panes -t '=$session' -F '#{pane_start_command}' 2>/dev/null | head -1")
[[ -n ${pane_command} ]] || { echo "source_session_missing:${host}:${session}" >&2; exit 3; }
metadata=$(printf '%s' "${pane_command}" | /usr/bin/python3 "${project}/scripts/parse_fleet_worker_command.py")
parsed_task=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["task"])' <<<"${metadata}")
[[ ${parsed_task} == "${task}" ]] || { echo "source_task_mismatch:${parsed_task}" >&2; exit 3; }
worker_id=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["worker_id"])' <<<"${metadata}")
uid_path=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["uid_path"])' <<<"${metadata}")
output_root=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["output_root"])' <<<"${metadata}")
resume_root=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["resume_root"])' <<<"${metadata}")
scratch=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["scratch"])' <<<"${metadata}")

python3 - "${run_root}/metadata.json" "${task}" "${host}" "${session}" \
  "${worker_id}" "${uid_path}" "${output_root}" "${resume_root}" \
  "${scratch}" "${copies}" <<'PY'
import json
from pathlib import Path
import sys

(
    destination,
    task,
    host,
    source_session,
    source_worker_id,
    source_uid_path,
    output_root,
    resume_root,
    scratch,
    copies,
) = sys.argv[1:]
Path(destination).write_text(
    json.dumps(
        {
            "schema": "fleet-worker-fanout/v1",
            "task": task,
            "host": host,
            "source_session": source_session,
            "source_worker_id": source_worker_id,
            "source_uid_path": source_uid_path,
            "output_root": output_root,
            "resume_root": resume_root,
            "scratch": scratch,
            "copies": int(copies),
            "status": "preparing",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

ssh "${ssh_options[@]}" "root@${host}" "/usr/bin/python3 - '$scratch'" \
  >"${run_root}/published.txt" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys
scratch = Path(sys.argv[1])
status = json.loads((scratch / 'runtime-state/request-parallel-status.json').read_text())
database = Path(status['final_registration_outbox']['spool_root']) / 'outbox.sqlite3'
connection = sqlite3.connect(database)
try:
    rows = connection.execute('SELECT DISTINCT input_record_uid FROM published').fetchall()
finally:
    connection.close()
for uid in sorted(str(row[0]) for row in rows if row and row[0]):
    print(uid)
PY
scp_retry "root@${host}:${uid_path}" "${run_root}/source.txt"
/usr/bin/python3 - "${run_root}/source.txt" "${run_root}/published.txt" "${run_root}/remaining.txt" <<'PY'
from pathlib import Path
import sys
source = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
published = {line.strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()}
remaining = [uid for uid in source if uid not in published]
if not remaining:
    raise SystemExit('source_has_no_remaining_uids')
Path(sys.argv[3]).write_text(''.join(f'{uid}\n' for uid in remaining))
print(f'fanout_scope source={len(source)} published={len(set(source) & published)} remaining={len(remaining)}')
PY

owners=()
for ((index=0; index<copies; index++)); do owners+=("p${index}"); done
/usr/bin/python3 "${project}/scripts/split_fleet_uid_partition.py" \
  --catalog "${catalog}" --input-uids "${run_root}/remaining.txt" \
  --output-dir "${run_root}/partitions" --owners "${owners[@]}"

sync_file_if_needed "${project}/scripts/launch_${task}_production_worker.sh" \
  "${project}/scripts/launch_${task}_production_worker.sh"
sync_file_if_needed "${project}/scripts/run_grd_sta_fleet_worker.sh" \
  "${project}/scripts/run_grd_sta_fleet_worker.sh"
sync_file_if_needed "${project}/scripts/terminate_fleet_worker_scope.py" \
  "${project}/scripts/terminate_fleet_worker_scope.py"
ssh "${ssh_options[@]}" "root@${host}" 'mkdir -p /run/ti/fleet-fanout'
for ((index=0; index<copies; index++)); do
  ssh_put_retry "${run_root}/partitions/p${index}.txt" \
    "/run/ti/fleet-fanout/${task}-${timestamp}-p${index}.txt"
done

frozen=0
launched=0
child_started=0
recover() {
  local code=$?
  trap - EXIT
  if (( frozen && ! launched )); then
    local recovery_index recovery_uid recovery_session
    for ((recovery_index=0; recovery_index<child_started; recovery_index++)); do
      recovery_uid="/run/ti/fleet-fanout/${task}-${timestamp}-p${recovery_index}.txt"
      recovery_session="${session}-p${recovery_index}"
      ssh "${ssh_options[@]}" "root@${host}" \
        "tmux kill-session -t '=${recovery_session}' 2>/dev/null || true; /usr/bin/python3 '${project}/scripts/terminate_fleet_worker_scope.py' '${recovery_uid}'" || true
    done
    if [[ ${task} == grd ]]; then
      ssh "${ssh_options[@]}" "root@${host}" \
        "bash '${project}/scripts/launch_grd_production_worker.sh' '${worker_id}' '${uid_path}' '${output_root}' '${resume_root}' large '${session}'" || true
    else
      ssh "${ssh_options[@]}" "root@${host}" \
        "bash '${project}/scripts/launch_sta_production_worker.sh' '${worker_id}' '${uid_path}' '${output_root}' '${resume_root}' '${session}'" || true
    fi
  fi
  exit "${code}"
}
trap recover EXIT
ssh "${ssh_options[@]}" "root@${host}" \
  "tmux kill-session -t '=$session' 2>/dev/null || true; /usr/bin/python3 '${project}/scripts/terminate_fleet_worker_scope.py' '$uid_path'; sleep 1"
frozen=1

for ((index=0; index<copies; index++)); do
  child_uid="/run/ti/fleet-fanout/${task}-${timestamp}-p${index}.txt"
  child_worker="${worker_id}-p${index}"
  child_session="${session}-p${index}"
  child_resume="${resume_root}-p${index}"
  # The source process may have thousands of LAS tasks already submitted but
  # not yet represented by final records.  Preserve those immutable task
  # receipts as a read-only fallback for every child; otherwise fanout would
  # submit the same model work again after terminating the source process.
  ssh "${ssh_options[@]}" "root@${host}" \
    "mkdir -p '${child_resume}/_state'; if [[ -d '${scratch}/las-operator-tasks' && ! -e '${child_resume}/_state/las-operator-tasks' ]]; then ln -s '${scratch}/las-operator-tasks' '${child_resume}/_state/las-operator-tasks'; fi"
  if [[ ${task} == grd ]]; then
    ssh "${ssh_options[@]}" "root@${host}" \
      "mkdir -p '${child_resume}'; bash '${project}/scripts/launch_grd_production_worker.sh' '${child_worker}' '${child_uid}' '${output_root}' '${child_resume}' large '${child_session}'"
  else
    ssh "${ssh_options[@]}" "root@${host}" \
      "mkdir -p '${child_resume}'; bash '${project}/scripts/launch_sta_production_worker.sh' '${child_worker}' '${child_uid}' '${output_root}' '${child_resume}' '${child_session}'"
  fi
  child_started=$((child_started + 1))
done
sleep 5
live=$(ssh "${ssh_options[@]}" "root@${host}" \
  "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^${session}-p[0-9]$' || true")
[[ ${live} -eq ${copies} ]] || { echo "fanout_live_session_mismatch:${host}:expected=${copies}:actual=${live}" >&2; exit 4; }
launched=1
python3 - "${run_root}/metadata.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_text())
value["status"] = "launched"
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
trap - EXIT
echo "fleet_worker_fanout_complete:task=${task}:host=${host}:copies=${copies}:run_root=${run_root}"
