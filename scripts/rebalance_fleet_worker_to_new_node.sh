#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 TASK DONOR_HOST DONOR_SESSION NEW_HOST [NEW_GRD_MODEL]" >&2
  exit 2
fi
task=$1
donor=$2
donor_session=$3
new_host=$4
new_grd_model=${5:-}
[[ ${task} == grd || ${task} == sta ]] || { echo "invalid_task:${task}" >&2; exit 2; }
if [[ -n ${new_grd_model} ]]; then
  [[ ${task} == grd ]] || { echo "new_grd_model_only_valid_for_grd" >&2; exit 2; }
  [[ ${new_grd_model} == doubao-seed-2-1-pro-260628 \
      || ${new_grd_model} == doubao-seed-2-1-turbo-260628 ]] || {
    echo "invalid_new_grd_model:${new_grd_model}" >&2
    exit 2
  }
fi

project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
catalog=/run/ti/cosmos3-physical-catalog/catalog/samples.parquet
timestamp=$(date +%Y%m%d-%H%M%S)
run_root="/run/ti/fleet-rebalance-${task}-${donor//_/-}-to-${new_host//_/-}-${timestamp}"
mkdir -p "${run_root}"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

remote() {
  local host=$1
  shift
  if [[ ${host} == local ]]; then bash -lc "$*"; else ssh "${ssh_options[@]}" "root@${host}" "$@"; fi
}
copy_from() {
  local host=$1 source=$2 target=$3
  if [[ ${host} == local ]]; then
    cp "${source}" "${target}"
  else
    # These are small ownership manifests.  Streaming them over an already
    # authenticated SSH command avoids SCP subsystem stalls on saturated
    # annotation nodes.
    timeout 30 ssh "${ssh_options[@]}" "root@${host}" \
      "gzip -c -- '${source}'" | gzip -dc >"${target}"
  fi
}
copy_to() {
  local source=$1 host=$2 target=$3
  if [[ ${host} == local ]]; then
    cp "${source}" "${target}"
  else
    # Do not use SCP or an SSH stdin stream for tiny manifests: on saturated
    # nodes those data channels can remain blocked even though command
    # channels are responsive.  A compressed manifest fits safely below the
    # remote command-size limit and completes in one command channel.
    local payload
    payload=$(gzip -c -- "${source}" | base64 -w0)
    (( ${#payload} < 120000 )) || {
      echo "compressed_manifest_too_large:${source}:${#payload}" >&2
      return 4
    }
    timeout 30 ssh "${ssh_options[@]}" "root@${host}" \
      "python3 -c \"import base64,gzip; open('${target}','wb').write(gzip.decompress(base64.b64decode('${payload}')))\""
  fi
}

new_ready=$(remote "${new_host}" "test -s /run/ti/vqa-subtask-index-20260916/.complete-319355-records && echo ready")
[[ ${new_ready} == ready ]] || { echo "new_node_not_provisioned:${new_host}" >&2; exit 3; }

pane_command=$(remote "${donor}" \
  "tmux list-panes -t '$donor_session' -F '#{pane_start_command}' 2>/dev/null | head -1")
[[ -n ${pane_command} ]] || { echo "donor_session_missing:${donor}:${donor_session}" >&2; exit 3; }
metadata=$(printf '%s' "${pane_command}" | /usr/bin/python3 "${project}/scripts/parse_fleet_worker_command.py")
parsed_task=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["task"])' <<<"${metadata}")
[[ ${parsed_task} == "${task}" ]] || { echo "donor_task_mismatch:${parsed_task}" >&2; exit 3; }
worker_id=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["worker_id"])' <<<"${metadata}")
uid_path=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["uid_path"])' <<<"${metadata}")
output_root=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["output_root"])' <<<"${metadata}")
resume_root=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["resume_root"])' <<<"${metadata}")
scratch=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["scratch"])' <<<"${metadata}")
donor_grd_model=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("grd_model", ""))' <<<"${metadata}")
donor_tos_fetch=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tos_upload_via_cos_fetch", ""))' <<<"${metadata}")
[[ ${donor_tos_fetch} == 0 || ${donor_tos_fetch} == 1 ]] || donor_tos_fetch=1

remote "${donor}" "/usr/bin/python3 - '$scratch'" >"${run_root}/published.txt" <<'PY'
import json
from pathlib import Path
import sqlite3
import sys

scratch = Path(sys.argv[1])
status = json.loads((scratch / 'runtime-state/request-parallel-status.json').read_text())
database = Path(status['final_registration_outbox']['spool_root']) / 'outbox.sqlite3'
connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True, timeout=2)
try:
    tables = {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if 'published' in tables:
        query = 'SELECT DISTINCT input_record_uid FROM published'
    elif {'accepted', 'jobs'} <= tables:
        query = (
            'SELECT DISTINCT a.input_record_uid FROM accepted a '
            'LEFT JOIN jobs j ON j.path=a.path WHERE j.path IS NULL'
        )
    else:
        raise SystemExit('unsupported_outbox_schema')
    rows = connection.execute(query).fetchall()
finally:
    connection.close()
for row in sorted(str(value[0]) for value in rows if value and value[0]):
    print(row)
PY
copy_from "${donor}" "${uid_path}" "${run_root}/source.txt"

/usr/bin/python3 - "${run_root}/source.txt" "${run_root}/published.txt" "${run_root}/remaining.txt" <<'PY'
from pathlib import Path
import sys
source = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
published = {line.strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()}
remaining = [uid for uid in source if uid not in published]
if not remaining:
    raise SystemExit('donor_has_no_remaining_uids')
Path(sys.argv[3]).write_text(''.join(f'{uid}\n' for uid in remaining))
print(f'donor_scope source={len(source)} published={len(set(source) & published)} remaining={len(remaining)}')
PY

/usr/bin/python3 "${project}/scripts/split_fleet_uid_partition.py" \
  --catalog "${catalog}" --input-uids "${run_root}/remaining.txt" \
  --output-dir "${run_root}/partitions" --owners donor new

if [[ ${task} == grd ]]; then
  memory_bytes=$(remote "${donor}" "awk '/MemTotal:/ {printf \"%.0f\\n\", \$2 * 1024}' /proc/meminfo")
  if (( memory_bytes >= 193273528320 )); then donor_profile=large
  elif (( memory_bytes >= 118111600640 )); then donor_profile=medium
  else donor_profile=shared
  fi
  new_memory_bytes=$(remote "${new_host}" "awk '/MemTotal:/ {printf \"%.0f\\n\", \$2 * 1024}' /proc/meminfo")
  if (( new_memory_bytes >= 193273528320 )); then new_profile=large
  elif (( new_memory_bytes >= 118111600640 )); then new_profile=medium
  else new_profile=shared
  fi
fi

for host in "${donor}" "${new_host}"; do
  [[ ${host} == local ]] && continue
  # The repository lives on a FUSE-backed mount on production nodes.  Writing
  # into it while thousands of requests are active can block indefinitely.
  # Provisioning already installs these launchers, so rebalance only verifies
  # them and keeps all mutable manifests under /run/ti.
  remote "${host}" "test -x '${project}/scripts/launch_${task}_production_worker.sh' -a -x '${project}/scripts/run_grd_sta_fleet_worker.sh'"
done

ownership_frozen=0
donor_restarted=0
recover_donor() {
  local code=$?
  trap - EXIT
  if (( ownership_frozen && ! donor_restarted )); then
    if [[ ${task} == grd ]]; then
      remote "${donor}" "GRD_MODEL='${donor_grd_model}' TOS_UPLOAD_VIA_COS_FETCH='${donor_tos_fetch}' bash '${project}/scripts/launch_grd_production_worker.sh' '${worker_id}' '${uid_path}' '${output_root}' '${resume_root}' '${donor_profile}' '${donor_session}'" || true
    else
      remote "${donor}" "bash '${project}/scripts/launch_sta_production_worker.sh' '${worker_id}' '${uid_path}' '${output_root}' '${resume_root}' '${donor_session}'" || true
    fi
  fi
  exit "${code}"
}
trap recover_donor EXIT

# Freeze ownership only after every potentially expensive catalog/snapshot
# operation has succeeded. A failure before the replacement donor starts
# automatically restores the original full-scope worker.
remote "${donor}" "pane_pid=\$(tmux list-panes -t '$donor_session' -F '#{pane_pid}' 2>/dev/null | head -1); if [[ -n \$pane_pid ]]; then kill -TERM -- -\$pane_pid 2>/dev/null || true; fi; tmux kill-session -t '$donor_session' 2>/dev/null || true; sleep 3"
ownership_frozen=1

donor_uid="/run/ti/fleet-rebalance/${task}-${timestamp}-donor.txt"
new_uid="/run/ti/fleet-rebalance/${task}-${timestamp}-${new_host}.txt"
remote "${donor}" "mkdir -p /run/ti/fleet-rebalance"
remote "${new_host}" "mkdir -p /run/ti/fleet-rebalance"
copy_to "${run_root}/partitions/donor.txt" "${donor}" "${donor_uid}"
copy_to "${run_root}/partitions/new.txt" "${new_host}" "${new_uid}"

if [[ ${task} == grd ]]; then
  # Keep the worker id (and therefore its local task/checkpoint state) stable
  # while narrowing only the UID ownership manifest.
  remote "${donor}" "mkdir -p '${scratch}'; [[ '${donor_tos_fetch}' == 1 ]] && touch '${scratch}/use-cos-fetch' || true; GRD_MODEL='${donor_grd_model}' TOS_UPLOAD_VIA_COS_FETCH='${donor_tos_fetch}' bash '${project}/scripts/launch_grd_production_worker.sh' '${worker_id}' '${donor_uid}' '${output_root}' '${resume_root}' '${donor_profile}' '${donor_session}'"
  donor_restarted=1
  new_worker="grd-tail-${new_host}-${timestamp}"
  new_output="/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-grd-sta-fleet-20260922/${new_worker}"
  new_resume="/run/ti/vqa-grd-resume/${new_worker}"
  new_scratch="/run/ti/vqa-grd-sta-fleet/production-${new_worker}"
  new_session="vqa-grd-production-${new_host}"
  # Same-host fanout, or a host that already owns an expansion worker, must
  # use a unique session name; otherwise launching the new half silently
  # terminates an existing worker.
  if [[ ${donor} == "${new_host}" ]] \
      || remote "${new_host}" "tmux has-session -t '=${new_session}' 2>/dev/null"; then
    new_session="${new_session}-${timestamp}"
  fi
  if [[ ${donor} == "${new_host}" ]]; then
    checkpoint_migrator=/run/ti/migrate_grd_checkpoint_partition.py
    copy_to "${project}/scripts/migrate_grd_checkpoint_partition.py" \
      "${new_host}" "${checkpoint_migrator}"
    # Workers created from the local catalog keep checkpoints under scratch,
    # while older workers that still read the mounted input tree keep them in
    # OUTPUT_ROOT/_state.  Copy both layouts before the new process starts;
    # copy-missing semantics makes this safe when one layout is absent or a
    # previous fanout attempt already populated part of the destination.
    remote "${new_host}" "python3 '${checkpoint_migrator}' --uids '${new_uid}' --source-root '${scratch}/tmp/grd-window-checkpoints' --target-root '${new_scratch}/tmp/grd-window-checkpoints'"
    remote "${new_host}" "python3 '${checkpoint_migrator}' --uids '${new_uid}' --source-root '${output_root}/_state/grd-window-checkpoints' --target-root '${new_output}/_state/grd-window-checkpoints'"
    new_task_state_fallback="${scratch}/las-operator-tasks"
  else
    # Preserve finished per-frame work when ownership moves to another host.
    # The archive is streamed directly between hosts, so no large temporary
    # checkpoint bundle is stored on the coordinator or either node.
    checkpoint_streamer=/run/ti/stream_grd_checkpoint_partition.py
    migration_uids="/run/ti/fleet-rebalance/${task}-${timestamp}-migrate.txt"
    copy_to "${project}/scripts/stream_grd_checkpoint_partition.py" \
      "${donor}" "${checkpoint_streamer}"
    copy_to "${run_root}/partitions/new.txt" "${donor}" "${migration_uids}"
    remote "${new_host}" "mkdir -p '${new_scratch}/tmp/grd-window-checkpoints' '${new_output}/_state/grd-window-checkpoints'"
    timeout 600 ssh "${ssh_options[@]}" "root@${donor}" \
      "python3 '${checkpoint_streamer}' --uids '${migration_uids}' --source-root '${scratch}/tmp/grd-window-checkpoints'" \
      | timeout 600 ssh "${ssh_options[@]}" "root@${new_host}" \
          "tar -xf - -C '${new_scratch}/tmp/grd-window-checkpoints'"
    timeout 600 ssh "${ssh_options[@]}" "root@${donor}" \
      "python3 '${checkpoint_streamer}' --uids '${migration_uids}' --source-root '${output_root}/_state/grd-window-checkpoints'" \
      | timeout 600 ssh "${ssh_options[@]}" "root@${new_host}" \
          "tar -xf - -C '${new_output}/_state/grd-window-checkpoints'"
    new_task_state_fallback=""
  fi
  remote "${new_host}" "mkdir -p '${new_resume}' '${new_scratch}'; [[ '${donor_tos_fetch}' == 1 ]] && touch '${new_scratch}/use-cos-fetch' || true; GRD_MODEL='${new_grd_model}' TOS_UPLOAD_VIA_COS_FETCH='${donor_tos_fetch}' VQA_LAS_TASK_STATE_FALLBACK_ROOT='${new_task_state_fallback}' bash '${project}/scripts/launch_grd_production_worker.sh' '${new_worker}' '${new_uid}' '${new_output}' '${new_resume}' '${new_profile}' '${new_session}'"
else
  remote "${donor}" "bash '${project}/scripts/launch_sta_production_worker.sh' '${worker_id}-r' '${donor_uid}' '${output_root}' '${resume_root}' '${donor_session}'"
  donor_restarted=1
  new_output="/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-grd-sta-fleet-20260921/sta-expansion-${new_host}"
  remote "${new_host}" "mkdir -p /run/ti/vqa-resume/${new_host}; bash '${project}/scripts/launch_sta_production_worker.sh' '${new_host}-prod' '${new_uid}' '${new_output}' '/run/ti/vqa-resume/${new_host}' 'vqa-sta-production-${new_host}'"
  bash "${project}/scripts/configure_cpa_dynamic_fleet.sh" --add-sta "${new_host}"
fi

sleep 5
new_check_session="vqa-${task}-production-${new_host}"
[[ ${task} == grd ]] && new_check_session=${new_session}
for spec in "${donor}:${donor_session}" "${new_host}:${new_check_session}"; do
  host=${spec%%:*}; session=${spec#*:}
  state=$(remote "${host}" "tmux list-panes -t '$session' -F '#{pane_dead}|#{pane_current_command}' 2>/dev/null | head -1")
  [[ ${state} == 0\|* ]] || { echo "rebalanced_worker_failed:${host}:${session}:${state}" >&2; exit 4; }
done
trap - EXIT
echo "fleet_worker_rebalanced:task=${task}:donor=${donor}:new_host=${new_host}:run_root=${run_root}"
