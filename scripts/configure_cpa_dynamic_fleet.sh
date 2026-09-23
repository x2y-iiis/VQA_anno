#!/usr/bin/env bash
set -euo pipefail

project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
state_root=/run/ti/cpa-dynamic-controller
session=vqa-cpa-dynamic-controller
new_slots_path="${state_root}/new-slots-per-node.txt"
tail_extra_nodes_path="${state_root}/tail-extra-nodes.txt"
tail_extra_slots_path="${state_root}/tail-extra-slots-per-node.txt"
tail2_extra_nodes_path="${state_root}/tail2-extra-nodes.txt"
tail2_extra_slots_path="${state_root}/tail2-extra-slots-per-node.txt"
historical_cpa=(
  x2y-6_12 x2y-6_13 x2y-6_14 x2y-6_15 x2y-6_16
  x2y-6_17 x2y-6_18 x2y-5_1 x2y-5_2 local
)
historical_sta=(
  x2y-5 x2y-6_4 x2y-6_5 x2y-6_7 x2y-6_9 x2y-6_10 x2y-6_19 x2y-6_20
)
add_cpa=()
add_sta=()
while (($#)); do
  case "$1" in
    --add-cpa) [[ $# -ge 2 ]] || exit 2; add_cpa+=("$2"); shift 2;;
    --add-sta) [[ $# -ge 2 ]] || exit 2; add_sta+=("$2"); shift 2;;
    *) echo "unknown_argument:$1" >&2; exit 2;;
  esac
done

mkdir -p "${state_root}"
exec 9>"${state_root}/fleet-configuration.lock"
flock -x 9

# Expansion slot density is durable because changing it after nodes have been
# appended would remap slot IDs to different hosts.  Allow a lower density on
# CPU-sharing nodes, but persist the first selected value across restarts.
if [[ -s ${new_slots_path} ]]; then
  new_slots_per_node=$(<"${new_slots_path}")
  if [[ -n ${CPA_NEW_SLOTS_PER_NODE:-} && ${CPA_NEW_SLOTS_PER_NODE} != "${new_slots_per_node}" ]]; then
    echo "cpa_new_slots_layout_is_immutable:stored=${new_slots_per_node}:requested=${CPA_NEW_SLOTS_PER_NODE}" >&2
    exit 3
  fi
else
  new_slots_per_node=${CPA_NEW_SLOTS_PER_NODE:-24}
fi
[[ ${new_slots_per_node} =~ ^[0-9]+$ && ${new_slots_per_node} -ge 4 ]] || {
  echo "invalid_cpa_new_slots_per_node:${new_slots_per_node}" >&2
  exit 3
}

requested_tail_extra_nodes=${CPA_NEW_TAIL_EXTRA_NODES:-}
requested_tail_extra_slots=${CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE:-0}
if [[ -s ${tail_extra_nodes_path} ]]; then
  stored_tail_extra_nodes=$(tr '\n' ' ' <"${tail_extra_nodes_path}" | xargs)
  if [[ -n ${requested_tail_extra_nodes} ]]; then
    # Extending the list at its tail is safe: every old node keeps the same
    # durable slot range and only new ranges are appended. Reordering or
    # removing the stored prefix would remap live slots and is rejected.
    [[ " ${requested_tail_extra_nodes} " == " ${stored_tail_extra_nodes} "* ]] || {
      echo "cpa_tail_extra_nodes_prefix_is_immutable:stored=${stored_tail_extra_nodes}:requested=${requested_tail_extra_nodes}" >&2
      exit 3
    }
    tail_extra_nodes=${requested_tail_extra_nodes}
  else
    tail_extra_nodes=${stored_tail_extra_nodes}
  fi
else
  tail_extra_nodes=${requested_tail_extra_nodes}
fi
if [[ -s ${tail_extra_slots_path} ]]; then
  tail_extra_slots=$(<"${tail_extra_slots_path}")
  [[ ${requested_tail_extra_slots} == 0 || ${requested_tail_extra_slots} == "${tail_extra_slots}" ]] || {
    echo "cpa_tail_extra_slots_are_immutable:stored=${tail_extra_slots}:requested=${requested_tail_extra_slots}" >&2
    exit 3
  }
else
  tail_extra_slots=${requested_tail_extra_slots}
fi
[[ ${tail_extra_slots} =~ ^[0-9]+$ ]] || {
  echo "invalid_cpa_tail_extra_slots_per_node:${tail_extra_slots}" >&2
  exit 3
}

requested_tail2_extra_nodes=${CPA_NEW_TAIL2_EXTRA_NODES:-}
requested_tail2_extra_slots=${CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE:-0}
if [[ -s ${tail2_extra_nodes_path} ]]; then
  stored_tail2_extra_nodes=$(tr '\n' ' ' <"${tail2_extra_nodes_path}" | xargs)
  if [[ -n ${requested_tail2_extra_nodes} ]]; then
    [[ " ${requested_tail2_extra_nodes} " == " ${stored_tail2_extra_nodes} "* ]] || {
      echo "cpa_tail2_extra_nodes_prefix_is_immutable:stored=${stored_tail2_extra_nodes}:requested=${requested_tail2_extra_nodes}" >&2
      exit 3
    }
    tail2_extra_nodes=${requested_tail2_extra_nodes}
  else
    tail2_extra_nodes=${stored_tail2_extra_nodes}
  fi
else
  tail2_extra_nodes=${requested_tail2_extra_nodes}
fi
if [[ -s ${tail2_extra_slots_path} ]]; then
  tail2_extra_slots=$(<"${tail2_extra_slots_path}")
  [[ ${requested_tail2_extra_slots} == 0 || ${requested_tail2_extra_slots} == "${tail2_extra_slots}" ]] || {
    echo "cpa_tail2_extra_slots_are_immutable:stored=${tail2_extra_slots}:requested=${requested_tail2_extra_slots}" >&2
    exit 3
  }
else
  tail2_extra_slots=${requested_tail2_extra_slots}
fi
[[ ${tail2_extra_slots} =~ ^[0-9]+$ ]] || {
  echo "invalid_cpa_tail2_extra_slots_per_node:${tail2_extra_slots}" >&2
  exit 3
}

load_nodes() {
  local path=$1
  shift
  if [[ -s ${path} ]]; then
    mapfile -t loaded < <(sed '/^[[:space:]]*$/d' "${path}")
  else
    loaded=("$@")
  fi
}
append_unique() {
  local -n target=$1
  shift
  local value existing
  for value in "$@"; do
    for existing in "${target[@]}"; do
      [[ ${value} != "${existing}" ]] || { value=; break; }
    done
    [[ -z ${value} ]] || target+=("${value}")
  done
}
atomic_lines() {
  local path=$1
  shift
  local temporary="${path}.tmp.$$"
  printf '%s\n' "$@" >"${temporary}"
  mv "${temporary}" "${path}"
}

load_nodes "${state_root}/cpa-nodes.txt" "${historical_cpa[@]}"
cpa_nodes=("${loaded[@]}")
load_nodes "${state_root}/sta-nodes.txt" "${historical_sta[@]}"
sta_nodes=("${loaded[@]}")
append_unique cpa_nodes "${add_cpa[@]}"
append_unique sta_nodes "${add_sta[@]}"

read -r -a tail_extra_node_array <<<"${tail_extra_nodes}"
for node in "${tail_extra_node_array[@]}"; do
  found=0
  for candidate in "${cpa_nodes[@]}"; do
    if [[ ${node} == "${candidate}" ]]; then
      found=1
      break
    fi
  done
  [[ ${found} -eq 1 ]] || {
    echo "cpa_tail_extra_node_not_in_fleet:${node}" >&2
    exit 3
  }
done
if [[ ${#tail_extra_node_array[@]} -gt 0 && ${tail_extra_slots} -eq 0 ]]; then
  echo "cpa_tail_extra_slots_must_be_positive" >&2
  exit 3
fi
read -r -a tail2_extra_node_array <<<"${tail2_extra_nodes}"
for node in "${tail2_extra_node_array[@]}"; do
  found=0
  for candidate in "${cpa_nodes[@]}"; do
    if [[ ${node} == "${candidate}" ]]; then found=1; break; fi
  done
  [[ ${found} -eq 1 ]] || {
    echo "cpa_tail2_extra_node_not_in_fleet:${node}" >&2
    exit 3
  }
done
if [[ ${#tail2_extra_node_array[@]} -gt 0 && ${tail2_extra_slots} -eq 0 ]]; then
  echo "cpa_tail2_extra_slots_must_be_positive" >&2
  exit 3
fi

sync_remote_file() {
  local host=$1 source=$2 destination=$3 expected actual payload
  expected=$(sha256sum "${source}" | awk '{print $1}')
  actual=$(timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "root@${host}" \
    "sha256sum '$destination' 2>/dev/null | cut -d ' ' -f 1" 2>/dev/null || true)
  [[ ${actual} != "${expected}" ]] || return 0
  payload=$(gzip -c "${source}" | base64 -w0)
  timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=8 "root@${host}" \
    "printf '%s' '$payload' | base64 -d | gzip -d > '${destination}.partial' && install -m 0755 '${destination}.partial' '$destination' && rm -f '${destination}.partial'"
}

for host in "${add_cpa[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "root@${host}" '
    set -e
    test -s /mnt/SAM3/checkpoints/sam3.pt
    test -s /run/ti/cosmos3-physical-catalog/RELEASE.json
    test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 4
  '
  sync_remote_file "${host}" "${project}/scripts/sta_contact_reuse.py" \
    "${project}/scripts/sta_contact_reuse.py"
  sync_remote_file "${host}" "${project}/scripts/launch_cpa_dynamic_generation.sh" \
    "${project}/scripts/launch_cpa_dynamic_generation.sh"
done
for host in "${add_sta[@]}"; do
  scp -q -o BatchMode=yes -o ConnectTimeout=8 \
    "${project}/scripts/collect_active_sta_ready.py" \
    "root@${host}:/run/ti/collect_active_sta_ready.py"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "root@${host}" \
    'chmod 755 /run/ti/collect_active_sta_ready.py'
done

if [[ "${cpa_nodes[*]:0:${#historical_cpa[*]}}" != "${historical_cpa[*]}" ]]; then
  echo historical_cpa_prefix_changed >&2
  exit 3
fi
if [[ "${sta_nodes[*]:0:${#historical_sta[*]}}" != "${historical_sta[*]}" ]]; then
  echo historical_sta_prefix_changed >&2
  exit 3
fi

atomic_lines "${state_root}/cpa-nodes.txt" "${cpa_nodes[@]}"
atomic_lines "${state_root}/sta-nodes.txt" "${sta_nodes[@]}"
atomic_lines "${new_slots_path}" "${new_slots_per_node}"
atomic_lines "${tail_extra_nodes_path}" "${tail_extra_node_array[@]}"
atomic_lines "${tail_extra_slots_path}" "${tail_extra_slots}"
atomic_lines "${tail2_extra_nodes_path}" "${tail2_extra_node_array[@]}"
atomic_lines "${tail2_extra_slots_path}" "${tail2_extra_slots}"
cpa_string=${cpa_nodes[*]}
sta_string=${sta_nodes[*]}
expected_slots=$((224 + new_slots_per_node * (${#cpa_nodes[@]} - ${#historical_cpa[@]}) + tail_extra_slots * ${#tail_extra_node_array[@]} + tail2_extra_slots * ${#tail2_extra_node_array[@]}))
actual_slots=$(cd "${project}" && env PYTHONPATH="${project}/scripts" \
  CPA_NODES="${cpa_string}" STA_NODES="${sta_string}" \
  CPA_NEW_SLOTS_PER_NODE="${new_slots_per_node}" \
  CPA_NEW_TAIL_EXTRA_NODES="${tail_extra_nodes}" \
  CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE="${tail_extra_slots}" \
  CPA_NEW_TAIL2_EXTRA_NODES="${tail2_extra_nodes}" \
  CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE="${tail2_extra_slots}" /usr/bin/python3 -c \
  'from cpa_fleet_controller import CPA_TOTAL_SLOTS; print(CPA_TOTAL_SLOTS)')
[[ ${actual_slots} -eq ${expected_slots} ]] || {
  echo "cpa_slot_layout_mismatch:expected=${expected_slots}:actual=${actual_slots}" >&2
  exit 3
}

tmux kill-session -t "${session}" 2>/dev/null || true
tmux new-session -d -s "${session}" \
  "cd '${project}' && export CPA_NODES='${cpa_string}' STA_NODES='${sta_string}' CPA_SLOTS_PER_NODE=16 CPA_LOCAL_SLOTS=4 CPA_EXTRA_SLOTS_PER_NODE=8 CPA_LOCAL_EXTRA_SLOTS=4 CPA_NEW_SLOTS_PER_NODE='${new_slots_per_node}' CPA_NEW_TAIL_EXTRA_NODES='${tail_extra_nodes}' CPA_NEW_TAIL_EXTRA_SLOTS_PER_NODE='${tail_extra_slots}' CPA_NEW_TAIL2_EXTRA_NODES='${tail2_extra_nodes}' CPA_NEW_TAIL2_EXTRA_SLOTS_PER_NODE='${tail2_extra_slots}' CPA_GENERATION_UIDS_PER_SLOT=512 CPA_INCLUDE_UNAUDITED_LIVE_STA=1 CPA_MAX_ATTEMPTS=20; exec /usr/bin/python3 -u scripts/cpa_fleet_controller.py --interval-seconds 30 2>&1 | tee -a '${state_root}/controller.log'"
sleep 4
[[ $(tmux list-panes -t "${session}" -F '#{pane_dead}') == 0 ]] || {
  echo cpa_controller_failed_to_start >&2
  exit 3
}
echo "cpa_dynamic_fleet_configured:cpa_nodes=${#cpa_nodes[@]}:sta_nodes=${#sta_nodes[@]}:slots=${expected_slots}:session=${session}"
