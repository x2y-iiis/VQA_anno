#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 HOST1 HOST2 HOST3 HOST4 HOST5 HOST6 HOST7 HOST8 HOST9 HOST10" >&2
  echo "allocation: first 8 GRD, last 2 STA; all hosts are CPA-ready" >&2
  exit 2
fi

project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
hosts=("$@")
roles=(grd grd grd grd grd grd grd grd sta sta)
run_root=/run/ti/new-fleet-provision-$(date +%Y%m%d-%H%M%S)
mkdir -p "${run_root}"

for index in "${!hosts[@]}"; do
  host=${hosts[$index]}
  role=${roles[$index]}
  (
    bash "${project}/scripts/provision_new_fleet_node.sh" "${host}" "${role}" \
      >"${run_root}/${host}.log" 2>&1
    printf 'ready\t%s\t%s\n' "${host}" "${role}" >"${run_root}/${host}.status"
  ) &
done
wait

for index in "${!hosts[@]}"; do
  host=${hosts[$index]}
  role=${roles[$index]}
  if [[ -s "${run_root}/${host}.status" ]]; then
    cat "${run_root}/${host}.status"
  else
    printf 'failed\t%s\t%s\tlog=%s\n' "${host}" "${role}" "${run_root}/${host}.log"
  fi
done

failed=$(find "${run_root}" -maxdepth 1 -type f -name '*.status' | wc -l)
[[ ${failed} -eq 10 ]] || exit 1
echo "new_fleet_provision_complete:run_root=${run_root}"
