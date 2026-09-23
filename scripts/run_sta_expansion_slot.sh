#!/usr/bin/env bash
set -euo pipefail
set +x

[[ $# -eq 4 ]] || {
  echo 'Usage: run_sta_expansion_slot.sh WORKER_ID UID_PATH OUTPUT_ROOT LOG_PATH' >&2
  exit 2
}
worker_id=$1
uid_path=$2
output_root=$3
log_path=$4
project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

set -a
source /mnt/cpa-ark-new/.env
set +a
export TOS_UPLOAD_VIA_COS_FETCH=1

mkdir -p "$(dirname "$log_path")"
exec > >(tee -a "$log_path") 2>&1
exec bash "$project/scripts/run_grd_sta_fleet_worker.sh" \
  sta "$worker_id" "$uid_path" "$output_root" '' \
  doubao-seed-2-1-turbo-260628 1024
