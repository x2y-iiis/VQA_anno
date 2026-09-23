#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 4 && $# -le 7 ]] || {
  echo 'Usage: launch_grd_sta_fleet_worker.sh TASK WORKER_ID UID_PATH OUTPUT_ROOT [RESUME_OUTPUT] [STA_MODEL] [ARK_UPLOAD_WORKERS]' >&2
  exit 2
}
TASK=$1
WORKER_ID=$2
UID_PATH=$3
OUTPUT_ROOT=$4
PROJECT=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SESSION="vqa-$TASK-$WORKER_ID"
LOG_ROOT=/run/ti/vqa-grd-sta-fleet/logs
mkdir -p "$LOG_ROOT"
tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "tmux_session_already_exists:$SESSION" >&2
  exit 3
}

printf -v worker_command '%q ' bash "$PROJECT/scripts/run_grd_sta_fleet_worker.sh" "$@"
printf -v monitor_command '%q ' /usr/bin/python3 "$PROJECT/scripts/monitor_grd_sta_fleet.py" --task "$TASK" --output "$OUTPUT_ROOT" --uid-path "$UID_PATH"
tmux new-session -d -s "$SESSION" "$worker_command 2>&1 | tee -a '$LOG_ROOT/$TASK-$WORKER_ID.log'"
tmux split-window -v -t "$SESSION":0 "$monitor_command"
tmux select-layout -t "$SESSION":0 even-vertical
echo "tmux_session_started:$SESSION"
echo "attach: tmux attach -t $SESSION"
