"""Print a compact live snapshot for one GRD or STA fleet worker."""

import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("grd", "sta"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid-path", type=Path, required=True)
    args = parser.parse_args()
    status_path = args.output / "_state" / "request-parallel-status.json"
    total = sum(1 for line in args.uid_path.open(encoding="utf-8") if line.strip())
    while True:
        os.system("clear")
        print(f"VQA FLEET {args.task.upper()} | output={args.output.name} | scope={total:,}")
        try:
            status = json.loads(status_path.read_text())
            age = time.time() - float(status.get("updated_at_unix", 0))
            durable = (status.get("durable_units_committed") or {}).get(args.task, 0)
            print(
                f"heartbeat_age={age:.1f}s active={status.get('active_requests', 0):,} "
                f"peak={status.get('peak_active_requests', 0):,} queued={status.get('queued_requests', 0):,} "
                f"durable_{args.task}_units={durable:,}"
            )
            stage = status.get("stage_pipeline") or {}
            print(
                f"episodes submitted={stage.get('submitted', 0):,} "
                f"completed={stage.get('completed', 0):,} failed={stage.get('failed', 0):,} "
                f"running={sum((stage.get('running_by_lane') or {}).values()):,}"
            )
            for lane, models in (status.get("model_admission") or {}).items():
                for model, metrics in models.items():
                    transport = metrics.get("http_transport") or {}
                    print(
                        f"{lane}/{model}: active={transport.get('active', 0):,} "
                        f"started={transport.get('started', 0):,} finished={transport.get('finished', 0):,} "
                        f"failed={transport.get('failed', 0):,} rate_limits={metrics.get('rate_limit_events', 0):,}"
                    )
        except (OSError, ValueError, TypeError) as error:
            print(f"status_unavailable: {type(error).__name__}: {error}")
        time.sleep(30)


if __name__ == "__main__":
    main()
