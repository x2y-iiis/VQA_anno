"""Quarantine only provably malformed cached LAS responses before record retry."""
import hashlib
import json
import os
from pathlib import Path
import time

from request_parallel import atomic_json


def invalidate_unparseable_response(item_dir, error, task_root):
    from las_annotation.models import PipelineDataError
    from las_annotation.parsing import parse_actions_response, parse_objects_response

    parsers = {'step1': parse_objects_response, 'step3': parse_actions_response}
    stage = error.get('stage')
    if stage not in parsers:
        return False
    path = Path(item_dir)/f'{stage}_raw.json'
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or not isinstance(record.get('request'), dict) or not isinstance(record.get('response'), dict):
        return False
    try:
        parsers[stage](record['response'])
    except PipelineDataError:
        pass
    else:
        # An unrelated transient error must never invalidate a valid response.
        return False
    digest = hashlib.sha256(json.dumps(record['request'], sort_keys=True).encode()).hexdigest()
    task_path = Path(task_root)/digest[:2]/f'{digest}.json'
    if task_path.exists():
        task = json.loads(task_path.read_text())
        task.update(status='FAILED', local_failure='unparseable_response', updated_at_unix=time.time())
        atomic_json(task_path, task)
    # Keep the exact original bytes for audit. Disable task-ID reuse before
    # removing the raw cache, so a crash cannot make the retry reuse it forever.
    backup = path.with_suffix(f'.invalid-response.{time.time_ns()}.json')
    os.replace(path, backup)
    print(f'las_invalid_response_quarantined stage={stage} request_hash={digest}', flush=True)
    return True
