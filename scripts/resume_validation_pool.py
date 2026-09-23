"""Read-only resume validation outside the HTTP interpreter; never bypass audits."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import re
import threading
import time


def signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def fast_incompatible_immutable_record(path, task, model):
    """Reject a per-episode file with a different task/model from its header.

    Immutable GRD records can be tens of megabytes because one JSON line owns
    every frame.  The canonical top-level UID is written before those payloads
    and already includes ``annotation:<task>:<model>``.  Reading the header is
    sufficient to reject an older model, while matching/unknown layouts still
    take the complete contract-validation path below.
    """
    path = Path(path)
    if not path.parent.name.endswith('.records'):
        return False
    try:
        with path.open('rb') as stream:
            header = stream.read(4096)
    except OSError:
        return False
    if b'"schema_version": "unified-vqa-record/v2"' not in header:
        return False
    match = re.search(rb'"uid"\s*:\s*("(?:[^"\\]|\\.)*")', header)
    if match is None:
        return False
    try:
        uid = json.loads(match.group(1))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return not str(uid).endswith(f':annotation:{task}:{model}')


def validate_current_file(path, task, model, provider, require_review):
    """Return compatible UIDs from an unchanged, structurally readable file.

    Incompatible, duplicate, or not-yet-reviewed rows are deliberately ignored.
    Immutable output mode will replace their per-episode files after successful
    re-annotation, so synchronously rewriting them on an object-store mount is
    unnecessary and can delay request startup for hours.
    """
    from annotate_videos import compatible_annotation_uid
    from audit_grounding_records import record_is_reviewed
    path = Path(path)
    try:
        before = signature(path)
        if fast_incompatible_immutable_record(path, task, model):
            if before != signature(path):
                return None
            return {'uids': [], 'ignored_rows': 1,
                    'signature': before, 'worker_pid': os.getpid(),
                    'fast_header_reject': True}
        seen = set()
        ignored = 0
        with path.open('rb', buffering=4 * 1024 * 1024) as stream:
            for raw in stream:
                if not raw.endswith(b'\n'):
                    return None
                if not raw.strip():
                    ignored += 1
                    continue
                # Strict UTF-8 is conservative: unusual encodings return to
                # the original audit/sanitization path instead of bypassing it.
                record = json.loads(raw.decode('utf-8'))
                uid = compatible_annotation_uid(record, task, model, provider)
                if uid is None or uid in seen or (require_review and not record_is_reviewed(record)):
                    ignored += 1
                    continue
                seen.add(uid)
        if before != signature(path):
            return None
        return {'uids': sorted(seen), 'ignored_rows': ignored,
                'signature': before, 'worker_pid': os.getpid()}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def warm_worker(delay_seconds):
    """Keep a probe busy briefly so every spawn worker becomes observable."""
    time.sleep(delay_seconds)
    return os.getpid()


class ResumeValidationPool:
    def __init__(self, workers):
        if not 1 <= workers <= 32:
            raise ValueError('resume_validation_workers_must_be_between_1_and_32')
        self.workers = workers
        self.gate = threading.BoundedSemaphore(workers)
        self.executor = None
        self._worker_pids = set()

    def __enter__(self):
        self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))
        return self

    def prewarm(self, timeout_seconds=60):
        """Spawn every validator before the parent creates thousands of threads."""
        if self.executor is None:
            raise RuntimeError('resume_validation_pool_not_entered')
        deadline = time.monotonic() + timeout_seconds
        while len(self._worker_pids) < self.workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f'resume_validation_pool_prewarm_timeout:{len(self._worker_pids)}:{self.workers}'
                )
            futures = [self.executor.submit(warm_worker, 0.05) for _ in range(self.workers)]
            for future in futures:
                self._worker_pids.add(future.result(timeout=remaining))
        return set(self._worker_pids)

    def validate(self, path, task, model, provider, require_review):
        if self.executor is None:
            raise RuntimeError('resume_validation_pool_not_entered')
        with self.gate:
            result = self.executor.submit(validate_current_file, path, task, model,
                                          provider, require_review).result()
        if result is None:
            return None
        if result.get('worker_pid'):
            self._worker_pids.add(result['worker_pid'])
        try:
            if signature(path) != result['signature']:
                return None
        except OSError:
            return None
        return set(result['uids'])

    def snapshot(self):
        return {'configured_workers': self.workers, 'worker_pids': sorted(self._worker_pids)}

    def __exit__(self, *_exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
