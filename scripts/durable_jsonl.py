"""Immutable JSONL cloud writes with an optional locally durable checkpoint outbox."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

IO_LIMITER = threading.BoundedSemaphore(8)
# Recovery cleanup must not consume publication slots or create a thread pool
# for each concurrently completed episode. The executor itself bounds deletes.
CLEANUP_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix='checkpoint-cleanup')
METRICS_LOCK = threading.Lock()
COMMITTED_UNITS = {}
CHECKPOINT_SPOOL = None
FINAL_RECORD_SPOOL = None


class PublicationLimiter:
    """Bound final writes separately from checkpoint replication and cleanup."""
    def __init__(self, workers):
        if type(workers) is not int or not 1 <= workers <= 256:
            raise ValueError('final_publication_workers_must_be_1_to_256')
        from http_transport_metrics import TimedOperationMetrics
        self.workers = workers
        self.slots = threading.BoundedSemaphore(workers)
        self.waiting = TimedOperationMetrics(backend='auto')
        self.writing = TimedOperationMetrics(backend='auto')

    @contextmanager
    def admit(self):
        acquired = False
        try:
            with self.waiting.track():
                self.slots.acquire()
                acquired = True
            with self.writing.track():
                yield
        finally:
            if acquired:
                self.slots.release()

    def snapshot(self):
        wait, write = self.waiting.snapshot(), self.writing.snapshot()
        return {'mode': 'bounded-final-publication/v1', 'workers': self.workers,
                'waiting': wait['active'], 'active': write['active'], 'peak_active': write['peak'],
                'completed': write['finished']-write['failed'], 'failed': write['failed'],
                'wait_median_seconds': wait['recent_call_median_seconds'],
                'write_median_seconds': write['recent_call_median_seconds']}


@contextmanager
def publication_capacity(workers=8):
    global IO_LIMITER
    if type(workers) is not int or not 1 <= workers <= 256:
        raise ValueError('final_publication_workers_must_be_1_to_256')
    if isinstance(IO_LIMITER, PublicationLimiter):
        raise RuntimeError('final_publication_already_configured')
    previous = IO_LIMITER
    IO_LIMITER = PublicationLimiter(workers)
    try:
        yield IO_LIMITER
    finally:
        IO_LIMITER = previous


def commit_metrics():
    """Process-local new durable units; excludes resume reads and duplicate puts."""
    with METRICS_LOCK:
        value = {'durable_units_committed': dict(COMMITTED_UNITS)}
    if CHECKPOINT_SPOOL is not None:
        value['checkpoint_sync'] = CHECKPOINT_SPOOL.snapshot()
    if FINAL_RECORD_SPOOL is not None:
        value['final_registration_outbox'] = FINAL_RECORD_SPOOL.snapshot()
    elif isinstance(IO_LIMITER, PublicationLimiter):
        value['final_publication'] = IO_LIMITER.snapshot()
    return value


@contextmanager
def checkpoint_storage(root, output, workers=32, max_mib=2048, max_files=20000,
                       local_batch_size=1, writer=None, batch_writer=None):
    global CHECKPOINT_SPOOL
    if root is None:
        yield
        return
    from checkpoint_spool import CheckpointSpool
    if CHECKPOINT_SPOOL is not None:
        raise RuntimeError('checkpoint_spool_already_configured')
    cloud_limit = threading.BoundedSemaphore(workers)
    publish = writer or (lambda path, record: atomic_jsonl(path, record, limiter=cloud_limit))
    batch_publish = batch_writer or (lambda rows: atomic_checkpoint_batch(rows, cloud_limit))
    spool = CheckpointSpool(root, output, publish,
                           workers=workers, max_bytes=max_mib*1024**2, max_files=max_files,
                           local_batch_size=local_batch_size,
                           batch_writer=batch_publish)
    CHECKPOINT_SPOOL = spool
    try:
        yield
    finally:
        try:
            spool.close()
        finally:
            CHECKPOINT_SPOOL = None


def check_checkpoint_storage_health():
    if CHECKPOINT_SPOOL is not None:
        CHECKPOINT_SPOOL.check_health()
    if FINAL_RECORD_SPOOL is not None:
        FINAL_RECORD_SPOOL.check_health()


@contextmanager
def final_record_storage(root, output, workers=128, max_mib=8192,
                         max_files=50000, local_batch_size=64, writer=None,
                         batch_writer=None):
    """Acknowledge a final record after local fsync; publish it asynchronously."""
    global FINAL_RECORD_SPOOL
    if root is None:
        yield
        return
    from checkpoint_spool import CheckpointSpool
    if FINAL_RECORD_SPOOL is not None:
        raise RuntimeError('final_record_spool_already_configured')
    cloud_limit = threading.BoundedSemaphore(workers)
    publish = writer or (lambda path, record: atomic_jsonl(path, record, limiter=cloud_limit))
    # COSFS small-file creation is substantially slower than payload transfer.
    # Production records remain individually acknowledged in the local WAL,
    # but records for the same output shard are replicated as one JSONL pack.
    # Existing readers already validate every line in every ``*.jsonl`` file.
    batch_publish = (batch_writer if batch_writer is not None else
                     (lambda rows: atomic_final_record_batch(rows, cloud_limit))
                     if writer is None else None)
    spool = CheckpointSpool(
        root, output, publish, workers=workers, max_bytes=max_mib*1024**2,
        max_files=max_files, local_batch_size=local_batch_size,
        batch_writer=batch_publish, batch_size=64, batch_delay=1,
        path_kind='final-record',
    )
    FINAL_RECORD_SPOOL = spool
    try:
        yield spool
    finally:
        try:
            spool.close()
        finally:
            FINAL_RECORD_SPOOL = None


def put_final_record(path, record):
    if FINAL_RECORD_SPOOL is not None:
        FINAL_RECORD_SPOOL.enqueue(path, record)
    else:
        atomic_jsonl(path, record)


def pending_final_uids(shard_path):
    if FINAL_RECORD_SPOOL is None:
        return set()
    return FINAL_RECORD_SPOOL.pending_input_uids(shard_path)


def final_record_spool_enabled():
    return FINAL_RECORD_SPOOL is not None


def retry_io(operation, path):
    for attempt in range(3):
        try:
            return operation()
        except OSError as error:
            if attempt == 2:
                raise
            print(f'checkpoint_io_retry path={path} attempt={attempt+1} error={error}', flush=True)
            time.sleep(attempt+1)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def unit_directory(path):
    return path.with_suffix('.units')


def checkpoint_paths(path):
    return retry_io(lambda: ([path] if path.is_file() else [])
                    + sorted(unit_directory(path).glob('*.jsonl')), path)


def load_checkpoint_files(checkpoint):
    """Use the existing identity/schema validator for legacy and immutable files."""
    original = checkpoint.path
    if CHECKPOINT_SPOOL is not None:
        CHECKPOINT_SPOOL.flush_checkpoint(original)
    combined = {}
    unit_files = 0
    try:
        for path in checkpoint_paths(original):
            checkpoint.path = path
            checkpoint.quiet_unit_load = path != original
            def load_one():
                checkpoint.entries = {}
                checkpoint._load()
            retry_io(load_one, path)
            for key, value in checkpoint.entries.items():
                if key in combined and fingerprint(combined[key]) != fingerprint(value):
                    raise ValueError(f'conflicting_checkpoint_unit:{original}:{key}')
                combined[key] = value
            unit_files += int(path != original)
    finally:
        checkpoint.path = original
        checkpoint.entries = combined
        checkpoint.quiet_unit_load = False
    if unit_files:
        print(f'checkpoint_units_loaded path={original} files={unit_files} units={len(combined)}', flush=True)


def atomic_jsonl(path, record, limiter=None):
    """Return only after the complete record is fsynced and published."""
    return atomic_jsonl_records(path, [record], limiter)


def atomic_checkpoint_batch(rows, limiter=None):
    """Pack same-episode immutable units using the existing multi-line reader."""
    parents = {path.parent for path, _ in rows}
    if len(parents) != 1:
        raise ValueError('checkpoint_batch_requires_one_episode_directory')
    records = [record for _, record in sorted(rows, key=lambda row: str(row[0]))]
    digest = fingerprint(records)
    atomic_jsonl_records(next(iter(parents))/f'pack-{digest}.jsonl', records, limiter)


def atomic_final_record_batch(rows, limiter=None):
    """Publish same-shard final records in one deterministic JSONL object."""
    parents = {path.parent for path, _ in rows}
    if len(parents) != 1:
        raise ValueError('final_record_batch_requires_one_output_shard')
    records = [record for _, record in sorted(rows, key=lambda row: str(row[0]))]
    digest = fingerprint(records)
    atomic_jsonl_records(next(iter(parents))/f'pack-{digest}.jsonl', records, limiter)


def atomic_jsonl_records(path, records, limiter=None):
    selected = IO_LIMITER if limiter is None else limiter
    with selected.admit() if isinstance(selected, PublicationLimiter) else selected:
        records = list(records)
        if not records:
            raise ValueError('durable_jsonl_refuses_empty_record_set')
        payload = ''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in records)
        if not payload.strip():
            raise ValueError('durable_jsonl_refuses_empty_payload')
        for attempt in range(3):
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                                 prefix='.write-', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                if not path.is_file() or path.stat().st_size != len(payload.encode('utf-8')):
                    raise OSError('durable_jsonl_post_publish_size_mismatch')
                # The source no longer exists after a successful rename. Avoid
                # an unnecessary remote unlink while retaining failure cleanup.
                temporary = None
                return
            except OSError as error:
                if attempt == 2:
                    raise
                print(f'durable_jsonl_retry path={path} attempt={attempt+1} error={error}', flush=True)
                time.sleep(attempt+1)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass


def put_checkpoint_unit(checkpoint, key, value, record):
    """Serialize duplicate keys, but do not hold the index lock during cloud I/O."""
    with checkpoint.lock:
        key_lock = checkpoint.key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with checkpoint.lock:
            if key in checkpoint.entries:
                if fingerprint(checkpoint.entries[key]) != fingerprint(value):
                    raise ValueError(f'conflicting_checkpoint_unit:{key}')
                return
        path = unit_directory(checkpoint.path)/(hashlib.sha256(key.encode()).hexdigest()+'.jsonl')
        if CHECKPOINT_SPOOL is not None:
            CHECKPOINT_SPOOL.enqueue(path, record)
        else:
            atomic_jsonl(path, record)
        # Do not expose completion in memory until durable publication succeeds.
        import copy
        with checkpoint.lock:
            checkpoint.entries[key] = copy.deepcopy(value)
        with METRICS_LOCK:
            task = getattr(checkpoint, 'task', 'grd')
            COMMITTED_UNITS[task] = COMMITTED_UNITS.get(task, 0) + 1


def put_checkpoint_units(checkpoint, items):
    """Publish several units from one checkpoint with one producer wait.

    The immutable files and record schema remain unchanged.  All absent units
    become visible in memory only after every corresponding local SQLite ticket
    is FULL-synchronous durable.
    """
    values = {}
    records = {}
    for key, value, record in items:
        if key in values and fingerprint(values[key]) != fingerprint(value):
            raise ValueError(f'conflicting_checkpoint_unit:{key}')
        values[key] = value
        records[key] = record
    if not values:
        return
    with checkpoint.lock:
        key_locks = [(key, checkpoint.key_locks.setdefault(key, threading.Lock()))
                     for key in sorted(values)]
    for _, key_lock in key_locks:
        key_lock.acquire()
    try:
        publish = []
        committed = []
        with checkpoint.lock:
            for key, value in values.items():
                previous = checkpoint.entries.get(key)
                if previous is not None:
                    if fingerprint(previous) != fingerprint(value):
                        raise ValueError(f'conflicting_checkpoint_unit:{key}')
                    continue
                path = unit_directory(checkpoint.path)/(hashlib.sha256(key.encode()).hexdigest()+'.jsonl')
                publish.append((path, records[key]))
                committed.append((key, value))
        if CHECKPOINT_SPOOL is not None:
            CHECKPOINT_SPOOL.enqueue_many(publish)
        else:
            for path, record in publish:
                atomic_jsonl(path, record)
        import copy
        with checkpoint.lock:
            for key, value in committed:
                checkpoint.entries[key] = copy.deepcopy(value)
        if committed:
            with METRICS_LOCK:
                task = getattr(checkpoint, 'task', 'grd')
                COMMITTED_UNITS[task] = COMMITTED_UNITS.get(task, 0) + len(committed)
    finally:
        for _, key_lock in reversed(key_locks):
            key_lock.release()


def clear_checkpoint_files(path):
    """Remove only this record's known recovery files after durable registration."""
    try:
        if CHECKPOINT_SPOOL is not None:
            CHECKPOINT_SPOOL.flush_checkpoint(path)
        files = checkpoint_paths(path)
    except OSError as error:
        print(f'checkpoint_cleanup_warning path={path} error={error}', flush=True)
        return
    def remove(file):
        try:
            file.unlink(missing_ok=True)
        except OSError as error:
            # Retained recovery data is harmless; the final annotation is durable.
            print(f'checkpoint_cleanup_warning path={file} error={error}', flush=True)
    list(CLEANUP_EXECUTOR.map(remove, files))
    try:
        unit_directory(path).rmdir()
    except OSError:
        # Preserve incompatible evidence, interrupted temporaries, and unknown files.
        pass
