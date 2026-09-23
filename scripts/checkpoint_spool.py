"""Locally durable, bounded SQLite outbox for immutable cloud checkpoints.

Acknowledgement means a FULL-synchronous local SQLite commit. Rows are removed
only after the cloud writer returns successfully. Losing the local disk before
replication can lose unsynced records; a local acknowledgement is not a cloud one.
"""
import fcntl
from collections import deque
from dataclasses import dataclass, field
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time


@dataclass
class _CommitTicket:
    payload: str
    size: int
    created: float = field(default_factory=time.time)
    ready: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass
class _PendingJob:
    payload: str
    size: int
    created: float
    parent: str
    attempts: int = 0
    retry_at: float = 0


@dataclass
class _CloudAck:
    claimed: list
    error_name: str | None
    retry_at: float | None
    ready: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class CheckpointSpool:
    def __init__(self, root, output, writer, workers=32, max_bytes=2048*1024**2,
                 max_files=20000, min_free_bytes=8*1024**3, batch_writer=None,
                 batch_size=64, batch_delay=1, local_batch_size=1,
                 local_batch_delay=0.005, path_kind='checkpoint'):
        if min(workers, max_bytes, max_files, local_batch_size) < 1 or local_batch_delay < 0:
            raise ValueError('checkpoint_spool_limits_must_be_positive')
        if path_kind not in {'checkpoint', 'final-record'}:
            raise ValueError('unsupported_spool_path_kind')
        self.path_kind = path_kind
        self.output = Path(output).absolute()
        self.root = Path(root)/hashlib.sha256(str(self.output).encode()).hexdigest()[:20]
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_fd = os.open(self.root/'writer.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock_fd)
            raise
        mutex = threading.RLock()
        self.condition = threading.Condition(mutex)
        self.local_condition = threading.Condition(mutex)
        self.db = sqlite3.connect(self.root/'outbox.sqlite3', check_same_thread=False, isolation_level=None)
        os.chmod(self.root/'outbox.sqlite3', 0o600)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (path TEXT PRIMARY KEY, payload TEXT NOT NULL, '
                        'size INTEGER NOT NULL, created REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, '
                        'retry_at REAL NOT NULL DEFAULT 0)')
        self.db.execute('CREATE INDEX IF NOT EXISTS retry_index ON jobs(retry_at, created)')
        if 'parent' not in {row[1] for row in self.db.execute('PRAGMA table_info(jobs)')}:
            self.db.execute("ALTER TABLE jobs ADD COLUMN parent TEXT NOT NULL DEFAULT ''")
        for (path,) in self.db.execute("SELECT path FROM jobs WHERE parent=''").fetchall():
            self.db.execute('UPDATE jobs SET parent=? WHERE path=?', (str(Path(path).parent), path))
        self.db.execute('CREATE INDEX IF NOT EXISTS parent_index ON jobs(parent,retry_at)')
        if self.path_kind == 'final-record':
            # ``jobs`` is an outbox, not a completion ledger: successful cloud
            # publication deletes its row.  Keep the accepted identity in the
            # same FULL-synchronous transaction so a restart cannot forget a
            # completed record and replay its (potentially billable) inference.
            self.db.execute(
                'CREATE TABLE IF NOT EXISTS accepted ('
                'path TEXT PRIMARY KEY, parent TEXT NOT NULL, '
                'input_record_uid TEXT NOT NULL, payload_sha256 TEXT NOT NULL, '
                'created REAL NOT NULL)'
            )
            self.db.execute('CREATE INDEX IF NOT EXISTS accepted_parent_index '
                            'ON accepted(parent,input_record_uid)')
            # ``accepted`` is a local-WAL acknowledgement.  Keep a distinct,
            # append-only publication ledger for records whose cloud writer
            # returned successfully.  Downstream stages can consume this table
            # incrementally without waiting for an unrelated, continuously
            # busy outbox to become completely empty.
            self.db.execute(
                'CREATE TABLE IF NOT EXISTS published ('
                'path TEXT PRIMARY KEY, parent TEXT NOT NULL, '
                'input_record_uid TEXT NOT NULL, published REAL NOT NULL)'
            )
            self.db.execute('CREATE INDEX IF NOT EXISTS published_parent_index '
                            'ON published(parent,input_record_uid)')
            self.db.execute('BEGIN IMMEDIATE')
            try:
                for path, payload, created, parent in self.db.execute(
                        'SELECT path,payload,created,parent FROM jobs').fetchall():
                    uid, digest = self._final_record_identity(payload)
                    self.db.execute(
                        'INSERT OR IGNORE INTO accepted('
                        'path,parent,input_record_uid,payload_sha256,created) '
                        'VALUES(?,?,?,?,?)',
                        (path, parent, uid, digest, created),
                    )
                # Migration is exact: an accepted path that is absent from the
                # durable outbox was removed only after a successful cloud
                # write.  Pending/retrying paths remain excluded.
                self.db.execute(
                    'INSERT OR IGNORE INTO published(path,parent,input_record_uid,published) '
                    'SELECT a.path,a.parent,a.input_record_uid,? FROM accepted a '
                    'LEFT JOIN jobs j ON j.path=a.path WHERE j.path IS NULL',
                    (time.time(),),
                )
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.writer, self.max_bytes, self.max_files = writer, max_bytes, max_files
        self.batch_writer, self.batch_size, self.batch_delay = batch_writer, max(1, batch_size), max(0, batch_delay)
        self.min_free_bytes = min_free_bytes
        self.pending, self.pending_bytes = self.db.execute('SELECT COUNT(*), COALESCE(SUM(size),0) FROM jobs').fetchone()
        # This bounded mirror is rebuilt from the durable database on startup.
        # Claiming a job must never scan SQLite rows while holding the condition.
        self.jobs = {}
        self.ready = {}
        self.ready_by_parent = {}
        self.ready_heap = []
        self.ready_serial = itertools.count()
        for path, payload, size, created, parent, attempts, retry_at in self.db.execute(
                'SELECT path,payload,size,created,parent,attempts,retry_at FROM jobs ORDER BY created').fetchall():
            self.jobs[path] = _PendingJob(payload, size, created, parent, attempts, retry_at)
            self._schedule(path)
        self.created_by_path = dict(self.db.execute('SELECT path,created FROM jobs').fetchall())
        self.created_heap = [(created, path) for path, created in self.created_by_path.items()]
        heapq.heapify(self.created_heap)
        self.local_batch_size, self.local_batch_delay = local_batch_size, local_batch_delay
        self.local_queue = {}
        self.accepted = {}
        self.accepted_by_parent = {}
        if self.path_kind == 'final-record':
            for path, parent, uid, digest in self.db.execute(
                    'SELECT path,parent,input_record_uid,payload_sha256 FROM accepted').fetchall():
                self.accepted[path] = (uid, digest, parent)
                self.accepted_by_parent.setdefault(parent, set()).add(uid)
        self.cloud_ack_queue = deque()
        self.local_queued_bytes = 0
        self.local_transactions = 0
        self.database_transactions = 0
        self.last_database_seconds = self.max_database_seconds = 0.0
        self.inflight = set()
        self.active_writers = 0
        self.local_commits = self.cloud_commits = self.errors = 0
        self.cloud_objects = 0
        self.last_error = None
        self.worker_failure = None
        self.last_success = time.time()
        self._publish_health()
        self.stopping = False
        self.threads = [threading.Thread(target=self._worker, name=f'checkpoint-sync-{i}', daemon=True)
                        for i in range(workers)]
        self.local_thread = (threading.Thread(target=self._commit_loop, name='checkpoint-local-commit', daemon=True)
                             if local_batch_size > 1 else None)
        self._publish_snapshot()
        if self.local_thread:
            self.local_thread.start()
        for thread in self.threads:
            thread.start()

    def _relative(self, path):
        path = Path(path).absolute()
        if '..' in path.parts:
            raise ValueError('checkpoint_spool_path_traversal')
        relative = path.relative_to(self.output)
        if self.path_kind == 'checkpoint':
            if relative.parts[:1] != ('_state',) or '.units' not in str(relative.parent):
                raise ValueError('checkpoint_spool_accepts_only_immutable_checkpoint_units')
        elif (len(relative.parts) != 5 or relative.parts[1] != 'shards'
              or not relative.parts[3].endswith('.records')
              or not re.fullmatch(r'[0-9a-f]{64}\.jsonl', relative.parts[4])):
            raise ValueError('final_record_spool_accepts_only_immutable_stage_records')
        return str(relative)

    @staticmethod
    def _final_record_identity(payload):
        try:
            uid = str(json.loads(payload)['provenance']['input_record_uid'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError('invalid_final_record_spool_payload') from error
        return uid, hashlib.sha256(payload.encode()).hexdigest()

    def _remember_accepted(self, relative, payload, created):
        uid, digest = self._final_record_identity(payload)
        parent = str(Path(relative).parent)
        self.accepted[relative] = (uid, digest, parent)
        self.accepted_by_parent.setdefault(parent, set()).add(uid)

    def pending_input_uids(self, shard_path):
        """Return durable final-record UIDs known before a resume scan starts."""
        if self.path_kind != 'final-record':
            raise RuntimeError('pending_input_uids_requires_final_record_spool')
        parent = str(Path(shard_path).with_suffix('.records').absolute().relative_to(self.output))
        with self.condition:
            result = set(self.accepted_by_parent.get(parent, ()))
            payloads = [ticket.payload for path, ticket in self.local_queue.items()
                        if str(Path(path).parent) == parent]
        for payload in payloads:
            result.add(self._final_record_identity(payload)[0])
        return result

    def published_input_uids(self, shard_path=None):
        """Return final-record UIDs acknowledged by the cloud writer."""
        if self.path_kind != 'final-record':
            raise RuntimeError('published_input_uids_requires_final_record_spool')
        if shard_path is None:
            rows = self.db.execute('SELECT input_record_uid FROM published').fetchall()
        else:
            parent = str(Path(shard_path).with_suffix('.records').absolute().relative_to(self.output))
            rows = self.db.execute(
                'SELECT input_record_uid FROM published WHERE parent=?', (parent,),
            ).fetchall()
        return {row[0] for row in rows}

    def _publish_health(self):
        # Replace one immutable snapshot while state is serialized by the
        # database condition. Readers must never wait behind SQLite/fsync.
        self._health_snapshot = (self.worker_failure, self.errors, self.last_success, self.last_error)

    def check_health(self):
        if shutil.disk_usage(self.root).free < self.min_free_bytes:
            raise RuntimeError('checkpoint_spool_local_disk_headroom_exhausted')
        self._check_worker_health()

    def _check_worker_health(self):
        failure, errors, last_success, last_error = self._health_snapshot
        if failure:
            raise RuntimeError(f'checkpoint_spool_worker_failed:{failure}')
        if errors >= 3 and time.time()-last_success > 120 and last_error:
            raise RuntimeError('checkpoint_spool_cloud_sync_stalled')

    def enqueue(self, path, record):
        self.enqueue_many([(path, record)])

    def enqueue_many(self, rows):
        """Durably enqueue one producer batch and wait for its local commits.

        Producers commonly create several independent frame checkpoints for one
        episode.  Queuing the whole group before waiting lets the single SQLite
        owner combine them into full batches instead of forcing every episode
        through one fsync rendezvous per frame.
        """
        prepared = {}
        for path, record in rows:
            relative = self._relative(path)
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True)+'\n'
            size = len(payload.encode())
            if size > self.max_bytes:
                raise ValueError('checkpoint_unit_exceeds_spool_capacity')
            previous = prepared.get(relative)
            if previous is not None and previous != (payload, size):
                raise ValueError('conflicting_checkpoint_spool_unit')
            prepared[relative] = (payload, size)
        if not prepared:
            return
        if len(prepared) > self.max_files or sum(size for _, size in prepared.values()) > self.max_bytes:
            raise ValueError('checkpoint_batch_exceeds_spool_capacity')
        # Filesystem metadata can block. The committer checks disk headroom
        # again before every transaction; never hold the queue mutex for statfs.
        self.check_health()
        tickets = {}
        with self.condition:
            while True:
                if self.stopping:
                    raise RuntimeError('checkpoint_spool_stopping')
                self._check_worker_health()
                new_rows = []
                tickets.clear()
                for relative, (payload, size) in prepared.items():
                    accepted = self.accepted.get(relative)
                    if accepted is not None:
                        _, digest, _ = accepted
                        if digest != hashlib.sha256(payload.encode()).hexdigest():
                            raise ValueError('conflicting_checkpoint_spool_unit')
                        continue
                    ticket = self.local_queue.get(relative)
                    if ticket is not None:
                        if ticket.payload != payload:
                            raise ValueError('conflicting_checkpoint_spool_unit')
                        tickets[relative] = ticket
                        continue
                    existing = self.jobs.get(relative)
                    if existing is not None:
                        if existing.payload != payload:
                            raise ValueError('conflicting_checkpoint_spool_unit')
                        continue
                    new_rows.append((relative, payload, size))
                new_bytes = sum(size for _, _, size in new_rows)
                if (self.pending+len(self.local_queue)+len(new_rows) <= self.max_files
                        and self.pending_bytes+self.local_queued_bytes+new_bytes <= self.max_bytes):
                    break
                self.condition.wait(timeout=1)
                self.condition.release()
                try:
                    self.check_health()
                finally:
                    self.condition.acquire()
            if self.local_thread:
                for relative, payload, size in new_rows:
                    ticket = _CommitTicket(payload, size)
                    self.local_queue[relative] = ticket
                    self.local_queued_bytes += size
                    tickets[relative] = ticket
                if new_rows:
                    self._publish_snapshot()
                    self.local_condition.notify()
            else:
                for relative, payload, size in new_rows:
                    self.check_health()
                    self._commit_single(relative, payload, size)
                return
        # A queued ticket is NOT a successful checkpoint. Even duplicate callers
        # wait until the transaction containing their record is fsynced. Waiting
        # outside the queue mutex lets the owner commit the entire producer batch.
        seen = set()
        for ticket in tickets.values():
            identity = id(ticket)
            if identity in seen:
                continue
            seen.add(identity)
            ticket.ready.wait()
            if ticket.error:
                raise ticket.error

    def _commit_single(self, relative, payload, size):
        created = time.time()
        parent = str(Path(relative).parent)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO jobs(path,payload,size,created,parent) VALUES(?,?,?,?,?)',
                            (relative, payload, size, created, parent))
            if self.path_kind == 'final-record':
                uid, digest = self._final_record_identity(payload)
                self.db.execute(
                    'INSERT INTO accepted(path,parent,input_record_uid,payload_sha256,created) '
                    'VALUES(?,?,?,?,?)', (relative, parent, uid, digest, created))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        if self.path_kind == 'final-record':
            self._remember_accepted(relative, payload, created)
        self.jobs[relative] = _PendingJob(payload, size, created, str(Path(relative).parent))
        self._schedule(relative)
        self.created_by_path[relative] = created
        heapq.heappush(self.created_heap, (created, relative))
        self.pending += 1
        self.pending_bytes += size
        self.local_commits += 1
        self.local_transactions += 1
        self.database_transactions += 1
        self._publish_snapshot()
        self.condition.notify_all()

    def _fail_local_queue(self, error):
        for ticket in self.local_queue.values():
            ticket.error = error
            ticket.ready.set()
        self.local_queue.clear()
        self.local_queued_bytes = 0

    def _fail_cloud_acks(self, error):
        while self.cloud_ack_queue:
            ticket = self.cloud_ack_queue.popleft()
            ticket.error = error
            ticket.ready.set()

    def _commit_loop(self):
        with self.condition:
            try:
                while True:
                    if self.worker_failure:
                        break
                    if self.stopping:
                        self._fail_local_queue(RuntimeError('checkpoint_spool_stopping'))
                        # Finish acknowledgements for already-running cloud
                        # writes. Never close SQLite beneath a late completion.
                        if not self.cloud_ack_queue and not self.active_writers:
                            break
                    if not self.local_queue and not self.cloud_ack_queue:
                        self.local_condition.wait()
                        continue
                    deadline = time.monotonic()+self.local_batch_delay
                    while (self.local_queue and len(self.local_queue) < self.local_batch_size
                           and not self.cloud_ack_queue and not self.stopping):
                        remaining = deadline-time.monotonic()
                        if remaining <= 0:
                            break
                        self.local_condition.wait(timeout=remaining)
                    if self.stopping:
                        self._fail_local_queue(RuntimeError('checkpoint_spool_stopping'))
                    batch = list(itertools.islice(self.local_queue.items(), self.local_batch_size))
                    acknowledgements = list(self.cloud_ack_queue)
                    if not batch and not acknowledgements:
                        continue
                    # This thread is the sole database writer in batched mode.
                    # Keep selected tickets in local_queue until COMMIT so
                    # duplicates still wait and capacity accounting includes
                    # in-flight commits. Producers/cloud completions may append
                    # while SQLite or statfs blocks; only this owner removes.
                    self.condition.release()
                    database_started = time.monotonic()
                    try:
                        self.check_health()
                        self._commit_database_batch(batch, acknowledgements)
                    finally:
                        database_seconds = time.monotonic()-database_started
                        self.condition.acquire()
                        self.last_database_seconds = database_seconds
                        self.max_database_seconds = max(self.max_database_seconds, database_seconds)
                    # Publish success only after FULL-synchronous COMMIT. The
                    # unchanged jobs table is replayable by previous versions.
                    for path, ticket in batch:
                        del self.local_queue[path]
                        self.local_queued_bytes -= ticket.size
                        self.jobs[path] = _PendingJob(ticket.payload, ticket.size, ticket.created,
                                                      str(Path(path).parent))
                        if self.path_kind == 'final-record':
                            self._remember_accepted(path, ticket.payload, ticket.created)
                        self._schedule(path)
                        self.created_by_path[path] = ticket.created
                        heapq.heappush(self.created_heap, (ticket.created, path))
                    self.pending += len(batch)
                    self.pending_bytes += sum(ticket.size for _, ticket in batch)
                    self.local_commits += len(batch)
                    self.local_transactions += bool(batch)
                    self.database_transactions += 1
                    for ack in acknowledgements:
                        self._apply_cloud_ack(ack)
                        self.cloud_ack_queue.popleft()
                    self._publish_health()
                    self._publish_snapshot()
                    self.condition.notify_all()
                    # Notify committed callers without retaining the shared
                    # database/index mutex across OS thread/GIL handoffs.
                    self.condition.release()
                    try:
                        for _, ticket in batch:
                            ticket.ready.set()
                        for ack in acknowledgements:
                            ack.ready.set()
                    finally:
                        self.condition.acquire()
                    # Producers and cloud acknowledgements must get the lock
                    # between commits, even under a continuously full queue.
                    self.condition.wait(timeout=0.0001)
            except BaseException as error:
                self.worker_failure = type(error).__name__
                self.stopping = True
                self._fail_local_queue(error)
                self._fail_cloud_acks(error)
                self._publish_health()
                print(f'checkpoint_local_commit_failed type={type(error).__name__}', flush=True)
            finally:
                self._fail_local_queue(RuntimeError('checkpoint_spool_stopping'))
                self._fail_cloud_acks(RuntimeError('checkpoint_spool_stopping'))
                self._publish_snapshot()
                self.condition.notify_all()

    def _commit_database_batch(self, batch, acknowledgements):
        """Called only by the commit owner, without the queue/index mutex."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for offset in range(0, len(batch), 128):
                chunk = batch[offset:offset+128]
                values = [(path, t.payload, t.size, t.created, str(Path(path).parent))
                          for path, t in chunk]
                self.db.execute('INSERT INTO jobs(path,payload,size,created,parent) VALUES '
                                + ','.join(['(?,?,?,?,?)']*len(chunk)),
                                tuple(value for row in values for value in row))
                if self.path_kind == 'final-record':
                    accepted = []
                    for path, ticket in chunk:
                        uid, digest = self._final_record_identity(ticket.payload)
                        accepted.append((path, str(Path(path).parent), uid, digest, ticket.created))
                    self.db.execute(
                        'INSERT INTO accepted('
                        'path,parent,input_record_uid,payload_sha256,created) VALUES '
                        + ','.join(['(?,?,?,?,?)']*len(accepted)),
                        tuple(value for row in accepted for value in row),
                    )
            completed_paths = [row[0] for ack in acknowledgements
                               if ack.error_name is None for row in ack.claimed]
            self._execute_cloud_rows(completed_paths)
            for ack in acknowledgements:
                if ack.error_name is not None:
                    self._execute_cloud_rows([row[0] for row in ack.claimed], ack.retry_at)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _worker(self):
        try:
            self._worker_loop()
        except Exception as error:
            with self.condition:
                self.worker_failure = self.worker_failure or type(error).__name__
                self.stopping = True
                self._publish_health()
                self._publish_snapshot()
                self.condition.notify_all()
                self.local_condition.notify_all()
            print(f'checkpoint_sync_worker_failed type={type(error).__name__}', flush=True)

    def _schedule(self, path):
        job = self.jobs[path]
        serial = next(self.ready_serial)
        self.ready[path] = serial
        self.ready_by_parent.setdefault(job.parent, {})[path] = serial
        due = max(job.retry_at, job.created+self.batch_delay if self.batch_writer else job.created)
        heapq.heappush(self.ready_heap, (due, job.created, serial, path))

    def _claim_batch(self, now):
        # Only paths/generations are kept in the heap. Stale entries cannot
        # retain payloads or claim a subsequently re-enqueued version of a path.
        if len(self.ready_heap) > max(1024, 2*len(self.ready)):
            self.ready_heap = [row for row in self.ready_heap if self.ready.get(row[3]) == row[2]]
            heapq.heapify(self.ready_heap)
        while self.ready_heap and self.ready.get(self.ready_heap[0][3]) != self.ready_heap[0][2]:
            heapq.heappop(self.ready_heap)
        if not self.ready_heap:
            return [], 1
        due, _, _, first = self.ready_heap[0]
        if due > now:
            return [], min(1, due-now)
        parent = self.jobs[first].parent
        paths = [first]
        if self.batch_writer and self.batch_size > 1:
            for path in self.ready_by_parent[parent]:
                if path != first and self.jobs[path].retry_at <= now:
                    paths.append(path)
                    if len(paths) == self.batch_size:
                        break
        claimed = []
        for path in paths:
            job = self.jobs[path]
            del self.ready[path]
            del self.ready_by_parent[parent][path]
            claimed.append((path, job.payload, job.size, job.attempts))
        if not self.ready_by_parent[parent]:
            del self.ready_by_parent[parent]
        self.inflight.update(paths)
        self.active_writers += 1
        return claimed, 0

    def _execute_cloud_rows(self, paths, retry_at=None):
        # Called inside the owner's transaction; bounded SQL placeholders.
        for offset in range(0, len(paths), 128):
            chunk = paths[offset:offset+128]
            placeholders = ','.join(['?']*len(chunk))
            if retry_at is None:
                if self.path_kind == 'final-record':
                    self.db.execute(
                        'INSERT OR IGNORE INTO published('
                        'path,parent,input_record_uid,published) '
                        f'SELECT path,parent,input_record_uid,? FROM accepted '
                        f'WHERE path IN ({placeholders})',
                        (time.time(), *chunk),
                    )
                self.db.execute(f'DELETE FROM jobs WHERE path IN ({placeholders})', tuple(chunk))
            else:
                self.db.execute(f'UPDATE jobs SET attempts=attempts+1,retry_at=? WHERE path IN ({placeholders})',
                                (retry_at, *chunk))

    def _update_cloud_rows(self, paths, retry_at=None):
        # Legacy unbatched mode retains its synchronous cloud acknowledgement.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self._execute_cloud_rows(paths, retry_at)
            self.db.execute('COMMIT')
            self.database_transactions += 1
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _apply_cloud_ack(self, ack):
        """Publish only after the transaction containing this ack commits."""
        paths = [row[0] for row in ack.claimed]
        self.inflight.difference_update(paths)
        self.active_writers -= 1
        if ack.error_name is None:
            self.pending -= len(ack.claimed)
            for path in paths:
                self.created_by_path.pop(path, None)
                del self.jobs[path]
            self.pending_bytes -= sum(row[2] for row in ack.claimed)
            self.cloud_commits += len(ack.claimed)
            self.cloud_objects += 1
            self.last_success = time.time()
            self.last_error = None
        else:
            self.errors += 1
            self.last_error = ack.error_name
            for path in paths:
                self.jobs[path].attempts += 1
                self.jobs[path].retry_at = ack.retry_at
                self._schedule(path)
            delay = min(60, 2**min(max(row[3] for row in ack.claimed)+1, 6))
            print(f'checkpoint_sync_retry type={ack.error_name} delay_seconds={delay}', flush=True)

    def _worker_loop(self):
        while True:
            with self.condition:
                if self.stopping:
                    return
                claimed, delay = self._claim_batch(time.time())
                if not claimed:
                    self.condition.wait(timeout=delay)
                    continue
                paths = [row[0] for row in claimed]
                self._publish_snapshot()
            error_name = None
            try:
                if self.batch_writer is not None:
                    self.batch_writer([(self.output/path, json.loads(payload)) for path,payload,_,_ in claimed])
                else:
                    self.writer(self.output/claimed[0][0], json.loads(claimed[0][1]))
            except Exception as error:
                # Exception strings may include annotation content or secrets.
                error_name = type(error).__name__
            delay = min(60, 2**min(max(row[3] for row in claimed)+1, 6))
            ack = _CloudAck(claimed, error_name, time.time()+delay if error_name else None)
            if self.local_thread:
                with self.condition:
                    if self.worker_failure:
                        raise RuntimeError('checkpoint_commit_owner_failed')
                    self.cloud_ack_queue.append(ack)
                    self._publish_snapshot()
                    self.local_condition.notify()
                # Each cloud writer holds at most one completion. This queue
                # is bounded by workers, even when local persistence is slow.
                ack.ready.wait()
                if ack.error:
                    raise ack.error
                continue
            with self.condition:
                self._update_cloud_rows(paths, ack.retry_at)
                self._apply_cloud_ack(ack)
                self._publish_health()
                self._publish_snapshot()
                self.condition.notify_all()

    def flush_checkpoint(self, path, timeout=300):
        """A read/delete barrier: never load stale cloud data or resurrect a unit."""
        from durable_jsonl import unit_directory
        prefix = str(unit_directory(Path(path).absolute()).relative_to(self.output))+'/'
        deadline = time.monotonic()+timeout
        with self.condition:
            while (any(path.startswith(prefix) for path in self.local_queue)
                   or any(path.startswith(prefix) for path in self.jobs)):
                if time.monotonic() >= deadline:
                    raise TimeoutError('checkpoint_cloud_barrier_timeout_local_data_retained')
                self.condition.wait(timeout=1)

    def _publish_snapshot(self):
        # Called only while database state is serialized (or at startup).
        # Lazy heap deletion makes the oldest-pending calculation cheap;
        # compact it if an old stalled item outlives many newer jobs.
        if len(self.created_heap) > max(1024, 2*len(self.created_by_path)):
            self.created_heap = [(created, path) for path, created in self.created_by_path.items()]
            heapq.heapify(self.created_heap)
        while self.created_heap and self.created_by_path.get(self.created_heap[0][1]) != self.created_heap[0][0]:
            heapq.heappop(self.created_heap)
        oldest = self.created_heap[0][0] if self.created_heap else None
        self._metrics_snapshot = {
            'mode': 'local-sqlite-outbox', 'path_kind': self.path_kind,
            'pending_files': self.pending,
            'pending_bytes': self.pending_bytes, 'inflight_syncs': len(self.inflight),
            'inflight_units': len(self.inflight), 'active_cloud_writers': self.active_writers,
            'cloud_writer_limit': len(self.threads),
            'max_pending_files': self.max_files, 'max_pending_bytes': self.max_bytes,
            '_oldest_pending_at': oldest, 'snapshot_updated_at_unix': time.time(),
            'local_queued_files': len(self.local_queue), 'local_queued_bytes': self.local_queued_bytes,
            'local_batch_size': self.local_batch_size,
            'claim_index_mode': 'bounded-memory/v1', 'ready_files': len(self.ready),
            'indexed_payload_bytes': self.pending_bytes,
            'local_transactions_this_process': self.local_transactions,
            'database_writer_mode': 'single-owner-local-and-cloud/v1' if self.local_thread else 'legacy-synchronous',
            'database_transactions_this_process': self.database_transactions,
            'database_io_lock_mode': 'outside-queue-mutex/v1' if self.local_thread else 'legacy-synchronous',
            'last_database_seconds': self.last_database_seconds,
            'max_database_seconds': self.max_database_seconds,
            'queued_cloud_acknowledgements': len(self.cloud_ack_queue),
            'local_commits_this_process': self.local_commits,
            'cloud_commits_this_process': self.cloud_commits,
            'cloud_objects_this_process': self.cloud_objects,
            'accepted_final_records': len(self.accepted),
            'batch_size': self.batch_size if self.batch_writer else 1,
            'sync_errors': self.errors, 'last_error_type': self.last_error,
            'worker_failure_type': self.worker_failure, 'spool_root': str(self.root)}

    def snapshot(self):
        # Copy one published state without competing for the SQLite lock.
        result = self._metrics_snapshot.copy()
        oldest = result.pop('_oldest_pending_at')
        result['oldest_pending_seconds'] = max(0, time.time()-oldest) if oldest else 0
        return result

    def close(self, timeout=300):
        deadline = time.monotonic()+timeout
        with self.condition:
            while (self.pending or self.local_queue) and time.monotonic() < deadline and not self.stopping:
                self.condition.wait(timeout=min(1, max(0, deadline-time.monotonic())))
            self.stopping = True
            self.condition.notify_all()
            self.local_condition.notify_all()
        # Do not close the database beneath a still-running cloud write. The
        # exclusive lock remains held until process exit in that case.
        join_deadline = max(deadline, time.monotonic()+1)
        all_threads = self.threads+([self.local_thread] if self.local_thread else [])
        for thread in all_threads:
            thread.join(timeout=max(0, join_deadline-time.monotonic()))
        live = any(thread.is_alive() for thread in all_threads)
        if not live:
            self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            self.db.close()
            os.close(self.lock_fd)
        if self.pending or live:
            raise TimeoutError('checkpoint_sync_shutdown_incomplete_local_data_retained')
