"""Shared request admission and independent LAS/downstream scheduling."""
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
import threading
import hashlib
import json
import os
import time
from pathlib import Path


class RequestAdmission:
    """FIFO admission before serialization, bounded by count and working bytes.

    Waiters use directed events instead of ``Condition.notify_all``.  The old
    broadcast implementation woke thousands of non-head FIFO waiters for every
    grant/release.  Under large STA runs that produced quadratic lock/GIL
    contention while most of the configured byte budget remained unused.
    """
    def __init__(self, maximum=256, byte_budget=12*1024**3):
        self.maximum = maximum
        self.byte_budget = byte_budget
        self.active = self.peak_active = self.bytes = self.peak_bytes = 0
        self.by_task = {}
        self.waiters = deque()
        self.condition = threading.Condition(threading.Lock())

    def _grant_waiters_locked(self):
        """Reserve every currently available permit before waking its owner.

        Waking only the FIFO head made a large fleet refill one OS thread at a
        time.  With thousands of waiters, the next thread could remain
        descheduled long enough for an 8k-wide gate to drain to a few hundred
        active requests despite ample memory and provider capacity.  Assigning
        permits while holding the queue lock keeps FIFO and byte accounting
        exact, but removes that scheduler-dependent hand-off chain.
        """
        while self.waiters and self.active < self.maximum:
            ticket = self.waiters[0]
            weight = ticket['weight']
            if self.bytes + weight > self.byte_budget:
                break
            self.waiters.popleft()
            ticket['granted'] = True
            self.active += 1
            self.bytes += weight
            task = ticket['task']
            self.by_task[task] = self.by_task.get(task, 0) + 1
            self.peak_active = max(self.active, self.peak_active)
            self.peak_bytes = max(self.bytes, self.peak_bytes)
            ticket['event'].set()

    def _release_grant_locked(self, ticket):
        if not ticket['granted']:
            return
        ticket['granted'] = False
        self.active -= 1
        self.bytes -= ticket['weight']
        task = ticket['task']
        self.by_task[task] -= 1
        self._grant_waiters_locked()

    @contextmanager
    def admit(self, task, weight=0):
        check = getattr(self, 'check_available', None)
        if check is not None:
            check()
        weight = max(0, int(weight))
        if weight > self.byte_budget:
            raise ValueError('single_request_exceeds_working_memory_budget')
        ticket = {
            'event': threading.Event(),
            'task': task,
            'weight': weight,
            'granted': False,
        }
        with self.condition:
            self.waiters.append(ticket)
            self._grant_waiters_locked()
        try:
            while not ticket['event'].wait(timeout=1):
                # Storage/provider checks must never hold the shared gate lock.
                if check is not None:
                    check()
            if check is not None:
                check()
        except BaseException:
            with self.condition:
                if ticket['granted']:
                    self._release_grant_locked(ticket)
                else:
                    try:
                        self.waiters.remove(ticket)
                    except ValueError:
                        pass
                    self._grant_waiters_locked()
            raise
        try:
            yield
        finally:
            with self.condition:
                self._release_grant_locked(ticket)

    def snapshot(self):
        with self.condition:
            return {'http_cap': self.maximum, 'active_requests': self.active,
                    'peak_active_requests': self.peak_active,
                    'queued_requests': len(self.waiters),
                    'reserved_working_bytes': self.bytes, 'peak_working_bytes': self.peak_bytes,
                    'working_byte_budget': self.byte_budget,
                    'active_by_task': dict(self.by_task)}


class PartitionedRequestAdmission:
    """Reserve memory per task family while retaining one provider HTTP ceiling.

    Large ECoT video requests cannot consume the GRD reserve or stand ahead of
    small GRD requests in a weighted FIFO. Total reserved bytes never exceed the
    original budget; this does not relax serialization or HTTP concurrency caps.
    """
    def __init__(self, maximum, byte_budget, ecot_byte_budget):
        if not 0 < ecot_byte_budget < byte_budget:
            raise ValueError('ecot_memory_partition_must_be_inside_total_budget')
        self.maximum, self.byte_budget = maximum, byte_budget
        self.slots = RequestAdmission(maximum, byte_budget)
        self.lanes = {
            'ecot': RequestAdmission(maximum, ecot_byte_budget),
            'grd': RequestAdmission(maximum, byte_budget - ecot_byte_budget),
        }
        self.peak_bytes = 0
        self.lock = threading.Lock()

    @contextmanager
    def admit(self, task, weight=0):
        check = getattr(self, 'check_available', None)
        if check is not None:
            check()
        lane = self.lanes['ecot' if task == 'ecot' else 'grd']
        if check is not None:
            lane.check_available = self.slots.check_available = check
        with lane.admit(task, weight), self.slots.admit(task, 0):
            if check is not None:
                check()
            self.snapshot()
            yield

    def snapshot(self):
        result = self.slots.snapshot()
        lanes = {name: gate.snapshot() for name, gate in self.lanes.items()}
        reserved = sum(value['reserved_working_bytes'] for value in lanes.values())
        with self.lock:
            self.peak_bytes = max(self.peak_bytes, reserved)
            peak = self.peak_bytes
        result.update(
            reserved_working_bytes=reserved, peak_working_bytes=peak,
            queued_requests=result['queued_requests'] + sum(value['queued_requests'] for value in lanes.values()),
            memory_partitions=lanes,
        )
        return result


def run_stage_jobs(jobs, upstream, downstream, finish, las_workers, video_workers, max_pending,
                   needs_upstream=None, las_executor=None):
    """LAS waits cannot occupy video workers; forward each completed LAS immediately.

    Jobs are (uid, needed_tasks, source, blobs). Pending queues retain only lazy
    source descriptors. No worker waits for another pool's future.
    """
    pending = {}
    las_context = (nullcontext(las_executor) if las_executor is not None else
                   ThreadPoolExecutor(las_workers, thread_name_prefix='las-stage'))
    with las_context as las_pool, \
         ThreadPoolExecutor(video_workers, thread_name_prefix='video-stage') as video_pool:
        def submit(job, phase):
            pool, function = (las_pool, upstream) if phase == 'las' else (video_pool, downstream)
            pending[pool.submit(function, job)] = (job, phase)

        def collect():
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job, phase = pending.pop(future)
                uid, needed, source, blobs = job
                succeeded = finish(future, uid, {'subtask'} if phase == 'las' else needed)
                if phase == 'las' and succeeded and needed - {'subtask'}:
                    submit((uid, needed - {'subtask'}, source, blobs), 'video')

        for job in jobs:
            while len(pending) >= max_pending:
                collect()
            required = needs_upstream(job) if needs_upstream is not None else 'subtask' in job[1]
            submit(job, 'las' if required else 'video')
        while pending:
            collect()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_bounded_batches(batches, process, workers=1):
    """Overlap bounded batch contexts without multiplying shared resource gates."""
    if workers == 1:
        for batch in batches:
            yield batch, process(batch)
        return
    with ThreadPoolExecutor(workers, thread_name_prefix='batch-stage') as pool:
        pending = {}
        def collect():
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                batch = pending.pop(future)
                yield batch, future.result()
        for batch in batches:
            while len(pending) >= workers:
                yield from collect()
            pending[pool.submit(process, batch)] = batch
        while pending:
            yield from collect()


def configure_las_admission(annotator, admission, task_root, max_operators=None, transport=None):
    """Wrap transport only; retain vendored LAS prompts, parsing and operators."""
    from las_annotation.client import LASClient
    from las_annotation.postprocessing import DoubaoTextClient
    operator_slots = threading.BoundedSemaphore(max_operators or admission.maximum)
    def check_available():
        check = getattr(admission, 'check_available', None)
        if check is not None:
            check()
    annotator.check_available = check_available

    class AdmittedLASClient(LASClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if transport is not None:
                from dataclasses import replace
                self.config = replace(self.config, poll_interval=0)

        def call_operator(self, request_body):
            # Count an asynchronous operator until completion, not only its
            # short submit HTTP call. Polling reuses this slot without nesting.
            with operator_slots, admission.admit('las_operator', 1024*1024):
                self.operator_admitted = True
                try:
                    return super().call_operator(request_body)
                finally:
                    self.operator_admitted = False

        def _post_json(self, url, payload):
            check = getattr(admission, 'check_available', None)
            if check is not None:
                check()
            if transport is not None:
                return transport.post_json(self, url, payload)
            if getattr(self, 'operator_admitted', False):
                return super()._post_json(url, payload)
            with admission.admit('las_http', 1024*1024):
                return super()._post_json(url, payload)

        def submit_operator(self, request_body):
            digest = hashlib.sha256(json.dumps(request_body, sort_keys=True).encode()).hexdigest()
            self.task_path = Path(task_root)/digest[:2]/f'{digest}.json'
            self.task_record = None
            if self.task_path.exists():
                self.task_record = json.loads(self.task_path.read_text())
                if self.task_record['status'] not in {'FAILED', 'TIMEOUT'}:
                    print(f'las_task_resume request_hash={digest} status={self.task_record["status"]}', flush=True)
                    if transport is not None:
                        transport.observe_task(self.task_record['submission'], submitted=True)
                    return self.task_record['submission']
                os.replace(self.task_path, self.task_path.with_suffix(f'.failed.{time.time_ns()}.json'))
            response = super().submit_operator(request_body)
            if not response.get('metadata', {}).get('task_id'):
                return response
            # Persist no credentials, signed URL or complete request body.
            self.task_record = {'request_hash': digest, 'status': 'SUBMITTED',
                                'submission': {'metadata': {'task_id': response['metadata']['task_id']}},
                                'updated_at_unix': time.time()}
            atomic_json(self.task_path, self.task_record)
            if transport is not None:
                transport.observe_task(response, submitted=True)
            print(f'las_task_submitted request_hash={digest}', flush=True)
            return response

        def poll_operator(self, request_body):
            if transport is not None:
                transport.wait_poll(request_body['task_id'])
            response = super().poll_operator(request_body)
            if transport is not None:
                transport.poll_finished(request_body['task_id'], response)
            status = str(response.get('metadata', {}).get('task_status', '')).upper()
            code = str(response.get('metadata', {}).get('business_code', ''))
            if status == 'COMPLETED' and code not in {'', '0'}:
                status = 'FAILED'
            record = getattr(self, 'task_record', None)
            if record is not None and status != record['status']:
                record.update(status=status, updated_at_unix=time.time())
                atomic_json(self.task_path, record)
                print(f'las_task_status request_hash={record["request_hash"]} status={status}', flush=True)
            return response

    class AdmittedTextClient(DoubaoTextClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if transport is not None:
                import httpx
                self._client = self._client.with_options(
                    max_retries=0,
                    timeout=httpx.Timeout(300, connect=30, write=60, pool=30),
                )

        def call(self, request):
            from rewrite_batches import rewrite_batches
            def call_one(value):
                with admission.admit('las_postprocess', 6*len(json.dumps(value).encode()) + 1024*1024):
                    if transport is not None:
                        return transport.text_call(super(AdmittedTextClient, self).call, value)
                    return super(AdmittedTextClient, self).call(value)
            return rewrite_batches(request, call_one, Path(task_root).parent/'las-step4-chunks')

    annotator.client_factory = AdmittedLASClient
    annotator.postprocess_client_factory = AdmittedTextClient
    annotator.release_published_media = True
    if transport is not None:
        from las_recovery import invalidate_unparseable_response
        annotator.on_failed_sample = lambda item_dir, error: invalidate_unparseable_response(
            item_dir, error, task_root)


@contextmanager
def report_admission(admission, path, interval=60, extra_metrics=None):
    """Emit real request counters independently of episode completion."""
    stopped = threading.Event()
    def report():
        while not stopped.is_set():
            value = dict(admission.snapshot(), pid=os.getpid(), updated_at_unix=time.time())
            if extra_metrics is not None:
                value.update(extra_metrics())
            try:
                atomic_json(path, value)
                print('request_parallel_heartbeat ' + json.dumps(value), flush=True)
            except OSError as error:
                print(f'request_metrics_write_failed type={type(error).__name__}', flush=True)
            stopped.wait(interval)
    thread = threading.Thread(target=report, daemon=True, name='request-metrics')
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=5)
