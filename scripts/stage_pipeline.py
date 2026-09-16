"""Bounded, independent stage lanes with shared asynchronous media preparation."""
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter, deque
import threading


def run_prepared_jobs(jobs, runtime, lane, prepare, run, finish, ready, check, max_pending):
    """Skip missing prerequisites without occupying threads or blocking later jobs.

    The caller revisits deferred batches after scanning other ready batches.
    Only lightweight source descriptors wait in this bounded lookahead queue.
    """
    iterator, backlog, pending = iter(jobs), deque(), {}
    exhausted = False
    deferred = 0
    while not exhausted or backlog or pending:
        check()
        runtime.changed.clear()
        while not exhausted and len(backlog) + len(pending) < max_pending:
            try:
                job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            if not ready(job):
                deferred += 1
                continue
            backlog.append(job)
        while backlog:
            job = backlog[0]
            future = runtime.try_submit(
                lane, job[0], lambda job=job: prepare(job),
                lambda value, job=job: run(job, value),
            )
            if future is None:
                break
            backlog.popleft()
            pending[future] = job
        for future in list(pending):
            if future.done():
                job = pending.pop(future)
                finish(future, job[0], job[1])
        if backlog or pending:
            runtime.changed.wait(.2)
    return deferred


class StagePipeline:
    """Prepare ahead of HTTP work; one stage's tail never holds another's slot.

    Resource contexts are shared by key while leased. Preparation happens in
    its own executor, never under an HTTP gate. Queues are bounded separately
    per lane, and resources outlive every consumer, including failing ones.
    """
    def __init__(self, lanes, workers, prepare_workers, prefetch):
        if min(workers, prepare_workers, prefetch) < 1:
            raise ValueError('stage_pipeline_limits_must_be_positive')
        self.workers, self.prefetch = workers, prefetch
        self.lock = threading.RLock()
        self.changed = threading.Event()
        self.preparers = ThreadPoolExecutor(prepare_workers, thread_name_prefix='media-prepare')
        self.runners = {lane: ThreadPoolExecutor(workers, thread_name_prefix=f'{lane}-episode')
                        for lane in lanes}
        self.entries = {}
        self.pending, self.running, self.prepared = Counter(), Counter(), Counter()
        self.metrics = Counter()
        self.preparing = 0
        self.closed = False

    def try_submit(self, lane, key, prepare, run):
        with self.lock:
            if self.closed:
                raise RuntimeError('stage_pipeline_closed')
            if self.pending[lane] >= self.workers + self.prefetch:
                return None
            self.pending[lane] += 1
            self.metrics['submitted'] += 1
            outer = Future()
            entry = self.entries.get(key)
            if entry is None:
                entry = {'refs': 0, 'context': None}
                self.entries[key] = entry
                def initialize():
                    with self.lock:
                        self.preparing += 1
                    try:
                        context = prepare()
                        value = context.__enter__()
                        entry['context'] = context
                        return value
                    finally:
                        with self.lock:
                            self.preparing -= 1
                entry['future'] = self.preparers.submit(initialize)
                self.metrics['media_preparations'] += 1
            else:
                self.metrics['media_reuses'] += 1
            entry['refs'] += 1

        def finish(result=None, error=None):
            # Cleanup must finish before the last lease is reported complete.
            cleanup = None
            with self.lock:
                entry['refs'] -= 1
                if entry['refs'] == 0:
                    self.entries.pop(key)
                    cleanup = entry['context']
            try:
                if cleanup is not None:
                    cleanup.__exit__(None, None, None)
            except BaseException as cleanup_error:
                if error is None:
                    error = cleanup_error
            with self.lock:
                self.pending[lane] -= 1
                self.metrics['failed' if error is not None else 'completed'] += 1
            if error is None:
                outer.set_result(result)
            else:
                outer.set_exception(error)
            self.changed.set()

        def execute(value):
            with self.lock:
                self.prepared[lane] -= 1
                self.running[lane] += 1
            try:
                result = run(value)
            except BaseException as error:
                finish(error=error)
            else:
                finish(result=result)
            finally:
                with self.lock:
                    self.running[lane] -= 1

        def ready(future):
            try:
                value = future.result()
            except BaseException as error:
                finish(error=error)
                return
            with self.lock:
                self.prepared[lane] += 1
            self.runners[lane].submit(execute, value)
        entry['future'].add_done_callback(ready)
        return outer

    def snapshot(self):
        with self.lock:
            return {'workers_per_lane': self.workers, 'prefetch_per_lane': self.prefetch,
                    'pending_by_lane': dict(self.pending), 'running_by_lane': dict(self.running),
                    'prepared_by_lane': dict(self.prepared), 'shared_media_entries': len(self.entries),
                    'preparing_active': self.preparing,
                    'prepare_queued': max(0, sum(not e['future'].done() for e in self.entries.values()) - self.preparing),
                    **self.metrics}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        with self.lock:
            self.closed = True
        # Preparation callbacks enqueue consumers; cancel work that has not
        # started after a provider stop, then drain only active preparation
        # before runners. Without cancel_futures a 2k-episode resident window
        # can turn a controlled reload into hours of unnecessary media work.
        self.preparers.shutdown(wait=True, cancel_futures=True)
        for executor in self.runners.values():
            executor.shutdown(wait=True)


class MediaPrefetchPool:
    """Speculative clips are bounded; a full queue never blocks inventory HTTP."""
    def __init__(self, workers, capacity):
        if workers < 1 or capacity < workers:
            raise ValueError('invalid_media_prefetch_limits')
        self.executor = ThreadPoolExecutor(workers, thread_name_prefix='future-media')
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.Lock()
        self.capacity = capacity
        self.outstanding = self.active = self.completed = self.peak = 0

    def submit(self, function, blocking=False):
        if not self.slots.acquire(blocking=blocking):
            return None
        with self.lock:
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
        def work():
            with self.lock:
                self.active += 1
            try:
                return function()
            finally:
                with self.lock:
                    self.active -= 1
                    self.completed += 1
        try:
            return self.executor.submit(work)
        except BaseException:
            self.release()
            raise

    def release(self):
        with self.lock:
            self.outstanding -= 1
        self.slots.release()

    def snapshot(self):
        with self.lock:
            return {'capacity': self.capacity, 'outstanding': self.outstanding,
                    'active': self.active, 'completed': self.completed, 'peak': self.peak}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.executor.shutdown(wait=True)


class DeferredMedia:
    """Start clipping alongside inventory; resolve only before future selection."""
    def __init__(self, pool, function):
        self.pool, self.function = pool, function
        self.future = None
        self.value = None

    def prefetch(self):
        if self.future is None and self.value is None:
            self.future = self.pool.submit(self.function)

    def resolve(self):
        if self.value is None:
            if self.future is None:
                self.future = self.pool.submit(self.function, blocking=True)
            try:
                self.value = self.future.result()
            finally:
                self.future = None
                self.pool.release()
        return self.value

    def close(self):
        if self.future is not None:
            try:
                self.future.result()
            finally:
                self.future = None
                self.pool.release()
        self.value = None
