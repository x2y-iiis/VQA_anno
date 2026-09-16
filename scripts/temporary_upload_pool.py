"""Exclusive reusable HTTP sessions and bounded temporary-upload phase metrics."""
from contextlib import contextmanager
import queue
import threading
import time
import weakref


class SessionPool:
    """Lease each Session to one caller; reuse TCP/TLS without shared mutation."""

    def __init__(self, capacity):
        self.queue = queue.LifoQueue(capacity)
        for _ in range(capacity):
            self.queue.put(None)

    @contextmanager
    def lease(self):
        import requests
        session = self.queue.get()
        try:
            if session is None:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=4, pool_maxsize=1, max_retries=0,
                    pool_block=True,
                )
                session.mount('https://', adapter)
            yield session
        finally:
            if session is not None:
                session.cookies.clear()
            self.queue.put(session)


class UploadMetrics:
    """Single-writer thread buckets; monitoring must not serialize uploads.

    Tuple replacement publishes each phase transition atomically under the
    CPython GIL. A thread never mutates another thread's bucket. The registry
    lock is needed only on a thread's first use and when collecting snapshots,
    not on the repeated phase entry/exit path. Dead-thread totals are retained
    while their buckets are compacted, bounding registry growth across batches.
    """

    mode = 'thread-local/v1'

    def __init__(self):
        self._registry_lock = threading.Lock()
        self._local = threading.local()
        self._buckets = []
        self._retired = {}

    def _bucket(self):
        bucket = getattr(self._local, 'bucket', None)
        if bucket is None:
            bucket = {}
            with self._registry_lock:
                self._buckets.append((weakref.ref(threading.current_thread()), bucket))
            self._local.bucket = bucket
        return bucket

    @staticmethod
    def _merge(target, name, value):
        active, completed, seconds, maximum = value
        previous = target.get(name, (0, 0, 0.0, 0.0))
        target[name] = (previous[0]+active, previous[1]+completed,
                        previous[2]+seconds, max(previous[3], maximum))

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        bucket = self._bucket()
        active, completed, seconds, maximum = bucket.get(name, (0, 0, 0.0, 0.0))
        bucket[name] = (active+1, completed, seconds, maximum)
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            active, completed, seconds, maximum = bucket[name]
            bucket[name] = (active-1, completed+1, seconds+elapsed, max(maximum, elapsed))

    def snapshot(self):
        with self._registry_lock:
            live = []
            for reference, bucket in self._buckets:
                thread = reference()
                if thread is None or not thread.is_alive():
                    for name, value in bucket.items():
                        self._merge(self._retired, name, value)
                else:
                    live.append((reference, bucket))
            self._buckets = live
            values = dict(self._retired)
            buckets = [bucket for _, bucket in live]
        for bucket in buckets:
            # Copy pairs in one C-level operation; phase-name insertion in a
            # different thread must not invalidate a Python dict iterator.
            for name, value in list(bucket.items()):
                self._merge(values, name, value)
        return {name: dict(zip(('active', 'completed', 'seconds', 'max_seconds'), value))
                for name, value in values.items()}
