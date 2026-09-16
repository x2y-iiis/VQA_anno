"""Process-local shared frame workers and a read-only upstream subtask follower."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import threading
import time


@contextmanager
def frame_executor(client, workers, task=None):
    pools = getattr(client, 'frame_executors', None)
    shared = pools.get(task) if isinstance(pools, dict) else None
    if shared is None:
        shared = getattr(client, 'frame_executor', None)
    if not isinstance(shared, ThreadPoolExecutor):
        with ThreadPoolExecutor(workers) as executor:
            yield executor
        return
    pending = set()
    lock = threading.Lock()

    class Lease:
        def submit(self, *args, **kwargs):
            future = shared.submit(*args, **kwargs)
            with lock:
                pending.add(future)
            def completed(done):
                with lock:
                    pending.discard(done)
            future.add_done_callback(completed)
            return future

    try:
        yield Lease()
    finally:
        # A source's frame/video cache must outlive all its submitted jobs,
        # including when another future raises. Never shut down the shared pool.
        with lock:
            remaining = list(pending)
        if remaining:
            wait(remaining)


class FollowingSubtaskIndex:
    """Follow one upstream shard without holding the entire catalog in memory."""
    def __init__(self, root, source_key, label, pipeline):
        self.pipeline = pipeline
        self.path = root/'shards'/source_key/f'{label}.jsonl'
        self.index = pipeline.SubtaskIndex()
        self.signature = None
        self.lock = threading.Lock()

    def find(self, source):
        uid = str(source['uid'])
        immutable = self.path.with_suffix('.records')/(hashlib.sha256(uid.encode()).hexdigest()+'.jsonl')
        if immutable.is_file():
            with immutable.open() as stream:
                row = json.loads(stream.readline())
            extracted = self.pipeline.subtask_from_record(row)
            if extracted is None or uid not in extracted[0]:
                raise ValueError(f'upstream_subtask_identity_mismatch:{uid}')
            return extracted[1]
        with self.lock:
            if self.path.is_file():
                stat = self.path.stat()
                signature = stat.st_size, stat.st_mtime_ns
                if signature != self.signature:
                    self.index = self.pipeline.load_subtask_index([self.path])
                    self.signature = signature
            return self.index.find(source)

    def wait_for(self, source, check_available, interval=30):
        started = time.monotonic()
        announced = False
        while True:
            check_available()
            if self.find(source) is not None:
                if announced:
                    print(f'upstream_subtask_ready uid={source["uid"]} '
                          f'wait_seconds={time.monotonic()-started:.1f}', flush=True)
                return
            if not announced:
                print(f'upstream_subtask_wait uid={source["uid"]} source={self.path}', flush=True)
                announced = True
            time.sleep(interval)
