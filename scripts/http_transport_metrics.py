"""Timed operation counters with transport-specific scope and an optional native backend."""
from contextlib import contextmanager
from collections import deque
import statistics
import threading
import time


class TimedOperationMetrics:
    mode = 'timed-operation/v1'
    scope = 'Elapsed time inside the tracked operation'

    def __init__(self, backend='python'):
        if backend not in {'python', 'auto', 'native'}:
            raise ValueError('unsupported_http_metrics_backend')
        self.native = None
        if backend != 'python':
            from native_http_metrics import NativeHTTPCounters, load_library
            if load_library() is not None:
                self.native = NativeHTTPCounters()
            elif backend == 'native':
                raise RuntimeError('native_http_metrics_not_built')
        self.lock = threading.Lock()
        self.active = self.peak = self.started = self.finished = self.failed = 0
        self.completed_call_seconds = 0.0
        self.recent_call_seconds = deque(maxlen=2048)

    @contextmanager
    def track(self):
        if self.native is not None:
            self.native.start()
        else:
            with self.lock:
                self.active += 1
                self.started += 1
                self.peak = max(self.peak, self.active)
        failed = False
        started_at = time.monotonic()
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.monotonic()-started_at
            if self.native is not None:
                self.native.finish(failed, elapsed)
            else:
                with self.lock:
                    self.active -= 1
                    self.finished += 1
                    self.failed += int(failed)
                    self.completed_call_seconds += elapsed
                    self.recent_call_seconds.append(elapsed)

    def snapshot(self):
        if self.native is not None:
            result, recent = self.native.snapshot()
        else:
            with self.lock:
                recent = list(self.recent_call_seconds)
                result = {'active': self.active, 'peak': self.peak, 'started': self.started,
                          'finished': self.finished, 'failed': self.failed,
                          'completed_call_seconds': self.completed_call_seconds}
        result.update(mode=self.mode,
                      counter_backend='gil-held-native/v1' if self.native is not None else 'python-mutex/v1',
                      scope=self.scope)
        result['recent_completed_call_samples'] = len(recent)
        result['recent_call_median_seconds'] = statistics.median(recent) if recent else None
        return result


class HTTPTransportMetrics(TimedOperationMetrics):
    mode = 'transport-call-through-body-read/v1'
    scope = 'Includes connection/upload/read; excludes admission and response parsing'
