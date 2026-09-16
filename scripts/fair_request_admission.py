"""Separate bounded serialization memory from fair per-attempt HTTP slots."""
from collections import Counter, deque
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import threading
import time



class CachedAvailabilityCheck:
    """Coalesce expensive provider/storage checks outside all admission locks."""
    def __init__(self, check, interval=1, notification_backend='private-locks'):
        self.check, self.interval = check, interval
        if notification_backend not in {'private-locks', 'auto', 'native-futex'}:
            raise ValueError('unsupported_availability_notification_backend')
        self._latch_factory = None
        if notification_backend != 'private-locks':
            from native_availability_latch import NativeRefreshLatch, load_library
            if load_library() is not None:
                self._latch_factory = NativeRefreshLatch
            elif notification_backend == 'native-futex':
                raise RuntimeError('native_availability_latch_not_built')
        self.lock = threading.Lock()
        self._state = (float('-inf'), None)
        self._pending_refresh = None
        self._refresh_thread = None
        self._refresh_stop = None
        self._background_error_type = None
        self._stats = {'mode': ('native-futex-broadcast/v1' if self._latch_factory
                               else 'per-waiter-lock-notification/v1'), 'refreshes': 0,
                       'failed_refreshes': 0, 'last_check_seconds': 0,
                       'max_check_seconds': 0, 'last_notify_seconds': 0,
                       'max_notify_seconds': 0, 'max_waiters': 0}

    @property
    def updated(self):
        return self._state[0]

    @updated.setter
    def updated(self, value):
        self._state = (value, self._state[1])

    @property
    def error(self):
        return self._state[1]

    @staticmethod
    def _notify_waiter(waiter):
        # No shared Event.condition to reacquire after notification. Each
        # follower owns one private, initially locked completion latch.
        waiter.release()

    def snapshot(self):
        pending = self._pending_refresh
        return dict(self._stats, refresh_active=pending is not None,
                    pending_waiters=len(pending) if pending is not None else 0,
                    foreground_max_age_seconds=self.interval,
                    background_refresh_alive=bool(self._refresh_thread and self._refresh_thread.is_alive()),
                    background_error_type=self._background_error_type)

    @contextmanager
    def refreshing(self):
        """Refresh early without changing the foreground freshness deadline."""
        if self.interval <= 0:
            raise ValueError('availability_refresh_interval_must_be_positive')
        with self.lock:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                raise RuntimeError('availability_background_refresh_already_running')
            stop = self._refresh_stop = threading.Event()
            self._background_error_type = None
            def run():
                while not stop.wait(self.interval/4):
                    try:
                        self._check(self.interval/4)
                    except Exception:
                        # Ordinary check failures are cached and still stop
                        # foreground requests. This worker never sends requests.
                        pass
                    except BaseException as error:
                        # Foreground callers retain the synchronous fallback
                        # and the same maximum age if this helper exits.
                        self._background_error_type = type(error).__name__
                        return
            self._refresh_thread = threading.Thread(target=run, name='availability-refresh', daemon=True)
            self._refresh_thread.start()
        try:
            yield self
        finally:
            stop.set()
            self._refresh_thread.join(timeout=5)

    def __call__(self):
        return self._check(self.interval)

    def _check(self, max_age):
        # One immutable snapshot gives coherent timestamp/error reads without
        # making thousands of cache hits queue on the refresh mutex.
        updated, error = self._state
        while time.monotonic() - updated >= max_age:
            leader = False
            with self.lock:
                updated, error = self._state
                if time.monotonic() - updated < max_age:
                    break
                pending = self._pending_refresh
                if pending is None:
                    pending = self._pending_refresh = self._latch_factory() if self._latch_factory else []
                    leader = True
                    refresh_id = self._stats['refreshes']+1
                    self._stats = dict(self._stats, refreshes=refresh_id)
                else:
                    if self._latch_factory:
                        pending.waiters += 1
                        waiter = pending
                    else:
                        waiter = threading.Lock()
                        waiter.acquire()
                        pending.append(waiter)
            if leader:
                started = time.monotonic()
                completed = False
                try:
                    try:
                        self.check()
                        error = None
                    except Exception as caught:
                        error = caught
                    self._state = (time.monotonic(), error)
                    completed = True
                finally:
                    checked = time.monotonic()
                    with self.lock:
                        self._pending_refresh = None
                        self._stats = dict(self._stats,
                            failed_refreshes=self._stats['failed_refreshes']+int(not completed or error is not None),
                            last_check_seconds=checked-started,
                            max_check_seconds=max(self._stats['max_check_seconds'], checked-started),
                            max_waiters=max(self._stats['max_waiters'], len(pending)))
                    if self._latch_factory:
                        pending.notify()
                    else:
                        for ticket in pending:
                            self._notify_waiter(ticket)
                    notify_seconds = time.monotonic()-checked
                    with self.lock:
                        self._stats = dict(self._stats,
                            max_notify_seconds=max(self._stats['max_notify_seconds'], notify_seconds),
                            last_notify_seconds=(notify_seconds if self._stats['refreshes'] == refresh_id
                                                 else self._stats['last_notify_seconds']))
            else:
                # A follower can proceed as soon as its own latch is released,
                # without waiting for the leader to notify every other caller.
                if self._latch_factory:
                    waiter.wait()
                else:
                    with waiter:
                        pass
            updated, error = self._state
        if error is not None:
            # Do not append thousands of waiter tracebacks to one shared error.
            raise type(error)(*error.args)


class FairNetworkSlots:
    """Wake only granted tickets; protect GRD from Flash request monopolization.

    A quarter is ECoT's contention share, not a hard ceiling. Either lane
    borrows all idle capacity. When demand returns, new grants restore the
    shares without cancelling already admitted requests.
    """
    def __init__(self, maximum):
        self.lock = threading.Lock()
        self.maximum = maximum
        self.ecot_limit = None
        self.active = self.peak = 0
        self.by_lane, self.by_task = Counter(), Counter()
        self.queues = {'ecot': deque(), 'grd': deque()}
        self.grants = Counter()

    def _grant(self):
        wakeups = []
        while self.active < self.maximum:
            ecot_ready = self.queues['ecot'] and self.by_lane['ecot'] < (
                self.maximum if self.ecot_limit is None else min(self.maximum, self.ecot_limit))
            lane = ('ecot' if ecot_ready and (
                        not self.queues['grd']
                        or self.by_lane['ecot'] < max(1, self.maximum // 4))
                    else 'grd' if self.queues['grd'] else None)
            if lane is None:
                break
            ticket = self.queues[lane].popleft()
            ticket['granted'] = True
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.by_lane[lane] += 1
            self.by_task[ticket['task']] += 1
            self.grants[ticket['task']] += 1
            if ticket['event'] is not None:
                wakeups.append(ticket['event'])
        return wakeups

    @staticmethod
    def _notify(wakeups):
        # Event notification can hand the GIL to its waiter. Never do this
        # while holding the shared admission lock or notify an immediate grant.
        for event in wakeups:
            event.set()

    def set_limit(self, maximum, ecot_limit=None):
        if maximum < 1:
            raise ValueError('http_limit_must_be_positive')
        if ecot_limit is not None and not 1 <= ecot_limit <= maximum:
            raise ValueError('ecot_limit_must_fit_global_limit')
        with self.lock:
            self.maximum = maximum
            self.ecot_limit = ecot_limit
            # Existing requests drain naturally when lowering the cap.
            wakeups = self._grant()
        self._notify(wakeups)

    def _release(self, lane, ticket):
        with self.lock:
            if ticket['granted']:
                self.active -= 1
                self.by_lane[lane] -= 1
                self.by_task[ticket['task']] -= 1
            else:
                self.queues[lane].remove(ticket)
            wakeups = self._grant()
        self._notify(wakeups)

    @contextmanager
    def admit(self, task, check=None):
        if check:
            check()
        lane = 'ecot' if task == 'ecot' else 'grd'
        ticket = {'task': task, 'event': None, 'granted': False}
        with self.lock:
            self.queues[lane].append(ticket)
            wakeups = self._grant()
            if not ticket['granted']:
                ticket['event'] = threading.Event()
        try:
            self._notify(wakeups)
            while ticket['event'] is not None and not ticket['event'].wait(.5):
                if check:
                    check()
            if check:
                check()
            yield
        finally:
            self._release(lane, ticket)

    def snapshot(self):
        with self.lock:
            return {'http_cap': self.maximum, 'active_requests': self.active,
                    'network_grant_mode': 'immediate-or-directed-outside-lock/v1',
                    'peak_active_requests': self.peak, 'active_by_task': dict(self.by_task),
                    'active_by_lane': dict(self.by_lane),
                    'queued_requests': sum(map(len, self.queues.values())),
                    'queued_by_lane': {k: len(v) for k, v in self.queues.items()},
                    'ecot_http_ceiling': self.maximum if self.ecot_limit is None else self.ecot_limit,
                    'ecot_contention_share': max(1, self.maximum // 4),
                    'work_conserving': self.ecot_limit is None or self.ecot_limit >= self.maximum,
                    'http_attempts_admitted': dict(self.grants)}


class MemoryReservations:
    """Weighted FIFO with directed wakeups, never broadcast lock contention."""
    def __init__(self, maximum, byte_budget):
        self.maximum, self.byte_budget = maximum, byte_budget
        self.lock = threading.Lock()
        self.queue = deque()
        self.active = self.bytes = self.peak_active = self.peak_bytes = 0
        self.by_task = Counter()
        self.check_available = None

    def _grant(self):
        wakeups = []
        while (self.queue and self.active < self.maximum
               and self.bytes+self.queue[0]['weight'] <= self.byte_budget):
            ticket = self.queue.popleft()
            ticket['granted'] = True
            self.active += 1
            self.bytes += ticket['weight']
            self.peak_active = max(self.peak_active, self.active)
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.by_task[ticket['task']] += 1
            if ticket['event'] is not None:
                wakeups.append(ticket['event'])
        return wakeups

    @contextmanager
    def admit(self, task, weight):
        weight = max(0, int(weight))
        if weight > self.byte_budget:
            raise ValueError('single_request_exceeds_working_memory_budget')
        check = self.check_available
        if check:
            check()
        ticket = {'task': task, 'weight': weight, 'event': None, 'granted': False}
        with self.lock:
            self.queue.append(ticket)
            wakeups = self._grant()
            if not ticket['granted']:
                ticket['event'] = threading.Event()
        try:
            FairNetworkSlots._notify(wakeups)
            while ticket['event'] is not None and not ticket['event'].wait(.5):
                if check:
                    check()
            if check:
                check()
            yield
        finally:
            with self.lock:
                if ticket['granted']:
                    self.active -= 1
                    self.bytes -= weight
                    self.by_task[task] -= 1
                else:
                    self.queue.remove(ticket)
                wakeups = self._grant()
            FairNetworkSlots._notify(wakeups)

    def snapshot(self):
        with self.lock:
            return {'http_cap': self.maximum, 'active_requests': self.active,
                    'peak_active_requests': self.peak_active, 'queued_requests': len(self.queue),
                    'reserved_working_bytes': self.bytes, 'peak_working_bytes': self.peak_bytes,
                    'working_byte_budget': self.byte_budget, 'active_by_task': dict(self.by_task),
                    'head_waiter_bytes': self.queue[0]['weight'] if self.queue else None,
                    'queue_implementation': 'weighted-ticket-events/v1'}


class FairRequestAdmission:
    """Memory spans serialization/retries; network slots span one HTTP attempt."""
    fair_network = True

    def __init__(self, maximum, byte_budget, ecot_byte_budget, control_path, output,
                 control_maximum=2048, memory_request_slots=2048, network_backend='python'):
        if not 0 < ecot_byte_budget < byte_budget:
            raise ValueError('ecot_memory_partition_must_be_inside_total_budget')
        if (control_maximum not in (2048, 4096, 8192) or not 1 <= maximum <= control_maximum
                or not control_maximum <= memory_request_slots <= 16384):
            raise ValueError('invalid_fair_request_capacity')
        self.control_maximum = control_maximum
        if network_backend not in {'python', 'auto', 'native-ecot'}:
            raise ValueError('unsupported_network_admission_backend')
        self.slots = FairNetworkSlots(maximum)
        if network_backend != 'python':
            from native_ecot_gate import NativeEcotNetworkSlots, load_library
            if load_library() is not None:
                self.slots = NativeEcotNetworkSlots(maximum)
            elif network_backend == 'native-ecot':
                raise RuntimeError('native_ecot_gate_not_built')
        self.lanes = {'ecot': MemoryReservations(memory_request_slots, ecot_byte_budget),
                      'grd': MemoryReservations(2048, byte_budget - ecot_byte_budget)}
        self.byte_budget = byte_budget
        self.check_available = None
        self.control_path = Path(control_path)
        self.output = str(Path(output).absolute())
        self.control_lock = threading.Lock()
        self.control_checked = float('-inf')
        self.control_revision = None
        self.control_applied_at = None
        self.control_error = None
        self.default_python_switch_interval = sys.getswitchinterval()
        self.applied_python_switch_interval = None
        self.metrics_lock = threading.Lock()
        self.annotation_counts = Counter()
        self.waiting_for_pacing = Counter()
        self.control_refresh(force=True)

    def control_refresh(self, force=False):
        if not self.control_lock.acquire(blocking=False):
            return
        try:
            if not force and time.monotonic()-self.control_checked < 1:
                return
            self.control_checked = time.monotonic()
            try:
                value = json.loads(self.control_path.read_text())
                if value.get('schema_version') != 'qwen-http-control/v1' or value.get('output') != self.output:
                    raise ValueError('http_control_identity_mismatch')
                limit = value['max_http_active']
                if (type(limit) is not int or limit not in (256, 512, 1024, 2048, 3072, 4096, 6144, 8192)
                        or limit > self.control_maximum):
                    raise ValueError('unsupported_http_control_limit')
                ecot_limit = value.get('ecot_http_limit')
                if ecot_limit is not None and (type(ecot_limit) is not int or not 1 <= ecot_limit <= limit):
                    raise ValueError('unsupported_ecot_http_limit')
                revision = value['revision']
                if not isinstance(revision, str) or not revision:
                    raise ValueError('invalid_http_control_revision')
                switch_ms = value.get('python_switch_interval_ms')
                if switch_ms is not None and (
                        type(switch_ms) not in (int, float) or not math.isfinite(switch_ms)
                        or not 0.1 <= switch_ms <= 5):
                    raise ValueError('python_switch_interval_ms_must_be_between_0_1_and_5')
                if revision != self.control_revision:
                    if switch_ms is not None or self.applied_python_switch_interval is not None:
                        interval = (self.default_python_switch_interval if switch_ms is None
                                    else switch_ms / 1000)
                        sys.setswitchinterval(interval)
                        self.applied_python_switch_interval = None if switch_ms is None else interval
                    self.slots.set_limit(limit, ecot_limit)
                    self.control_revision = revision
                    self.control_applied_at = time.time()
                    print(f'http_control_applied limit={limit} revision={revision} ecot_limit={ecot_limit} '
                          f'python_switch_interval_ms={sys.getswitchinterval()*1000:g}', flush=True)
                self.control_error = None
            except Exception as error:
                self.control_error = type(error).__name__
                if force:
                    raise
        finally:
            self.control_lock.release()

    @contextmanager
    def reserve(self, task, weight):
        lane = self.lanes['ecot' if task == 'ecot' else 'grd']
        lane.check_available = self.check_available
        with lane.admit(task, weight):
            yield

    @contextmanager
    def pacing(self, task):
        with self.metrics_lock:
            self.waiting_for_pacing[task] += 1
        try:
            yield
        finally:
            with self.metrics_lock:
                self.waiting_for_pacing[task] -= 1

    @contextmanager
    def network(self, task):
        self.control_refresh()
        with self.slots.admit(task, self.check_available):
            yield

    def record_grd_completion(self, accepted):
        with self.metrics_lock:
            self.annotation_counts['grd_reviewed_new'] += 1
            self.annotation_counts['grd_accepted_new' if accepted else 'grd_rejected_new'] += 1

    def snapshot(self):
        self.control_refresh()
        result = self.slots.snapshot()
        lanes = {name: lane.snapshot() for name, lane in self.lanes.items()}
        with self.metrics_lock:
            counts, pacing = dict(self.annotation_counts), dict(self.waiting_for_pacing)
        result.update(scheduler='fair-per-attempt/v3-work-conserving', memory_partitions=lanes,
                      reserved_working_bytes=sum(x['reserved_working_bytes'] for x in lanes.values()),
                      working_byte_budget=self.byte_budget,
                      memory_queued_requests=sum(x['queued_requests'] for x in lanes.values()),
                      pacing_waiters=pacing, annotation_completions=counts,
                      control_revision=self.control_revision, control_applied_at=self.control_applied_at,
                      python_switch_interval_ms=sys.getswitchinterval()*1000,
                      control_error_type=self.control_error, control_path=str(self.control_path))
        checker = getattr(self, 'availability_checker', None)
        if checker is not None:
            result['availability_check'] = checker.snapshot()
        return result
