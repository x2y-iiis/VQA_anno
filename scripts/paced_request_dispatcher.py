"""Grant paced permits without depending on the previous request thread waking."""
from collections import deque
import threading
import time


class PacedRequestDispatcher:
    mode = 'independent-paced-dispatch/v3-bounded-catchup'
    # CPython can deschedule this single dispatcher for tens of milliseconds
    # when thousands of request threads are runnable.  Crediting a small,
    # bounded slice of that delay preserves the configured average start rate
    # without releasing the entire queue as one provider-visible burst.
    catchup_window_seconds = .04
    max_grant_batch = 16

    def __init__(self, pacer):
        self.pacer = pacer
        self.condition = threading.Condition(pacer.lock)
        self.pending = deque()
        self.thread = None
        self.grants = 0
        self.cancelled = 0
        self.immediate_grants = 0
        self.catchup_batches = 0
        self.catchup_permits = 0
        self.peak_grant_batch = 1

    def snapshot_unlocked(self):
        """Called while holding the pacer's shared lock."""
        return {'dispatcher_mode': self.mode, 'queued_tickets': len(self.pending),
                'dispatched_permits': self.grants, 'cancelled_tickets': self.cancelled,
                'immediate_grants': self.immediate_grants,
                'catchup_batches': self.catchup_batches,
                'catchup_permits': self.catchup_permits,
                'peak_grant_batch': self.peak_grant_batch}

    def wait(self, check=None):
        if check is not None:
            check()
        with self.condition:
            now = time.monotonic()
            self.pacer._recover(now)
            if not self.pending and self.pacer.next_start <= now:
                # A ready permit with no older waiter needs no dispatcher
                # handoff. Retain exact positive spacing and no catch-up credit.
                self.pacer.next_start = now + self.pacer.interval_seconds
                self.grants += 1
                self.immediate_grants += 1
                ticket = None
            else:
                ticket = {'event': threading.Event(), 'granted': False}
                self.pending.append(ticket)
                if self.thread is None:
                    self.thread = threading.Thread(target=self._run,
                                                   name='request-start-dispatch', daemon=True)
                    self.thread.start()
                self.condition.notify()
        if ticket is None:
            if check is not None:
                check()
            return
        try:
            while not ticket['event'].wait(1):
                if check is not None:
                    check()
            if check is not None:
                check()
        finally:
            # Grant is terminal and published before notification. Only a
            # cancellation needs to acquire the shared mutex; recheck inside
            # it because dispatch may win the race with cancellation.
            if not ticket['granted']:
                with self.condition:
                    if not ticket['granted']:
                        self.pending.remove(ticket)
                        self.cancelled += 1
                        self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                if not self.pending:
                    self.condition.wait(60)
                    if not self.pending:
                        self.thread = None
                        return
                now = time.monotonic()
                self.pacer._recover(now)
                delay = self.pacer.next_start - now
                if delay > 0:
                    self.condition.wait(min(delay, 1))
                    continue
                interval = self.pacer.interval_seconds
                batch_size = 1
                if interval > 0:
                    # Discard old debt, but preserve at most a 40 ms scheduling
                    # slip.  At the current Ark ramp rates this grants a small
                    # cohort, never an unbounded queue flush.
                    credited_lateness = min(
                        max(0.0, now - self.pacer.next_start),
                        self.catchup_window_seconds,
                    )
                    batch_size += int(credited_lateness / interval)
                else:
                    batch_size = self.max_grant_batch
                batch_size = min(len(self.pending), self.max_grant_batch, batch_size)
                tickets = [self.pending.popleft() for _ in range(batch_size)]
                for ticket in tickets:
                    ticket['granted'] = True
                self.grants += batch_size
                if batch_size > 1:
                    self.catchup_batches += 1
                    self.catchup_permits += batch_size - 1
                    self.peak_grant_batch = max(self.peak_grant_batch, batch_size)
                # Schedule the next cohort from now.  No debt older than the
                # bounded cohort survives into another dispatcher iteration.
                self.pacer.next_start = now + interval
            # No provider/storage checks or HTTP work happen in this dispatcher.
            # Granted waiters are notified outside the shared mutex.
            for ticket in tickets:
                ticket['event'].set()
