"""Grant paced permits without depending on the previous request thread waking."""
from collections import deque
import threading
import time
import traceback


class PacedRequestDispatcher:
    mode = 'independent-paced-dispatch/v4-single-watchdog'
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
        self.watchdog_thread = None
        self.grants = 0
        self.cancelled = 0
        self.immediate_grants = 0
        self.catchup_batches = 0
        self.catchup_permits = 0
        self.peak_grant_batch = 1
        self.dispatcher_restarts = 0
        self.stalled_dispatcher_restarts = 0
        self.generation = 0
        self.last_grant_at = time.monotonic()
        self.last_restart_at = 0.0

    def _ensure_watchdog_unlocked(self):
        """Keep one recovery observer per dispatcher, never one per waiter.

        Having every queued request wake once per second and take the shared
        condition lock starves the dispatcher when several thousand requests
        are waiting.  A single watchdog retains dead/stalled-thread recovery
        without creating that lock convoy.
        """
        if self.watchdog_thread is not None and self.watchdog_thread.is_alive():
            return
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name='request-start-watchdog',
            daemon=True,
        )
        self.watchdog_thread.start()

    def _ensure_thread_unlocked(self):
        """Start or replace the dispatcher while ``self.condition`` is held.

        A stale ``Thread`` object can survive after its worker has exited (and
        after a fork).  Treating a non-None object as a live dispatcher leaves
        every subsequent pacing ticket blocked forever.
        """
        now = time.monotonic()
        thread_alive = self.thread is not None and self.thread.is_alive()
        # A Thread object can remain nominally alive while its dispatch loop is
        # no longer making progress (observed after very large worker cohorts
        # and process-pool activity).  If permits are due and the queue has not
        # advanced for five seconds, replace the dispatcher.  Generation checks
        # make a late-recovering predecessor exit before it can grant again.
        stalled = bool(
            thread_alive
            and self.pending
            and self.pacer.next_start <= now
            # Thousands of queued waiters wake at roughly the same time.  A
            # replacement thread still needs to acquire the shared condition;
            # without a restart cooldown each waiter can supersede it before
            # it gets scheduled, creating an endless replacement storm.
            and now - max(self.last_grant_at, self.last_restart_at) >= 5.0
        )
        if thread_alive and not stalled:
            return
        if self.thread is not None:
            self.dispatcher_restarts += 1
        if stalled:
            self.stalled_dispatcher_restarts += 1
        self.generation += 1
        generation = self.generation
        self.last_restart_at = now
        self.thread = threading.Thread(target=self._run, args=(generation,),
                                       name='request-start-dispatch', daemon=True)
        self.thread.start()

    def snapshot_unlocked(self):
        """Called while holding the pacer's shared lock."""
        return {'dispatcher_mode': self.mode, 'queued_tickets': len(self.pending),
                'dispatched_permits': self.grants, 'cancelled_tickets': self.cancelled,
                'immediate_grants': self.immediate_grants,
                'catchup_batches': self.catchup_batches,
                'catchup_permits': self.catchup_permits,
                'peak_grant_batch': self.peak_grant_batch,
                'dispatcher_thread_alive': bool(
                    self.thread is not None and self.thread.is_alive()),
                'dispatcher_restarts': self.dispatcher_restarts,
                'stalled_dispatcher_restarts': self.stalled_dispatcher_restarts,
                'last_restart_age_seconds': (
                    max(0.0, time.monotonic() - self.last_restart_at)
                    if self.last_restart_at else None),
                'last_grant_age_seconds': max(0.0, time.monotonic() - self.last_grant_at)}

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
                self._ensure_thread_unlocked()
                self._ensure_watchdog_unlocked()
                self.condition.notify()
        if ticket is None:
            if check is not None:
                check()
            return
        try:
            while not ticket['event'].wait(1):
                # Recovery is deliberately owned by one watchdog.  Thousands
                # of waiters taking this condition here formed a lock convoy
                # that prevented the dispatcher itself from making progress.
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

    def _watchdog_loop(self):
        while True:
            time.sleep(1)
            with self.condition:
                if not self.pending:
                    return
                self._ensure_thread_unlocked()
                self.condition.notify()

    def _run(self, generation):
        current = threading.current_thread()
        try:
            self._dispatch_loop(generation)
        except Exception:
            traceback.print_exc()
        finally:
            with self.condition:
                if self.thread is current and self.generation == generation:
                    self.thread = None
                # Pending waiters must recover even when the dispatcher exits
                # unexpectedly and no new request arrives.
                if self.pending and self.generation == generation:
                    self._ensure_thread_unlocked()
                    self.condition.notify_all()

    def _dispatch_loop(self, generation):
        while True:
            with self.condition:
                if generation != self.generation:
                    return
                if not self.pending:
                    self.condition.wait(60)
                    if generation != self.generation:
                        return
                    if not self.pending:
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
                self.last_grant_at = now
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
