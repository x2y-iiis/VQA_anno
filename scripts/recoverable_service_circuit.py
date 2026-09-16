"""Pause one failing model and allow one bounded half-open inference attempt."""
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import math
import threading
import time
import urllib.error


class ServiceAdmissionExpired(Exception):
    """An unsent attempt must reacquire its invalidated service permit."""


def transient_service_status(status):
    return status in {500, 502, 503, 504}


def transient_transport_timeout(error):
    """Match typed socket timeouts, never infer billing/auth from error text."""
    return isinstance(error, TimeoutError) or (
        isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError))


def retry_after_delay(headers):
    value = (headers or {}).get('Retry-After')
    if not value:
        return 0
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, delay) if math.isfinite(delay) else 0


class RecoverableServiceCircuit:
    def __init__(self, threshold=20, cooldown=60, clock=time.monotonic):
        if threshold < 1 or cooldown <= 0:
            raise ValueError('invalid_service_circuit_configuration')
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.condition = threading.Condition()
        self.state, self.generation = 'closed', 0
        self.failures = self.openings = self.recoveries = self.waiters = 0
        self.service_errors = self.stale_responses = self.probes = 0
        self.transport_timeouts = 0
        self.cancelled_admissions = 0
        self.until = 0
        self._view = (self.generation, self.state, self.failures)

    def _publish_view(self):
        # Writers hold the condition. CPython readers obtain one coherent,
        # immutable reference without joining the shared condition mutex queue.
        self._view = (self.generation, self.state, self.failures)

    def _open(self, delay=0):
        self.state = 'open'
        self.generation += 1
        self.openings += 1
        self.until = self.clock() + max(self.cooldown, delay)
        self._publish_view()
        self.condition.notify_all()

    def finish(self, ticket, result, retry_after=0):
        generation, probe = ticket
        current_generation, state, failures = self._view
        if generation == current_generation and not probe and state == 'closed':
            if result == 'other' or (result == 'success' and failures == 0):
                # This no-op linearizes at the snapshot read. A later failure
                # must not be reset by a success that was already accounted for.
                return
        with self.condition:
            if result == 'admission_cancelled':
                self.cancelled_admissions += 1
                if generation == self.generation and probe:
                    self._open()
                return
            if result == 'transport_timeout':
                self.transport_timeouts += 1
            if result == 'service_error':
                self.service_errors += 1
                if retry_after > 0:
                    remaining = max(0, self.until-self.clock()) if self.state == 'open' else 0
                    self._open(max(remaining, retry_after))
                    return
            if generation != self.generation:
                self.stale_responses += 1
                return
            if probe:
                if result == 'success':
                    self.state = 'closed'
                    self.failures = 0
                    self.generation += 1
                    self.recoveries += 1
                    self.condition.notify_all()
                else:
                    self._open()
            elif self.state == 'closed':
                if result == 'success':
                    self.failures = 0
                elif result in {'service_error', 'transport_timeout'}:
                    self.failures += 1
                    if self.failures >= self.threshold:
                        self._open()
            self._publish_view()

    def allows(self, ticket):
        current_generation, state, _ = self._view
        generation, probe = ticket
        return generation == current_generation and (
            state == 'closed' or (probe and state == 'half_open'))

    def _wait_for_recovery(self, check):
        with self.condition:
            self.waiters += 1
        try:
            while True:
                # Cancellation/provider checks must not run under this mutex.
                if check is not None:
                    check()
                with self.condition:
                    if self.state == 'closed':
                        return (self.generation, False)
                    if self.state == 'open' and self.clock() >= self.until:
                        self.state = 'half_open'
                        self.probes += 1
                        self._publish_view()
                        return (self.generation, True)
                    self.condition.wait(timeout=1)
        finally:
            with self.condition:
                self.waiters -= 1

    @contextmanager
    def admit(self, check=None):
        if check is not None:
            check()
        generation, state, _ = self._view
        ticket = (generation, False) if state == 'closed' else self._wait_for_recovery(check)
        try:
            yield ticket
        except ServiceAdmissionExpired:
            self.finish(ticket, 'admission_cancelled')
            raise
        except urllib.error.HTTPError as error:
            transient = transient_service_status(error.code)
            self.finish(ticket, 'service_error' if transient else 'other',
                        retry_after_delay(error.headers) if transient else 0)
            raise
        except BaseException as error:
            # An interrupted half-open attempt must never strand the gate.
            self.finish(ticket, 'transport_timeout' if transient_transport_timeout(error) else 'other')
            raise
        else:
            self.finish(ticket, 'success')

    def snapshot(self):
        with self.condition:
            return {'mode': 'bounded-half-open/v2-closed-fastpath', 'state': self.state,
                    'closed_state_checks': 'immutable-view/v1',
                    'cooldown_remaining_seconds': max(0, self.until-self.clock()) if self.state == 'open' else 0,
                    'cooldown_seconds': self.cooldown, 'failure_threshold': self.threshold,
                    'consecutive_service_failures': self.failures, 'waiters': self.waiters,
                    'service_errors': self.service_errors, 'openings': self.openings,
                    'transport_timeouts': self.transport_timeouts,
                    'typed_timeout_recovery': True,
                    'pacing_ticket_revalidation': 'cancel-expired/v1',
                    'cancelled_admissions': self.cancelled_admissions,
                    'probes': self.probes, 'recoveries': self.recoveries,
                    'stale_responses': self.stale_responses}
