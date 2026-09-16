"""Bounded best-effort rate telemetry; never persist advisory state under admission locks."""
import json
import os
from pathlib import Path
import threading
import time


class CoalescedRateState:
    def __init__(self, path, interval=1.0):
        self.path = Path(path)
        self.interval = interval
        self.condition = threading.Condition()
        self.latest = None
        self.latest_sequence = self.written_sequence = 0
        self.errors = 0
        self.last_error_type = None
        self.stopped = False
        self.thread = None

    def submit(self, state):
        sequence = int(state['rate_limit_events'])
        with self.condition:
            if self.stopped:
                return
            # A caller can be descheduled after releasing the admission lock.
            # Do not let its older snapshot overwrite a newer counter.
            if sequence <= self.latest_sequence:
                return
            self.latest, self.latest_sequence = dict(state), sequence
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='rate-state-writer', daemon=True)
                self.thread.start()
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {'mode': 'coalesced-advisory/v1', 'submitted_sequence': self.latest_sequence,
                    'written_sequence': self.written_sequence, 'pending': self.latest_sequence > self.written_sequence,
                    'errors': self.errors, 'last_error_type': self.last_error_type}

    def _persist(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f'.{self.path.name}.tmp.{os.getpid()}.{threading.get_ident()}')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.replace(temporary, self.path)

    def _run(self):
        due = 0.0
        while True:
            with self.condition:
                while not self.stopped and (self.latest is None or time.monotonic() < due):
                    self.condition.wait(max(0.001, due-time.monotonic()) if self.latest is not None else None)
                if self.stopped:
                    return
                state, self.latest = self.latest, None
            try:
                self._persist(state)
            except Exception as error:
                with self.condition:
                    self.errors += 1
                    self.last_error_type = type(error).__name__
                    if self.latest is None:
                        self.latest = state
            else:
                with self.condition:
                    self.written_sequence = int(state['rate_limit_events'])
                    self.last_error_type = None
            due = time.monotonic() + self.interval

    def close(self, timeout=1.0):
        """Stop telemetry only; no annotation/checkpoint data is owned here."""
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout)
