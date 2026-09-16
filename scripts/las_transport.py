"""Paced LAS transport with resumable polls and independently bounded remote work."""
from collections import Counter, deque
import json
from pathlib import Path
import random
import threading
import time

import requests


def rate_kind(message):
    value = str(message).lower()
    if any(s in value for s in ('tpm', 'tokens per minute', 'tokenratelimit', 'token rate')):
        return 'tpm'
    if any(s in value for s in ('rpm', 'requests per minute', 'requestratelimit')):
        return 'rpm'
    return 'unknown'


class RatePacer:
    """FIFO, non-bursting request starts; no reservation of distant future slots."""
    def __init__(self, rpm, check=lambda: None):
        self.maximum = self.rpm = float(rpm)
        self.check = check
        self.condition = threading.Condition()
        self.queue = deque()
        self.next_start = self.cooldown_until = 0.
        self.last_backoff = -1e12
        self.last_growth = time.monotonic()
        self.rate_events = 0

    def wait(self):
        ticket = threading.Event()
        with self.condition:
            self.queue.append(ticket)
        try:
            while True:
                self.check()
                with self.condition:
                    now = time.monotonic()
                    if now-max(self.last_backoff, self.last_growth) >= 60:
                        self.rpm = min(self.maximum, max(self.rpm+5, self.rpm*1.1))
                        self.last_growth = now
                    remaining = max(self.next_start, self.cooldown_until)-now
                    if self.queue[0] is ticket and remaining <= 0:
                        self.queue.popleft()
                        self.next_start = now+60/self.rpm
                        if self.queue:
                            self.queue[0].set()
                        return
                    delay = (max(.005, min(1., remaining)) if self.queue[0] is ticket else 1.)
                    ticket.clear()
                # Wake the next head only, not thousands of request waiters.
                ticket.wait(delay)
        except BaseException:
            with self.condition:
                self.queue.remove(ticket)
                if self.queue:
                    self.queue[0].set()
            raise

    def backoff(self, retry_after=5):
        with self.condition:
            now = time.monotonic()
            self.rate_events += 1
            # One burst is one backoff, not 2048 cumulative penalties.
            if now-self.last_backoff >= 30:
                self.rpm = max(30., self.rpm*.7)
                self.last_backoff = now
                self.cooldown_until = now+min(60., max(2., retry_after))
            if self.queue:
                self.queue[0].set()

    def set_maximum(self, value):
        with self.condition:
            old = self.maximum
            self.maximum = float(value)
            self.rpm = min(self.rpm, self.maximum)
            if self.rpm == old and self.maximum > old:
                self.rpm = self.maximum
            if self.queue:
                self.queue[0].set()

    def snapshot(self):
        with self.condition:
            return {'rpm': self.rpm, 'maximum_rpm': self.maximum,
                    'queued': len(self.queue), 'rate_events': self.rate_events,
                    'cooldown_seconds': max(0, self.cooldown_until-time.monotonic())}


class LasTransport:
    def __init__(self, admission, control_path):
        self.admission = admission
        self.control_path = Path(control_path)
        self.lock = threading.RLock()
        self.counts = Counter()
        self.remote = {}
        self.terminal = set()
        self.next_poll = {}
        self.config = {}
        self.last_reload = 0.
        self.events_path = self.control_path.with_name('las-transport-events.jsonl')
        self.pacers = {name: RatePacer(rpm, self.check) for name,rpm in
                       [('submit', 600), ('poll', 1200), ('postprocess', 600)]}
        self.reload(force=True)

    def check(self):
        check = getattr(self.admission, 'check_available', None)
        if check:
            check()
        self.reload()

    def reload(self, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and now-self.last_reload < 3:
                return
            self.last_reload = now
            try:
                config = json.loads(self.control_path.read_text())
                cap = int(config['operator_limit'])
                if not 1 <= cap <= 2048:
                    raise ValueError('operator_limit_out_of_range')
                for name in self.pacers:
                    rpm = float(config.get(name+'_rpm', self.pacers[name].maximum))
                    if not 30 <= rpm <= 12000:
                        raise ValueError('rpm_out_of_range')
                interval = float(config.get('poll_interval_seconds', 30))
                if not 5 <= interval <= 300:
                    raise ValueError('poll_interval_out_of_range')
            except (OSError, ValueError, KeyError, TypeError):
                if force:
                    raise
                return
            self.config = config
            with self.admission.condition:
                self.admission.maximum = cap
                self.admission.condition.notify_all()
            for name, pacer in self.pacers.items():
                pacer.set_maximum(config.get(name+'_rpm', pacer.maximum))

    def record(self, endpoint, status, code='', duration=0., request_id=None):
        kind = rate_kind(code) if status == 429 or rate_kind(code) != 'unknown' else None
        with self.lock:
            self.counts[endpoint+'.responses'] += 1
            self.counts[endpoint+'.status.'+str(status)] += 1
            if kind:
                self.counts[endpoint+'.rate.'+kind] += 1
            row = {'time_unix': time.time(), 'endpoint': endpoint, 'status': status,
                   'business_code': str(code)[:150], 'rate_kind': kind,
                   'elapsed_seconds': round(duration, 3), 'request_id': request_id}
            # Local diagnostic append only: never include keys, URLs or request bodies.
            with self.events_path.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        return kind

    def observe_task(self, result, submitted=False):
        meta = result.get('metadata', {})
        tid = meta.get('task_id')
        if not tid:
            return
        status = str(meta.get('task_status') or ('SUBMITTED' if submitted else 'UNKNOWN')).upper()
        code = str(meta.get('business_code', ''))
        if status == 'COMPLETED' and code not in {'', '0'}:
            status = 'FAILED'
        with self.lock:
            if tid in self.terminal:
                return
            if status in {'COMPLETED','FAILED','TIMEOUT'}:
                self.terminal.add(tid)
                self.remote.pop(tid, None)
                self.next_poll.pop(tid, None)
                self.counts['operator.'+status.lower()] += 1
            else:
                self.remote[tid] = status
        if status in {'FAILED','TIMEOUT'} or (status == 'COMPLETED' and code not in {'','0'}):
            self.record('operator', 200, code, request_id=meta.get('request_id'))
            if rate_kind(code) != 'unknown':
                self.pacers['submit'].backoff(15)

    def wait_poll(self, tid):
        with self.lock:
            due = self.next_poll.get(tid, 0.)
        while time.monotonic() < due:
            self.check()
            time.sleep(min(1., max(0., due-time.monotonic())))

    def poll_finished(self, tid, result):
        self.observe_task({'metadata': {**result.get('metadata', {}), 'task_id': tid}})
        with self.lock:
            if tid in self.remote:
                # Budget polls across live remote tasks, not admission waiters.
                period = max(float(self.config.get('poll_interval_seconds', 30)),
                             len(self.remote)*60/max(1, self.pacers['poll'].rpm*.85))
                self.next_poll[tid] = time.monotonic()+period*random.uniform(.9, 1.1)

    def post_json(self, client, url, payload):
        from las_annotation.client import LASClientError
        endpoint = 'poll' if url.rstrip('/').endswith('/poll') else 'submit'
        pacer = self.pacers[endpoint]
        deadline = time.monotonic()+1800
        attempt = 0
        while True:
            pacer.wait()
            attempt += 1
            started = time.monotonic()
            try:
                response = client._session.post(url, json=payload,
                    headers={'Content-Type':'application/json', 'Authorization':f'Bearer {client._api_key}'},
                    timeout=client.config.timeout)
            except (requests.Timeout, requests.ConnectionError) as error:
                self.record(endpoint, 'network_error', type(error).__name__, time.monotonic()-started)
                if attempt >= 6 or time.monotonic() >= deadline:
                    # Poll retry will resume the saved task ID; never resubmit here.
                    raise LASClientError(f'LAS {endpoint} network error: {type(error).__name__}') from error
                time.sleep(min(20, 2**min(attempt, 4))+random.random())
                continue
            try:
                result = response.json()
            except ValueError:
                result = {}
            meta = result.get('metadata', {}) if isinstance(result, dict) else {}
            code = meta.get('business_code', '')
            self.record(endpoint, response.status_code, code, time.monotonic()-started, meta.get('request_id'))
            if response.status_code == 429:
                try:
                    retry_after = float(response.headers.get('Retry-After', '5'))
                except (ValueError, TypeError):
                    retry_after = 5
                pacer.backoff(retry_after)
                if time.monotonic() < deadline:
                    continue
            elif response.status_code in {500,502,503,504} and attempt < 8:
                time.sleep(min(20, 2**min(attempt, 4))+random.random())
                continue
            if not response.ok:
                raise LASClientError(f'LAS {endpoint} HTTP {response.status_code}: {response.text[:2000]}')
            if not isinstance(result, dict) or not isinstance(result.get('metadata'), dict):
                raise LASClientError(f'LAS {endpoint} response schema invalid')
            return result

    def text_call(self, call, request):
        for attempt in range(8):
            self.pacers['postprocess'].wait()
            started = time.monotonic()
            try:
                value = call(request)
            except Exception as error:
                status = getattr(error, 'status_code', None)
                kind = rate_kind(str(error))
                # Log only the classification; SDK errors may embed request data.
                self.record('postprocess', status or 'error', kind if kind != 'unknown' else type(error).__name__, time.monotonic()-started)
                if status == 429 and attempt < 7:
                    self.pacers['postprocess'].backoff(10)
                    continue
                if status in {500,502,503,504} and attempt < 7:
                    time.sleep(min(20, 2**attempt)+random.random())
                    continue
                if type(error).__name__ in {'APITimeoutError', 'APIConnectionError'} and attempt < 7:
                    time.sleep(min(20, 2**attempt)+random.random())
                    continue
                raise
            self.record('postprocess', 200, duration=time.monotonic()-started)
            return value

    def snapshot(self):
        self.reload()
        with self.lock:
            return {'control': self.config, 'counts': dict(self.counts),
                    'remote_unfinished': len(self.remote),
                    'remote_by_status': dict(Counter(self.remote.values())),
                    'endpoints': {k:v.snapshot() for k,v in self.pacers.items()}}
