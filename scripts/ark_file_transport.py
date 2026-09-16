"""Bounded, content-addressed Ark Files transport for Responses media inputs."""
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
from pathlib import Path
import threading
import time
import uuid

import httpx


class ArkFileReference:
    def __init__(self, publisher, source, model, mime_type, digest, size):
        self.publisher, self.source = publisher, source
        self.model, self.mime_type = model, mime_type
        self.sha256, self.size_bytes = digest, size
        self.lock = threading.Lock()
        self.record = None
        self.lease = None
        self.closed = False

    @property
    def identity(self):
        return hashlib.sha256(json.dumps([self.sha256, self.mime_type, self.model,
                                         'default-preprocess-v1']).encode()).hexdigest()

    def acquire(self):
        self.lease = (self.publisher.root / (self.identity + '.lease')).open('a')
        fcntl.flock(self.lease, fcntl.LOCK_SH)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.lease is not None:
                # Explicitly release the shared open-file-description lock even
                # if a concurrent subprocess temporarily inherited its FD.
                fcntl.flock(self.lease, fcntl.LOCK_UN)
                self.lease.close()
                self.lease = None
        if self.publisher is not None:
            try:
                if not self.publisher.delete_unused(self.identity, self.model):
                    self.publisher.defer_cleanup(self.identity, self.model)
            except Exception as error:
                print('ark_file_cleanup_failed error_type=' + type(error).__name__, flush=True)

    def __len__(self):
        # The inference body contains only an opaque ID, never original bytes.
        return 256

    def resolve(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('ark_file_reference_is_closed')
            if not self.record or self.record.get('expire_at', 0) < time.time() + 900:
                self.record = self.publisher.resolve(self)
            return self.record['id']


class ArkFilePublisher:
    def __init__(self, cache_root, api_key_env, endpoint, workers=16, check=None, on_fatal=None):
        if not endpoint.endswith('/chat/completions'):
            raise ValueError('ark_files_requires_chat_completions_base_endpoint')
        self.base_url = endpoint.removesuffix('/chat/completions') + '/'
        self.api_key_env, self.check = api_key_env, check or (lambda: None)
        self.on_fatal, self.fatal_reported = on_fatal, False
        key = os.environ[api_key_env]
        account = hashlib.sha256((self.base_url + key).encode()).hexdigest()
        self.root = Path(cache_root) / account
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.slots = threading.BoundedSemaphore(workers)
        self.status_slots = threading.BoundedSemaphore(128)
        self.client = httpx.Client(base_url=self.base_url, trust_env=False,
                                  headers={'Authorization': 'Bearer ' + key},
                                  limits=httpx.Limits(max_connections=workers,
                                                     max_keepalive_connections=workers),
                                  timeout=httpx.Timeout(180, connect=30, pool=180))
        self.status_client = httpx.Client(base_url=self.base_url, trust_env=False,
            headers={'Authorization': 'Bearer ' + key},
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=128),
            timeout=httpx.Timeout(60, connect=20, pool=60))
        self.cleanup_slots = threading.BoundedSemaphore(64)
        self.cleanup_pacing_lock = threading.Lock()
        self.cleanup_next_start = 0.0
        self.cleanup_interval = .01
        self.cleanup_client = httpx.Client(base_url=self.base_url, trust_env=False,
            headers={'Authorization': 'Bearer ' + key},
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
            timeout=httpx.Timeout(20, connect=10, pool=20))
        self.metrics_lock = threading.Lock()
        self.metrics = {'uploads': 0, 'uploaded_bytes': 0, 'cache_hits': 0,
                        'active_uploads': 0, 'peak_uploads': 0,
                        'waiting_uploads': 0, 'upload_wait_seconds': 0,
                        'upload_transfer_seconds': 0, 'processing_seconds': 0,
                        'processed_files': 0, 'upload_workers': workers,
                        'deleted_files': 0, 'deleted_source_bytes': 0,
                        'cleanup_errors': 0, 'cleanup_deferred_in_use': 0}
        self.pending_cleanup_root = self.root / 'pending-cleanup'
        self.pending_cleanup_root.mkdir(exist_ok=True, mode=0o700)
        self.cleanup_stop = threading.Event()
        self.cleanup_thread = None
        self.cleanup_thread_lock = threading.Lock()
        self.cleanup_cursor = ''

    def defer_cleanup(self, identity, model):
        """Persist a missed cleanup so a transient lease never becomes a leak."""
        self._save(self.pending_cleanup_root / (identity + '.json'),
                   {'model': model, 'generation': uuid.uuid4().hex})

    def reap_pending(self, limit=128):
        available = sorted(self.pending_cleanup_root.glob('*.json'))
        later = [p for p in available if p.name > self.cleanup_cursor]
        paths = (later + [p for p in available if p.name <= self.cleanup_cursor])[:limit]
        if paths:
            self.cleanup_cursor = paths[-1].name
        from concurrent.futures import ThreadPoolExecutor
        def reclaim(path):
            try:
                original = path.read_text()
                model = json.loads(original)['model']
                deleted = self.delete_unused(path.stem, model)
                record_path = self.root / path.name
                if deleted or not record_path.exists() or (
                        json.loads(record_path.read_text()).get('status') == 'deleted'):
                    if path.exists() and path.read_text() == original:
                        path.unlink(missing_ok=True)
            except Exception as error:
                print('ark_file_gc_error error_type=' + type(error).__name__, flush=True)
        if paths:
            with ThreadPoolExecutor(8, thread_name_prefix='ark-file-gc') as pool:
                list(pool.map(reclaim, paths))
        return len(paths)

    def start_cleanup_reaper(self):
        with self.cleanup_thread_lock:
            if self.cleanup_thread is not None:
                return
            def run():
                while not self.cleanup_stop.wait(5):
                    self.reap_pending()
            self.cleanup_thread = threading.Thread(target=run, name='ark-file-reaper', daemon=True)
            self.cleanup_thread.start()

    def file_status(self, file_id):
        started = time.monotonic()
        with self.metrics_lock:
            self.metrics['waiting_status_queries'] = self.metrics.get('waiting_status_queries', 0) + 1
        try:
            while not self.status_slots.acquire(timeout=.5):
                self.check()
        finally:
            with self.metrics_lock:
                self.metrics['waiting_status_queries'] -= 1
        try:
            self.check()
            with self.metrics_lock:
                self.metrics['status_queries'] = self.metrics.get('status_queries', 0) + 1
                self.metrics['status_wait_seconds'] = self.metrics.get('status_wait_seconds', 0) + time.monotonic() - started
            return self.status_client.get('files/' + file_id)
        finally:
            self.status_slots.release()

    def snapshot(self):
        with self.metrics_lock:
            return {**self.metrics, 'cleanup_start_qps': 1 / self.cleanup_interval}

    def publish(self, path, model, mime_type='video/mp4'):
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        reference = ArkFileReference(self, path, model, mime_type,
                                     digest.hexdigest(), path.stat().st_size)
        reference.acquire()
        try:
            reference.resolve()
        except BaseException:
            reference.close()
            raise
        return reference

    @contextlib.contextmanager
    def published_bytes(self, data, model, mime_type):
        reference = ArkFileReference(self, data, model, mime_type,
                                     hashlib.sha256(data).hexdigest(), len(data))
        reference.acquire()
        try:
            reference.resolve()
            yield reference
        finally:
            reference.close()

    def delete_unused(self, identity, model):
        """Delete only this cache's unleased temporary file, never caller data."""
        path = self.root / (identity + '.json')
        with (self.root / (identity + '.lease')).open('a') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                with self.metrics_lock:
                    self.metrics['cleanup_deferred_in_use'] += 1
                return False
            try:
                with (self.root / (identity + '.lock')).open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not path.exists():
                        return False
                    record = json.loads(path.read_text())
                    if record.get('status') == 'deleted':
                        return False
                    if record.get('object') != 'file' or record.get('purpose') != 'user_data':
                        raise RuntimeError('ark_file_cleanup_ownership_mismatch')
                    filename = record.get('filename', '')
                    if (not re.fullmatch(r'[0-9a-f]{64}\.(mp4|jpg)', filename)
                            or not re.fullmatch(r'file-[A-Za-z0-9_-]+', record.get('id', ''))):
                        raise RuntimeError('ark_file_cleanup_identifier_mismatch')
                    expected = ArkFileReference(None, None, model, record.get('mime_type'), filename[:64], 0)
                    if expected.identity != identity:
                        raise RuntimeError('ark_file_cleanup_cache_identity_mismatch')
                    with self.cleanup_slots:
                        with self.cleanup_pacing_lock:
                            now = time.monotonic()
                            delay = max(0, self.cleanup_next_start - now)
                            self.cleanup_next_start = max(now, self.cleanup_next_start) + self.cleanup_interval
                        if delay:
                            time.sleep(delay)
                        response = self.cleanup_client.delete('files/' + record['id'])
                    if response.status_code not in (200, 204, 404):
                        try:
                            code = str(response.json().get('error', {}).get('code', 'unknown'))
                        except (ValueError, AttributeError):
                            code = 'non_json_error'
                        with self.metrics_lock:
                            errors = self.metrics.setdefault('cleanup_http_errors', {})
                            key = str(response.status_code) + ':' + code
                            errors[key] = errors.get(key, 0) + 1
                        print('ark_file_cleanup_http_error status=' + str(response.status_code) +
                              ' code=' + re.sub(r'[^A-Za-z0-9_.-]', '', code)[:120], flush=True)
                        raise RuntimeError(f'ark_file_cleanup_http_error:{response.status_code}')
                    if response.status_code == 200 and response.json().get('deleted') is not True:
                        raise RuntimeError('ark_file_cleanup_missing_acknowledgement')
                    record.update(status='deleted', expire_at=0, deleted_at_unix=time.time())
                    self._save(path, record)
                    with self.metrics_lock:
                        self.metrics['deleted_files'] += 1
                        self.metrics['deleted_source_bytes'] += record.get('bytes', 0)
                    return True
            except Exception as error:
                # Annotation results must survive a failed best-effort cleanup.
                with self.metrics_lock:
                    self.metrics['cleanup_errors'] += 1
                print('ark_file_cleanup_failed error_type=' + type(error).__name__ +
                      ' identity=' + identity, flush=True)
                return False

    def _save(self, path, record):
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with open(temporary, 'x', opener=lambda p, f: os.open(p, f, 0o600)) as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _body(self, response):
        if response.status_code >= 400:
            # Avoid exception objects that embed authenticated request details.
            try:
                code = (response.json().get('error') or {}).get('code', 'unknown')
            except (ValueError, AttributeError):
                code = 'non_json_error'
            message = f'ark_files_http_error:{response.status_code}:{code}'
            with self.metrics_lock:
                errors = self.metrics.setdefault('file_http_errors', {})
                label = f'{response.request.method}:{response.status_code}:{code}'
                errors[label] = errors.get(label, 0) + 1
            if response.status_code == 401 or code in (
                    'OperationDenied.FileQuotaExceeded', 'AccessDenied', 'InvalidApiKey'):
                with self.metrics_lock:
                    notify = not self.fatal_reported
                    self.fatal_reported = True
                if notify and self.on_fatal is not None:
                    self.on_fatal(response.status_code, message)
            raise RuntimeError(message)
        return response.json()

    def resolve(self, reference):
        identity = reference.identity
        path = self.root / (identity + '.json')
        # File locks deduplicate uploads across episode workers and process resumes.
        with (self.root / (identity + '.lock')).open('a') as lock:
            while True:
                self.check()
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.1)
            record = None
            if path.exists():
                try:
                    record = json.loads(path.read_text())
                except (ValueError, OSError):
                    pass
            if record and record.get('expire_at', 0) > time.time() + 900:
                # Confirm a persisted ID is still available once per reference.
                response = self.file_status(record['id'])
                if response.status_code == 404:
                    record = None
                else:
                    record = self._body(response)
                if record and record.get('status') == 'active':
                    with self.metrics_lock:
                        self.metrics['cache_hits'] += 1
                    return record
            else:
                record = None
            if not record:
                waiting_started = time.monotonic()
                with self.metrics_lock:
                    self.metrics['waiting_uploads'] += 1
                try:
                    while not self.slots.acquire(timeout=.5):
                        self.check()
                finally:
                    wait_seconds = time.monotonic() - waiting_started
                    with self.metrics_lock:
                        self.metrics['waiting_uploads'] -= 1
                        self.metrics['upload_wait_seconds'] += wait_seconds
                try:
                    with self.metrics_lock:
                        self.metrics['active_uploads'] += 1
                        self.metrics['peak_uploads'] = max(self.metrics['peak_uploads'], self.metrics['active_uploads'])
                    self.check()
                    stream = (reference.source.open('rb') if isinstance(reference.source, Path)
                              else io.BytesIO(reference.source))
                    with stream:
                        suffix = '.mp4' if reference.mime_type.startswith('video/') else '.jpg'
                        transfer_started = time.monotonic()
                        response = self.client.post('files', data={'purpose': 'user_data'},
                            files={'file': (reference.sha256 + suffix, stream, reference.mime_type)})
                        transfer_seconds = time.monotonic() - transfer_started
                    record = self._body(response)
                    self._save(path, record)
                    with self.metrics_lock:
                        self.metrics['uploads'] += 1
                        self.metrics['uploaded_bytes'] += reference.size_bytes
                        self.metrics['upload_transfer_seconds'] += transfer_seconds
                    print('ark_file_upload ' + json.dumps({'bytes': reference.size_bytes,
                          'mime_type': reference.mime_type, 'sha256': reference.sha256,
                          'wait_seconds': round(wait_seconds, 3),
                          'transfer_seconds': round(transfer_seconds, 3),
                          'completed_at_unix': time.time()}), flush=True)
                finally:
                    with self.metrics_lock:
                        self.metrics['active_uploads'] -= 1
                    self.slots.release()
            processing_started = time.monotonic()
            deadline = processing_started + 600
            while record.get('status') != 'active':
                self.check()
                if record.get('status') == 'failed':
                    raise RuntimeError('ark_file_processing_failed')
                if time.monotonic() >= deadline:
                    raise RuntimeError('ark_file_processing_timeout')
                time.sleep(3)
                record = self._body(self.file_status(record['id']))
            if record.get('expire_at', 0) <= time.time() + 900:
                raise RuntimeError('ark_file_expiry_too_close')
            self._save(path, record)
            with self.metrics_lock:
                self.metrics['processing_seconds'] += time.monotonic() - processing_started
                self.metrics['processed_files'] += 1
            return record


def responses_payload(model, system, prompt, media, response_format):
    content = []
    for mime, value in media:
        if not isinstance(value, ArkFileReference) or value.model != model or value.mime_type != mime:
            raise ValueError('ark_files_media_model_or_type_mismatch')
        kind = 'input_video' if mime.startswith('video/') else 'input_image'
        if getattr(value, 'transport_method', None) == 'cos-presigned':
            url_field = 'video_url' if kind == 'input_video' else 'image_url'
            content.append({'type': kind, url_field: value.resolve()})
        else:
            content.append({'type': kind, 'file_id': value.resolve()})
    content.append({'type': 'input_text', 'text': prompt})
    format_spec = ({'type': 'json_schema', **response_format['json_schema']}
                   if response_format['type'] == 'json_schema' else response_format)
    return {'model': model, 'store': False, 'temperature': 0,
            'thinking': {'type': 'disabled'}, 'text': {'format': format_spec},
            'input': [{'role': 'system', 'content': system},
                      {'role': 'user', 'content': content}]}


def responses_as_chat(body):
    if body.get('status') != 'completed':
        raise ValueError('ark_response_not_completed:' + str(body.get('status')))
    output = [part['text'] for item in body.get('output', []) if item.get('type') == 'message'
              for part in item.get('content', []) if part.get('type') == 'output_text']
    if not output:
        raise ValueError('ark_response_missing_output_text')
    usage = body.get('usage') or {}
    return {'choices': [{'message': {'content': ''.join(output)}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': usage.get('input_tokens'),
                      'completion_tokens': usage.get('output_tokens'),
                      'prompt_tokens_details': usage.get('input_tokens_details'),
                      'completion_tokens_details': usage.get('output_tokens_details')}}
