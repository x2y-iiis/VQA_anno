"""DashScope getPolicy -> multipart upload -> model-bound, 48-hour oss:// URL.

This is the temporary storage API, not customer-managed OSS or COS signing.
https://help.aliyun.com/zh/model-studio/get-temporary-file-url
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import threading
import tempfile
import time
import urllib.parse

UPLOAD_ENDPOINT = 'https://dashscope.aliyuncs.com/api/v1/uploads'
LIFETIME_SECONDS = 48 * 3600
REFRESH_MARGIN = 3600
RESOLVE_HEADER = {'X-DashScope-OssResourceResolve': 'enable'}
MEDIA_EXTENSIONS = {'video/mp4': '.mp4', 'image/jpeg': '.jpg', 'image/png': '.png'}
# Requests uses the connect budget while sending a request body, then switches
# to the read budget. Large uploads must not inherit the 15-second policy RPC
# connect budget; both upload phases remain bounded and retries stay unchanged.
UPLOAD_TIMEOUT = (180, 180)


def redact_media_urls(value):
    return re.sub(r'(?:oss|https?)://[^\s\"\'<>]+', '[MEDIA_URL_REDACTED]', str(value))


def safe_transport_failure(error, phase):
    """Describe nested transport failures without serializing exception text."""
    allowed = {'ConnectionError', 'ConnectionResetError', 'ConnectionAbortedError',
               'BrokenPipeError', 'TimeoutError', 'ConnectTimeout', 'ReadTimeout',
               'ReadTimeoutError', 'ConnectTimeoutError', 'SSLError', 'SSLEOFError',
               'SSLCertVerificationError', 'ProxyError', 'ProtocolError',
               'RemoteDisconnected', 'NewConnectionError', 'NameResolutionError',
               'gaierror', 'MaxRetryError', 'OSError', 'RequestException'}
    pending, visited, kinds, errnos, operations = [error], set(), set(), set(), set()
    while pending and len(visited) < 16:
        item = pending.pop()
        if not isinstance(item, BaseException) or id(item) in visited:
            continue
        visited.add(id(item))
        name = type(item).__name__
        kinds.add(name if name in allowed else 'OtherError')
        number = getattr(item, 'errno', None)
        if type(number) is int and -4096 <= number <= 4096:
            errnos.add(number)
        # Exact fixed OpenSSL messages only; never serialize arbitrary text.
        if isinstance(item, TimeoutError):
            operation = {'The write operation timed out': 'write',
                         'The read operation timed out': 'read'}.get(str(item))
            if operation:
                operations.add(operation)
        pending.extend([item.__cause__, item.__context__, getattr(item, 'reason', None)])
        pending.extend(arg for arg in item.args if isinstance(arg, BaseException))
    result = {'phase': phase if phase in {'policy', 'upload'} else 'unknown',
              'types': sorted(kinds), 'errnos': sorted(errnos)}
    if operations:
        result['timeout_operations'] = sorted(operations)
    return json.dumps(result, separators=(',', ':'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def file_lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield stream


def private_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class TemporaryVideo:
    publisher: object = field(repr=False)
    path: Path = field(repr=False)
    model: str
    sha256: str
    size_bytes: int
    mime_type: str = 'video/mp4'
    _url: str = field(default='', repr=False)
    _expires_at: float = field(default=0, repr=False)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def __len__(self):
        return 8192  # URL overhead, not the remote video size.

    def resolve(self):
        with self._lock:
            if self._expires_at <= time.time() + REFRESH_MARGIN:
                self._url, self._expires_at = self.publisher.resolve(self)
            return self._url


class DashScopeTemporaryPublisher:
    def __init__(self, root, api_key_env='DASHSCOPE_API_KEY', upload_workers=2, policy_qps=5,
                 compact_video_cache=False, pooled_uploads=False, isolated_policy=False):
        if upload_workers < 1:
            raise ValueError('temporary_upload_workers_must_be_positive')
        if not 0 < policy_qps <= 90:
            raise ValueError('temporary_upload_policy_qps_must_be_between_0_and_90')
        self.policy_interval = 1 / policy_qps
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.api_key_env = api_key_env
        self.compact_video_cache = compact_video_cache
        self.slots = threading.BoundedSemaphore(upload_workers)
        self.pooled_uploads = pooled_uploads
        from temporary_upload_pool import SessionPool, UploadMetrics
        self.policy_sessions = SessionPool(min(upload_workers, 64)) if pooled_uploads else None
        self.upload_sessions = SessionPool(upload_workers) if pooled_uploads else None
        self.metrics = UploadMetrics()
        self.policy_process = None
        if isolated_policy:
            if not pooled_uploads:
                raise ValueError('isolated_policy_requires_pooled_uploads')
            from temporary_policy_process import PolicyProcess
            self.policy_process = PolicyProcess(self.root, api_key_env, policy_qps,
                                                min(upload_workers, 64))
            # Constructed with ApiClient before the annotation executors start.
            self.policy_process.start()
        self._image_db_lock = threading.Lock()
        self._image_db_roots = set()

    def publish(self, path, model, mime_type='video/mp4'):
        if mime_type not in MEDIA_EXTENSIONS:
            raise ValueError('temporary_media_mime_type_unsupported')
        path = Path(path)
        size = path.stat().st_size
        if not 0 < size <= 1024**3:
            raise ValueError('temporary_video_size_must_be_between_1_byte_and_1GiB')
        reference = TemporaryVideo(self, path, model, file_sha256(path), size, mime_type=mime_type)
        reference.resolve()
        return reference

    @contextmanager
    def published_bytes(self, payload, model, mime_type):
        """Keep exact local bytes alive through HTTP and validation retries."""
        if mime_type not in MEDIA_EXTENSIONS:
            raise ValueError('temporary_media_mime_type_unsupported')
        with tempfile.NamedTemporaryFile(prefix='media-', suffix=MEDIA_EXTENSIONS[mime_type],
                                         dir=self.root) as stream:
            stream.write(payload)
            stream.flush()
            yield self.publish(stream.name, model, mime_type=mime_type)

    def _namespace(self, model):
        key = os.environ.get(self.api_key_env, '').strip()
        if not key:
            raise RuntimeError(f'missing_api_key_environment_variable:{self.api_key_env}')
        # Separate keys conservatively, even if they belong to the same account.
        identity = hashlib.sha256((UPLOAD_ENDPOINT + '\0' + model + '\0' + key).encode()).hexdigest()
        root = self.root / identity
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root, key

    def _pace(self, root):
        # Credential acquisitions are paced across processes using this cache.
        # The configured cap retains headroom below the provider's 100 QPS cap.
        # Model inference concurrency is not restricted here.
        with file_lock(root / 'policy-rate.lock') as stream:
            stream.seek(0)
            try:
                previous = float(stream.read() or '0')
            except ValueError:
                previous = 0
            interval = self.policy_interval
            time.sleep(max(0, min(interval, previous + interval - time.time())))
            stream.seek(0)
            stream.truncate()
            stream.write(str(time.time()))
            stream.flush()

    def _upload(self, reference, key, root):
        if self.pooled_uploads:
            return self._upload_pooled(reference, key, root)
        import requests
        last_error = 'unknown'
        for attempt in range(3):
            self._pace(root)
            started = time.time()  # Conservative expiry, before the upload.
            try:
                with requests.get(UPLOAD_ENDPOINT, headers={
                    'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                }, params={'action': 'getPolicy', 'model': reference.model},
                    timeout=(15, 60), allow_redirects=False) as response:
                    if response.status_code != 200:
                        last_error = f'get_policy_http_{response.status_code}'
                        if response.status_code != 429 and response.status_code < 500:
                            break
                        raise requests.RequestException()
                    policy = response.json()['data']
                host = urllib.parse.urlsplit(policy['upload_host'])
                if host.scheme != 'https' or not (host.hostname or '').endswith('.aliyuncs.com'):
                    raise ValueError('temporary_upload_host_must_be_aliyun_https')
                filename = reference.sha256 + MEDIA_EXTENSIONS[reference.mime_type]
                object_key = f"{policy['upload_dir'].rstrip('/')}/{filename}"
                fields = {
                    'OSSAccessKeyId': (None, policy['oss_access_key_id']),
                    'Signature': (None, policy['signature']),
                    'policy': (None, policy['policy']),
                    'x-oss-object-acl': (None, policy['x_oss_object_acl']),
                    'x-oss-forbid-overwrite': (None, policy['x_oss_forbid_overwrite']),
                    'key': (None, object_key), 'success_action_status': (None, '200'),
                }
                with reference.path.open('rb') as stream:
                    fields['file'] = (filename, stream, reference.mime_type)
                    with requests.post(policy['upload_host'], files=fields,
                                       timeout=UPLOAD_TIMEOUT, allow_redirects=False) as response:
                        if response.status_code != 200:
                            last_error = f'upload_http_{response.status_code}'
                            if response.status_code not in (403, 429) and response.status_code < 500:
                                break
                            raise requests.RequestException()
                url = f'oss://{object_key}'
                if len(url) > 8192:
                    raise ValueError('temporary_url_too_long')
                return url, started + LIFETIME_SECONDS
            except (requests.RequestException, OSError) as error:
                if last_error == 'unknown':
                    last_error = type(error).__name__
            except (KeyError, ValueError, TypeError):
                raise RuntimeError('dashscope_temporary_invalid_upload_policy') from None
            if attempt < 2:
                time.sleep(2**attempt + random.random())
        # Do not leak policy signatures, URLs, or echoed request credentials.
        raise RuntimeError(f'dashscope_temporary_upload_failed:{last_error}') from None

    def _upload_pooled(self, reference, key, root):
        """Acquire each fresh policy before taking a POST slot; keep retry bounds."""
        import requests
        last_error = 'unknown'
        for attempt in range(3):
            started = time.time()
            last_error = 'unknown'
            phase = 'policy'
            try:
                # Policy queue/pacing never occupies an upload POST slot.
                if self.policy_process is not None:
                    status, policy = self.policy_process.get_policy(reference.model, root.name, self.metrics)
                else:
                    with self.metrics.phase('policy_queue'):
                        lease = self.policy_sessions.lease()
                        session = lease.__enter__()
                    try:
                        with self.metrics.phase('policy_pacing'):
                            self._pace(root)
                        with self.metrics.phase('policy_http'):
                            with session.get(UPLOAD_ENDPOINT, headers={
                                'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                            }, params={'action': 'getPolicy', 'model': reference.model},
                                timeout=(15, 60), allow_redirects=False) as response:
                                status = response.status_code
                                policy = response.json()['data'] if status == 200 else None
                    finally:
                        lease.__exit__(None, None, None)
                if status != 200:
                    last_error = f'get_policy_http_{status}'
                    if status != 429 and status < 500:
                        break
                    raise requests.RequestException()
                host = urllib.parse.urlsplit(policy['upload_host'])
                if host.scheme != 'https' or not (host.hostname or '').endswith('.aliyuncs.com'):
                    raise ValueError('temporary_upload_host_must_be_aliyun_https')
                filename = reference.sha256 + MEDIA_EXTENSIONS[reference.mime_type]
                phase = 'upload'
                object_key = f"{policy['upload_dir'].rstrip('/')}/{filename}"
                fields = {
                    'OSSAccessKeyId': (None, policy['oss_access_key_id']),
                    'Signature': (None, policy['signature']), 'policy': (None, policy['policy']),
                    'x-oss-object-acl': (None, policy['x_oss_object_acl']),
                    'x-oss-forbid-overwrite': (None, policy['x_oss_forbid_overwrite']),
                    'key': (None, object_key), 'success_action_status': (None, '200'),
                }
                with self.metrics.phase('upload_queue'):
                    self.slots.acquire()
                try:
                    with self.upload_sessions.lease() as session, reference.path.open('rb') as stream:
                        fields['file'] = (filename, stream, reference.mime_type)
                        with self.metrics.phase('upload_http'):
                            with session.post(policy['upload_host'], files=fields,
                                              timeout=UPLOAD_TIMEOUT, allow_redirects=False) as response:
                                # Consume the body before returning this connection to the pool.
                                _ = response.content
                                if response.status_code != 200:
                                    last_error = f'upload_http_{response.status_code}'
                                    if response.status_code not in (403, 429) and response.status_code < 500:
                                        break
                                    raise requests.RequestException()
                finally:
                    self.slots.release()
                url = f'oss://{object_key}'
                if len(url) > 8192:
                    raise ValueError('temporary_url_too_long')
                return url, started + LIFETIME_SECONDS
            except (requests.RequestException, OSError) as error:
                if last_error == 'unknown':
                    last_error = safe_transport_failure(error, phase)
            except (KeyError, ValueError, TypeError):
                raise RuntimeError('dashscope_temporary_invalid_upload_policy') from None
            if attempt < 2:
                time.sleep(2**attempt + random.random())
        raise RuntimeError(f'dashscope_temporary_upload_failed:{last_error}') from None

    def _upload_admitted(self, reference, key, root):
        if self.pooled_uploads:
            return self._upload(reference, key, root)
        with self.slots:
            return self._upload(reference, key, root)

    def resolve(self, reference):
        root, key = self._namespace(reference.model)
        if reference.mime_type.startswith('image/') or self.compact_video_cache:
            return self._resolve_image(reference, root, key)
        cache_path = root / f'{reference.sha256}.json'
        with file_lock(root / f'{reference.sha256}.lock'):
            try:
                cached = json.loads(cache_path.read_text())
                if (cached['sha256'] == reference.sha256 and cached['model'] == reference.model
                    and cached['size_bytes'] == reference.size_bytes
                    and cached.get('mime_type', 'video/mp4') == reference.mime_type
                    and cached['expires_at_unix'] > time.time() + REFRESH_MARGIN
                    and cached['url'].startswith('oss://') and len(cached['url']) <= 8192):
                    return cached['url'], cached['expires_at_unix']
            except (OSError, ValueError, KeyError, TypeError):
                pass
            # Unlike signed COS URLs, temporary files cannot be renewed or
            # queried; expired/missing local records require another upload.
            if file_sha256(reference.path) != reference.sha256:
                raise RuntimeError('temporary_video_changed_before_upload')
            url, expires_at = self._upload_admitted(reference, key, root)
            private_json(cache_path, {
                'sha256': reference.sha256, 'model': reference.model,
                'size_bytes': reference.size_bytes, 'url': url,
                'mime_type': reference.mime_type,
                'expires_at_unix': expires_at,
            })
            print(f'dashscope_temporary_uploaded model={reference.model} '
                  f'sha256={reference.sha256} bytes={reference.size_bytes}', flush=True)
            return url, expires_at

    def _resolve_image(self, reference, root, key):
        """Compact media metadata and bounded locks; legacy video caches stay readable."""
        database = root / 'image-urls.sqlite3'
        with self._image_db_lock:
            if root not in self._image_db_roots:
                with file_lock(root / 'image-db-init.lock'):
                    fd = os.open(database, os.O_CREAT | os.O_RDWR, 0o600)
                    os.close(fd)
                    with closing(sqlite3.connect(database, timeout=60)) as db:
                        db.execute('PRAGMA journal_mode=WAL')
                        db.execute('CREATE TABLE IF NOT EXISTS images '
                                   '(sha256 TEXT PRIMARY KEY, mime TEXT, size INTEGER, url TEXT, expires REAL)')
                        db.commit()
                self._image_db_roots.add(root)
        def cached_url():
            with closing(sqlite3.connect(database, timeout=60)) as db:
                row = db.execute('SELECT mime,size,url,expires FROM images WHERE sha256=?',
                                 (reference.sha256,)).fetchone()
            if (row and row[0] == reference.mime_type and row[1] == reference.size_bytes
                    and row[3] > time.time() + REFRESH_MARGIN
                    and row[2].startswith('oss://') and len(row[2]) <= 8192):
                return row[2], row[3]
            return None

        # Committed hits need no upload lock. A colliding new hash must not
        # stall downstream review of an already uploaded frame.
        if self.pooled_uploads:
            cached = cached_url()
            if cached is not None:
                return cached
        # At most 4096 image locks/model, rather than one inode per target frame.
        with file_lock(root / f'image-{reference.sha256[:3]}.lock'):
            cached = cached_url()
            if cached is not None:
                return cached
            if file_sha256(reference.path) != reference.sha256:
                raise RuntimeError('temporary_image_changed_before_upload')
            url, expires_at = self._upload_admitted(reference, key, root)
            with closing(sqlite3.connect(database, timeout=60)) as db:
                db.execute('INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)',
                           (reference.sha256, reference.mime_type, reference.size_bytes, url, expires_at))
                db.commit()
            print(f'dashscope_temporary_uploaded model={reference.model} '
                  f'sha256={reference.sha256} bytes={reference.size_bytes} mime={reference.mime_type}', flush=True)
            return url, expires_at


def prepare_request_media(media, prepare_inline):
    if not any(isinstance(value, TemporaryVideo) for _, value in media):
        return prepare_inline(media)
    inline = iter(prepare_inline([(mime, value) for mime, value in media
                                  if not isinstance(value, TemporaryVideo)]))
    return [(mime, value) if isinstance(value, TemporaryVideo) else next(inline)
            for mime, value in media]


def refresh_request_urls(request_bytes, references):
    updates = []
    for index, reference, old_url in references:
        new_url = reference.resolve()
        if new_url != old_url:
            updates.append((index, new_url))
    if not updates:
        return request_bytes
    payload = json.loads(request_bytes)
    content = payload['messages'][1]['content']
    for index, url in updates:
        kind = 'video_url' if 'video_url' in content[index] else 'image_url'
        content[index][kind]['url'] = url
    for position, (index, reference, _) in enumerate(references):
        kind = 'video_url' if 'video_url' in content[index] else 'image_url'
        references[position] = (index, reference, content[index][kind]['url'])
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')
