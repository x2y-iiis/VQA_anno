"""Private, short-lived ECoT image URLs using the existing LAS COS account."""
from contextlib import contextmanager
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from ark_file_transport import ArkFileReference
from request_parallel import atomic_json

BUCKET = os.environ.get('VQA_COS_BUCKET', '')
PREFIX = os.environ.get('VQA_COS_IMAGE_PREFIX', 'vqa-annotation/ecot-target-images/').strip('/') + '/'
VIDEO_PREFIX = os.environ.get('VQA_COS_VIDEO_PREFIX', 'vqa-annotation/ecot-videos/').strip('/') + '/'
COS_ENDPOINT = os.environ.get('VQA_COS_ENDPOINT', 'https://cos.ap-shanghai.myqcloud.com')
COS_REGION = os.environ.get('VQA_COS_REGION', 'ap-shanghai')
COSCLI = os.environ.get('LAS_COSCLI', 'coscli')


class ShardedCosClient:
    """Route object operations across independent SDK connection-pool locks."""

    def __init__(self, clients):
        self.clients = tuple(clients)
        if not self.clients:
            raise ValueError('cos_client_shards_must_not_be_empty')
        self.shard_count = len(self.clients)

    def _for_key(self, key):
        digest = hashlib.sha256(key.encode()).digest()
        return self.clients[int.from_bytes(digest[:8], 'big') % self.shard_count]

    def put_object(self, **kwargs):
        return self._for_key(kwargs['Key']).put_object(**kwargs)

    def delete_object(self, **kwargs):
        return self._for_key(kwargs['Key']).delete_object(**kwargs)

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        return self._for_key(Params['Key']).generate_presigned_url(
            operation, Params=Params, ExpiresIn=ExpiresIn)


def make_client(workers):
    import boto3
    from botocore.config import Config
    # Decode through the already-installed official CLI once, never per image.
    # Captured credentials must never be logged or persisted in a new file.
    result = subprocess.run([COSCLI, 'config', 'show', '--disable-log'],
                            capture_output=True, text=True, timeout=30)
    fields = dict(re.findall(r'^[ \t]*(Secret ID|Secret Key|Session Token):[ \t]*([^\r\n]*)',
                             result.stdout, flags=re.MULTILINE))
    if result.returncode or not fields.get('Secret ID', '').startswith('AKID') or not fields.get('Secret Key'):
        raise RuntimeError('cos_existing_credentials_unavailable')
    # Construct clients serially before workers start. Credentials are decoded
    # once, while each client owns a separate urllib3 pool and signing state.
    shards = min(16, max(1, (workers + 31) // 32))
    pool_size = max(1, (workers + shards - 1) // shards)
    clients = [boto3.client('s3', endpoint_url=COS_ENDPOINT,
        region_name=COS_REGION, aws_access_key_id=fields['Secret ID'].strip(),
        aws_secret_access_key=fields['Secret Key'].strip(),
        aws_session_token=fields.get('Session Token', '').strip() or None,
        config=Config(signature_version='s3', s3={'addressing_style': 'virtual'},
            max_pool_connections=pool_size, connect_timeout=20, read_timeout=90,
            # Storage retries are cheaper than retrying a whole annotated episode.
            retries={'max_attempts': 3, 'mode': 'standard'}, proxies={},
            request_checksum_calculation='when_required', response_checksum_validation='when_required'))
        for _ in range(shards)]
    return ShardedCosClient(clients)


class CosImageReference(ArkFileReference):
    transport_method = 'cos-presigned'

    def __init__(self, publisher, record):
        super().__init__(publisher, None, record['model'], 'image/jpeg', record['sha256'], record['bytes'])
        self.object_record = record

    def __len__(self):
        return 2048

    @property
    def identity(self):
        return self.object_record['id']

    def resolve(self):
        if self.closed:
            raise RuntimeError('cos_image_reference_is_closed')
        return self.publisher.client.generate_presigned_url('get_object',
            Params={'Bucket': BUCKET, 'Key': self.object_record['key']}, ExpiresIn=3600)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.lease is not None:
                fcntl.flock(self.lease, fcntl.LOCK_UN)
                self.lease.close()
                self.lease = None
        # The durable manifest already makes this object recoverable by the
        # cleanup reaper. A synchronous COS delete here makes every completed
        # annotation wait on storage and can strand thousands of episode
        # workers during a provider drain.
        self.publisher.defer_delete(self.object_record)


class CosVideoReference(ArkFileReference):
    """One episode-level video held by a private, expiring HTTPS URL."""
    transport_method = 'cos-presigned'

    def __init__(self, publisher, record):
        super().__init__(publisher, None, record['model'], 'video/mp4',
                         record['sha256'], record['bytes'])
        self.object_record = record

    def __len__(self):
        return 2048

    @property
    def identity(self):
        return self.object_record['id']

    def resolve(self):
        if self.closed:
            raise RuntimeError('cos_video_reference_is_closed')
        return self.publisher.client.generate_presigned_url(
            'get_object', Params={'Bucket': BUCKET, 'Key': self.object_record['key']},
            ExpiresIn=3600,
        )

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.lease is not None:
                fcntl.flock(self.lease, fcntl.LOCK_UN)
                self.lease.close()
                self.lease = None
        self.publisher.defer_delete(self.object_record)


class CosImagePublisher:
    schema = 'ecot-cos-temporary-image/v1'
    prefix = PREFIX
    suffix = '.jpg'
    cleanup_log_name = 'cos_image'

    def __init__(self, cache_root, workers=512, client=None, check=None):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if client is None and not BUCKET:
            raise RuntimeError('missing_required_configuration:VQA_COS_BUCKET')
        self.client = client if client is not None else make_client(workers)
        self.check = check or (lambda: None)
        # A Python Condition-backed semaphore convoys badly once thousands of
        # frame workers queue for hundreds of upload slots.  Reuse the native
        # FIFO gate already exercised by Ark HTTP admission: each waiter sleeps
        # on its own ticket, while the exact upload ceiling remains bounded.
        from native_ecot_gate import NativeEcotNetworkSlots, load_library
        if load_library() is not None:
            self.slots = NativeEcotNetworkSlots(workers)
            self.slot_mode = 'native-ecot-fifo/v1'
        else:
            self.slots = threading.BoundedSemaphore(workers)
            self.slot_mode = 'python-bounded-semaphore/v1'
        self.lock = threading.Lock()
        self.metrics = {'uploads': 0, 'uploaded_bytes': 0, 'deleted': 0, 'cleanup_errors': 0,
                        'active_uploads': 0, 'waiting_uploads': 0, 'workers': workers,
                        'upload_slot_mode': self.slot_mode,
                        'client_pool_shards': self.client.shard_count if isinstance(self.client, ShardedCosClient) else 1,
                        'upload_wait_seconds': 0, 'upload_transfer_seconds': 0}
        self.gc_stop = threading.Event()
        self.gc_wake = threading.Event()
        self.gc_thread = None
        self.gc_cursor = ''

    def start_cleanup_reaper(self):
        if self.gc_thread is not None:
            return
        def run():
            while not self.gc_stop.is_set():
                self.gc_wake.wait(5)
                self.gc_wake.clear()
                if self.gc_stop.is_set():
                    return
                self.reap_pending()
        self.gc_thread = threading.Thread(target=run, name='cos-image-gc', daemon=True)
        self.gc_thread.start()

    def defer_delete(self, record):
        """Wake the asynchronous reaper; ownership remains in its manifest."""
        if (record.get('schema') != self.schema or record.get('bucket') != BUCKET):
            raise RuntimeError(self.cleanup_log_name + '_cleanup_ownership_mismatch')
        self.gc_wake.set()

    def reap_pending(self):
        from concurrent.futures import ThreadPoolExecutor
        all_paths = sorted(self.root.glob('*.json'))
        paths = ([p for p in all_paths if p.name > self.gc_cursor] +
                 [p for p in all_paths if p.name <= self.gc_cursor])[:128]
        if paths:
            self.gc_cursor = paths[-1].name
        def reap(path):
            try:
                self.delete_owned(json.loads(path.read_text()))
            except FileNotFoundError:
                pass
            except Exception as error:
                print(self.cleanup_log_name + '_gc_error error_type=' + type(error).__name__, flush=True)
        if paths:
            with ThreadPoolExecutor(8, thread_name_prefix='cos-image-cleanup') as pool:
                list(pool.map(reap, paths))

    def snapshot(self):
        with self.lock:
            result = dict(self.metrics)
        if self.slot_mode == 'native-ecot-fifo/v1':
            slots = self.slots.snapshot()
            result['slot_active'] = slots['active_requests']
            result['slot_queued'] = slots['queued_requests']
            result['slot_peak'] = slots['peak_active_requests']
        return result

    @contextmanager
    def upload_slot(self):
        if self.slot_mode == 'native-ecot-fifo/v1':
            with self.slots.admit('ecot', self.check):
                yield
            return
        while not self.slots.acquire(timeout=.5):
            self.check()
        try:
            yield
        finally:
            self.slots.release()

    def delete_owned(self, record):
        if (record.get('schema') != self.schema or record.get('bucket') != BUCKET
                or not re.fullmatch(r'[0-9a-f]{32}', record.get('id', ''))
                or not re.fullmatch(r'[0-9a-f]{64}', record.get('sha256', ''))
                or record.get('key') != self.prefix + record['id'] + '/' + record['sha256'] + self.suffix):
            raise RuntimeError(self.cleanup_log_name + '_cleanup_ownership_mismatch')
        with (self.root / (record['id'] + '.lease')).open('a') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if not (self.root / (record['id'] + '.json')).exists():
                return
            self._delete_unleased(record)

    def _delete_unleased(self, record):
        try:
            self.client.delete_object(Bucket=BUCKET, Key=record['key'])
            (self.root / (record['id'] + '.json')).unlink(missing_ok=True)
            (self.root / (record['id'] + '.lease')).unlink(missing_ok=True)
            with self.lock:
                self.metrics['deleted'] += 1
        except Exception as error:
            with self.lock:
                self.metrics['cleanup_errors'] += 1
            print(self.cleanup_log_name + '_cleanup_failed error_type=' + type(error).__name__, flush=True)

    @contextmanager
    def published_bytes(self, data, model, mime_type):
        if mime_type != 'image/jpeg':
            raise ValueError('cos_ecot_images_requires_normalized_jpeg')
        identity, digest = uuid.uuid4().hex, hashlib.sha256(data).hexdigest()
        record = {'schema': self.schema, 'id': identity,
            'bucket': BUCKET, 'key': self.prefix + identity + '/' + digest + self.suffix,
            'sha256': digest, 'bytes': len(data), 'model': model}
        ref = CosImageReference(self, record)
        ref.acquire()
        try:
            atomic_json(self.root / (identity + '.json'), record)
            self.check()
            waiting_started = time.monotonic()
            with self.lock:
                self.metrics['waiting_uploads'] += 1
            waiting = True
            try:
                with self.upload_slot():
                    with self.lock:
                        self.metrics['waiting_uploads'] -= 1
                        self.metrics['active_uploads'] += 1
                    waiting = False
                    wait_seconds = time.monotonic() - waiting_started
                    try:
                        transfer_started = time.monotonic()
                        self.client.put_object(Bucket=BUCKET, Key=record['key'], Body=data, ContentType=mime_type)
                        transfer_seconds = time.monotonic() - transfer_started
                        with self.lock:
                            self.metrics['uploads'] += 1
                            self.metrics['uploaded_bytes'] += len(data)
                            self.metrics['upload_wait_seconds'] += wait_seconds
                            self.metrics['upload_transfer_seconds'] += transfer_seconds
                        print('cos_image_upload ' + json.dumps({'bytes': len(data),
                            'wait_seconds': wait_seconds, 'transfer_seconds': transfer_seconds,
                            'completed_at_unix': time.time()}), flush=True)
                    finally:
                        with self.lock:
                            self.metrics['active_uploads'] -= 1
            except Exception as error:
                if waiting:
                    with self.lock:
                        # Admission failed before the active transition.
                        self.metrics['waiting_uploads'] -= 1
                raise RuntimeError('cos_image_upload_failed:' + type(error).__name__) from None
            yield ref
        finally:
            ref.close()


class CosVideoPublisher(CosImagePublisher):
    """Upload one exact 2 FPS episode video and reuse its URL for all frames."""
    schema = 'ecot-cos-temporary-video/v1'
    prefix = VIDEO_PREFIX
    suffix = '.mp4'
    cleanup_log_name = 'cos_video'

    def publish(self, path, model, mime_type='video/mp4'):
        if mime_type != 'video/mp4':
            raise ValueError('cos_ecot_videos_require_mp4')
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        identity = uuid.uuid4().hex
        record = {'schema': self.schema, 'id': identity, 'bucket': BUCKET,
                  'key': self.prefix + identity + '/' + digest.hexdigest() + self.suffix,
                  'sha256': digest.hexdigest(), 'bytes': path.stat().st_size, 'model': model}
        ref = CosVideoReference(self, record)
        ref.acquire()
        try:
            atomic_json(self.root / (identity + '.json'), record)
            self.check()
            waiting_started = time.monotonic()
            with self.lock:
                self.metrics['waiting_uploads'] += 1
            waiting = True
            try:
                with self.upload_slot():
                    with self.lock:
                        self.metrics['waiting_uploads'] -= 1
                        self.metrics['active_uploads'] += 1
                    waiting = False
                    wait_seconds = time.monotonic() - waiting_started
                    try:
                        transfer_started = time.monotonic()
                        with path.open('rb') as stream:
                            self.client.put_object(Bucket=BUCKET, Key=record['key'],
                                                   Body=stream, ContentType=mime_type)
                        transfer_seconds = time.monotonic() - transfer_started
                        with self.lock:
                            self.metrics['uploads'] += 1
                            self.metrics['uploaded_bytes'] += record['bytes']
                            self.metrics['upload_wait_seconds'] += wait_seconds
                            self.metrics['upload_transfer_seconds'] += transfer_seconds
                        print('cos_video_upload ' + json.dumps({
                            'bytes': record['bytes'], 'wait_seconds': wait_seconds,
                            'transfer_seconds': transfer_seconds,
                            'completed_at_unix': time.time(),
                        }), flush=True)
                    finally:
                        with self.lock:
                            self.metrics['active_uploads'] -= 1
            except Exception:
                if waiting:
                    with self.lock:
                        self.metrics['waiting_uploads'] -= 1
                raise
            return ref
        except BaseException:
            ref.close()
            raise
