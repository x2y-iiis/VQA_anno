"""Direct, atomic COS publication for durable JSONL annotation objects."""

import base64
import hashlib
import json
from pathlib import Path
import threading

from cos_ecot_images import BUCKET, make_client
from http_transport_metrics import TimedOperationMetrics


DEFAULT_MOUNT_ROOT = Path('/mnt/human_data/video_cleaning')
DEFAULT_MOUNT_ROOT_ALIASES = (
    DEFAULT_MOUNT_ROOT,
    Path('/mnt/human_data/video-cleaning'),
)
KEY_PREFIX = 'video-cleaning/'


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class CosOutputPublisher:
    """Map the writable COSFS mount to direct S3-compatible object PUTs.

    The caller's local SQLite outbox retains each logical record until the PUT
    returns, so direct publication changes transport without weakening recovery.
    """

    def __init__(self, workers=128, client=None, mount_root=DEFAULT_MOUNT_ROOT):
        self.mount_root = Path(mount_root).absolute()
        # The production COSFS mount exists under both historical spellings.
        # They address the same object namespace, so accept both only when the
        # caller uses the production default.  Explicit test/custom roots stay
        # restricted to the path supplied by the caller.
        self.mount_roots = tuple(
            root.absolute() for root in DEFAULT_MOUNT_ROOT_ALIASES
        ) if self.mount_root == DEFAULT_MOUNT_ROOT.absolute() else (self.mount_root,)
        self.client = client if client is not None else make_client(workers)
        self.operations = TimedOperationMetrics(backend='auto')
        self.lock = threading.Lock()
        self.objects = self.bytes = 0

    def key_for_path(self, path):
        absolute = Path(path).absolute()
        relative = None
        for root in self.mount_roots:
            try:
                relative = absolute.relative_to(root)
                break
            except ValueError:
                continue
        if relative is None:
            raise ValueError('cos_output_path_outside_writable_mount')
        if not relative.parts or '..' in relative.parts:
            raise ValueError('cos_output_path_invalid')
        return KEY_PREFIX + relative.as_posix()

    def write_records(self, path, records):
        payload = ''.join(
            json.dumps(record, ensure_ascii=False) + '\n' for record in records
        ).encode('utf-8')
        content_md5 = base64.b64encode(hashlib.md5(payload).digest()).decode('ascii')
        with self.operations.track():
            self.client.put_object(
                Bucket=BUCKET,
                Key=self.key_for_path(path),
                Body=payload,
                ContentType='application/x-ndjson',
                ContentMD5=content_md5,
            )
        with self.lock:
            self.objects += 1
            self.bytes += len(payload)

    def write_one(self, path, record):
        self.write_records(path, [record])

    def _write_batch(self, rows, directory_suffix):
        parents = {Path(path).parent for path, _ in rows}
        if len(parents) != 1:
            raise ValueError('cos_output_batch_requires_one_parent')
        records = [record for _, record in sorted(rows, key=lambda row: str(row[0]))]
        path = next(iter(parents)) / f'{directory_suffix}-{fingerprint(records)}.jsonl'
        self.write_records(path, records)

    def write_checkpoint_batch(self, rows):
        self._write_batch(rows, 'pack')

    def write_final_record_batch(self, rows):
        self._write_batch(rows, 'pack')

    def snapshot(self):
        operation = self.operations.snapshot()
        with self.lock:
            objects, payload_bytes = self.objects, self.bytes
        return {
            'mode': 'cos-direct-put/v1',
            'active': operation['active'],
            'peak_active': operation['peak'],
            'started': operation['started'],
            'finished': operation['finished'],
            'failed': operation['failed'],
            'recent_put_median_seconds': operation['recent_call_median_seconds'],
            'objects': objects,
            'bytes': payload_bytes,
            'bucket': BUCKET,
            'key_prefix': KEY_PREFIX,
        }
