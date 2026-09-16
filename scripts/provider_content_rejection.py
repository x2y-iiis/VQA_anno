"""Keep explicit input-video refusals scoped to an episode/task, not the account."""
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading


class ProviderContentRejected(RuntimeError):
    def __init__(self, task, model, reason):
        self.task, self.model, self.reason = task, model, reason
        super().__init__(task, model, reason)

    def __str__(self):
        return f'provider_input_video_rejected:task={self.task}:model={self.model}:{self.reason}'


def is_input_video_rejection(status, body):
    return is_input_media_rejection(status, body, kinds=('input video',))


def is_input_media_rejection(status, body, kinds=('input video', 'input image')):
    if status != 400:
        return False
    try:
        error = json.loads(body).get('error', {})
        return (error.get('code') == 'data_inspection_failed'
                and any(kind in error.get('message', '').lower() for kind in kinds))
    except (ValueError, AttributeError, TypeError):
        return False


_current = threading.local()


class EpisodeContentScope:
    def __init__(self):
        self.error_args = None

    def check(self):
        if self.error_args is not None:
            raise ProviderContentRejected(*self.error_args)

    @contextmanager
    def activate(self):
        previous = getattr(_current, 'scope', None)
        _current.scope = self
        try:
            self.check()
            yield
        finally:
            _current.scope = previous


def check_content_scope():
    scope = getattr(_current, 'scope', None)
    if scope is not None:
        scope.check()


def reject_current_video(task, model, reason):
    error = ProviderContentRejected(task, model, reason)
    scope = getattr(_current, 'scope', None)
    if scope is not None:
        scope.error_args = error.args
    raise error


@contextmanager
def drain_content_jobs(jobs):
    """Keep shared media alive until submitted jobs exit after any failure."""
    try:
        yield
    except BaseException:
        for future in jobs:
            future.cancel()
        if jobs:
            concurrent.futures.wait(jobs)
        raise


def rejection_identity(uid, model, contract, task='ecot'):
    if task not in {'ecot', 'grd'}:
        raise ValueError('unsupported_content_rejection_task')
    return {'uid': str(uid), 'task': task, 'provider': 'dashscope',
            'model': model, 'contract_id': contract}


def rejection_path(output, identity):
    if identity['task'] not in {'ecot', 'grd'}:
        raise ValueError('unsupported_content_rejection_task')
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(output)/'_state/provider-content-rejections'/identity['task']/f'{digest}.jsonl'


def check_saved_rejection(output, identity):
    path = rejection_path(output, identity)
    if path.is_file():
        record = json.loads(path.read_text())
        if record.get('identity') != identity:
            raise ValueError('provider_content_rejection_identity_mismatch')
        raise ProviderContentRejected(identity['task'], identity['model'],
                                      f'persisted_rejection:{path}:{record["reason"]}')


def save_rejection(output, identity, reason):
    from durable_jsonl import atomic_jsonl
    path = rejection_path(output, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(path, {'schema_version': 'provider-content-rejection/v1',
                       'identity': identity, 'reason': reason, 'status': 'provider_rejected',
                       'registered_as_success': False})
    return path
