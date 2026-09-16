"""Documented LAS image contact grounding with the operator's default models."""
from __future__ import annotations

import copy
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import requests

from local_json import atomic_json


OPERATOR = {'operator_id': 'las_spatial_perception', 'operator_version': 'v1'}
VERSION = 'las-default-contact-points/v1'


def contact_request(uri, agent_type):
    if agent_type not in {'human', 'robot', 'auto'}:
        raise ValueError('Invalid LAS contact agent type')
    return {**OPERATOR, 'data': {
        'task_template': 'embodied_interaction_grounding', 'action': 'detect',
        'media': {'uri': uri, 'type': 'image'}, 'task_context': {'agent_type': agent_type},
    }}


def decode_points(response, image_size, event):
    data = response['data']
    if not isinstance(data.get('frames'), list):
        raise ValueError('LAS completed response has no frame array')
    width, height = map(int, data['resolution'].split('x'))
    if [width, height] != list(image_size):
        raise ValueError(f'LAS image resolution mismatch: {[width,height]} != {image_size}')
    pairs = []
    for detection_index, frame in enumerate(data['frames']):
        if frame.get('source_frame_index', 0) != 0:
            raise ValueError('LAS image response refers to an unexpected source frame')
        hand = frame.get('agent', {}).get('label') or str(event.get('agent_role', '')).replace('_', ' ')
        obj = frame.get('interacted_object', {}).get('label') or event.get('object_name', '')
        for points in frame.get('contact_points', []):
            pair = {'pair_index': len(pairs), 'las_detection_index': detection_index,
                    'hand_description': hand, 'object_description': obj}
            for source, target in (('h_point', 'h_xy_1000'), ('o_point', 'o_xy_1000')):
                xy = points.get(source)
                if (not isinstance(xy, list) or len(xy) != 2 or
                        any(isinstance(v, bool) or not math.isfinite(float(v)) for v in xy) or
                        not 0 <= xy[0] <= width - 1 or not 0 <= xy[1] <= height - 1):
                    raise ValueError(f'Invalid LAS contact coordinate: detection={detection_index} role={source}')
                pair[target] = [round(xy[0] / max(1, width - 1) * 1000),
                                round(xy[1] / max(1, height - 1) * 1000)]
                pair[source + '_pixel_xy'] = list(xy)
            pairs.append(pair)
    return {
        'hand_description': pairs[0]['hand_description'] if pairs else str(event.get('agent_role', '')).replace('_', ' '),
        'object_description': pairs[0]['object_description'] if pairs else event.get('object_name', ''),
        'contact_pairs': pairs,
    }


class LasContactPointSelector:
    def __init__(self, root: Path, *, cos_prefix=None,
                 coscli=None, base_url=None, check_available=None, request_admission=None):
        self.root = Path(root)
        self.cos_prefix = str(cos_prefix or os.environ.get('CPA_LAS_COS_PREFIX', '')).rstrip('/')
        self.coscli = coscli or os.environ.get('LAS_COSCLI', 'coscli')
        self.base_url = (base_url or os.environ.get('LAS_BASE_URL', 'https://operator.las.cn-beijing.volces.com')).rstrip('/')
        self.check_available = check_available or (lambda: None)
        self.request_admission = request_admission

    def post(self, session, endpoint, payload, retries=1, before_send=None):
        for attempt in range(retries):
            self.check_available()
            try:
                admission=(self.request_admission.admit('las_contact_http', 16384)
                           if self.request_admission is not None else contextlib.nullcontext())
                with admission:
                    if before_send is not None:before_send()
                    response = session.post(self.base_url + endpoint, json=payload,
                        headers={'Authorization': 'Bearer ' + os.environ['LAS_API_KEY']}, timeout=(30,120))
                response.raise_for_status()
                value = response.json()
                if not isinstance(value.get('metadata'), dict):
                    raise ValueError('LAS response metadata missing')
                return value
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
                if attempt + 1 == retries:
                    raise
                time.sleep(min(30, 3 * 2 ** attempt))

    def publish(self, source, digest, directory):
        if not self.cos_prefix.startswith('cos://'):
            raise RuntimeError('cpa_las_cos_prefix_must_be_configured')
        cache = directory / 'private-url.json'
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved['expires_at'] > time.time() + 7200:
                return saved['url']
        remote = f'{self.cos_prefix}/{digest}.jpg'
        subprocess.run([self.coscli,'cp',str(source),remote,'--disable-log','--routines','1','--thread-num','4'],
                       check=True, capture_output=True)
        result = subprocess.run([self.coscli,'signurl',remote,'--simple-output','--time','604800'],
                                check=True,capture_output=True,text=True)
        uri = result.stdout.strip()
        if not uri.startswith('https://'):
            raise ValueError('LAS image publication did not return an HTTPS URL')
        # The URL cache is private and is never included in returned audit records.
        atomic_json(cache, {'url':uri,'expires_at':time.time()+604800})
        cache.chmod(0o600)
        return uri

    def _retire_failed_submission(self, directory, submitted):
        """Journal a retryable terminal task before retiring its current pointers."""
        task_id = submitted['metadata'].get('task_id')
        if not task_id:
            return False
        history = directory/'failed-submissions'
        archived = history/(hashlib.sha256(task_id.encode()).hexdigest()+'.json')
        if not archived.exists():
            poll_path = directory/'poll.json'
            if not poll_path.exists():
                return False
            polled = json.loads(poll_path.read_text())
            meta = polled.get('metadata', {})
            if (meta.get('task_id') != task_id or meta.get('task_status') != 'FAILED'
                    or meta.get('business_code') != 'Video.ModelFailed'):
                return False
            if len(list(history.glob('*.json'))) >= 2:
                raise RuntimeError('LAS contact model failed after 3 distinct submissions: '+task_id)
            history.mkdir(exist_ok=True)
            started_path = directory/'submission-started.json'
            atomic_json(archived, {'submit': submitted, 'terminal_response': polled,
                'submission_started': json.loads(started_path.read_text()) if started_path.exists() else None,
                'retry_reason': 'terminal_Video.ModelFailed', 'archived_at': time.time()})
        # Keep submit.json until last so interrupted retirement is replayable.
        for name in ('poll.json', 'submission-started.json', 'submit.json'):
            (directory/name).unlink(missing_ok=True)
        return True

    def _resolve(self, session, directory, source, digest, family):
        """Reuse live/successful tasks and retry only a proven terminal model failure."""
        completed = directory/'response.json'
        submitted_path = directory/'submit.json'
        while True:
            if completed.exists():
                return json.loads(completed.read_text())
            if submitted_path.exists():
                submitted = json.loads(submitted_path.read_text())
                if self._retire_failed_submission(directory, submitted):
                    continue
            else:
                if (directory/'submission-started.json').exists():
                    raise RuntimeError(f'LAS submission outcome unknown; inspect {directory}')
                uri = self.publish(source, digest, directory)
                payload = contact_request(uri, family)
                audit_request = copy.deepcopy(payload)
                audit_request['data']['media']['uri'] = 'sha256:'+digest
                audit_request['data']['ark_api_key'] = '[injected from environment]'
                atomic_json(directory/'request.json', audit_request)
                payload['data']['ark_api_key'] = os.environ['ARK_API_KEY']
                print(f'las_contact_submit image={digest[:12]} model_policy=operator_default', flush=True)
                submitted = self.post(session, '/api/v1/submit', payload,
                    before_send=lambda: atomic_json(directory/'submission-started.json', {'started_at': time.time()}))
                atomic_json(submitted_path, submitted)
            meta = submitted['metadata']
            if not meta.get('task_id'):
                raise RuntimeError('LAS contact submit failed: '+json.dumps(meta))
            delay = 5
            while True:
                response = self.post(session, '/api/v1/poll', {**OPERATOR, 'task_id': meta['task_id']}, retries=4)
                atomic_json(directory/'poll.json', response)
                status = response['metadata']['task_status']
                if status == 'COMPLETED':
                    if str(response['metadata'].get('business_code', '0')) != '0':
                        raise RuntimeError('LAS contact completion contains a failure code')
                    atomic_json(completed, response)
                    return response
                if status in {'FAILED', 'TIMEOUT', 'CANCELLED'}:
                    if self._retire_failed_submission(directory, submitted):
                        print(f'las_contact_retry_terminal_model_failure task_id={meta["task_id"]}', flush=True)
                        break
                    raise RuntimeError('LAS contact task failed: '+json.dumps(response['metadata']))
                time.sleep(delay)
                delay = min(60, delay*1.5)

    def select(self, image_bytes: bytes, event: dict):
        for key in ('ARK_API_KEY', 'LAS_API_KEY'):
            if not os.environ.get(key, '').strip():
                raise RuntimeError(f'Missing environment variable: {key}')
        image = cv2.imdecode(np.frombuffer(image_bytes,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid LAS contact image')
        height, width = image.shape[:2]
        digest = hashlib.sha256(image_bytes).hexdigest()
        family = 'robot' if event.get('agent_role') == 'robot_gripper' else 'human'
        identity = {'version':VERSION,'image_sha256':digest,'agent_type':family,'model_policy':'operator_default'}
        key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        directory = self.root / key
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            atomic_json(directory/'identity.json',identity)
            source = directory/'source.jpg'
            if not source.exists():
                source.write_bytes(image_bytes)
            with requests.Session() as session:
                response = self._resolve(session, directory, source, digest, family)
            value = decode_points(response,[width,height],event)
            audit = {
                'version':VERSION,'model_policy':'operator_default','image_sha256':digest,
                'request':json.loads((directory/'request.json').read_text()),
                'response':response,'actual_models':list(response['data'].get('token_usage',{})),
                'local_artifact_directory':str(directory), 'coordinate_space':'enlarged_contact_crop',
            }
            return value, audit
