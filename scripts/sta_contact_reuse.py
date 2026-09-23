"""Strict read-only reuse of CPA-final contacts published by the STA task."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time


STA_CONTRACT_ID = 'vqa-anno-raw-sta-cpa-final-random-event-bbox/v5'


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
    ).hexdigest()


def _first_json_object(raw: str) -> dict:
    decoder = json.JSONDecoder()
    text = str(raw or '').strip()
    start = text.find('{')
    if start < 0:
        raise ValueError('sta_reuse_review_response_has_no_json_object')
    value, _ = decoder.raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError('sta_reuse_review_response_must_be_object')
    return value


def _valid_bbox(value) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        values = [int(round(float(item))) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0 <= values[0] < values[2] <= 1000 and 0 <= values[1] < values[3] <= 1000):
        return None
    return values


class StaFinalContactIndex:
    """Locate immutable STA records without scanning a multi-million-file tree."""

    def __init__(self, roots):
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.memory = {}
        cache = os.environ.get('VQA_STA_PACK_CACHE', '').strip()
        self.pack_cache = Path(cache) if cache else None
        raw_prefixes = os.environ.get('VQA_COS_STA_PREFIXES', '')
        prefixes = raw_prefixes.split(':') if raw_prefixes else []
        self.cos_prefixes = {
            root: prefix.rstrip('/')
            for root, prefix in zip(self.roots, prefixes, strict=False)
            if prefix.strip()
        }
        location_path = os.environ.get('VQA_STA_LOCATION_INDEX', '').strip()
        if location_path and Path(location_path).is_file():
            value = json.loads(Path(location_path).read_text(encoding='utf-8'))
            self.locations = value if isinstance(value, dict) else {}
        else:
            self.locations = {}
        # A freshly provisioned GPU node intentionally has no COS/FUSE mount.
        # Build deterministic local roots for every exact published prefix so
        # the location index itself is sufficient to materialize STA records.
        known_prefixes = {prefix.rstrip('/') for prefix in self.cos_prefixes.values()}
        dynamic_base = Path(os.environ.get(
            'VQA_STA_EXACT_ROOT_CACHE', '/run/ti/sta-exact-location-roots',
        ))
        for key in self.locations.values():
            if not isinstance(key, str) or '/shards/' not in key:
                continue
            task_prefix = key.split('/shards/', 1)[0].rstrip('/')
            if task_prefix in known_prefixes:
                continue
            digest = hashlib.sha256(task_prefix.encode()).hexdigest()
            task_root = dynamic_base / digest / 'sta'
            (task_root / 'shards').mkdir(parents=True, exist_ok=True)
            self.roots += (task_root,)
            self.cos_prefixes[task_root] = task_prefix
            known_prefixes.add(task_prefix)

    @staticmethod
    def _cache_path(cache_root: Path, prefix: str, uid: str) -> Path:
        root_digest = hashlib.sha256(prefix.rstrip('/').encode()).hexdigest()
        uid_digest = hashlib.sha256(uid.encode()).hexdigest()
        return cache_root / root_digest / uid_digest[:2] / f'{uid_digest}.jsonl'

    @staticmethod
    def _cos_call(operation):
        """Retry the small targeted COS reads used by the legacy-pack bridge."""
        retryable = {
            'SlowDown', 'RequestTimeout', 'InternalError', 'ServiceUnavailable',
            '500', '502', '503', '504',
        }
        for attempt in range(7):
            try:
                return operation()
            except Exception as error:
                code = str(
                    getattr(error, 'response', {}).get('Error', {}).get('Code', '')
                )
                if code not in retryable or attempt == 6:
                    raise
                time.sleep(min(15.0, 0.25 * (2 ** attempt)) * random.uniform(.8, 1.2))

    def _materialize_pack_parent(self, prefix: str, batch: int) -> None:
        """Expand legacy pack objects for one catalog parent into a shared cache."""
        if self.pack_cache is None:
            return
        root_digest = hashlib.sha256(prefix.rstrip('/').encode()).hexdigest()
        state_root = self.pack_cache / root_digest / '.catalog-materialized'
        state_root.mkdir(parents=True, exist_ok=True)
        marker = state_root / f'catalog-{batch:06d}.complete'
        lock_path = state_root / f'catalog-{batch:06d}.lock'
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if marker.is_file():
                return
            import annotate_videos as core
            client = core._cos_input_client()
            object_prefix = (
                prefix.rstrip('/') + '/shards/cosmos3_v1_5/'
                f'catalog-{batch:06d}.records/pack-'
            )
            token = None
            keys = []
            while True:
                request = {
                    'Bucket': 'datasets-1409717487',
                    'Prefix': object_prefix,
                    'MaxKeys': 1000,
                }
                if token:
                    request['ContinuationToken'] = token
                page = self._cos_call(lambda request=request: client.list_objects_v2(**request))
                keys.extend(
                    item['Key'] for item in page.get('Contents', [])
                    if int(item.get('Size') or 0) > 0
                    and item['Key'].endswith('.jsonl')
                )
                if not page.get('IsTruncated'):
                    break
                token = page['NextContinuationToken']
            for key in keys:
                response = self._cos_call(
                    lambda key=key: client.get_object(
                        Bucket='datasets-1409717487', Key=key,
                    )
                )
                body = response['Body']
                try:
                    raw = body.read()
                finally:
                    close = getattr(body, 'close', None)
                    if close is not None:
                        close()
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    uid = str(record.get('provenance', {}).get('input_record_uid') or '')
                    if not uid:
                        continue
                    path = self._cache_path(self.pack_cache, prefix, uid)
                    if path.is_file() and path.stat().st_size > 0:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    file_descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f'.{path.name}.', dir=path.parent,
                    )
                    try:
                        with os.fdopen(file_descriptor, 'wb') as stream:
                            stream.write(line.rstrip() + b'\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary_name, path)
                    finally:
                        if os.path.exists(temporary_name):
                            os.unlink(temporary_name)
            temporary_marker = marker.with_name(
                f'.{marker.name}.{os.getpid()}.{threading.get_ident()}'
            )
            temporary_marker.write_text(
                json.dumps({'objects': len(keys), 'completed_at_unix': time.time()}) + '\n',
                encoding='utf-8',
            )
            os.replace(temporary_marker, marker)
        finally:
            os.close(descriptor)

    @staticmethod
    def _task_root(root: Path) -> Path:
        return root if root.name == 'sta' else root / 'sta'

    def _record_paths(self, source: dict, aliases: list[str]):
        row_index = source.get('_catalog_row_index')
        if not isinstance(row_index, int) or row_index < 0:
            raise ValueError('sta_reuse_requires_catalog_row_index')
        # Output labels preserve the producer's physical batch width.  The
        # expanded STA fleet uses 8,192- and 16,384-record batches, while older
        # runs used 512-4,096.  Omitting the two production widths made a
        # cloud-acknowledged record look absent to CPA (for example global row
        # 151,144 is catalog-000009 at width 16,384, not catalog-000147).
        batches = tuple(dict.fromkeys(
            row_index // size for size in (16384, 8192, 4096, 2048, 1024, 512)
        ))
        # Live STA producers expose the exact cloud-acknowledged object key.
        # Prefer that one deterministic GET over probing every historical root.
        for alias in aliases:
            key = self.locations.get(alias)
            if not isinstance(key, str) or '/shards/' not in key:
                continue
            task_prefix, relative = key.split('/shards/', 1)
            for root, prefix in self.cos_prefixes.items():
                if prefix.rstrip('/') != task_prefix.rstrip('/'):
                    continue
                task_root = self._task_root(root)
                path = task_root / 'shards' / relative
                if not path.is_file() or path.stat().st_size == 0:
                    import annotate_videos as core
                    core.materialize_cos_file_from_root(
                        path, task_root, prefix, refresh_empty=True,
                    )
                if path.is_file():
                    yield path, alias
                break
        candidates = []
        for root in self.roots:
            task_root = self._task_root(root)
            for alias in aliases:
                name = hashlib.sha256(alias.encode('utf-8')).hexdigest() + '.jsonl'
                for batch in batches:
                    path = task_root / 'shards' / 'cosmos3_v1_5' / f'catalog-{batch:06d}.records' / name
                    if path.is_file():
                        if path.stat().st_size > 0 or root not in self.cos_prefixes:
                            yield path, alias
                    if (not path.is_file() or path.stat().st_size == 0) and root in self.cos_prefixes:
                        candidates.append((path, alias, task_root, self.cos_prefixes[root]))
        # Try every deterministic individual-object location before any legacy
        # pack listing. The former loop scanned all packs under each old root
        # before advancing to a newer producer root, so a valid current STA
        # record could spend minutes behind unrelated COS listings.
        import annotate_videos as core
        for path, alias, task_root, prefix in candidates:
            core.materialize_cos_file_from_root(
                path, task_root, prefix, refresh_empty=True,
            )
            if path.is_file():
                yield path, alias
        for root in self.roots:
            prefix = self.cos_prefixes.get(root)
            if self.pack_cache is not None and prefix:
                for alias in aliases:
                    path = self._cache_path(self.pack_cache, prefix, alias)
                    if path.is_file():
                        yield path, alias
        # Older STA publishers stored multiple immutable records in packs.
        # Materialize their targeted parents only after exact objects and the
        # existing cache have all been exhausted.
        for root in self.roots:
            prefix = self.cos_prefixes.get(root)
            if self.pack_cache is not None and prefix:
                for batch in batches:
                    self._materialize_pack_parent(prefix, batch)
                    for alias in aliases:
                        path = self._cache_path(self.pack_cache, prefix, alias)
                        if path.is_file():
                            yield path, alias

    def find(self, source: dict, subtasks: dict) -> dict:
        uid = str(source.get('uid') or '')
        primary_uid = str(source.get('_primary_record_uid') or '')
        aliases = [uid]
        if primary_uid and primary_uid not in aliases:
            aliases.append(primary_uid)
        elif ':view=' in uid:
            aliases.append(uid.split(':view=', 1)[0])
        cache_key = (tuple(aliases), digest_json(subtasks))
        if cache_key in self.memory:
            return copy.deepcopy(self.memory[cache_key])
        invalid = []
        found = False
        for path, matched_uid in self._record_paths(source, aliases):
            found = True
            try:
                lines = [
                    line for line in path.read_text(encoding='utf-8').splitlines()
                    if line.strip()
                ]
                if len(lines) != 1:
                    raise ValueError(
                        f'sta_reuse_immutable_record_requires_one_line:{path}:{len(lines)}'
                    )
                record = json.loads(lines[0])
                annotation = record.get('provenance', {}).get('annotation', {})
                if annotation.get('contract_id') != STA_CONTRACT_ID:
                    raise ValueError(
                        f'sta_reuse_contract_mismatch:{path}:'
                        f'{annotation.get("contract_id")}'
                    )
                record_uid = str(record.get('provenance', {}).get('input_record_uid') or '')
                if record_uid != matched_uid:
                    raise ValueError(
                        f'sta_reuse_uid_mismatch:{path}:{record_uid}:{matched_uid}'
                    )
                extension = record.get('annotations', {}).get('robot_extension', {})
                result = extension.get('result')
                if (
                    not isinstance(result, dict)
                    or result.get('contact_frame_authority') != 'cpa_final'
                ):
                    raise ValueError(f'sta_reuse_contact_authority_mismatch:{path}')
                expected_subtask_hash = digest_json(subtasks)
                if result.get('subtask_result_sha256') != expected_subtask_hash:
                    raise ValueError(f'sta_reuse_subtask_hash_mismatch:{path}')
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                invalid.append(f'{path}:{type(error).__name__}:{error}')
                continue
            value = {
                'path': str(path),
                'record_uid': record_uid,
                'target_uid': uid,
                'primary_view': uid == record_uid,
                'record_sha256': hashlib.sha256(lines[0].encode('utf-8')).hexdigest(),
                'result': copy.deepcopy(result),
                'requests': copy.deepcopy(extension.get('requests') or []),
            }
            self.memory[cache_key] = copy.deepcopy(value)
            return value
        if not found:
            raise FileNotFoundError(f'sta_final_contact_record_not_found:{uid}')
        # Preserve the original validation reason when there is no usable
        # fallback, while allowing a corrupt earlier root to be bypassed.
        raise ValueError(invalid[-1])


def cpa_seed_from_sta(reused: dict, subtasks: dict) -> dict:
    """Rebuild point-ready CPA events from a validated STA-final record."""
    result = reused['result']
    expected_steps = {str(step.get('id')): step for step in subtasks.get('subtasks') or []}
    review_events = {}
    for request in reused.get('requests') or []:
        try:
            parsed = _first_json_object(request.get('raw_response', ''))
        except (ValueError, json.JSONDecodeError):
            continue
        events = parsed.get('reviewed_contact_events')
        if not isinstance(events, list):
            continue
        step_id = str(request.get('subtask_id'))
        for event in events:
            if isinstance(event, dict) and event.get('event_id') is not None:
                review_events[(step_id, str(event['event_id']))] = event
    output = []
    observed_steps = set()
    for segment in result.get('subtask_results') or []:
        step_id = str(segment.get('subtask', {}).get('id'))
        if step_id not in expected_steps:
            raise ValueError(f'sta_reuse_unknown_subtask_id:{step_id}')
        observed_steps.add(step_id)
        events = []
        for final_event in segment.get('result', {}).get('contact_events') or []:
            if final_event.get('accepted') is not True:
                continue
            selection = final_event.get('contact_frame_selection')
            contact_time = final_event.get('contact_time_seconds')
            if not isinstance(selection, dict) or selection.get('valid_contact') is not True:
                raise ValueError(f'sta_reuse_invalid_contact_selection:{step_id}:{final_event.get("event_id")}')
            if not isinstance(contact_time, (int, float)):
                raise ValueError(f'sta_reuse_invalid_contact_time:{step_id}:{final_event.get("event_id")}')
            event = copy.deepcopy(final_event)
            event.pop('observations', None)
            event.pop('observation_sampling', None)
            event.pop('sta_proposed_observations', None)
            review = review_events.get((step_id, str(event.get('event_id'))), {})
            bbox = _valid_bbox(event.get('interaction_bbox_xyxy_1000'))
            if bbox is None:
                bbox = _valid_bbox(review.get('interaction_bbox_xyxy_1000'))
            if bbox is not None:
                event['interaction_bbox_xyxy_1000'] = bbox
            event['accepted'] = True
            event['clip_time_seconds'] = round(
                float(contact_time) - float(expected_steps[step_id]['start_time_seconds']), 6
            )
            event['contact_frame_source'] = 'sta_final_reuse'
            events.append(event)
        output.append({
            'subtask': copy.deepcopy(expected_steps[step_id]),
            'task_instruction': expected_steps[step_id]['subtask'],
            'task_instruction_source': 'subtask',
            'media_scope': {
                'kind': 'video_clip',
                'start_time_seconds': expected_steps[step_id]['start_time_seconds'],
                'end_time_seconds': expected_steps[step_id]['end_time_seconds'],
                'duration_seconds': (
                    expected_steps[step_id]['end_time_seconds']
                    - expected_steps[step_id]['start_time_seconds']
                ),
            },
            'result': {'reviewed_contact_events': events},
        })
    if observed_steps != set(expected_steps):
        raise ValueError(f'sta_reuse_subtask_set_mismatch:{sorted(observed_steps)}:{sorted(expected_steps)}')
    return {
        'task_instruction_source': 'subtask',
        'subtask_result_sha256': digest_json(subtasks),
        'subtask_results': output,
        'contact_frame_authority': 'sta_cpa_final_reuse',
        'sta_contact_reuse': {
            'source_record_uid': reused['record_uid'],
            'source_record_sha256': reused['record_sha256'],
            'source_path': reused['path'],
            'primary_view': reused['primary_view'],
        },
    }
