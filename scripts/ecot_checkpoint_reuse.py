"""Read compatible foreign ECoT units without changing their stored identity."""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_prior_units(checkpoint, sources, indices, system_prompt, user_prompt,
                     training_question, source_frame_index, pipeline):
    found, signatures = {}, {}
    rejected = Counter()
    selected = set(indices)
    for source in sources:
        expected = copy.deepcopy(checkpoint.identity)
        expected.update(provider=source.provider, models={'ecot': source.model})
        identity_hash = fingerprint(expected)
        path = pipeline.annotation_checkpoint_path(
            source.root / '_state/annotation-unit-checkpoints', 'ecot', checkpoint.uid)
        candidates = []
        try:
            path.stat()
            candidates.append(path)
        except FileNotFoundError:
            pass
        try:
            with os.scandir(path.with_suffix('.units')) as entries:
                candidates += sorted(Path(entry.path) for entry in entries
                                     if entry.name.endswith('.jsonl') and not entry.name.startswith('.'))
        except FileNotFoundError:
            pass
        for candidate in candidates:
            with candidate.open(encoding='utf-8', buffering=4 * 1024 * 1024) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not (row.get('schema_version') == checkpoint.SCHEMA_VERSION
                                and row.get('input_record_uid') == checkpoint.uid
                                and row.get('task') == 'ecot'
                                and row.get('identity') == expected
                                and row.get('identity_sha256') == identity_hash):
                            rejected['identity'] += 1
                            continue
                        value = row['value']
                        frame, request = value['frame'], value['request']
                        index = frame['sampled_frame_index']
                        if type(index) is not int or index not in selected:
                            rejected['index'] += 1
                            continue
                        key = f'ecot_frame:{index}'
                        instruction = expected['dependency']['task_instruction']
                        if not (row['unit_key'] == key
                                and request.get('model') == source.model
                                and request.get('provider', source.provider) == source.provider
                                and request.get('stage') == 'privileged_complete_video_ecot'
                                and request.get('media_kind') == pipeline.ECOT_MEDIA_KIND
                                and request.get('sampled_frame_index') == index
                                and request.get('sampled_time_seconds') == round(index / pipeline.ECOT_FPS, 6)
                                and request.get('task_instruction_source') == 'global_task'
                                and request.get('task_instruction') == instruction
                                and request.get('system_prompt') == system_prompt
                                and request.get('base_prompt', request.get('prompt')) == user_prompt):
                            rejected['request_contract'] += 1
                            continue
                        structured = pipeline.parse_ecot_result(frame['structured_ecot'])
                        rebuilt = pipeline.ecot_frame_record(
                            structured, instruction, training_question, index, source_frame_index(index))
                        if frame != rebuilt:
                            rejected['frame_contract'] += 1
                            continue
                    except (ValueError, KeyError, TypeError, AttributeError):
                        rejected['malformed'] += 1
                        continue
                    signature = fingerprint(value)
                    if key in signatures and signatures[key] != signature:
                        raise ValueError(f'conflicting_prior_ecot_unit:{checkpoint.uid}:{key}')
                    if key in found:
                        continue
                    copied = copy.deepcopy(value)
                    copied['request']['provider'] = source.provider
                    copied['request']['reused_from'] = {
                        'provider': source.provider, 'model': source.model,
                        'output_root': str(source.root), 'checkpoint_file': str(candidate),
                        'checkpoint_identity_sha256': identity_hash,
                        'checkpoint_record_sha256': fingerprint(row),
                    }
                    found[key], signatures[key] = copied, signature
    if found or rejected:
        print(f'ecot_prior_units_validated uid={checkpoint.uid} reused={len(found)} '
              f'rejected={json.dumps(dict(rejected), sort_keys=True)} '
              'source_unchanged=true imported_units_count_as_new_inference=false', flush=True)
    return found


def provenance_summary(requests, provider, model):
    counts = Counter((request.get('provider', provider), request.get('model', model))
                     for request in requests)
    if not any(pair != (provider, model) for pair in counts):
        return None
    return {
        'completion_provider': provider, 'completion_model': model,
        'mixed_provider_or_model': True,
        'frame_counts': [dict(provider=p, model=m, frames=n)
                         for (p, m), n in sorted(counts.items())],
        'provenance_locator': 'requests:sampled_frame_index',
    }
