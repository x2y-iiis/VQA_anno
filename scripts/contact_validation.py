"""Validate contact units before caching and recover malformed legacy evidence."""
import copy
import json
import math


def validate_contact_events(result, task, video=True, require_time=True):
    key = 'contact_events' if task == 'sta' else 'reviewed_contact_events'
    if not isinstance(result, dict):
        raise ValueError(f'{task}_contact_result_must_be_object')
    events = result.get(key)
    if not isinstance(events, list):
        raise ValueError(f'{task}_{key}_must_be_list')
    seen_ids = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f'{task}_event_must_be_object:{index}')
        if not require_time or event.get('accepted') is False:
            continue
        event_id = event.get('event_id')
        if (not event_id or isinstance(event_id, bool)
                or not isinstance(event_id, (str, int)) or not str(event_id).strip()
                or str(event_id) in seen_ids):
            raise ValueError(f'{task}_event_requires_unique_nonempty_id:{index}')
        seen_ids.add(str(event_id))
        field = 'contact_time_seconds' if video else 'contact_media_index'
        number = event.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError(f'{task}_event_requires_finite_nonnegative_{field}:{index}')
        if not video and int(number) != number:
            raise ValueError(f'{task}_event_requires_integer_media_index:{index}')
    return result


def inherit_cpa_event_ids(result, proposal):
    """Restore only omitted identifiers using the existing positional review mapping."""
    value = copy.deepcopy(result)
    if not isinstance(value, dict) or not isinstance(proposal, dict):
        return value
    proposals = proposal.get('contact_events')
    events = value.get('reviewed_contact_events')
    if not isinstance(proposals, list) or not isinstance(events, list):
        return value
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get('event_id') is not None or index >= len(proposals):
            continue
        original = proposals[index]
        if isinstance(original, dict) and original.get('event_id'):
            event['event_id'] = copy.deepcopy(original['event_id'])
            fields = list(event.get('inherited_from_sta_fields') or [])
            if 'event_id' not in fields:
                fields.append('event_id')
            event['inherited_from_sta_fields'] = fields
    return value


def checked_unit(checkpoint, key, validator):
    """Prefer a validated replacement; never mutate or overwrite old evidence."""
    invalid = []
    if checkpoint is None:
        return None, key, invalid
    replacement = key + ':recovered-v2'
    for candidate in (replacement, key + ':recovered-v1', key):
        value = checkpoint.get(candidate)
        if value is None:
            continue
        try:
            validated = validator(value['result'])
            value = copy.deepcopy(value)
            value['result'] = validated
            return value, candidate, invalid
        except (ValueError, TypeError, KeyError) as error:
            invalid.append({'checkpoint_key': candidate, 'error': str(error),
                            'cached_value': copy.deepcopy(value)})
            print(f'contact_checkpoint_invalid_preserved key={candidate} error={error}', flush=True)
    return None, replacement if invalid else key, invalid


def request_contact_unit(checkpoint, key, task, prompt, request, validator, fallback=None):
    cached, destination, invalid = checked_unit(checkpoint, key, validator)
    if cached is not None:
        return cached, destination
    if fallback is not None and not invalid:
        try:
            value = validator(fallback['result'])
            return dict(copy.deepcopy(fallback), result=value), key
        except (ValueError, TypeError, KeyError):
            pass
    attempts = list(invalid)
    current_prompt = prompt
    for attempt in range(3):
        result, raw = request(current_prompt)
        try:
            value = validator(result)
        except (ValueError, TypeError, KeyError) as error:
            attempts.append({'prompt': current_prompt, 'raw_response': raw,
                             'result': copy.deepcopy(result), 'error': str(error)})
            print(f'contact_validation_retry task={task} attempt={attempt+1} error={error}', flush=True)
            if attempt == 2:
                # Preserve all failed responses without declaring a valid unit.
                if checkpoint is not None:
                    import hashlib
                    digest = hashlib.sha256(json.dumps(attempts, sort_keys=True).encode()).hexdigest()
                    checkpoint.put(f'contact_validation_failure:{key}:{digest}', {'attempts': attempts})
                raise ValueError(f'contact_validation_exhausted:{task}:{error}') from error
            current_prompt = (prompt + '\nYour previous response failed validation: ' + str(error)
                              + '\nPrevious response: ' + json.dumps(result, ensure_ascii=False)
                              + '\nRecheck the SAME supplied media and return the complete corrected JSON. '
                              'Keep each event as an object, with a numeric contact timestamp for video '
                              'or integer contact_media_index for ordered images. Do not invent times '
                              'or silently remove a supported event to satisfy validation; verify its '
                              'onset from the media. Reject an unsupported reviewed event explicitly.')
            continue
        unit = {'result': value, 'raw': raw, 'prompt': current_prompt}
        if attempts:
            unit['validation_attempts'] = attempts
        if checkpoint is not None:
            checkpoint.put(destination, unit)
        return unit, destination
    raise AssertionError('unreachable contact validation branch')
