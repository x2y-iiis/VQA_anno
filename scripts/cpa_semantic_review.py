"""Audit final SAM contact points without changing contact-frame selection."""
from __future__ import annotations

import copy
import json
import math

import cv2


REVIEW_MODEL = 'doubao-seed-2-1-turbo-260628'
REVIEW_VERSION = 'sam3-contact-semantic-review-paired/v2'
CRITERIA = ('in_contact', 'on_correct_participant', 'clearly_visible', 'unambiguous')


def point_rows(pairs: list[dict]) -> list[dict]:
    rows, seen = [], set()
    for pair in pairs:
        index = int(pair['pair_index'])
        for prefix, field, role in (
            ('H', 'h_xy_1000', 'contact_agent'),
            ('O', 'o_xy_1000', 'contacted_object'),
        ):
            if field not in pair:
                continue
            point_id = f'{prefix}{index}'
            xy = pair[field]
            if point_id in seen or len(xy) != 2 or any(
                not math.isfinite(float(v)) or not 0 <= float(v) <= 1000 for v in xy
            ):
                raise ValueError(f'cpa_review_invalid_point:{point_id}')
            seen.add(point_id)
            rows.append({'point_id': point_id, 'pair_index': index,
                         'role': role, 'xy_1000': list(xy)})
    return rows


def review_prompt(event: dict, rows: list[dict], crop_rows: list[dict]) -> str:
    context = {key: event.get(key) for key in (
        'contact_event_name', 'agent_role', 'object_name', 'contact_source_frame_index',
        'contact_time_seconds', 'interaction_bbox_pixel_xyxy_expanded',
    )}
    return (
        'Audit these SAM-snapped points on the EXACT selected contact frame. '
        'Image 1 is the unmarked full frame. Image 2 is its unmarked enlarged crop. '
        'Coordinates use a 0..1000 grid with image corners at 0 and 1000, separately '
        'for each image. Full-frame points are the final proposed points; crop points '
        'are the earlier crop-SAM stage for context, not replacement coordinates. '
        'H is on the named contact agent (hand, gripper or held tool); O is on the '
        'named contacted object. Review EACH final point independently. '
        'in_contact: is this point at the actual visible physical contact interface, '
        'rather than in a gap, at an unrelated region, or merely near the object? '
        'on_correct_participant: is the point on its assigned participant material? '
        'clearly_visible: can the point and interface be seen without occlusion? '
        'unambiguous: is the exact participant and contact assignment unambiguous? '
        'Think carefully. Prefer explicit Yes/No decisions, not hedged prose. '
        'Use No whenever the image does not establish a criterion. Do not move points '
        'or change the contact frame. Return JSON with exactly one item per point: '
        '{"points":[{"point_id":"H0","in_contact":"Yes",'
        '"on_correct_participant":"Yes","clearly_visible":"Yes",'
        '"unambiguous":"Yes","reason":"Brief visual evidence"}]}.\n'
        f'Event: {json.dumps(context, ensure_ascii=False)}\n'
        f'Final full-frame points: {json.dumps(rows)}\n'
        f'Earlier crop-SAM points: {json.dumps(crop_rows)}'
    )


def validate_review(value: dict, rows: list[dict]) -> list[dict]:
    """Require complete, explicit verdicts; malformed responses are never approval."""
    items = value.get('points')
    if not isinstance(items, list):
        raise ValueError('cpa_review_missing_points')
    expected = {row['point_id'] for row in rows}
    decisions = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('cpa_review_invalid_item')
        point_id = item.get('point_id')
        if point_id not in expected or point_id in decisions:
            raise ValueError(f'cpa_review_unknown_or_duplicate_point:{point_id}')
        if any(item.get(key) not in ('Yes', 'No') for key in CRITERIA):
            raise ValueError(f'cpa_review_requires_yes_no:{point_id}')
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError(f'cpa_review_missing_reason:{point_id}')
        decisions[point_id] = {**copy.deepcopy(item),
                               'accepted': all(item[key] == 'Yes' for key in CRITERIA)}
    if set(decisions) != expected:
        raise ValueError('cpa_review_incomplete_point_coverage')
    return [decisions[row['point_id']] for row in rows]


def filter_points(pairs: list[dict], decisions: list[dict]) -> list[dict]:
    """Retain a pair only when both participant points passed review."""
    keep = {item['point_id'] for item in decisions if item['accepted']}
    result = []
    for pair in pairs:
        index = int(pair['pair_index'])
        if not {f'H{index}',f'O{index}'} <= keep:
            continue
        if pair.get('h_xy_1000') is None or pair.get('o_xy_1000') is None:
            continue
        item = {'pair_index': index}
        for prefix, field in (('H', 'h_xy_1000'), ('O', 'o_xy_1000')):
            if f'{prefix}{index}' in keep:
                item[field] = copy.deepcopy(pair[field])
        if len(item) > 1:
            result.append(item)
    return result


def review_contact_points(client, event: dict, frame_bgr, crop_bgr, *, attempt_callback=None) -> dict:
    """Return a new event with full initial evidence and only approved final points."""
    if client.api not in {'ark', 'las'}:
        raise ValueError('cpa_semantic_review_requires_volc_client')
    result = copy.deepcopy(event)
    initial = copy.deepcopy(event.get('contact_point_pairs', []))
    rows = point_rows(initial)
    result['initial_contact_point_pairs'] = initial
    if not rows:
        result['contact_semantic_review'] = {
            'version': REVIEW_VERSION, 'model': REVIEW_MODEL, 'thinking_enabled': True,
            'status': 'no_initial_points', 'decisions': [], 'attempts': [],
        }
        return result
    crop_rows = point_rows([
        {'pair_index': pair['pair_index'], **pair['sam3_crop']} for pair in initial
    ])
    prompt = review_prompt(event, rows, crop_rows)
    media = []
    for frame in (frame_bgr, crop_bgr):
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            raise ValueError('cpa_review_image_encode_failed')
        media.append(('image/jpeg', encoded.tobytes()))
    attempts = []
    request_prompt = prompt
    for attempt in range(1, 4):
        value, raw = client.request_json(
            'cpa_semantic_review', REVIEW_MODEL,
            'You audit visible contact semantics. Think carefully and return JSON with Yes/No verdicts.',
            request_prompt, media,
        )
        record = {'attempt': attempt, 'prompt': request_prompt,
                  'raw_response': raw, 'parsed_response': copy.deepcopy(value)}
        attempts.append(record)
        try:
            decisions = validate_review(value, rows)
            record['status'] = 'valid'
            if attempt_callback is not None:
                attempt_callback(copy.deepcopy(attempts))
            break
        except ValueError as error:
            record.update(status='invalid', error=str(error))
            if attempt_callback is not None:
                attempt_callback(copy.deepcopy(attempts))
            if attempt == 3:
                # Preserve evidence for a resumable caller without accepting any point.
                error.review_attempts = attempts
                raise
            request_prompt = prompt + '\nCorrect the invalid response: ' + str(error)
    result['contact_point_pairs'] = filter_points(initial, decisions)
    retained_ids={r['point_id'] for r in point_rows(result['contact_point_pairs'])}
    result['contact_semantic_review'] = {
        'version': REVIEW_VERSION, 'model': REVIEW_MODEL, 'thinking_enabled': True,
        'status': 'reviewed', 'prompt': prompt, 'decisions': decisions,
        'initial_point_count': len(rows),
        'accepted_point_count': len(retained_ids),
        'individually_accepted_point_count': sum(item['accepted'] for item in decisions),
        'rejected_point_ids': [item['point_id'] for item in decisions if not item['accepted']],
        'removed_point_ids': [item['point_id'] for item in decisions if item['point_id'] not in retained_ids],
        'pair_policy': 'reject_both_if_either_fails',
        'attempts': attempts,
    }
    return result
