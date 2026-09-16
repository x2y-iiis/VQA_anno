"""Fail closed on invisible tracks; audit material identity with paired frame evidence."""
import json
from pathlib import Path

import cv2

VERSION = 'cpa-visible-same-material-pairs/v1'
SYSTEM = '''You audit exact material-point correspondences for contact anticipation.
The contact frame is teacher evidence only. A point on the correct hand or object
is insufficient: it must be the SAME physical surface point in both frames.
Judge the pixel center using the raw images; markers can cover the evidence.
For each H/O pair independently assess visible surface, correct participant, and
same material point. Occluded, hidden behind another surface, ambiguous, blurred,
or visually unsupported correspondences are false. A projected location behind
an occluder is not a visible point. Do not relocate, invent or repair coordinates.
One failed or uncertain H/O judgment removes the entire pair. Return JSON only.'''
FIELDS = ('hand_on_end_effector', 'object_on_contacted_object', 'hand_visible',
          'object_visible', 'hand_same_material_point', 'object_same_material_point')


def visibility_reasons(pair):
    reasons = []
    for role, prefix in (('hand', 'H'), ('object', 'O')):
        if pair.get(f'{role}_visible') is not True:
            reasons.append(prefix + '_tracker_invisible_or_missing')
        xy = pair.get(f'{role}_point_xy')
        if not isinstance(xy, list) or len(xy) != 2:
            reasons.append(prefix + '_coordinate_missing')
        elif any(type(v) not in (int, float) or not 0 <= v <= 1 for v in xy):
            reasons.append(prefix + '_coordinate_out_of_bounds')
    return reasons


def validate_verdict(value, indices):
    if not isinstance(value, dict) or not isinstance(value.get('reason'), str):
        raise ValueError('Material review requires a reason')
    checks = value.get('checks')
    if not isinstance(checks, list) or len(checks) != len(indices):
        raise ValueError('Material review pair count mismatch')
    seen = set()
    for check in checks:
        i = check.get('pair_index')
        if type(i) is not int or i not in indices or i in seen:
            raise ValueError('Material review pair identities mismatch')
        seen.add(i)
        if any(type(check.get(field)) is not bool for field in FIELDS):
            raise ValueError('Material review requires explicit booleans for every judgment')
        if not isinstance(check.get('reason'), str) or not check['reason'].strip():
            raise ValueError('Material review requires per-pair evidence')
    return value


def evidence_images(audit, source, pairs, legacy, directory):
    """Save full raw/marked frames and raw/marked local crops in a declared order."""
    capture = cv2.VideoCapture(audit['source_video'])
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, audit['candidate_frame'])
        ok, contact = capture.read()
    finally:
        capture.release()
    query = cv2.imread(source['query_image_path'])
    if not ok or contact is None or query is None:
        raise ValueError('Material review evidence could not be decoded')
    indices = {p['pair_index'] for p in pairs}
    contact_pairs = [p for p in audit['contact_point_pairs_full_xy'] if p['pair_index'] in indices]
    records = []
    for label, raw, coordinates in (('contact', contact, contact_pairs), ('query', query, pairs)):
        marked = raw.copy()
        height, width = raw.shape[:2]
        xs, ys = [], []
        for pair in coordinates:
            i = pair['pair_index']
            for role, prefix in (('hand', 'H'), ('object', 'O')):
                xy = pair[f'{role}_point_xy']
                xs.append(xy[0] * (width - 1)); ys.append(xy[1] * (height - 1))
                legacy.draw_cpa_point(marked, xy, f'{prefix}{i}', legacy.CPA_POINT_COLORS[i % len(legacy.CPA_POINT_COLORS)], role)
        pad = max(48, round(min(width, height) * .08))
        box = [max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
               min(width, int(max(xs)) + pad + 1), min(height, int(max(ys)) + pad + 1)]
        x1, y1, x2, y2 = box
        for view, image in (('raw-full', raw), ('marked-full', marked),
                            ('raw-crop', raw[y1:y2, x1:x2]), ('marked-crop', marked[y1:y2, x1:x2])):
            path = directory / f'{source["observation_id"]}-{label}-{view}.jpg'
            if not cv2.imwrite(str(path), image):
                raise RuntimeError('Material review evidence write failed')
            records.append({'frame': label, 'view': view, 'path': str(path),
                            'source_frame_index': audit['candidate_frame'] if label == 'contact' else source['query_frame_index'],
                            'crop_box': box if 'crop' in view else [0, 0, width, height]})
    return records


def review_tracks(client, legacy, audit, source, tracked, directory, *, digest, atomic_json, pipeline_version):
    path = directory / f'{source["observation_id"]}-review.json'
    identity = digest({'audit': {**audit,'source_video':audit.get('source_sha256',audit.get('source_video'))}, 'source': source, 'tracked': tracked,
                       'version': VERSION, 'pipeline_version': pipeline_version, 'model': legacy.MODEL})
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != identity:
            raise ValueError('cpa_tracking_review_checkpoint_mismatch')
        return saved['review']
    eligible, gates, distances = [], [], []
    for pair in tracked['contact_pairs']:
        reasons = visibility_reasons(pair)
        if all(pair.get(f'{role}_point_xy') is not None for role in ('hand', 'object')):
            distance = legacy.cpa_pair_separation_review(audit, pair)
        else:
            distance = {'pair_index': pair['pair_index'], 'distance_growth_sufficient': None,
                        'reason': 'missing_partner'}
        distances.append(distance)
        if distance['distance_growth_sufficient'] is False:
            reasons.append('pair_distance_review_failed')
        gates.append({'pair_index': pair['pair_index'], 'reasons': reasons,
                      'hand_tracker_visible': pair.get('hand_visible'),
                      'object_tracker_visible': pair.get('object_visible')})
        if not reasons:
            eligible.append(pair)
    checks = []; attempts = []; evidence = []; prompt = ''
    reason = 'No pairs passed the tracker visibility and distance gates; no model request.'
    if eligible:
        evidence = evidence_images(audit, source, eligible, legacy, directory)
        schema = {'reason': 'Overall evidence summary', 'checks': [
            {'pair_index': p['pair_index'], **{field: 'boolean' for field in FIELDS},
             'reason': 'Visible anatomical/surface landmarks supporting or contradicting identity; uncertainty means false'}
            for p in eligible]}
        prompt = (f'Subtask instruction: {audit.get("subtask_instruction", audit["task_instruction"])}\n'
                  f'Contact agent: {audit["end_effector_type"]}. Object: {audit["contact"]["noun"]}.\n'
                  f'Future contact source frame: {audit["candidate_frame"]}. Query source frame: {source["query_frame_index"]}.\n'
                  'All coordinates below are normalized in the FULL original frame, including for crops.\n'
                  'Contact pairs: ' + json.dumps([p for p in audit['contact_point_pairs_full_xy'] if p['pair_index'] in {r['pair_index'] for r in eligible}]) + '\n'
                  'Query pairs: ' + json.dumps(eligible) + '\nImages in order: ' +
                  json.dumps([{k:v for k,v in e.items() if k != 'path'} for e in evidence]) + '\n'
                  'Return this exact structure with actual JSON booleans. Same material point requires affirmative visual evidence across both frames; same object alone is insufficient.\n' + json.dumps(schema))
        for attempt in range(3):
            value, raw = client.request_json('cpa_material_point_review', legacy.MODEL, SYSTEM, prompt,
                [('image/jpeg', Path(e['path']).read_bytes()) for e in evidence])
            record = {'parsed_response': value, 'raw_response': raw}; attempts.append(record)
            try:
                parsed = validate_verdict(value, {p['pair_index'] for p in eligible})
            except (ValueError, TypeError, AttributeError) as error:
                record['error'] = str(error)
                atomic_json(path.with_name(path.stem + '-attempts.json'), attempts)
                if attempt == 2: raise
                continue
            atomic_json(path.with_name(path.stem + '-attempts.json'), attempts)
            checks, reason = parsed['checks'], parsed['reason']
            break
    by_index = {c['pair_index']: c for c in checks}
    validated = []; removed = []
    for gate, pair in zip(gates, tracked['contact_pairs']):
        i = pair['pair_index']; check = by_index.get(i)
        if check:
            gate['reasons'].extend(field + '_failed' for field in FIELDS if not check[field])
        gate['material_review_status'] = 'reviewed' if check else 'skipped_hard_gate'
        gate['retained'] = not gate['reasons']
        if gate['reasons']:
            removed.extend([f'H{i}', f'O{i}']); continue
        for role, prefix in (('hand', 'H'), ('object', 'O')):
            validated.append({'point_id': f'{prefix}{i}', 'pair_index': i, 'role': role,
                              'xy': pair[f'{role}_point_xy'], 'tracker_visible': True})
    review = {'candidate_id': audit['candidate_id'], 'observation_id': source['observation_id'],
              'query_frame_index': source['query_frame_index'], 'tracked_contact_pairs': tracked['contact_pairs'],
              'validated_points': validated, 'removed_point_ids': removed, 'pair_distance_checks': distances,
              'tracking_review_model': legacy.MODEL if eligible else None, 'tracking_review_prompt': prompt,
              'tracking_review_system_prompt': SYSTEM, 'evidence_images': evidence,
              'checks': checks, 'reason': reason, 'attempts': attempts,
              'visibility_policy': VERSION, 'pair_gate_checks': gates}
    atomic_json(path, {'identity': identity, 'review': review})
    return review
