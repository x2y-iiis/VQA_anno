"""Label contact-point handedness from full-frame context and a marked crop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2

from prepare_cpa_comparison import atomic_json

VERSION = 'contact-point-handedness-egocentric-context/v2'
MODEL = 'doubao-seed-2-0-lite-260215'
SIDES = ('left', 'right', 'unknown')
SYSTEM = 'You label anatomical handedness at identified contact points. Use visual evidence and return JSON only.'


def contact_points(event):
    return [{'point_id': f'H{p["pair_index"]}', 'pair_index': p['pair_index'],
             'xy_1000': p['h_xy_1000']} for p in event.get('contact_point_pairs', [])
            if p.get('h_xy_1000') is not None]


def validate_labels(value, points):
    labels = value.get('points') if isinstance(value, dict) else None
    expected = {p['point_id'] for p in points}
    if not isinstance(labels, list) or len(labels) != len(expected):
        raise ValueError('Return exactly one label per requested H point')
    if {p.get('point_id') for p in labels} != expected:
        raise ValueError('Point IDs must match the requested IDs exactly, without duplicates')
    if any(p.get('hand_side') not in SIDES or not isinstance(p.get('reason'), str) for p in labels):
        raise ValueError('Each point requires hand_side left/right/unknown and a reason')
    return labels


def marked_images(sample, event, points, directory):
    cap = cv2.VideoCapture(sample['video'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, event['contact_source_frame_index'])
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise ValueError('Unable to read the exact contact frame')
    h, w = frame.shape[:2]
    box = event.get('interaction_bbox_pixel_xyxy_expanded', [0, 0, w, h])
    xs = [p['xy_1000'][0] / 1000 * (w - 1) for p in points]
    ys = [p['xy_1000'][1] / 1000 * (h - 1) for p in points]
    # Include every requested point, even when the stored interaction box is too small.
    x0, y0 = max(0, int(min([box[0]] + xs)) - 24), max(0, int(min([box[1]] + ys)) - 24)
    x1, y1 = min(w, int(max([box[2]] + xs)) + 25), min(h, int(max([box[3]] + ys)) + 25)
    crop = frame[y0:y1, x0:x1].copy()
    scale = min(2.0, 1400 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for image, ox, oy, sx, sy in [(frame, 0, 0, 1, 1),
            (crop, x0, y0, crop.shape[1] / (x1-x0), crop.shape[0] / (y1-y0))]:
        for point in points:
            x = round((point['xy_1000'][0] / 1000 * (w - 1) - ox) * sx)
            y = round((point['xy_1000'][1] / 1000 * (h - 1) - oy) * sy)
            cv2.circle(image, (x,y), 6, (0,255,255), 2, cv2.LINE_AA)
            position = (max(0,min(image.shape[1]-65,x+9)), max(24,min(image.shape[0]-8,y-9)))
            cv2.putText(image, point['point_id'], position, cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(image, point['point_id'], position, cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x0,y0), (x1-1,y1-1), (255,180,30), 2)
    paths = []
    for name, image in [('full.jpg', frame), ('crop.jpg', crop)]:
        path = directory / name
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ValueError('Unable to encode handedness evidence')
        paths.append(path)
    return paths, [x0,y0,x1,y1]


def label_event(client, sample, event, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    points = contact_points(event)
    identity = {'version':VERSION, 'model':MODEL, 'video_sha256':sample['sha256'],
                'frame':event['contact_source_frame_index'], 'points':points,
                'crop_box':event.get('interaction_bbox_pixel_xyxy_expanded'),
                'camera_viewpoint':sample.get('camera_viewpoint','infer_from_full_frame')}
    path = directory / 'labels.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record['identity'] != identity:
            raise ValueError('Handedness checkpoint mismatch')
        validate_labels({'points':record['points']}, points)
        return record
    if not points:
        record = {'identity':identity, 'status':'no_hand_points', 'points':[], 'attempts':[]}
        atomic_json(path, record); return record
    paths, crop = marked_images(sample, event, points, directory)
    viewpoint = (
        'CAMERA CONTEXT: This is an egocentric, FIRST-PERSON frame: the camera wearer is the person '
        'performing the action, and we look out in the SAME direction as that person. '
        'Do NOT apply the left/right reversal used when looking at a person facing the camera. '
        'In a normal uncrossed pose, the wearer\'s right forearm enters from image right and left forearm '
        'from image left. Hands can cross, so trace wrist/forearm continuity and thumb anatomy rather '
        'than labeling from horizontal pixel position alone. '
        if sample.get('camera_viewpoint')=='egocentric' else
        'First establish the camera viewpoint from the full frame. A first-person view does not use '
        'the left/right reversal of a person facing the camera. '
    )
    prompt = viewpoint + (
        'Image 1 is the full contact frame; its blue rectangle locates the crop. Image 2 is the enlarged crop '
        'of the SAME frame. Yellow rings and H IDs mark the exact contact-agent material points to label. '
        'For EACH H ID, identify whether its pixel belongs to the person\'s anatomical LEFT or RIGHT hand '
        '(including its fingers), not the left/right side of the image. Use visible thumb, palm/back-of-hand, '
        'wrist and arm context. The two hands may both contact the object; label every point independently. '
        'Check palm versus back of hand before interpreting thumb position. '
        'Do not infer all points from the dominant action or merge both hands. If a point is not on a hand, '
        'is occluded, or cannot be determined from the images, use unknown and explain why. '
        'Do not move or invent points. Required IDs: ' + ', '.join(p['point_id'] for p in points) +
        '. Return {"points":[{"point_id":"H0","hand_side":"left|right|unknown","reason":"visual evidence"}]}.'
    )
    schema = {'name':'contact_handedness','strict':True,'schema':{'type':'object',
        'additionalProperties':False,'required':['points'],'properties':{'points':{'type':'array',
        'minItems':len(points),'maxItems':len(points),'items':{'type':'object','additionalProperties':False,
        'required':['point_id','hand_side','reason'],'properties':{
            'point_id':{'type':'string','enum':[p['point_id'] for p in points]},
            'hand_side':{'type':'string','enum':list(SIDES)},'reason':{'type':'string'}}}}}}}
    request = {'model':MODEL,'system':SYSTEM,'prompt':prompt,'schema':schema,
               'media':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
               'thinking':'disabled','temperature':0}
    attempt_path = directory / 'attempts.json'
    attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else []
    if attempts:
        try:
            labels = validate_labels(attempts[-1]['parsed'], points)
        except ValueError:
            labels = None
    else:
        labels = None
    while labels is None and len(attempts) < 3:
        sent_prompt = prompt if not attempts else prompt + '\nFormat correction: ' + attempts[-1]['error']
        value, raw = client.request_json('cpa_hand_side', MODEL, SYSTEM, sent_prompt,
            [('image/jpeg', p.read_bytes()) for p in paths], schema)
        item = {'parsed':value, 'response':json.loads(raw), 'prompt':sent_prompt}
        attempts.append(item)
        try:
            labels = validate_labels(value, points)
        except ValueError as error:
            item['error'] = str(error)
        atomic_json(attempt_path, attempts)
    if labels is None:
        raise ValueError('Handedness format validation failed after three attempts')
    pairs = {p['point_id']:p for p in points}
    labels = [{**p, 'pair_index':pairs[p['point_id']]['pair_index'],
               'xy_1000':pairs[p['point_id']]['xy_1000']} for p in labels]
    record = {'identity':identity,'status':'completed','model':MODEL,'points':labels,
              'request':request,'attempts':attempts,'crop_box_pixel_xyxy':crop,'completed_at':time.time()}
    atomic_json(path,record)
    print(f'contact_handedness sample={sample["id"]} frame={identity["frame"]} labels=' +
          json.dumps({p['point_id']:p['hand_side'] for p in labels}), flush=True)
    return record


def add_question_handedness(question, event):
    """Count only this query's surviving tracked hand points; never expose contact images."""
    record = event['contact_handedness']
    labels = {p['point_id']:p for p in record['points']}
    for key in ('gt_high_precision','gt_three_significant'):
        for point in question[key]:
            hand_id = point['point_id'] if point['role']=='hand' else 'H'+point['point_id'][1:]
            label = labels.get(hand_id)
            if point['role']=='hand' and label is None:
                raise ValueError('Tracked hand point has no contact handedness annotation')
            point['contact_hand_side'] = label['hand_side'] if label else 'unknown'
            if point['role']=='hand':
                point['hand_side'] = label['hand_side']
    counts = {side:sum(p['role']=='hand' and p['hand_side']==side for p in question['gt_high_precision']) for side in SIDES}
    response_counts = {side+'_hand':count for side,count in counts.items()}
    response_counts['object'] = question['target_point_counts']['object']
    question['hand_point_counts'] = counts
    question['response_point_counts'] = response_counts
    question['answer'] = {side+'_hand':[p['xy'] for p in question['gt_three_significant']
        if p['role']=='hand' and p['hand_side']==side] for side in SIDES}
    question['answer']['object'] = [p['xy'] for p in question['gt_three_significant'] if p['role']=='object']
    prefix = question['question'].split('Return only JSON of the form ')[0]
    question['question'] = prefix + (
        f'Of the {question["target_point_counts"]["hand"]} hand points, exactly {counts["left"]} belong to the anatomical LEFT hand, '
        f'{counts["right"]} to the anatomical RIGHT hand, and {counts["unknown"]} have unknown hand side. '
        'Left/right refers to the person\'s own hands, not the side of the image. '
        'Put each predicted hand point in its corresponding array. '
        'Return only JSON {"left_hand":[[x,y],...],"right_hand":[[x,y],...],'
        '"unknown_hand":[[x,y],...],"object":[[x,y],...]}. Use [] for every zero-count array. '
        'Do not return choice labels or coordinates from the future contact frame.'
    )
    question['handedness_provenance'] = {'version':VERSION,'model':MODEL,
        'source_contact_frame':event['contact_source_frame_index'],
        'labels':[{'point_id':p['point_id'],'hand_side':p['hand_side']} for p in record['points']],
        'policy':'Contact-frame labels propagate by tracked point ID; counts include only surviving query points.'}
    question['version'] = 'cpa-subtask-handedness-video-coordinates-3sig/v3'
    return question


def review_event(client, directory, feedback):
    """Recheck a visually inconsistent label, preserving the original model reply."""
    directory=Path(directory);path=directory/'labels.json';record=json.loads(path.read_text())
    prompt=record['request']['prompt']+'\nReview feedback: '+feedback
    request=record['request']
    value,raw=client.request_json('cpa_hand_side',MODEL,SYSTEM,prompt,
        [('image/jpeg',Path(m['path']).read_bytes()) for m in request['media']],request['schema'])
    labels=validate_labels(value,record['identity']['points'])
    points={p['point_id']:p for p in record['identity']['points']}
    record.setdefault('review_history',[]).append({'previous_points':record['points'],'feedback':feedback,
        'prompt':prompt,'response':json.loads(raw),'parsed':value,'reviewed_at':time.time()})
    record['points']=[{**p,'pair_index':points[p['point_id']]['pair_index'],
        'xy_1000':points[p['point_id']]['xy_1000']} for p in labels]
    atomic_json(path,record)
    return record
