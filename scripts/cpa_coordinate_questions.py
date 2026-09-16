"""Coordinate-only anticipation questions over unmarked 5 FPS video history."""
from __future__ import annotations

import copy
import hashlib
import json
import math

from cpa_student_video import student_frame_indices

VERSION = 'cpa-object-only-final-frame-two-decimal/v3'
MODELS = {'seed_2_1_pro': 'doubao-seed-2-1-pro-260628',
          'seed_2_0_lite': 'doubao-seed-2-0-lite-260215'}


def three_significant(value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Coordinate must be finite and normalized to [0,1]')
    return float(format(value, '.3g'))


def two_decimal(value):
    from decimal import Decimal, ROUND_HALF_UP
    if isinstance(value,bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Coordinate must be finite and normalized to [0,1]')
    return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def answer_text(answer):
    return '{' + ', '.join(json.dumps(role) + ': [' + ', '.join(
        '[' + ', '.join(format(v, '.2f') for v in xy) + ']' for xy in values
    ) + ']' for role,values in answer.items()) + '}'


def coordinate_question(sample, event, review, *, history_seconds=2.0, robot_agent=False):
    """Use surviving observation points directly, without choice-SAM2 or distractors."""
    query, contact = review['query_frame_index'], event['contact_source_frame_index']
    if not 0 <= query < contact:
        raise ValueError('Student query must precede contact')
    points = [copy.deepcopy(p) for p in review['validated_points']
              if p['role'] == 'object']
    if not points:
        return None
    raw_track = next(o for o in event['backtracking']['tracked_observations']
                     if o['query_frame_index'] == query)
    raw_points = {p['point_id']: p for p in raw_track['points']}
    high, low = [], []
    for point in points:
        xy = [float(v) / 1000 for v in raw_points[point['point_id']]['xy_1000']]
        if any(abs(a-b)>1e-12 for a,b in zip(xy, point['xy'])):
            raise ValueError('Reviewed coordinates differ from original tracked point')
        item = {'point_id': point['point_id'], 'role': point['role'], 'xy': xy}
        high.append(item)
        rounded = [two_decimal(v) for v in xy]
        low.append({**item, 'xy': rounded,
                    'xy_text': [format(v, '.2f') for v in rounded]})
    counts = {role: sum(p['role'] == role for p in points) for role in ('hand','object')}
    count_instruction = (f'Predict exactly {counts["object"]} point(s) on the contacted object; '
                         f'{counts["object"]} point(s) in total. ')
    prompt = (
        'Watch the entire provided 5 FPS video. Predict the locations IN ITS LAST FRAME '
        'of the material points that will participate in the next intentional physical contact. '
        'Return coordinates on the participants as they appear in the LAST FRAME, before contact occurs. '
        'The video is unmarked. Earlier frames provide motion context. '
        + count_instruction +
        'Coordinates must be normalized to [0,1]: top-left pixel center is (0,0), '
        'bottom-right pixel center is (1,1), x increases rightward and y downward. '
        'Report each coordinate rounded to exactly TWO DECIMAL PLACES (for example, 0.10). '
        'Return only JSON of the form {"object":[[x,y],...]}. '
        'Only object points are requested. Do not return hand points or choice labels.'
    )
    indices = student_frame_indices(max(0,query-round(history_seconds*sample['fps'])),query,sample['fps'])
    answer={'object':[p['xy'] for p in low]}
    return {'id': review['observation_id']+'-coordinates', 'version': VERSION,
            'sample_id': sample['id'], 'query_frame_index': query, 'question_frame_index': query,
            'contact_frame_index': contact, 'tracking_target_frame_indices': [query],
            'task_instruction': sample['task_instruction'], 'question': prompt,
            'target_point_counts': counts, 'known_conditions': {'type':'video','sampling_fps':5,
                'source_frame_indices':indices,'frame_index_convention':'zero_based_original_video',
                'markers':'none'}, 'teacher_raw_points':high, 'gt_two_decimal':low,
            'answer': answer, 'answer_text':answer_text(answer),
            'answer_precision':'two_decimal_places', 'response_point_counts':{'object':counts['object']},
            'student_scope':'object_only',
            'gt_provenance': {'source':'LAS contact points → crop/full SAM3 → contact review → native CoTracker → observation review',
                'precision':'Student GT rounded to two decimal places; original tracks retained only in teacher audit',
                'observation_id':review['observation_id']},
            'learner_inputs':['task_instruction','unmarked_5fps_video','question_with_point_counts']}


def questions_from_result(sample, result, *, history_seconds=2.0, robot_agent=False):
    questions=[]
    for segment in result['subtask_results']:
        instruction=segment['subtask'].get('subtask')
        if not isinstance(instruction,str) or not instruction.strip():
            raise ValueError('Coordinate student question requires a subtask instruction')
        subtask_sample={**sample,'task_instruction':instruction}
        for event in segment['result'].get('reviewed_contact_events',[]):
            if not event.get('accepted'):
                continue
            from cpa_pair_policy import enforce_event_pairs
            event=enforce_event_pairs(copy.deepcopy(event),require_hand_side=False)
            for review in event.get('tracking_reviews',[]):
                question=coordinate_question(subtask_sample,event,review,history_seconds=history_seconds,robot_agent=robot_agent)
                if question:
                    question['pair_filter']=copy.deepcopy(review.get('pair_filter',{}))
                    question['instruction_source']='subtask'
                    question['subtask_id']=segment['subtask']['id']
                    question['event_id']=event['event_id']
                    questions.append(question)
    return questions


def prediction_schema(counts, decimal_places=None):
    return {'name':'contact_coordinates','strict':True,'schema':{
        'type':'object','additionalProperties':False,'required':list(counts),
        'properties':{role:{'type':'array','minItems':count,'maxItems':count,
            'items':{'type':'array','minItems':2,'maxItems':2,
                     'items':{'type':'number','minimum':0,'maximum':1,
                              **({'multipleOf':.01} if decimal_places==2 else {})}}}
                      for role,count in counts.items()}}}


def validate_prediction(value, counts, decimal_places=None):
    if not isinstance(value,dict) or set(value) != set(counts):
        raise ValueError('Expected exactly these coordinate arrays: '+', '.join(counts))
    points=[]
    for role,count in counts.items():
        values=value[role]
        if not isinstance(values,list) or len(values)!=count:
            raise ValueError(f'Expected exactly {count} {role} points')
        for index,xy in enumerate(values):
            if not isinstance(xy,list) or len(xy)!=2 or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in xy):
                raise ValueError('Predicted coordinates must be normalized numeric pairs')
            if decimal_places==2 and any(abs(v-two_decimal(v))>1e-9 for v in xy):
                raise ValueError('Predicted coordinates must be rounded to two decimal places')
            points.append({'point_id':f'{role}-{index+1}','role':role,'xy':xy})
    return points


def learner_example(question, source_video):
    return {key:copy.deepcopy(question[key]) for key in (
        'id','known_conditions','question_frame_index','task_instruction','question','target_point_counts','answer')} | {
        'source_video':source_video,'question_mode':'coordinate_prediction','answer_precision':question.get('answer_precision','three_significant_digits'),
        **{key:copy.deepcopy(question[key]) for key in ('answer_text','student_scope') if key in question},
        'instruction_source':question.get('instruction_source','subtask'),
        **{key:copy.deepcopy(question[key]) for key in ('hand_point_counts','response_point_counts') if key in question}}


def display_prediction(value):
    """Keep valid predictions at their exact positions; expose invalid values separately."""
    valid, invalid = [], []
    roles = ('left_hand','right_hand','unknown_hand','object') if isinstance(value,dict) and any(k in value for k in ('left_hand','right_hand','unknown_hand')) else ('hand','object')
    for role in roles:
        entries = value.get(role, []) if isinstance(value, dict) else []
        if not isinstance(entries, list):
            invalid.append({'role':role, 'raw':entries})
            continue
        for index,xy in enumerate(entries):
            point = {'point_id':f'{role}-{index+1}', 'role':role, 'xy':xy}
            if role.endswith('_hand'):
                point.update(role='hand', hand_side=role.removesuffix('_hand'))
            if isinstance(xy,list) and len(xy)==2 and all(
                not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in xy):
                valid.append(point)
            else:
                invalid.append(point)
    return valid, invalid
