"""Require both H and O to pass before retaining either participant."""
import copy

VERSION = 'cpa-strict-visible-ho-pairs/v2'


def pair_index(point):
    return int(point['pair_index']) if 'pair_index' in point else int(point['point_id'][1:])


def enforce_event_pairs(event, *, require_hand_side=False):
    """Apply the pair gate to saved contact/observation verdicts without new inference."""
    audit=event.get('pair_filter')
    same_policy=audit and audit.get('require_hand_side')==require_hand_side
    before=copy.deepcopy(event.get('pre_pair_filter_contact_point_pairs',[]) if same_policy else event.get('contact_point_pairs',[]))
    event['pre_pair_filter_contact_point_pairs']=before
    labels={p['point_id']:p['hand_side'] for p in event.get('contact_handedness',{}).get('points',[])}
    decisions={p['point_id']:p.get('accepted',False) for p in event.get('contact_semantic_review',{}).get('decisions',[])}
    kept=[];removed=[]
    for pair in before:
        index=int(pair['pair_index']);reasons=[]
        for prefix,field in [('H','h_xy_1000'),('O','o_xy_1000')]:
            if pair.get(field) is None:reasons.append(prefix+'_missing_or_rejected_at_contact')
            if decisions.get(f'{prefix}{index}') is False:reasons.append(prefix+'_contact_review_failed')
        if require_hand_side and labels.get(f'H{index}') not in {'left','right'}:
            reasons.append('H_handedness_unknown_or_missing')
        if reasons:removed.append({'pair_index':index,'removed_point_ids':[f'H{index}',f'O{index}'],'reasons':reasons})
        else:kept.append(pair)
    event['contact_point_pairs']=kept
    retained={int(p['pair_index']) for p in kept}
    for review in event.get('tracking_reviews',[]):
        points=copy.deepcopy(review.get('pre_pair_filter_validated_points',review.get('validated_points',[])))
        review['pre_pair_filter_validated_points']=points
        groups={}
        for point in points:groups.setdefault(pair_index(point),set()).add(point['role'])
        checks={c['pair_index']:c for c in review.get('checks',[])}
        distances={c['pair_index']:c for c in review.get('pair_distance_checks',[])}
        tracked={p['pair_index']:p for p in review.get('tracked_contact_pairs',[])}
        point_visibility={p['point_id']:p.get('tracker_visible') for p in points}
        gates={p['pair_index']:p for p in review.get('pair_gate_checks',[])}
        allowed=set();dropped=[]
        candidates=set(groups)|{p['pair_index'] for p in review.get('tracked_contact_pairs',[])}
        for index in sorted(candidates):
            reasons=[]
            if index not in retained:reasons.append('contact_pair_rejected')
            if groups.get(index)!= {'hand','object'}:reasons.append('H_or_O_missing_or_rejected_at_observation')
            for role,prefix in [('hand','H'),('object','O')]:
                visible=tracked[index].get(role+'_visible') if index in tracked else point_visibility.get(f'{prefix}{index}')
                if visible is not True:reasons.append(prefix+'_tracker_invisible_or_missing')
            reasons.extend(gates.get(index,{}).get('reasons',[]))
            check=checks.get(index,{})
            if check.get('hand_on_end_effector') is False:reasons.append('H_observation_review_failed')
            if check.get('object_on_contacted_object') is False:reasons.append('O_observation_review_failed')
            if distances.get(index,{}).get('distance_growth_sufficient') is False:reasons.append('pair_distance_review_failed')
            reasons=list(dict.fromkeys(reasons))
            if reasons:dropped.append({'pair_index':index,'removed_point_ids':[f'H{index}',f'O{index}'],'reasons':reasons})
            else:allowed.add(index)
        review['validated_points']=[p for p in points if pair_index(p) in allowed]
        review['removed_point_ids']=sorted(set(review.get('removed_point_ids',[]))|
            {pid for pair in dropped for pid in pair['removed_point_ids']})
        review['pair_filter']={'version':VERSION,'removed_pairs':dropped,'retained_pair_indices':sorted(allowed)}
    event['pair_filter']={'version':VERSION,'require_hand_side':require_hand_side,
        'removed_pairs':removed,'retained_pair_indices':sorted(retained)}
    return event
