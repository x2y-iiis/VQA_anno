"""CoTracker3 backward tracking from audited contacts to existing query frames.

The native-frame decode, 640-pixel resize, query convention, offline window (60)
and backward_tracking=True match robot_vqa_sta_cpa/scripts/track_cpa_points.py.
Point identities remain stable when pair filtering removes previously approved points.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

import cv2
import numpy as np

from cpa_semantic_review import point_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_tracker(repo=Path(os.environ.get('COTRACKER_REPO', PROJECT_ROOT/'third_party/co-tracker')),
                 checkpoint=Path(os.environ.get(
                     'COTRACKER_CHECKPOINT', PROJECT_ROOT/'models/cotracker/scaled_offline.pth')),
                 device='cuda'):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor
    model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).to(device)
    model.eval()
    return model


def track_contact_points(model, source_video: Path, event: dict,
                         query_frame_indices: list[int], *, device='cuda', max_width=640,
                         require_review=True) -> dict:
    """Keep native-rate tracking and emit predictions only for requested query frames."""
    import torch

    if require_review and event.get('contact_semantic_review', {}).get('status') not in {'reviewed', 'no_initial_points'}:
        raise ValueError('cpa_tracking_requires_completed_semantic_review')
    contact_frame = int(event['contact_source_frame_index'])
    targets = sorted(set(query_frame_indices))
    if not targets or any(type(t) is not int or t < 0 or t >= contact_frame for t in targets):
        raise ValueError('cpa_tracking_invalid_query_frames')
    points = point_rows(event['contact_point_pairs'])
    accepted_ids = {d['point_id'] for d in event.get('contact_semantic_review', {}).get('decisions', [])
                    if d['accepted']}
    # Pair/hand-side filtering can remove an individually approved partner.
    if require_review and not {p['point_id'] for p in points} <= accepted_ids:
        raise ValueError('cpa_tracking_point_review_mismatch')
    result = {
        'tracker': 'CoTracker3 offline scaled checkpoint', 'backward_tracking': True,
        'tracking_fps': 'original', 'query_contact_source_frame_index': contact_frame,
        'contact_review_required': require_review,
        'target_source_frame_indices': targets, 'query_points': points,
        'tracked_observations': [],
    }
    if not points:
        result['status'] = 'no_approved_contact_points'
        return result
    start = targets[0]
    capture = cv2.VideoCapture(str(source_video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(width, height) <= 1:
        capture.release()
        raise ValueError('cpa_tracking_invalid_video_dimensions')
    scale = min(1., max_width / width)
    track_width, track_height = max(2, round(width * scale)), max(2, round(height * scale))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    try:
        for frame_index in range(start, contact_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'cpa_tracking_decode_failed:{frame_index}')
            if (width, height) != (track_width, track_height):
                frame = cv2.resize(frame, (track_width, track_height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.tensor([[
        [contact_frame - start, p['xy_1000'][0] / 1000 * (track_width - 1),
         p['xy_1000'][1] / 1000 * (track_height - 1)] for p in points
    ]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        tracks, visibility = model(video, queries=queries, backward_tracking=True)
    tracks = tracks[0].detach().cpu().numpy()
    visibility = visibility[0].detach().cpu().numpy().astype(bool)
    for target in targets:
        tracked = []
        for index, point in enumerate(points):
            xy = tracks[target - start, index] / np.array([track_width - 1, track_height - 1])
            tracked.append({
                **point, 'xy_1000': (np.clip(xy, 0, 1) * 1000).round(6).tolist(),
                'visible': bool(visibility[target - start, index]),
            })
        result['tracked_observations'].append({'query_frame_index': target, 'points': tracked})
    result.update(status='completed', source_resolution=[width, height],
                  tracking_resolution=[track_width, track_height],
                  tracking_frame_start=start, tracking_frame_end_inclusive=contact_frame)
    return result
