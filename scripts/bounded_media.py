"""Bound inline request media without dropping video frames or changing image geometry."""
from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np

VIDEO_BUDGET = 12 * 1024 * 1024
REQUEST_BUDGET = 32 * 1024 * 1024


class FrameCache(MutableMapping):
    """Store exact JPEG bytes on disk with a byte-bounded in-memory LRU."""
    def __init__(self, root: Path, budget=64*1024*1024, write_through=True):
        self.root, self.budget = root, budget
        self.write_through = write_through
        self.dirty = set()
        root.mkdir(parents=True, exist_ok=True)
        self.paths = {}
        self.hot = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    def __contains__(self, key):
        return key in self.paths

    def __getitem__(self, key):
        with self.lock:
            if key in self.hot:
                self.hot.move_to_end(key)
                return self.hot[key]
            value = ('image/jpeg', self.paths[key].read_bytes())
            self._remember(key, value)
            return value

    def _remember(self, key, value):
        if key in self.hot:
            self.bytes -= len(self.hot.pop(key)[1])
        while self.hot and self.bytes + len(value[1]) > self.budget:
            victim = next(iter(self.hot))
            if victim in self.dirty:
                self._persist(victim, self.hot[victim])
            self.bytes -= len(self.hot.popitem(last=False)[1][1])
        if len(value[1]) <= self.budget:
            self.hot[key] = value
            self.bytes += len(value[1])

    def __setitem__(self, key, value):
        with self.lock:
            path = self.root/f'{key:08d}.jpg'
            self.paths[key] = path
            if self.write_through or len(value[1]) > self.budget:
                self._persist(key, value)
            self._remember(key, value)
            if not self.write_through and key in self.hot:
                self.dirty.add(key)

    def _persist(self, key, value):
        if shutil.disk_usage(self.root).free < 1024**3:
            raise RuntimeError('frame_cache_disk_headroom_below_1GiB')
        with self.paths[key].open('wb') as stream:
            stream.write(value[1])
        self.dirty.discard(key)

    def __delitem__(self, key):
        with self.lock:
            self.paths.pop(key).unlink(missing_ok=True)
            self.dirty.discard(key)
            if key in self.hot:
                self.bytes -= len(self.hot.pop(key)[1])

    def clear(self):
        # The owning TemporaryDirectory removes disk files after all users exit.
        self.paths.clear()
        self.hot.clear()
        self.dirty.clear()
        self.bytes = 0


class FrameSelection(Mapping):
    """A lazy subset, avoiding re-materializing the entire JPEG cache as a dict."""
    def __init__(self, cache, keys):
        self.cache, self.keys_set = cache, frozenset(keys)

    def __len__(self):
        return len(self.keys_set)

    def __iter__(self):
        return iter(sorted(self.keys_set))

    def __getitem__(self, key):
        if key not in self.keys_set:
            raise KeyError(key)
        return self.cache[key]

    def __contains__(self, key):
        return key in self.keys_set

    def clear(self):
        self.keys_set = frozenset()


def video_info(path: Path) -> tuple[int, float]:
    response = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_frames:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    value = json.loads(response.stdout)
    return int(value['streams'][0]['nb_frames']), float(value['format']['duration'])


def fit_video_file(path: Path, destination: Path, budget: int = VIDEO_BUDGET) -> Path:
    """Keep the entire temporal input; only resize/re-encode the transport copy."""
    if path.stat().st_size <= budget:
        return path
    frames, duration = video_info(path)
    bitrate = max(1000, int(budget * 8 * 0.80 / max(duration, 0.1)))
    for edge in (960, 640, 384):
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
             '-threads', '1', '-i', str(path), '-an',
             '-vf', f"scale=w='min({edge},iw)':h='min({edge},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
             '-vsync', '0', '-c:v', 'libx264', '-preset', 'veryfast',
             '-b:v', str(bitrate), '-maxrate', str(bitrate), '-bufsize', str(bitrate*2),
             '-threads', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
             str(destination)],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f'video_transport_encode_failed:{result.stderr[-1000:]}')
        actual_frames, actual_duration = video_info(destination)
        if actual_frames != frames or abs(actual_duration-duration) > 0.1:
            raise RuntimeError('video_transport_changed_temporal_coverage')
        if destination.stat().st_size <= budget:
            print(f'video_transport_prepared source_bytes={path.stat().st_size} '
                  f'request_bytes={destination.stat().st_size} frames={frames} '
                  f'duration_seconds={duration} max_edge={edge}', flush=True)
            return destination
        bitrate = int(bitrate * 0.7)
    raise RuntimeError('video_transport_cannot_fit_request_budget')


def fit_video_bytes(payload: bytes, budget: int = VIDEO_BUDGET) -> bytes:
    if len(payload) <= budget:
        return payload
    with tempfile.TemporaryDirectory(prefix='vqa-request-video-') as directory:
        root = Path(directory)
        source = root/'source.mp4'
        with source.open('wb') as stream:
            stream.write(payload)
        return fit_video_file(source, root/'transport.mp4', budget).read_bytes()


def prepare_inline_media(media, budget: int = REQUEST_BUDGET):
    """Keep combined binary media below its Base64-expanded request allowance."""
    if not media:
        return []
    binary_budget = max(1, (budget - 1024*1024) * 3 // 4)
    video_count = sum(mime.startswith('video/') for mime, _ in media)
    video_budget = min(VIDEO_BUDGET, binary_budget // max(1, video_count+1))
    prepared = [(mime, fit_video_bytes(payload, video_budget) if mime.startswith('video/')
                 else payload) for mime, payload in media]
    total = sum(len(payload) for _, payload in prepared)
    if total <= binary_budget:
        return prepared
    images = sum(mime.startswith('image/') for mime, _ in prepared)
    remaining = binary_budget - sum(len(p) for m, p in prepared if not m.startswith('image/'))
    if not images or remaining < 1024*images:
        raise ValueError('request_media_exceeds_binary_budget')
    per_image = remaining // images
    result = []
    for mime, payload in prepared:
        if not mime.startswith('image/') or len(payload) <= per_image:
            result.append((mime, payload))
            continue
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('request_image_decode_failed')
        for _ in range(16):
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok and encoded.nbytes <= per_image:
                result.append(('image/jpeg', encoded.tobytes()))
                break
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (max(1, int(width*.75)), max(1, int(height*.75))))
        else:
            raise ValueError('request_image_cannot_fit_budget')
    return result
