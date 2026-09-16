"""Bounded persistent workers for identical media clipping outside a large parent."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing
import os
import threading


def _ready():
    import annotate_videos as pipeline
    pipeline.cv2.setNumThreads(1)
    return os.getpid()


def _clip(source, start, end, fps, minimum_duration, codec_threads=None):
    import annotate_videos as pipeline
    return pipeline._clip_video_path_admitted(source, start, end, fps, minimum_duration,
                                              codec_threads=codec_threads)


class MediaClipPool:
    def __init__(self, workers):
        if workers < 1:
            raise ValueError('media_clip_workers_must_be_positive')
        self.workers = workers
        self.gate = threading.BoundedSemaphore(workers)
        self.executor = None
        self.codec_budget = int(os.environ.get('VQA_VIDEO_CLIP_CPU_BUDGET', '0'))
        if self.codec_budget < 0:
            raise ValueError('video_clip_cpu_budget_must_be_nonnegative')
        self.codec_threads = int(os.environ.get('VQA_VIDEO_CLIP_CODEC_THREADS', '1'))
        if self.codec_budget and (not 1 <= self.codec_threads <= 8 or self.codec_budget < self.codec_threads):
            raise ValueError('video_clip_cpu_budget_must_fit_one_encoder')
        from fair_request_admission import MemoryReservations
        self.cpu_slots = MemoryReservations(workers, self.codec_budget) if self.codec_budget else None

    def __enter__(self):
        self.executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))
        try:
            futures = [self.executor.submit(_ready) for _ in range(self.workers)]
            for future in futures:
                future.result()
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            raise
        return self

    def clip(self, source, start, end, fps=None, minimum_duration=None):
        if self.executor is None:
            raise RuntimeError('media_clip_pool_not_entered')
        threads = (1 if max(end-start, minimum_duration or 0) <= 15 else self.codec_threads)
        reservation = self.cpu_slots.admit('clip', threads) if self.cpu_slots else nullcontext()
        with reservation, self.gate:
            return self.executor.submit(_clip, source, start, end, fps, minimum_duration,
                                        threads if self.cpu_slots else None).result()

    def snapshot(self):
        value = {'workers':self.workers, 'codec_cpu_budget':self.codec_budget,
                 'long_clip_codec_threads':self.codec_threads}
        if self.cpu_slots:
            state = self.cpu_slots.snapshot()
            value['codec_admission'] = {
                'active_clips':state['active_requests'], 'queued_clips':state['queued_requests'],
                'reserved_codec_threads':state['reserved_working_bytes'],
                'codec_thread_budget':self.codec_budget,
                'note':'Nominal codec-thread budget, not a measured CPU-core or total OS-thread limit.'}
        return value

    def __exit__(self, *_exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
