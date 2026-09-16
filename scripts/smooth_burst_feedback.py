"""Opt-in continuous rate growth with bounded, fresh-response feedback.

Keep the existing arrival-window feedback and prompt congestion response.
Only upward rate changes are slewed; there is no wait for a request cohort
to finish, no concurrency-cap reduction, and no growth driven by idle time.

Experimental: the September 11 production trial did not establish sustained
throughput improvement. Source production rolled back to arrival feedback.
Do not enable this experiment automatically on later resumes.
"""
import math

from cohort_burst_feedback import CohortBurstFeedback


class SmoothBurstFeedback(CohortBurstFeedback):
    mode = 'request-start-cohort-smooth-growth/v1'

    def __init__(self, now):
        super().__init__(now)
        self.current = None
        self.target = None
        self.updated = now
        self.growth_pauses = 0

    def reset_epoch(self, now):
        super().reset_epoch(now)
        if self.current is not None:
            self.target = self.current
            self.updated = now

    def advance(self, now, interval):
        # An explicit external rate change (e.g. RPM/TPM backoff) invalidates
        # any earlier growth target instead of being undone by that target.
        if self.current is None or interval != self.current:
            self.current = self.target = interval
            self.updated = now
            return interval
        elapsed = max(0, now-self.updated)
        self.updated = max(self.updated, now)
        if self.target < self.current:
            self.current = max(self.target, self.current * math.exp(
                -math.log(1.2) * min(elapsed, 60)/60))
        return self.current

    def pause_growth(self, now, interval, started_at=None):
        interval = self.advance(now, interval)
        if started_at is None or started_at >= self.epoch:
            self.target = interval
            self.growth_pauses += 1
        return interval

    def observe(self, now, started_at, throttled, interval, floor, ceiling, ticket=None):
        interval = self.advance(now, interval)
        if throttled and (started_at is None or started_at >= self.epoch):
            # Stop pending acceleration on the first fresh burst response;
            # the existing aggregate rule still controls the actual backoff.
            interval = self.pause_growth(now, interval, started_at)
        previous = self.decisions
        target = super().observe(now, started_at, throttled, interval,
                                 floor, ceiling, ticket=ticket)
        if self.decisions != previous:
            self.target = target
            if target >= interval:
                self.current = target
            self.last_decision.update(
                interval_after=self.current, target_interval=target,
                rate_growth_factor_per_minute=1.2)
        return self.current

    def snapshot(self):
        return dict(super().snapshot(), target_interval_seconds=self.target,
                    current_interval_seconds=self.current,
                    rate_growth_factor_per_minute=1.2,
                    growth_pauses=self.growth_pauses)
