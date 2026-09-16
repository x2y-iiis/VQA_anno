"""Bounded burst-rate feedback attributed to the request's actual start epoch.

Old responses remain visible in metrics and normal retry handling, but cannot
repeatedly penalize a sending rate that those requests never used. No provider
quota, Retry-After, authentication, or content policy is bypassed here.
"""
class CohortBurstFeedback:
    mode = 'request-start-cohort/v1'

    def __init__(self, now):
        self.epoch = now
        self.first_feedback = None
        self.successes = 0
        self.throttles = 0
        self.stale_responses = 0
        self.decisions = 0
        self.last_decision = None

    def observe(self, now, started_at, throttled, interval, floor, ceiling, ticket=None):
        if started_at is not None and started_at < self.epoch:
            self.stale_responses += 1
            return interval
        if self.first_feedback is None:
            self.first_feedback = now
        self.throttles += int(throttled)
        self.successes += int(not throttled)
        total = self.successes + self.throttles
        # Aggregate responses for ten seconds and at least eight observations.
        # No recovery is driven merely by silence or by stale successes.
        if now - self.first_feedback < 10 or total < 8:
            return interval
        if self.throttles:
            target = min(ceiling, max(interval, .005) * 1.25)
            reason = 'fresh_burst_feedback'
        else:
            target = max(floor, interval / 1.2)
            reason = 'fresh_successful_feedback'
        target = max(floor, target)
        self.last_decision = {'reason': reason, 'successes': self.successes,
                              'throttles': self.throttles, 'interval_before': interval,
                              'interval_after': target, 'window_seconds': now-self.first_feedback}
        self.decisions += 1
        self.reset_epoch(now)
        return target

    def reset_epoch(self, now):
        self.epoch = now
        self.first_feedback = None
        self.successes = self.throttles = 0

    def snapshot(self):
        return {'mode': self.mode, 'stale_responses': self.stale_responses,
                'fresh_successes': self.successes, 'fresh_throttles': self.throttles,
                'decisions': self.decisions, 'last_decision': self.last_decision}
