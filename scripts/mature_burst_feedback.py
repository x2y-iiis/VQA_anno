"""Bounded request-start samples whose outcomes settle before rate adaptation.

This is an opt-in alternative to arrival-window feedback. It does not change
provider limits, Retry-After, retry budgets, request content, or HTTP capacity.
The caller serializes access using the request pacer's lock.

Experimental only: the loaded 2026-09-11 trial failed because slow samples
delayed congestion response. Production supervisors no longer offer this mode.
"""


class MatureBurstFeedback:
    mode = 'settled-request-sample/v1'

    def __init__(self, now, sample_limit=32, sample_seconds=10, minimum_sample=8):
        self.sample_limit = sample_limit
        self.sample_seconds = sample_seconds
        self.minimum_sample = minimum_sample
        self.sequence = 0
        self.decisions = 0
        self.stale_responses = 0
        self.excluded_responses = 0
        self.last_decision = None
        self.reset_epoch(now)

    def reset_epoch(self, now):
        self.epoch = now
        self.first_start = None
        self.pending = set()
        self.sampled = 0
        self.successes = self.throttles = self.unknown = 0
        self.sealed = False

    def _seal_if_ready(self, now):
        if self.first_start is not None:
            self.sealed |= (self.sampled >= self.sample_limit or
                            (self.sampled >= self.minimum_sample and
                             now-self.first_start >= self.sample_seconds))

    def start(self, now):
        # If an idle sample has already settled, include this start so there
        # will be a terminal callback to perform the decision under the pacer.
        if self.pending:
            self._seal_if_ready(now)
        if self.sealed:
            return None
        self.sequence += 1
        ticket = (self.epoch, self.sequence)
        self.pending.add(ticket)
        self.sampled += 1
        if self.first_start is None:
            self.first_start = now
        self._seal_if_ready(now)
        return ticket

    def observe(self, now, started_at, throttled, interval, floor, ceiling, ticket=None):
        return self._finish(now, ticket, 'throttles' if throttled else 'successes',
                            interval, floor, ceiling)

    def abandon(self, now, ticket, interval, floor, ceiling):
        return self._finish(now, ticket, 'unknown', interval, floor, ceiling)

    def _finish(self, now, ticket, outcome, interval, floor, ceiling):
        if ticket is None:
            self.excluded_responses += 1
            return interval
        if ticket not in self.pending:
            self.stale_responses += 1
            return interval
        self.pending.remove(ticket)
        setattr(self, outcome, getattr(self, outcome)+1)
        self._seal_if_ready(now)
        if not self.sealed or self.pending:
            return interval
        if self.throttles:
            target = min(ceiling, max(interval, .005)*1.25)
            reason = 'settled_sample_burst_feedback'
        elif self.unknown:
            target = interval
            reason = 'settled_sample_unknown_hold'
        else:
            target = max(floor, interval/1.2)
            reason = 'settled_sample_successful_feedback'
        target = max(floor, target)
        self.last_decision = {
            'reason': reason, 'successes': self.successes, 'throttles': self.throttles,
            'unknown': self.unknown, 'sampled': self.sampled,
            'interval_before': interval, 'interval_after': target,
            'window_seconds': now-self.first_start,
        }
        self.decisions += 1
        self.reset_epoch(now)
        return target

    def snapshot(self):
        return {'mode': self.mode, 'stale_responses': self.stale_responses,
                'excluded_responses': self.excluded_responses,
                'fresh_successes': self.successes, 'fresh_throttles': self.throttles,
                'unknown_outcomes': self.unknown, 'sampled_requests': self.sampled,
                'pending_sample_requests': len(self.pending), 'sample_sealed': self.sealed,
                'sample_limit': self.sample_limit,
                'decisions': self.decisions, 'last_decision': self.last_decision}
