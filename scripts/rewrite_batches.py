"""Checkpointed, cardinality-checked LAS English rewrite requests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading

from request_parallel import atomic_json

_LOCKS = [threading.Lock() for _ in range(256)]


def rewrite_batches(request, call, cache_root: Path, batch_size=32):
    """Keep the vendored model/system prompt and every source segment in order.

    Split only the numbered rewrite input, never the video or its time intervals.
    Mismatched outputs are never truncated, padded or registered as valid.
    """
    from las_annotation.postprocessing import parse_rewrite_response
    from las_annotation.models import PipelineDataError
    try:
        text = request['input'][-1]['content'][0]['text']
    except (KeyError, IndexError, TypeError):
        return call(request)
    lines = text.splitlines()
    if not lines or not all(re.match(r'^\d+\. \[Skill: .*\] \[Description: .*\]$', line) for line in lines):
        return call(request)

    def combined(children, count):
        output = []
        for response, expected in children:
            output.extend(parse_rewrite_response(response, expected))
        if len(output) != count:
            raise PipelineDataError('Batched rewrite cardinality mismatch.')
        return {'object': 'checkpointed_las_rewrite_batches/v1',
                'output_text': '\n'.join(output),
                'source_description_count': count,
                'chunks': [{'count': expected, 'response': response} for response, expected in children]}

    def run(part):
        current = deepcopy(request)
        # Preserve the original source numbers and exact description/skill text.
        current['input'][-1]['content'][0]['text'] = '\n'.join(part)
        digest = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
        path = cache_root/digest[:2]/f'{digest}.json'
        with _LOCKS[int(digest[:2], 16)]:
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get('request') == current:
                    parse_rewrite_response(cached['response'], len(part))
                    return cached['response']
            response = call(current)
            try:
                parse_rewrite_response(response, len(part))
            except PipelineDataError:
                # Retain invalid output for audit, then re-request smaller groups.
                atomic_json(path.with_suffix('.invalid.json'), {'request': current, 'response': response})
                if len(part) == 1:
                    response = call(current)
                    parse_rewrite_response(response, 1)
                else:
                    # Recursion runs outside this stripe lock (different hashes
                    # can share a stripe); release it before taking child locks.
                    response = None
            if response is not None:
                atomic_json(path, {'request': current, 'response': response})
                return response
        middle = len(part)//2
        response = combined([(run(part[:middle]), middle),
                             (run(part[middle:]), len(part)-middle)], len(part))
        atomic_json(path, {'request': current, 'response': response})
        return response

    parts = [lines[start:start+batch_size] for start in range(0, len(lines), batch_size)]
    results = [(run(part), len(part)) for part in parts]
    return results[0][0] if len(results) == 1 else combined(results, len(lines))
