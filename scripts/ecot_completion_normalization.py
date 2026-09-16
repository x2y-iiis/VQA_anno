"""Normalize an explicit final incompletion statement without inferring state."""
import re


FINAL_INCOMPLETE = re.compile(
    r'(?:^|(?<=[.!?;]))\s*(?:the\s+)?task\s+(?:is\s+)?not\s+(?:yet\s+)?complete'
    r'(?P<reason>\s+(?:as|because)\s+[^.!?;]+)?[.!]?\s*$',
    re.IGNORECASE,
)


def normalize_completion_suffix(text):
    """Keep the canonical output contract; reject conditional/implicit claims."""
    stripped = text.rstrip()
    for suffix, canonical in [('task not yet complete.', 'Task not yet complete.'),
                              ('task complete.', 'Task complete.')]:
        if stripped.lower().endswith(suffix):
            return stripped[:-len(suffix)] + canonical
    match = FINAL_INCOMPLETE.search(stripped)
    if match is None:
        return text
    if match.group('reason'):
        # Preserve every descriptive word; only add the equivalent marker.
        return stripped.rstrip('.!') + '; Task not yet complete.'
    return stripped[:match.start()].rstrip() + (' ' if match.start() else '') + 'Task not yet complete.'
