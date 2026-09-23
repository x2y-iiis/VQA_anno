"""Normalize an explicit final incompletion statement without inferring state."""
import re


FINAL_INCOMPLETE = re.compile(
    r'(?:^|(?<=[.!?;]))\s*(?:the\s+)?task\s+(?:is\s+)?not\s+(?:yet\s+)?complete'
    r'(?P<reason>\s+(?:as|because)\s+[^.!?;]+)?[.!]?\s*$',
    re.IGNORECASE,
)

# Models often make the required judgment explicitly but phrase the subject as
# "the overall task", "the wipe-screen task", or "all required tasks".  That
# is a representation difference, not a different visual judgment.  Preserve
# the complete original text and append the canonical machine-readable suffix.
EXPLICIT_FINAL_INCOMPLETE = re.compile(
    r'\b(?:is|are)\s+not\s+(?:yet\s+)?complete[.!]?\s*$', re.IGNORECASE,
)
EXPLICIT_FINAL_COMPLETE = re.compile(
    r'\b(?:is|are)\s+complete[.!]?\s*$', re.IGNORECASE,
)


def _explicit_task_judgment(text):
    """Return a canonical suffix only for an unqualified final task claim."""
    final_sentence = re.split(r'(?<=[.!;])\s+', text)[-1].strip()
    lowered = final_sentence.lower()
    if (not re.search(r'(?<!sub)\btasks?\b', lowered)
            or lowered.startswith('if ')
            or re.search(r'\b(?:may|might|could)\b', lowered)
            or '"' in final_sentence or "'" in final_sentence
            or final_sentence.endswith('?')):
        return None
    if EXPLICIT_FINAL_INCOMPLETE.search(final_sentence):
        return 'Task not yet complete.'
    if EXPLICIT_FINAL_COMPLETE.search(final_sentence):
        return 'Task complete.'
    return None


def normalize_completion_suffix(text):
    """Keep the canonical output contract; reject conditional/implicit claims."""
    stripped = text.rstrip()
    for suffix, canonical in [('task not yet complete.', 'Task not yet complete.'),
                              ('task complete.', 'Task complete.')]:
        if stripped.lower().endswith(suffix):
            return stripped[:-len(suffix)] + canonical
    match = FINAL_INCOMPLETE.search(stripped)
    if match is not None:
        if match.group('reason'):
            # Preserve every descriptive word; only add the equivalent marker.
            return stripped.rstrip('.!') + '; Task not yet complete.'
        return stripped[:match.start()].rstrip() + (' ' if match.start() else '') + 'Task not yet complete.'
    canonical = _explicit_task_judgment(stripped)
    if canonical is not None:
        return stripped.rstrip('.!') + '; ' + canonical
    return text
