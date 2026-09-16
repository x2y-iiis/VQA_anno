"""Shared structural checks for the approved immediate atomic-action target."""
import re


def validate_atomic_action(value: object) -> str:
    """Validate representation, without claiming to verify visual semantics."""
    if not isinstance(value, str):
        raise ValueError('ecot_atomic_action_not_string')
    text = re.sub(r'\s+', ' ', value.strip())
    if text == 'None':
        return text
    if text.lower() == 'none':
        raise ValueError('ecot_atomic_action_invalid_phrase')
    if not text or text != text.lower() or len(text) > 240:
        raise ValueError('ecot_atomic_action_invalid_phrase')
    if any(ord(character) < 32 for character in text):
        raise ValueError('ecot_atomic_action_invalid_english_phrase')
    return text
