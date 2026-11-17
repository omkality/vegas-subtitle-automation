"""Keep a display transcript separate from the chosen Russian character vocabulary."""
from .metrics import words

RUSSIAN_ALPHABET=set(' абвгдежзийклмнопрстуфхцчшщъыьэюя')


def gigaam_label(text: str) -> dict:
    normalized=' '.join(words(text))
    unsupported=sorted(set(normalized)-RUSSIAN_ALPHABET)
    return {'text':normalized,'supported':bool(normalized) and not unsupported,
            'unsupported_characters':unsupported}
