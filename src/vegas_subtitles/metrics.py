from __future__ import annotations
import re


def words(text: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))


def word_errors(reference: str, hypothesis: str) -> dict:
    a, b = words(reference), words(hypothesis)
    previous = list(range(len(b)+1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    errors = previous[-1]
    return {"errors": errors, "reference_words": len(a),
            "wer": errors/len(a) if a else (0. if not b else None)}
