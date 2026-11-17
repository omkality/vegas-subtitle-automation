"""Secondary lexical diagnostics, not an assessment of censor timing or coverage."""
from collections import Counter
import re
from .metrics import words


def family(word: str) -> str | None:
    if word=='бля' or word.startswith(('бляд','блят')):return 'blyad'
    if re.match(r'^(?:за|вы|по|рас|раз|от|на|под|про|об|о|с)?пиз(?:д|ж)',word):return 'pizda'
    if word=='хули' or re.match(r'^(?:на|по|за|до|ни|о|а|под|рас|раз|с|от|об|при|про|в|не)?ху[йяеи]',word):return 'huy'
    if re.match(r'^(?:за|на|вы|от|при|по|про|до|у|о|об|объ|пере|раз|разъ|с|съ|в|въ|под|подъ)?еб',word):return 'ebat'
    if word in ('сука','суки','суку','суке','сукой','сучка','сучки','сучку','сучья','сучье'):return 'suka'
    return None


def diagnose(samples: list[dict]) -> dict:
    reference=predicted=matched=0
    for item in samples:
        a=Counter(f for w in words(item['reference']) if (f:=family(w)))
        b=Counter(f for w in words(item['hypothesis']) if (f:=family(w)))
        reference+=sum(a.values());predicted+=sum(b.values())
        matched+=sum((a&b).values())
    return {'reference_occurrences':reference,'predicted_occurrences':predicted,'matched_families':matched,
            'missed_occurrences':reference-matched,'extra_occurrences':predicted-matched,
            'recall':matched/reference if reference else None,'precision':matched/predicted if predicted else None,
            'limitation':'Conservative word-family counts per utterance. No word timing, censor completeness, or independently annotated profanity labels.'}
