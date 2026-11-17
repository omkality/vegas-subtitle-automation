"""Prepare a reproducible review queue without overwriting an active batch."""
from __future__ import annotations
import json
from pathlib import Path
import random

from .evaluation import fingerprint


def prepare_batch(dataset: Path, output: Path, audit_root: Path, limit: int=120) -> dict:
    if output.exists():raise ValueError('Review batch already exists. Choose a new output.')
    if limit<1:raise ValueError('Review limit must be positive')
    package=json.loads(dataset.read_text(encoding='utf-8'))
    rows=[r for r in package['items'] if r['split'] in ('train','validation')]
    models={}
    for path in sorted(audit_root.glob('*/suggestions.json')):
        models[path.parent.name]={r['id']:r for r in json.loads(path.read_text(encoding='utf-8'))}
    random.Random(20261117).shuffle(rows)
    def priority(row):
        errors=max((m.get(row['id'],{}).get('reference_word_errors') or 0 for m in models.values()),default=0)
        return (not bool(row['issues']),-errors)
    rows.sort(key=priority)
    # Corrected validation labels help model selection, while training labels help adaptation.
    selected=[r for r in rows if r['split']=='validation'][:limit//2]
    selected += [r for r in rows if r['split']=='train'][:limit-len(selected)]
    chosen={r['id'] for r in selected}
    selected += [r for r in rows if r['id'] not in chosen][:limit-len(selected)]
    ids=[r['id'] for r in selected]
    predictions={i:{name:values[i]['hypothesis'] for name,values in models.items() if i in values} for i in ids}
    result={'title':f'Проверка разметки: {len(ids)} фрагментов','dataset_sha256':fingerprint(dataset),
            'ids':ids,'predictions':predictions,'selection':'prioritized_review_not_random_evaluation'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'examples':len(ids),'batch':str(output),'excluded_splits':['test','holdout']}
