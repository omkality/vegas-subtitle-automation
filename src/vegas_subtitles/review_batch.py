"""Prepare a reproducible review queue without overwriting an active batch."""
from __future__ import annotations
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import random
import wave

from .evaluation import fingerprint


def prepare_batch(dataset: Path, output: Path, audit_root: Path, limit: int=120,
                  exclude_batches: tuple[Path, ...]=(), selection: str='prioritized') -> dict:
    if output.exists():raise ValueError('Review batch already exists. Choose a new output.')
    if limit<1:raise ValueError('Review limit must be positive')
    package=json.loads(dataset.read_text(encoding='utf-8'))
    dataset_hash=fingerprint(dataset)
    originals={r['id']:r for r in package['items']}
    excluded=set();sources=[]
    for path in exclude_batches:
        previous=json.loads(path.read_text(encoding='utf-8'))
        # Legacy queues had no dataset hash. IDs must still belong to this base.
        if previous.get('dataset_sha256',dataset_hash)!=dataset_hash:
            raise ValueError('Excluded batch belongs to a different dataset')
        if not set(previous['ids'])<=originals.keys():raise ValueError('Unknown excluded review IDs')
        excluded.update(previous['ids'])
        sources.append({'file':path.name,'sha256':fingerprint(path)})
    rows=[r for r in package['items'] if r['split'] in ('train','validation') and r['id'] not in excluded]
    models={}
    for path in sorted(audit_root.glob('*/suggestions.json')):
        models[path.parent.name]={r['id']:r for r in json.loads(path.read_text(encoding='utf-8'))}
    random.Random(20261117).shuffle(rows)
    def priority(row):
        errors=max((m.get(row['id'],{}).get('reference_word_errors') or 0 for m in models.values()),default=0)
        return (not bool(row['issues']),-errors)
    if selection=='prioritized':
        rows.sort(key=priority)
        selected=[r for r in rows if r['split']=='validation'][:limit//2]
        selected += [r for r in rows if r['split']=='train'][:limit-len(selected)]
        chosen={r['id'] for r in selected}
        selected += [r for r in rows if r['id'] not in chosen][:limit-len(selected)]
    elif selection=='balanced':
        buckets=defaultdict(list)
        for row in rows:
            duration=row.get('duration',0)
            length='short' if duration<1 else 'medium' if duration<2 else 'long'
            buckets[(row['split'],row.get('group',row.get('project','')),row.get('speaker','unknown'),length)].append(row)
        selected=[]
        keys=list(buckets)
        random.Random(20261117).shuffle(keys)
        while keys and len(selected)<limit:
            for key in keys:
                if len(selected)>=limit:break
                selected.append(buckets[key].pop())
            keys=[k for k in keys if buckets[k]]
    else:raise ValueError('Unknown review selection policy')
    ids=[r['id'] for r in selected]
    predictions={i:{name:values[i]['hypothesis'] for name,values in models.items() if i in values} for i in ids}
    result={'title':f'Проверка разметки: {len(ids)} фрагментов','dataset_sha256':dataset_hash,
            'ids':ids,'predictions':predictions,'selection':selection+'_review_not_random_evaluation',
            'excluded_batches':sources}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'examples':len(ids),'batch':str(output),'excluded_splits':['test','holdout']}


def extend_context(dataset: Path, batch: Path, output: Path, padding: float=2.5, workers: int=4) -> dict:
    """Create a new queue-specific dataset, preserving the original audio and reviews."""
    from .audio import extract_row
    if output.exists():raise ValueError('Context destination must be new')
    if not math.isfinite(padding) or not 0<padding<=10:raise ValueError('Padding must be between 0 and 10 seconds')
    if workers<1:raise ValueError('Workers must be positive')
    package=json.loads(dataset.read_text(encoding='utf-8'))
    spec=json.loads(batch.read_text(encoding='utf-8'))
    dataset_hash=fingerprint(dataset)
    if spec.get('dataset_sha256')!=dataset_hash:raise ValueError('Batch belongs to a different dataset')
    original={r['id']:r for r in package['items']}
    ids=spec['ids']
    if len(set(ids))!=len(ids) or not set(ids)<=original.keys():raise ValueError('Invalid review IDs')
    rows=[original[i] for i in ids]
    for row in rows:
        if row['split'] not in ('train','validation'):raise ValueError('Context preparation excludes test and holdout')
        if len(row['spans'])!=1 or row['spans'][0]['rate']!=1:raise ValueError('Context requires one contiguous source span')
        if fingerprint(dataset.parent/row['audio'])!=row['audio_sha256']:raise ValueError('Original audio checksum mismatch')
    output.mkdir(parents=True)
    def prepare(row):
        span=dict(row['spans'][0]);before=min(padding,span['source_start'])
        span['source_start']-=before;span['source_end']+=padding
        context=output/'context'/(row['id']+'.wav')
        extract_row({**row,'candidate':True,'spans':[span]},context)
        with wave.open(str(context),'rb') as source:
            duration=source.getnframes()/source.getframerate()
        if duration<before+row['duration']-.002:raise ValueError('Extended context does not cover the original clip')
        return {**row,'audio':Path(os.path.relpath((dataset.parent/row['audio']).resolve(),output.resolve())).as_posix(),
            'context_audio':context.relative_to(output).as_posix(),'context_sha256':fingerprint(context),
            'context_duration':duration,'trim_start':before,'trim_end':min(before+row['duration'],duration),
            'training_approved':False,'alignment_reviewed':False,'reviewed_text':None}
    prepared={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(prepare,row):row['id'] for row in rows}
        for index,future in enumerate(as_completed(futures),1):
            result=future.result();prepared[result['id']]=result
            if index%50==0 or index==len(rows):print(f'Extended context {index}/{len(rows)}',flush=True)
    derived={'schema':1,'source_manifest_sha256':package.get('source_manifest_sha256'),
        'parent_dataset_sha256':dataset_hash,'source_batch_sha256':fingerprint(batch),
        'holdout_groups':package.get('holdout_groups',[]),'context_padding':padding,
        'items':[prepared[i] for i in ids]}
    destination=output/'dataset.json'
    destination.write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    spec={**spec,'parent_dataset_sha256':dataset_hash,'dataset_sha256':fingerprint(destination)}
    (output/'batch.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'examples':len(rows),'dataset':str(destination),'batch':str(output/'batch.json'),
        'context_padding_seconds':padding,'excluded_splits':['test','holdout']}
