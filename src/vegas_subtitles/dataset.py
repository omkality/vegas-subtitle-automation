"""Private staged audio, isolated holdout groups, and explicit reviewed training exports."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import wave

from .audio import extract_row
from .corpus import load_rows
from .evaluation import fingerprint
from .training_text import gigaam_label

SOFT_ISSUES={'masked_profanity_label','obfuscated_label_needs_review'}


def stage(manifest: Path, output: Path, workers: int=4) -> dict:
    if (output/'dataset.json').exists():
        raise ValueError('Staged dataset already exists. Choose a new output to preserve human reviews.')
    rows=load_rows(manifest)
    selected=[r for r in rows if r['candidate'] or (r['issues'] and set(r['issues'])<=SOFT_ISSUES)]
    group_counts=Counter(r['group'] for r in selected if r['split']=='train')
    eligible=[g for g,n in group_counts.items() if n>=8]
    held=sorted(eligible,key=lambda g:hashlib.sha256(('unseen-final:'+g).encode()).hexdigest())[:4]
    output.mkdir(parents=True,exist_ok=True)
    def prepare(row):
        ident=row['id']
        audio=output/'audio'/(ident+'.wav')
        context=output/'context'/(ident+'.wav')
        extract_row({**row,'candidate':True},audio)
        span=dict(row['spans'][0])
        before=min(.5,span['source_start'])
        span['source_start']-=before
        span['source_end']+=.5
        extract_row({**row,'candidate':True,'spans':[span]},context)
        with wave.open(str(audio),'rb') as source:duration=source.getnframes()/source.getframerate()
        with wave.open(str(context),'rb') as source:context_duration=source.getnframes()/source.getframerate()
        return {**row,'split_original':row['split'],'split':'holdout' if row['group'] in held else row['split'],
                'audio':audio.relative_to(output).as_posix(),'audio_sha256':fingerprint(audio),'duration':duration,
                'context_audio':context.relative_to(output).as_posix(),'context_sha256':fingerprint(context),
                'context_duration':context_duration,'trim_start':before,'trim_end':min(before+duration,context_duration),
                'training_approved':False,'alignment_reviewed':False,'reviewed_text':None}
    items=[];failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,future in enumerate(as_completed({pool.submit(prepare,r):r['id'] for r in selected}),1):
            try:items.append(future.result())
            except (OSError,RuntimeError,ValueError) as exc:failures.append(str(exc))
            if i%50==0 or i==len(selected):print(f'Staged {i}/{len(selected)}',flush=True)
    items.sort(key=lambda r:r['id'])
    package={'schema':1,'source_manifest_sha256':fingerprint(manifest),'holdout_groups':held,'items':items}
    (output/'dataset.json').write_text(json.dumps(package,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'examples':len(items),'minutes':sum(r['duration'] for r in items)/60,
        'splits':dict(Counter(r['split'] for r in items)),'holdout_groups':held,
        'needs_unmasking':sum(bool(set(r['issues']) & SOFT_ISSUES) for r in items),
        'approved':0,'failures':failures,'status':'staged_for_review_not_approved_training_data'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def export_reviewed(dataset_path: Path, review_path: Path, output: Path) -> dict:
    package=json.loads(dataset_path.read_text(encoding='utf-8'))
    review_bytes=review_path.read_bytes()
    review=json.loads(review_bytes.decode('utf-8-sig'))
    if review['dataset_sha256']!=fingerprint(dataset_path):raise ValueError('Review belongs to a different dataset version')
    originals={r['id']:r for r in package['items']}
    approved=[];seen=set()
    for decision in review['decisions']:
        ident=decision['id']
        if ident not in originals or ident in seen:raise ValueError('Unknown or duplicated review ID')
        seen.add(ident)
        if not isinstance(decision.get('training_approved',False),bool):raise ValueError('Approval must be boolean')
        if not decision.get('training_approved'):continue
        if decision.get('rejected'):raise ValueError('Rejected examples cannot be approved')
        if decision.get('alignment_reviewed') is not True:raise ValueError('Approved text also requires alignment review')
        if review.get('schema',1)>=2 and (decision.get('text_reviewed') is not True or decision.get('coverage')!='full'):
            raise ValueError('Approved examples require reviewed text and complete audio')
        text=decision['text'].strip()
        if not text or '*' in text:raise ValueError('Training labels must contain the actual spoken words')
        if decision['speaker'] not in ('self','friend'):raise ValueError('Speaker must be explicitly reviewed')
        row=originals[ident]
        start,end=float(decision['trim_start']),float(decision['trim_end'])
        if not 0<=start<end<=row['context_duration']+.0001:raise ValueError('Invalid reviewed audio boundaries')
        if fingerprint(dataset_path.parent/row['context_audio'])!=row['context_sha256']:
            raise ValueError('Context audio checksum mismatch')
        approved.append((row,decision,text,start,end))
    if output.exists():raise ValueError('Export destination must be new')
    output.mkdir(parents=True)
    manifests={};counts=Counter();excluded=[];giga_counts=Counter()
    for row,decision,text,start,end in approved:
        context=dataset_path.parent/row['context_audio']
        if fingerprint(context)!=row['context_sha256']:raise ValueError('Context audio checksum mismatch')
        audio=output/'audio'/(row['id']+'.wav');audio.parent.mkdir(exist_ok=True)
        with wave.open(str(context),'rb') as source:
            source.setpos(round(start*16000))
            data=source.readframes(round((end-start)*16000))
        with wave.open(str(audio),'wb') as target:
            target.setnchannels(1);target.setsampwidth(2);target.setframerate(16000);target.writeframes(data)
        split=row['split'];counts[split]+=1
        normalized=gigaam_label(text)
        manifests.setdefault(split,[]).append({'id':row['id'],'audio':audio.relative_to(output).as_posix(),
            'audio_sha256':fingerprint(audio),'duration':len(data)/32000,'text':text,'speaker':decision['speaker'],
            'group':row['group'],'label_source':'explicit_human_review','alignment_reviewed':True,
            'project':row.get('project',row['group']),'split':split,
            'gigaam_normalization':normalized})
        if not normalized['supported']:
            excluded.append({'id':row['id'],'split':split,'text':text,
                'reason':'Write the spoken Russian form of digits/Latin names before GigaAM training',
                'unsupported_characters':normalized['unsupported_characters']})
    for split,rows in manifests.items():
        (output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        # Model-neutral JSONL is canonical. GigaAM's native trainer also accepts this TSV.
        import csv
        with (output/(split+'.tsv')).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t');writer.writerow(['path','duration','transcription'])
            usable=[r for r in rows if r['gigaam_normalization']['supported']]
            writer.writerows((r['audio'],r['duration'],r['gigaam_normalization']['text']) for r in usable)
            giga_counts[split]=len(usable)
    development=[r for split,rows in manifests.items() if split in ('train','validation') for r in rows]
    (output/'development.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in development),encoding='utf-8')
    summary={'approved':len(approved),'splits':dict(counts),'dataset_sha256':fingerprint(dataset_path),
             'review_sha256':hashlib.sha256(review_bytes).hexdigest(),'training_run':False,
             'gigaam_splits':dict(giga_counts),'gigaam_excluded_labels':len(excluded)}
    (output/'gigaam-exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
