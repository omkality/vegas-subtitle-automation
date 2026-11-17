"""Frozen audio sets and paired model evaluation without repeated media extraction."""
from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import time
import wave

from .audio import extract_row
from .corpus import load_rows
from .metrics import word_errors, character_errors


def fingerprint(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(manifest: Path, output: Path, split: str='validation') -> dict:
    rows=[r for r in load_rows(manifest) if r['candidate'] and r['split']==split]
    if not rows:raise ValueError('No candidate examples in this split')
    output.mkdir(parents=True,exist_ok=True)
    items=[]
    for row in sorted(rows,key=lambda r:r['id']):
        audio=output/'audio'/(row['id']+'.wav')
        extract_row(row,audio)
        with wave.open(str(audio),'rb') as source:
            duration=source.getnframes()/source.getframerate()
        items.append({**row,'audio':audio.relative_to(output).as_posix(),
                      'audio_sha256':fingerprint(audio),'duration':duration})
    result={'schema':1,'split':split,'source_manifest_sha256':fingerprint(manifest),
            'reference_status':'editorial_timings_not_manually_aligned','items':items}
    (output/'set.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'examples':len(items),'seconds':sum(r['duration'] for r in items),'set':str(output/'set.json')}


def from_reviewed(manifest: Path, output: Path, split: str='validation') -> dict:
    rows=load_rows(manifest)
    if not rows:raise ValueError('No reviewed examples')
    allowed={'train','validation'} if split=='development' else {split}
    if any(r['split'] not in allowed or r['label_source']!='explicit_human_review' or not r['alignment_reviewed'] for r in rows):
        raise ValueError('Every evaluation example must have the requested split and explicit review')
    if output.exists():raise ValueError('Evaluation destination must be new')
    for row in rows:
        if fingerprint(manifest.parent/row['audio'])!=row['audio_sha256']:raise ValueError('Reviewed audio checksum mismatch')
    (output/'audio').mkdir(parents=True)
    items=[]
    for row in rows:
        target=output/'audio'/(row['id']+'.wav')
        shutil.copyfile(manifest.parent/row['audio'],target)
        items.append({**row,'audio':target.relative_to(output).as_posix()})
    package={'schema':1,'split':split,'source_manifest_sha256':fingerprint(manifest),
        'reference_status':'human_reviewed_text_and_boundaries','items':items,
        'evaluation_use':'development_diagnostic_includes_train' if split=='development' else split}
    (output/'set.json').write_text(json.dumps(package,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'examples':len(items),'seconds':sum(r['duration'] for r in items),'set':str(output/'set.json')}


def evaluate(set_path: Path, kind: str, model: Path, output: Path) -> dict:
    from .engines import load_engine
    dataset=json.loads(set_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=True)
    hashes={p.name:fingerprint(p) for p in sorted(model.iterdir()) if p.is_file()}
    started=time.monotonic()
    engine=load_engine(kind,model)
    load_seconds=time.monotonic()-started
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch=None
    results=[];failures=[]
    for row in dataset['items']:
        audio=set_path.parent/row['audio']
        try:
            if fingerprint(audio)!=row['audio_sha256']:raise ValueError('Audio checksum mismatch')
            started=time.monotonic()
            hypothesis=engine.transcribe(audio)
            elapsed=time.monotonic()-started
            item={'id':row['id'],'project':row['project'],'speaker':row['speaker'],
                  'split':row.get('split',dataset['split']),'audio_sha256':row['audio_sha256'],
                  'reference':row['text'],'hypothesis':hypothesis,'seconds':row['duration'],
                  'inference_seconds':elapsed,**word_errors(row['text'],hypothesis),
                  **character_errors(row['text'],hypothesis)}
            results.append(item)
            print(json.dumps(item,ensure_ascii=False),flush=True)
        except (ValueError,RuntimeError,OSError) as exc:
            failures.append({'id':row['id'],'error':str(exc)})
            print(json.dumps(failures[-1],ensure_ascii=False),flush=True)
        (output/'samples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    count=sum(r['reference_words'] for r in results)
    chars=sum(r['reference_characters'] for r in results)
    seconds=sum(r['seconds'] for r in results)
    versions={}
    for name in ('torch','torchaudio','transformers','faster-whisper','ctranslate2'):
        try:versions[name]=version(name)
        except PackageNotFoundError:pass
    report={'set_sha256':fingerprint(set_path),'audio_ids':[r['id'] for r in dataset['items']],
            'model':str(model),'model_sha256':hashes,'settings':engine.settings,'versions':versions,
            'examples':len(results),'failed_samples':failures,'reference_words':count,
            'word_errors':sum(r['errors'] for r in results),
            'wer':sum(r['errors'] for r in results)/count if count else None,
            'reference_characters':chars,'character_errors':sum(r['character_errors'] for r in results),
            'cer':sum(r['character_errors'] for r in results)/chars if chars else None,
            'audio_seconds':seconds,'model_load_seconds':load_seconds,
            'inference_seconds':sum(r['inference_seconds'] for r in results),
            'real_time_factor':sum(r['inference_seconds'] for r in results)/seconds if seconds else None,
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch else None,
            'projects':dict(Counter(r['project'] for r in results)),
            'reference_status':dataset.get('reference_status','unspecified'),
            'evaluation_use':dataset.get('evaluation_use',dataset.get('split')),
            'limitations':(['Editorial labels and timings need review'] if dataset.get('reference_status')!='human_reviewed_text_and_boundaries' else [])+
                          ['No independent final test score','Profanity recall requires a separate reviewed challenge set']}
    report['split_scores']={}
    for split in sorted({r['split'] for r in results}):
        part=[r for r in results if r['split']==split]
        n=sum(r['reference_words'] for r in part);c=sum(r['reference_characters'] for r in part)
        report['split_scores'][split]={'examples':len(part),'reference_words':n,
            'word_errors':sum(r['errors'] for r in part),'wer':sum(r['errors'] for r in part)/n if n else None,
            'cer':sum(r['character_errors'] for r in part)/c if c else None}
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
