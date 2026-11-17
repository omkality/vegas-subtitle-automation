"""Suggest word boundaries without granting human-review or training approval."""
from __future__ import annotations
import json
from pathlib import Path

from .audio import read_wav
from .evaluation import fingerprint


def suggest(dataset: Path, model: Path, output: Path) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoProcessor
    torch.set_num_threads(8)
    processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    aligner=AutoModelForTokenClassification.from_pretrained(model,dtype=torch.bfloat16,
        device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
    package=json.loads(dataset.read_text(encoding='utf-8'))
    selected=[r for r in package['items'] if r['split'] in ('train','validation') and not r['issues']]
    output.mkdir(parents=True,exist_ok=True)
    results=[];failures=[]
    for i,row in enumerate(selected,1):
        try:
            audio=dataset.parent/row['context_audio']
            if fingerprint(audio)!=row['context_sha256']:raise ValueError('Context checksum mismatch')
            inputs,word_lists=processor.prepare_forced_aligner_inputs(
                audio=read_wav(audio),transcript=row['text'],language='Russian')
            inputs=inputs.to(aligner.device,aligner.dtype)
            with torch.inference_mode():logits=aligner(**inputs).logits
            timings=processor.decode_forced_alignment(logits=logits,input_ids=inputs['input_ids'],
                word_lists=word_lists,timestamp_token_id=aligner.config.timestamp_token_id)[0]
            flags=[]
            if not timings:flags.append('no_word_boundaries')
            if any(w['end_time']<=w['start_time'] for w in timings):flags.append('zero_or_negative_word_duration')
            if any(w['start_time']<0 or w['end_time']>row['context_duration']+.08 for w in timings):flags.append('outside_context')
            if any(a['start_time']>b['start_time'] for a,b in zip(timings,timings[1:])):flags.append('non_monotonic_alignment')
            start=max(0,min(w['start_time'] for w in timings)-.08) if timings else None
            end=min(row['context_duration'],max(w['end_time'] for w in timings)+.12) if timings else None
            results.append({'id':row['id'],'words':timings,'suggested_start':start,'suggested_end':end,
                'flags':flags,'alignment_reviewed':False,'training_approved':False})
        except (ValueError,RuntimeError,OSError) as exc:failures.append({'id':row['id'],'error':str(exc)})
        if i%50==0 or i==len(selected):
            print(f'Aligned {i}/{len(selected)}',flush=True)
            (output/'suggestions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'dataset_sha256':fingerprint(dataset),'model_sha256':{p.name:fingerprint(p) for p in model.iterdir() if p.is_file()},
        'aligned':len(results),'flagged':sum(bool(r['flags']) for r in results),'failures':failures,
        'excluded_splits':['test','holdout'],'limitation':'Forced alignment cannot prove that the supplied transcript was spoken.'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
