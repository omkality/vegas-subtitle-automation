"""ASR suggestions for label review, kept separate from benchmark scores."""
from __future__ import annotations
import json
from pathlib import Path
from .engines import load_engine
from .evaluation import fingerprint
from .metrics import word_errors


def predict(dataset: Path, model: Path, kind: str, output: Path) -> dict:
    package=json.loads(dataset.read_text(encoding='utf-8'))
    rows=[r for r in package['items'] if r['split'] in ('train','validation')]
    engine=load_engine(kind,model)
    output.mkdir(parents=True,exist_ok=True)
    results=[];failures=[]
    for i,row in enumerate(rows,1):
        try:
            audio=dataset.parent/row['audio']
            if fingerprint(audio)!=row['audio_sha256']:raise ValueError('Audio checksum mismatch')
            hypothesis=engine.transcribe(audio)
            results.append({'id':row['id'],'hypothesis':hypothesis,'reference':row['text'],
                'reference_word_errors':word_errors(row['text'],hypothesis)['errors'] if not row['issues'] else None,
                'issues':row['issues'],'training_approved':False})
        except (RuntimeError,ValueError,OSError) as exc:failures.append({'id':row['id'],'error':str(exc)})
        if i%50==0 or i==len(rows):
            print(f'Predicted {i}/{len(rows)}',flush=True)
            (output/'suggestions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'dataset_sha256':fingerprint(dataset),'model_sha256':{p.name:fingerprint(p) for p in model.iterdir() if p.is_file()},
        'examples':len(results),'failures':failures,'excluded_splits':['test','holdout'],
        'status':'review_suggestions_not_ground_truth_or_benchmark'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
