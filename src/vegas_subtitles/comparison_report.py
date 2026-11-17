"""Audit paired runs and generate an offline report without publishing private data."""
from __future__ import annotations
import html
import json
from pathlib import Path

from .profanity_metrics import diagnose

NAMES={
    'gigaam-rnnt':'GigaAM v3 RNN-T',
    'gigaam-e2e-rnnt':'GigaAM v3 E2E RNN-T',
    'gigaam-ctc':'GigaAM v3 CTC',
    'gigaam-e2e-ctc':'GigaAM v3 E2E CTC',
    'gigaam-multilingual':'GigaAM Multilingual 220M',
    'gigaam-multilingual-large':'GigaAM Multilingual 600M',
    'large-v3':'Whisper large-v3',
    'qwen3-asr-1.7b':'Qwen3-ASR 1.7B',
    'parakeet-tdt-v3':'NVIDIA Parakeet TDT v3',
    't-one':'T-one, greedy без KenLM',
    'whisper-large-v3-russian':'Whisper Russian, beam 5',
    'whisper-large-v3-russian-greedy':'Whisper Russian, greedy',
}


def generate(runs: Path, output: Path) -> dict:
    models={}
    for file in sorted(runs.glob('*/summary.json')):
        report=json.loads(file.read_text(encoding='utf-8'))
        samples=json.loads((file.parent/'samples.json').read_text(encoding='utf-8'))
        if report['failed_samples']:raise ValueError(f'Incomplete run: {file.parent.name}')
        if len(samples)!=report['examples'] or {r['id'] for r in samples}!=set(report['audio_ids']):
            raise ValueError(f'Sample/report mismatch: {file.parent.name}')
        models[file.parent.name]={'label':NAMES.get(file.parent.name,file.parent.name),
            'report':report,'lexical_profanity_diagnostic':diagnose(samples),
            'worst_cases':sorted(samples,key=lambda r:r['errors'],reverse=True)[:3]}
    if not models:raise ValueError('No completed comparison runs')
    reports=[v['report'] for v in models.values()]
    if len({r['set_sha256'] for r in reports})!=1 or len({tuple(r['audio_ids']) for r in reports})!=1:
        raise ValueError('Models were not evaluated on the same frozen audio set')
    expected=reports[0]['reference_words']
    if any(r['reference_words']!=expected for r in reports):raise ValueError('Reference denominators differ')
    ranked=sorted(models,key=lambda k:(models[k]['report']['split_scores']['validation']['wer'],
        models[k]['report']['split_scores']['validation']['cer']))
    best=min(v['report']['split_scores']['validation']['wer'] for v in models.values())
    tied=[k for k in ranked if models[k]['report']['split_scores']['validation']['wer']==best]
    validation=reports[0]['split_scores']['validation']
    result={'protocol':'Paired ASR comparison on human-reviewed audio before any fine-tuning',
        'set_sha256':reports[0]['set_sha256'],'examples':reports[0]['examples'],'reference_words':expected,
        'audio_seconds':reports[0]['audio_seconds'],'best_validation_wer_models':tied,'models':models,
        'limitations':['Validation is small and prioritizes difficult examples.',
            'Overall development scores include future training examples and are not an independent test.',
            'Swear-family counts are heuristic lexical diagnostics, not censor timing or safety guarantees.',
            'Backends and decoder settings differ and are recorded per model.']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    body=[]
    for key in ranked:
        value=models[key];r=value['report'];v=r['split_scores']['validation'];d=value['lexical_profanity_diagnostic']
        body.append(f"<tr><td>{html.escape(value['label'])}</td><td>{v['word_errors']}/{v['reference_words']} · {v['wer']*100:.1f}%</td>"
            f"<td>{r['wer']*100:.1f}%</td><td>{d['matched_families']}/{d['reference_occurrences']}</td><td>{r['inference_seconds']:.2f} с</td></tr>")
    details=[]
    for key in ranked:
        value=models[key];parts=[]
        for sample in value['worst_cases']:
            parts.append('<p><b>Разметка:</b> '+html.escape(sample['reference'])+'<br><b>Модель:</b> '+html.escape(sample['hypothesis'])+'</p>')
        settings=html.escape(json.dumps(value['report']['settings'],ensure_ascii=False,indent=2))
        details.append('<details><summary>'+html.escape(value['label'])+'</summary><pre>'+settings+'</pre>'+''.join(parts)+'</details>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Сравнение ASR по проверенной разметке</title><style>
body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17243c;background:#f8fafc}
h1{font-size:30px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:11px;border-bottom:1px solid #d8e2ef;text-align:left}th{background:#e8eff9}details{border:1px solid #cddaea;background:white;padding:12px;margin:10px 0;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap}p{max-width:980px}.scroll{overflow-x:auto}.note{background:#e9f2ff;padding:14px;border-radius:8px}
</style><h1>Сравнение моделей по твоей разметке</h1>'''
    page+=f"<p>{result['examples']} общих фрагментов, {result['reference_words']} слов, {result['audio_seconds']:.1f} секунды звука. Модели запускались последовательно на GPU.</p>"
    page+=f'<p class="note">Validation содержит {validation["examples"]} фрагментов и {validation["reference_words"]} слов, преимущественно сложные случаи. Одна ошибка меняет WER примерно на {100/validation["reference_words"]:.1f} процентного пункта. Общий результат на {result["examples"]} примерах включает будущий train. Эти оценки не являются независимым финальным тестом.</p>'
    page+=f'<div class="scroll"><table><tr><th>Модель / конфигурация</th><th>Validation: ошибки / слова, WER</th><th>Все {result["examples"]}: WER</th><th>Семейства мата: совпадения</th><th>Inference</th></tr>'+''.join(body)+'</table></div>'
    page+='<p>WER: меньше лучше. Регистр, пунктуация и ё/е нормализованы. Числа и латинские названия не заменялись догадками. Время исключает загрузку модели и подготовку файлов, настройки декодирования приведены ниже. T-one проверен без KenLM.</p>'
    page+='<p>Подсчёт мата сравнивает число вхождений семейств слов внутри каждой реплики. Он допускает варианты написания вроде «блять» и «блядь», но учитывает пропущенные повторы. Словарь эвристический. Этот показатель не измеряет границы сигнала или оставшиеся слышимые части слова.</p>'
    page+='<h2>Настройки и сложные случаи</h2>'+''.join(details)+'</html>'
    (output/'comparison.html').write_text(page,encoding='utf-8')
    return {'configurations':len(models),'examples':result['examples'],'best_validation_wer_models':tied,
            'json':str(output/'comparison.json'),'html':str(output/'comparison.html')}
