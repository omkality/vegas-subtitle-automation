from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from .corpus import build_manifest, load_rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="vegas-subtitles")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Build conservative candidate labels from native exports")
    manifest.add_argument("--exports", type=Path, default=Path("local/corpus/projects"))
    manifest.add_argument("--output", type=Path, default=Path("local/dataset"))
    download = commands.add_parser("download-model", help="Download a revision-pinned model with SHA256 verification")
    download.add_argument("name", help="Alias from configs/models.lock.json")
    download.add_argument("--registry", type=Path, default=Path("configs/models.lock.json"))
    download.add_argument("--output", type=Path, default=Path("local/models"))
    evaluation=commands.add_parser('prepare-evaluation',help='Freeze identical audio and labels for every ASR backend')
    evaluation.add_argument('--manifest',type=Path,default=Path('local/dataset/manifest.jsonl'))
    evaluation.add_argument('--output',type=Path,default=Path('local/evaluation/validation'))
    evaluation.add_argument('--split',choices=['train','validation','test'],default='validation')
    gold=commands.add_parser('prepare-reviewed-evaluation',help='Freeze human-corrected shared audio for a new paired comparison')
    gold.add_argument('--manifest',type=Path,required=True)
    gold.add_argument('--output',type=Path,required=True)
    gold.add_argument('--split',choices=['validation','holdout','development'],default='validation')
    paired=commands.add_parser('evaluate',help='Evaluate a backend on a frozen paired audio set')
    paired.add_argument('--set',type=Path,default=Path('local/evaluation/validation/set.json'))
    paired.add_argument('--engine',choices=['whisper','qwen','gigaam','parakeet','russian-whisper','russian-whisper-beam5','t-one'],required=True)
    paired.add_argument('--model',type=Path,required=True)
    paired.add_argument('--output',type=Path,required=True)
    compare=commands.add_parser('comparison-report',help='Verify paired runs and render a private comparison report')
    compare.add_argument('--runs',type=Path,required=True)
    compare.add_argument('--output',type=Path,required=True)
    staging=commands.add_parser('stage-dataset',help='Extract private audio and reserve unseen recording groups')
    staging.add_argument('--manifest',type=Path,default=Path('local/dataset/manifest.jsonl'))
    staging.add_argument('--output',type=Path,default=Path('local/training-staged'))
    staging.add_argument('--workers',type=int,default=4)
    reviewed=commands.add_parser('export-reviewed',help='Export only explicitly reviewed audio-text pairs')
    reviewed.add_argument('--dataset',type=Path,default=Path('local/training-staged/dataset.json'))
    reviewed.add_argument('--review',type=Path,required=True)
    reviewed.add_argument('--output',type=Path,required=True)
    review_ui=commands.add_parser('review',help='Serve a local human annotation batch')
    review_ui.add_argument('--dataset',type=Path,default=Path('local/training-staged/dataset.json'))
    review_ui.add_argument('--batch',type=Path,default=Path('local/review-batches/current.json'))
    review_ui.add_argument('--saved',type=Path,default=Path('local/reviews/batch-001.json'))
    review_ui.add_argument('--host',default='127.0.0.1')
    review_ui.add_argument('--port',type=int,default=8765)
    batch=commands.add_parser('prepare-review-batch',help='Create a new private queue without touching active annotations')
    batch.add_argument('--dataset',type=Path,default=Path('local/training-staged/dataset.json'))
    batch.add_argument('--output',type=Path,default=Path('local/review-batches/current.json'))
    batch.add_argument('--audit-root',type=Path,default=Path('local/label-audit'))
    batch.add_argument('--limit',type=int,default=120)
    align=commands.add_parser('suggest-alignment',help='Suggest boundaries without approving labels')
    align.add_argument('--dataset',type=Path,default=Path('local/training-staged/dataset.json'))
    align.add_argument('--model',type=Path,default=Path('local/models/qwen3-aligner'))
    align.add_argument('--output',type=Path,default=Path('local/alignment'))
    audit=commands.add_parser('audit-labels',help='Generate private ASR suggestions for train/validation only')
    audit.add_argument('--dataset',type=Path,default=Path('local/training-staged/dataset.json'))
    audit.add_argument('--model',type=Path,required=True)
    audit.add_argument('--engine',choices=['gigaam','whisper','qwen','parakeet','russian-whisper','russian-whisper-beam5','t-one'],required=True)
    audit.add_argument('--output',type=Path,required=True)
    benchmark = commands.add_parser("benchmark", help="Evaluate a local ASR model on held-out recordings")
    benchmark.add_argument("--manifest", type=Path, default=Path("local/dataset/manifest.jsonl"))
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--output", type=Path, default=Path("local/benchmarks/baseline"))
    benchmark.add_argument("--limit", type=int, default=30)
    benchmark.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--compute-type", default="float16")
    export = commands.add_parser("export-corpus", help="Launch one VEGAS instance to inspect projects without saving them")
    export.add_argument("--vegas-exe", type=Path, required=True)
    export.add_argument("--projects-dir", type=Path, required=True)
    export.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    export.add_argument("--exclude", action="append", default=[], help="Exclude project filename prefixes")
    prepare = commands.add_parser("prepare-job", help="Create an expendable VEGAS input and a reviewed edit plan")
    for name in ("source", "review", "output"):
        prepare.add_argument("--"+name, type=Path, required=True)
    prepare.add_argument("--reference", type=Path)
    prepare.add_argument("--bleep", type=Path)
    prepare.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    apply = commands.add_parser("run-job", help="Apply the current reviewed plan using one new VEGAS instance")
    apply.add_argument("--vegas-exe", type=Path, required=True)
    apply.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.command == "manifest":
        result = build_manifest(args.exports, args.output)
        print(json.dumps({k:v for k,v in result.items() if k != "projects_detail"}, ensure_ascii=False, indent=2))
    elif args.command == "download-model":
        from .models import fetch
        print(fetch(args.name, args.registry, args.output))
    elif args.command == 'prepare-evaluation':
        from .evaluation import prepare
        print(json.dumps(prepare(args.manifest,args.output,args.split),ensure_ascii=False,indent=2))
    elif args.command == 'prepare-reviewed-evaluation':
        from .evaluation import from_reviewed
        print(json.dumps(from_reviewed(args.manifest,args.output,args.split),ensure_ascii=False,indent=2))
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        result=evaluate(args.set,args.engine,args.model,args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['failed_samples']:raise SystemExit(1)
    elif args.command == 'comparison-report':
        from .comparison_report import generate
        print(json.dumps(generate(args.runs,args.output),ensure_ascii=False,indent=2))
    elif args.command == 'stage-dataset':
        from .dataset import stage
        result=stage(args.manifest,args.output,args.workers)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['failures']:raise SystemExit(1)
    elif args.command == 'export-reviewed':
        from .dataset import export_reviewed
        print(json.dumps(export_reviewed(args.dataset,args.review,args.output),ensure_ascii=False,indent=2))
    elif args.command == 'review':
        from .review_server import serve
        serve(args.dataset,args.batch,args.saved,args.host,args.port)
    elif args.command == 'prepare-review-batch':
        from .review_batch import prepare_batch
        print(json.dumps(prepare_batch(args.dataset,args.output,args.audit_root,args.limit),ensure_ascii=False,indent=2))
    elif args.command == 'suggest-alignment':
        from .alignment import suggest
        result=suggest(args.dataset,args.model,args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['failures']:raise SystemExit(1)
    elif args.command == 'audit-labels':
        from .audit import predict
        result=predict(args.dataset,args.model,args.engine,args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['failures']:raise SystemExit(1)
    elif args.command == "benchmark":
        from .benchmark import run
        result = run(args.manifest, args.model, args.output, args.limit, args.split,
                     args.device, args.compute_type)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["failed_samples"]:
            raise SystemExit(1)
    elif args.command == "export-corpus":
        if not args.vegas_exe.is_file() or not args.projects_dir.is_dir():
            parser.error("VEGAS executable or projects directory does not exist")
        repo = args.repo.resolve()
        config = ET.Element("settings", projectsDirectory=str(args.projects_dir.resolve()),
                            corpusDirectory=str(repo/"local/corpus"))
        for prefix in args.exclude:
            ET.SubElement(config,"exclude").text=prefix
        (repo/"local").mkdir(exist_ok=True)
        ET.ElementTree(config).write(repo/"local/settings.xml", encoding="utf-8", xml_declaration=True)
        process = subprocess.Popen([str(args.vegas_exe), "/NOLOGO", "/SCRIPT", str(repo/"vegas/ExportCorpus.cs")])
        print(f"VEGAS exporter started as PID {process.pid}. Progress: {repo/'local/corpus/progress.log'}")
    elif args.command == "prepare-job":
        from .plan import prepare_job
        print(prepare_job(args.repo.resolve(), args.source, args.review, args.output, args.reference, args.bleep))
    elif args.command == "run-job":
        repo=args.repo.resolve()
        plan=ET.parse(repo/"local/jobs/current/edit-plan.xml").getroot()
        if not args.vegas_exe.is_file():
            parser.error("VEGAS executable does not exist")
        process=subprocess.Popen([str(args.vegas_exe),"/NOLOGO","/OPEN",plan.attrib["source"],"/SCRIPT",str(repo/"vegas/ApplyPlan.cs")])
        print(f"VEGAS builder started as PID {process.pid}. Result: {repo/'local/jobs/current/build-result.txt'}")
