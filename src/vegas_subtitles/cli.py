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
    download.add_argument("name", choices=["large-v3", "turbo"])
    download.add_argument("--registry", type=Path, default=Path("configs/models.lock.json"))
    download.add_argument("--output", type=Path, default=Path("local/models"))
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
