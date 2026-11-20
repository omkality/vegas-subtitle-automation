from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import textwrap
import xml.etree.ElementTree as ET


WHISPER_SUBTITLE_CREDIT = re.compile(r"\bсубтитры\s+(?:сделал|создавал)\s+dimatorzok\b", re.IGNORECASE)


def check_caption_for_whisper_credit(text: str, start: float) -> None:
    if WHISPER_SUBTITLE_CREDIT.search(text):
        raise ValueError(f"В субтитре на {start:.3f} с обнаружена ложная подпись Whisper. Исправь текст перед монтажом.")


def validate_plan(captions: list[dict], bleeps: list[dict]) -> None:
    for item in captions:
        if item["speaker"] not in {"self", "friend"}:
            raise ValueError("Each caption needs a reviewed self/friend speaker label")
        start = float(item["start"])
        if not 0 <= start < float(item["end"]):
            raise ValueError("Invalid caption interval")
        if not item["text"].strip():
            raise ValueError("Empty caption")
        check_caption_for_whisper_credit(item["text"], start)
    for speaker in ("self", "friend"):
        rows = sorted((c for c in captions if c["speaker"] == speaker), key=lambda c:c["start"])
        if any(a["end"] > b["start"]+.001 for a,b in zip(rows,rows[1:])):
            raise ValueError("Overlapping captions for the same speaker")
    for item in bleeps:
        if not 0 <= float(item["start"]) < float(item["end"]):
            raise ValueError("Invalid bleep interval")


def prepare_job(repo: Path, source: Path, review: Path, output: Path,
                reference: Path | None, bleep: Path | None) -> Path:
    if not source.is_file() or not review.is_file():
        raise ValueError("Source project or reviewed plan does not exist")
    if output.exists() or output.resolve() == source.resolve():
        raise ValueError("Output must be a new path, distinct from the original")
    data = json.loads(review.read_text(encoding="utf-8-sig"))
    captions, bleeps = data["captions"], data.get("bleeps", [])
    validate_plan(captions, bleeps)
    if bleeps and (bleep is None or not bleep.is_file()):
        raise ValueError("An existing bleep sample is required")
    if reference is not None and not reference.is_file():
        raise ValueError("Reference project does not exist")
    job = repo/"local/jobs/current"
    job.mkdir(parents=True, exist_ok=True)
    # Only an expendable copy is ever opened by ApplyPlan.cs.
    private_input = job/"input.veg"
    shutil.copy2(source, private_input)
    shutil.copy2(repo/"configs/reference-style.xml", job/"reference-style.xml")
    root = ET.Element("edit", source=str(private_input.resolve()), output=str(output.resolve()),
                      referenceProject=str(reference.resolve()) if reference else "",
                      bleepSource=str(bleep.resolve()) if bleep else "", hotbar="true" if reference else "false")
    fps = 60000/1001
    for c in sorted(captions,key=lambda c:c["start"]):
        text = "\n".join(textwrap.wrap(c["text"],width=28 if c["speaker"] == "friend" else 22,
                                     break_long_words=False, break_on_hyphens=False))
        if c["speaker"] == "friend":
            text = "* "+text+" *"
        ET.SubElement(root,"caption",start=f"{round(c['start']*fps)/fps:.8f}",
                      end=f"{round(c['end']*fps)/fps:.8f}",speaker=c["speaker"]).text=text
    for b in bleeps:
        ET.SubElement(root,"bleep",start=f"{round(b['start']*fps)/fps:.8f}",end=f"{round(b['end']*fps)/fps:.8f}")
    ET.indent(root)
    ET.ElementTree(root).write(job/"edit-plan.xml",encoding="utf-8",xml_declaration=True)
    return job/"edit-plan.xml"
