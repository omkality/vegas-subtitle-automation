"""Turn native VEGAS XML exports into auditable, conservative ASR candidates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath
import xml.etree.ElementTree as ET


def local_path(value: str) -> Path:
    """Accept Windows source paths when the tooling runs under WSL."""
    import os
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        root = os.environ.get("VEGAS_MEDIA_ROOT_"+value[0].upper(), "/mnt/"+value[0].lower())
        return Path(root) / value[3:].replace("\\", "/")
    return Path(value)


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def unwrap_speaker_stars(value: str) -> str:
    value = normalize_text(value)
    if value.startswith("*") and value.endswith("*"):
        return value[1:-1].strip()
    return value


def merge_spans(spans: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for span in sorted(spans, key=lambda s: s["timeline_start"]):
        if merged:
            old = merged[-1]
            contiguous = abs(old["timeline_end"] - span["timeline_start"]) < .002
            source_contiguous = abs(old["source_end"] - span["source_start"]) < .002
            same = all(old[k] == span[k] for k in ("path", "stream", "rate"))
            if contiguous and source_contiguous and same:
                old["timeline_end"] = span["timeline_end"]
                old["source_end"] = span["source_end"]
                continue
        merged.append(dict(span))
    return merged


def parse_project(path: Path) -> tuple[list[dict], dict]:
    root = ET.parse(path).getroot()
    tracks = root.findall("track")
    video_paths = {
        e.get("path") for t in tracks if t.get("type") == "Video"
        for e in t.findall("event") if e.find("generator") is None
    }
    audio_tracks = [t for t in tracks if t.get("type") == "Audio"]
    captions: list[dict] = []
    for t in tracks:
        for e in t.findall("event"):
            gen = e.find("generator")
            if gen is None or gen.find("text") is None:
                continue
            a, b = float(e.get("start", "0")), float(e.get("end", "0"))
            text = unwrap_speaker_stars(gen.findtext("text", ""))
            speaker = gen.get("speaker", "unknown")
            cyrillic = len(re.findall(r"[а-яё]", text, re.I))
            latin = len(re.findall(r"[a-z]", text, re.I))
            label_language = "en" if latin > max(3, cyrillic*2) else "ru"
            choices: list[tuple[float, list[dict]]] = []
            for audio_track in audio_tracks:
                spans = []
                for event in audio_track.findall("event"):
                    source = event.get("path", "")
                    if source not in video_paths:
                        continue  # Avoid music and sound-effect tracks.
                    lo = max(a, float(event.get("start", "0")))
                    hi = min(b, float(event.get("end", "0")))
                    if hi <= lo:
                        continue
                    rate = float(event.get("rate", "1"))
                    offset = float(event.get("offset", "0"))
                    source_start = offset + (lo - float(event.get("start", "0"))) * rate
                    # Deliberately ignore mute/gain. We need ORIGINAL uncensored audio.
                    spans.append({"path": source, "stream": int(event.get("stream", "0")),
                                  "timeline_start": lo, "timeline_end": hi,
                                  "source_start": source_start,
                                  "source_end": source_start + (hi-lo)*rate, "rate": rate})
                if spans:
                    choices.append((sum(s["timeline_end"]-s["timeline_start"] for s in spans), spans))
            spans = merge_spans(max(choices, key=lambda x: x[0])[1]) if choices else []
            issues = []
            if not text:
                issues.append("empty_text")
            if "*" in text:
                issues.append("masked_profanity_label")
            if re.search(r"\d[а-яё]|[а-яё]\d", text, re.I):
                issues.append("obfuscated_label_needs_review")
            if label_language != "ru":
                issues.append("non_russian_label_needs_review")
            if speaker == "unknown":
                issues.append("unknown_speaker")
            if not .35 <= b-a <= 20:
                issues.append("duration_out_of_range")
            if len(spans) != 1 or (spans and (spans[0]["timeline_start"] > a+.03 or spans[-1]["timeline_end"] < b-.03)):
                issues.append("audio_mapping_needs_review")
            if any(abs(s["rate"]-1) > .00001 for s in spans):
                issues.append("speed_changed")
            if any(not local_path(s["path"]).is_file() for s in spans):
                issues.append("missing_source_media")
            if spans and any(PureWindowsPath(s["path"]).suffix.lower() == ".veg" for s in spans):
                issues.append("nested_project_source")
            ident = hashlib.sha256(f"{path.stem}:{t.get('index')}:{e.get('index')}".encode()).hexdigest()[:16]
            captions.append({"id": ident, "project": path.stem, "track": int(t.get("index", "0")),
                             "event": int(e.get("index", "0")), "start": a, "end": b,
                             "text": text, "speaker": speaker, "label_language": label_language, "spans": spans,
                             "issues": issues, "label_source": "human_edited_vegas",
                             "alignment_reviewed": False})
    # A clean single-speaker label must not cover a second person's caption.
    for c in captions:
        for other in captions:
            if c is other or c["speaker"] == other["speaker"]:
                continue
            if min(c["end"], other["end"]) - max(c["start"], other["start"]) > .08:
                if "overlapping_speakers" not in c["issues"]:
                    c["issues"].append("overlapping_speakers")
    return captions, {"project": path.stem, "captions": len(captions),
                      "timeline_seconds": float(root.get("duration", "0"))}


def assign_groups(rows: list[dict]) -> None:
    """Connected projects sharing a source recording cannot leak across splits."""
    parent = {r["project"]: r["project"] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for row in rows:
        for span in row["spans"]:
            source = span["path"].lower()
            if source in owners:
                a, b = find(row["project"]), find(owners[source])
                parent[max(a,b)] = min(a,b)
            owners[source] = row["project"]
    groups = sorted({find(project) for project in parent})
    ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    test_count = max(1, round(len(groups)*.15)) if len(groups) >= 3 else 0
    val_count = max(1, round(len(groups)*.1)) if len(groups) >= 4 else 0
    split = {g: "test" if i < test_count else "validation" if i < test_count+val_count else "train"
             for i, g in enumerate(ordered)}
    for row in rows:
        row["group"] = find(row["project"])
        row["split"] = split[row["group"]]


def build_manifest(exports: Path, destination: Path) -> dict:
    rows, projects, errors = [], [], []
    for path in sorted(exports.glob("*.xml")):
        try:
            samples, info = parse_project(path)
            rows.extend(samples)
            projects.append(info)
        except (ET.ParseError, ValueError, KeyError) as exc:
            errors.append({"project": path.name, "error": str(exc)})
    seen = set()
    for row in rows:
        signature = (row["text"], tuple((s["path"].lower(), s["stream"], round(s["source_start"], 2),
                                        round(s["source_end"], 2)) for s in row["spans"]))
        if row["spans"] and signature in seen:
            row["issues"].append("duplicate_source_and_label")
        seen.add(signature)
    assign_groups(rows)
    for row in rows:
        row["candidate"] = not row["issues"]
        row["training_approved"] = False  # Never silently treat editorial timings as alignment.
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/"manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    counts = Counter(issue for row in rows for issue in row["issues"])
    candidates = [r for r in rows if r["candidate"]]
    report = {"projects": len(projects), "captions": len(rows),
              "timeline_minutes": sum(p["timeline_seconds"] for p in projects)/60,
              "speakers": dict(Counter(r["speaker"] for r in rows)),
              "candidate_count": len(candidates),
              "candidate_minutes": sum(r["end"]-r["start"] for r in candidates)/60,
              "candidate_speakers": dict(Counter(r["speaker"] for r in candidates)),
              "recording_groups": len({r["group"] for r in rows}),
              "candidate_splits": dict(Counter(r["split"] for r in candidates)),
              "issues": dict(counts), "projects_detail": projects, "errors": errors,
              "training_started": False}
    (destination/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]
