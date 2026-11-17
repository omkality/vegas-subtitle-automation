from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

from .audio import extract_row, read_wav
from .corpus import load_rows
from .metrics import word_errors, words


def run(manifest: Path, model: str, output: Path, limit: int = 30,
        split: str = "validation", device: str = "cuda", compute_type: str = "float16") -> dict:
    from faster_whisper import WhisperModel
    rows = [r for r in load_rows(manifest) if r["candidate"] and r["split"] == split
            and r["end"]-r["start"] >= 1.1 and len(words(r["text"])) >= 3]
    # Mix projects, rather than selecting only the first alphabetic project.
    random.Random(17).shuffle(rows)
    selected = rows[:limit]
    if not selected:
        raise ValueError("No suitable benchmark samples")
    output.mkdir(parents=True, exist_ok=True)
    def sha256(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    model_path = Path(model)
    fingerprints = {p.name: sha256(p) for p in sorted(model_path.iterdir()) if p.is_file()} if model_path.is_dir() else {}
    asr = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=4)
    results, failures = [], []
    started = time.monotonic()
    for row in selected:
        try:
            path = output/"audio"/(row["id"]+".wav")
            extract_row(row, path)
            segments, _ = asr.transcribe(read_wav(path), language="ru", beam_size=5,
                                         word_timestamps=True, vad_filter=True,
                                         condition_on_previous_text=False)
            segments = list(segments)
            hypothesis = " ".join(s.text.strip() for s in segments)
            item = {"id": row["id"], "project": row["project"], "speaker": row["speaker"],
                    "reference": row["text"], "hypothesis": hypothesis,
                    **word_errors(row["text"], hypothesis)}
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append({"id": row["id"], "error": str(exc)})
        (output/"samples.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    count = sum(r["reference_words"] for r in results)
    report = {"model": model, "device": device, "compute_type": compute_type, "split": split,
              "manifest_sha256": sha256(manifest), "model_sha256": fingerprints,
              "sample_ids": [r["id"] for r in selected], "selection_seed": 17,
              "versions": {name: version(name) for name in ("faster-whisper", "ctranslate2", "numpy", "imageio-ffmpeg")},
              "samples": len(results), "failed_samples": failures, "reference_words": count,
              "word_errors": sum(r["errors"] for r in results),
              "wer": sum(r["errors"] for r in results)/count if count else None,
              "elapsed_seconds": time.monotonic()-started,
              "elapsed_scope": "Audio extraction and inference, excluding model load and fingerprinting",
              "projects": dict(Counter(r["project"] for r in results)),
              "limitation": "Editorial caption timings were not manually realigned. WER includes timing and label mismatch."}
    (output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
