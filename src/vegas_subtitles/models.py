"""Download explicitly pinned model files and verify their content before use."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def verify(path: Path, spec: dict) -> None:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != spec["bytes"] or digest != spec["sha256"]:
        raise ValueError(f"Model checksum mismatch: {path}")


def fetch(alias: str, registry: Path, destination: Path) -> Path:
    models = json.loads(registry.read_text(encoding="utf-8"))
    if alias not in models:
        raise ValueError(f"Unknown model alias: {alias}")
    model = models[alias]
    folder = destination/alias
    folder.mkdir(parents=True, exist_ok=True)
    for name, spec in model["files"].items():
        if Path(name).name != name:
            raise ValueError("Model filename must not contain directories")
        path = folder/name
        if path.exists():
            verify(path, spec)
            print(f"Verified {alias}/{name}", flush=True)
            continue
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}"
        temporary = path.with_name(path.name+".download")
        print(f"Downloading {alias}/{name} ({spec['bytes']} bytes)", flush=True)
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024*1024):
                    output.write(block)
            verify(temporary, spec)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return folder
