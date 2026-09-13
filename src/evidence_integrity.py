"""Integrity checks for the frozen evaluation set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evaluation_manifest(manifest_path: str | Path) -> Dict[str, str]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("frozen_after_label_audit"):
        raise ValueError("Evaluation manifest is not a recognized frozen manifest")
    verified: Dict[str, str] = {}
    for relative_name, expected in manifest.get("files", {}).items():
        target = manifest_file.parent / relative_name
        actual = sha256_file(target)
        if actual.lower() != str(expected).lower():
            raise ValueError(
                f"Frozen evaluation file changed: {target}; expected {expected}, got {actual}"
            )
        verified[relative_name] = actual
    if not verified:
        raise ValueError("Evaluation manifest contains no files")
    return verified
