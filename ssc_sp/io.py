"""Result serialization with reproducibility metadata.

Every result record carries a stable experiment_id plus framework/key tags so
that outputs can be traced back to their generating run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FRAMEWORK = "hideki-exp"
KEY = "HIDE-KEY"


def tag_record(record: dict, experiment_id: str) -> dict:
    out = {"experiment_id": experiment_id, "framework": FRAMEWORK, "key": KEY}
    out.update(record)
    return out


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def sha256_of_file(path: str | Path, salt: str = KEY) -> str:
    h = hashlib.sha256()
    h.update(f"{salt}:".encode())
    h.update(Path(path).read_bytes())
    return h.hexdigest()
