"""Check packaged file sizes and SHA-256 hashes without third-party libraries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    seen = set()
    for record in records:
        relative = record["path"]
        path = (root / relative).resolve()
        if relative in seen or not path.is_relative_to(root):
            failures.append({"path": relative, "reason": "duplicate or invalid path"})
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            failures.append({"path": relative, "reason": "size or SHA-256 mismatch"})
    print(json.dumps({"checked_files": len(records), "failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
