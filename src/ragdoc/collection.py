"""Collection directory layout: manifest.json, parsed/, chunks.parquet, index/, eval/, logs/."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import collections_dir
from .errors import RagDocError
from .security import safe_collection_name

_ROW_CACHE: dict[str, tuple[tuple, list[dict]]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Collection:
    def __init__(self, cdir: Path):
        self.dir = Path(cdir)

    # layout
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def parsed_dir(self) -> Path:
        return self.dir / "parsed"

    @property
    def rows_path(self) -> Path:
        return self.dir / "chunks.parquet"

    @property
    def lance_dir(self) -> Path:
        return self.dir / "index" / "lance"

    @property
    def eval_dir(self) -> Path:
        return self.dir / "eval"

    @property
    def logs_dir(self) -> Path:
        return self.dir / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.parsed_dir, self.lance_dir, self.eval_dir / "runs", self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def open(name: str) -> Collection:
        safe_collection_name(name)
        c = Collection(collections_dir() / name)
        if not c.manifest_path.exists():
            raise RagDocError(f"Collection '{name}' not found.", hint="Run plan_rag then build_index first, or check ragdoc://collections.", code="NO_COLLECTION")
        return c

    # manifest
    def load_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text("utf-8"))

    def save_manifest(self, m: dict) -> None:
        m["updated"] = now_iso()
        atomic_write_json(self.manifest_path, m)

    # rows
    def read_rows(self) -> list[dict]:
        if not self.rows_path.exists():
            return []
        st = self.rows_path.stat()
        key = (st.st_mtime_ns, st.st_size)
        hit = _ROW_CACHE.get(str(self.rows_path))
        if hit and hit[0] == key:
            return hit[1]
        rows = pq.read_table(self.rows_path).to_pylist()
        _ROW_CACHE[str(self.rows_path)] = (key, rows)
        return rows

    def write_rows(self, rows: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("chunk_id", pa.string()), ("file", pa.string()), ("file_hash", pa.string()), ("page", pa.int32()),
            ("heading_path", pa.string()), ("ctype", pa.string()), ("text", pa.string()), ("embed_text", pa.string()),
            ("tokens", pa.int32()), ("seq", pa.int32()),
        ])
        tmp = self.rows_path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp)
        os.replace(tmp, self.rows_path)
        _ROW_CACHE.pop(str(self.rows_path), None)

    def load_parsed(self, rel: str):
        from .parsers import ParsedDoc

        m = self.load_manifest()
        entry = m.get("files", {}).get(rel)
        if not entry or not entry.get("parsed"):
            return None
        p = self.dir / entry["parsed"]
        return ParsedDoc.from_json(json.loads(p.read_text("utf-8"))) if p.exists() else None


def list_collections() -> list[dict]:
    root = collections_dir()
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            mp = d / "manifest.json"
            if d.is_dir() and mp.exists():
                try:
                    m = json.loads(mp.read_text("utf-8"))
                    out.append({"name": m.get("name", d.name), "files": len(m.get("files", {})), "chunks": m.get("stats", {}).get("chunks"), "updated": m.get("updated")})
                except Exception:  # noqa: BLE001
                    out.append({"name": d.name, "error": "unreadable manifest"})
    return out
