"""LanceDB vector store (default, local)."""
from __future__ import annotations

import re
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

_HEX = re.compile(r"^[0-9a-f]{8,64}$")
TABLE = "chunks"


class LanceStore:
    def __init__(self, path: Path, dim: int):
        path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.db = lancedb.connect(str(path))
        self._t = None

    def _schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("file_key", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.dim)),
        ])

    @property
    def table(self):
        if self._t is None:
            names = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
            names = getattr(names, "tables", names)
            if TABLE in list(names):
                self._t = self.db.open_table(TABLE)
            else:
                self._t = self.db.create_table(TABLE, schema=self._schema())
        return self._t

    def drop(self) -> None:
        names = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
        if TABLE in list(getattr(names, "tables", names)):
            self.db.drop_table(TABLE)
        self._t = None

    def delete_file(self, file_key: str) -> None:
        if not _HEX.match(file_key):
            raise ValueError("bad file key")
        self.table.delete(f"file_key = '{file_key}'")

    def add(self, ids: list[str], file_key: str, vectors: np.ndarray) -> None:
        if not ids:
            return
        rows = [{"id": i, "file_key": file_key, "vector": v.astype(np.float32).tolist()} for i, v in zip(ids, vectors)]
        self.table.add(rows)

    def search(self, vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        n = self.count()
        if n == 0:
            return []
        res = self.table.search(vec.astype(np.float32)).metric("cosine").limit(min(limit, n)).to_list()
        return [(r["id"], 1.0 - float(r["_distance"])) for r in res]

    def count(self) -> int:
        return int(self.table.count_rows())

    def has_id(self, chunk_id: str) -> bool:
        if not _HEX.match(chunk_id):
            return False
        return int(self.table.count_rows(f"id = '{chunk_id}'")) > 0

    def delete_ids(self, ids: list[str]) -> None:
        for i in ids:
            if _HEX.match(i):
                self.table.delete(f"id = '{i}'")
