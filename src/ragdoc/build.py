"""Resumable, batched, hash-cached index build. Writes only inside the collection directory."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import __version__
from .chunkers import chunk_document
from .collection import Collection, now_iso
from .config import Limits, collections_dir
from .embedders import get_embedder
from .errors import RagDocError
from .parsers import SUPPORTED_EXTS, ParserUnavailable, parse_file
from .security import Deadline, is_within, safe_collection_name, safe_path
from .stores import LanceStore
from .textutil import sha1_hex, sha256_file

SKIP_DIRS = {".git", ".ragdoc", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
BATCH = 64


def discover_files(src: Path, base: Path | None = None) -> tuple[list[tuple[Path, str]], list[dict]]:
    """Return ([(abs_path, rel_path)], skipped). Never follows symlinks out of the base directory."""
    from .config import base_dir

    base = base or base_dir()
    files: list[tuple[Path, str]] = []
    skipped: list[dict] = []
    if src.is_file():
        cands = [(src, src.name)]
    else:
        cands = []
        for root, dirs, names in os.walk(src, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for n in sorted(names):
                p = Path(root) / n
                cands.append((p, p.relative_to(src).as_posix()))
    for p, rel in cands:
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if p.is_symlink() and not is_within(p, base):
            skipped.append({"file": rel, "reason": "symlink points outside base directory"})
            continue
        try:
            size = p.stat().st_size
        except OSError:
            skipped.append({"file": rel, "reason": "unreadable"})
            continue
        if size > Limits.max_file_bytes():
            skipped.append({"file": rel, "reason": f"larger than {Limits.max_file_bytes() // (1024 * 1024)} MB limit"})
            continue
        files.append((p, rel))
        if len(files) > Limits.max_files():
            raise RagDocError(f"More than {Limits.max_files()} files.", hint="Index a sub-folder or raise RAGDOC_MAX_FILES.", code="TOO_MANY_FILES")
    return files, skipped


def backend_for(plan: dict, ext: str, rel: str | None = None) -> str:
    fb = plan.get("file_backends") or {}
    if rel and rel in fb:
        return fb[rel]
    p = plan.get("parsers", {})
    return p.get(ext, p.get("*", "fast"))


def file_sig(plan: dict, ext: str, rel: str | None = None) -> str:
    return sha1_hex(json.dumps({
        "backend": backend_for(plan, ext, rel), "chunking": plan.get("chunking"), "embedder": plan["embedder"]["name"],
        "fault": (plan.get("_faults") or {}).get("parse"),
    }, sort_keys=True, default=str), 12)


def build_collection(plan: dict, *, resume: bool = True, progress=None, deadline: Deadline | None = None,
                     collection_dir: Path | None = None, force_files: set[str] | None = None) -> dict:
    t0 = time.time()
    deadline = deadline or Deadline()
    name = safe_collection_name(plan["collection"])
    coll = Collection(collection_dir or (collections_dir() / name))
    coll.ensure_dirs()
    src = safe_path(plan["source_path"])
    files, skipped = discover_files(src)
    warnings: list[str] = []

    try:
        embedder = get_embedder(plan["embedder"]["name"])
    except RagDocError as e:
        warnings.append(f"{e.message} Falling back to the offline hashing embedder.")
        plan["embedder"] = {"name": "hashing-384", "dim": 384, "fallback_from": plan["embedder"]["name"]}
        embedder = get_embedder("hashing-384")

    manifest = coll.load_manifest() if coll.manifest_path.exists() else {
        "schema": 1, "name": name, "created": now_iso(), "files": {}, "consent": {}, "stats": {}}
    prev_emb = manifest.get("plan", {}).get("embedder", {}).get("name")
    if prev_emb and prev_emb != plan["embedder"]["name"]:
        resume = False  # vectors from another embedder are not comparable
        LanceStore(coll.lance_dir, embedder.dim).drop()
        manifest["files"] = {}
    manifest.update({
        "name": name, "source_path": str(src), "plan": plan,
        "consent": {**manifest.get("consent", {}), **plan.get("consent", {})},
        "versions": {"ragdoc": __version__, "embedder": plan["embedder"]["name"], **_lib_versions()},
    })
    store = LanceStore(coll.lance_dir, embedder.dim)
    rows_by_file: dict[str, list[dict]] = {}
    for r in coll.read_rows():
        rows_by_file.setdefault(r["file"], []).append(r)

    known = manifest.setdefault("files", {})
    current = {rel for _, rel in files}
    removed = [rel for rel in list(known) if rel not in current]
    for rel in removed:
        store.delete_file(sha1_hex(rel, 12))
        known.pop(rel, None)
        rows_by_file.pop(rel, None)

    done = reused = 0
    last_ckpt = time.time()

    def checkpoint() -> None:
        coll.write_rows([r for rel in sorted(rows_by_file) for r in rows_by_file[rel]])
        manifest["stats"] = {"chunks": sum(len(v) for v in rows_by_file.values()), "files": len(known)}
        coll.save_manifest(manifest)

    for idx, (path, rel) in enumerate(files):
        deadline.check()
        if progress:
            progress(idx, len(files), f"{rel}")
        ext = path.suffix.lower()
        h = sha256_file(path)
        sig = file_sig(plan, ext, rel)
        entry = known.get(rel)
        if (resume and rel not in (force_files or ()) and entry and entry.get("hash") == h and entry.get("sig") == sig and entry.get("status") == "done"
                and (entry.get("n_chunks", 0) == 0 or rel in rows_by_file)):
            reused += 1
            continue
        fkey = sha1_hex(rel, 12)
        store.delete_file(fkey)
        rows_by_file.pop(rel, None)
        backend = backend_for(plan, ext, rel)
        try:
            doc = parse_file(path, rel, h, backend, manifest=manifest, faults=plan.get("_faults"))
        except ParserUnavailable as e:
            warnings.append(f"{rel}: {e} Using the fast parser instead.")
            backend = "fast"
            doc = parse_file(path, rel, h, "fast", manifest=manifest, faults=plan.get("_faults"))
        except RagDocError as e:
            skipped.append({"file": rel, "reason": e.message})
            known.pop(rel, None)
            continue
        except Exception as e:  # noqa: BLE001
            skipped.append({"file": rel, "reason": f"parse error: {type(e).__name__}: {e}"})
            known.pop(rel, None)
            continue
        pname = f"parsed/{sha1_hex(rel, 10)}-{h[:12]}-{doc.parser}.json"
        (coll.dir / pname).write_text(json.dumps(doc.to_json()), encoding="utf-8")
        chunks = chunk_document(doc, plan.get("chunking"))
        vecs = []
        for i in range(0, len(chunks), BATCH):
            vecs.append(embedder.embed_docs([c.embed_text for c in chunks[i : i + BATCH]]))
        if chunks:
            import numpy as np

            store.add([c.chunk_id for c in chunks], fkey, np.vstack(vecs))
        rows_by_file[rel] = [c.to_dict() for c in chunks]
        known[rel] = {"hash": h, "size": path.stat().st_size, "mtime": path.stat().st_mtime, "sig": sig, "backend": doc.parser,
                      "n_chunks": len(chunks), "parsed": pname, "status": "done",
                      "scanned_pages": doc.meta.get("scanned_pages", [])}
        done += 1
        if time.time() - last_ckpt > 20 or done % 25 == 0:
            checkpoint()
            last_ckpt = time.time()
    checkpoint()
    if progress:
        progress(len(files), len(files), "done")
    return {
        "collection": name, "files_total": len(files), "files_indexed": done, "files_unchanged": reused,
        "files_removed": len(removed), "skipped": skipped[:20], "skipped_count": len(skipped),
        "chunks": manifest["stats"]["chunks"], "warnings": warnings, "seconds": round(time.time() - t0, 2),
        "collection_dir": str(coll.dir),
    }


def _lib_versions() -> dict:
    out = {}
    for lib in ("lancedb", "pypdf", "pyarrow"):
        try:
            import importlib.metadata as md

            out[lib] = md.version(lib)
        except Exception:  # noqa: BLE001
            pass
    return out
