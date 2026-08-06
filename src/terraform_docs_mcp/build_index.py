"""Build the packaged search index.

Run via ``make index``. Not part of the installed tool: it imports the
``build`` dependency group (langchain-text-splitters, huggingface-hub, pyyaml)
and writes into the package's ``_data`` directory, which is then shipped by
``uv_build``.

``uv_build`` runs no build hooks, so this cannot be triggered from inside
``uv build``. The Makefile enforces the ordering instead.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

from . import PROVIDERS
from .corpus import chunk_document, iter_documents
from .embed import MODEL_ID, SentenceTransformerEmbedder, download_model
from .index import INDEX_FILENAME, SCHEMA, VECTORS_FILENAME, quantize


def _log(message: str) -> None:
    print(f"[build-index] {message}", flush=True)


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _copy_docs(project_root: Path, data_dir: Path) -> None:
    """Copy provider markdown and licenses into the packaged data directory.

    Copied rather than symlinked: the submodules do not exist inside a wheel.
    """
    for provider, repo in PROVIDERS.items():
        src = project_root / repo / "website" / "docs"
        dst = data_dir / "docs" / provider
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        license_src = project_root / repo / "LICENSE"
        license_dst = data_dir / "licenses" / provider
        license_dst.mkdir(parents=True, exist_ok=True)
        if license_src.exists():
            shutil.copyfile(license_src, license_dst / "LICENSE")
        else:  # pragma: no cover - MPL redistribution requires this file
            raise FileNotFoundError(
                f"{license_src} is missing; it must ship with the redistributed docs."
            )


def build(project_root: Path, data_dir: Path) -> dict[str, int]:
    data_dir.mkdir(parents=True, exist_ok=True)

    _log("fetching embedding model")
    model_dir = download_model(data_dir / "model")

    _log("loading and chunking documents")
    t0 = time.time()
    documents, chunks = [], []
    for provider, repo in PROVIDERS.items():
        for doc in iter_documents(project_root / repo, provider):
            documents.append(doc)
            chunks.extend(chunk_document(doc))
    _log(f"{len(documents)} documents, {len(chunks)} chunks ({time.time() - t0:.1f}s)")

    _log("embedding")
    t0 = time.time()
    embedder = SentenceTransformerEmbedder(model_dir)
    # encode() handles batching, length-sorting to minimise padding, and the
    # progress bar itself; there is nothing to hand-roll here.
    vectors = embedder.embed_documents([c.text for c in chunks], progress=True)
    _log(f"embedded {len(chunks)} chunks, dim {vectors.shape[1]} ({time.time() - t0:.1f}s)")

    _log("writing vectors")
    quantized = quantize(vectors)
    np.save(data_dir / VECTORS_FILENAME, quantized)

    _log("writing sqlite index")
    _write_db(project_root, data_dir, documents, chunks, vectors.shape[1])

    _log("copying documentation and licenses")
    _copy_docs(project_root, data_dir)

    return {"documents": len(documents), "chunks": len(chunks)}


def _write_db(project_root: Path, data_dir: Path, documents, chunks, dim: int) -> None:
    db_path = data_dir / INDEX_FILENAME
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _ = conn.executemany(
            "INSERT INTO documents (doc_id, provider, kind, name, title, subcategory,"
            " description, rel_path) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    d.doc_id,
                    d.provider,
                    d.kind,
                    d.name,
                    d.title,
                    d.subcategory,
                    d.description,
                    d.rel_path,
                )
                for d in documents
            ],
        )

        # Chunk ids are assigned densely from 1 so that vectors row i maps to
        # chunk id i + 1 with no lookup table.
        conn.executemany(
            "INSERT INTO chunks (id, doc_id, ordinal, heading_path, snippet) VALUES (?,?,?,?,?)",
            [
                (i, c.doc_id, c.ordinal, c.heading_path, c.snippet)
                for i, c in enumerate(chunks, start=1)
            ],
        )
        conn.executemany(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?,?)",
            [(i, c.text) for i, c in enumerate(chunks, start=1)],
        )

        meta = {
            "model_id": MODEL_ID,
            "dim": str(dim),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunk_count": str(len(chunks)),
            "document_count": str(len(documents)),
        }
        for provider, repo in PROVIDERS.items():
            meta[f"{provider}_commit"] = _git_sha(project_root / repo)
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?,?)", sorted(meta.items())
        )

        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the packaged search index.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the provider submodules.",
    )
    args = parser.parse_args(argv)

    data_dir = Path(__file__).resolve().parent / "_data"
    stats = build(args.project_root, data_dir)

    total = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    _log(f"done: {stats['documents']} documents, {stats['chunks']} chunks")
    _log(f"_data size: {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
