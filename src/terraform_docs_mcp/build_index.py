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
import time
from typing import Sequence

# Disabled alongside the vector-embed build below -- see build(). `_write_db`
# is left defined but uncalled, so INDEX_FILENAME/SCHEMA (which it still
# references) stay imported; VECTORS_FILENAME/quantize do not, since those are
# only used by the disabled vector-writing block.
# import numpy as np

from ._config import DATA_DIR, DOCUMENTS_INDEX_FILENAME, PROJECT_ROOT
from .manifest import (
    current_inputs as _current_inputs,
    read as _read_manifest,
    staleness as _staleness,
    summary as _summary,
    write as _write_manifest,
)

from .corpus import Document, iter_documents, PROVIDERS  # chunk_document: see below
from .db import Db

# from .embed import SentenceTransformerEmbedder, download_model
from .index import INDEX_FILENAME, SCHEMA  # VECTORS_FILENAME, quantize: disabled


def _log(message: str) -> None:
    print(f"[build-index] {message}", flush=True)


def _copy_docs() -> None:
    """Copy provider markdown and licenses into the packaged data directory.

    Copied rather than symlinked: the submodules do not exist inside a wheel.
    """
    for config in PROVIDERS.values():
        src = PROJECT_ROOT / config.source_docs_dir
        dst = DATA_DIR / config.destination_docs_dir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        license_src = PROJECT_ROOT / config.source_license
        license_dst = DATA_DIR / config.destination_docs_dir
        license_dst.mkdir(parents=True, exist_ok=True)
        if license_src.exists():
            shutil.copyfile(license_src, license_dst / "LICENSE")
        else:  # pragma: no cover - MPL redistribution requires this file
            raise FileNotFoundError(
                f"{license_src} is missing; it must ship with the redistributed docs."
            )


def build(force: bool = False) -> dict[str, object] | None:
    """Regenerate ``_data`` if any input changed. ``None`` if nothing did.

    Everything is rebuilt or nothing is: the database is unlinked and the docs
    tree is replaced wholesale, so there is no partial state to reason about.
    The one exception is the embedding model, which ``download_model`` skips
    when its ``MODEL_REPO.txt`` marker already names the right repo -- weights
    are content-identified by that marker, and re-pulling 48 MB because a
    Python file changed would be pure waste.
    """
    reason = _staleness(DATA_DIR, PROVIDERS)
    if reason is None and not force:
        recorded = _read_manifest(DATA_DIR)
        _log(f"up to date ({recorded.get('fingerprint', '')[:12]}); nothing to do")
        for line in _summary(recorded):
            _log(line)
        return None
    _log(f"rebuilding: {reason or 'forced'}")

    # Captured before the build, not after: a source edit made during these ~96
    # seconds did not go into this index, and recording it would mark a stale
    # index fresh. Recording the older hash merely triggers one more rebuild.
    inputs = _current_inputs(PROVIDERS)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- vector-embed build: disabled -------------------------------------
    # Focus is on the document-level trigram search path (db.py) for now.
    # Re-enable this block, the "writing sqlite index" block below it, and the
    # commented imports above, together -- they are one unit.
    #
    # _log("fetching embedding model")
    # model_dir = download_model(DATA_DIR / MODEL_DIRNAME)

    _log("loading documents")
    t0 = time.time()
    documents = []
    for provider in PROVIDERS.values():
        for doc in iter_documents(provider):
            documents.append(doc)
            # chunks.extend(chunk_document(doc))  # embedding-only; see above
    _log(f"{len(documents)} documents ({time.time() - t0:.1f}s)")

    # _log("embedding")
    # t0 = time.time()
    # embedder = SentenceTransformerEmbedder(model_dir)
    # # encode() handles batching, length-sorting to minimise padding, and the
    # # progress bar itself; there is nothing to hand-roll here.
    # vectors = embedder.embed_documents([c.text for c in chunks], progress=True)
    # _log(
    #     f"embedded {len(chunks)} chunks, dim {vectors.shape[1]} ({time.time() - t0:.1f}s)"
    # )
    #
    # _log("writing vectors")
    # quantized = quantize(vectors)
    # np.save(DATA_DIR / VECTORS_FILENAME, quantized)
    #
    # _log("writing sqlite index")
    # _write_db(documents, chunks)
    # -- end vector-embed build --------------------------------------------

    # The document-level trigram search path. A separate artifact from
    # index.sqlite3 -- not wired into the (currently disabled) hybrid pipeline.
    _log("writing document-level trigram index")
    _write_documents_db(documents)

    _log("copying documentation and licenses")
    _copy_docs()

    # Last, deliberately: the manifest's presence is what says the build
    # finished. A crash above leaves none, and the next run starts over.
    # `dim`/`chunk_count` are omitted while the vector-embed build is
    # disabled -- there is no vectors array or chunk list to report on.
    document = _write_manifest(
        DATA_DIR,
        inputs,
        {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "document_count": len(documents),
        },
    )
    _log(f"wrote manifest ({document['fingerprint'][:12]})")
    return document


def _write_db(documents: Sequence[Document], chunks) -> None:
    db_path = DATA_DIR / INDEX_FILENAME
    db_path.unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _ = conn.executemany(
            "INSERT INTO documents (doc_id, provider, kind, title, subcategory,"
            " description, rel_path) VALUES (?,?,?,?,?,?,?)",
            [
                (
                    d.doc_id,
                    d.provider.value,
                    d.kind,
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

        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


def _write_documents_db(documents: Sequence[Document]) -> None:
    db_path = DATA_DIR / DOCUMENTS_INDEX_FILENAME
    db_path.unlink(missing_ok=True)
    db = Db(db_path, readonly=False)
    try:
        db.create_schema()
        db.add_documents(documents)
        db.rebuild_fts()
        db.commit()
        db.vacuum()
    finally:
        db.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m terraform_docs_mcp.build_index",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when nothing changed.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Report whether a rebuild is needed and exit 1 if so; build nothing.",
    )
    args = parser.parse_args(argv)

    if args.check:
        reason = _staleness(DATA_DIR, PROVIDERS)
        if reason is None:
            _log("up to date")
            return 0
        _log(f"stale: {reason}")
        return 1

    try:
        build(force=args.force)
    except KeyboardInterrupt:
        # The manifest is written last, so an interrupted build leaves none and
        # the next run starts over. Nothing to clean up.
        _log("interrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
