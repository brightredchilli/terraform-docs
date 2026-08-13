"""Build the packaged search index and per-document summaries.

Bare invocation builds everything, skipping whatever stage's inputs haven't
changed:

    uv run src/terraform_docs_mcp/build_index.py

Or run one stage explicitly:

    uv run src/terraform_docs_mcp/build_index.py index
    uv run src/terraform_docs_mcp/build_index.py summaries

Not part of the installed tool: it imports the ``build`` dependency group
(langchain-text-splitters, huggingface-hub, pyyaml) and writes into the
package's ``_data``/``src/summaries`` directories, which are then shipped (or,
for summaries, read) by later steps.

Imports are absolute (``terraform_docs_mcp.x``), not relative (``.x``),
specifically so this file can be run directly by path as well as via
``-m`` -- a relative import has no meaning when a file is executed as
``__main__`` rather than imported as a submodule, and running it directly
would fail with "attempted relative import with no known parent package"
otherwise. This is the one module in the package where that matters.

``uv_build`` runs no build hooks, so this cannot be triggered from inside
``uv build``. The Makefile enforces the ordering instead.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from typing import Annotated, Sequence

import typer

# Disabled alongside the vector-embed build below -- see _run_index(). `_write_db`
# is left defined but uncalled, so INDEX_FILENAME/SCHEMA (which it still
# references) stay imported; VECTORS_FILENAME/quantize do not, since those are
# only used by the disabled vector-writing block.
# import numpy as np

from terraform_docs_mcp._config import DATA_DIR, DOCUMENTS_INDEX_FILENAME, PROJECT_ROOT
from terraform_docs_mcp.manifest import Manifest, git_sha, repo_of

from terraform_docs_mcp.corpus import (
    Document,
    iter_documents,
    PROVIDERS,
)  # chunk_document: see below
from terraform_docs_mcp.db import Db
from terraform_docs_mcp.summarize import ensure_summaries
from terraform_docs_mcp.util import handle_broken_pipe

# from terraform_docs_mcp.embed import SentenceTransformerEmbedder, download_model
from terraform_docs_mcp.index import (
    INDEX_FILENAME,
    SCHEMA,
)  # VECTORS_FILENAME, quantize: disabled

app = typer.Typer()


def _log(message: str) -> None:
    print(f"[build-index] {message}", flush=True)


def _current_commits() -> dict[str, str]:
    return {name: git_sha(repo_of(config)) for name, config in PROVIDERS.items()}


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


def _index_stale(commits: dict[str, str]) -> bool:
    recorded = Manifest.read(DATA_DIR)
    return (
        recorded.documents_aws_commit_sha != commits["aws"]
        or recorded.documents_google_commit_sha != commits["google"]
        or not (DATA_DIR / DOCUMENTS_INDEX_FILENAME).exists()
    )


def _run_index(force: bool = False) -> None:
    """Rebuild ``documents.sqlite3`` if either provider commit moved.

    Wholesale, not incremental: the database is unlinked and every document
    reloaded and reinserted, so there is no partial state to reason about.
    """
    commits = _current_commits()
    if not force and not _index_stale(commits):
        _log("documents up to date")
        return
    _log("rebuilding documents.sqlite3")

    _log("loading documents")
    t0 = time.time()
    documents = []
    for provider in PROVIDERS.values():
        for doc in iter_documents(provider):
            documents.append(doc)
    _log(f"{len(documents)} documents ({time.time() - t0:.1f}s)")

    _log("writing document-level trigram index")
    _write_documents_db(documents)

    _log("copying documentation and licenses")
    _copy_docs()

    Manifest.update(
        DATA_DIR, documents_aws=commits["aws"], documents_google=commits["google"]
    )
    _log("wrote manifest (documents_aws, documents_google)")


def _run_summaries(force: bool = False) -> None:
    """Refresh ``src/summaries/`` for any document whose checksum changed.

    Reads from ``documents.sqlite3`` (via ``Db.iter_all``), so it always runs
    against whatever the last ``_run_index`` actually produced -- run ``index``
    first if the database itself is stale.
    """
    db = Db(DATA_DIR / DOCUMENTS_INDEX_FILENAME, readonly=True)
    try:
        counts = ensure_summaries(db, force=force)
    finally:
        db.close()
    _log(
        f"summaries: {counts['written']} written, {counts['regenerated']} regenerated, "
        f"{counts['skipped']} skipped"
    )

    commits = _current_commits()
    Manifest.update(
        DATA_DIR, summaries_aws=commits["aws"], summaries_google=commits["google"]
    )
    _log("wrote manifest (summaries_aws, summaries_google)")


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


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    """Build everything: the document index, then summaries. Each stage skips
    itself if its own inputs haven't changed. Run `index` or `summaries`
    directly for just one stage."""
    if ctx.invoked_subcommand is not None:
        return
    _run_index()
    _run_summaries()


@app.command()
def index(
    force: Annotated[
        bool, typer.Option(help="Rebuild even when nothing changed.")
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            help="Report whether a rebuild is needed and exit 1 if so; build nothing."
        ),
    ] = False,
):
    """Build the packaged search index in `_data/`."""
    if force and check:
        raise typer.BadParameter("--force and --check are mutually exclusive")

    if check:
        commits = _current_commits()
        if _index_stale(commits):
            _log("stale: a provider commit moved (or documents.sqlite3 is missing)")
            raise typer.Exit(1)
        _log("up to date")
        raise typer.Exit(0)

    try:
        _run_index(force=force)
    except KeyboardInterrupt:
        _log("interrupted")
        raise typer.Exit(130)


@app.command()
def summaries(
    force: Annotated[
        bool,
        typer.Option(help="Regenerate every summary, even ones that already exist."),
    ] = False,
):
    """Refresh `src/summaries/` for any document whose content changed."""
    try:
        _run_summaries(force=force)
    except KeyboardInterrupt:
        _log("interrupted")
        raise typer.Exit(130)


@handle_broken_pipe
def main() -> int:
    try:
        app()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
