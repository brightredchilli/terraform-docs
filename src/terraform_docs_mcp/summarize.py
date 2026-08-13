"""Per-document summaries, cached as source under ``src/summaries/``.

A library module, not an entry point -- run via ``build_index.py summaries``
(``make summaries``, or ``uv run src/terraform_docs_mcp/build_index.py
summaries``). Build-time only, like ``build_index.py`` -- not a ``cli.py``
command, since nothing installed needs to produce these. Also, unlike the rest
of ``build_index.py``, not wired into ``make index`` at all: nothing
downstream reads these files yet (see ``summarize()``), and generating them is
a fundamentally different kind of operation from the rest of the build --
potentially networked, slow, and non-deterministic -- so it stays a separate,
explicit step rather than something `make index` does silently.

Caching is per-document and checksum-based, read from ``documents.sqlite3``
(:meth:`~terraform_docs_mcp.db.Db.iter_all`), not staleness-based like the
rest of this build: a summary is regenerated exactly when the document it was
built from changed content, tracked by embedding that content's checksum in
the summary's own filename -- see :func:`_summary_dir_and_stem`.
"""

from __future__ import annotations

from pathlib import Path

from ._config import DOC_SUFFIXES, SUMMARIES_DIR
from .corpus import extract_intro
from .db import Db, StoredDocument

#: Hex characters of the checksum kept in a summary's filename. Short enough
#: to keep filenames sane, long enough that a collision within one provider's
#: ~2-4k documents is not a practical concern.
CHECKSUM_CHARS = 12


def summarize(text: str) -> str:
    """Placeholder. Returns ``text`` unchanged.

    The eventual implementation is a smolagents agent (``ToolCallingAgent`` or
    ``CodeAgent``) equipped with ``VisitWebpageTool``, so it can follow links
    found in the intro text -- this is *why* smolagents rather than a plain
    completion call: no model backend has been chosen yet, and there is
    nothing here to plug one into until there is. Tools sourced from an MCP
    server (``smolagents.ToolCollection.from_mcp``) are a plausible later
    addition, not built here.

    This is the only function that needs to change to go from placeholder to
    real: ``ensure_summaries`` calls it exactly once per document and writes
    whatever it returns, so the surrounding pipeline does not need to know
    the difference.
    """
    return text


def _summary_dir_and_stem(doc: StoredDocument) -> tuple[Path, str]:
    """Directory and filename stem (no checksum, no extension) for a
    document's summary.

    Mirrors the provider's own ``website/docs`` layout underneath a
    provider-scoped directory -- e.g. ``d/s3_bucket.html.markdown`` in the aws
    provider becomes a stem of ``s3_bucket`` inside ``src/summaries/aws/d/``.
    Provider-scoped because AWS and Google each have their own ``d/``, ``r/``,
    ... and would otherwise collide in a single shared directory.
    """
    rel = doc.rel_path
    for suffix in DOC_SUFFIXES:
        if rel.endswith(suffix):
            rel = rel[: -len(suffix)]
            break
    path = Path(rel)
    return SUMMARIES_DIR / doc.provider.value / path.parent, path.name


def _find_existing_summary(doc: StoredDocument) -> Path | None:
    """The document's cached summary file, whatever checksum it was written
    with, or ``None`` if there is none yet."""
    directory, stem = _summary_dir_and_stem(doc)
    matches = sorted(directory.glob(f"{stem}.___*.md"))
    return matches[0] if matches else None


def _summary_path_for_checksum(doc: StoredDocument, checksum: str) -> Path:
    directory, stem = _summary_dir_and_stem(doc)
    return directory / f"{stem}.___{checksum[:CHECKSUM_CHARS]}.md"


def ensure_summaries(db: Db, force: bool = False) -> dict[str, int]:
    """Write any summary that is missing or stale. Current ones are untouched.

    Reads documents from ``db`` (for their checksums), not the raw markdown
    directly -- summaries are keyed off what is actually in the database, the
    same content ``get_document``/search would return, not a fresh re-parse.

    A document's checksum is embedded in its summary's filename
    (``stem.___<checksum>.md``), so staleness is a filename comparison: if the
    checksum there matches the document's current one, the summary is left
    alone; if it differs, the old file is deleted and a fresh one written;
    if none exists yet, one is written. ``force`` regenerates every summary
    regardless of whether an existing one already matches.
    """
    written = 0
    regenerated = 0
    skipped = 0
    for doc in db.iter_all():
        target = _summary_path_for_checksum(doc, doc.checksum)
        existing = _find_existing_summary(doc)

        if existing is not None:
            if existing == target and not force:
                skipped += 1
                continue
            existing.unlink()
            regenerated += 1
        else:
            written += 1

        target.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize(extract_intro(doc.body))
        target.write_text(summary, encoding="utf-8")
    return {"written": written, "regenerated": regenerated, "skipped": skipped}
