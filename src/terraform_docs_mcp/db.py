"""Document-level storage and trigram full-text search.

Documents only -- no chunks. Chunking exists solely because the embedding
model truncates at 512 tokens (see ``corpus.py``); FTS5 has no such limit, so
text search never needed it. Chunk-level bookkeeping for the vector channel is
a separate, not-yet-designed concern and does not belong here.

Nothing outside this module opens a database, names a table or a column, or
sees a :class:`sqlite3.Row`. Callers hand in a path and domain objects
(:class:`~terraform_docs_mcp.corpus.Document`,
:class:`~terraform_docs_mcp.corpus.Provider`,
:class:`~terraform_docs_mcp.corpus.Kind`) and get :class:`StoredDocument`
objects back.

.. note::

   **Independent of the production index.** This module is built into its own
   artifact (``_data/documents.sqlite3``, see ``build_index.py``), separate
   from ``index.sqlite3`` and the existing hybrid search in ``index.py``. Both
   are built from the same corpus; nothing about the existing pipeline reads
   from or writes to this one.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .corpus import Document, Kind, Provider
from .search import to_fts_match

#: Documents returned by :meth:`Db.search` when the caller names no limit.
#: Generous, because it may feed rank fusion rather than a user directly.
CANDIDATES = 60

# Columns of `documents`, in insert order. Named explicitly and used to build
# both the INSERT and every SELECT, so that adding a column cannot leave a
# `SELECT *` silently handing back rows with fields nobody reads.
_DOCUMENT_COLUMNS = ("doc_id", "provider", "kind", "body", "heading", "checksum", "rel_path")

SCHEMA = """
CREATE TABLE documents (
    doc_id   TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    kind     TEXT NOT NULL,
    body     TEXT NOT NULL,
    -- title, and the same string with underscores replaced by spaces, e.g.
    -- "aws_s3_bucket aws s3 bucket" -- so a query that omits underscores
    -- still finds an identifier that uses them. Not a GENERATED column:
    -- title itself is not stored (it is redundant with doc_id's trailing
    -- segment), so this is computed once in Python at write time instead.
    heading  TEXT NOT NULL,
    -- sha256(body), hex. Lets summarize.py detect exactly which documents'
    -- content actually changed, independent of whether the provider commit
    -- as a whole moved.
    checksum TEXT NOT NULL,
    -- Re-added after StoredDocument originally dropped it: summarize.py needs
    -- it to reconstruct the original website/docs/<kind-dir>/<stem> layout
    -- under src/summaries/, which doc_id/kind alone cannot do (the kind
    -- enum's value, e.g. "datasource", is not the same string as the source
    -- directory name, "d").
    rel_path TEXT NOT NULL
);
CREATE INDEX idx_documents_provider ON documents(provider);
CREATE INDEX idx_documents_kind     ON documents(kind);

-- Built into SQLite core (3.34+); no extension to load or ship. Trigram
-- indexes every substring rather than whole words, so "s3bucket" or
-- "lifecyc" match inside a longer identifier, while a query like "s3bucket"
-- still does not spuriously match "aws s3 bucket" prose (there is no such
-- literal substring). content='documents' means FTS5 reads heading straight
-- from the table rather than storing a second copy.
--
-- heading only, not body: indexing full page bodies let a page merely
-- *mentioning* a term outrank -- or bury -- a page whose identifier *is* that
-- term (e.g. a query for "aws_s3_bucket" failed to surface
-- aws:resource:aws_s3_bucket at all in the top 8, beaten by every longer
-- name containing it as a prefix, such as aws_s3_bucket_versioning). Search
-- is scoped to the identifier signal; body remains in `documents` for
-- display and is read directly by get_document, just not indexed.
CREATE VIRTUAL TABLE documents_fts USING fts5(
    heading,
    content='documents',
    content_rowid='rowid',
    tokenize='trigram'
);
"""


class ReadOnly(RuntimeError):
    """A write was attempted against a :class:`Db` opened read-only."""


@dataclass(frozen=True)
class Counts:
    documents: int


@dataclass(frozen=True)
class StoredDocument:
    """What `documents` actually persists -- narrower than corpus.Document.

    ``title`` is not stored: it is redundant with ``doc_id``'s trailing
    segment (``f"{provider}:{kind}:{title}"``, see ``corpus.iter_documents``).
    ``subcategory``/``description`` are not stored either -- they are
    build-time-only inputs to the embedded chunk text
    (``corpus._summary_chunk``) and are never queried back out of storage.
    """

    doc_id: str
    provider: Provider
    kind: Kind
    body: str
    heading: str
    checksum: str
    rel_path: str


def _heading(title: str) -> str:
    """title, plus the same string with underscores turned into spaces.

    Lets a query that omits underscores ("s3 bucket") still match an
    identifier that uses them ("aws_s3_bucket") via the trigram index.
    """
    return f"{title} {title.replace('_', ' ')}"


def _checksum(body: str) -> str:
    """sha256(body), hex. Used by summarize.py to detect exactly which
    documents' content changed, independent of whether the provider commit as
    a whole moved."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class Db:
    """The document store.

    Connections are per-thread, because sqlite connections are thread-affine
    and a single ``Db`` may be shared across a threadpool.
    """

    def __init__(self, path: Path, readonly: bool) -> None:
        self.path = Path(path)
        self.readonly = readonly
        self._local = threading.local()
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- lifecycle

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            if self.readonly:
                conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            else:
                conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def commit(self) -> None:
        self._conn().commit()

    def close(self) -> None:
        """Close this thread's connection. Safe to call more than once.

        Only this thread's: sqlite3 refuses to close a connection from a thread
        other than the one that opened it, so connections made on other threads
        are released when they are garbage collected.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def vacuum(self) -> None:
        """Compact the database.

        Commits first -- VACUUM cannot run inside a transaction, and any
        preceding write leaves one open.
        """
        self._require_writable()
        self._conn().commit()
        self._conn().execute("VACUUM")

    def _require_writable(self) -> None:
        if self.readonly:
            raise ReadOnly(f"{self.path} was opened read-only")

    # ----------------------------------------------------------------- schema

    def create_schema(self) -> None:
        self._require_writable()
        self._conn().executescript(SCHEMA)

    # ---------------------------------------------------------------- writing

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Store documents.

        Accepts the richer ``corpus.Document`` -- that is what callers have on
        hand -- and persists ``doc_id, provider, kind, body, rel_path`` plus a
        ``heading`` computed from ``title`` and a ``checksum`` computed from
        ``body``. ``title``/``subcategory``/``description`` are read and
        discarded; see :class:`StoredDocument`.
        """
        self._require_writable()
        placeholders = ",".join("?" * len(_DOCUMENT_COLUMNS))
        self._conn().executemany(
            f"INSERT INTO documents ({','.join(_DOCUMENT_COLUMNS)})"
            f" VALUES ({placeholders})",
            [
                (
                    d.doc_id,
                    d.provider.value,
                    d.kind.value,
                    d.body,
                    _heading(d.title),
                    _checksum(d.body),
                    d.rel_path,
                )
                for d in documents
            ],
        )

    def rebuild_fts(self) -> None:
        """(Re)build the FTS index from the documents table's current contents.

        External content tables have no automatic sync -- normally you would
        add triggers to keep them current row by row, but this project always
        replaces the whole database in one pass, so a single rebuild after the
        bulk load is simpler and correct.
        """
        self._require_writable()
        self._conn().execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")

    # ---------------------------------------------------------------- reading

    def get_documents(self, doc_ids: Sequence[str]) -> list[StoredDocument]:
        """Documents for the given ids, in the order asked for.

        Unknown ids are skipped rather than raising, so a caller ranking ids
        from elsewhere does not have to pre-validate them.
        """
        if not doc_ids:
            return []
        placeholders = ",".join("?" * len(doc_ids))
        rows = self._conn().execute(
            f"SELECT {','.join(_DOCUMENT_COLUMNS)} FROM documents"
            f" WHERE doc_id IN ({placeholders})",
            list(doc_ids),
        )
        by_id = {row["doc_id"]: _document(row) for row in rows}
        return [by_id[i] for i in doc_ids if i in by_id]

    def get_documents_by_identifier(
        self,
        identifiers: Iterable[str],
        provider: Provider | None = None,
        kind: Kind | None = None,
    ) -> list[StoredDocument]:
        """Documents whose doc_id ends with one of these identifiers.

        ``doc_id`` has the shape ``<provider>:<kind>:<identifier>`` (see
        ``corpus.iter_documents``), so the identifier is recovered from its
        trailing segment rather than a stored title column -- title is
        redundant with doc_id and is not stored separately.

        Matched with ``GLOB``, not ``LIKE``: every identifier here contains
        underscores, and ``LIKE``'s ``_`` wildcard matches any single
        character, so ``LIKE '%:aws_s3_bucket'`` would silently also match
        ``awsXs3Xbucket``. ``GLOB`` has no such trap, but is case-sensitive --
        callers must lowercase their own input, matching the fact that doc_ids
        are already lowercase by construction.
        """
        wanted = list(identifiers)
        if not wanted:
            return []
        clauses = " OR ".join(["d.doc_id GLOB '*:' || ?"] * len(wanted))
        where, params = _filters(provider, kind)
        sql = f"SELECT {','.join(_DOCUMENT_COLUMNS)} FROM documents d WHERE ({clauses})"
        if where:
            sql += f" AND {where}"
        rows = self._conn().execute(sql, [*wanted, *params])
        return [_document(row) for row in rows]

    def search(
        self,
        query: str,
        provider: Provider | None = None,
        kind: Kind | None = None,
        limit: int = CANDIDATES,
    ) -> list[StoredDocument]:
        """Documents matching ``query``, best first.

        Trigram full-text search over ``heading`` alone -- not ``body``; see
        the comment on ``documents_fts`` in ``SCHEMA``. One row per document
        -- ``documents_fts`` is 1:1 with ``documents`` -- so this is a plain
        join and order-by, with no chunk-to-document rollup needed.
        """
        match = to_fts_match(query)
        if not match:
            return []
        where, params = _filters(provider, kind)
        sql = (
            f"SELECT {','.join(f'd.{c}' for c in _DOCUMENT_COLUMNS)}"
            " FROM documents_fts f"
            " JOIN documents d ON d.rowid = f.rowid"
            " WHERE documents_fts MATCH ?"
        )
        if where:
            sql += f" AND {where}"
        # bm25() is negative and more negative is better, hence ascending.
        sql += " ORDER BY bm25(documents_fts) LIMIT ?"
        rows = self._conn().execute(sql, [match, *params, limit])
        return [_document(row) for row in rows]

    def counts(self) -> Counts:
        conn = self._conn()
        return Counts(
            documents=conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()[
                "n"
            ],
        )

    def iter_all(self) -> Iterator[StoredDocument]:
        """Every stored document, ordered by doc_id for determinism.

        Feeds summarize.py's per-document checksum comparison -- it needs
        every document's checksum, not a filtered or ranked subset.
        """
        rows = self._conn().execute(
            f"SELECT {','.join(_DOCUMENT_COLUMNS)} FROM documents ORDER BY doc_id"
        )
        for row in rows:
            yield _document(row)


def _document(row: sqlite3.Row) -> StoredDocument:
    return StoredDocument(
        doc_id=row["doc_id"],
        provider=Provider(row["provider"]),
        kind=Kind(row["kind"]),
        body=row["body"],
        heading=row["heading"],
        checksum=row["checksum"],
        rel_path=row["rel_path"],
    )


def _filters(
    provider: Provider | None, kind: Kind | None
) -> tuple[str, list[str]]:
    """Build the WHERE fragment for the document filters.

    Private on purpose: callers pass ``Provider``/``Kind`` values and never see
    SQL.
    """
    clauses: list[str] = []
    params: list[str] = []
    if provider is not None:
        clauses.append("d.provider = ?")
        params.append(provider.value)
    if kind is not None:
        clauses.append("d.kind = ?")
        params.append(kind.value)
    return " AND ".join(clauses), params
