"""On-disk index: SQLite (metadata + BM25) alongside a quantized vector array.

Read path only. Writing lives in :mod:`terraform_docs_mcp.build_index`, which
is build-time and pulls dependencies this module deliberately avoids.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._config import data_dir as _default_data_dir
from .embed import MODEL_ID, SentenceTransformerEmbedder
from .search import (
    aggregate_to_documents,
    infer_provider,
    reciprocal_rank_fusion,
    to_fts_match,
)

INDEX_FILENAME = "index.sqlite3"
VECTORS_FILENAME = "vectors.i8.npy"

#: Chunks pulled from each retrieval channel before fusion. Generous relative
#: to the default limit of 10 because fusion and document-level aggregation
#: both collapse candidates.
CHANNEL_CANDIDATES = 60

# Unit vectors have components in [-1, 1], so a fixed scale beats a per-row one
# and needs no side table. int8 holds the array to ~15 MB on disk versus ~59 MB
# at float32, which matters because it ships inside the wheel.
QUANT_SCALE = 127.0

#: Terraform identifiers always contain an underscore, which distinguishes them
#: from ordinary prose words.
_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

#: Resources are the usual intent when a bare identifier is searched; data
#: sources share the same name and should follow rather than displace them.
_KIND_PRIORITY = {"resource": 0, "datasource": 1}

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE documents (
    doc_id      TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    subcategory TEXT,
    description TEXT,
    rel_path    TEXT NOT NULL
);
CREATE INDEX idx_documents_provider ON documents(provider);
CREATE INDEX idx_documents_kind     ON documents(kind);

-- chunks.id is 1-based and dense: vectors row i corresponds to chunk id i + 1.
CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    ordinal      INTEGER NOT NULL,
    heading_path TEXT,
    snippet      TEXT NOT NULL
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);

-- External content: FTS5 keeps the inverted index but not the text, since the full
-- markdown already ships alongside and chunk snippets live in `chunks`.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='',
    tokenize='unicode61 remove_diacritics 2'
);
"""


class IndexUnavailable(RuntimeError):
    """The packaged index is missing or was built by a different model."""


def quantize(vectors: np.ndarray) -> np.ndarray:
    """Convert unit float32 vectors to int8."""
    return np.round(vectors * QUANT_SCALE).clip(-127, 127).astype(np.int8)


class Index:
    """Query interface over the packaged index.

    Holds no per-request state, so instances are safe to share across the
    threadpool that serves MCP tool calls. SQLite connections are the one
    exception -- they are thread-affine, so each thread gets its own.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir = data_dir or _default_data_dir()
        self._db_path = self._dir / INDEX_FILENAME
        self._vectors_path = self._dir / VECTORS_FILENAME
        self._local = threading.local()
        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None
        self._embedder: SentenceTransformerEmbedder | None = None

        if not self._db_path.exists():
            raise IndexUnavailable(
                f"No search index at {self._db_path}.\n"
                "The installed package was built without one. The index is "
                "generated rather than committed, so building straight from a "
                "git checkout or source tree produces a package with no data "
                "in it.\n"
                "  In the source tree:  make build   (then install dist/*.whl)\n"
                "  As a dependency:     depend on the built wheel, not on the "
                "git repository."
            )

    # ---------------------------------------------------------------- lazy

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @property
    def vectors(self) -> np.ndarray:
        """Dequantized document matrix, ``(n_chunks, dim)`` float32."""
        if self._vectors is None:
            with self._lock:
                if self._vectors is None:
                    raw = np.load(self._vectors_path, mmap_mode="r")
                    # One-off dequantization (~40 ms) buys a plain float32
                    # matmul per query; leaving it int8 would force numpy to
                    # upcast the whole matrix on every search instead. Scaled
                    # in place so only one float32 copy is ever live.
                    vectors = np.asarray(raw, dtype=np.float32)
                    vectors /= QUANT_SCALE
                    self._vectors = vectors
        return self._vectors

    @property
    def embedder(self) -> SentenceTransformerEmbedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = SentenceTransformerEmbedder(self._dir / "model")
        return self._embedder

    # ------------------------------------------------------------- metadata

    def meta(self) -> dict[str, str]:
        rows = self._conn().execute("SELECT key, value FROM meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def _check_compatible(self) -> None:
        """Refuse to search an index built by a different model.

        Compares the recorded model id rather than asking the embedder, so
        this stays cheap: it must not force a multi-hundred-megabyte torch
        model to load just to reject a stale index.
        """
        meta = self.meta()
        if meta.get("model_id") != MODEL_ID:
            raise IndexUnavailable(
                f"Index was built with {meta.get('model_id')} (dim {meta.get('dim')}) "
                f"but this build embeds with {MODEL_ID}. Rebuild with `make index`."
            )
        recorded = int(meta.get("dim", 0))
        if self.vectors.shape[1] != recorded:
            raise IndexUnavailable(
                f"Vector array has {self.vectors.shape[1]} dimensions but the index "
                f"records {recorded}. Rebuild with `make index`."
            )

    # ------------------------------------------------------------ retrieval

    def _filter_clause(
        self, provider: str | None, kind: str | None
    ) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if provider:
            clauses.append("d.provider = ?")
            params.append(provider)
        if kind:
            clauses.append("d.kind = ?")
            params.append(kind)
        return (" AND ".join(clauses), params)

    def _bm25(self, query: str, provider: str | None, kind: str | None) -> list[int]:
        match = to_fts_match(query)
        if not match:
            return []
        where, params = self._filter_clause(provider, kind)
        sql = (
            "SELECT f.rowid AS id FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE chunks_fts MATCH ?"
        )
        if where:
            sql += f" AND {where}"
        # bm25() is negative with better matches more negative, so ascending.
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        rows = (
            self._conn().execute(sql, [match, *params, CHANNEL_CANDIDATES]).fetchall()
        )
        return [r["id"] for r in rows]

    def _allowed_rows(
        self, provider: str | None, kind: str | None
    ) -> np.ndarray | None:
        """Chunk ids passing the filters, or ``None`` when unfiltered."""
        if not provider and not kind:
            return None
        where, params = self._filter_clause(provider, kind)
        rows = (
            self._conn()
            .execute(
                f"SELECT c.id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE {where}",
                params,
            )
            .fetchall()
        )
        return np.array([r["id"] for r in rows], dtype=np.int64)

    def _dense(self, query: str, provider: str | None, kind: str | None) -> list[int]:
        vectors = self.vectors
        scores = vectors @ self.embedder.embed_query(query)

        allowed = self._allowed_rows(provider, kind)
        if allowed is not None:
            if allowed.size == 0:
                return []
            # Restrict *before* taking the top-k. Filtering afterwards would
            # return far fewer than `limit` results for narrow filters such as
            # kind="functions", which covers only a handful of documents.
            subset = scores[allowed - 1]
            k = min(CHANNEL_CANDIDATES, subset.size)
            top = np.argpartition(-subset, k - 1)[:k]
            top = top[np.argsort(-subset[top])]
            return [int(allowed[i]) for i in top]

        k = min(CHANNEL_CANDIDATES, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [int(i) + 1 for i in top]

    def search(
        self,
        query: str,
        provider: str | None = None,
        kind: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search returning ranked documents."""
        query = (query or "").strip()
        if not query:
            return []
        self._check_compatible()

        # An explicit filter always wins; otherwise honour a provider the query
        # names itself.
        if provider is None:
            provider = infer_provider(query)

        fused = reciprocal_rank_fusion(
            [self._bm25(query, provider, kind), self._dense(query, provider, kind)]
        )
        chunk_rows = self._chunk_details(fused.keys())
        ranked = aggregate_to_documents(fused, chunk_rows, limit)

        # A query naming a resource outright should return that resource, full
        # stop. Relevance ranking alone does not guarantee it: `aws_lambda_function`
        # appears verbatim in dozens of related pages, so BM25 happily ranks
        # `aws_lambda_function_url` or a data source above the resource itself.
        exact = self._exact_name_matches(query, provider, kind)
        if exact:
            ranked = self._promote(exact, ranked)

        return self._decorate(ranked[:limit])

    def _exact_name_matches(
        self, query: str, provider: str | None, kind: str | None
    ) -> list[str]:
        """Documents whose terraform identifier the query states verbatim."""
        candidates = {query.strip().lower()}
        candidates.update(t.lower() for t in _IDENTIFIER_TOKEN.findall(query))
        candidates = {c for c in candidates if "_" in c}
        if not candidates:
            return []

        placeholders = ",".join("?" * len(candidates))
        where, params = self._filter_clause(provider, kind)
        sql = f"SELECT doc_id, kind FROM documents d WHERE lower(d.title) IN ({placeholders})"
        if where:
            sql += f" AND {where}"
        rows = self._conn().execute(sql, [*candidates, *params]).fetchall()
        return [
            r["doc_id"]
            for r in sorted(rows, key=lambda r: _KIND_PRIORITY.get(r["kind"], 2))
        ]

    def _promote(
        self, doc_ids: list[str], ranked: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Move the given documents to the front, preserving the rest's order."""
        existing = {e["doc_id"]: e for e in ranked}
        head: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            entry = existing.pop(doc_id, None)
            head.append(entry if entry is not None else self._summary_entry(doc_id))
        return head + [e for e in ranked if e["doc_id"] in existing]

    def _summary_entry(self, doc_id: str) -> dict[str, Any]:
        """Build a result entry for a document neither channel surfaced."""
        row = (
            self._conn()
            .execute(
                "SELECT heading_path, snippet FROM chunks WHERE doc_id = ? ORDER BY ordinal LIMIT 1",
                (doc_id,),
            )
            .fetchone()
        )
        return {
            "doc_id": doc_id,
            "score": 0.0,
            "heading_path": row["heading_path"] if row else None,
            "snippet": row["snippet"] if row else "",
        }

    def _chunk_details(self, ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        ids = list(ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = (
            self._conn()
            .execute(
                f"SELECT id, doc_id, heading_path, snippet FROM chunks WHERE id IN ({placeholders})",
                ids,
            )
            .fetchall()
        )
        return {r["id"]: dict(r) for r in rows}

    def _decorate(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not ranked:
            return []
        ids = [r["doc_id"] for r in ranked]
        placeholders = ",".join("?" * len(ids))
        rows = (
            self._conn()
            .execute(f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", ids)
            .fetchall()
        )
        by_id = {r["doc_id"]: dict(r) for r in rows}

        out = []
        for entry in ranked:
            doc = by_id.get(entry["doc_id"])
            if not doc:
                continue
            out.append(
                {
                    "doc_id": doc["doc_id"],
                    "provider": doc["provider"],
                    "kind": doc["kind"],
                    # "name": doc["name"],
                    "title": doc["title"],
                    "subcategory": doc["subcategory"],
                    "description": doc["description"],
                    "heading": entry["heading_path"],
                    "snippet": entry["snippet"],
                    "score": round(entry["score"], 6),
                }
            )
        return out

    # ------------------------------------------------------------- document

    def get_document(self, doc_id: str) -> str:
        """Return a document's markdown"""
        row = (
            self._conn()
            .execute(
                "SELECT provider, rel_path, title FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(f"Unknown doc_id: {doc_id!r}")

        path = self._dir / "docs" / row["provider"] / row["rel_path"]
        text = path.read_text(encoding="utf-8", errors="replace")
        body = _strip_frontmatter(text)
        return body

    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {"documents": docs, "chunks": chunks, **self.meta()}


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    return text[end + 4 :].lstrip("\n") if end != -1 else text


# def _extract_section(body: str, section: str, doc_id: str) -> str:
#     import re
#
#     wanted = section.strip().lower().lstrip("#").strip()
#     matches = list(re.finditer(r"^##\s+(.+)$", body, re.MULTILINE))
#     for i, m in enumerate(matches):
#         if m.group(1).strip().lower() == wanted:
#             end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
#             return body[m.start() : end].rstrip()
#     available = ", ".join(m.group(1).strip() for m in matches) or "none"
#     raise KeyError(
#         f"No section {section!r} in {doc_id}. Available sections: {available}"
#     )
