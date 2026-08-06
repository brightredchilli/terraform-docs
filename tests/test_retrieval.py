"""End-to-end tests against the built index.

Skipped when no index is present, so `make test` works before `make index`.
"""

from __future__ import annotations

import concurrent.futures

import pytest

from terraform_docs_mcp.index import Index, IndexUnavailable


@pytest.fixture(scope="module")
def index() -> Index:
    try:
        return Index()
    except IndexUnavailable as exc:
        pytest.skip(str(exc))


# (query, provider, expected doc_id). Three query shapes, because they exercise
# different halves of the hybrid retriever: exact identifiers are won by BM25,
# paraphrases by the vector channel, attribute phrases by both.
#
# `provider` is set only where the query names no cloud. "delete old files in a
# bucket" is a fair question to ask of either provider -- GCS lifecycle rules
# are as correct an answer as S3's -- so pinning a single expected document
# without scoping the search would be testing a coin flip, not retrieval.
# Queries that do say "aws" or "gcp" are left unscoped on purpose: resolving
# the provider from the wording is part of what should work.
RECALL_CASES = [
    # -- exact identifier -------------------------------------------------
    ("aws_instance", None, "aws:r/instance"),
    ("google_compute_firewall", None, "google:r/compute_firewall"),
    ("aws_lambda_function", None, "aws:r/lambda_function"),
    # -- attribute level --------------------------------------------------
    ("s3 bucket lifecycle expiration rules", None, "aws:r/s3_bucket_lifecycle_configuration"),
    ("iam role assume role policy document", None, "aws:r/iam_role"),
    ("rds db instance backup retention period", None, "aws:r/db_instance"),
    ("gke cluster node pool configuration", None, "google:r/container_cluster"),
    # -- paraphrase, provider stated in the query -------------------------
    ("create a virtual machine in google cloud", None, "google:r/compute_instance"),
    ("managed relational database in google cloud", None, "google:r/sql_database_instance"),
    ("allow inbound traffic on a port in gcp", None, "google:r/compute_firewall"),
    ("object storage bucket on google cloud", None, "google:r/storage_bucket"),
    ("grant an aws service permission to act on my behalf", None, "aws:r/iam_role"),
    # -- paraphrase, provider-agnostic wording (scoped) -------------------
    ("how do I automatically delete old files in a bucket", "aws",
     "aws:r/s3_bucket_lifecycle_configuration"),
    ("run code without managing servers", "aws", "aws:r/lambda_function"),
    ("look up an existing machine image to boot from", "aws", "aws:d/ami"),
]

# Queries the retriever is measurably weak on, kept in the suite rather than
# deleted so the weakness stays visible and any improvement shows up as XPASS.
# Not strict, because they sit close to the cut-off and can flip.
# Which queries are weak depends on the embedding model, so this set is
# re-measured whenever the model changes rather than carried forward.
KNOWN_WEAK = {
    "object storage bucket on google cloud":
        "'object storage' collides lexically with the storage_bucket_object "
        "family, which outranks the bucket resource itself (target rank ~8)",
    "grant an aws service permission to act on my behalf":
        "highly indirect phrasing of assume-role semantics; no lexical overlap "
        "with aws_iam_role at all (target outside top 20)",
    "how do I automatically delete old files in a bucket":
        "S3 table/lifecycle pages and the v4 upgrade guide outrank the "
        "lifecycle resource itself (target rank ~7)",
    "run code without managing servers":
        "a pure serverless paraphrase; App Runner and Bedrock agent runtimes "
        "are semantically just as good a match as Lambda (target outside "
        "top 20)",
}


def _case_params():
    for query, provider, expected in RECALL_CASES:
        marks = []
        if query in KNOWN_WEAK:
            marks.append(pytest.mark.xfail(reason=KNOWN_WEAK[query], strict=False))
        yield pytest.param(query, provider, expected, id=query, marks=marks)


@pytest.mark.parametrize("query,provider,expected", list(_case_params()))
def test_recall_at_5(index: Index, query: str, provider: str | None, expected: str):
    results = index.search(query, provider=provider, limit=5)
    found = [r["doc_id"] for r in results]
    assert expected in found, f"{expected!r} not in top-5: {found}"


def test_overall_recall_does_not_regress(index: Index):
    """Aggregate guard, so an improvement to one query cannot mask a regression.

    Measured 11/15 at rank 5 and 12/15 at rank 10 when this was written; the
    floor is set just below to catch real regressions without failing on noise.
    """
    hits = sum(
        1
        for query, provider, expected in RECALL_CASES
        if expected in [r["doc_id"] for r in index.search(query, provider=provider, limit=5)]
    )
    assert hits >= 10, f"recall@5 dropped to {hits}/{len(RECALL_CASES)}"


class TestFilters:
    def test_provider_filter(self, index: Index):
        results = index.search("instance", provider="google", limit=10)
        assert results
        assert {r["provider"] for r in results} == {"google"}

    def test_kind_filter(self, index: Index):
        results = index.search("ami image", kind="d", limit=10)
        assert results
        assert {r["kind"] for r in results} == {"d"}

    def test_narrow_filter_still_fills_results(self, index: Index):
        """Filtering must happen inside each channel, not after the top-k.

        `functions` covers only a handful of documents corpus-wide. If the
        filter were applied after taking the top-60 candidates, an unrelated
        query would return nothing at all.
        """
        results = index.search("parse an arn", kind="functions", limit=5)
        assert results, "post-filtering would have emptied this result set"
        assert {r["kind"] for r in results} == {"functions"}

    def test_combined_filters(self, index: Index):
        results = index.search("network", provider="aws", kind="r", limit=5)
        assert results
        assert all(r["provider"] == "aws" and r["kind"] == "r" for r in results)

    def test_impossible_filter_returns_empty(self, index: Index):
        assert index.search("anything", kind="does-not-exist", limit=5) == []


class TestQueryHandling:
    def test_empty_query(self, index: Index):
        assert index.search("", limit=5) == []
        assert index.search("   ", limit=5) == []

    @pytest.mark.parametrize(
        "query",
        ['bucket " unbalanced', "wildcard*", "a AND", "col:val", "s3-bucket", "(paren"],
    )
    def test_fts_syntax_does_not_raise(self, index: Index, query: str):
        index.search(query, limit=5)  # must not raise OperationalError

    def test_limit_is_respected(self, index: Index):
        assert len(index.search("bucket", limit=3)) <= 3

    def test_results_are_unique_documents(self, index: Index):
        ids = [r["doc_id"] for r in index.search("instance", limit=10)]
        assert len(ids) == len(set(ids))

    def test_results_carry_expected_fields(self, index: Index):
        r = index.search("aws_instance", limit=1)[0]
        assert set(r) >= {
            "doc_id", "provider", "kind", "name", "title", "snippet", "score",
        }
        assert r["snippet"]


class TestGetDocument:
    def test_returns_full_markdown(self, index: Index):
        text = index.get_document("aws:r/instance")
        assert text.startswith("# Resource: aws_instance")
        assert "## Argument Reference" in text

    def test_frontmatter_is_stripped(self, index: Index):
        assert not index.get_document("aws:r/instance").startswith("---")

    def test_section_extraction(self, index: Index):
        section = index.get_document("aws:r/instance", section="Argument Reference")
        assert section.startswith("## Argument Reference")
        assert "## Attribute Reference" not in section

    def test_section_is_case_insensitive(self, index: Index):
        assert index.get_document("aws:r/instance", section="argument reference")

    def test_unknown_document(self, index: Index):
        with pytest.raises(KeyError):
            index.get_document("aws:r/does-not-exist")

    def test_unknown_section_lists_available(self, index: Index):
        with pytest.raises(KeyError, match="Argument Reference"):
            index.get_document("aws:r/instance", section="No Such Section")


class TestConcurrency:
    def test_parallel_searches(self, index: Index):
        """SQLite connections are thread-affine.

        The MCP server dispatches tool calls onto a threadpool, so a single
        shared connection would raise "SQLite objects created in a thread can
        only be used in that same thread".
        """
        queries = ["aws_instance", "bucket lifecycle", "firewall", "iam role"] * 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda q: index.search(q, limit=3), queries))
        assert all(r for r in results)

    def test_parallel_document_reads(self, index: Index):
        ids = ["aws:r/instance", "google:r/compute_instance", "aws:d/ami"] * 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            texts = list(pool.map(index.get_document, ids))
        assert all(t for t in texts)


class TestIndexMetadata:
    def test_stats(self, index: Index):
        stats = index.stats()
        assert stats["documents"] > 4000
        assert stats["chunks"] > 10000
        assert stats["aws_commit"] != "unknown"

    def test_vectors_align_with_chunks(self, index: Index):
        """vectors row i must correspond to chunk id i+1."""
        n_chunks = index.stats()["chunks"]
        assert index.vectors.shape[0] == n_chunks
        assert index.vectors.shape[1] == index.embedder.dim

    def test_model_identity_recorded(self, index: Index):
        from terraform_docs_mcp.embed import MODEL_ID

        assert index.meta()["model_id"] == MODEL_ID
