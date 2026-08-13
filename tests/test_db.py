"""Tests for the document store and trigram full-text search.

Everything here builds its own small database in ``tmp_path``, independent of
the packaged index -- ``db.py`` is a separate artifact from ``index.sqlite3``,
not wired into the existing hybrid pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terraform_docs_mcp.corpus import Document, Kind, Provider
from terraform_docs_mcp.db import CANDIDATES, Db, ReadOnly, StoredDocument, _heading


def _doc(
    doc_id: str,
    provider: Provider = Provider.aws,
    kind: Kind = Kind.resource,
    title: str = "aws_instance",
    body: str = "Provides an EC2 instance.",
) -> Document:
    return Document(
        doc_id=doc_id,
        provider=provider,
        kind=kind,
        title=title,
        subcategory="EC2",
        description="Provides an EC2 instance.",
        rel_path="r/instance.html.markdown",
        body=body,
    )


@pytest.fixture
def db(tmp_path: Path) -> Db:
    database = Db(tmp_path / "documents.sqlite3", readonly=False)
    database.create_schema()
    return database


@pytest.fixture
def populated(db: Db) -> Db:
    db.add_documents(
        [
            _doc(
                "aws:resource:aws_s3_bucket",
                title="aws_s3_bucket",
                body="Provides an S3 bucket resource. Supports lifecycle rules "
                "and expiration policies.",
            ),
            _doc(
                "aws:resource:aws_instance",
                title="aws_instance",
                body="Provides an EC2 instance resource.",
            ),
            _doc(
                "google:resource:google_compute_instance",
                provider=Provider.google,
                title="google_compute_instance",
                body="Provides a compute engine instance.",
            ),
            _doc(
                "aws:datasource:aws_instance",
                kind=Kind.datasource,
                title="aws_instance",
                body="Look up an existing EC2 instance.",
            ),
            _doc(
                "aws:resource:aws_s3_bucket_lifecycle_configuration",
                title="aws_s3_bucket_lifecycle_configuration",
                body="Manages an S3 bucket's lifecycle configuration.",
            ),
        ]
    )
    db.rebuild_fts()
    db.commit()
    return db


class TestHeading:
    def test_is_title_plus_de_underscored_title(self):
        assert _heading("aws_s3_bucket") == "aws_s3_bucket aws s3 bucket"

    def test_a_title_with_no_underscores_is_unchanged_by_the_second_half(self):
        assert _heading("guide") == "guide guide"


class TestDocuments:
    def test_round_trip(self, db: Db):
        db.add_documents([_doc("aws:resource:aws_instance")])
        (found,) = db.get_documents(["aws:resource:aws_instance"])
        assert found == StoredDocument(
            doc_id="aws:resource:aws_instance",
            provider=Provider.aws,
            kind=Kind.resource,
            body="Provides an EC2 instance.",
            heading="aws_instance aws instance",
            checksum=found.checksum,  # sha256(body), not worth duplicating here
            rel_path="r/instance.html.markdown",
        )

    def test_title_subcategory_description_are_not_persisted(self, db: Db):
        """StoredDocument is narrower than Document by design -- title is
        redundant with doc_id's trailing segment, and subcategory/description
        are build-time-only inputs to the embedded chunk text. rel_path *is*
        persisted (unlike the others) -- summarize.py needs it to reconstruct
        each document's original website/docs layout."""
        db.add_documents([_doc("aws:resource:aws_instance")])
        (found,) = db.get_documents(["aws:resource:aws_instance"])
        assert not hasattr(found, "title")
        assert not hasattr(found, "subcategory")
        assert not hasattr(found, "description")
        assert found.rel_path == "r/instance.html.markdown"

    def test_returned_in_the_order_asked_for(self, populated: Db):
        ids = ["google:resource:google_compute_instance", "aws:resource:aws_instance"]
        assert [d.doc_id for d in populated.get_documents(ids)] == ids

    def test_unknown_ids_are_skipped(self, populated: Db):
        found = populated.get_documents(["aws:resource:aws_instance", "nope"])
        assert [d.doc_id for d in found] == ["aws:resource:aws_instance"]

    def test_empty_request(self, populated: Db):
        assert populated.get_documents([]) == []


class TestGetDocumentsByIdentifier:
    def test_matches_the_doc_id_suffix(self, populated: Db):
        found = populated.get_documents_by_identifier(["aws_s3_bucket"])
        assert [d.doc_id for d in found] == ["aws:resource:aws_s3_bucket"]

    def test_returns_every_kind_sharing_an_identifier(self, populated: Db):
        """Resource and data source share a name; ordering is the caller's call."""
        found = populated.get_documents_by_identifier(["aws_instance"])
        assert {d.kind for d in found} == {Kind.resource, Kind.datasource}

    def test_filters_apply(self, populated: Db):
        found = populated.get_documents_by_identifier(
            ["aws_instance"], kind=Kind.datasource
        )
        assert [d.doc_id for d in found] == ["aws:datasource:aws_instance"]

    def test_empty_request(self, populated: Db):
        assert populated.get_documents_by_identifier([]) == []

    def test_uses_glob_not_like(self, db: Db):
        """LIKE's `_` wildcard matches any character, and every identifier here
        contains underscores -- LIKE '%:aws_s3_bucket' would silently also
        match a document literally named 'awsXs3Xbucket'. GLOB has no such
        trap."""
        db.add_documents(
            [
                _doc("aws:resource:aws_s3_bucket", title="aws_s3_bucket"),
                _doc("aws:resource:awsXs3Xbucket", title="awsXs3Xbucket"),
            ]
        )
        found = db.get_documents_by_identifier(["aws_s3_bucket"])
        assert [d.doc_id for d in found] == ["aws:resource:aws_s3_bucket"]

    def test_case_sensitive_caller_must_lowercase(self, populated: Db):
        """doc_ids are lowercase by construction; GLOB is case-sensitive, so an
        uppercase identifier simply finds nothing rather than matching loosely."""
        assert populated.get_documents_by_identifier(["AWS_S3_BUCKET"]) == []


class TestSearch:
    def test_body_text_is_not_searched(self, populated: Db):
        """heading is indexed, body is not -- see the comment on documents_fts
        in SCHEMA. A term appearing only in a page's body must not match,
        however distinctive: it was body text alone that used to bury exact
        identifier matches under longer, unrelated superstrings."""
        assert populated.search("expiration policies") == []

    def test_substring_match(self, populated: Db):
        """Trigram matches inside a longer token, not just whole words."""
        found = populated.search("lifecyc")
        assert "aws:resource:aws_s3_bucket_lifecycle_configuration" in [
            d.doc_id for d in found
        ]

    def test_de_underscored_heading_match(self, populated: Db):
        """A query omitting underscores still finds the identifier that uses
        them, via the heading field's de-underscored half."""
        found = populated.search("s3 bucket")
        assert "aws:resource:aws_s3_bucket" in [d.doc_id for d in found]

    def test_no_cross_word_false_positive(self, populated: Db):
        """'s3bucket' (no space) is not a literal substring of 'aws_s3_bucket'
        or its de-underscored form 'aws s3 bucket' -- trigram must not match it."""
        assert populated.search("s3bucket") == []

    def test_exact_identifier_is_not_buried_by_body_matches(self, db: Db):
        """This is the concrete failure that motivated excluding body: a page
        merely *mentioning* a term in its body must not compete with -- let
        alone outrank -- a page whose identifier *is* that term."""
        db.add_documents(
            [
                _doc(
                    "aws:resource:aws_gateway",
                    title="aws_gateway",  # "gateway" lands in the heading
                    body="Provides a networking primitive with no special terms.",
                ),
                _doc(
                    "aws:resource:aws_widget",
                    title="aws_widget",
                    body="This resource is sometimes used behind a gateway.",
                ),
            ]
        )
        db.rebuild_fts()
        db.commit()
        found = db.search("gateway")
        assert [d.doc_id for d in found] == ["aws:resource:aws_gateway"]

    def test_each_document_appears_once(self, populated: Db):
        found = populated.search("instance")
        ids = [d.doc_id for d in found]
        assert len(ids) == len(set(ids))

    def test_respects_provider(self, populated: Db):
        found = populated.search("instance", provider=Provider.google)
        assert {d.provider for d in found} == {Provider.google}

    def test_respects_kind(self, populated: Db):
        found = populated.search("instance", kind=Kind.datasource)
        assert {d.kind for d in found} == {Kind.datasource}

    def test_respects_limit(self, populated: Db):
        assert len(populated.search("instance", limit=1)) == 1

    def test_returns_whole_documents(self, populated: Db):
        """Matched via heading; body is still returned in full even though it
        played no part in finding the document."""
        (found,) = populated.search("lifecycle")
        assert found.body.startswith("Manages an S3 bucket's lifecycle")

    @pytest.mark.parametrize(
        "query",
        [
            'bucket "unbalanced',
            "bucket*",
            "bucket AND OR NEAR",
            "aws-instance",
            "column:value",
            "^caret",
            "(unclosed",
        ],
    )
    def test_fts_metacharacters_do_not_raise(self, populated: Db, query: str):
        populated.search(query)

    def test_empty_query(self, populated: Db):
        assert populated.search("") == []
        assert populated.search("   ") == []

    def test_no_matches(self, populated: Db):
        assert populated.search("kubernetes helm chart") == []

    def test_default_candidate_pool(self):
        assert CANDIDATES > 0


class TestLifecycle:
    def test_readonly_rejects_writes(self, populated: Db):
        reader = Db(populated.path, readonly=True)
        with pytest.raises(ReadOnly):
            reader.add_documents([_doc("aws:resource:aws_vpc")])
        with pytest.raises(ReadOnly):
            reader.create_schema()
        with pytest.raises(ReadOnly):
            reader.vacuum()
        with pytest.raises(ReadOnly):
            reader.rebuild_fts()

    def test_readonly_reads(self, populated: Db):
        reader = Db(populated.path, readonly=True)
        assert reader.counts() == populated.counts()

    def test_counts(self, populated: Db):
        assert populated.counts().documents == 5

    def test_vacuum(self, populated: Db):
        populated.vacuum()
        assert populated.counts().documents == 5

    def test_close_is_idempotent(self, populated: Db):
        populated.close()
        populated.close()

    def test_reopens_after_close(self, populated: Db):
        populated.close()
        assert populated.counts().documents == 5

    def test_creates_parent_directories(self, tmp_path: Path):
        nested = Db(tmp_path / "a" / "b" / "documents.sqlite3", readonly=False)
        nested.create_schema()
        assert nested.counts().documents == 0
