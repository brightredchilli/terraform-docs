"""Tests for per-document summary extraction and caching.

Most of these run against synthetic `Document`s and a `tmp_path` standing in
for `SUMMARIES_DIR`. `TestExtractIntroOnRealDocument` is the exception: it
reads one real file from the submodule and pins extract_intro's exact output
against it, so a change to the splitting logic that alters real output gets
caught here, not just against hand-built synthetic markdown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terraform_docs_mcp import summarize
from terraform_docs_mcp._config import PROJECT_ROOT
from terraform_docs_mcp.corpus import Document, Kind, Provider, extract_intro, strip_frontmatter
from terraform_docs_mcp.db import Db


def _doc(
    doc_id: str = "aws:resource:example",
    provider: Provider = Provider.aws,
    kind: Kind = Kind.resource,
    rel_path: str = "r/example.html.markdown",
    body: str = "# Resource: example\n\nAn example resource.\n\n## Example Usage\n\nblah\n",
) -> Document:
    return Document(
        doc_id=doc_id,
        provider=provider,
        kind=kind,
        title="aws_example",
        subcategory="EC2",
        description=None,
        rel_path=rel_path,
        body=body,
    )


class TestExtractIntro:
    def test_text_before_first_h2(self):
        body = "# Title\n\nIntro paragraph.\n\n## Example Usage\n\nThe rest.\n"
        assert extract_intro(body) == "# Title\nIntro paragraph."

    def test_no_h2_returns_whole_body(self):
        """No h2 at all -- the whole body is "the intro". The splitter still
        normalizes the paragraph break into its own hard-break convention, so
        this checks content survives, not byte-for-byte passthrough."""
        body = "# Title\n\nJust a short guide with no sections."
        intro = extract_intro(body)
        assert "Title" in intro
        assert "Just a short guide with no sections." in intro

    def test_fenced_code_containing_a_doubled_hash_comment_does_not_split(self):
        """The concrete case a naive `^## ` line scan gets wrong: an HCL
        example can contain a comment that itself starts with `##`."""
        body = """# Resource: aws_example

Intro paragraph.

## Example Usage

```
resource "aws_example" "this" {
  ## not a markdown heading, just an HCL comment
  name = "demo"
}
```

## Argument Reference

...
"""
        intro = extract_intro(body)
        assert intro == "# Resource: aws_example\nIntro paragraph."
        assert "Example Usage" not in intro
        assert "not a markdown heading" not in intro


S3_BUCKET_RESOURCE_DOC = (
    PROJECT_ROOT / "terraform-provider-aws/website/docs/r/s3_bucket.html.markdown"
)


class TestExtractIntroOnRealDocument:
    """Pins extract_intro's exact output against a real corpus file, rather
    than only ever exercising it against hand-built synthetic markdown."""

    @pytest.mark.skipif(
        not S3_BUCKET_RESOURCE_DOC.is_file(),
        reason="provider submodule not checked out; run `make bootstrap`",
    )
    def test_s3_bucket_resource(self):
        raw = S3_BUCKET_RESOURCE_DOC.read_text(encoding="utf-8")
        body, _frontmatter = strip_frontmatter(raw)

        intro = extract_intro(body)

        expected = """# Resource: aws_s3_bucket
Provides a S3 bucket resource.
-> This resource provides functionality for managing S3 general purpose buckets in an AWS Partition. To manage Amazon S3 Express directory buckets, use the [`aws_s3_directory_bucket`](/docs/providers/aws/r/s3_directory_bucket.html) resource. To manage [S3 on Outposts](https://docs.aws.amazon.com/AmazonS3/latest/dev/S3onOutposts.html), use the [`aws_s3control_bucket`](/docs/providers/aws/r/s3control_bucket.html) resource.
-> Object Lock can be enabled by using the `object_lock_enabled` attribute or by using the [`aws_s3_bucket_object_lock_configuration`](/docs/providers/aws/r/s3_bucket_object_lock_configuration.html) resource. Please note, that by using the resource, Object Lock can be enabled/disabled without destroying and recreating the bucket.
-> To support ABAC (Attribute Based Access Control) in general purpose buckets, this resource will now attempt to send tags in the create request and use the S3 Control tagging APIs [`TagResource`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_control_TagResource.html), [`UntagResource`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_control_UntagResource.html), and [`ListTagsForResource`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_control_ListTagsForResource.html) for read and update operations. The calling principal must have the corresponding `s3:TagResource`, `s3:UntagResource`, and `s3:ListTagsForResource` [IAM permissions](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazons3.html#amazons3-actions-as-permissions). If the principal lacks the appropriate permissions, the provider will fall back to tagging after creation and using the S3 tagging APIs [`PutBucketTagging`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutBucketTagging.html), [`DeleteBucketTagging`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_DeleteBucketTagging.html), and [`GetBucketTagging`](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetBucketTagging.html) instead. With ABAC enabled, tag modifications may fail with the fall back behavior. See the [AWS documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/buckets-tagging-enable-abac.html) for additional details on enabling ABAC in general purpose buckets."""

        assert intro == expected
        # The h2 that follows the intro must not have leaked in.
        assert "Example Usage" not in intro


class TestSummarizeStub:
    def test_returns_input_unchanged(self):
        assert summarize.summarize("some intro text") == "some intro text"
        assert summarize.summarize("") == ""


class TestSummaryDirAndStem:
    def test_scoped_by_provider_to_avoid_collision(self, db: Db):
        """AWS and Google both have a d/xyz.html.markdown; without provider
        scoping they would collide in the same summaries/d/ directory."""
        db.add_documents(
            [
                _doc("aws:datasource:xyz", provider=Provider.aws, rel_path="d/xyz.html.markdown"),
                _doc(
                    "google:datasource:xyz",
                    provider=Provider.google,
                    rel_path="d/xyz.html.markdown",
                ),
            ]
        )
        aws_doc, google_doc = db.get_documents(["aws:datasource:xyz", "google:datasource:xyz"])
        aws_dir, aws_stem = summarize._summary_dir_and_stem(aws_doc)
        google_dir, google_stem = summarize._summary_dir_and_stem(google_doc)
        assert aws_dir != google_dir
        assert aws_dir.parts[-2:] == ("aws", "d")
        assert google_dir.parts[-2:] == ("google", "d")
        assert aws_stem == google_stem == "xyz"

    def test_html_markdown_suffix_is_stripped(self, db: Db):
        db.add_documents([_doc("aws:resource:instance", rel_path="r/instance.html.markdown")])
        (doc,) = db.get_documents(["aws:resource:instance"])
        _, stem = summarize._summary_dir_and_stem(doc)
        assert stem == "instance"

    def test_plain_markdown_suffix_is_stripped(self, db: Db):
        """A handful of Google data-source pages use plain .markdown, not
        .html.markdown -- both must strip to the same bare stem."""
        db.add_documents([_doc("aws:datasource:bare", rel_path="d/bare.markdown")])
        (doc,) = db.get_documents(["aws:datasource:bare"])
        _, stem = summarize._summary_dir_and_stem(doc)
        assert stem == "bare"


@pytest.fixture
def db(tmp_path: Path) -> Db:
    database = Db(tmp_path / "documents.sqlite3", readonly=False)
    database.create_schema()
    return database


@pytest.fixture
def summaries_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "summaries"
    monkeypatch.setattr(summarize, "SUMMARIES_DIR", root)
    return root


class TestEnsureSummaries:
    def test_writes_a_missing_summary(self, db: Db, summaries_dir: Path):
        doc = _doc("aws:resource:example", body="# Title\n\nIntro.\n\n## Example Usage\n\nblah\n")
        db.add_documents([doc])
        (stored,) = db.get_documents(["aws:resource:example"])

        counts = summarize.ensure_summaries(db)

        assert counts == {"written": 1, "regenerated": 0, "skipped": 0}
        path = summarize._summary_path_for_checksum(stored, stored.checksum)
        assert path.read_text(encoding="utf-8") == extract_intro(doc.body)

    def test_existing_summary_with_matching_checksum_is_left_untouched(
        self, db: Db, summaries_dir: Path
    ):
        doc = _doc("aws:resource:example")
        db.add_documents([doc])
        (stored,) = db.get_documents(["aws:resource:example"])
        path = summarize._summary_path_for_checksum(stored, stored.checksum)
        path.parent.mkdir(parents=True)
        path.write_text("a hand-written summary, not what extract_intro would produce")

        counts = summarize.ensure_summaries(db)

        assert counts == {"written": 0, "regenerated": 0, "skipped": 1}
        assert path.read_text() == "a hand-written summary, not what extract_intro would produce"

    def test_changed_checksum_deletes_the_old_file_and_regenerates(
        self, db: Db, summaries_dir: Path
    ):
        """The concrete mechanism this whole design exists for: a document's
        content changing must be detected via the checksum embedded in the
        existing summary's filename, not an existence check alone."""
        doc = _doc("aws:resource:example")
        db.add_documents([doc])
        (stored,) = db.get_documents(["aws:resource:example"])
        stale_path = summarize._summary_path_for_checksum(stored, "0" * 64)
        stale_path.parent.mkdir(parents=True)
        stale_path.write_text("stale content from an old body")

        counts = summarize.ensure_summaries(db)

        assert counts == {"written": 0, "regenerated": 1, "skipped": 0}
        assert not stale_path.exists()
        fresh_path = summarize._summary_path_for_checksum(stored, stored.checksum)
        assert fresh_path.read_text(encoding="utf-8") == extract_intro(doc.body)

    def test_force_regenerates_even_a_matching_summary(self, db: Db, summaries_dir: Path):
        doc = _doc("aws:resource:example")
        db.add_documents([doc])
        (stored,) = db.get_documents(["aws:resource:example"])
        path = summarize._summary_path_for_checksum(stored, stored.checksum)
        path.parent.mkdir(parents=True)
        path.write_text("stale")

        counts = summarize.ensure_summaries(db, force=True)

        assert counts == {"written": 0, "regenerated": 1, "skipped": 0}
        assert path.read_text(encoding="utf-8") == extract_intro(doc.body)
