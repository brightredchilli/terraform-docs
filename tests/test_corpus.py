"""Tests for document loading and chunking."""

from __future__ import annotations


import pytest

from terraform_docs_mcp._config import PROJECT_ROOT
from terraform_docs_mcp.corpus import (
    PROVIDER_AWS,
    PROVIDER_GOOGLE,
    Document,
    Kind,
    Provider,
    chunk_document,
    iter_documents,
    strip_frontmatter,
)

AWS = PROJECT_ROOT / "terraform-provider-aws"
GOOGLE = PROJECT_ROOT / "terraform-provider-google"

requires_submodules = pytest.mark.skipif(
    not (AWS / "website/docs").is_dir() or not (GOOGLE / "website/docs").is_dir(),
    reason="provider submodules not checked out; run `make bootstrap`",
)


class TestStripFrontmatter:
    def test_removes_yaml_block(self):
        body, fm = strip_frontmatter('---\nsubcategory: "EC2"\n---\n\n# Title\n')
        assert body.startswith("# Title")
        assert "subcategory" in fm

    def test_passes_through_when_absent(self):
        body, fm = strip_frontmatter("# Title\n\ntext")
        assert body.startswith("# Title")
        assert fm == ""

    def test_handles_comment_banner(self):
        """Google pages open with a '#'-commented banner inside the frontmatter.

        If this leaked into the body a markdown splitter would read those lines
        as H1 headings.
        """
        text = (
            "---\n"
            "# ------------------------------\n"
            "#   *** AUTO GENERATED CODE ***\n"
            "# ------------------------------\n"
            'subcategory: "Compute Engine"\n'
            "---\n\n# google_compute_instance\n\nBody.\n"
        )
        body, _ = strip_frontmatter(text)
        assert body.startswith("# google_compute_instance")
        assert "AUTO GENERATED" not in body


class TestChunking:
    @staticmethod
    def _doc(body: str) -> Document:
        return Document(
            doc_id="aws:resource:example",
            provider=Provider.aws,
            kind=Kind.resource,
            title="Resource: aws_example",
            subcategory="EC2",
            description=None,
            rel_path="r/example.html.markdown",
            body=body,
        )

    def test_hcl_comments_do_not_create_sections(self):
        """'#' inside a fenced HCL block is a comment, not a heading.

        These docs are full of them, so a naive regex split would shred every
        example block into phantom sections.
        """
        body = (
            "# Resource: aws_example\n\n"
            "## Example Usage\n\n"
            "```terraform\n"
            "# Canonical\n"
            "## not a heading either\n"
            'resource "aws_example" "x" {\n  ami = "abc"\n}\n'
            "```\n\n"
            "## Argument Reference\n\n* `ami` - (Required) The AMI.\n"
        )
        chunks = chunk_document(self._doc(body))
        headings = {c.heading_path for c in chunks}
        assert not any("Canonical" in h or "not a heading" in h for h in headings)
        assert any("Argument Reference" in h for h in headings)

    def test_breadcrumb_is_prepended_to_text(self):
        body = (
            "# Resource: aws_example\n\n## Argument Reference\n\n* `ami` - The AMI.\n"
        )
        chunk = next(
            c for c in chunk_document(self._doc(body)) if "Argument" in c.heading_path
        )
        first_line = chunk.text.split("\n", 1)[0]
        assert "aws_example" in first_line
        assert "Argument Reference" in first_line

    def test_breadcrumb_does_not_repeat_the_h1(self):
        body = (
            "# Resource: aws_example\n\n## Argument Reference\n\n* `ami` - The AMI.\n"
        )
        chunk = next(
            c for c in chunk_document(self._doc(body)) if "Argument" in c.heading_path
        )
        first_line = chunk.text.split("\n", 1)[0]
        assert first_line.count("aws_example") == 1

    def test_oversized_sections_are_split(self):
        from terraform_docs_mcp.corpus import MAX_CHUNK_CHARS

        body = "# Resource: aws_example\n\n## Argument Reference\n\n" + ("word " * 2000)
        chunks = chunk_document(self._doc(body))
        assert len(chunks) > 1
        assert all(len(c.text) < MAX_CHUNK_CHARS * 2 for c in chunks)

    def test_document_without_headings_still_chunks(self):
        chunks = chunk_document(self._doc("Just prose, no headings at all.\n"))
        # Summary chunk plus the body.
        assert len(chunks) == 2
        assert any("prose" in c.text for c in chunks)

    def test_first_chunk_is_a_document_summary(self):
        """Every document gets one short synthetic passage standing for it.

        Body sections describe parts of a resource, so a topical query has to
        outrank thousands of argument-level chunks to reach the right page.
        """
        body = (
            "# Resource: aws_example\n\n## Argument Reference\n\n* `ami` - The AMI.\n"
        )
        doc = self._doc(body)
        summary = chunk_document(doc)[0]
        assert summary.ordinal == 0
        assert "aws_example" in summary.text
        assert "EC2" in summary.text
        assert len(summary.text) < 300

    def test_summary_chunk_does_not_repeat_itself(self):
        doc = Document(
            doc_id="aws:guide:x",
            provider=Provider.aws,
            kind=Kind.guide,
            title="Upgrade Guide",
            subcategory=None,
            description="Upgrade Guide",  # guides repeat their title here
            rel_path="guides/x.html.markdown",
            body="# Upgrade Guide\n\ntext\n",
        )
        assert chunk_document(doc)[0].text == "Upgrade Guide"


@requires_submodules
class TestRealCorpus:
    def test_aws_resource_metadata(self):
        docs = {d.doc_id: d for d in iter_documents(PROVIDER_AWS)}
        doc = docs["aws:resource:instance"]
        assert doc.kind == Kind.resource
        assert doc.subcategory and "EC2" in doc.subcategory

    def test_google_backticked_title_resolves(self):
        """Google wraps identifiers in backticks: `google_bigquery_dataset`."""
        docs = {d.doc_id: d for d in iter_documents(PROVIDER_GOOGLE)}
        assert (
            docs["google:datasource:bigquery_dataset"].title
            == "google_bigquery_dataset"
        )

    def test_plain_markdown_suffix_is_picked_up(self):
        """A handful of Google pages use .markdown, not .html.markdown."""
        docs = {d.doc_id: d for d in iter_documents(PROVIDER_GOOGLE)}
        assert "google:datasource:dns_record_set" in docs

    def test_real_page_chunks_without_phantom_headings(self):
        docs = {d.doc_id: d for d in iter_documents(PROVIDER_AWS)}
        chunks = chunk_document(docs["aws:resource:instance"])
        assert len(chunks) > 5
        # The page contains '# Canonical' inside an HCL example.
        assert not any("Canonical" in c.heading_path for c in chunks)
