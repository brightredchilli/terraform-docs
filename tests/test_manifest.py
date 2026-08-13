"""Tests for the build-provenance manifest.

manifest.json is now a flat dict of provider commit SHAs, updated
incrementally by each build_index.py stage -- see manifest.py.
"""

from __future__ import annotations

from pathlib import Path

from terraform_docs_mcp.manifest import Manifest


class TestManifest:
    def test_missing_manifest_reads_as_empty(self, tmp_path: Path):
        m = Manifest.read(tmp_path)
        assert m.documents_aws_commit_sha is None
        assert m.summaries_google_commit_sha is None

    def test_update_writes_and_reads_back(self, tmp_path: Path):
        Manifest.update(tmp_path, documents_aws="abc123", documents_google="def456")
        m = Manifest.read(tmp_path)
        assert m.documents_aws_commit_sha == "abc123"
        assert m.documents_google_commit_sha == "def456"

    def test_update_is_incremental_not_a_replace(self, tmp_path: Path):
        """Each build stage writes its own keys; an earlier stage's keys must
        survive a later stage's update."""
        Manifest.update(tmp_path, documents_aws="abc123", documents_google="def456")
        Manifest.update(tmp_path, summaries_aws="ghi789", summaries_google="jkl012")
        m = Manifest.read(tmp_path)
        assert m.documents_aws_commit_sha == "abc123"
        assert m.documents_google_commit_sha == "def456"
        assert m.summaries_aws_commit_sha == "ghi789"
        assert m.summaries_google_commit_sha == "jkl012"
