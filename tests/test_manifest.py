"""Tests for the build-provenance manifest.

manifest.json is now a flat dict of provider commit SHAs. A Manifest is read
once (at construction, given the full path to the file) and mutated/saved in
place as a build progresses -- see manifest.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terraform_docs_mcp.manifest import Manifest


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.json"


class TestManifest:
    def test_missing_manifest_reads_as_empty(self, path: Path):
        m = Manifest(path)
        assert m.documents_aws_commit_sha is None
        assert m.summaries_google_commit_sha is None

    def test_filepath_is_exactly_what_was_passed_in(self, path: Path):
        assert Manifest(path).filepath == path

    def test_save_writes_and_a_fresh_read_sees_it(self, path: Path):
        m = Manifest(path)
        m.documents_aws_commit_sha = "abc123"
        m.documents_google_commit_sha = "def456"
        m.save()

        reloaded = Manifest(path)
        assert reloaded.documents_aws_commit_sha == "abc123"
        assert reloaded.documents_google_commit_sha == "def456"

    def test_save_creates_missing_parent_directories(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "manifest.json"
        m = Manifest(nested)
        m.documents_aws_commit_sha = "abc123"
        m.save()
        assert nested.exists()

    def test_updates_are_incremental_not_a_replace(self, path: Path):
        """Each build stage sets its own fields on the same in-memory Manifest
        and saves; an earlier stage's fields must survive a later stage's
        save, and a later save must see what an earlier one wrote."""
        m = Manifest(path)
        m.documents_aws_commit_sha = "abc123"
        m.documents_google_commit_sha = "def456"
        m.save()

        m.summaries_aws_commit_sha = "ghi789"
        m.summaries_google_commit_sha = "jkl012"
        m.save()

        reloaded = Manifest(path)
        assert reloaded.documents_aws_commit_sha == "abc123"
        assert reloaded.documents_google_commit_sha == "def456"
        assert reloaded.summaries_aws_commit_sha == "ghi789"
        assert reloaded.summaries_google_commit_sha == "jkl012"

    def test_a_fresh_manifest_picks_up_what_another_instance_saved(self, path: Path):
        """Simulates two separate build_index.py invocations (e.g. `index`
        then a later `summaries` run): each constructs its own Manifest, but
        must still see the other's saved fields."""
        first = Manifest(path)
        first.documents_aws_commit_sha = "abc123"
        first.save()

        second = Manifest(path)
        second.summaries_aws_commit_sha = "ghi789"
        second.save()

        assert Manifest(path).documents_aws_commit_sha == "abc123"
        assert Manifest(path).summaries_aws_commit_sha == "ghi789"
