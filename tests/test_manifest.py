"""Tests for build provenance and the staleness check `make index` drives.

None of these run a real build. The source tree and git are both stubbed, so
the logic can be exercised in milliseconds rather than 96 seconds -- which is
the whole point of the manifest existing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terraform_docs_mcp import manifest
from terraform_docs_mcp._config import MANIFEST_FILENAME, IndexUnavailable
from terraform_docs_mcp.corpus import PROVIDERS


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for ``src/`` that tests can edit."""
    root = tmp_path / "src" / "pkg"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("x = 1\n")
    (root / "index.py").write_text("y = 2\n")
    (root / "py.typed").write_text("")
    monkeypatch.setattr(manifest, "SOURCE_DIR", tmp_path / "src")
    return root


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A ``_data`` directory holding every artifact a build produces."""
    root = tmp_path / "_data"
    root.mkdir()
    for name in manifest.REQUIRED_ARTIFACTS:
        target = root / name
        if "." in name:
            target.write_bytes(b"")
        else:
            target.mkdir()
    return root


class FakeGit:
    """Stubbed submodule state, so tests need no real repositories."""

    def __init__(self) -> None:
        self.shas = {"aws": "a" * 40, "google": "b" * 40}
        self.dirty: set[str] = set()

    def _provider(self, repo: Path) -> str:
        return repo.name.removeprefix("terraform-provider-")

    def sha(self, repo: Path) -> str:
        return self.shas[self._provider(repo)]

    def is_dirty(self, repo: Path) -> bool:
        return self._provider(repo) in self.dirty


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> FakeGit:
    fake = FakeGit()
    monkeypatch.setattr(manifest, "_git_sha", fake.sha)
    monkeypatch.setattr(manifest, "_git_dirty", fake.is_dirty)
    return fake


def _seed(data_dir: Path) -> dict:
    """Write a manifest describing the tree as it currently stands."""
    return manifest.write(
        data_dir,
        manifest.current_inputs(PROVIDERS),
        {"built_at": "2026-01-01T00:00:00Z", "dim": 384},
    )


class TestSourceChecksum:
    def test_is_stable(self, source_tree: Path):
        assert manifest.source_sha256() == manifest.source_sha256()

    def test_changes_when_a_file_changes(self, source_tree: Path):
        before = manifest.source_sha256()
        (source_tree / "index.py").write_text("y = 3\n")
        assert manifest.source_sha256() != before

    def test_changes_when_a_file_is_renamed(self, source_tree: Path):
        """The path is hashed alongside the bytes, so a pure rename counts."""
        before = manifest.source_sha256()
        (source_tree / "index.py").rename(source_tree / "retrieval.py")
        assert manifest.source_sha256() != before

    def test_covers_non_python_files(self, source_tree: Path):
        before = manifest.source_sha256()
        (source_tree / "py.typed").write_text("# now non-empty\n")
        assert manifest.source_sha256() != before

    def test_ignores_generated_data(self, source_tree: Path):
        """`_data` lives under src/ and holds the manifest this digest goes into.

        Including it would make the fingerprint depend on its own output.
        """
        before = manifest.source_sha256()
        generated = source_tree / "_data"
        generated.mkdir()
        (generated / "index.sqlite3").write_bytes(b"\x00" * 1024)
        (generated / MANIFEST_FILENAME).write_text("{}")
        assert manifest.source_sha256() == before

    def test_ignores_bytecode(self, source_tree: Path):
        before = manifest.source_sha256()
        cache = source_tree / "__pycache__"
        cache.mkdir()
        (cache / "index.cpython-314.pyc").write_bytes(b"\x00")
        (source_tree / "stray.pyc").write_bytes(b"\x00")
        assert manifest.source_sha256() == before


class TestFingerprint:
    def test_ignores_key_order(self):
        a = {"model_id": "m", "source_sha256": "s"}
        b = {"source_sha256": "s", "model_id": "m"}
        assert manifest.fingerprint(a) == manifest.fingerprint(b)

    def test_excludes_build_outputs(self, source_tree: Path, data_dir: Path, git: FakeGit):
        """Otherwise every build would look stale the moment it finished.

        `built_at` differs between any two builds, so if it fed the
        fingerprint, no index could ever compare equal to its own inputs.
        """
        first = manifest.write(
            data_dir, manifest.current_inputs(PROVIDERS), {"built_at": "2026-01-01T00:00:00Z"}
        )
        second = manifest.write(
            data_dir, manifest.current_inputs(PROVIDERS), {"built_at": "2027-06-06T12:00:00Z"}
        )
        assert first["fingerprint"] == second["fingerprint"]


class TestRead:
    def test_missing_manifest_names_the_fix(self, data_dir: Path):
        with pytest.raises(IndexUnavailable, match="make index"):
            manifest.read(data_dir)

    def test_rejects_a_future_version(self, data_dir: Path):
        (data_dir / MANIFEST_FILENAME).write_text(
            json.dumps({"manifest_version": manifest.MANIFEST_VERSION + 1})
        )
        with pytest.raises(IndexUnavailable, match="version"):
            manifest.read(data_dir)

    def test_rejects_corruption(self, data_dir: Path):
        """A half-written manifest must not read as a completed build."""
        (data_dir / MANIFEST_FILENAME).write_text('{"manifest_version": 1, "inp')
        with pytest.raises(IndexUnavailable, match="unreadable"):
            manifest.read(data_dir)


class TestStaleness:
    def test_fresh_after_writing(self, source_tree: Path, data_dir: Path, git: FakeGit):
        _seed(data_dir)
        assert manifest.staleness(data_dir, PROVIDERS) is None

    def test_no_manifest(self, source_tree: Path, data_dir: Path, git: FakeGit):
        assert manifest.staleness(data_dir, PROVIDERS) == "no manifest"

    def test_source_changed(self, source_tree: Path, data_dir: Path, git: FakeGit):
        _seed(data_dir)
        (source_tree / "index.py").write_text("y = 99\n")
        assert manifest.staleness(data_dir, PROVIDERS) == "source changed"

    def test_provider_moved(self, source_tree: Path, data_dir: Path, git: FakeGit):
        _seed(data_dir)
        git.shas["aws"] = "c" * 40
        reason = manifest.staleness(data_dir, PROVIDERS)
        assert reason is not None
        assert reason.startswith("aws moved aaaaaaaa -> cccccccc")

    def test_dirty_provider_is_always_stale(
        self, source_tree: Path, data_dir: Path, git: FakeGit
    ):
        """A dirty submodule cannot be shown fresh, so it rebuilds every time.

        `git status` reports which files changed, not what they now contain, so
        an index built from a dirty tree cannot be matched against it later.
        """
        _seed(data_dir)
        git.dirty.add("google")
        reason = manifest.staleness(data_dir, PROVIDERS)
        assert reason is not None
        assert "google has uncommitted changes" in reason

    def test_model_changed(self, source_tree: Path, data_dir: Path, git: FakeGit):
        document = _seed(data_dir)
        document["inputs"]["model_id"] = "some/other-model"
        (data_dir / MANIFEST_FILENAME).write_text(json.dumps(document))
        reason = manifest.staleness(data_dir, PROVIDERS)
        assert reason is not None
        assert reason.startswith("model changed: some/other-model ->")

    @pytest.mark.parametrize("artifact", manifest.REQUIRED_ARTIFACTS)
    def test_deleted_artifact(
        self, source_tree: Path, data_dir: Path, git: FakeGit, artifact: str
    ):
        """A manifest whose artifacts are gone describes a build that isn't there."""
        _seed(data_dir)
        target = data_dir / artifact
        target.rmdir() if target.is_dir() else target.unlink()
        assert manifest.staleness(data_dir, PROVIDERS) == f"artifact missing: {artifact}"


class TestIndexRequiresManifest:
    def test_missing_manifest_fails_at_construction(self, tmp_path: Path):
        """Not on first search: `serve()` builds an Index at startup so that a
        broken install fails before an MCP client ever connects."""
        from terraform_docs_mcp.index import INDEX_FILENAME, Index

        (tmp_path / INDEX_FILENAME).write_bytes(b"")
        with pytest.raises(IndexUnavailable, match="manifest"):
            Index(data_dir=tmp_path)
