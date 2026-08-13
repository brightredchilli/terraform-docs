"""Build provenance for ``_data``: which provider commit each build stage last
ran against.

Deliberately minimal: one flat dict, updated incrementally as each stage of
``build_index.py`` completes, not written once at the end of a single build.
``documents_aws``/``documents_google`` record the commit ``documents.sqlite3``
was last (re)built from; ``summaries_aws``/``summaries_google`` record the
commit ``src/summaries/`` was last confirmed current for. Each stage compares
its own pair against the provider's current commit (``util.git.sha``) to
decide whether it has anything to do.

A single ``Manifest`` is meant to be read once at the start of a build, then
mutated and saved as each stage completes, rather than re-read from disk on
every comparison.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Manifest:
    """Parsed contents of a manifest file, read once and kept in memory.

    Mutable on purpose: a build reads one of these at the start, sets fields
    as its stages complete, and calls :meth:`save` after each one -- so a
    crash between stages leaves a manifest that correctly describes partial
    progress rather than none at all.
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Missing or corrupt both read as ``{}`` rather than raising.

        There is no single "build complete" marker to protect -- every caller
        compares one specific key, and a key that was never written compares
        unequal to any real commit SHA, which is already the correct
        "needs building" answer.
        """
        try:
            return json.loads(self.filepath.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return {}

    @property
    def documents_aws_commit_sha(self) -> str | None:
        return self.data.get("documents_aws")

    @documents_aws_commit_sha.setter
    def documents_aws_commit_sha(self, value: str) -> None:
        self.data["documents_aws"] = value

    @property
    def documents_google_commit_sha(self) -> str | None:
        return self.data.get("documents_google")

    @documents_google_commit_sha.setter
    def documents_google_commit_sha(self, value: str) -> None:
        self.data["documents_google"] = value

    @property
    def summaries_aws_commit_sha(self) -> str | None:
        return self.data.get("summaries_aws")

    @summaries_aws_commit_sha.setter
    def summaries_aws_commit_sha(self, value: str) -> None:
        self.data["summaries_aws"] = value

    @property
    def summaries_google_commit_sha(self) -> str | None:
        return self.data.get("summaries_google")

    @summaries_google_commit_sha.setter
    def summaries_google_commit_sha(self, value: str) -> None:
        self.data["summaries_google"] = value

    def save(self) -> None:
        """Write the current state to :attr:`filepath`."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temporary file: a half-written manifest would still
        # look like a valid (if stale) one to the next read.
        tmp = self.filepath.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self.filepath)
