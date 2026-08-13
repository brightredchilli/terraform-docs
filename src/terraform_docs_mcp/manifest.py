"""Build provenance for ``_data``: which provider commit each build stage last
ran against.

Deliberately minimal: one flat dict, updated incrementally as each stage of
``build_index.py`` completes, not written once at the end of a single build.
``documents_aws``/``documents_google`` record the commit ``documents.sqlite3``
was last (re)built from; ``summaries_aws``/``summaries_google`` record the
commit ``src/summaries/`` was last confirmed current for. Each stage compares
its own pair against the provider's current commit to decide whether it has
anything to do.

Build-time only: computing a *current* commit SHA needs ``git`` and the
submodule checkouts, neither of which exist in an installed wheel. Nothing at
runtime needs this module at all -- the packaged artifacts it describes
(``documents.sqlite3``, and eventually the vector index) carry no
runtime-checked provenance of their own right now.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._config import MANIFEST_FILENAME

if TYPE_CHECKING:  # build-time only; keeps `corpus` off the runtime import path
    from .corpus import ProviderConfig

#: Path within a provider submodule that actually feeds the index. Scoping the
#: SHA lookup to it would matter for a dirty-tree check; this module no longer
#: does one -- see Db/build_index.py for why that was dropped.
DOCS_PATHSPEC = "website/docs"


@dataclass(frozen=True)
class Manifest:
    """Parsed contents of ``manifest.json``.

    A thin, read-only view over the flat dict :meth:`update` persists. The
    only ways to get one are :meth:`read` and :meth:`update`.
    """

    data: dict[str, Any]

    @property
    def documents_aws_commit_sha(self) -> str | None:
        return self.data.get("documents_aws")

    @property
    def documents_google_commit_sha(self) -> str | None:
        return self.data.get("documents_google")

    @property
    def summaries_aws_commit_sha(self) -> str | None:
        return self.data.get("summaries_aws")

    @property
    def summaries_google_commit_sha(self) -> str | None:
        return self.data.get("summaries_google")

    @classmethod
    def read(cls, data_dir: Path) -> Manifest:
        """Parse the manifest, or an empty one if there is none yet.

        Missing or corrupt both read as ``Manifest({})`` rather than raising.
        There is no single "build complete" marker to protect anymore --
        every caller compares one specific key, and a key that has never been
        written compares unequal to any real commit SHA, which is already the
        correct "needs building" answer.
        """
        path = data_dir / MANIFEST_FILENAME
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return cls({})

    @classmethod
    def update(cls, data_dir: Path, **fields: str) -> Manifest:
        """Merge ``fields`` into the manifest on disk and write it back.

        Incremental, not atomic-at-the-end-of-a-build: each stage calls this
        once it finishes, so a crash between stages leaves a manifest that
        correctly describes partial progress rather than none at all.
        """
        data = dict(cls.read(data_dir).data)
        data.update(fields)
        path = data_dir / MANIFEST_FILENAME
        data_dir.mkdir(parents=True, exist_ok=True)
        # Written via a temporary file: a half-written manifest would still
        # look like a valid (if stale) one to the next read.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
        return cls(data)


def repo_of(config: ProviderConfig) -> Path:
    """Submodule root for a provider.

    ``source_docs_dir`` is repo-relative (``terraform-provider-aws/website/docs``);
    its first component is the submodule.
    """
    from ._config import PROJECT_ROOT

    return PROJECT_ROOT / config.source_docs_dir.parts[0]


def _git(repo: Path, *args: str) -> str | None:
    """Run git in ``repo``, or ``None`` if it cannot be run at all."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return out.stdout.strip()


def git_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD") or "unknown"
