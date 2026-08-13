"""Shelling out to git for a repo's current commit SHA.

Build-time only -- needs a real git checkout, which does not exist in an
installed wheel. build_index.py is the only caller.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    """The commit SHA of the git repo containing ``path``.

    ``path`` need not be a repo root -- checked, not assumed:
    ``git -C <subdirectory> rev-parse HEAD`` returns the exact same SHA as
    running it at the repo root, since git resolves the enclosing repository
    itself. Callers can point this directly at whatever directory they
    actually care about.

    Returns ``"unknown"`` if the SHA cannot be determined (no git checkout,
    ``git`` not on PATH, ...) rather than raising -- a submodule directory
    existing without git metadata is a real, recoverable situation upstream
    callers already handle by treating an "unknown" commit as always stale.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return out.stdout.strip()
