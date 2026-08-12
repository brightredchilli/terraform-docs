"""Build provenance for ``_data``, and the staleness check Make drives.

The manifest answers two questions that used to have no home:

* *What produced this index?* -- the provider commits, the model, a checksum of
  the source tree, and when it ran.
* *Is it still current?* -- by comparing those same inputs against the tree as
  it is now.

It lives in a file rather than in ``index.sqlite3`` so that Make can use it as
a target: a value inside the database is invisible to the build system.

**Two halves, deliberately separated.** Everything that inspects the source
tree or the submodules is build-time only, because an installed wheel has no
``src/`` and no submodules -- ``PROJECT_ROOT`` there points at whatever happens
to sit above ``site-packages``. The runtime path uses :func:`read` and nothing
else.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ._config import (
    DOCS_DIRNAME,
    DOCUMENTS_INDEX_FILENAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    MODEL_DIRNAME,
    SOURCE_DIR,
    VECTORS_FILENAME,
    IndexUnavailable,
    __version__,
)

if TYPE_CHECKING:  # build-time only; keeps `corpus` off the runtime import path
    from .corpus import ProviderConfig

#: Bumped when the manifest's shape changes. It feeds the fingerprint, so a
#: bump invalidates every existing index -- which is the point: an older
#: manifest cannot be trusted to describe the new format's inputs.
MANIFEST_VERSION = 1

#: Everything `build()` produces. A manifest whose artifacts have been deleted
#: describes a build that no longer exists.
REQUIRED_ARTIFACTS = (
    INDEX_FILENAME,
    VECTORS_FILENAME,
    MODEL_DIRNAME,
    DOCS_DIRNAME,
    DOCUMENTS_INDEX_FILENAME,
)

#: Path within a provider submodule that actually feeds the index. Scoping the
#: dirty check to it keeps unrelated churn in the submodule from forcing a
#: 96-second rebuild.
DOCS_PATHSPEC = "website/docs"

_SKIP_DIRS = frozenset({"_data", "__pycache__"})
_SKIP_SUFFIXES = frozenset({".pyc", ".pyo"})


# ------------------------------------------------------------------- runtime


def read(data_dir: Path) -> dict[str, Any]:
    """Parse the manifest shipped alongside the index.

    The only function here safe to call from an installed package. Raises
    :class:`IndexUnavailable` rather than returning a default, because every
    caller needs the model id and dimension it carries.
    """
    path = data_dir / MANIFEST_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IndexUnavailable(
            f"No build manifest at {path}.\n"
            "The index records what built it in this file; without it the "
            "model and dimension cannot be verified.\n"
            "  In the source tree:  make index\n"
            "  As a dependency:     reinstall a wheel built by `make build`."
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IndexUnavailable(
            f"Build manifest at {path} is unreadable ({exc}). Rebuild with `make index`."
        ) from exc

    version = document.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise IndexUnavailable(
            f"Build manifest at {path} is version {version}, but this build "
            f"reads version {MANIFEST_VERSION}. Rebuild with `make index`."
        )
    return document


# ---------------------------------------------------------------- build-time


def current_inputs(providers: Mapping[str, ProviderConfig]) -> dict[str, Any]:
    """Everything that determines what a build would produce.

    Inputs only. Results of the build -- counts, dimension, timestamp -- are
    recorded separately, since including them would make the fingerprint depend
    on itself.
    """
    from .embed import MODEL_ID  # build-time import; cheap, but keep it local

    return {
        "source_sha256": source_sha256(),
        "model_id": MODEL_ID,
        "providers": {
            name: {
                "commit": _git_sha(_repo_of(config)),
                "dirty": _git_dirty(_repo_of(config)),
            }
            for name, config in sorted(providers.items())
        },
    }


def fingerprint(inputs: Mapping[str, Any]) -> str:
    """Stable digest of an inputs mapping.

    Canonical JSON -- sorted keys, no incidental whitespace -- so that a
    reordered dict does not read as a change.
    """
    payload = json.dumps(
        {"manifest_version": MANIFEST_VERSION, "inputs": inputs},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write(
    data_dir: Path, inputs: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Record a completed build. Call last: this file marks the build done."""
    document = {
        "manifest_version": MANIFEST_VERSION,
        "fingerprint": fingerprint(inputs),
        "inputs": dict(inputs),
        "outputs": {"package_version": __version__, **outputs},
    }
    path = data_dir / MANIFEST_FILENAME
    # Written via a temporary file: a half-written manifest would still look
    # like a completed build to the next `make index`.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return document


def staleness(
    data_dir: Path, providers: Mapping[str, ProviderConfig]
) -> str | None:
    """Why the index needs rebuilding, or ``None`` if it does not.

    A reason rather than a bool, so `make index` can say what changed instead
    of silently spending 96 seconds.
    """
    if not (data_dir / MANIFEST_FILENAME).exists():
        return "no manifest"
    try:
        document = read(data_dir)
    except IndexUnavailable as exc:
        return str(exc).splitlines()[0]

    missing = [name for name in REQUIRED_ARTIFACTS if not (data_dir / name).exists()]
    if missing:
        return f"artifact missing: {', '.join(missing)}"

    current = current_inputs(providers)

    # A dirty submodule is always stale. Its commit still matches, but the
    # working tree no longer does, and `git status` reports *that* files
    # changed, not what they now contain -- so freshness cannot be established.
    # Rebuilding every time is the safe direction, and it stops as soon as the
    # submodule is clean again.
    dirty = [name for name, p in current["providers"].items() if p["dirty"]]
    if dirty:
        return (
            f"{', '.join(dirty)} has uncommitted changes under {DOCS_PATHSPEC} "
            "(cannot verify freshness)"
        )

    # Recomputed from the recorded inputs rather than trusting the stored
    # `fingerprint` field, which is there for humans and `stats`. Comparing
    # against it would let a manifest whose two halves disagree read as fresh.
    recorded = document.get("inputs") or {}
    if fingerprint(current) == fingerprint(recorded):
        return None
    return _explain(recorded, current)


def _explain(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    reasons: list[str] = []
    if recorded.get("model_id") != current["model_id"]:
        reasons.append(
            f"model changed: {recorded.get('model_id')} -> {current['model_id']}"
        )
    was_providers = recorded.get("providers") or {}
    for name, now in current["providers"].items():
        was = was_providers.get(name) or {}
        if was.get("commit") != now["commit"]:
            reasons.append(
                f"{name} moved {_short(was.get('commit'))} -> {_short(now['commit'])}"
            )
    if recorded.get("source_sha256") != current["source_sha256"]:
        reasons.append("source changed")
    # Falls back to a generic reason rather than claiming nothing changed: the
    # fingerprint disagreed, so something did, even if it is a key this
    # function does not know how to describe.
    return "; ".join(reasons) or "inputs changed"


def _short(sha: str | None) -> str:
    return (sha or "unknown")[:8]


# ------------------------------------------------------------ source hashing


def source_sha256() -> str:
    """Digest of every file under ``src/``, excluding generated output.

    Build-time only -- an installed wheel has no ``src/``.

    ``_data`` is skipped because it lives *under* ``src/`` and holds the
    manifest this digest goes into; including it would make the fingerprint
    depend on its own output. ``__pycache__`` is skipped because it changes
    without the source changing.

    The relative path is hashed alongside each file's contents, so renaming a
    module counts as a change even though no bytes moved.
    """
    digest = hashlib.sha256()
    for path in _source_files():
        digest.update(path.relative_to(SOURCE_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_files() -> list[Path]:
    """Hashable files under ``src/``, in a stable order.

    Prunes ``os.walk`` in place rather than filtering afterwards: ``_data``
    holds tens of thousands of files, and descending into it just to discard
    the results would dominate the cost of this check.
    """
    found: list[Path] = []
    for root, dirnames, filenames in os.walk(SOURCE_DIR):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1] in _SKIP_SUFFIXES:
                continue
            found.append(Path(root) / name)
    # os.walk order is filesystem-dependent; the digest must not be.
    return sorted(found)


# --------------------------------------------------------------------- git


def _repo_of(config: ProviderConfig) -> Path:
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


def _git_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD") or "unknown"


def _git_dirty(repo: Path) -> bool:
    """Whether the provider's docs have uncommitted changes.

    ``None`` from git (no submodule checked out, git absent) reads as clean:
    the commit is already ``unknown`` in that case, which is the louder signal.
    """
    status = _git(repo, "status", "--porcelain", "--", DOCS_PATHSPEC)
    return bool(status)


def summary(document: Mapping[str, Any]) -> Iterable[str]:
    """Human-readable lines describing a manifest, for logs and `stats`."""
    inputs = document.get("inputs") or {}
    outputs = document.get("outputs") or {}
    yield f"built {outputs.get('built_at')} by {inputs.get('model_id')}"
    for name, provider in sorted((inputs.get("providers") or {}).items()):
        flag = " (dirty)" if provider.get("dirty") else ""
        yield f"  {name} {_short(provider.get('commit'))}{flag}"
