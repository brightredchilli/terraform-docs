"""Offline hybrid search over Terraform provider documentation."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__version__ = "0.1.0"

#: Providers indexed by this tool, mapped to their submodule directory name.
PROVIDERS: dict[str, str] = {
    "aws": "terraform-provider-aws",
    "google": "terraform-provider-google",
}

#: Path within each provider repo holding the user-facing registry docs.
DOCS_SUBPATH = "website/docs"

#: Glob matching provider documentation files.
DOC_GLOB = "*.markdown"

#: Suffixes documentation files carry, longest first. Most use
#: ``.html.markdown``, but a handful of Google data-source pages use plain
#: ``.markdown``.
DOC_SUFFIXES = (".html.markdown", ".markdown")


def data_dir() -> Path:
    """Locate the packaged ``_data`` directory.

    Wheels install unpacked, so this resolves to a real directory on disk and
    the vector array can be memory-mapped without copying.
    """
    return Path(str(files(__package__) / "_data"))
