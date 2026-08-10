"""Constants and paths shared by every module.

Kept separate from ``__init__`` so that the package root can re-export the
public API (``Index``) without a circular import: ``index`` needs ``data_dir``,
and the root needs ``index``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__version__ = "0.1.0"


#: Glob matching provider documentation files.
DOC_GLOB = "*.markdown"

#: Suffixes documentation files carry, longest first. Most use
#: ``.html.markdown``, but a handful of Google data-source pages use plain
#: ``.markdown``.
DOC_SUFFIXES = (".html.markdown", ".markdown")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "terraform_docs_mcp" / "_data"


def data_dir() -> Path:
    """Locate the packaged ``_data`` directory.

    Wheels install unpacked, so this resolves to a real directory on disk and
    the vector array can be memory-mapped without copying.
    """
    return Path(str(files(__package__) / "_data"))
