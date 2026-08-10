"""Offline hybrid search over Terraform provider documentation.

Usable as a library as well as an MCP server:

    from terraform_docs_mcp import Index

    index = Index()
    for hit in index.search("s3 bucket lifecycle", provider="aws", limit=5):
        print(hit["doc_id"], hit["snippet"])
    print(index.get_document(hit["doc_id"], section="Argument Reference"))

``Index`` is safe to share across threads and holds no per-request state, but
the first search loads an embedding model -- build one per process and reuse it.

This module is a facade over the public API and nothing else. Shared constants
and paths live in ``_config`` so that submodules never import from the package
root, which would make the root unable to re-export ``Index``.
"""

from __future__ import annotations

from ._config import __version__
from .index import Index, IndexUnavailable

__all__ = ["Index", "IndexUnavailable", "__version__"]
