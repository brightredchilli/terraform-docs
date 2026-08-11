"""MCP server exposing search over the packaged Terraform provider docs.

The server holds no per-request state: the index is read-only and every tool
call is a pure function of its arguments. That makes it safe to serve calls
concurrently and lets the HTTP transport run in stateless mode, where each
request is self-contained and no session is tracked.
"""

from __future__ import annotations

from enum import StrEnum
import threading
from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field

from terraform_docs_mcp.corpus import Kind, Provider
from terraform_docs_mcp.util import all_values

from ._config import __version__
from .index import Index, IndexUnavailable

INSTRUCTIONS = """\
Searches the official Terraform provider documentation for AWS and Google Cloud
(resources, data sources, guides, functions).

Use `search` to find the relevant documentation pages, then `get_document` to
read one. Prefer requesting a specific `section` (for example "Argument
Reference") over fetching a whole page, since the largest pages are very long.
"""


class Transport(StrEnum):
    """How the MCP server talks to its client."""

    stdio = "stdio"
    http = "http"


mcp = MCPServer(
    name="terraform-docs",
    version=__version__,
    instructions=INSTRUCTIONS,
)

_index: Index | None = None
_index_lock = threading.Lock()

#: Guards against a caller requesting an unreasonable page of results.
MAX_LIMIT = 50


def get_index() -> Index:
    """Return the process-wide index, loading it on first use.

    Stateless request handling does not mean re-loading per call: loading the
    embedding model costs seconds, so it is built once and shared. Both it and
    the vector array are read-only and safe to use from several threads.
    """
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = Index()
    return _index


@mcp.tool()
def terraform_mcp_search(
    query: Annotated[
        str, Field(description="Natural-language question or Terraform identifier.")
    ],
    provider: Annotated[
        str | None,
        Field(
            description=f"Restrict results to one provider. Optional. Supported values: {all_values(Provider)}"
        ),
    ] = None,
    kind: Annotated[
        str | None,
        Field(
            description=f"Restrict to a document kind. Optional. Values: {all_values(Kind)}"
        ),
    ] = None,
    limit: Annotated[int, Field(description="Maximum documents to return.")] = 10,
) -> list[dict[str, Any]]:
    """Search Terraform provider documentation and return matching documents.

    Combines BM25 keyword matching with vector similarity, so both exact
    identifiers (`aws_s3_bucket lifecycle`) and paraphrases ("how do I delete
    old objects in a bucket automatically") work.

    Each result carries a `doc_id` to pass to `get_document`.
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    return get_index().search(query, provider=provider, kind=kind, limit=limit)


@mcp.tool()
def terraform_mcp_get_document(
    doc_id: Annotated[
        str,
        Field(description="Document id from `search`, e.g. 'aws:resource:instance'."),
    ],
) -> str:
    """Return the markdown for one documentation page"""
    try:
        return get_index().get_document(doc_id)
    except KeyError as exc:
        # KeyError stringifies with quotes; unwrap for a readable tool error.
        raise ValueError(str(exc.args[0]) if exc.args else str(exc)) from exc


def serve(
    transport: Transport = Transport.stdio, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """Run the server until the transport closes.

    Command-line parsing lives in :mod:`terraform_docs_mcp.cli`; this is the
    plain function underneath it. Raises :class:`IndexUnavailable` if the
    packaged index is missing -- deliberately before any client handshake, so a
    broken install fails visibly at startup rather than on the first search.
    """
    get_index()

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Stateless: no session id, no persistent server->client stream, so
        # requests can be spread across replicas. We give up sampling,
        # elicitation and progress notifications, none of which these
        # read-only tools use.
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            stateless_http=True,
        )
