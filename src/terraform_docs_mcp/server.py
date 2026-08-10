"""MCP server exposing search over the packaged Terraform provider docs.

The server holds no per-request state: the index is read-only and every tool
call is a pure function of its arguments. That makes it safe to serve calls
concurrently and lets the HTTP transport run in stateless mode, where each
request is self-contained and no session is tracked.
"""

from __future__ import annotations

import argparse
import threading
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import Field

from ._config import __version__
from .index import Index, IndexUnavailable

INSTRUCTIONS = """\
Searches the official Terraform provider documentation for AWS and Google Cloud
(resources, data sources, guides, functions).

Use `search` to find the relevant documentation pages, then `get_document` to
read one. Prefer requesting a specific `section` (for example "Argument
Reference") over fetching a whole page, since the largest pages are very long.
"""

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
def search(
    query: Annotated[
        str, Field(description="Natural-language question or Terraform identifier.")
    ],
    provider: Annotated[
        Literal["aws", "google"] | None,
        Field(description="Restrict results to one provider."),
    ] = None,
    kind: Annotated[
        str | None,
        Field(
            description=(
                "Restrict to a document kind: 'r' (resources), 'd' (data sources), "
                "'guides', 'functions', 'ephemeral-resources', 'list-resources', "
                "'actions'."
            )
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
def get_document(
    doc_id: Annotated[str, Field(description="Document id from `search`, e.g. 'aws:r/instance'.")],
    section: Annotated[
        str | None,
        Field(
            description=(
                "Return only this top-level section, e.g. 'Argument Reference', "
                "'Example Usage', 'Attribute Reference', 'Import'. Omit for the "
                "whole page."
            )
        ),
    ] = None,
) -> str:
    """Return the markdown for one documentation page, or a single section.

    Large resource pages run to tens of thousands of tokens, so prefer passing
    `section` when you know which part you need.
    """
    try:
        return get_index().get_document(doc_id, section=section)
    except KeyError as exc:
        # KeyError stringifies with quotes; unwrap for a readable tool error.
        raise ValueError(str(exc.args[0]) if exc.args else str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="stdio (default) for a local MCP client; http to serve over Streamable HTTP.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    try:
        get_index()
    except IndexUnavailable as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Stateless: no session id, no persistent server->client stream, so
        # requests can be spread across replicas. We give up sampling,
        # elicitation and progress notifications, none of which these
        # read-only tools use.
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            stateless_http=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
