"""Terminal front-end for the same search the MCP server exposes.

Exists so retrieval can be exercised and tuned without wiring up an MCP client.
"""

import argparse
from enum import Enum, StrEnum
import json
import sys
from typing import Annotated, Literal

import typer

from terraform_docs_mcp.corpus import Kind, Provider
from terraform_docs_mcp.util import all_values, handle_broken_pipe

from .index import Index, IndexUnavailable

app = typer.Typer(no_args_is_help=True)


# @handle_broken_pipe
def main() -> int:
    try:
        app()
    except KeyboardInterrupt:
        return 130

    return 0


def _get_index() -> Index:
    try:
        return Index()
    except IndexUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(2)


@app.command()
def stats():
    index = _get_index()
    print(json.dumps(index.stats(), indent=2, sort_keys=True))


@app.command()
def get(doc_id: Annotated[str, typer.Argument()]):
    index = _get_index()
    try:
        print(index.get_document(doc_id))
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        raise typer.Exit(1)
    return 0


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search term")],
    provider: Annotated[
        Provider | None,
        typer.Option(help=f"Only search this provider: {all_values(Provider)}"),
    ] = None,
    kind: Annotated[
        Kind | None,
        typer.Option(
            help=f"Only search this type of documentation: {all_values(Kind)}"
        ),
    ] = None,
    limit: Annotated[
        int, typer.Option(max=100, help="Limit search results. Default: 10")
    ] = 10,
):
    """
    Search documents.

    """
    index = _get_index()

    query = " ".join(query).strip()

    results = index.search(query, provider=provider, kind=kind, limit=limit)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
