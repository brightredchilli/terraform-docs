"""Terminal front-end for the same search the MCP server exposes.

Exists so retrieval can be exercised and tuned without wiring up an MCP client.
"""

import dataclasses
import json
from enum import StrEnum
from pathlib import Path
import sys
from typing import Annotated

import typer

from terraform_docs_mcp._config import DOCUMENTS_DB_FILENAME, PROJECT_ROOT
from terraform_docs_mcp._config import data_dir as _data_dir
from terraform_docs_mcp.corpus import Kind, Provider
from terraform_docs_mcp.util.strenum import all_values
from terraform_docs_mcp.util.handle_broken_pipe import handle_broken_pipe

from .db import Db
from .index import Index, IndexUnavailable
from .server import Transport, serve as _serve

app = typer.Typer(no_args_is_help=True)


@handle_broken_pipe
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


def _get_documents_db() -> Db:
    path = _data_dir() / DOCUMENTS_DB_FILENAME
    if not path.exists():
        print(f"error: no document index at {path}. Run `make index`.", file=sys.stderr)
        raise typer.Exit(2)
    return Db(path, readonly=True)


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
    db: Annotated[
        bool,
        typer.Option(
            "--db",
            help=(
                "Text search only (trigram full-text over document headings "
                "and bodies). Skips the embedding model entirely -- no torch "
                "import, no hybrid fusion."
            ),
        ),
    ] = False,
):
    """
    Search documents.

    """
    if db:
        database = _get_documents_db()
        results = database.search(query, provider=provider, kind=kind, limit=limit)
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
        return

    index = _get_index()
    results = index.search(query, provider=provider, kind=kind, limit=limit)
    print(json.dumps(results, indent=2))


@app.command()
def serve(
    transport: Annotated[
        Transport,
        typer.Option(help="stdio for a local MCP client; http for Streamable HTTP."),
    ] = Transport.stdio,
    host: Annotated[
        str, typer.Option(help="Bind address, http transport only.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port, http transport only.")] = 8000,
):
    """
    Run the MCP server.

    stdio is what a local MCP client launches; http serves the same tools in
    stateless mode so requests can be spread across replicas.
    """
    try:
        _serve(transport=transport, host=host, port=port)
    except IndexUnavailable as exc:
        # stderr, never stdout: on the stdio transport anything written to
        # stdout that is not JSON-RPC corrupts the protocol stream.
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(2)


@handle_broken_pipe
def serve_main() -> int:
    """Entry point for the ``terraform-docs-mcp`` alias.

    Kept so MCP client configurations naming that executable keep working.
    ``typer.run`` builds a one-command app around ``serve``, so the flags are
    identical to ``terraform-docs serve``.
    """
    try:
        typer.run(serve)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
