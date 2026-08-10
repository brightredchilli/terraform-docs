"""End-to-end test driving the server over a real stdio MCP session.

Spawns the server as a subprocess and speaks the protocol to it, so this
catches wiring problems the in-process tests cannot: tool registration, schema
generation, serialization of return values, and the console-script entrypoint.

Uses ``asyncio.run`` inside sync tests rather than an async pytest plugin, to
avoid adding a test-only dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from terraform_docs_mcp._config import data_dir

pytestmark = pytest.mark.skipif(
    not (data_dir() / "index.sqlite3").exists(),
    reason="no packaged index; run `make index`",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def _call(tool: str, arguments: dict):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "terraform_docs_mcp.server"],
        cwd=str(PROJECT_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(tool, arguments)
            return tools, result


def _run(tool: str, arguments: dict):
    return asyncio.run(asyncio.wait_for(_call(tool, arguments), timeout=120))


def _text_of(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")


def _rows_of(result) -> list[dict]:
    """Decode a list-returning tool result.

    The server emits one JSON content block per element rather than a single
    JSON array, so each block is parsed independently.
    """
    return [json.loads(c.text) for c in result.content if getattr(c, "type", None) == "text"]


class TestOverStdio:
    def test_tools_are_advertised(self):
        tools, _ = _run("search", {"query": "aws_instance", "limit": 1})
        names = {t.name for t in tools.tools}
        assert names == {"search", "get_document"}

    def test_search_returns_documents(self):
        _, result = _run("search", {"query": "s3 bucket lifecycle expiration", "limit": 5})
        assert not result.is_error
        rows = _rows_of(result)
        assert rows, "no results returned over MCP"
        assert any("s3_bucket_lifecycle" in r["doc_id"] for r in rows)
        assert {"doc_id", "provider", "kind", "snippet", "score"} <= set(rows[0])

    def test_search_honours_provider_filter(self):
        _, result = _run("search", {"query": "instance", "provider": "google", "limit": 5})
        assert not result.is_error
        assert {r["provider"] for r in _rows_of(result)} == {"google"}

    def test_get_document_section(self):
        _, result = _run(
            "get_document", {"doc_id": "aws:r/instance", "section": "Argument Reference"}
        )
        assert not result.is_error
        assert _text_of(result).startswith("## Argument Reference")

    def test_unknown_document_is_a_tool_error(self):
        _, result = _run("get_document", {"doc_id": "aws:r/nope"})
        assert result.is_error
        assert "nope" in _text_of(result)
