"""Standalone diagnostic for the MCP server over stdio.

Speaks raw newline-delimited JSON-RPC to a freshly spawned server, so it
exercises the same path an MCP client uses without depending on the client
library agreeing with the server. Point it at exactly the command your client
launches:

    terraform-docs-probe                                  # this package's server
    terraform-docs-probe -- /path/to/terraform-docs-mcp   # a specific binary
    terraform-docs-probe --command "uv run terraform-docs-mcp"

It reports each protocol phase separately, because "boots and lists tools but
calls fail" is a distinct failure from "never starts": tool discovery is
answered from static metadata, while the first search additionally loads the
embedding model and touches the index.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shlex
import subprocess
import sys
import time
from typing import Any

PROTOCOL_VERSION = "2025-06-18"

#: Generous by default: the first search loads the embedding model, which is
#: seconds rather than milliseconds. A client whose tool-call timeout is
#: shorter than this will fail on the first call and succeed afterwards.
DEFAULT_TIMEOUT = 60.0

_OK = "  ok  "
_FAIL = " FAIL "


class ProbeFailure(RuntimeError):
    """A phase of the probe failed; the message is already user-facing."""


class Server:
    """A spawned MCP server, spoken to over raw JSON-RPC."""

    def __init__(self, command: list[str], cwd: str | None, timeout: float) -> None:
        self.command = command
        self.timeout = timeout
        self._next_id = 0
        # stderr is kept separate and surfaced on failure: a server that
        # crashes mid-call usually explains itself there, and stdio clients
        # routinely swallow it.
        # Binary pipes, so reads can be polled with a real deadline. A text
        # readline() blocks uninterruptibly, which would make this tool hang on
        # exactly the situation it exists to diagnose.
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        self.junk: list[str] = []
        self._buffer = b""
        self._selector = selectors.DefaultSelector()
        assert self.proc.stdout is not None
        self._selector.register(self.proc.stdout, selectors.EVENT_READ)

    def _send(self, payload: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        )
        return self._read_response(request_id)

    def _readline(self, deadline: float) -> bytes | None:
        """One line from the server's stdout, or ``None`` if the deadline passes.

        Polled rather than blocking, so a wedged server produces a timeout
        report instead of hanging this process too.
        """
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line, self._buffer = self._buffer[:newline], self._buffer[newline + 1 :]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if not self._selector.select(timeout=min(remaining, 0.5)):
                continue
            assert self.proc.stdout is not None
            # os.read on the raw fd: the selector already told us data is
            # ready, and this avoids the buffered reader's blocking semantics.
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:  # EOF: the server closed stdout
                if self._buffer:
                    line, self._buffer = self._buffer, b""
                    return line
                raise ProbeFailure(
                    "server closed stdout without replying (it exited or crashed) — "
                    "see its stderr below"
                )
            self._buffer += chunk

    def _read_response(self, request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            raw = self._readline(deadline)
            if raw is None:
                raise ProbeFailure(
                    f"timed out after {self.timeout:.0f}s waiting for a reply to "
                    f"request {request_id}. If a real client shows the same, raise "
                    "its tool-call timeout: the first search loads the embedding "
                    "model."
                )
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # Anything non-JSON on stdout corrupts the protocol stream.
                # This is the classic stdio failure: a stray print() or a
                # progress bar written to stdout rather than stderr.
                self.junk.append(line)
                continue
            if message.get("id") == request_id:
                return message
            # Notifications and replies to other ids are not our concern here.

    def drain_stderr(self) -> str:
        self.proc.kill()
        try:
            _, err = self.proc.communicate(timeout=5)
        except Exception:  # pragma: no cover - best effort on teardown
            return ""
        finally:
            self._selector.close()
        return (err or b"").decode("utf-8", "replace")


def _report(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{_OK if ok else _FAIL}] {label}" + (f"  {detail}" if detail else ""))


def _tool_names(listed: dict[str, Any]) -> list[str]:
    return [t.get("name", "?") for t in listed.get("result", {}).get("tools", [])]


def probe(
    command: list[str],
    cwd: str | None,
    tool: str | None,
    arguments: dict[str, Any],
    timeout: float,
) -> int:
    print(f"launching: {' '.join(shlex.quote(c) for c in command)}")
    print(f"cwd:       {cwd or '.'}\n")

    server = Server(command, cwd, timeout)
    failures = 0
    try:
        # -- handshake ---------------------------------------------------
        t0 = time.perf_counter()
        reply = server.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "terraform-docs-probe", "version": "1"},
            },
        )
        if "error" in reply:
            _report("initialize", False, json.dumps(reply["error"])[:200])
            return 1
        info = reply.get("result", {}).get("serverInfo", {})
        _report(
            "initialize",
            True,
            f"{info.get('name')} {info.get('version')}  ({time.perf_counter() - t0:.2f}s)",
        )
        server.notify("notifications/initialized")

        # -- discovery ---------------------------------------------------
        t0 = time.perf_counter()
        listed = server.request("tools/list")
        if "error" in listed:
            _report("tools/list", False, json.dumps(listed["error"])[:200])
            return 1
        names = _tool_names(listed)
        _report("tools/list", True, f"{names}  ({time.perf_counter() - t0:.2f}s)")

        # -- invocation --------------------------------------------------
        # Discovery is answered from static metadata; only a call touches the
        # index and the model. Failing here while discovery succeeds is the
        # single most common report, so always exercise a call.
        targets = [tool] if tool else names
        for name in targets:
            if name not in names:
                _report(f"tools/call {name}", False, f"not advertised; have {names}")
                failures += 1
                continue
            args = arguments if tool else _default_args(name)
            if args is None:
                print(f"[ skip ] tools/call {name}  (no default arguments known)")
                continue
            t0 = time.perf_counter()
            try:
                called = server.request("tools/call", {"name": name, "arguments": args})
            except ProbeFailure as exc:
                _report(f"tools/call {name}", False, str(exc))
                failures += 1
                continue
            elapsed = time.perf_counter() - t0

            if "error" in called:
                _report(f"tools/call {name}", False, json.dumps(called["error"])[:300])
                failures += 1
                continue
            result = called.get("result", {})
            # A tool that raises reports isError with the message in content,
            # rather than a JSON-RPC error. Both are failures worth showing.
            if result.get("isError") or result.get("is_error"):
                text = " ".join(
                    c.get("text", "") for c in result.get("content", [])
                )[:300]
                _report(f"tools/call {name}", False, f"tool error: {text}")
                failures += 1
                continue
            blocks = result.get("content", [])
            preview = (blocks[0].get("text", "") if blocks else "").replace("\n", " ")[:90]
            _report(
                f"tools/call {name}",
                True,
                f"{len(blocks)} block(s) in {elapsed:.2f}s  | {preview}",
            )
            if elapsed > 10:
                print(
                    f"         note: {elapsed:.0f}s is longer than some clients' "
                    "default tool timeout"
                )
    except ProbeFailure as exc:
        _report("protocol", False, str(exc))
        failures += 1
    finally:
        stderr = server.drain_stderr()
        if server.junk:
            print("\nNON-JSON OUTPUT ON STDOUT — this corrupts the stdio protocol:")
            for line in server.junk[:10]:
                print("   ", line[:160])
            print(
                "  Anything printed to stdout by the server breaks JSON-RPC framing.\n"
                "  Send diagnostics to stderr instead."
            )
            failures += 1
        if stderr.strip():
            print("\nserver stderr:")
            for line in stderr.strip().splitlines()[-20:]:
                print("   ", line[:200])

    print()
    if failures:
        print(f"{failures} check(s) failed")
    else:
        print("all checks passed")
    return 1 if failures else 0


def _default_args(tool_name: str) -> dict[str, Any] | None:
    """Reasonable smoke-test arguments for this server's own tools."""
    if "search" in tool_name:
        return {"query": "s3 bucket lifecycle", "limit": 2}
    if "get_document" in tool_name:
        return {"doc_id": "aws:resource:s3_bucket"}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--command",
        help="Shell command that starts the server. Defaults to this package's server.",
    )
    parser.add_argument("--cwd", help="Working directory to launch the server in.")
    parser.add_argument("--tool", help="Call only this tool (default: every tool).")
    parser.add_argument(
        "--args", default="{}", help="JSON arguments for --tool, e.g. '{\"query\":\"vpc\"}'."
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "argv", nargs="*", help="Server command after `--` (alternative to --command)."
    )
    args = parser.parse_args(argv)

    if args.command:
        command = shlex.split(args.command)
    elif args.argv:
        command = args.argv
    else:
        command = [sys.executable, "-m", "terraform_docs_mcp.cli", "serve"]

    try:
        arguments = json.loads(args.args)
    except json.JSONDecodeError as exc:
        parser.error(f"--args is not valid JSON: {exc}")

    try:
        return probe(command, args.cwd, args.tool, arguments, args.timeout)
    except FileNotFoundError:
        print(f"[{_FAIL}] cannot execute: {command[0]}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
