"""Terminal front-end for the same search the MCP server exposes.

Exists so retrieval can be exercised and tuned without wiring up an MCP client.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .index import Index, IndexUnavailable
from .search import format_results


def main(argv: list[str] | None = None) -> int:
    """Entry point that survives a closed pipe.

    Documents run to thousands of lines, so `--get ... | head` is the normal
    way to use this. Without handling the resulting BrokenPipeError, Python
    prints a traceback at shutdown when it flushes stdout.
    """
    try:
        return _main(argv)
    except BrokenPipeError:
        # Point stdout at devnull so the interpreter's final flush cannot
        # raise again, then report the conventional SIGPIPE status.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 128 + 13
    except KeyboardInterrupt:
        return 130


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Search terms.")
    parser.add_argument("--provider", choices=("aws", "google"))
    parser.add_argument("--kind")
    parser.add_argument("-n", "--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON.")
    parser.add_argument("--get", metavar="DOC_ID", help="Print a document instead of searching.")
    parser.add_argument("--section", help="With --get, print only this section.")
    parser.add_argument("--stats", action="store_true", help="Show index statistics.")
    args = parser.parse_args(argv)

    try:
        index = Index()
    except IndexUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.stats:
        print(json.dumps(index.stats(), indent=2, sort_keys=True))
        return 0

    if args.get:
        try:
            print(index.get_document(args.get, section=args.section))
        except KeyError as exc:
            print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            return 1
        return 0

    query = " ".join(args.query).strip()
    if not query:
        parser.error("provide a query, --get DOC_ID, or --stats")

    results = index.search(query, provider=args.provider, kind=args.kind, limit=args.limit)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
