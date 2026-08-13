# Working in this repository

Hybrid (BM25 + vector) search over the Terraform AWS and Google provider
documentation, exposed as an MCP server and a Python library. The installed
artifact is meant to be self-contained and offline: the wheel ships the search
index and documentation alongside the code.

## Before calling any edit done

Run the type checker — this repo has a `pyrightconfig.json`
(`typeCheckingMode: standard`, `reportArgumentType: error`). Fix what it
reports rather than silencing it.

```bash
make typecheck      # uv run basedpyright src tests
make check          # typecheck + tests
```

## Style

**Prefer `"""` triple-quoted strings over adjacent-literal concatenation for
multiline text** (`"line one\n" "line two\n"` → `"""line one\nline two"""`).
This matters most for expected-value fixtures in tests — concatenation is easy
to get subtly wrong (a missing space between fragments, mismatched embedded
quotes) and harder to eyeball against the real content it is meant to match.

## Two entry points

**`cli.py`** is what gets installed and exposed to users (`terraform-docs-mcp`
on `$PATH`): `search`/`get`/`stats` for the command line, and `serve` to run
the MCP server (stdio or Streamable HTTP). This is the only thing an installed
user ever runs.

**`build_index.py`** is development-only, never installed. It builds
`_data/` (the packaged search index) and `src/summaries/` (per-document
summaries, checked into git as source, not generated at install time). Bare
invocation builds both, skipping whichever hasn't changed:

```bash
uv run src/terraform_docs_mcp/build_index.py            # everything
uv run src/terraform_docs_mcp/build_index.py index       # just the index
uv run src/terraform_docs_mcp/build_index.py summaries   # just the summaries
```

Its imports are absolute (`terraform_docs_mcp.x`), not relative, specifically
so it can run by path as shown above, not just via `-m` — a relative import
breaks the moment a file executes as `__main__`. This is the one module in the
package where that matters; everywhere else, relative imports are the norm.

Staleness for both stages is tracked in `_data/manifest.json`, a flat dict of
provider commit SHAs (`documents_aws`, `summaries_google`, ...), updated
incrementally as each stage completes — not a single atomic write at the end.

## Layout

```
src/terraform_docs_mcp/
  cli.py          installed entry point: search/get/stats/serve
  server.py       MCP tools
  index.py        sqlite + vectors, hybrid search (read path)
  db.py           document-level sqlite store + trigram search
  search.py       query prep, RRF fusion (pure functions)
  corpus.py       load + chunk markdown, extract_intro()   (build-time only)
  summarize.py    per-document summaries                   (build-time only)
  build_index.py  dev entry point: builds _data/ and src/summaries/
  manifest.py     _data/manifest.json read/write
  embed.py        sentence-transformers wrapper
  _data/          GENERATED — gitignored, ships in the wheel
src/summaries/    GENERATED once — committed, not gitignored
```
