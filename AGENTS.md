# Working in this repository

An MCP server and Python library providing hybrid (BM25 + vector) search over
the Terraform AWS and Google provider documentation. The distinguishing
constraint is that **the installed artifact is self-contained and offline**: the
wheel ships the search index, the embedding model, and the documentation itself.

## Before calling any edit done

**Run the type checker.** This repo has a `pyrightconfig.json`
(`typeCheckingMode: standard`, `reportArgumentType: error`, with
`reportGeneralTypeIssues` relaxed under `tests/`). Edits are not finished until
it is clean:

```bash
make typecheck      # uv run basedpyright src tests
make check          # typecheck + tests
```

Fix what it reports rather than silencing it. When a third-party function is
overloaded too widely to narrow — `SentenceTransformer.encode_*` returns
tensor/array/list/dict — convert at the boundary (`np.asarray(...)`) instead of
adding `# type: ignore`.

Also run `make test` (96 tests, ~20 s). Some tests are skipped without a built
index; that is expected before `make index`.

## Invariants that are easy to break

**`_data/` is generated, gitignored, and must ship in the wheel.** It is built
by `make index`. Do **not** add `source-exclude` for it in
`[tool.uv.build-backend]` — despite being documented as applying to source
distributions, it also strips paths from the *wheel*, which silently produces a
34 KB package containing no index. uv_build includes non-`.py` files under the
module directory automatically, gitignore notwithstanding.

**Never depend on this project from git.** A git checkout has no `_data`, so the
build succeeds and produces a package that raises `IndexUnavailable` on first
use. Distribute `dist/*.whl` built after `make index`.

**Submodules must not import from the package root.** `__init__.py` imports
`.index` to re-export `Index`; if a submodule does `from . import X`, that
becomes a circular import. Shared constants live in `_config.py`. There is a
test enforcing this.

**The package root is a facade.** Public API is `Index`, `IndexUnavailable`,
`__version__` — nothing else. Corpus globs, provider→submodule mappings and
`data_dir()` are internals; import them from `._config`.

**torch must stay out of import time.** `sentence_transformers` is imported
inside methods, not at module scope, so importing the package (or constructing
an `Index`) does not load torch — only the first search does. Tests assert this
in a subprocess.

**Changing the embedding model requires a re-index.** `MODEL_ID` is recorded in
the index and re-checked at query time. `download_model` compares a
`MODEL_REPO.txt` marker, so changing `MODEL_REPO` correctly re-downloads.

## Things measured, not assumed

Several decisions here were made from measurement and should not be "corrected"
back on intuition:

- **The query instruction is per-model and its direction flips.** bge needs one
  (recall@5 5/15 → 9/15); the model we ship is hurt by one (8/15 → 5/15). It is
  written into the packaged model's config at build time, not into our code.
- **Weighting the vector channel above BM25 makes retrieval worse.** Equal RRF
  weights were measured best.
- **CoreML is 4.2x slower than CPU here**, because the graph fragments into 74
  partitions. Do not "enable acceleration".
- **The secondary-chunk bonus is capped at one chunk.** Summing over all
  matching chunks rewards long documents, which let the 12,000-word upgrade
  guide win unrelated queries.

Known-weak benchmark queries are tracked as `xfail` with written reasons in
`tests/test_retrieval.py::KNOWN_WEAK`. The benchmark is 15 queries — small
enough that tuning parameters against it fits noise. Prefer structural fixes.

## Commands

| | |
|---|---|
| `make bootstrap` | fetch submodules, sparse-checkout `website/docs` only |
| `make index` | build the index into `_data/` (~96 s on Apple silicon) |
| `make check` | typecheck + tests |
| `make build` | wheel (runs `index` first — `uv_build` has no build hooks) |
| `make install` | `uv tool install` locally |
| `make probe` | exercise the stdio MCP server end to end |

## Layout

```
src/terraform_docs_mcp/
  _config.py     constants and packaged-data path (no intra-package imports)
  corpus.py      load + chunk markdown          (build-time only)
  embed.py       sentence-transformers wrapper
  index.py       sqlite + vectors, hybrid search
  search.py      query prep, RRF fusion, aggregation (pure functions)
  server.py      MCP tools + serve(); no argument parsing
  cli.py         typer app: search/get/stats/build/serve — all entry points
  probe.py       stdio MCP diagnostic
  build_index.py index builder                  (build-time only)
  _data/         GENERATED — never committed, always shipped
```
