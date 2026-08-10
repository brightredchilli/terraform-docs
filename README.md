# terraform-docs-mcp

An MCP server exposing hybrid (BM25 + vector) search over the Terraform **AWS**
and **Google** provider documentation.

The installed tool is fully self-contained: the wheel ships the prebuilt search
index, the embedding model, and the documentation itself. After installation it
works offline, with no model download, no index build, and no dependency on the
git submodules.

## Install

```bash
make bootstrap   # fetch submodules, sparse-checkout the docs
make index       # build the search index into src/terraform_docs_mcp/_data
make install     # uv tool install .
```

## Register with an MCP client

```bash
claude mcp add terraform-docs -- terraform-docs-mcp
```

Or by configuration file:

```json
{
  "mcpServers": {
    "terraform-docs": { "command": "terraform-docs-mcp" }
  }
}
```

The server speaks stdio by default. `--transport http` serves Streamable HTTP
in stateless mode instead, so requests can be spread across replicas without
sticky sessions.

## Tools

| Tool | Purpose |
|---|---|
| `search(query, provider=None, kind=None, limit=10)` | Ranked list of matching documents with a snippet |
| `get_document(doc_id, section=None)` | Full markdown for a document, or one `##` section |

`kind` filters by document type: `r` (resources), `d` (data sources), `guides`,
`functions`, `ephemeral-resources`, `list-resources`, `actions`.

## How search works

Two channels run over the same chunks and are combined with reciprocal rank
fusion, then rolled up so each document appears once:

- **BM25** (SQLite FTS5, Porter-stemmed) carries exact identifiers such as
  `aws_s3_bucket`.
- **Vector similarity** (`mxbai-embed-xsmall-v1` via sentence-transformers)
  carries paraphrases such as *"how do I delete old objects automatically"*,
  where the query shares no words with the target page.

Documents are chunked on markdown headings, and every chunk is prefixed with a
breadcrumb (`aws_instance — EC2 > Argument Reference > CPU Options`) so a
section still matches queries naming the resource it belongs to. Each document
also gets one short synthetic "summary" chunk, giving topical queries something
compact to match instead of competing with thousands of argument-level chunks.

Two rules sit on top of ranking:

- **Exact identifier wins.** A query naming `aws_lambda_function` returns that
  page, rather than whichever of the dozens of pages mentioning it ranks best.
- **A named provider is respected.** "…in google cloud" or "…in gcp" scopes the
  search, because returning AWS pages for those is simply wrong. A query naming
  both clouds (a migration question) stays unscoped.

## Scale and measured quality

| | |
|---|---|
| Documents | 4,338 (2,576 AWS + 1,762 Google) |
| Chunks | 42,446 |
| Packaged data | 123 MB (48 MB model, 16 MB vectors, 26 MB index, 36 MB docs) |
| Wheel | 70 MB |
| Installed | ~900 MB, of which torch is ~485 MB |
| Index build | ~96 s on Apple silicon (MPS) |
| Query latency | a few ms once the model is loaded (~2 s at first query) |

The install is dominated by torch, which comes in via sentence-transformers.
That is a deliberate trade: delegating tokenization, pooling, normalization and
batching to the library removes a class of silent correctness bug — an earlier
hand-rolled ONNX embedder mean-pooled a model that requires CLS pooling, which
produces plausible-looking but measurably worse vectors and raises no error.

On a 15-query benchmark spanning exact identifiers, attribute phrases and
paraphrases, **recall@5 is 11/15 and recall@10 is 13/15**. All identifier and
attribute-level queries pass. The four known failures are indirect paraphrases,
kept in the test suite as `xfail` with written reasons rather than deleted —
see `KNOWN_WEAK` in `tests/test_retrieval.py`. The benchmark is small enough
that tuning parameters against it further would fit noise rather than improve
retrieval.

## Choosing the embedding model

Candidates were compared on this project's own benchmark rather than by
leaderboard position, using a fast proxy (rank documents by their summary chunk
alone). Two findings shaped the choice:

| Model | query prompt | recall@5 | MRR | weights |
|---|---|---|---|---|
| bge-small-en-v1.5 | none / instruction | 4 → 7 | 0.21 → 0.29 | 133 MB |
| **mxbai-embed-xsmall-v1** | **none** | **8** | 0.30 | **48 MB** |
| bge-base-en-v1.5 | none / instruction | 5 → 9 | 0.26 → 0.36 | 439 MB |
| arctic-embed-m-v1.5 | instruction | 5 | 0.27 | 436 MB |

First, **the query instruction matters more than the model**: bge gains 3-4
recall points from it, and no candidate declared it to sentence-transformers,
so `encode_query` would silently have skipped it. Second, **bigger was barely
better** — bge-base leads the proxy but costs 9x the weights, and on the full
hybrid system `mxbai-embed-xsmall` reaches the same recall@5 as the previous
setup and a better recall@10.

Whether a model wants a query instruction is per-model and measured, not
assumed: bge is helped by one (5/15 → 9/15), this model is hurt by one
(8/15 → 5/15). The chosen prompt is written into the packaged model's own
config at build time, so runtime code carries no model-specific knowledge.

## Rebuilding

`make index` re-reads the submodules and regenerates everything. The index
records the commit SHA of each provider repo (`terraform-docs-search --stats`),
so you can tell which upstream revision a given artifact was built from.

## Use as a library

The same search is available to any Python project:

```python
from terraform_docs_mcp import Index

index = Index()                       # loads the packaged index
for hit in index.search("s3 bucket lifecycle", provider="aws", limit=5):
    print(hit["doc_id"], hit["score"], hit["snippet"])

print(index.get_document("aws:r/s3_bucket", section="Argument Reference"))
```

`Index` is safe to share across threads and keeps no per-request state, but the
first search loads an embedding model — build one `Index` per process and reuse
it.

Importing the package costs ~80 ms and pulls numpy, which the vector search
runs on. It does **not** pull torch: that arrives only when a search actually
needs to embed a query, so importing this package at module scope stays cheap
even in a process that never searches.

### Depending on it

**Depend on the built wheel, not on the git repository.**

```toml
[project]
dependencies = ["terraform-docs-mcp"]

[tool.uv.sources]
terraform-docs-mcp = { path = "/path/to/terraform_docs_mcp-0.1.0-py3-none-any.whl" }
```

The index is generated by `make index` and deliberately not committed, so
building straight from a git checkout **succeeds and produces a 34 KB package
with no data in it**. Nothing fails until the consumer calls `Index()`, which
then raises `IndexUnavailable`. A git dependency, an editable install of a
freshly cloned tree, and `uv add git+…` all hit this. Produce the artifact with
`make build` and distribute `dist/*.whl` — via a path, a private index, or an
artifact store.

Two smaller notes:

- `uv.lock` records the wheel's hash, so rebuilding at the same version breaks
  the lock with a hash mismatch. Bump the version, or run
  `uv lock --upgrade-package terraform-docs-mcp`.
- The package ships `py.typed`, so annotations are visible to type checkers in
  consuming projects.

## Debugging without an MCP client

```bash
terraform-docs-search "s3 bucket lifecycle expiration"
```

## Licensing

This tool redistributes documentation from
[terraform-provider-aws](https://github.com/hashicorp/terraform-provider-aws)
and
[terraform-provider-google](https://github.com/hashicorp/terraform-provider-google),
both licensed under the Mozilla Public License 2.0. See `NOTICE` and
`src/terraform_docs_mcp/_data/licenses/`.
