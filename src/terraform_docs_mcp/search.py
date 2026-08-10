"""Query preparation and rank fusion.

Pure functions over ranked id lists -- no I/O, so they are cheap to unit test
independently of the index.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

#: RRF damping constant. 60 is the value from the original Cormack et al. paper
#: and is not sensitive enough to be worth tuning here.
RRF_K = 60

#: Weight applied to a document's runner-up chunk. A page matching in more than
#: one section is a slightly better bet than one matching in a single spot.
SECONDARY_CHUNK_WEIGHT = 0.2

#: How many runner-up chunks may contribute. Deliberately small: summing over
#: every matching chunk rewards documents for being *long* rather than
#: relevant, which let the 12,000-word provider upgrade guides win unrelated
#: queries purely by having more chunks in the candidate pool.
MAX_SECONDARY_CHUNKS = 1

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

# FTS5 ships no stopword list, and these terms are ruinous here. Under OR
# semantics a natural-language query like "how do I delete old files in a
# bucket" scores every long page that happens to contain "how", "do" and "in" --
# in practice the provider version-upgrade guides, which are enormous and
# mention everything. IDF alone does not rescue it because those pages are long
# enough to match many stopwords each.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    with without from by as is are was were be been being do does did doing
    have has had having i me my we our you your it its they them their
    how what when where which who whom why can could should would will shall
    may might must not no nor so such about into over under again further
    here there all any both each few more most other some only own same too
    very just also get got make makes made use used using want need
    """.split()
)


#: Words that name a cloud unambiguously. Deliberately limited to provider
#: names -- service names like "s3" or "bigquery" are already strong lexical
#: signals, and inferring from them would misfire on comparative questions.
_PROVIDER_WORDS = {
    "aws": "aws",
    "amazon": "aws",
    "google": "google",
    "gcp": "google",
    "gcs": "google",
}


def infer_provider(query: str) -> str | None:
    """Detect a provider named in the query, if exactly one is named.

    A query saying "in google cloud" that returns AWS documents is wrong
    regardless of how similar those documents look, and this was the single
    largest source of bad results: "managed relational database in google
    cloud" returned four AWS RDS pages in its top five.

    Returns ``None`` when the query names both providers (a migration
    question) or neither, so the caller searches unfiltered.
    """
    found = {
        _PROVIDER_WORDS[t]
        for t in (m.lower() for m in _TOKEN.findall(query))
        if t in _PROVIDER_WORDS
    }
    return found.pop() if len(found) == 1 else None


def _informative_tokens(query: str) -> list[str]:
    """Query tokens with stopwords removed, falling back if that empties it."""
    tokens = _TOKEN.findall(query)
    kept = [t for t in tokens if t.lower() not in _STOPWORDS]
    # An all-stopword query ("how to") still deserves its literal terms.
    return kept or tokens


def to_fts_match(query: str) -> str:
    """Build an FTS5 MATCH expression from free text.

    User queries routinely contain characters FTS5 treats as syntax (``-``,
    ``*``, ``:``, quotes), which would otherwise raise
    ``sqlite3.OperationalError``. Extracting bare tokens and quoting each one
    sidesteps the grammar entirely.

    Quoting also turns an identifier like ``aws_instance`` into a phrase query,
    since the tokenizer splits on the underscore -- so it matches documents
    where "aws" and "instance" are adjacent rather than merely co-occurring.
    """
    tokens = _informative_tokens(query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Fuse ranked id lists into a single scored mapping.

    RRF combines rankings by position rather than score, so the channels need
    no calibration against each other -- which is what makes hybrid retrieval
    practical without tuning a score blend.

    ``weights`` scales each channel's contribution. It is needed because the
    two channels differ in how they behave when they have nothing useful to
    say: cosine similarity degrades gracefully, but BM25 under OR semantics
    returns a full page of candidates for *any* query with matching tokens. On
    a paraphrased question its entire ranking can be noise, and unweighted RRF
    would let that noise outvote a genuinely strong vector hit.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def aggregate_to_documents(
    fused: Mapping[int, float],
    chunks: Mapping[int, Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Roll chunk scores up to their parent documents.

    ``search`` returns documents, so competing chunks from one page must not
    occupy several result slots. The best-matching chunk supplies the snippet.
    """
    best: dict[str, dict[str, Any]] = {}
    for chunk_id, score in fused.items():
        row = chunks.get(chunk_id)
        if row is None:
            continue
        doc_id = row["doc_id"]
        entry = best.get(doc_id)
        if entry is None:
            best[doc_id] = {
                "doc_id": doc_id,
                "score": score,
                "extra": 0.0,
                "seconds": 0,
                "heading_path": row["heading_path"],
                "snippet": row["snippet"],
            }
        elif entry["seconds"] < MAX_SECONDARY_CHUNKS:
            # fused is ordered by descending score, so the first chunk seen for
            # a document is its best and keeps ownership of the snippet.
            entry["extra"] += score
            entry["seconds"] += 1

    ranked = sorted(
        best.values(),
        key=lambda e: -(e["score"] + SECONDARY_CHUNK_WEIGHT * e["extra"]),
    )
    for entry in ranked:
        entry["score"] += SECONDARY_CHUNK_WEIGHT * entry.pop("extra")
        entry.pop("seconds")
    return ranked[:limit]


# def format_results(results: Iterable[Mapping[str, Any]]) -> str:
#     """Render results as plain text for the CLI."""
#     lines = []
#     for i, r in enumerate(results, start=1):
#         label = r.get("name") or r.get("title")
#         subcat = f" [{r['subcategory']}]" if r.get("subcategory") else ""
#         lines.append(f"{i:2}. {label}  ({r['provider']}/{r['kind']}){subcat}")
#         lines.append(f"    {r['doc_id']}  score={r['score']:.4f}")
#         if r.get("heading"):
#             lines.append(f"    § {r['heading']}")
#         lines.append(f"    {r['snippet']}")
#         lines.append("")
#     return "\n".join(lines) if lines else "No results."
