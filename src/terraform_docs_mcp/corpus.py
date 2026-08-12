"""Load Terraform provider documentation and split it into retrievable chunks.

Build-time only. This module imports ``langchain_text_splitters`` and ``yaml``,
which live in the ``build`` dependency group and are absent from an installed
tool.
"""

from __future__ import annotations

from enum import StrEnum
from os import PathLike
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._config import DOC_GLOB, DOC_SUFFIXES, PROJECT_ROOT

# Splitting on h3 as well as h1/h2 matters for this corpus: `Argument Reference`
# on a large resource page carries a dozen `###` sub-blocks (`CPU Options`,
# `EBS, Ephemeral, and Root Block Devices`, ...) and a query about one of them
# should retrieve that block, not the whole page.
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Character budgets. The embedding model truncates at 512 tokens; ~1600 chars
# keeps us comfortably inside that for English prose mixed with HCL.
MAX_CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200

#: Length of the human-readable excerpt stored per chunk for display.
SNIPPET_CHARS = 320

# AWS prefixes its H1 with the doc kind ("Resource: aws_instance"); Google uses
# the bare identifier ("google_compute_instance"). Strip the prefix so both
# providers yield a comparable title.
_TITLE_PREFIX = re.compile(
    r"^(?:Resource|Data Source|List Resource|Ephemeral(?: Resource)?|Action|Function):\s*"
)
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_H1 = re.compile(r"^# (.+)$", re.MULTILINE)

# Google wraps most of its H1 identifiers in code backticks
# (``` `google_bigquery_dataset` ```); AWS does not. Asterisks appear
# occasionally as emphasis. Underscores are load-bearing in identifiers, so
# they are deliberately left alone.
_TITLE_NOISE = re.compile(r"[`*]")

# Google's list-resource pages annotate the heading, e.g.
# "google_bigquery_dataset (list)".
_TITLE_ANNOTATION = re.compile(r"\s*\([^)]*\)\s*$")

#: Kinds whose pages describe a named terraform entity, so a filename-derived
#: identifier is a sensible fallback when the heading is prose.
ENTITY_KINDS = frozenset({"r", "d", "list-resources", "ephemeral-resources", "actions"})


# def _clean_heading(text: str) -> str:
#     return _TITLE_NOISE.sub("", text).strip()


def _strip_doc_suffix(name: str) -> str:
    for suffix in DOC_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class Provider(StrEnum):
    aws = "aws"
    google = "google"


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    source_docs_dir: Path
    source_license: Path
    destination_docs_dir: Path


PROVIDER_AWS = ProviderConfig(
    provider=Provider.aws,
    source_docs_dir=Path("terraform-provider-aws/website/docs"),
    source_license=Path("terraform-provider-aws/LICENSE"),
    destination_docs_dir=Path("docs/aws"),
)
PROVIDER_GOOGLE = ProviderConfig(
    provider=Provider.google,
    source_docs_dir=Path("terraform-provider-google/website/docs"),
    source_license=Path("terraform-provider-google/LICENSE"),
    destination_docs_dir=Path("docs/google"),
)
#: Providers indexed by this tool, mapped to their submodule directory name.
PROVIDERS: dict[str, ProviderConfig] = {
    "aws": PROVIDER_AWS,
    "google": PROVIDER_GOOGLE,
}


class Kind(StrEnum):
    resource = "resource"
    datasource = "datasource"
    ephemeral_resource = "ephemeral_resource"
    list_resource = "list_resource"
    action = "action"
    function = "function"
    guide = "guide"


def _parse_kind(s: str) -> Kind | None:
    if s == "d":
        return Kind.datasource
    elif s == "r":
        return Kind.resource
    elif s == "functions":
        return Kind.function
    elif s == "ephemeral-resources":
        return Kind.ephemeral_resource
    elif s == "guides":
        return Kind.guide
    elif s == "actions":
        return Kind.action
    elif s == "list-resources":
        return Kind.list_resource
    return None


@dataclass(frozen=True)
class Document:
    """A single provider documentation page."""

    doc_id: str  # "aws:r/instance"
    provider: Provider  # "aws" | "google"
    kind: Kind  # "r" | "d" | "guides" | ...
    # name: str | None  # "aws_instance"; None for guides and other prose pages
    title: str
    subcategory: str | None
    description: str | None
    rel_path: str  # "r/instance.html.markdown"
    body: str  # frontmatter removed


@dataclass(frozen=True)
class _Section:
    """One heading-delimited span, normalised from the splitter's output.

    The splitter returns langchain ``Document`` objects; normalising into this
    lets the headingless fallback be an ordinary value rather than a synthesised
    stand-in object.
    """

    page_content: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a document."""

    doc_id: str
    ordinal: int
    heading_path: str
    text: str  # breadcrumb-enriched; what gets embedded and FTS-indexed
    snippet: str  # short excerpt for display


def strip_frontmatter(text: str) -> tuple[str, str]:
    """Split leading YAML frontmatter from the markdown body.

    Returns ``(body, frontmatter)``; ``frontmatter`` is ``""`` when absent.

    This must run before any markdown splitting. Google's pages open with an
    auto-generation banner made of YAML comment lines that begin with ``#``, and
    a markdown splitter would otherwise read them as H1 headings and shred every
    one of those pages into nonsense sections.
    """
    if not text.startswith("---"):
        return text, ""
    end = text.find("\n---", 3)
    if end == -1:
        return text, ""
    return text[end + 4 :].lstrip("\n"), text[3:end]


def _parse_frontmatter(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    import yaml  # build-time dependency

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return _parse_frontmatter_fallback(raw)
    if not isinstance(data, dict):
        return {}
    return {k: str(v).strip() for k, v in data.items() if v is not None}


def _parse_frontmatter_fallback(raw: str) -> dict[str, str]:
    """Recover the scalar keys we care about from unparseable YAML."""
    out: dict[str, str] = {}
    for key in ("subcategory", "page_title", "description"):
        m = re.search(rf"^{key}:\s*(.+)$", raw, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip().strip('"')
    return out


# def _title_and_name(
#     body: str, meta: dict[str, str], stem: str, provider: str, kind: str
# ) -> tuple[str, str | None]:
#     """Derive a display title and the terraform identifier for a page."""
#     m = _H1.search(body)
#     raw = m.group(1) if m else meta.get("page_title", stem)
#     heading = _clean_heading(raw)
#
#     bare = _clean_heading(_TITLE_PREFIX.sub("", heading))
#     bare = _TITLE_ANNOTATION.sub("", bare)
#     if _IDENTIFIER.match(bare):
#         return heading, bare
#
#     # Google's IAM pages title themselves in prose ("IAM policy for Apigee
#     # Environment") because one page documents the _binding, _member and
#     # _policy resources together. Fall back to the filename so those pages are
#     # still reachable by identifier.
#     if kind in ENTITY_KINDS:
#         candidate = stem if stem.startswith(f"{provider}_") else f"{provider}_{stem}"
#         if _IDENTIFIER.match(candidate):
#             return heading, candidate
#
#     return heading, None


def iter_documents(config: ProviderConfig) -> Iterator[Document]:
    docs_root = PROJECT_ROOT / config.source_docs_dir
    """Yield every documentation page for one provider."""
    if not docs_root.is_dir():
        raise FileNotFoundError(
            f"{docs_root} not found. Run `make bootstrap` to fetch the submodules."
        )

    for path in sorted(docs_root.rglob(DOC_GLOB)):
        rel = path.relative_to(docs_root)
        # Top-level index.html.markdown has no parent directory to name it.
        if len(rel.parts) == 1:
            continue
        kind = _parse_kind(rel.parts[0])

        if kind is None:
            raise ValueError(f"{path} has no known kind")

        # hard assumption here that filenames are always the name of the resource
        # suffixed with provider_
        title = f"{config.provider.value}_{_strip_doc_suffix(path.name)}"

        raw = path.read_text(encoding="utf-8", errors="replace")
        body, frontmatter = strip_frontmatter(raw)
        meta = _parse_frontmatter(frontmatter)
        # title, name = _title_and_name(body, meta, stem, provider, kind)

        yield Document(
            doc_id=f"{config.provider.value}:{kind}:{title}",
            provider=config.provider,
            kind=kind,
            title=title,
            subcategory=meta.get("subcategory") or None,
            description=meta.get("description") or None,
            rel_path=rel.as_posix(),
            body=body,
        )


def _breadcrumb(doc: Document, headings: list[str]) -> str:
    """Build the context prefix prepended to every chunk.

    ``headings`` is ordered h1, h2, h3. The h1 is dropped because it merely
    restates the page identifier we already lead with, and repeating it dilutes
    the embedding.
    """
    head = doc.title
    if doc.subcategory:
        head = f"{head} — {doc.subcategory}"
    trail = " > ".join(h for h in headings[1:] if h)
    return f"{head} > {trail}" if trail else head


def chunk_document(doc: Document) -> list[Chunk]:
    """Split a document into breadcrumb-enriched chunks.

    Two passes: split on markdown headings for structure, then size-bound any
    oversized section. The heading splitter is fence-aware, so ``#`` characters
    inside HCL examples (``# Canonical``) are correctly treated as comments
    rather than headings.
    """
    from langchain_text_splitters import (  # build-time dependency
        MarkdownHeaderTextSplitter,
        MarkdownTextSplitter,
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = MarkdownTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
    )

    sections = [
        _Section(s.page_content, {str(k): str(v) for k, v in s.metadata.items()})
        for s in header_splitter.split_text(doc.body)
    ] or [_Section(doc.body, {})]  # a page with no headings at all

    chunks: list[Chunk] = [_summary_chunk(doc)]
    for section in sections:
        headings = [section.metadata.get(k, "") for _, k in HEADERS_TO_SPLIT_ON]
        crumb = _breadcrumb(doc, headings)
        heading_path = " > ".join(h for h in headings if h) or doc.title

        pieces = (
            size_splitter.split_text(section.page_content)
            if len(section.page_content) > MAX_CHUNK_CHARS
            else [section.page_content]
        )
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    ordinal=len(chunks),
                    heading_path=heading_path,
                    # The breadcrumb is prepended into the embedded text, not
                    # just kept as metadata: it is what lets a chunk about
                    # `root_block_device` still match the query "aws_instance
                    # root block device", where the resource name never appears
                    # in the section body.
                    text=f"{crumb}\n\n{piece}",
                    snippet=_make_snippet(piece),
                )
            )
    return chunks


def _summary_chunk(doc: Document) -> Chunk:
    """A short synthetic chunk standing for the document as a whole.

    Body sections describe *parts* of a resource, so a topical query ("object
    storage bucket on google cloud") has to beat thousands of argument-level
    chunks to reach the right page. This gives every document one compact,
    high-signal passage -- identifier, category and one-line purpose -- that
    such queries can match directly.
    """
    text = doc.title
    if doc.subcategory:
        text = f"{text} — {doc.subcategory}"
    if doc.description:
        text = f"{text}\n\n{doc.description}"

    return Chunk(
        doc_id=doc.doc_id,
        ordinal=0,
        heading_path=doc.title,
        text=text,
        snippet=_make_snippet(doc.description or doc.title),
    )


def _make_snippet(text: str) -> str:
    """Condense chunk text into a short display excerpt."""
    # The header splitter joins lines with markdown hard-breaks ("  \n"); undo
    # that so snippets read as prose.
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    if len(flat) <= SNIPPET_CHARS:
        return flat
    cut = flat[:SNIPPET_CHARS]
    space = cut.rfind(" ")
    return (cut[:space] if space > SNIPPET_CHARS // 2 else cut).rstrip() + "…"
