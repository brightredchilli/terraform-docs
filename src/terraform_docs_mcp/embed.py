"""Sentence embeddings via sentence-transformers.

The model is downloaded once at build time and saved into the package's
``_data`` directory, then loaded from that path at runtime -- so the installed
tool never reaches the network.

This module deliberately delegates rather than reimplements. Tokenization,
truncation, padding, batching, pooling, normalization and the model's own
query/document prompts are all sentence-transformers' responsibility. An
earlier hand-rolled ONNX version of this file got the pooling mode wrong
(bge pools the CLS token, not the token mean) in a way that produced
plausible-looking but measurably worse vectors, which is exactly the class of
bug that disappears when the library reads the model's own configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

#: Model to vendor. Chosen by measuring candidates on this project's own
#: retrieval benchmark -- see "Choosing the embedding model" in the README.
#: 24M parameters and 48 MB of weights, which matters because the model ships
#: inside the wheel; bge-base scored marginally higher on a summary-only proxy
#: but costs 439 MB.
MODEL_REPO = "mixedbread-ai/mxbai-embed-xsmall-v1"

#: Instruction prepended to queries but not to documents. Empty for this model.
#:
#: Whether a model wants a query instruction is per-model and *not* reliably
#: declared: every candidate reported empty prompts to sentence-transformers,
#: yet bge measurably needs one (recall@5 5/15 without, 9/15 with) while this
#: model is measurably hurt by one (8/15 without, 5/15 with). So it is measured
#: rather than assumed, and written into the packaged model's own config at
#: build time (see :func:`download_model`) to keep runtime code free of
#: model-specific knowledge.
MODEL_QUERY_PROMPT = ""

#: Recorded in the index and re-checked at query time, so an index built with
#: one model can never be searched with another.
MODEL_ID = MODEL_REPO

#: Override the torch device (``cpu``, ``mps``, ``cuda``). Default is
#: sentence-transformers' own auto-detection.
DEVICE_ENV = "TERRAFORM_DOCS_DEVICE"


class Embedder(Protocol):
    """Minimal surface the index depends on, so the model stays swappable."""

    model_id: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Loads a sentence-transformers model saved on disk."""

    model_id = MODEL_ID

    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        import logging

        # sentence-transformers logs model loading at INFO and prints a tqdm
        # bar per encode call. On an MCP stdio server that is a stream of
        # stderr noise around every single query, so quieten it here rather
        # than in each caller.
        for name in ("sentence_transformers", "transformers"):
            logging.getLogger(name).setLevel(logging.WARNING)
        try:  # transformers also draws a bar while loading the weights
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:  # pragma: no cover - cosmetic only
            pass

        from sentence_transformers import SentenceTransformer

        if not (model_dir / "modules.json").exists():
            raise FileNotFoundError(
                f"No sentence-transformers model in {model_dir}. Run `make index` to build it."
            )
        self._model = SentenceTransformer(
            str(model_dir),
            device=device or os.environ.get(DEVICE_ENV) or None,
            local_files_only=True,
        )

    @property
    def dim(self) -> int:
        # Renamed in sentence-transformers 5.6; keep the old spelling working.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        value = getter()
        if value is None:
            raise RuntimeError(
                f"{MODEL_REPO} did not report an embedding dimension; the saved "
                "model folder may be incomplete."
            )
        return int(value)

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = 64, progress: bool = False
    ) -> np.ndarray:
        """Embed passages. Returns an ``(n, dim)`` float32 array of unit vectors.

        No manual batching or length-sorting here: ``encode`` already sorts by
        length internally to minimise padding and restores the original order
        afterwards.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode_document(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        # encode_* is overloaded over tensor/array/list/dict returns; asarray
        # both narrows the type and is a no-op for the array we asked for.
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns a 1-D unit vector of length ``dim``.

        ``encode_query`` applies whatever query prompt the model declares in
        its own config (bge, e5 and arctic all expect different ones), so the
        asymmetry is handled by the model rather than hardcoded here.
        """
        vector = self._model.encode_query(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)


def download_model(model_dir: Path) -> Path:
    """Fetch the model and save it into ``model_dir`` for packaging.

    Build-time only. ``SentenceTransformer.save`` writes a self-contained
    folder -- weights, tokenizer, pooling config and prompts -- which
    :class:`SentenceTransformerEmbedder` then loads by path with
    ``local_files_only``. Saving rather than snapshotting the repo also avoids
    shipping duplicate weight formats, since many repos carry both
    ``pytorch_model.bin`` and ``model.safetensors``.
    """
    import shutil

    # Record which repo produced the saved folder. Without this, changing
    # MODEL_REPO leaves the previous model in place -- the folder still looks
    # valid, so the build silently embeds with the old model while the index
    # records the new name.
    marker = model_dir / "MODEL_REPO.txt"
    if marker.exists() and marker.read_text().strip() == MODEL_REPO:
        return model_dir
    if model_dir.exists():
        shutil.rmtree(model_dir)

    from sentence_transformers import SentenceTransformer

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        MODEL_REPO,
        # Baked into the saved config so the packaged model carries its own
        # query/document asymmetry and the runtime never has to know about it.
        prompts={"query": MODEL_QUERY_PROMPT, "document": ""},
    )
    model.save(str(model_dir))
    marker.write_text(MODEL_REPO + "\n")
    return model_dir
