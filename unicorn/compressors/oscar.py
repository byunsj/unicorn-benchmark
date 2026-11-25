"""OSCAR (naver/oscar-qwen2-7B) soft-compression + generation pipeline.

OSCAR compresses document chunks into embeddings and answers directly from them
with its own decoder, so it can't be split into the prune-then-generate phases:
this class does doc selection, chunking, compression and generation in one call.

Documents are selected by the shared full-document BGE reranker, then split into
fixed 128-token windows for memory compression. ``max_chunks_per_doc`` keeps only
each document's leading windows, which is how Comp-S bounds a document's memory
blocks: with both capped at 16 the two soft methods hand their decoder the same
span of the same documents, so the comparison is about compression rather than
about how much of the document each one happened to read.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ._crossencoder import DEFAULT_BGE_RERANKER_MODEL
from .base import ExternalRerankerPlugin

logger = logging.getLogger(__name__)

DEFAULT_OSCAR_MODEL = "naver/oscar-qwen2-7B"
# OSCAR's internal compressor tokenizer; used here only to count tokens so
# chunk boundaries match OSCAR's 128-token window. Gated on HuggingFace.
DEFAULT_CHUNK_TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"


class OscarPipeline(ExternalRerankerPlugin):
    """External-reranker top-k selection + OSCAR, using fixed-size token chunks."""

    SPECIAL_TOKEN_BUDGET = 4
    # OSCAR's decoder prompt holds one (n_mem_tokens + sep) = 9-token block per
    # chunk and is truncated to 2048 tokens internally; too many chunks overflow it
    # (mem-token slots get cut -> replace_embeddings assertion) and blow up memory.
    # This global floor keeps the highest-scoring chunks; with max_chunks_per_doc
    # set it cannot bind unless max_chunks_per_doc * top_k exceeds it.
    MAX_OSCAR_CHUNKS = 180

    def __init__(
        self,
        model_name: str = DEFAULT_OSCAR_MODEL,
        chunk_tokenizer_name: str = DEFAULT_CHUNK_TOKENIZER,
        chunk_max_tokens: int = 128,
        max_chunks_per_doc: int = 16,
        display_name: str | None = None,
        lazy: bool = True,
        **kwargs,
    ):
        if not kwargs.get("external_reranker_name"):
            kwargs["external_reranker_name"] = DEFAULT_BGE_RERANKER_MODEL
        super().__init__(**kwargs)
        self.model_name = model_name
        self.chunk_tokenizer_name = chunk_tokenizer_name
        self.chunk_max_tokens = chunk_max_tokens
        # Windows kept per selected document; 0 keeps all of them.
        self.max_chunks_per_doc = max_chunks_per_doc
        self.display_name = display_name or model_name

        self.model = None
        self._chunk_tokenizer = None
        self.device: str | None = None

        if not lazy:
            self.load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "auto") -> None:
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        try:
            self._chunk_tokenizer = AutoTokenizer.from_pretrained(self.chunk_tokenizer_name)
        except OSError as e:
            raise RuntimeError(
                f"Could not load chunk tokenizer '{self.chunk_tokenizer_name}'. "
                "This repo is gated on HuggingFace -- accept the license at "
                f"https://huggingface.co/{self.chunk_tokenizer_name} and set "
                "HF_TOKEN / run `huggingface-cli login` with an approved account, "
                "or pass a different `chunk_tokenizer_name` in the reranker config."
            ) from e

        self.load_external_reranker(device)

        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(device)
        self.model.eval()

    def unload(self) -> None:
        self.unload_external_reranker()
        del self.model
        del self._chunk_tokenizer
        self.model = None
        self._chunk_tokenizer = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None and self.is_external_reranker_loaded()

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            **self.get_external_reranker_info(),
            "chunk_tokenizer_name": self.chunk_tokenizer_name,
            "chunk_max_tokens": self.chunk_max_tokens,
            "max_chunks_per_doc": self.max_chunks_per_doc,
        }

    @property
    def tokenizer(self):
        """Chunk tokenizer, exposed for token-count bookkeeping like other compressors."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return self._chunk_tokenizer

    @property
    def compression_rate(self) -> int:
        """Document tokens represented by one memory embedding (16 for oscar-qwen2-7B)."""
        rate = getattr(self.model, "compr_rate", None)
        return int(rate) if isinstance(rate, int) and rate > 0 else 16

    @property
    def n_mem_tokens(self) -> int:
        """Number of soft memory embeddings OSCAR produces per chunk.

        OSCAR compresses each chunk into ``doc_max_length // compr_rate`` memory
        vectors (8 for oscar-qwen2-7B: 128 // 16). Read from the loaded model so the
        compression-ratio bookkeeping matches what the decoder actually consumes.
        """
        model = self.model
        n = getattr(model, "n_mem_tokens", None)
        if isinstance(n, int) and n > 0:
            return n
        dm = getattr(model, "doc_max_length", None)
        cr = getattr(model, "compr_rate", None)
        if isinstance(dm, int) and isinstance(cr, int) and cr > 0:
            return dm // cr
        return 8

    def _chunk_documents(self, query: str, docs: list[str]) -> list[list[str]]:
        """Split docs into OSCAR's fixed windows, reserving query space.

        With ``max_chunks_per_doc`` set, a document contributes only its leading
        windows, so the text past that point never reaches the decoder.
        """
        query_tokens = len(self._chunk_tokenizer.encode(query or "", add_special_tokens=False))
        budget = max(self.chunk_max_tokens - query_tokens - self.SPECIAL_TOKEN_BUDGET, 1)
        chunks_by_doc: list[list[str]] = []
        for doc in docs:
            if not isinstance(doc, str) or not doc.strip():
                chunks_by_doc.append([])
                continue
            token_ids = self._chunk_tokenizer.encode(doc, add_special_tokens=False)
            if self.max_chunks_per_doc:
                token_ids = token_ids[: budget * self.max_chunks_per_doc]
            chunks = [
                self._chunk_tokenizer.decode(token_ids[start : start + budget], skip_special_tokens=True).strip()
                for start in range(0, len(token_ids), budget)
            ]
            # The fallback reinstates the whole document when every window decoded
            # to whitespace, which would otherwise slip past the cap.
            chunks = [chunk for chunk in chunks if chunk] or [doc]
            chunks_by_doc.append(chunks[: self.max_chunks_per_doc] if self.max_chunks_per_doc else chunks)
        return chunks_by_doc

    def _fit_query_to_compressor_window(self, query: str) -> str:
        """Shorten a query that cannot share OSCAR's compression window with a chunk.

        OSCAR tokenizes (query, chunk) as a *pair* into ``chunk_max_tokens + 3``
        positions with ``truncation='only_second'``, so only the chunk may be cut and
        a query that fills the window on its own makes the tokenizer raise. Such a
        query is cut to half the window, leaving room for document tokens in every
        window. Document selection still uses the full query.
        """
        tokenizer = self._chunk_tokenizer
        limit = self.chunk_max_tokens + 3 - tokenizer.num_special_tokens_to_add(pair=True)
        token_ids = tokenizer.encode(query or "", add_special_tokens=False)
        if len(token_ids) < limit:
            return query

        keep = max(self.chunk_max_tokens // 2, 1)
        logger.warning(
            "[OSCAR] truncating query from %d to %d tokens: it does not fit the compressor's "
            "%d-token (query, chunk) window, which truncates only the chunk.",
            len(token_ids),
            keep,
            self.chunk_max_tokens + 3,
        )
        return tokenizer.decode(token_ids[:keep], skip_special_tokens=True)

    def _fit_to_budget(self, query: str, chunks_by_doc: list[list[str]]) -> list[list[str]]:
        """Drop the lowest-scoring chunks until the decoder's capacity is met.

        BGE ranks whole documents, so chunk-level scores are computed only here,
        to choose between chunks once there are more than the decoder can hold.
        """
        flat = [(doc_i, chunk) for doc_i, chunks in enumerate(chunks_by_doc) for chunk in chunks]
        scores = super().compute_relevance_scores(query, [chunk for _, chunk in flat])
        keep = sorted(range(len(flat)), key=lambda i: scores[i], reverse=True)[: self.MAX_OSCAR_CHUNKS]
        kept: list[list[str]] = [[] for _ in chunks_by_doc]
        for i in sorted(keep):  # restore reading order
            doc_i, chunk = flat[i]
            kept[doc_i].append(chunk)
        return kept

    def rerank_and_generate(
        self,
        query: str,
        docs: list[str],
        top_k: int,
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        """Select top-k documents, then let OSCAR generate from their chunks."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        n = len(docs)
        # The compressor shares a fixed window with the query, so an over-long query
        # is cut for compression only; document selection below keeps the full query.
        compressor_query = self._fit_query_to_compressor_window(query)
        chunks_by_doc = self._chunk_documents(compressor_query, docs)

        scores = super().compute_relevance_scores(query, docs)
        ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        top_k_indices = ranked_indices[:top_k]

        # Selected docs' chunks in reading order, top document first.
        selected_by_doc = [list(chunks_by_doc[i]) for i in top_k_indices]
        n_selected = sum(len(chunks) for chunks in selected_by_doc)
        if n_selected > self.MAX_OSCAR_CHUNKS:
            selected_by_doc = self._fit_to_budget(query, selected_by_doc)

        selected_chunks = [chunk for chunks in selected_by_doc for chunk in chunks]
        if selected_chunks:
            # OSCAR's modelling code runs compression/generation without disabling
            # autograd, so without this guard the compressor builds a full graph over
            # every chunk and OOMs on long documents. inference_mode also speeds it up.
            with torch.inference_mode():
                answers = self.model.generate_from_text(
                    questions=[compressor_query],
                    documents=[selected_chunks],
                    max_new_tokens=max_new_tokens,
                    query_dependent=True,
                )
            generated_answer = (answers[0] or "").strip()
        else:
            generated_answer = ""

        logger.info(
            "[OSCAR] docs=%d selected_docs=%d selected_chunks=%d (of %d)",
            n,
            len(top_k_indices),
            len(selected_chunks),
            n_selected,
        )

        return {
            "scores": scores,
            "ranked_indices": ranked_indices,
            "top_k_indices": top_k_indices,
            "selected_chunks_by_doc": selected_by_doc,
            "n_mem_tokens": self.n_mem_tokens,
            "generated_answer": generated_answer,
        }


__all__: list[str] = [
    "DEFAULT_CHUNK_TOKENIZER",
    "DEFAULT_OSCAR_MODEL",
    "OscarPipeline",
]
