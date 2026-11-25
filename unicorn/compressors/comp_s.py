"""Comp-S adapter for the rerank side of the pipeline.

Comp-S does not prune text: it compresses each document into ``<MEM>`` embeddings
that its own decoder consumes, so the compression and answer generation happen in
:class:`unicorn.generator.CompSGenerator`. Here the model only *scores* documents
for top-k selection and passes the selected ones through as full text.

The released checkpoint has no native reranking head, so selection is driven by
the same external BGE reranker the other methods use. That is deliberate: it
holds document selection fixed across methods, so the comparison isolates what
compression does to the answer rather than what reranking does to the pool.
"""

from __future__ import annotations

import logging

import torch

from .base import CompressionModel

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Escape literal ``<MEM>`` / ``<AE>`` markers, matching the training collator."""
    return text.replace("<MEM>", "< MEM >").replace("<AE>", "< AE >")


class CompS(CompressionModel):
    def __init__(
        self,
        model_name: str,
        display_name: str | None = None,
        lazy: bool = True,
        compressor_max_length: int = 128,
        query_dependent: bool | None = None,
        attn_implementation: str | None = None,
        rerank_batch_size: int = 8,
        max_chunks_per_doc: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.display_name = display_name or model_name
        self.compressor_max_length = compressor_max_length
        # None means inherit from the checkpoint config at load time.
        self._query_dependent_override = query_dependent
        self.query_dependent = True
        # None picks FlashAttention-2 when it is installed and eager otherwise.
        self.attn_implementation = attn_implementation
        self.rerank_batch_size = rerank_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.HAS_SEPARATE_SCORING = True

        self.model = None
        self.device: str | None = None

        if not lazy:
            self.load()

    def load(self, device: str = "auto") -> None:
        from ..models.soft_compression import SoftCompressionEncoder

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # The decoder is only needed for generation, so skip it here.
        model = SoftCompressionEncoder.from_pretrained(
            self.model_name, load_decoder=False, attn_implementation=self.attn_implementation
        )
        self.model = model.to(device).eval()

        self.query_dependent = (
            self._query_dependent_override
            if self._query_dependent_override is not None
            else bool(model.config.query_dependent)
        )

        if self.external_reranker_name:
            self.load_external_reranker(device)
        else:
            logger.warning(
                "Comp-S has no reranker configured; falling back to retrieval order (all scores 0)."
            )

    def unload(self) -> None:
        self.unload_external_reranker()
        del self.model
        self.model = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "display_name": self.display_name,
            "loaded": self.is_loaded(),
            "device": str(self.device) if self.device else None,
            **self.get_external_reranker_info(),
        }

    @property
    def tokenizer(self):
        """Compressor tokenizer, exposed for token-count bookkeeping."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        return self.model.compressor_tokenizer

    def compute_relevance_scores(self, query: str, docs: list[str]) -> list[float]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        if not docs:
            return []
        if self._external_reranker is not None:
            return super().compute_relevance_scores(query or "", docs)
        return [0.0] * len(docs)

    def compute_relevance_score(self, query: str, document: str) -> float:
        return self.compute_relevance_scores(query, [document])[0]

    @property
    def compression_rate(self) -> int:
        """Document tokens represented by one memory embedding."""
        return int(self.model.compr_rate)

    def memory_vectors(self, query: str, docs: list[str]) -> int:
        """Memory embeddings the decoder will actually consume for *docs*.

        Lower than the documents' full compressed size whenever a document runs
        past ``max_chunks_per_doc`` windows, since the rest is never read. Shares
        the chunking helper with the generator, so this is the count the decoder
        really sees.
        """
        from ..models.chunking import add_memory_tokens_to_inputs, chunk_doc_to_rows

        tokenizer = self.model.compressor_tokenizer
        rows: list[list[int]] = []
        for doc in docs:
            rows.extend(
                chunk_doc_to_rows(
                    tokenizer,
                    _clean_text(query or ""),
                    _clean_text(doc).strip(),
                    compressor_max_length=self.compressor_max_length,
                    query_dependent=self.query_dependent,
                    max_chunks_per_doc=self.max_chunks_per_doc,
                )
            )
        if not rows:
            return 0
        _, per_row = add_memory_tokens_to_inputs(
            rows, mem_token_id=tokenizer.mem_token_id, compr_rate=self.model.compr_rate
        )
        return sum(per_row)

    def process_with_highlights(
        self,
        query: str,
        context: str,
        threshold: float | None = 0.1,
        compute_spans: bool = True,
    ) -> dict:
        """Return the document unchanged; compression happens at generation time."""
        text = _clean_text(context) if isinstance(context, str) else ""
        score = self.compute_relevance_score(query, text) if (text and self._external_reranker) else 0.0
        result: dict = {"pruned_context": text, "reranking_score": score, "compression_rate": 0.0}
        if compute_spans:
            result["spans"] = [{"start": 0, "end": len(text), "selected": True}]
        return result
